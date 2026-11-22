"""Speaker output: a jitter buffer between the GPU's chunks and sounddevice.

The model renders at 16 kHz. Some devices won't open at 16 kHz, so if that
fails we open at 48 kHz and upsample by 3 with a STATEFUL polyphase FIR (a
per-chunk resample would click at every chunk boundary).

`played` is reported in the 16 kHz stream index the server uses.

FIXED-DELAY PLAYOUT (online mode). `schedule(t_play)` holds the speaker silent
until wall time t_play (= press time + delay) and then plays continuously. The
moment real audio first leaves, an ANCHOR is recorded -- (stream sample, wall
time) -- and the app forwards it to the server, which uses it to know when
every future measure will be heard. If the queue ever runs dry, the output
does not stutter chunk by chunk: it goes silent once, waits until
`rebuffer_s` is queued, resumes, and records a new anchor. A smooth stream
with one audible gap beats a stream that stalls every take.
"""

from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np
from scipy import signal

SR = 16000


class _Upsampler:
    def __init__(self, L):
        self.L = L
        self.h = signal.firwin(24 * L + 1, 0.9 / L) * L
        self.K = int(np.ceil(len(self.h) / L))
        self.tail = np.zeros(self.K, np.float32)

    def __call__(self, x):
        xx = np.concatenate([self.tail, x])
        y = signal.upfirdn(self.h, xx, up=self.L)
        out = y[self.K * self.L:(self.K + len(x)) * self.L]
        self.tail = xx[-self.K:]
        return out.astype(np.float32)


class AudioOut:
    def __init__(self, device=None):
        self.device = device
        self.volume = 0.8
        self.muted = False
        self._q = deque()             # output-domain float32 arrays
        self._qlen = 0
        self._lock = threading.Lock()
        self.ratio = 1
        self.out_sr = SR
        self._up = None
        self.stream = None
        self.start16 = None           # stream index of the first queued sample
        self.next16 = None            # stream index expected next
        self.consumed_out = 0
        self.underruns = 0
        self.level = 0.0
        self.error = ""
        self.listeners = []           # f(start16, chunk16)
        self.gate_t = None            # don't play before this wall time
        self.rebuffer_s = 4.0
        self._playing = False
        self._rebuffering = False
        self.anchors = []             # (sample16, wall time) events, drained by app
        self.rebuffers = 0

    def open(self):
        import sounddevice as sd
        for sr in (SR, 48000):
            try:
                self.stream = sd.OutputStream(samplerate=sr, channels=1, dtype="float32",
                                              device=self.device, callback=self._cb,
                                              blocksize=0, latency="low")
                self.stream.start()
                self.out_sr, self.ratio = sr, sr // SR
                self._up = _Upsampler(self.ratio) if self.ratio > 1 else None
                return
            except Exception as e:  # noqa: BLE001
                self.error = f"{type(e).__name__}: {e}"
                self.stream = None
        raise RuntimeError(f"could not open audio output: {self.error}")

    def close(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def reset(self):
        with self._lock:
            self._q.clear()
            self._qlen = 0
            self.start16 = self.next16 = None
            self.consumed_out = 0
            self.underruns = 0
            self.rebuffers = 0
            self.gate_t = None
            self._playing = False
            self._rebuffering = False
            self.anchors = []
            if self._up is not None:
                self._up = _Upsampler(self.ratio)

    def schedule(self, t_play: float | None, rebuffer_s: float = 4.0):
        """Hold output until wall time t_play (None = play as soon as possible)."""
        with self._lock:
            self.gate_t = t_play
            self.rebuffer_s = rebuffer_s

    def push(self, start16: int, x: np.ndarray):
        x = np.asarray(x, np.float32)
        for f in self.listeners:
            f(start16, x)
        with self._lock:
            if self.next16 is None:
                self.start16 = self.next16 = start16
            if start16 > self.next16:                    # gap -> silence
                x = np.concatenate([np.zeros(start16 - self.next16, np.float32), x])
            elif start16 < self.next16:                  # overlap -> skip
                x = x[self.next16 - start16:]
            self.next16 += len(x)
            y = self._up(x) if self._up is not None else x
            self._q.append(y)
            self._qlen += len(y)

    def _cb(self, out, frames, time_info, _status):
        out.fill(0)
        now = time.time()
        with self._lock:
            if self.gate_t is not None and now < self.gate_t:
                return                                   # pre-roll: hold silent
            if self._rebuffering:
                if self._qlen < self.rebuffer_s * self.out_sr:
                    return
                self._rebuffering = False
                self._playing = False                    # re-anchor below
            got = 0
            if self._q and not self._playing:
                # first sample of a (re)start: when it will physically sound
                lat = 0.0
                try:
                    lat = float(time_info.outputBufferDacTime - time_info.currentTime)
                except Exception:  # noqa: BLE001
                    pass
                s16 = (self.start16 or 0) + self.consumed_out // self.ratio
                self.anchors.append((int(s16), now + max(0.0, lat)))
                self._playing = True
            while got < frames and self._q:
                a = self._q[0]
                k = min(len(a), frames - got)
                out[got:got + k, 0] = a[:k]
                if k == len(a):
                    self._q.popleft()
                else:
                    self._q[0] = a[k:]
                got += k
            self._qlen -= got
            self.consumed_out += got
            if got < frames and self._playing:
                self.underruns += 1
                self.rebuffers += 1
                self._rebuffering = True
                self._playing = False
        g = 0.0 if self.muted else self.volume
        out *= g
        self.level = float(np.sqrt(np.mean(out ** 2))) if frames else 0.0

    @property
    def played(self):
        if self.start16 is None:
            return 0
        return self.start16 + self.consumed_out // self.ratio

    @property
    def buffered_s(self):
        return self._qlen / float(self.out_sr)

    def info(self):
        wait = None if self.gate_t is None else max(0.0, self.gate_t - time.time())
        return {"sr": self.out_sr, "buffered_s": round(self.buffered_s, 2),
                "underruns": self.underruns, "rebuffers": self.rebuffers,
                "preroll_s": wait, "rebuffering": self._rebuffering, "played": self.played,
                "level": self.level, "error": self.error if self.stream is None else ""}
