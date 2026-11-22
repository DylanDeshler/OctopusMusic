"""Session recording -- the raw material for answering the normalization
question offline.

recordings/<stamp>/
    eeg.csv          t, ch1..ch16   raw uV as received (replayable)
    features.jsonl   {t, feats, artifact, dials, z}  every feature hop
    takes.jsonl      every take / measured message from the server
    output.wav       the generated audio as streamed (16 kHz)
    config.json      the full client config at start (routes, dsp, params)
    events.jsonl     calibration marks, notes, parameter changes
    labels.json      [[t0, t1, "label"], ...] seconds from the first EEG sample --
                     what the evaluation harness and the Record tab read
    meta.json        subject, placement, protocol, device, contact quality at start
"""

from __future__ import annotations

import json
import os
import threading
import time

import numpy as np

from octomusic import protocol


class Recorder:
    def __init__(self, root="recordings"):
        self.root = root
        self.dir = None
        self._eeg = self._feat = self._takes = self._events = None
        self._wav = None
        self._wav_next = None
        self.started = None
        self._lock = threading.Lock()      # eeg() runs on the USB reader thread
        self._t_first = None
        self._segments = []                # closed [t0_abs, t1_abs, label]
        self._open = None                  # (t0_abs, label) of the current label
        self.meta = {}

    @property
    def active(self):
        return self.dir is not None

    def start(self, config: dict, fs: float, meta: dict | None = None):
        import soundfile as sf
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.dir = os.path.join(self.root, stamp)
        os.makedirs(self.dir, exist_ok=True)
        self._eeg = open(os.path.join(self.dir, "eeg.csv"), "w")
        self._eeg.write("t," + ",".join(f"ch{i + 1}" for i in range(16)) + "\n")
        self._feat = open(os.path.join(self.dir, "features.jsonl"), "w")
        self._takes = open(os.path.join(self.dir, "takes.jsonl"), "w")
        self._events = open(os.path.join(self.dir, "events.jsonl"), "w")
        self._wav = sf.SoundFile(os.path.join(self.dir, "output.wav"), "w",
                                 samplerate=16000, channels=1, subtype="PCM_16")
        self._wav_next = None
        with open(os.path.join(self.dir, "config.json"), "w") as f:
            f.write(protocol.dumps({**config, "fs": fs}))
        self.started = time.time()
        self._fs = fs
        self._n = 0
        self._t_first = None
        self._segments, self._open = [], None
        self.meta = {"started": self.started, "fs": fs, **(meta or {})}
        self._write_meta()
        return self.dir

    # ---- labels -------------------------------------------------------------------
    def label(self, lab: str | None):
        """Switch the current label (None = unlabelled) at this instant."""
        if not self.active:
            return
        now = time.time()
        if self._open is not None:
            t0, prev = self._open
            if lab == prev:
                return
            if now - t0 > 0.05:
                self._segments.append([t0, now, prev])
        self._open = (now, lab) if lab else None
        self.event("label", label=lab)
        self._write_labels()

    @property
    def current_label(self):
        return self._open[1] if self._open else None

    def label_summary(self):
        out = {}
        segs = list(self._segments) + ([[self._open[0], time.time(), self._open[1]]] if self._open else [])
        for t0, t1, lab in segs:
            d = out.setdefault(lab, {"seconds": 0.0, "segments": 0})
            d["seconds"] += t1 - t0
            d["segments"] += 1
        return out

    def _write_labels(self):
        if not self.dir:
            return
        base = self._t_first or self.started
        segs = list(self._segments)
        if self._open is not None:
            segs.append([self._open[0], time.time(), self._open[1]])
        rel = [[round(t0 - base, 3), round(t1 - base, 3), lab] for t0, t1, lab in segs]
        with open(os.path.join(self.dir, "labels.json"), "w") as f:
            json.dump(rel, f)

    def _write_meta(self):
        if self.dir:
            with open(os.path.join(self.dir, "meta.json"), "w") as f:
                f.write(protocol.dumps(self.meta))

    def update_meta(self, **kw):
        self.meta.update(kw)
        self._write_meta()

    def stop(self):
        if self.active:
            self.label(None)
            self._write_labels()
            self.update_meta(stopped=time.time(), labels=self.label_summary())
        with self._lock:
            for f in (self._eeg, self._feat, self._takes, self._events):
                if f:
                    f.close()
        if self._wav:
            self._wav.close()
        d = self.dir
        self.dir = None
        self._eeg = self._feat = self._takes = self._events = self._wav = None
        return d

    def eeg(self, block: np.ndarray):
        with self._lock:
            if not self._eeg:
                return
            t = time.time()
            n = len(block)
            ts = t - (n - 1 - np.arange(n)) / self._fs
            if self._t_first is None:
                self._t_first = float(ts[0])
            lines = [f"{ts[i]:.4f}," + ",".join(f"{v:.2f}" for v in block[i]) for i in range(n)]
            self._eeg.write("\n".join(lines) + "\n")

    def features(self, row: dict):
        if self._feat:
            self._feat.write(protocol.dumps(row) + "\n")

    def take(self, msg: dict):
        if self._takes:
            self._takes.write(protocol.dumps(msg) + "\n")

    def event(self, kind: str, **kw):
        if self._events:
            self._events.write(protocol.dumps({"t": time.time(), "kind": kind, **kw}) + "\n")
            self._events.flush()

    def audio(self, start16: int, x: np.ndarray):
        if not self._wav:
            return
        if self._wav_next is None:
            self._wav_next = start16
        if start16 > self._wav_next:
            self._wav.write(np.zeros(start16 - self._wav_next, np.float32))
            self._wav_next = start16
        elif start16 < self._wav_next:
            x = x[self._wav_next - start16:]
        self._wav.write(x)
        self._wav_next += len(x)
