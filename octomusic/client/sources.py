"""EEG sources. Each one pushes (n_samples, 16) microvolt blocks into an
EEGBuffer. All run as asyncio tasks on the client's event loop.

  OctopusBLE   the real headset, over Bluetooth LE (bleak)
  Synthetic    fake EEG whose band powers follow UI sliders -- for exercising
               the whole chain (and the model's response) without the headset
  Replay       plays back a recording made by this app at real-time speed --
               for iterating on normalization/mapping offline

OCTOPUS 16 PROTOCOL (pieeg-club/Octopus_16 + hello_world_octopus_16):
  2x TI ADS131M08, 24-bit, gain 4, Vref 1.2 V, ~250 SPS (some boards with an
  8.000 MHz crystal run at ~244 SPS -- we measure the real rate and report it).
  Controller: Seeed XIAO ESP32-S3, BLE only.
  service  4fafc201-1fb5-459e-8fcc-c5c9c331914b
  data     beb5483e-36e1-4688-b7f5-ea07361b26a8   (notify)
  command  c0de0001-36e1-4688-b7f5-ea07361b26a8   (register bridge)
  Packet: 51 bytes = 0xA0, counter(0..255), 16 x int24 big-endian, 0xC0.
  A notification may carry several packets back to back.
  uV = raw * 1.2 / 4 / 8388607 * 1e6  (~0.0358 uV / LSB)
"""

from __future__ import annotations

import asyncio
import csv
import os
import threading
import time

import numpy as np

N_CH = 16
NOMINAL_FS = 250.0

SERVICE_UUID = "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
DATA_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"
CMD_UUID = "c0de0001-36e1-4688-b7f5-ea07361b26a8"
PACKET_LEN = 51
UV_SCALE = 1.2 / 4.0 / 8388607.0 * 1e6
_NAME_PREFIXES = ("octopus", "octopos", "bioron", "pieeg")


class EEGBuffer:
    """Ring buffer of the last `seconds` of raw uV, plus counters.

    `total` counts samples ever written, so consumers keep their own read
    cursor and never miss or double-read a block."""

    def __init__(self, seconds: float = 60.0, fs: float = NOMINAL_FS):
        self.fs = fs
        self.cap = int(seconds * fs)
        self.data = np.zeros((self.cap, N_CH), dtype=np.float32)
        self.total = 0
        self.lost = 0
        self.received = 0
        self._lock = threading.Lock()
        self._rate_t0 = None
        self._rate_n0 = 0
        self.measured_fs = None
        self.listeners = []          # callables(block) -- e.g. the recorder

    def write(self, block: np.ndarray):
        block = np.asarray(block, dtype=np.float32).reshape(-1, N_CH)
        n = len(block)
        if n == 0:
            return
        with self._lock:
            i = self.total % self.cap
            k = min(n, self.cap - i)
            self.data[i:i + k] = block[:k]
            if k < n:
                self.data[:n - k] = block[k:]
            self.total += n
        now = time.time()
        if self._rate_t0 is None:
            self._rate_t0, self._rate_n0 = now, self.total
        elif now - self._rate_t0 >= 5.0:
            self.measured_fs = (self.total - self._rate_n0) / (now - self._rate_t0)
            self._rate_t0, self._rate_n0 = now, self.total
        for f in self.listeners:
            f(block)

    def read_since(self, cursor: int):
        """Samples written since `cursor` (clipped to what the ring holds)."""
        with self._lock:
            total = self.total
            n = min(total - cursor, self.cap)
            if n <= 0:
                return np.zeros((0, N_CH), np.float32), total
            start = (total - n) % self.cap
            if start + n <= self.cap:
                out = self.data[start:start + n].copy()
            else:
                out = np.concatenate([self.data[start:], self.data[:(start + n) % self.cap]])
        return out, total

    def reset_stats(self):
        self.lost = self.received = 0


class Source:
    name = "source"

    def __init__(self, buf: EEGBuffer):
        self.buf = buf
        self.status = "idle"
        self.error = ""
        self._task = None

    def start(self):
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._guard())

    async def _guard(self):
        try:
            await self.run()
        except asyncio.CancelledError:
            pass
        except Exception as e:  # noqa: BLE001
            self.status, self.error = "error", f"{type(e).__name__}: {e}"
        finally:
            if self.status not in ("error",):
                self.status = "stopped"

    async def stop(self):
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self._task = None
        self.status = "stopped"

    async def run(self):
        raise NotImplementedError

    def info(self):
        return {"name": self.name, "status": self.status, "error": self.error}


# ---- the real device ----------------------------------------------------------

def parse_packets(data: bytes, last_counter):
    """Decode every valid 51-byte packet in one notification.
    Returns (uv array (n, 16), last_counter, n_lost)."""
    rows, lost = [], 0
    for off in range(0, len(data) - PACKET_LEN + 1, PACKET_LEN):
        p = data[off:off + PACKET_LEN]
        if p[0] != 0xA0 or p[50] != 0xC0:
            continue
        ctr = p[1]
        if last_counter is not None:
            gap = (ctr - last_counter - 1) % 256
            if 0 < gap < 128:
                lost += gap
        last_counter = ctr
        raw = np.frombuffer(bytes(p[2:50]), dtype=np.uint8).reshape(16, 3).astype(np.int32)
        v = (raw[:, 0] << 16) | (raw[:, 1] << 8) | raw[:, 2]
        v = np.where(v & 0x800000, v - (1 << 24), v)
        rows.append(v.astype(np.float32) * UV_SCALE)
    if not rows:
        return np.zeros((0, N_CH), np.float32), last_counter, lost
    return np.stack(rows), last_counter, lost


class OctopusBLE(Source):
    name = "octopus"

    def __init__(self, buf, address: str | None = None, scan_timeout=10.0):
        super().__init__(buf)
        self.address = address
        self.scan_timeout = scan_timeout
        self.device_name = ""
        self.mtu = None

    async def run(self):
        from bleak import BleakClient, BleakScanner
        self.status = "scanning"
        if self.address:
            dev = await BleakScanner.find_device_by_address(self.address,
                                                            timeout=self.scan_timeout)
        else:
            def match(d, adv):
                uu = [u.lower() for u in (adv.service_uuids or [])]
                nm = (d.name or adv.local_name or "").lower()
                return SERVICE_UUID in uu or nm.startswith(_NAME_PREFIXES)
            dev = await BleakScanner.find_device_by_filter(match, timeout=self.scan_timeout)
        if dev is None:
            raise RuntimeError("no Octopus found (is it on, charged, ESP32 seated the "
                               "right way round, and does this terminal have "
                               "Bluetooth permission?)")
        self.device_name = dev.name or dev.address
        self.status = "connecting"
        last = {"ctr": None}
        disconnected = asyncio.Event()

        def on_data(_, data: bytearray):
            uv, last["ctr"], lost = parse_packets(bytes(data), last["ctr"])
            self.buf.lost += lost
            self.buf.received += len(uv)
            if len(uv):
                self.buf.write(uv)

        async with BleakClient(dev, disconnected_callback=lambda _: disconnected.set()) as client:
            self.mtu = getattr(client, "mtu_size", None)
            await client.start_notify(DATA_UUID, on_data)
            self.status = "streaming"
            await disconnected.wait()
        raise RuntimeError("device disconnected")

    def info(self):
        return {**super().info(), "device": self.device_name, "mtu": self.mtu}


# ---- fake EEG -------------------------------------------------------------------

class Synthetic(Source):
    """Pink-ish background plus band-limited oscillators whose amplitudes are
    live knobs (uV). Channels share the rhythms with per-channel gain and
    independent noise, which is roughly what 16 closely spaced dry pins see."""
    name = "synthetic"

    def __init__(self, buf, fs=NOMINAL_FS):
        super().__init__(buf)
        self.fs = fs
        self.knobs = {"theta": 4.0, "alpha": 10.0, "beta": 3.0, "emg": 0.0,
                      "noise": 6.0, "line": 2.0, "blink": 0.0}
        self._rng = np.random.default_rng(1)
        self._phase = {}
        self._pink = np.zeros(N_CH)

    def _osc(self, name, f0, bw, n):
        # narrowband: random-walk frequency around f0
        ph = self._phase.get(name, self._rng.uniform(0, 2 * np.pi, N_CH))
        f = f0 + bw * np.sin(np.linspace(0, 1, n)[:, None] * 0.3 + ph[None, :])
        inc = 2 * np.pi * f / self.fs
        p = ph[None, :] + np.cumsum(inc, axis=0)
        self._phase[name] = p[-1] % (2 * np.pi)
        return np.sin(p)

    async def run(self):
        self.status = "streaming"
        gains = 0.7 + 0.6 * self._rng.random(N_CH)
        block = 10
        t_next = time.perf_counter()
        n_done = 0
        while True:
            k = self.knobs
            n = block
            t = (n_done + np.arange(n)) / self.fs
            white = self._rng.normal(size=(n, N_CH))
            out = np.empty((n, N_CH))
            for i in range(n):            # 1-pole "pink-ish" background
                self._pink = 0.97 * self._pink + white[i]
                out[i] = self._pink * 0.25
            out *= k["noise"]
            out += k["theta"] * self._osc("theta", 6.0, 1.0, n) * gains
            out += k["alpha"] * self._osc("alpha", 10.0, 0.8, n) * gains
            out += k["beta"] * self._osc("beta", 20.0, 4.0, n) * gains
            if k["emg"] > 0:
                out += k["emg"] * self._rng.normal(size=(n, N_CH)) * \
                    np.abs(self._osc("emgmod", 0.5, 0.2, n))
            out += k["line"] * np.sin(2 * np.pi * 60.0 * t)[:, None]
            if k["blink"] > 0 and self._rng.random() < 0.02 * n / self.fs * 10:
                out += k["blink"] * 20 * np.hanning(n)[:, None]
            out += 2000.0                  # DC offset, like the real ADC
            self.buf.received += n
            self.buf.write(out.astype(np.float32))
            n_done += n
            t_next += n / self.fs
            await asyncio.sleep(max(0.0, t_next - time.perf_counter()))


# ---- replay -----------------------------------------------------------------------

class Replay(Source):
    """Replays `eeg.csv` from a recording folder (written by recorder.py):
    columns t, ch1..ch16 in uV. Loops at the end."""
    name = "replay"

    def __init__(self, buf, path: str, speed: float = 1.0):
        super().__init__(buf)
        self.path = path
        self.speed = speed
        self.progress = 0.0

    def _load(self):
        p = self.path
        if os.path.isdir(p):
            p = os.path.join(p, "eeg.csv")
        with open(p) as f:
            r = csv.reader(f)
            header = next(r)
            rows = [list(map(float, row)) for row in r if row]
        a = np.asarray(rows, dtype=np.float64)
        chans = [i for i, h in enumerate(header) if h.startswith("ch")]
        t = a[:, header.index("t")] if "t" in header else np.arange(len(a)) / NOMINAL_FS
        return t, a[:, chans].astype(np.float32)

    async def run(self):
        self.status = "loading"
        t, x = await asyncio.to_thread(self._load)
        if len(x) < 10:
            raise RuntimeError("recording is empty")
        dur = t[-1] - t[0]
        fs = (len(t) - 1) / dur if dur > 0 else NOMINAL_FS
        self.status = "streaming"
        block = 10
        i = 0
        t_next = time.perf_counter()
        while True:
            j = min(len(x), i + block)
            self.buf.received += j - i
            self.buf.write(x[i:j])
            self.progress = j / len(x)
            t_next += (j - i) / fs / max(0.05, self.speed)
            i = 0 if j >= len(x) else j
            await asyncio.sleep(max(0.0, t_next - time.perf_counter()))

    def info(self):
        return {**super().info(), "path": self.path, "progress": self.progress}
