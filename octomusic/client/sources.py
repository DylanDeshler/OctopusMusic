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
from collections import deque

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
        self.reconnects = 0
        self.events = deque(maxlen=50)     # (time, what went wrong)
        self._task = None

    def start(self):
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._guard())

    reconnect = False          # hardware sources retry instead of dying

    async def _guard(self):
        backoff = 1.0
        while True:
            t0 = time.time()
            try:
                await self.run()
                if not self.reconnect:
                    break
                raise RuntimeError("stream ended")
            except asyncio.CancelledError:
                break
            except Exception as e:  # noqa: BLE001
                self.error = f"{type(e).__name__}: {e}"
                self.events.append((time.time(), self.error))
                if not self.reconnect or getattr(e, "fatal", False):
                    self.status = "error"
                    return
                if time.time() - t0 > 30:
                    backoff = 1.0                  # it had been working: retry fast
                self.reconnects += 1
                self.status = f"reconnecting ({self.reconnects})"
                try:
                    await asyncio.sleep(backoff)
                except asyncio.CancelledError:
                    break
                backoff = min(10.0, backoff * 2)
        if self.status != "error":
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
        return {"name": self.name, "status": self.status, "error": self.error,
                "reconnects": self.reconnects}


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
    reconnect = True

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


# ---- the real device, over USB ------------------------------------------------

ESPRESSIF_VID = 0x303A


def find_usb_port():
    """The Octopus's XIAO ESP32-S3 USB serial port, or None."""
    from serial.tools import list_ports
    for p in list_ports.comports():
        if p.vid == ESPRESSIF_VID:
            return p.device
    for p in list_ports.comports():
        if "usbmodem" in p.device or "ttyACM" in p.device:
            return p.device
    return None


def split_stream(buf: bytes | bytearray):
    """Split a USB byte stream into 51-byte packets and text lines.

    The firmware prints text (boot log, heartbeat) until streaming starts and
    after it stops; packets are framed A0 ... C0. Text never contains 0xA0, so
    anything that is not a framed packet is either a text line or junk to skip
    up to the next 0xA0. Returns (packets, lines, bytes consumed); a partial
    packet or line at the end is left for the next call."""
    packets, lines = [], []
    i, n = 0, len(buf)
    while i < n:
        if buf[i] == 0xA0:
            if i + PACKET_LEN > n:
                break
            if buf[i + 50] == 0xC0:
                packets.append(bytes(buf[i:i + PACKET_LEN]))
                i += PACKET_LEN
            else:
                i += 1
            continue
        j = buf.find(b"\n", i)
        k = buf.find(b"\xa0", i)
        if k != -1 and (j == -1 or k < j):
            i = k                                   # junk before a packet
            continue
        if j == -1:
            if n - i > 512:
                i = n                               # runaway junk
            break
        line = bytes(buf[i:j]).decode("utf-8", "replace").strip()
        if line:
            lines.append(line)
        i = j + 1
    return packets, lines, i


class OctopusUSB(Source):
    """The headset over its USB-C port. Needs the octomusic firmware
    (firmware/octopus_usb): send 'S' and it writes the same 51-byte packets the
    BLE characteristic carries, every sample, with no loss.

    SAFETY: the board has no isolation. While the headset is worn over USB the
    computer must run on battery with the charger unplugged and nothing
    mains-powered attached.

    Opening the port resets the ESP32 (the USB-serial peripheral does that on
    connect), so this waits for the boot to finish before asking for data, and
    keeps the firmware's text lines (boot log, heartbeat) for the UI."""
    name = "octopus-usb"
    reconnect = True

    def __init__(self, buf, port: str | None = None):
        super().__init__(buf)
        self.port = port
        self.log = deque(maxlen=20)
        self.device_name = ""
        self.reset_reasons = deque(maxlen=20)   # board's own reason for each boot
        self.last_hb = ""
        self._stop_evt = threading.Event()

    async def run(self):
        self._stop_evt.clear()
        self.status = "connecting"
        await asyncio.to_thread(self._worker)

    async def stop(self):
        self._stop_evt.set()
        await super().stop()

    def _worker(self):
        import serial
        port = self.port or find_usb_port()
        if not port:
            raise RuntimeError("no Octopus USB port found (is the USB-C cable data-capable?)")
        self.device_name = port
        s = serial.Serial()
        s.port, s.baudrate, s.timeout = port, 115200, 0.05
        s.dtr = False
        s.rts = False
        s.open()
        try:
            self._loop(s)
        finally:
            try:
                s.write(b"X")
                s.close()
            except Exception:  # noqa: BLE001
                pass

    def _loop(self, s):
        pending = bytearray()
        last_ctr = None
        t_open = time.time()
        t_sent = 0.0
        t_last_pkt = None
        booted = False
        while not self._stop_evt.is_set():
            now = time.time()
            if not booted and (now - t_open > 3.0):
                booted = True
            if booted and (t_last_pkt is None or now - t_last_pkt > 2.0) and now - t_sent > 2.0:
                s.write(b"S")
                t_sent = now
                if t_last_pkt is None and now - t_open > 10.0:
                    e = RuntimeError(
                        "board answers but sends no EEG over USB: it is running firmware "
                        "without USB streaming. Flash firmware/octopus_usb (see README), "
                        "or use the Bluetooth source.")
                    e.fatal = True
                    raise e
            chunk = s.read(4096)
            if not chunk:
                continue
            pending += chunk
            raw, lines, used = split_stream(pending)
            del pending[:used]
            for line in lines:
                self.log.append(line)
                if "Setup complete" in line:
                    booted = True
                if line.startswith("[BOOT] reset reason:"):
                    self.reset_reasons.append((time.time(), line.split(":", 1)[1].strip()))
                if line.startswith("[HB]"):
                    self.last_hb = line
            rows = []
            if raw:
                uv, last_ctr, lost = parse_packets(b"".join(raw), last_ctr)
                self.buf.lost += lost
                rows.append(uv)
            if rows:
                block = np.concatenate(rows)
                self.buf.received += len(block)
                self.buf.write(block)
                t_last_pkt = time.time()
                self.status = "streaming"

    def info(self):
        return {**super().info(), "device": self.device_name, "log": list(self.log)[-3:],
                "last_hb": self.last_hb,
                "reset_reasons": [r for _, r in self.reset_reasons][-5:],
                "events": [e for _, e in self.events][-5:]}


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
