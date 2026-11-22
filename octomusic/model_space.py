"""What the music model will accept, in one place both ends import.

Everything here is copied from the Neural Music Generator repo (weather.py and
backend.py) rather than imported, because the CLIENT must not need that repo,
torch, or a GPU. If the checkpoint changes, these are the numbers to revisit.

THE CONTRACT. The client turns EEG into a "dial" per target: a number in 0..1.
The server turns each dial into model units by lerping across that target's
band. Keeping the client in 0..1 means every normalization experiment on the
laptop is model-agnostic, and the "how far may this signal go" question lives
in exactly one table that the UI can edit live (bands travel with the params).
"""

from __future__ import annotations

import math

import numpy as np

# Conditioning targets the client can drive. The first five are real model
# signals; "style" is a dial that slerps between two presets' style vectors
# (only read when the server's style mode is "morph").
# Every model input. Scalars take a 0..1 dial; chroma takes a 12-vector the
# client builds (key template / EEG spectrum / dial-chosen key); style takes a
# dial whose meaning depends on the style mode (morph amount, axis angle).
TARGETS = ("rms", "bpm", "density", "zcr", "flatness", "chroma", "style")
SCALAR_TARGETS = ("rms", "bpm", "density", "zcr", "flatness")
DSP_TARGETS = ("rms", "density", "zcr", "flatness")   # local_embedder signals

TARGET_LABELS = {
    "rms": "energy (rms)",
    "bpm": "tempo (bpm)",
    "density": "density (onsets/s)",
    "zcr": "brightness (zcr)",
    "flatness": "flatness (noisiness)",
    "chroma": "harmony (chroma)",
    "style": "style",
}

# From weather.TARGET_BANDS: pooled 2nd/98th percentiles of the five shipped
# presets, pulled in slightly -- i.e. the range this checkpoint has actually
# been conditioned on. (lo, hi, log). `log` lerps geometrically: flatness spans
# two decades. The bpm numbers are per-MEASURE rates (60*4/measure_seconds),
# which is why they look high; 240 means one measure per second.
DEFAULT_BANDS = {
    "rms":      (0.060, 0.220, False),
    "density":  (1.000, 6.000, False),
    "zcr":      (0.055, 0.135, False),
    "flatness": (0.0004, 0.0250, True),
    "bpm":      (175.0, 290.0, False),
    "style":    (0.0, 1.0, False),
    "chroma":   (0.0, 1.0, False),
}

# ---- tempo grid (backend.snap_bpm) -------------------------------------------
# The bpm embedding is an nn.Embedding(350) and only rows BPM_GRID_NUM/k were
# ever trained (the label pipeline measured measures in whole 320-sample frames
# at 16 kHz). Off-grid rows are untrained noise vectors, so every bpm the model
# sees must be snapped. The server snaps again with the real function; this copy
# exists so the UI can show the tempo the model will actually get.
SR = 16000
TARGET_SIG = 4
BPM_GRID_NUM = 60.0 * TARGET_SIG * SR / 320           # 12000
BPM_MIN, BPM_MAX = 40.0, 350.0
BPM_K_MIN = int(math.ceil(BPM_GRID_NUM / BPM_MAX))    # 35
BPM_K_MAX = int(math.floor(BPM_GRID_NUM / BPM_MIN))   # 300


def snap_bpm(bpm: float) -> float:
    if not np.isfinite(bpm) or bpm <= 0:
        return BPM_MIN
    k = int(np.clip(round(BPM_GRID_NUM / bpm), BPM_K_MIN, BPM_K_MAX))
    return BPM_GRID_NUM / k


def measure_seconds(bpm: float) -> float:
    return 60.0 * TARGET_SIG / float(bpm)


def dial_to_value(target: str, u: float, bands: dict | None = None) -> float:
    """0..1 dial -> model units for `target`. Clipped; bpm snapped."""
    lo, hi, log = (bands or DEFAULT_BANDS)[target]
    u = float(np.clip(u, 0.0, 1.0))
    if log:
        v = math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))
    else:
        v = lo + u * (hi - lo)
    if target == "bpm":
        v = snap_bpm(v)
    return v


def value_to_dial(target: str, v: float, bands: dict | None = None) -> float:
    lo, hi, log = (bands or DEFAULT_BANDS)[target]
    if log:
        u = (math.log(max(v, 1e-12)) - math.log(lo)) / (math.log(hi) - math.log(lo))
    else:
        u = (v - lo) / (hi - lo)
    return float(np.clip(u, 0.0, 1.0))


def bands_from_json(d) -> dict:
    """Accept bands as sent over JSON ({t: [lo, hi, log]}), merged on defaults."""
    out = dict(DEFAULT_BANDS)
    for k, v in (d or {}).items():
        if k in out and v is not None and len(v) >= 2:
            lo, hi = float(v[0]), float(v[1])
            log = bool(v[2]) if len(v) > 2 else out[k][2]
            if log and (lo <= 0 or hi <= 0):
                continue
            out[k] = (lo, hi, log)
    return out


# ---- harmony -----------------------------------------------------------------
# backend.key_template, copied: the chroma row that stands for a key.
def key_template(root: int, minor: bool) -> np.ndarray:
    scale = [0, 2, 3, 5, 7, 8, 10] if minor else [0, 2, 4, 5, 7, 9, 11]
    t = np.full(12, 0.35)
    for i in scale:
        t[(int(root) + i) % 12] += 0.4
    t[int(root) % 12] += 0.2
    return np.clip(t, 0, 1)


# Keys along the circle of fifths, so a dial sweeping 0..1 walks through
# neighbouring keys (each step changes one note) instead of jumping around.
CIRCLE_OF_FIFTHS = [(7 * i) % 12 for i in range(12)]     # C G D A E B F# ...
NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
