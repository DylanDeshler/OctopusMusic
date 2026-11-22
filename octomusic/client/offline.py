"""Offline mode: recorded EEG -> features -> dials -> a whole generated piece.

For finding normalizations and stats before trusting them live. The pass is
the SAME code the live path runs (same causal filters, same feature function,
same Mapper), stepped at the same 0.25 s hop on the recording's own clock, so
whatever sounds right here behaves the same way live. The one thing offline can
do that live cannot is look at the whole session at once: `analyze` fills
every feature's session stats (mean, std, full distribution), which the
`session_z` / `session_quantile` normalizers read -- here non-causally, and in
later live runs as a fixed reference.
"""

from __future__ import annotations

import csv
import os

import numpy as np
from octomusic.client.dsp import (DEFAULT_DSP, FEATURES, FilterBank, channel_quality,
                                  compute_features, used_channels)
from octomusic.client.sources import NOMINAL_FS

HOP_S = 0.25


def load_recording(path: str):
    p = os.path.join(path, "eeg.csv") if os.path.isdir(path) else path
    with open(p) as f:
        r = csv.reader(f)
        header = next(r)
        rows = [row for row in r if row]
    a = np.asarray(rows, dtype=np.float64)
    chans = [i for i, h in enumerate(header) if h.startswith("ch")]
    t = a[:, header.index("t")] if "t" in header else np.arange(len(a)) / NOMINAL_FS
    x = a[:, chans].astype(np.float32)
    dur = t[-1] - t[0] if len(t) > 1 else 0.0
    fs = (len(t) - 1) / dur if dur > 0 else NOMINAL_FS
    return t, x, fs


def extract(t, x, fs, dsp_cfg: dict, start_s=0.0, duration_s=None):
    """Feature series over [start_s, start_s+duration_s) of the recording.
    Same FilterBank, quality test and feature function as live, stepped at
    the live hop. Returns dict(times, feats{name: array}, spec12, artifact)."""
    cfg = {**DEFAULT_DSP, **{k: v for k, v in dsp_cfg.items() if k in DEFAULT_DSP}}
    bank = FilterBank(fs, cfg)
    clean, wide, hp = bank(x)
    win = int(cfg["window_s"] * fs)
    hop = max(1, int(HOP_S * fs))
    i0 = max(win + int(2 * fs), int(start_s * fs))        # skip filter warm-up
    i1 = len(x) if duration_s is None else min(len(x), int((start_s + duration_s) * fs))
    k = int(fs)
    times, rows, specs, arts = [], [], [], []
    for i in range(i0, i1 + 1, hop):
        q = channel_quality(clean[i - k:i], hp[i - 2 * k:i], x[i - k:i], fs, cfg["notch_hz"])
        use = used_channels(cfg, q["good"])
        feats, sp, art, _, _ = compute_features(clean[i - win:i], wide[i - win:i], fs, use,
                                                cfg["artifact_uv"])
        times.append(float(t[i - 1]))
        rows.append(feats)
        specs.append(sp)
        arts.append(art > 0.5)
    names = list(rows[0]) if rows else list(FEATURES)
    return {
        "times": np.asarray(times),
        "feats": {n: np.asarray([r[n] for r in rows]) for n in names},
        "spec12": np.asarray(specs),
        "artifact": np.asarray(arts, bool),
    }


def session_series(ex: dict, routes: dict):
    """Every feature plus every route expression, as whole-session arrays."""
    from octomusic.client.mapping import eval_expr
    out = dict(ex["feats"])
    n = len(ex["times"])
    for r in routes.values():
        e = r["feature"]
        if e not in out:
            vals = [eval_expr(e, {k: v[i] for k, v in ex["feats"].items()}) for i in range(n)]
            if all(v is not None for v in vals):
                out[e] = np.asarray(vals, float)
    clean = ~ex["artifact"]
    return {k: (v[clean] if clean.any() else v) for k, v in out.items()}


def dial_frames(ex: dict, mapper):
    """Step a (cloned) Mapper through the recording on its own clock."""
    frames = []
    feats = ex["feats"]
    for i, t in enumerate(ex["times"]):
        f = {k: float(v[i]) for k, v in feats.items()}
        dials, chroma = mapper.step(f, bool(ex["artifact"][i]), float(t), ex["spec12"][i])
        frames.append({"t": float(t), "values": dials,
                       "chroma": None if chroma is None else [round(float(c), 4) for c in chroma]})
    return frames


def summarize(series: dict):
    """Per-feature table for the UI: mean, std, p5, p50, p95."""
    out = {}
    for k, v in series.items():
        if len(v) < 2:
            continue
        p5, p50, p95 = np.percentile(v, [5, 50, 95])
        out[k] = {"mean": float(v.mean()), "std": float(v.std()),
                  "p5": float(p5), "p50": float(p50), "p95": float(p95)}
    return out


def decimate_series(times, arr, max_points=1500):
    step = max(1, len(times) // max_points)
    return times[::step].tolist(), np.round(arr[::step], 4).tolist()
