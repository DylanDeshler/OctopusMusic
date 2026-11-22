"""Recorded sessions: listing, per-label analysis, and turning one into a baseline.

`analyze` answers "which features actually respond to my states, on this
board, on this head?" -- the question every mapping decision depends on, and
one PiEEG's sample data can't answer for you:

  per label        seconds, segments, artifact %, and each feature's median
  per feature      eta^2 (share of its variance explained by the label),
                   the best pairwise AUC, and Cohen's d for that pair
  overall          cross-validated decoding of the label from all features
                   (leave-one-segment-out, best over 0-6 s lags) -- the same
                   number the hyper-parameter search optimizes, so a session
                   that decodes near chance here won't rescue any mapping

`apply_baseline` writes a session into the live Mapper: one label's mean/std
become the `baseline` normalizer, two labels' medians become the `range`
normalizer's low/high anchors, and the whole session becomes the reference for
`session_z` / `session_quantile`.
"""

from __future__ import annotations

import json
import os

import numpy as np

from octomusic.client import offline as OFF
from octomusic.eval.metrics import _auc, best_over_lags

KEY_FEATURES = [
    "alpha_rel", "beta_rel", "theta_rel", "delta_rel", "relax", "engage", "theta_beta",
    "focus_idx", "relax_idx", "meditation_idx", "sef90", "entropy", "iaf", "total_log",
    "emg_log", "gamma_rel", "sig_rms", "sig_zcr", "sig_flatness", "sig_density",
]


def list_sessions(root="recordings"):
    out = []
    if not os.path.isdir(root):
        return out
    dirs = [d for d in os.listdir(root) if os.path.isfile(os.path.join(root, d, "eeg.csv"))]
    for d in sorted(dirs, key=lambda d: os.path.getmtime(os.path.join(root, d, "eeg.csv")), reverse=True):
        p = os.path.join(root, d)
        if not os.path.isfile(os.path.join(p, "eeg.csv")):
            continue
        meta, labels = {}, []
        try:
            meta = json.load(open(os.path.join(p, "meta.json")))
        except (OSError, ValueError):
            pass
        try:
            labels = json.load(open(os.path.join(p, "labels.json")))
        except (OSError, ValueError):
            pass
        per = {}
        for t0, t1, lab in labels:
            e = per.setdefault(lab, [0.0, 0])
            e[0] += t1 - t0
            e[1] += 1
        dur = (meta.get("stopped", 0) - meta.get("started", 0)) if meta.get("stopped") else None
        if dur is None:
            dur = os.path.getsize(os.path.join(p, "eeg.csv")) / (16 * 9 + 16) / 250.0
        out.append({"name": d, "subject": meta.get("subject", ""), "placement": meta.get("placement", ""),
                    "protocol": meta.get("protocol", ""), "notes": meta.get("notes", ""),
                    "duration_s": round(float(dur), 1),
                    "labels": {k: {"seconds": round(v[0], 1), "segments": v[1]} for k, v in per.items()}})
    return out


class SessionAnalysis:
    def __init__(self, path, dsp_cfg):
        from octomusic.eval.recordings import load_dir
        self.path = path
        self.rec = load_dir(path)
        self.ex = OFF.extract(self.rec.t, self.rec.x, self.rec.fs, dsp_cfg)
        self.labels = self.rec.label_at(self.ex["times"])
        self.clean = ~self.ex["artifact"]

    def mask(self, lab):
        return np.array([l == lab for l in self.labels]) & self.clean

    def report(self):
        ex, lab = self.ex, self.labels
        classes = sorted(set(l for l in lab if l is not None))
        per_label = {}
        for c in classes:
            m_all = np.array([l == c for l in lab])
            m = m_all & self.clean
            segs = [s for s in self.rec.labels if s[2] == c]
            per_label[c] = {
                "seconds": round(float(sum(t1 - t0 for t0, t1, _ in segs)), 1),
                "segments": len(segs),
                "artifact_pct": round(100 * float(1 - m.sum() / max(1, m_all.sum())), 1),
                "medians": {f: float(np.median(ex["feats"][f][m])) for f in KEY_FEATURES
                            if f in ex["feats"] and m.any()},
            }
        feats = []
        for f in KEY_FEATURES:
            if f not in ex["feats"] or len(classes) < 2:
                continue
            v = ex["feats"][f]
            groups = [v[self.mask(c)] for c in classes]
            if any(len(g) < 3 for g in groups):
                continue
            allv = np.concatenate(groups)
            ss_tot = ((allv - allv.mean()) ** 2).sum() + 1e-12
            ss_b = sum(len(g) * (g.mean() - allv.mean()) ** 2 for g in groups)
            best = (0.5, None, 0.0)
            for i in range(len(classes)):
                for j in range(i + 1, len(classes)):
                    a, b = groups[i], groups[j]
                    auc = _auc(a, b)
                    auc = max(auc, 1 - auc)
                    if auc > best[0]:
                        sp = np.sqrt((a.var() + b.var()) / 2) + 1e-12
                        best = (auc, f"{classes[i]} vs {classes[j]}", float((a.mean() - b.mean()) / sp))
            feats.append({"feature": f, "eta2": float(ss_b / ss_tot), "auc": float(best[0]),
                          "pair": best[1], "d": best[2]})
        feats.sort(key=lambda r: -r["eta2"])
        decode = {"bacc": None, "chance": None, "lag_s": None, "norm": None}
        if len(classes) >= 2:
            X = np.stack([ex["feats"][f] for f in KEY_FEATURES if f in ex["feats"]], axis=1)
            keep = self.clean
            t = ex["times"][keep]
            b, ch, k, lag = best_over_lags(X[keep], t, self.rec.label_at)
            if np.isfinite(b):
                decode = {"bacc": b, "chance": ch, "lag_s": lag, "norm": (b - ch) / (1 - ch)}
        warn = []
        for c, d in per_label.items():
            if d["segments"] < 2:
                warn.append(f"'{c}' has only {d['segments']} segment: the search can't cross-validate it")
            if d["artifact_pct"] > 30:
                warn.append(f"'{c}' is {d['artifact_pct']:.0f}% artifacts: check contact / stillness")
        if not classes:
            warn.append("no labels in this session")
        return {"type": "session_report", "name": os.path.basename(self.path.rstrip("/")),
                "duration_s": round(self.rec.duration, 1), "classes": classes,
                "artifact_pct": round(100 * float(self.ex["artifact"].mean()), 1),
                "per_label": per_label, "features": feats[:20], "decode": decode, "warnings": warn}

    def apply_baseline(self, mapper, baseline=None, low=None, high=None):
        """Write this session's numbers into the live mapper. Returns what changed."""
        from octomusic.client.mapping import FeatureStats
        done = {"baseline": baseline, "low": low, "high": high, "features": 0}
        series = OFF.session_series(self.ex, mapper.routes)
        mapper.set_session_stats(series)
        for f, v in self.ex["feats"].items():
            st = mapper.stats.setdefault(f, FeatureStats())
            if baseline and self.mask(baseline).sum() >= 4:
                x = v[self.mask(baseline)]
                st.base = (float(x.mean()), float(x.std()))
            for kind, lab in (("low", low), ("high", high)):
                if lab and self.mask(lab).sum() >= 4:
                    st.anchors[kind] = float(np.median(v[self.mask(lab)]))
            done["features"] += 1
        return done
