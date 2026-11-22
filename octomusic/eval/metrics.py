"""Metrics for choosing mappings without listening. Numpy only.

Three properties, matching what "good" means here (responsive, reasonable
quality, representative of the recorded signal):

  REPRESENTATIVE  can the recorded state be recovered from what we send
                  (conditioning) and, finally, from the audio itself?
                  -> `separability`, `audibility` (balanced accuracy,
                  leave-one-SEGMENT-out so neighbouring measures can't leak)
  RESPONSIVE      how fast the conditioning moves after the state changes,
                  and how much of each signal's band it actually uses
                  -> `responsiveness`, `band_usage`
  QUALITY         is the conditioning something the model saw in training,
                  does the model follow it, and is the audio healthy / real
                  in the model's own CLAP space
                  -> `plausibility`, `temporal`, `compliance`, `health`, `clap`

They are reported separately and combined in `score_*` as an objective
(representative) under constraints (quality), never as a plain weighted sum:
constant conditioning is perfectly plausible, perfectly followed and perfectly
healthy, and says nothing about the brain.
"""

from __future__ import annotations

import numpy as np

from octomusic.eval.refstats import knn_dist, to_feature_space
from octomusic.model_space import SCALAR_TARGETS, dial_to_value, snap_bpm

DIM_OF = {"rms": 0, "density": 1, "zcr": 2, "flatness": 3, "bpm": 4}


class Ref:
    """refstats.npz, with kNN nulls for any subset of dimensions."""

    def __init__(self, path):
        r = np.load(path)
        self.r = {k: r[k] for k in r.files}
        z = self.r["ref_z"]
        n_null = min(3000, len(z) // 10)
        self.base, self.null = z[n_null:], z[:n_null]
        self._null = {}
        self.clap_ref = self.r["clap_ref"].astype(np.float32)

    def z(self, vals):
        X = to_feature_space(vals["rms"], vals["density"], vals["zcr"], vals["flatness"], vals["bpm"])
        return (X - self.r["mu"]) / self.r["sd"]

    def knn_null(self, dims):
        key = tuple(dims)
        if key not in self._null:
            self._null[key] = knn_dist(self.null[:, dims], self.base[:, dims])
        return self._null[key]


# ---- conditioning trajectory (exactly what the server would author) -----------

class _Stub:
    def __init__(self, n, preset_vals):
        for s in SCALAR_TARGETS:
            setattr(self, s, np.full(n, preset_vals.get(s, 0.0), dtype=np.float64))


def simulate(frames, params, duration_s, time_scale=1.0, offset_s=0.0, preset_vals=None):
    """Run the server's LiveConditioner authoring over a dial timeline, offline
    mode (song time -> recording time), and return per-measure arrays."""
    from octomusic.model_space import bands_from_json
    from octomusic.server.conditioner import LiveConditioner
    c = LiveConditioner(snap_bpm=snap_bpm)
    c.load(frames)
    c.configure(mode=params.get("mode", "locked"), smooth_s=params.get("smooth_s", 2.0),
                glide=params.get("glide", 0.5), active=params["active"],
                bands=bands_from_json(params.get("bands")),
                bpm_jitter=0.0, style_mode=params.get("style_mode", "off"))
    pv = preset_vals or {"rms": 0.12, "density": 3.0, "zcr": 0.09, "flatness": 0.003, "bpm": 222.2}
    c.bpm_fixed = snap_bpm(pv["bpm"])
    t0 = c.first_t() + offset_s
    n = int(duration_s / 0.8) + 32            # upper bound; trimmed by song time
    c.plan = {"start_s": 0.0, "eeg_time": lambda s: t0 + s * time_scale}
    c._author(_Stub(n, pv), list(range(n)), 0)
    recs = [c.record(i) for i in range(n)]
    recs = [r for r in recs if r["song_s"] <= duration_s]
    out = {s: np.array([r["vals"][s] for r in recs]) for s in SCALAR_TARGETS}
    out["dials"] = {k: np.array([r["dials"].get(k, np.nan) for r in recs])
                    for k in ("rms", "bpm", "density", "zcr", "flatness", "chroma", "style")}
    out["song_s"] = np.array([r["song_s"] for r in recs])
    out["eeg_t"] = np.array([r["t"][1] for r in recs])
    return out


# ---- helpers ----------------------------------------------------------------------

def _auc(a, b):
    """P(a > b) with ties at 1/2; symmetric version max(AUC, 1-AUC) used below."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    allv = np.concatenate([a, b])
    ranks = allv.argsort().argsort() + 1.0
    # average ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, ranks)
    ranks = (sums / cnt)[inv]
    u = ranks[:len(a)].sum() - len(a) * (len(a) + 1) / 2
    return u / (len(a) * len(b))


def segments(labels):
    """Contiguous runs of the same label -> segment id per item (None = -1)."""
    seg = np.full(len(labels), -1)
    k, prev = -1, object()
    for i, l in enumerate(labels):
        if l is None:
            prev = object()
            continue
        if l != prev:
            k += 1
            prev = l
        seg[i] = k
    return seg


def decode_bacc(X, labels):
    """Leave-one-segment-out nearest-centroid on z-scored features ->
    (balanced accuracy, chance, n_classes). NaN when it can't be measured.

    A fold is scored only if every class still has training data without the
    held-out segment; a class with a single segment therefore can't be scored
    (holding it out leaves nothing to learn it from), and if any class goes
    unscored the whole result is NaN rather than a fake "always wrong"."""
    lab = np.asarray(labels, dtype=object)
    keep = np.array([l is not None for l in lab]) & np.all(np.isfinite(X), axis=1)
    X, lab = X[keep], lab[keep]
    classes = sorted(set(lab))
    if len(classes) < 2 or len(X) < 8:
        return np.nan, np.nan, len(classes)
    chance = 1.0 / len(classes)
    seg = segments(list(lab))
    pred = np.array([None] * len(lab), dtype=object)
    for sid in set(seg):
        tr, te = seg != sid, seg == sid
        if any(not (tr & (lab == c)).any() for c in classes):
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Z = (X - mu) / sd
        C = np.stack([Z[tr & (lab == c)].mean(0) for c in classes])
        d = ((Z[te][:, None, :] - C[None]) ** 2).sum(-1)
        # exact ties (e.g. constant features) -> no information -> chance
        tie = np.isclose(d.min(1), d.max(1))
        p = np.array(classes, dtype=object)[d.argmin(1)]
        p[tie] = "__tie__"
        pred[te] = p
    recalls = []
    for c in classes:
        m = (lab == c) & (pred != None)  # noqa: E711
        if not m.any():
            return np.nan, chance, len(classes)
        hit = pred[m] == c
        ties = pred[m] == "__tie__"
        recalls.append(float(np.mean(hit) + np.mean(ties) * chance))
    return float(np.mean(recalls)), chance, len(classes)


LAGS_S = (0.0, 1.0, 2.0, 3.0, 4.0, 6.0)


def best_over_lags(X, times, label_fn, lags=LAGS_S):
    """Representative-with-delay: decode labels taken `lag` seconds BEFORE each
    measure's EEG time, keep the best lag. Speed is judged separately (t50)."""
    best = (np.nan, np.nan, 0, np.nan)
    for lag in lags:
        b, ch, k = decode_bacc(X, label_fn(np.asarray(times) - lag))
        if np.isfinite(b) and (not np.isfinite(best[0]) or b > best[0]):
            best = (b, ch, k, lag)
    return best


def norm_bacc(b, chance):
    return np.nan if not np.isfinite(b) else (b - chance) / (1 - chance)


# ---- conditioning metrics ----------------------------------------------------------

def plausibility(ref: Ref, traj, active):
    dims = [DIM_OF[s] for s in SCALAR_TARGETS if s in active]
    if not dims or len(traj["rms"]) == 0:
        return {"offmanifold_frac": 0.0, "knn_pct_median": 50.0, "dims": []}
    Z = ref.z(traj)[:, dims]
    d = knn_dist(Z, ref.base[:, dims])
    null = ref.knn_null(dims)
    p99 = np.percentile(null, 99)
    pct = np.searchsorted(np.sort(null), d) / len(null) * 100
    return {"offmanifold_frac": float(np.mean(d > p99)), "knn_pct_median": float(np.median(pct)),
            "dims": dims}


def temporal(ref: Ref, traj, active):
    """Per active dim: jitter (steps beyond training p99), and how much the
    signal moves within a 24-measure window relative to real music."""
    out = {}
    Z = ref.z(traj)
    for s in SCALAR_TARGETS:
        if s not in active or len(Z) < 25:
            continue
        i = DIM_OF[s]
        st = np.abs(np.diff(Z[:, i]))
        p99 = np.percentile(ref.r["steps_abs"][:, i], 99)
        wins = [Z[k:k + 24, i].std() for k in range(0, len(Z) - 23, 24)]
        out[s] = {"jump_frac": float(np.mean(st > p99)),
                  "winstd_ratio": float(np.median(wins) / (np.median(ref.r["win_std"][:, i]) + 1e-9))}
    return out


def band_usage(traj, active):
    out = {}
    for s in list(SCALAR_TARGETS) + ["style", "chroma"]:
        if s not in active:
            continue
        u = traj["dials"][s]
        u = u[np.isfinite(u)]
        if len(u) < 4:
            continue
        h, _ = np.histogram(u, bins=10, range=(0, 1))
        p = h / max(1, h.sum())
        ent = float(-(p[p > 0] * np.log(p[p > 0])).sum() / np.log(10))
        out[s] = {"std": float(u.std()), "pinned": float(np.mean((u < 0.02) | (u > 0.98))),
                  "entropy": ent}
    return out


def separability(traj, label_fn, active, times=None):
    feats = [s for s in list(SCALAR_TARGETS) + ["style"] if s in active]
    if not feats:
        return {"bacc": np.nan, "norm": np.nan, "auc": {}, "lag_s": np.nan}
    X = np.stack([traj["dials"][s] for s in feats], axis=1)
    t = traj["eeg_t"] if times is None else times
    b, ch, k, lag = best_over_lags(X, t, label_fn)
    auc = {}
    lab = label_fn(np.asarray(t) - (lag if np.isfinite(lag) else 0.0))
    classes = sorted(set(l for l in lab if l is not None))
    if len(classes) == 2:
        a, bmask = lab == classes[0], lab == classes[1]
        for j, s in enumerate(feats):
            v = _auc(X[a, j], X[bmask, j])
            auc[s] = float(max(v, 1 - v)) if np.isfinite(v) else np.nan
    return {"bacc": b, "chance": ch, "norm": norm_bacc(b, ch), "auc": auc,
            "n_classes": k, "lag_s": lag}


def responsiveness(frame_t, frame_dials, frame_labels, active):
    """Seconds for the dial vector to cover half the distance between the
    previous and next segment means after each label change (frame level,
    i.e. mapping latency only; online mode adds the fixed delay on top)."""
    keys = [s for s in list(SCALAR_TARGETS) + ["style"] if s in active]
    if not keys:
        return {"t50_median_s": np.nan, "n": 0}
    D = np.stack([np.array([d.get(k, np.nan) for d in frame_dials]) for k in keys], axis=1)
    lab = list(frame_labels)
    seg = segments(lab)
    t50 = []
    ids = [s for s in sorted(set(seg)) if s >= 0]
    for a, b in zip(ids, ids[1:]):
        ia, ib = np.where(seg == a)[0], np.where(seg == b)[0]
        if lab[ia[0]] == lab[ib[0]] or len(ia) < 4 or len(ib) < 4:
            continue
        ma, mb = D[ia[len(ia) // 2:]].mean(0), D[ib[len(ib) // 2:]].mean(0)
        dvec = mb - ma
        n = np.linalg.norm(dvec)
        if n < 1e-3:
            continue
        proj = (D[ib] - ma) @ dvec / n ** 2
        hit = np.where(proj >= 0.5)[0]
        if len(hit):
            t50.append(frame_t[ib[hit[0]]] - frame_t[ib[0]])
    return {"t50_median_s": float(np.median(t50)) if t50 else np.nan, "n": len(t50)}


def artifact_sensitivity(frame_dials, artifact, active):
    keys = [s for s in list(SCALAR_TARGETS) + ["style"] if s in active]
    if not keys or artifact.sum() < 2 or (~artifact).sum() < 2:
        return np.nan
    D = np.stack([np.array([d.get(k, np.nan) for d in frame_dials]) for k in keys], axis=1)
    step = np.abs(np.diff(D, axis=0)).mean(1)
    a = artifact[1:]
    return float(np.nanmean(step[a]) / (np.nanmean(step[~a]) + 1e-9))


# ---- audio metrics (stage 2) ---------------------------------------------------------

def compliance(ref: Ref, requested, measured, active):
    """Per active DSP signal: does the model's output follow the request?"""
    out = {}
    Zr, Zm = ref.z(requested), ref.z(measured)
    for s in ("rms", "density", "zcr", "flatness"):
        if s not in active:
            continue
        i = DIM_OF[s]
        a, b = Zr[:, i], Zm[:, i]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 6 or a[ok].std() < 1e-6:
            out[s] = {"r": np.nan, "slope": np.nan, "req_std_z": float(a[ok].std()) if ok.any() else 0}
            continue
        r = float(np.corrcoef(a[ok], b[ok])[0, 1])
        slope = float(np.polyfit(a[ok], b[ok], 1)[0])
        out[s] = {"r": r, "slope": slope, "req_std_z": float(a[ok].std())}
    return out


def health(ref: Ref, audio, measured, loop_cos):
    rms = np.asarray(measured["rms"], float)
    return {
        "clip_frac": float(np.mean(np.abs(audio) > 0.99)) if len(audio) else np.nan,
        "loud_frac": float(np.mean(rms > ref.r["pct"][8, 0] * 1.25)) if len(rms) else np.nan,
        "silent_frac": float(np.mean(rms < ref.r["pct"][0, 0] * 0.5)) if len(rms) else np.nan,
        "loop_cos_median": float(np.median(loop_cos)) if len(loop_cos) else np.nan,
        "loop_cos_max": float(np.max(loop_cos)) if len(loop_cos) else np.nan,
    }


def clap(ref: Ref, emb):
    """Realism and motion in the model's own CLAP audio space."""
    if emb is None or len(emb) < 4:
        return None
    E = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    best = (E @ ref.clap_ref.T).max(1)
    null = np.sort(ref.r["clap_knn_null"])
    p1 = null[int(0.01 * len(null))]
    drift = 1.0 - (E[1:] * E[:-1]).sum(1)
    # Fréchet distance with diagonal shrinkage (few hundred samples vs 512 dims),
    # reported against the same estimator on an equal-size REAL sample.
    def fd(A, B, lam=0.5):
        from scipy import linalg
        ma, mb = A.mean(0), B.mean(0)
        def cov(X):
            S = np.cov(X, rowvar=False)
            return (1 - lam) * S + lam * np.diag(np.diag(S))
        Ca, Cb = cov(A), cov(B)
        cs = linalg.sqrtm(Ca @ Cb).real
        return float(((ma - mb) ** 2).sum() + np.trace(Ca + Cb - 2 * cs))
    rng = np.random.default_rng(0)
    n = len(E)
    R = ref.clap_ref
    i1 = rng.choice(len(R), n, replace=False)
    i2 = rng.choice(len(R), min(len(R), 4 * n), replace=False)
    fd_gen = fd(E, R[i2])
    fd_real = fd(R[i1], R[i2])
    return {"unreal_frac": float(np.mean(best < p1)),
            "nn_cos_median": float(np.median(best)),
            "drift_median": float(np.median(drift)),
            "drift_ratio": float(np.median(drift) / (np.median(ref.r["clap_drift"]) + 1e-9)),
            "fd": fd_gen, "fd_real_baseline": fd_real, "fd_ratio": fd_gen / max(fd_real, 1e-9)}


def audibility(measured, emb, durations, times, label_fn):
    """Decode the recorded state from the AUDIO: measured DSP signals,
    log measure duration (tempo is audible), and the first 8 CLAP PCs."""
    cols = [np.asarray(measured[s], float) for s in ("rms", "density", "zcr")]
    cols.append(np.log10(np.maximum(np.asarray(measured["flatness"], float), 1e-6)))
    cols.append(np.log(np.asarray(durations, float)))
    X = np.stack(cols, axis=1)
    if emb is not None and len(emb) == len(X):
        E = emb - emb.mean(0)
        U, S, Vt = np.linalg.svd(E, full_matrices=False)
        X = np.concatenate([X, E @ Vt[:8].T], axis=1)
    X = X[:, np.isfinite(X).all(0)] if np.isfinite(X).all(0).any() else X
    b, ch, k, lag = best_over_lags(X, times, label_fn)
    return {"bacc": b, "chance": ch, "norm": norm_bacc(b, ch), "n_classes": k, "lag_s": lag}


# ---- scoring -------------------------------------------------------------------------

LIMITS = {
    "offmanifold_frac": 0.10,   # measures outside the training density's 99th pct
    "jump_frac": 0.05,          # measure steps beyond training p99
    "winstd_ratio_max": 3.0,    # moving 3x more than real music within a window
    "pinned": 0.40,             # dial stuck at an end of its band
    "unreal_frac": 0.25,        # measures less CLAP-real than real music's 1st pct
    "clip_frac": 0.005,
    "loud_frac": 0.05,
    "loop_cos_median": 0.90,
}


def violations_stage1(m):
    v = []
    if m["plausibility"]["offmanifold_frac"] > LIMITS["offmanifold_frac"]:
        v.append("off-manifold")
    for s, t in m["temporal"].items():
        if t["jump_frac"] > LIMITS["jump_frac"]:
            v.append(f"{s}:jumpy")
        if t["winstd_ratio"] > LIMITS["winstd_ratio_max"]:
            v.append(f"{s}:jittery")
    for s, b in m["band_usage"].items():
        if b["pinned"] > LIMITS["pinned"]:
            v.append(f"{s}:pinned")
    return v


def score_stage1(m):
    """Objective: how well the conditioning separates the labelled states
    (normalized balanced accuracy, 0 = chance). Without labels: band usage."""
    sep = m["separability"]["norm"]
    usage = np.mean([b["entropy"] for b in m["band_usage"].values()]) if m["band_usage"] else 0.0
    base = sep if np.isfinite(sep) else 0.25 * usage
    return float(base - 0.5 * len(m["violations"]) * 0.1)


def violations_stage2(m):
    v = list(m.get("violations", []))
    h = m["health"]
    if h["clip_frac"] > LIMITS["clip_frac"]:
        v.append("clipping")
    if h["loud_frac"] > LIMITS["loud_frac"]:
        v.append("loud")
    if np.isfinite(h["loop_cos_median"]) and h["loop_cos_median"] > LIMITS["loop_cos_median"]:
        v.append("looping")
    c = m.get("clap")
    if c and c["unreal_frac"] > LIMITS["unreal_frac"]:
        v.append("unrealistic")
    return v
