"""Reference statistics of what the model was trained on. Run once, on the GPU box:

    python -m octomusic.eval.refstats --out eval_ref/refstats.npz

Reads the same memmaps train_modern_fancy_dit_measures.py trains from (paths
and row counts copied from its get_batch; override with --data-dir). Uses the
VALIDATION split by default -- 99k measures is plenty for distributions and it
is the split the model was not fit to -- plus an optional train subsample.

Everything is computed WITHIN SONGS (clap_nopre_clap_index_{split}.npz), because
training windows never cross a song boundary (respect_song_boundaries=True),
so a measure-to-measure step across two songs is a configuration the model
never saw and must not leak into "what normal motion looks like".

Saved:
  scalars        the five per-measure conditioning scalars, in "feature space":
                 rms, density, zcr, log10(flatness), log2(bpm) -- log where the
                 signal spans decades (flatness) or octaves (bpm, which also has
                 the beat tracker's 2x doublings in it, kept as trained)
  mu, sd         per-dimension standardization
  ref_z          a random subsample of standardized measures (kNN density base)
  knn_null       kNN distances of held-out REAL measures to ref_z -- the null
                 distribution any generated trajectory is scored against
  steps_abs      |measure-to-measure step| per dimension, within songs (z units)
  win_std        per-24-measure-window std per dimension, within songs
  lag1           per-window lag-1 autocorrelation per dimension
  clap_ref       random L2-normalized CLAP audio-tower embeddings (style bin)
  clap_mu/cov    their mean/covariance (Fréchet distance)
  clap_knn_null  max-cos of held-out real embeddings to clap_ref
  clap_drift     measure-to-measure cosine distance within songs
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

DIMS = ("rms", "density", "zcr", "log_flatness", "log2_bpm")
N_CHUNKS = 24

ROWS = {"train": 4490789, "val": 99131}
FILES = {
    "meta": "low_large_24576_subset_chroma_rms_density_zcr_flatness_{split}.bin",
    "bpm": "low_large_24576_subset_adapter_longtrain_v2_64_bpm_{split}.bin",
    "style": "clap_nopre_clap_style_{split}.bin",
    "index": "clap_nopre_clap_index_{split}.npz",
}


def to_feature_space(rms, density, zcr, flatness, bpm):
    """The 5 scalars -> the space all plausibility stats live in."""
    return np.stack([np.asarray(rms, float), np.asarray(density, float),
                     np.asarray(zcr, float),
                     np.log10(np.maximum(np.asarray(flatness, float), 1e-6)),
                     np.log2(np.maximum(np.asarray(bpm, float), 1.0))], axis=-1)


def knn_dist(q, ref, k=10, chunk=2048):
    """Mean distance to the k nearest rows of `ref` (brute force, chunked)."""
    out = np.empty(len(q))
    r2 = (ref ** 2).sum(1)
    for i in range(0, len(q), chunk):
        a = q[i:i + chunk]
        d2 = (a ** 2).sum(1)[:, None] + r2[None, :] - 2 * a @ ref.T
        d2 = np.partition(np.maximum(d2, 0), k, axis=1)[:, :k]
        out[i:i + chunk] = np.sqrt(d2).mean(1)
    return out


def load_split(data_dir, split):
    n = ROWS[split]
    meta = np.memmap(os.path.join(data_dir, FILES["meta"].format(split=split)),
                     dtype=np.float32, mode="r", shape=(n, 16))
    bpm = np.memmap(os.path.join(data_dir, FILES["bpm"].format(split=split)),
                    dtype=np.float32, mode="r")
    style = np.memmap(os.path.join(data_dir, FILES["style"].format(split=split)),
                      dtype=np.float16, mode="r", shape=(n, 512))
    ix = np.load(os.path.join(data_dir, FILES["index"].format(split=split)), allow_pickle=True)
    start = np.asarray(ix["song_measure_start"], np.int64)
    stop = np.asarray(ix["song_measure_stop"], np.int64)
    assert len(bpm) == n, f"bpm rows {len(bpm)} != {n}"
    return meta, bpm, style, start, stop


def build(data_dir, split="val", n_ref=50000, n_null=5000, n_clap=20000, seed=0,
          train_extra=0):
    rng = np.random.default_rng(seed)
    t0 = time.time()
    meta, bpm, style, start, stop = load_split(data_dir, split)
    X = to_feature_space(meta[:, 12], meta[:, 13], meta[:, 14], meta[:, 15], bpm[:])
    rows = np.arange(len(X))
    if train_extra:
        tm, tb, _, _, _ = load_split(data_dir, "train")
        pick = np.sort(rng.choice(len(tb), train_extra, replace=False))
        X = np.concatenate([X, to_feature_space(tm[pick, 12], tm[pick, 13], tm[pick, 14],
                                                tm[pick, 15], tb[pick])])
    ok = np.all(np.isfinite(X), axis=1)
    mu, sd = X[ok].mean(0), X[ok].std(0) + 1e-9
    Z = (X - mu) / sd
    good = np.where(ok)[0]
    perm = rng.permutation(good)
    ref = Z[perm[:n_ref]]
    null = Z[perm[n_ref:n_ref + n_null]]
    knn_null = knn_dist(null, ref)

    # within-song temporal stats on the base split only
    Zs = Z[:len(rows)]
    steps, wstd, lag1 = [], [], []
    for s, e in zip(start, stop):
        if e - s < N_CHUNKS:
            continue
        seg = Zs[s:e]
        if not np.all(np.isfinite(seg)):
            continue
        steps.append(np.abs(np.diff(seg, axis=0)))
        for w0 in range(0, e - s - N_CHUNKS + 1, N_CHUNKS):
            w = seg[w0:w0 + N_CHUNKS]
            wstd.append(w.std(0))
            a, b = w[:-1] - w[:-1].mean(0), w[1:] - w[1:].mean(0)
            lag1.append((a * b).sum(0) / (np.sqrt((a * a).sum(0) * (b * b).sum(0)) + 1e-12))
    steps = np.concatenate(steps)
    steps = steps[rng.choice(len(steps), min(len(steps), 200000), replace=False)]

    # CLAP audio-tower reference
    cidx = np.sort(rng.choice(len(rows), n_clap + n_null, replace=False))
    S = np.asarray(style[cidx], dtype=np.float32)
    S = S[rng.permutation(len(S))]
    S /= np.linalg.norm(S, axis=1, keepdims=True) + 1e-9
    clap_ref, clap_null = S[:n_clap], S[n_clap:]
    clap_knn_null = (clap_null @ clap_ref.T).max(1)
    drift = []
    for s, e in zip(start, stop):
        if e - s < 2:
            continue
        v = np.asarray(style[s:e], dtype=np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
        drift.append(1.0 - (v[1:] * v[:-1]).sum(1))
        if sum(len(d) for d in drift) > 200000:
            break
    drift = np.concatenate(drift)

    out = dict(dims=np.array(DIMS), mu=mu, sd=sd, ref_z=ref.astype(np.float32),
               knn_null=knn_null, steps_abs=steps.astype(np.float32),
               win_std=np.asarray(wstd, np.float32), lag1=np.asarray(lag1, np.float32),
               clap_ref=clap_ref.astype(np.float16), clap_mu=clap_ref.mean(0),
               clap_cov=np.cov(clap_ref, rowvar=False), clap_knn_null=clap_knn_null,
               clap_drift=drift.astype(np.float32),
               pct=np.percentile(X[ok], [1, 2, 5, 25, 50, 75, 95, 98, 99], axis=0))
    print(f"[refstats] {split}: {ok.sum()} measures, {len(start)} songs, "
          f"{len(wstd)} windows, built in {time.time() - t0:.0f}s")
    return out


def summary(r):
    d = {}
    for i, name in enumerate(r["dims"]):
        d[str(name)] = {
            "mean": float(r["mu"][i]), "sd": float(r["sd"][i]),
            "p1_p50_p99": [float(r["pct"][0, i]), float(r["pct"][4, i]), float(r["pct"][8, i])],
            "step_abs_p50_p99_z": [float(np.percentile(r["steps_abs"][:, i], 50)),
                                   float(np.percentile(r["steps_abs"][:, i], 99))],
            "window_std_p50_z": float(np.median(r["win_std"][:, i])),
            "lag1_p50": float(np.nanmedian(r["lag1"][:, i])),
        }
    d["knn_null_p50_p95_p99"] = [float(np.percentile(r["knn_null"], q)) for q in (50, 95, 99)]
    d["clap_knn_null_p1_p50"] = [float(np.percentile(r["clap_knn_null"], q)) for q in (1, 50)]
    d["clap_drift_p50_p99"] = [float(np.percentile(r["clap_drift"], q)) for q in (50, 99)]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/data/binaries")
    ap.add_argument("--split", default="val")
    ap.add_argument("--train-extra", type=int, default=0,
                    help="also pool this many random train measures into the scalar stats")
    ap.add_argument("--out", default="eval_ref/refstats.npz")
    a = ap.parse_args()
    r = build(a.data_dir, a.split, train_extra=a.train_extra)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez_compressed(a.out, **r)
    s = summary(r)
    with open(os.path.splitext(a.out)[0] + ".json", "w") as f:
        json.dump(s, f, indent=1)
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
