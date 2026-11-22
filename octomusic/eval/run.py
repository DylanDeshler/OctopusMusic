"""Find mapping/normalization settings without listening. Runs on the GPU box.

    # once: reference statistics from the training data
    python -m octomusic.eval.refstats --out eval_ref/refstats.npz
    # once: PiEEG's public recordings (plus any of yours under recordings/)
    python -m octomusic.eval.recordings fetch eval_data
    # the search
    python -m octomusic.eval.run --refstats eval_ref/refstats.npz \\
        --recordings eval_data/pieeg16_alpha eval_data/octopus16_pz_open_closed recordings/2026* \\
        --n-random 300 --top 6

STAGE 1 (CPU, seconds per config): every config's dial timeline is computed
over every recording with the client's own DSP + Mapper, then authored into
per-measure conditioning with the server's own LiveConditioner -- i.e. exactly
what the model would receive -- and scored: does the conditioning separate the
labelled states, how fast does it respond, how much of each band does it use,
is it on the training manifold, does it move like real music.

STAGE 2 (GPU): the top configs (plus baselines) are generated offline with the
real model and the AUDIO is scored: can the states be decoded from the music,
does the model follow each signal, CLAP realism/motion in the model's own
embedding space, loudness/clipping/looping.

Output in eval_runs/<stamp>/: report.md (start here), results.json, wav for
every stage-2 run (the listening shortlist), and configs/eval_*.json that the
UI's "Load config" opens directly (session stats included).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time

import numpy as np

from octomusic import protocol
from octomusic.client import offline as OFF
from octomusic.client.dsp import DEFAULT_DSP
from octomusic.eval import metrics as M
from octomusic.eval import space
from octomusic.eval.recordings import load as load_rec

GEN_DEFAULTS = {
    "seed": 0, "n_new": 8, "n_steps": 32, "guidance": 3.0, "hist_guidance": 1.5,
    "ctx_noise": 0.2, "mode": "locked", "smooth_s": 2.0, "glide": 0.5, "bpm_jitter": 0.0,
    "style_b": None, "style_text": "", "style_axis": "", "style_axis_deg": 60.0,
    "bands": None, "analyze_output": True, "time_scale": 1.0, "offset_s": 0.0,
}


# ---- preparation ----------------------------------------------------------------------

def prepare(paths, dsp_cfg):
    preps = []
    for p in paths:
        rec = load_rec(p)
        ex = OFF.extract(rec.t, rec.x, rec.fs, dsp_cfg)
        if len(ex["times"]) < 20:
            print(f"[prep] skip {p}: too short")
            continue
        preps.append({"rec": rec, "ex": ex, "frame_labels": rec.label_at(ex["times"])})
        labs = sorted(set(l for *_, l in rec.labels))
        print(f"[prep] {rec.name}: {rec.duration:.0f}s, {len(ex['times'])} hops, "
              f"artifact {100 * ex['artifact'].mean():.0f}%, labels {labs}")
    return preps


def frames_for(cfg_mapper, prep):
    m = cfg_mapper.clone_for_offline()
    m.set_session_stats(OFF.session_series(prep["ex"], m.routes))   # this recording's own stats
    return OFF.dial_frames(prep["ex"], m), m


# ---- stage 1 -----------------------------------------------------------------------------

def stage1_one(cfg, preps, ref, gen_defaults, max_music_s):
    mapper, gen = space.apply(cfg, gen_defaults)
    active = set(gen["active"]) | ({"style"} if gen["style_mode"] in ("morph", "axis") else set())
    per = []
    for prep in preps:
        frames, _ = frames_for(mapper, prep)
        ex = prep["ex"]
        dur = min(max_music_s, float(ex["times"][-1] - ex["times"][0]))
        traj = M.simulate(frames, gen, dur)
        fd = [f["values"] for f in frames]
        r = {
            "rec": prep["rec"].name,
            "plausibility": M.plausibility(ref, traj, active),
            "temporal": M.temporal(ref, traj, active),
            "band_usage": M.band_usage(traj, active),
            "separability": M.separability(traj, prep["rec"].label_at, active),
            "responsiveness": M.responsiveness(ex["times"], fd, prep["frame_labels"], active),
            "artifact_sensitivity": M.artifact_sensitivity(fd, ex["artifact"], active),
            "n_measures": len(traj["rms"]),
        }
        r["violations"] = M.violations_stage1(r)
        per.append(r)
    seps = [r["separability"]["norm"] for r in per if np.isfinite(r["separability"]["norm"])]
    agg = {
        "name": cfg["name"], "desc": space.describe(cfg, mapper), "config": cfg,
        "per_recording": per,
        "separability": float(np.mean(seps)) if seps else np.nan,
        "offmanifold": float(np.mean([r["plausibility"]["offmanifold_frac"] for r in per])),
        "t50_s": float(np.nanmedian([r["responsiveness"]["t50_median_s"] for r in per]
                                    + [np.nan])),
        "violations": sorted(set(v for r in per for v in r["violations"])),
    }
    usage = [b["entropy"] for r in per for b in r["band_usage"].values()]
    agg["usage"] = float(np.mean(usage)) if usage else 0.0
    base = agg["separability"] if np.isfinite(agg["separability"]) else 0.25 * agg["usage"]
    agg["score"] = float(base - 0.05 * len(agg["violations"]))
    return agg


def stage1(cfgs, preps, ref, gen_defaults, max_music_s):
    out = []
    t0 = time.time()
    for i, cfg in enumerate(cfgs):
        try:
            out.append(stage1_one(cfg, preps, ref, gen_defaults, max_music_s))
        except Exception as e:  # noqa: BLE001
            print(f"[stage1] {cfg['name']} failed: {type(e).__name__}: {e}")
        if (i + 1) % 25 == 0:
            print(f"[stage1] {i + 1}/{len(cfgs)} configs ({time.time() - t0:.0f}s)")
    return out


def pick_top(results, k):
    cand = [r for r in results if not r["name"].startswith("baseline:")]
    feasible = [r for r in cand if not r["violations"]]
    pool = sorted(feasible, key=lambda r: -r["score"]) + \
        sorted([r for r in cand if r["violations"]], key=lambda r: -r["score"])
    top = pool[:k]
    return top + [r for r in results if r["name"].startswith("baseline:")]


# ---- stage 2 -----------------------------------------------------------------------------

def generate(engine, frames, params):
    from octomusic.server.engine import Session
    msgs, chunks = [], []

    def send(o):
        if isinstance(o, tuple):
            chunks.append((o[2], np.asarray(o[3], np.float32)))
        else:
            msgs.append(o)

    sess = Session(engine, send)
    sess.start(params, "offline", None, frames)
    sess._thread.join()
    errs = [m for m in msgs if m.get("type") == "status" and m.get("state") == "error"]
    if errs:
        raise RuntimeError(errs[-1].get("msg"))
    n = max((s + len(c) for s, c in chunks), default=0)
    audio = np.zeros(n, np.float32)
    for s, c in chunks:
        audio[s:s + len(c)] = c
    return audio, msgs


def stage2_one(r1, preps, engine, ref, gen_defaults, max_music_s, outdir, embed):
    cfg = r1["config"]
    mapper, gen = space.apply(cfg, gen_defaults)
    active = set(gen["active"])
    sr = engine.B.SR
    per = []
    for prep in preps:
        frames, m_rec = frames_for(mapper, prep)
        ex = prep["ex"]
        dur = min(max_music_s, float(ex["times"][-1] - ex["times"][0]))
        params = {**gen, "duration_s": dur}
        t0 = time.time()
        audio, msgs = generate(engine, frames, params)
        takes = [m for m in msgs if m.get("type") == "take"]
        meas = {row["m"]: row for m in msgs if m.get("type") == "measured" for row in m["measures"]}
        rows = [row for tk in takes for row in tk["measures"]
                if row["start"] is not None and row["end"] is not None and row["end"] <= len(audio)]
        req = {s: np.array([row["vals"][s] for row in rows]) for s in ("rms", "density", "zcr", "flatness", "bpm")}
        mea = {s: np.array([meas.get(row["m"], {}).get(s, np.nan) for row in rows])
               for s in ("rms", "density", "zcr", "flatness")}
        mea["bpm"] = req["bpm"]
        durs = np.array([(row["end"] - row["start"]) / sr for row in rows])
        times = np.array([row["t"][1] for row in rows])
        emb = None
        if embed is not None and rows:
            try:
                emb = embed(audio, [(row["start"] / sr, row["end"] / sr) for row in rows])
            except Exception as e:  # noqa: BLE001
                print(f"[stage2] CLAP embed failed: {e}")
        loops = [tk.get("loop_cos", 0.0) for tk in takes if not tk.get("first")]
        traj = {**req, "dials": {k: np.array([row["dials"].get(k, np.nan) for row in rows])
                                 for k in ("rms", "bpm", "density", "zcr", "flatness", "chroma", "style")}}
        r = {
            "rec": prep["rec"].name, "music_s": len(audio) / sr, "gen_s": time.time() - t0,
            "audibility": M.audibility(mea, emb, durs, times, prep["rec"].label_at),
            "separability": M.separability(traj, prep["rec"].label_at, active, times=times),
            "compliance": M.compliance(ref, req, mea, active),
            "health": M.health(ref, audio, mea, loops),
            "clap": M.clap(ref, emb),
            "plausibility": M.plausibility(ref, req, active),
        }
        r["violations"] = M.violations_stage2(r)
        wav = os.path.join(outdir, "wav", f"{cfg['name'].replace(':', '_')}__{prep['rec'].name}.wav")
        os.makedirs(os.path.dirname(wav), exist_ok=True)
        import soundfile as sf
        sf.write(wav, protocol.soft_limit(audio), sr, subtype="PCM_16")
        r["wav"] = os.path.relpath(wav, outdir)
        per.append(r)
        print(f"[stage2] {cfg['name']} × {prep['rec'].name}: {r['music_s']:.0f}s music in "
              f"{r['gen_s']:.0f}s, audibility {r['audibility']['norm']}, violations {r['violations']}")
    # the config to load in the UI (session stats of the last recording)
    ui = space.to_ui_config(m_rec, DEFAULT_DSP, {k: v for k, v in gen.items() if k != "duration_s"})
    with open(os.path.join(outdir, "configs", f"eval_{cfg['name'].replace(':', '_')}.json"), "w") as f:
        f.write(protocol.dumps(ui))

    def mean(key, sub):
        v = [x[key][sub] for x in per if x[key] and np.isfinite(x[key][sub])]
        return float(np.mean(v)) if v else np.nan
    comp = [c["r"] for x in per for c in x["compliance"].values() if np.isfinite(c["r"])]
    return {
        "name": cfg["name"], "desc": r1["desc"], "per_recording": per,
        "audibility": mean("audibility", "norm"), "separability": mean("separability", "norm"),
        "compliance_r": float(np.mean(comp)) if comp else np.nan,
        "clap_unreal": mean("clap", "unreal_frac") if any(x["clap"] for x in per) else np.nan,
        "clap_fd_ratio": mean("clap", "fd_ratio") if any(x["clap"] for x in per) else np.nan,
        "clap_drift_ratio": mean("clap", "drift_ratio") if any(x["clap"] for x in per) else np.nan,
        "loop_cos": mean("health", "loop_cos_median"),
        "violations": sorted(set(v for x in per for v in x["violations"])),
    }


# ---- report --------------------------------------------------------------------------------

def fmt(v, nd=2):
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def report(outdir, s1, s2, args, ref_summary):
    L = ["# OctopusMusic mapping search", "",
         f"recordings: {', '.join(args.recordings)}  ",
         f"configs scored: {len(s1)} (stage 1), {len(s2)} generated (stage 2), "
         f"music per run ≤ {args.max_music_s:.0f}s", ""]
    if s2:
        ok = [r for r in s2 if not r["violations"] and not r["name"].startswith("baseline:")]
        best = max(ok, key=lambda r: (np.nan_to_num(r["audibility"], nan=-1), -np.nan_to_num(r["clap_fd_ratio"], nan=9)), default=None)
        L += ["## Recommendation", ""]
        if best:
            L += [f"**{best['name']}** — {best['desc']}", "",
                  f"audibility {fmt(best['audibility'])} (0 = chance, 1 = perfect), "
                  f"compliance r {fmt(best['compliance_r'])}, CLAP FD ratio {fmt(best['clap_fd_ratio'])} "
                  f"(1 = as real as real music), drift ratio {fmt(best['clap_drift_ratio'])}.  ",
                  f"Load `configs/eval_{best['name'].replace(':', '_')}.json` in the UI, or listen "
                  f"to `wav/{best['name'].replace(':', '_')}__*.wav`.", ""]
        else:
            L += ["No config passed every quality limit; see the table.", ""]
        L += ["## Stage 2 — generated audio", "",
              "| config | audibility | separability (cond.) | compliance r | CLAP unreal | CLAP FD ratio | drift ratio | loop cos | violations | mapping |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for r in sorted(s2, key=lambda r: -np.nan_to_num(r["audibility"], nan=-9)):
            L.append(f"| {r['name']} | {fmt(r['audibility'])} | {fmt(r['separability'])} | {fmt(r['compliance_r'])} | "
                     f"{fmt(r['clap_unreal'])} | {fmt(r['clap_fd_ratio'])} | {fmt(r['clap_drift_ratio'])} | "
                     f"{fmt(r['loop_cos'])} | {', '.join(r['violations']) or '—'} | {r['desc']} |")
        L += ["", "Per-signal compliance (r between requested and measured) for each run is in results.json.", ""]
    L += ["## Stage 1 — conditioning only (top 25)", "",
          "| config | score | separability | off-manifold | t50 s | band usage | violations | mapping |",
          "|---|---|---|---|---|---|---|---|"]
    for r in sorted(s1, key=lambda r: -r["score"])[:25]:
        L.append(f"| {r['name']} | {fmt(r['score'])} | {fmt(r['separability'])} | {fmt(r['offmanifold'])} | "
                 f"{fmt(r['t50_s'], 1)} | {fmt(r['usage'])} | {', '.join(r['violations']) or '—'} | {r['desc']} |")
    L += ["", "## How to read this", "",
          "- **audibility / separability**: leave-one-segment-out decoding of the labelled state, from the audio "
          "(measured rms/density/zcr/flatness, measure duration, CLAP PCs) or from the conditioning. "
          "Normalized: 0 = chance, 1 = perfect. Baselines should sit near 0 — if they don't, the labels leak "
          "through something other than the mapping.",
          "- **compliance r**: correlation between what was requested and what the generated audio measured, per "
          "active DSP signal. Low = the model ignores that signal at these settings.",
          "- **CLAP**: per-measure embeddings from the model's own style tower. *unreal* = fraction of measures less "
          "similar to any real training measure than real music's 1st percentile. *FD ratio* = shrinkage Fréchet "
          "distance to real measures ÷ the same for an equal-size real sample (1 ≈ indistinguishable). *drift ratio* "
          "= measure-to-measure motion ÷ real music's (≪1 static/looping, ≫1 incoherent).",
          "- **off-manifold**: fraction of measures outside the training conditioning density's 99th percentile.",
          "- **t50**: seconds for the dials to move half way to a new state's level after a label change "
          "(mapping only; online adds the fixed delay).", "",
          "Reference statistics: see refstats.json.", "",
          "```json", json.dumps(ref_summary, indent=1)[:4000], "```"]
    with open(os.path.join(outdir, "report.md"), "w") as f:
        f.write("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refstats", required=True)
    ap.add_argument("--recordings", nargs="+", required=True)
    ap.add_argument("--n-random", type=int, default=300)
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--max-music-s", type=float, default=120.0)
    ap.add_argument("--notch", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stage1-only", action="store_true")
    ap.add_argument("--mock", action="store_true", help="mock backend (plumbing test)")
    ap.add_argument("--configs", help="JSON list of configs to evaluate instead of sampling")
    ap.add_argument("--out", default="eval_runs")
    args = ap.parse_args()

    outdir = os.path.join(args.out, time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(os.path.join(outdir, "configs"), exist_ok=True)
    ref = M.Ref(args.refstats)
    ref_summary_path = os.path.splitext(args.refstats)[0] + ".json"
    ref_summary = json.load(open(ref_summary_path)) if os.path.exists(ref_summary_path) else {}
    dsp_cfg = {**DEFAULT_DSP, "notch_hz": args.notch}
    preps = prepare(args.recordings, dsp_cfg)
    if not preps:
        raise SystemExit("no usable recordings")
    if not any(p["rec"].labels for p in preps):
        print("[warn] no labelled recordings: separability/audibility can't be measured; "
              "stage 1 will rank by band usage only")

    cfgs = json.load(open(args.configs)) if args.configs else space.sample(args.n_random, args.seed)
    s1 = stage1(cfgs, preps, ref, GEN_DEFAULTS, args.max_music_s)
    s2 = []
    if not args.stage1_only:
        from octomusic.server.engine import Engine
        engine = Engine(mock=args.mock)
        embed = None
        if not args.mock:
            try:
                from preset_extractor import _extract_style
                embed = lambda a, mt: _extract_style(a, mt, sr=engine.B.SR)  # noqa: E731
            except Exception as e:  # noqa: BLE001
                print(f"[stage2] CLAP unavailable ({e}); realism metrics skipped")
        for r1 in pick_top(s1, args.top):
            try:
                s2.append(stage2_one(r1, preps, engine, ref, GEN_DEFAULTS, args.max_music_s, outdir, embed))
            except Exception as e:  # noqa: BLE001
                print(f"[stage2] {r1['name']} failed: {type(e).__name__}: {e}")
    with open(os.path.join(outdir, "results.json"), "w") as f:
        f.write(protocol.dumps({"stage1": s1, "stage2": s2, "args": vars(args)}))
    report(outdir, s1, s2, args, ref_summary)
    print(f"\n[done] {outdir}/report.md")


if __name__ == "__main__":
    main()
