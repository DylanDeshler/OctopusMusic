"""The search space: which feature drives which model input, how it is
normalized and smoothed, and which inputs are left unconditional.

A config is plain JSON:
    {"name": ..., "preset": "features", "routes": {target: {route overrides}},
     "gen": {generation/authoring overrides}}
`apply()` turns it into a Mapper + gen params exactly as the UI would, and
`to_ui_config()` writes it in the format the UI's "Load config" reads, so a
winner can be loaded and listened to live in one click.
"""

from __future__ import annotations

import copy
import random

from octomusic.client.mapping import PRESETS, Mapper
from octomusic.model_space import SCALAR_TARGETS

# Features worth trying per input. Broad on purpose; stage 1 is cheap.
EEG_FEATURES = [
    "alpha_rel", "beta_rel", "theta_rel", "delta_rel", "relax", "engage", "theta_beta",
    "focus_idx", "relax_idx", "meditation_idx", "sef90", "entropy", "total_log",
    "emg_log", "sig_rms", "sig_zcr", "sig_flatness", "sig_density",
]
NORMS = [("adaptive", {"tau_s": 30.0}), ("adaptive", {"tau_s": 90.0}),
         ("adaptive", {"tau_s": 240.0}), ("session_z", {}), ("session_quantile", {})]

BASELINES = [
    # "nothing from the brain": calibrates audibility's floor and quality's ceiling
    {"name": "baseline:unconditional", "preset": "unconditional", "routes": {}, "gen": {}},
    {"name": "baseline:constant", "preset": "all manual",
     "routes": {"chroma": {"source": "off"}, "style": {"source": "off"}}, "gen": {}},
]
NAMED = [{"name": f"preset:{p}", "preset": p, "routes": {}, "gen": {}}
         for p in ("features", "raw signal (EEG as audio)", "brightness only")]


def random_config(rng: random.Random, i: int):
    routes = {}
    for t in SCALAR_TARGETS:
        p_off = 0.45 if t == "bpm" else 0.3
        r = rng.random()
        if r < p_off:
            routes[t] = {"source": "off"} if t != "bpm" else {"source": "manual", "manual": 0.5}
            continue
        norm, extra = rng.choice(NORMS)
        routes[t] = {
            "source": "eeg", "feature": rng.choice(EEG_FEATURES), "norm": norm,
            "gain": rng.choice([1.0, 1.5, 2.5, 4.0]), "center": rng.choice([0.35, 0.5, 0.65]),
            "smooth_s": rng.choice([0.5, 2.0, 5.0]) * (2.0 if t == "bpm" else 1.0),
            "invert": rng.random() < 0.5, **extra,
        }
    if all(routes[t]["source"] != "eeg" for t in SCALAR_TARGETS):
        routes["zcr"] = {"source": "eeg", "feature": "alpha_rel", "norm": "session_z",
                         "gain": 2.5, "invert": True}
    routes["chroma"] = {"source": rng.choice(["off", "off", "spectrum", "dial_key"]),
                        "drive": "eeg", "feature": rng.choice(EEG_FEATURES), "norm": "session_quantile"}
    routes["style"] = {"source": "off"}
    gen = {"smooth_s": rng.choice([1.0, 2.0, 4.0]), "mode": "locked"}
    return {"name": f"rand{i:03d}", "preset": "unconditional", "routes": routes, "gen": gen}


def sample(n, seed=0):
    rng = random.Random(seed)
    return BASELINES + NAMED + [random_config(rng, i) for i in range(n)]


def apply(cfg, gen_defaults):
    m = Mapper()
    m.apply_preset(cfg.get("preset", "features"))
    for t, r in cfg.get("routes", {}).items():
        m.set_route(t, **r)
    gen = {**copy.deepcopy(gen_defaults), **cfg.get("gen", {})}
    gen["active"] = m.active_targets()
    gen["style_mode"] = m.routes["style"]["source"]
    return m, gen


def describe(cfg, mapper):
    parts = []
    for t in list(SCALAR_TARGETS) + ["chroma", "style"]:
        r = mapper.routes[t]
        if r["source"] == "off":
            continue
        if r["source"] == "eeg":
            parts.append(f"{t}←{'-' if r['invert'] else ''}{r['feature']}[{r['norm']},g{r['gain']:g},s{r['smooth_s']:g}]")
        else:
            parts.append(f"{t}:{r['source']}")
    return "; ".join(parts) or "all unconditional"


def to_ui_config(mapper, dsp_cfg, gen):
    g = {k: v for k, v in gen.items() if k not in ("active", "style_mode")}
    return {"mapper": mapper.state(), "dsp": dsp_cfg, "gen": g}


assert set(PRESETS) >= {"features", "unconditional", "all manual"}
