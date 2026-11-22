"""EEG features -> model inputs. One ROUTE per model input.

THIS IS THE OPEN QUESTION, so it is built to be argued with live rather than
decided here. Every model input -- rms, bpm, density, zcr, flatness, chroma,
style -- has a route, and every part of a route is a UI control.

SOURCES (what the model is told):

  scalars (rms bpm density zcr flatness)
      off       not conditioned: the server drops the signal window-wide and the
                model makes it up (bpm keeps a fixed tempo, since the decoder needs
                a measure length either way)
      eeg       driven by a feature or an expression of features
      manual    the slider IS the dial -- hear what one signal does on its own
  chroma
      off       harmony free
      key       a fixed key template (root + major/minor)
      spectrum  the EEG's own spectrum folded into 12 bins (2–40 Hz, log-spaced)
                -- raw, unmusical, and a legitimate experiment
      dial_key  a dial walks the circle of fifths (driven like a scalar)
  style
      off | preset (A, constant) | morph (A→B or A→text by a dial)
      | axis (a named direction, rotated by up to N degrees by a dial)

DIALS. Driven routes produce a dial in 0..1 via

    u = clip( center + gain * SPREAD * z , 0, 1 )      then EMA-smoothed

which is weather's two-term normalizer: `center` is where the music sits at your
own baseline, `z` the deviation that makes anything move, `gain` weather's w (at
1, one standard deviation = 15% of the band). Normalizers differ only in how z
is obtained:

  adaptive          running mean/std, time constant tau_s. Self-calibrating;
                    adapts away sustained states after ~2 tau.
  baseline          mean/std from a calibration recording (Calibrate → Baseline).
  quantile          rank within the last window_s seconds.
  session_z         mean/std of the WHOLE analyzed recording (offline analysis).
  session_quantile  rank within the whole analyzed recording.
                    Both session_* are non-causal offline, and in live mode they
                    use the stats of the last recording you analyzed -- which is
                    how offline exploration hands its numbers to a live run.
  range             linear between recorded Low and High anchors.
  fixed             linear between hand-entered lo/hi feature values.

EXPRESSIONS. `feature` may be a single feature name or a linear combination,
"0.5*alpha_rel - beta_rel + emg_log". The combination is normalized as one
value, so every normalizer works on it.

ARTIFACTS. When most channels exceed the artifact threshold, driven routes hold
their value and stats freeze -- except routes reading emg_log, for which the
clench is the intended control.
"""

from __future__ import annotations

import copy
import math
import re
import time
from collections import deque

import numpy as np

from octomusic.model_space import CIRCLE_OF_FIFTHS, SCALAR_TARGETS, TARGETS, key_template

SPREAD = 0.15
NORMS = ["adaptive", "baseline", "quantile", "session_z", "session_quantile",
         "range", "fixed"]
SOURCES = {
    **{t: ["off", "eeg", "manual"] for t in SCALAR_TARGETS},
    "chroma": ["off", "key", "spectrum", "dial_key"],
    "style": ["off", "preset", "morph", "axis"],
}

ROUTE_DEFAULTS = {
    "source": "eeg",
    "drive": "eeg",           # chroma/style: what moves their dial (eeg | manual)
    "feature": "alpha_rel",   # name or linear expression
    "norm": "adaptive",
    "invert": False,
    "center": 0.5,
    "gain": 1.5,
    "smooth_s": 2.0,          # EMA on the dial
    "tau_s": 60.0,            # adaptive memory
    "window_s": 120.0,        # quantile window
    "fixed_lo": 0.0, "fixed_hi": 1.0,
    "manual": 0.5,
    # chroma
    "key_root": 0, "key_minor": False,
    "spec_gamma": 1.0,        # spectrum→chroma contrast (exponent)
    "key_hysteresis": 0.65,   # dial_key: fraction of a step needed to change key
}

# Argued defaults, the way weather.MAPPING was. Starting points, not truths.
PRESETS = {
    "features": {
        # brightness <- alpha, inverted: eyes closing is the most reliable effect
        # in EEG, so "close your eyes and it goes dark" is the first test.
        "zcr":      {"source": "eeg", "feature": "alpha_rel", "invert": True},
        # energy <- engagement index beta/(alpha+theta)
        "rms":      {"source": "eeg", "feature": "engage"},
        # density <- muscle: a jaw clench is deliberate and unmistakable, and
        # density is the one signal that literally counts events.
        "density":  {"source": "eeg", "feature": "emg_log", "gain": 2.0, "center": 0.35},
        # tempo <- spectral edge, slow so the most disruptive signal moves least
        "bpm":      {"source": "eeg", "feature": "sef90", "smooth_s": 6.0},
        # flatness <- spectral entropy: noise-like EEG, noise-like music
        "flatness": {"source": "eeg", "feature": "entropy"},
        "chroma":   {"source": "off"},
        "style":    {"source": "off", "drive": "eeg", "feature": "relax"},
    },
    "raw signal (EEG as audio)": {
        # The EEG waveform measured with the music's own descriptors.
        "rms":      {"source": "eeg", "feature": "sig_rms"},
        "zcr":      {"source": "eeg", "feature": "sig_zcr"},
        "flatness": {"source": "eeg", "feature": "sig_flatness"},
        "density":  {"source": "eeg", "feature": "sig_density"},
        "bpm":      {"source": "manual", "manual": 0.5},
        "chroma":   {"source": "spectrum"},
        "style":    {"source": "off"},
    },
    "brightness only": {
        "zcr":      {"source": "eeg", "feature": "alpha_rel", "invert": True},
        "rms": {"source": "off"}, "density": {"source": "off"},
        "flatness": {"source": "off"}, "bpm": {"source": "manual", "manual": 0.5},
        "chroma": {"source": "off"}, "style": {"source": "off"},
    },
    "style morph only": {
        **{t: {"source": "off"} for t in ("rms", "density", "zcr", "flatness", "chroma")},
        "bpm": {"source": "manual", "manual": 0.5},
        "style": {"source": "morph", "drive": "eeg", "feature": "relax"},
    },
    "all manual": {**{t: {"source": "manual"} for t in SCALAR_TARGETS},
                   "chroma": {"source": "key"}, "style": {"source": "preset"}},
    "unconditional": {**{t: {"source": "off"} for t in SCALAR_TARGETS},
                      "chroma": {"source": "off"}, "style": {"source": "off"}},
}

_TERM = re.compile(r"([+-]?)\s*(\d*\.?\d+)?\s*\*?\s*([A-Za-z_][A-Za-z0-9_]*)")


def parse_expr(expr: str):
    """'0.5*alpha_rel - beta_rel' -> [(0.5, 'alpha_rel'), (-1.0, 'beta_rel')]."""
    terms = []
    for sign, num, name in _TERM.findall(expr.replace(" ", "")):
        w = float(num) if num else 1.0
        terms.append((-w if sign == "-" else w, name))
    return terms


def eval_expr(expr: str, feats: dict):
    if expr in feats:
        return feats[expr]
    terms = parse_expr(expr)
    if not terms or any(n not in feats for _, n in terms):
        return None
    return float(sum(w * feats[n] for w, n in terms))


class FeatureStats:
    """Running statistics for one feature or expression."""

    def __init__(self):
        self.mean = None
        self.var = None
        self.hist = deque()           # (t, x)
        self.base = None              # (mean, std) from calibration
        self.anchors = {}             # {"low": median, "high": median}
        self.session = None           # (mean, std, sorted values) from offline

    def update(self, t, x, tau_s, dt=0.25):
        if self.mean is None:
            self.mean, self.var = x, 0.0
        else:
            a = 1.0 - math.exp(-dt / max(1.0, tau_s))
            d = x - self.mean
            self.mean += a * d
            self.var = (1 - a) * (self.var + a * d * d)
        self.hist.append((t, x))
        while self.hist and t - self.hist[0][0] > 600:
            self.hist.popleft()

    def z(self, kind, x, t, window_s):
        if kind == "baseline" and self.base is not None:
            m, s = self.base
            return (x - m) / max(s, 1e-9)
        if kind == "session_z" and self.session is not None:
            m, s, _ = self.session
            return (x - m) / max(s, 1e-9)
        if kind == "session_quantile" and self.session is not None:
            srt = self.session[2]
            q = float(np.searchsorted(srt, x)) / max(1, len(srt))
            return (q - 0.5) / 0.25
        if kind == "quantile":
            vals = [v for (tt, v) in self.hist if t - tt <= window_s]
            if len(vals) >= 8:
                return (float(np.mean(np.asarray(vals) < x)) - 0.5) / 0.25
            return 0.0
        if self.mean is None:                         # adaptive / fallbacks
            return 0.0
        return (x - self.mean) / max(math.sqrt(self.var or 0.0), 1e-9)

    def window(self, t, seconds):
        return np.asarray([v for (tt, v) in self.hist if t - tt <= seconds])


class Mapper:
    def __init__(self):
        self.routes = {t: dict(ROUTE_DEFAULTS) for t in TARGETS}
        self.apply_preset("features")
        self.stats: dict[str, FeatureStats] = {}
        self.dials = {t: 0.5 for t in TARGETS}
        self.z = {t: 0.0 for t in TARGETS}
        self.chroma = None
        self._key_idx = None
        self._last_t = None

    # ---- config ------------------------------------------------------------------
    def apply_preset(self, name):
        for t in TARGETS:
            self.routes[t] = {**ROUTE_DEFAULTS, **self.routes.get(t, {}),
                              **PRESETS[name].get(t, {})}

    def set_route(self, target, **kw):
        if target not in self.routes:
            return
        r = self.routes[target]
        for k, v in kw.items():
            if k not in ROUTE_DEFAULTS:
                continue
            d = ROUTE_DEFAULTS[k]
            r[k] = str(v) if isinstance(d, str) else bool(v) if isinstance(d, bool) \
                else int(v) if isinstance(d, int) else float(v)

    def active_targets(self):
        return [t for t, r in self.routes.items() if r["source"] != "off"]

    def _drive(self, t):
        r = self.routes[t]
        if t in SCALAR_TARGETS:
            return r["source"]                     # off | eeg | manual
        if r["source"] in ("off", "key", "spectrum", "preset"):
            return "off"                           # no dial needed
        return r["drive"]

    # ---- one hop -------------------------------------------------------------------
    def step(self, feats, artifact, t=None, spec12=None):
        """Advance one hop. Returns (dials, chroma vector or None)."""
        t = time.time() if t is None else t
        dt = 0.25 if self._last_t is None else min(5.0, max(1e-3, t - self._last_t))
        self._last_t = t
        if feats:
            names = set(feats) | {r["feature"] for r in self.routes.values()}
            for name in names:
                x = eval_expr(name, feats)
                if x is None:
                    continue
                if artifact and "emg_log" not in name:
                    continue
                tau = min((r["tau_s"] for r in self.routes.values()
                           if r["feature"] == name), default=60.0)
                self.stats.setdefault(name, FeatureStats()).update(t, x, tau, dt)
        for tgt, r in self.routes.items():
            drive = self._drive(tgt)
            if drive == "off":
                continue
            if drive == "manual":
                target, self.z[tgt] = float(r["manual"]), 0.0
            else:
                x = eval_expr(r["feature"], feats) if feats else None
                if x is None or (artifact and "emg_log" not in r["feature"]):
                    continue                       # hold
                target, self.z[tgt] = self._normalize(r, self.stats[r["feature"]], x, t)
            a = 1.0 - math.exp(-dt / max(0.05, float(r["smooth_s"])))
            self.dials[tgt] = float(np.clip(self.dials[tgt] + a * (target - self.dials[tgt]), 0, 1))
        self.chroma = self._chroma(spec12)
        return dict(self.dials), self.chroma

    def _normalize(self, r, st, x, t):
        n = r["norm"]
        if n in ("range", "fixed"):
            if n == "range" and {"low", "high"} <= set(st.anchors):
                lo, hi = st.anchors["low"], st.anchors["high"]
            else:
                lo, hi = r["fixed_lo"], r["fixed_hi"]
            u = (x - lo) / (hi - lo) if abs(hi - lo) > 1e-12 else 0.5
            if r["invert"]:
                u = 1 - u
            u = r["center"] + r["gain"] * (u - 0.5)
            return float(np.clip(u, 0, 1)), (u - r["center"]) / SPREAD
        z = float(np.clip(st.z(n, x, t, r["window_s"]), -6, 6))
        if r["invert"]:
            z = -z
        return float(np.clip(r["center"] + r["gain"] * SPREAD * z, 0, 1)), z

    def _chroma(self, spec12):
        r = self.routes["chroma"]
        src = r["source"]
        if src == "key":
            return key_template(r["key_root"], r["key_minor"])
        if src == "spectrum":
            if spec12 is None:
                return self.chroma
            v = np.clip(np.asarray(spec12, float), 0, 1) ** max(0.1, r["spec_gamma"])
            return 0.2 + 0.8 * v                  # keep within the template's range
        if src == "dial_key":
            u = self.dials["chroma"] * 11.999
            i = self._key_idx
            if i is None or abs(u - (i + 0.5)) > 0.5 + r["key_hysteresis"] * 0.5:
                i = int(u)
            self._key_idx = i
            return key_template(CIRCLE_OF_FIFTHS[i], r["key_minor"])
        return None

    # ---- calibration -----------------------------------------------------------------
    def calibrate(self, kind: str, seconds: float, t: float | None = None):
        t = time.time() if t is None else t
        out = {}
        for name, st in self.stats.items():
            w = st.window(t, seconds)
            if len(w) < 4:
                continue
            if kind == "baseline":
                st.base = (float(np.mean(w)), float(np.std(w)))
            else:
                st.anchors[kind] = float(np.median(w))
            out[name] = float(np.mean(w))
        return out

    def set_session_stats(self, series: dict):
        """series: {feature_or_expr: np.ndarray over a whole recording}."""
        for name, x in series.items():
            x = np.asarray(x, float)
            x = x[np.isfinite(x)]
            if len(x) < 8:
                continue
            st = self.stats.setdefault(name, FeatureStats())
            st.session = (float(x.mean()), float(x.std()), np.sort(x))

    def state(self):
        return {
            "routes": self.routes,
            "calib": {n: {"base": s.base, "anchors": s.anchors,
                          "session": None if s.session is None else s.session[:2]}
                      for n, s in self.stats.items()
                      if s.base or s.anchors or s.session is not None},
        }

    def load(self, d):
        for tgt, r in (d.get("routes") or {}).items():
            self.set_route(tgt, **r)
        for n, c in (d.get("calib") or {}).items():
            st = self.stats.setdefault(n, FeatureStats())
            st.base = tuple(c["base"]) if c.get("base") else None
            st.anchors = dict(c.get("anchors") or {})

    def clone_for_offline(self):
        """A fresh Mapper with this one's routes and calibrations, but no
        running state -- so an offline pass never disturbs the live one."""
        m = Mapper()
        m.routes = copy.deepcopy(self.routes)
        for n, st in self.stats.items():
            s = FeatureStats()
            s.base, s.anchors, s.session = st.base, dict(st.anchors), st.session
            m.stats[n] = s
        return m
