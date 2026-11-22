"""LiveConditioner: the EEG counterpart of weather.WeatherTrajectory.

It plugs into the same seam -- `Track.cond_hook`, called by `_continue_step`
after the window slides and before the latents are read -- and keeps the same
two rules weather.py documents:

1. UNIFORM PRESENCE. Every signal is on for all 24 measures of the window or
   off for all 24. This checkpoint never saw a partially-present signal
   (multi_token_drop's per-measure path is dead code in both stages); a
   partial pattern comes back as new measures at clipping volume.

2. A PURE FUNCTION OF ABSOLUTE MEASURE, for measures that already exist. The
   context measures are frozen latents whose audio has been sent; their
   conditioning must be what they were generated with (for bpm it is their
   decoded length). EEG is only known up to "now", so this keeps a LEDGER:
   the first time a measure index is seen it is authored, every later time it
   is read back verbatim.

TIME. Every EEG frame carries the client's timestamp. Before each take the
engine sets `plan`: where in the song the new measures start and a function
song-seconds -> EEG-time. Online that function is "the wall time this sample
will play, minus the delay"; offline it is "the recording's start plus song
time times the time scale". Authoring modes:

  locked  each new measure is authored from the EEG at ITS OWN mapped time,
          averaged over `smooth_s` back from there. Measure durations depend on
          their own bpm, so the walk is sequential. Online this makes the
          brain-to-ear delay a constant; offline it lays the recording along
          the music measure by measure.
  hold    every new measure gets the EEG at the take's start time; the first
          `glide` fraction of the take ramps from the previous measure. A
          staircase per take -- kept for comparison.
"""

from __future__ import annotations

import bisect
import threading
import time

import numpy as np

from octomusic.model_space import (DEFAULT_BANDS, DSP_TARGETS, SCALAR_TARGETS,
                                   dial_to_value, measure_seconds)

DEFAULT_CFG = {
    "mode": "locked",        # locked | hold
    "smooth_s": 2.0,         # averaging window per measure, seconds of EEG
    "glide": 0.5,            # hold only
    "active": ["rms", "bpm", "density", "zcr", "flatness"],
    "bands": DEFAULT_BANDS,
    "bpm_jitter": 0.0,       # relative white jitter on bpm (training data ~0.015)
    "style_mode": "off",     # off | preset | morph | axis
    "style_axis_deg": 60.0,  # axis mode: angle at dial = 1
}


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / (float(np.linalg.norm(v)) + 1e-9)


def _slerp(a, b, t):
    a, b = _unit(a), _unit(b)
    d = float(np.clip(a @ b, -1.0, 1.0))
    om = np.arccos(d)
    if om < 1e-4:
        return _unit((1 - t) * a + t * b)
    return _unit((np.sin((1 - t) * om) * a + np.sin(t * om) * b) / np.sin(om))


def _rotate(a, axis, deg):
    a = _unit(a)
    p = np.asarray(axis, np.float32) - float(np.asarray(axis) @ a) * a
    p = _unit(p)
    th = np.deg2rad(deg)
    return _unit(np.cos(th) * a + np.sin(th) * p)


class LiveConditioner:
    def __init__(self, snap_bpm=None):
        self._lock = threading.Lock()
        self._t: list[float] = []            # client timestamps, sorted
        self._v: list[dict] = []
        self._c: list = []                   # chroma (12,) or None
        self._off: list[float] = []          # recv_server_time - client_t
        self._ledger: dict[int, dict] = {}
        self.cfg = dict(DEFAULT_CFG)
        self.style_a = self.style_b = self.style_axis = None
        self.bpm_fixed = None
        self.plan = None                     # {"start_s", "eeg_time"}
        self.keep_s = 900.0                  # online: frames kept
        self._snap = snap_bpm
        self._rng = np.random.default_rng(0)
        self.last_authored: list[int] = []

    def __deepcopy__(self, memo):           # shared, never cloned (see weather)
        return self

    # ---- input --------------------------------------------------------------
    def push(self, t, values, chroma=None):
        now = time.time()
        clean = {k: float(np.clip(v, 0.0, 1.0)) for k, v in (values or {}).items()
                 if isinstance(v, (int, float)) and np.isfinite(v)}
        ch = None if chroma is None else np.asarray(chroma, np.float32)
        with self._lock:
            t = float(t)
            if self._t and t < self._t[-1]:
                i = bisect.bisect(self._t, t)
                self._t.insert(i, t); self._v.insert(i, clean); self._c.insert(i, ch)
            else:
                self._t.append(t); self._v.append(clean); self._c.append(ch)
            self._off.append(now - t)
            self._off = self._off[-40:]
            while self._t and self._t[-1] - self._t[0] > self.keep_s:
                self._t.pop(0); self._v.pop(0); self._c.pop(0)

    def load(self, frames):
        """Offline: replace everything with a whole recording's frames."""
        with self._lock:
            fr = sorted(frames, key=lambda f: f["t"])
            self._t = [float(f["t"]) for f in fr]
            self._v = [{k: float(v) for k, v in (f.get("values") or {}).items()} for f in fr]
            self._c = [None if f.get("chroma") is None else np.asarray(f["chroma"], np.float32)
                       for f in fr]
            self._off = []
            self.keep_s = float("inf")

    def has_frames(self):
        return bool(self._t)

    def first_t(self):
        return self._t[0] if self._t else None

    def latest_t(self):
        return self._t[-1] if self._t else None

    def client_now(self):
        """Best estimate of the client's clock right now (offline: None)."""
        if not self._off:
            return None
        return time.time() - float(np.median(self._off))

    def configure(self, **cfg):
        with self._lock:
            for k, v in cfg.items():
                if k in DEFAULT_CFG:
                    self.cfg[k] = v

    def reset(self):
        with self._lock:
            self._ledger.clear()

    def record(self, m):
        return self._ledger.get(int(m))

    def forget_before(self, m):
        with self._lock:
            for k in [k for k in self._ledger if k < m]:
                del self._ledger[k]

    # ---- lookup -----------------------------------------------------------------
    def _mean(self, a, b):
        """Mean dials/chroma over client time [a, b]. Falls back to the frame
        nearest b when the window is empty. Returns (dials, chroma, early)."""
        if not self._t:
            return {}, None, True
        early = b > self._t[-1] + 0.3
        i0 = bisect.bisect_left(self._t, a)
        i1 = bisect.bisect_right(self._t, b)
        if i1 <= i0:
            j = min(max(i1 - 1, 0), len(self._t) - 1)
            i0, i1 = j, j + 1
        acc, n, chs = {}, {}, []
        for k in range(i0, i1):
            for key, v in self._v[k].items():
                acc[key] = acc.get(key, 0.0) + v
                n[key] = n.get(key, 0) + 1
            if self._c[k] is not None:
                chs.append(self._c[k])
        dials = {k: acc[k] / n[k] for k in acc}
        ch = np.mean(chs, axis=0) if chs else None
        return dials, ch, early

    # ---- authoring ----------------------------------------------------------------
    def _author(self, cond, idxs, first):
        cfg, bands = self.cfg, self.cfg["bands"]
        plan = self.plan or {}
        eeg_time = plan.get("eeg_time")
        s = float(plan.get("start_s", 0.0))
        w = max(0.25, float(cfg["smooth_s"]))
        prev = self._ledger.get(first + idxs[0] - 1)
        if eeg_time is None:                       # no clock yet: latest EEG
            lt = self.latest_t() or 0.0
            eeg_time = lambda _s: lt                # noqa: E731
        if cfg["mode"] == "hold":
            te = eeg_time(s)
            if self.latest_t() is not None:
                te = min(te, self.latest_t())
            target, tch, early = self._mean(te - w, te)
        n = len(idxs)
        for j, i in enumerate(idxs):
            if cfg["mode"] == "hold":
                g = float(np.clip(cfg["glide"], 0.0, 1.0))
                frac = 1.0 if g <= 0 else min(1.0, (j + 1) / max(1.0, g * n))
                dials = {k: (prev["dials"].get(k, v) if prev else v) * (1 - frac) + v * frac
                         for k, v in target.items()}
                ch, t_used = tch, te
            else:
                t_used = eeg_time(s)
                dials, ch, early = self._mean(t_used - w, t_used)
            vals = {}
            for sname in SCALAR_TARGETS:
                if sname in dials and (sname in cfg["active"] or sname != "bpm"):
                    vals[sname] = dial_to_value(sname, dials[sname], bands)
                else:
                    vals[sname] = float(getattr(cond, sname)[i])
            if "bpm" not in cfg["active"] and self.bpm_fixed:
                vals["bpm"] = float(self.bpm_fixed)
            if cfg["bpm_jitter"] > 0:
                vals["bpm"] *= float(1.0 + self._rng.normal() * cfg["bpm_jitter"])
            if self._snap:
                vals["bpm"] = float(self._snap(vals["bpm"]))
            style = None
            if self.style_a is not None:
                u = float(dials.get("style", 0.0))
                mode = cfg["style_mode"]
                if mode == "morph" and self.style_b is not None:
                    style = _slerp(self.style_a, self.style_b, u)
                elif mode == "axis" and self.style_axis is not None:
                    style = _rotate(self.style_a, self.style_axis, u * cfg["style_axis_deg"])
                else:
                    style = _unit(self.style_a)
            self._ledger[first + i] = {
                "dials": dict(dials), "vals": vals, "style": style,
                "chroma": None if ch is None else np.asarray(ch, np.float32),
                "t": [t_used - w, t_used], "early": bool(early), "song_s": s,
            }
            s += measure_seconds(vals["bpm"])
            prev = self._ledger[first + i]

    # ---- the cond_hook contract ----------------------------------------------------
    def __call__(self, cond, first: int):
        c = cond.copy()
        T = c.T
        with self._lock:
            new = [i for i in range(T) if (first + i) not in self._ledger]
            if new:
                self._author(c, new, first)
            self.last_authored = [first + i for i in new]
            cfg = self.cfg
            active = set(cfg["active"])
            have_chroma = True
            for i in range(T):
                rec = self._ledger[first + i]
                for sname, v in rec["vals"].items():
                    getattr(c, sname)[i] = v
                if rec["style"] is not None and cfg["style_mode"] != "off":
                    c.style[i] = rec["style"]
                if rec["chroma"] is not None:
                    c.chroma[i] = rec["chroma"]
                else:
                    have_chroma = False
            # window-wide presence -- rule 1
            for sname in DSP_TARGETS:
                c.active[sname][:] = sname in active
            c.active["bpm"][:] = "bpm" in active
            c.active["style"][:] = cfg["style_mode"] != "off" and self.style_a is not None
            # chroma only when EVERY measure in the window has a real one, so a
            # switch-on mid-run waits until the window is full rather than
            # conditioning the context on placeholder chords.
            c.active["chroma"][:] = "chroma" in active and have_chroma
        return c
