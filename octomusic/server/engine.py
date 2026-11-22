"""The generation side: load the model once, run one live session at a time.

This reuses the Neural Music Generator code as-is (put its checkout on
NMG_PATH). Nothing in that repo is modified; we drive `RealBackend` directly,
without the ZeroGPU wrapper, which removes the per-leg GPU reservation, the
pickling of the whole Track across a process boundary, and the
STREAM_LEG_MEASURES / headroom machinery that existed only to live inside
ZeroGPU quotas. On a dedicated GPU the loop is just: wait until the client's
buffer needs audio, run one `_continue_step`, send the new measures.
"""

from __future__ import annotations

import glob
import os
import sys
import threading
import time
import traceback

import numpy as np

from octomusic.server.conditioner import DEFAULT_CFG, LiveConditioner

# Parameters the client may set. Applied at the start of the next take.
DEFAULT_PARAMS = {
    "preset": None,           # seeds the track; style A
    "style_b": None,          # morph target preset ...
    "style_text": "",         # ... or text (CLAP text tower), if style_b unset
    "style_axis": "",         # axis mode: a label like "→ brighter" (STYLE_AXES)
    "seed": 0,
    "n_new": 8,               # measures written per take (context = 24 - n_new)
    "n_steps": 32,            # sampler steps; 16 = "draft"
    "guidance": 3.0,          # signal CFG
    "hist_guidance": 1.5,     # history guidance (1.0 = off, saves DiT passes)
    "ctx_noise": 0.2,         # noise the frozen context is held at
    # online
    "delay_s": 20.0,          # fixed brain->ear delay; the playout buffer
    "auto_draft": True,       # halve steps for a take that would arrive late
    "lead_in_measures": 6,    # of the opening render, how much to play
    # offline
    "time_scale": 1.0,        # seconds of recording per second of music
    "offset_s": 0.0,          # where in the recording to start
    "duration_s": None,       # music to generate; None = whole recording
    "analyze_output": True,   # measure what the model actually played
    **{k: v for k, v in DEFAULT_CFG.items()},
}
_COND_KEYS = set(DEFAULT_CFG)


def _add_nmg_path():
    """Where the model code lives: $NMG_PATH if set, else the copy bundled in
    this repo (OctopusMusic/nmg), so a fresh GPU box needs nothing else."""
    here = os.path.dirname(os.path.abspath(__file__))
    bundled = os.path.abspath(os.path.join(here, "..", "..", "nmg"))
    p = os.environ.get("NMG_PATH") or bundled
    if not os.path.isfile(os.path.join(p, "real_backend.py")):
        raise RuntimeError(f"model code not found in {p} (set NMG_PATH, or restore OctopusMusic/nmg)")
    if p not in sys.path:
        sys.path.insert(0, p)
    return p


def _resolve_ckpt(nmg, env_var, default):
    """Local path if it exists (as given, or inside NMG_PATH), else download
    from MODEL_REPO on the Hugging Face hub -- same rule as app_hf.py."""
    val = os.environ.get(env_var, default)
    for cand in (val, os.path.join(nmg, val)):
        if os.path.exists(cand):
            return cand
    from huggingface_hub import hf_hub_download
    repo = os.environ.get("MODEL_REPO", "ddeshler/NeuralMusic")
    print(f"[engine] fetching {val} from {repo} ...", flush=True)
    return hf_hub_download(repo_id=repo, filename=val, repo_type="model",
                           token=os.environ.get("HF_TOKEN"))


class Engine:
    def __init__(self, mock: bool = False):
        self.nmg = _add_nmg_path()
        import backend as B
        self.B = B
        self.mock = mock
        preset_dir = os.environ.get("PRESET_DIR", os.path.join(self.nmg, "presets"))
        self.presets = {}
        for path in sorted(glob.glob(os.path.join(preset_dir, "*.npz"))):
            name = os.path.splitext(os.path.basename(path))[0]
            try:
                self.presets[name] = B.Conditioning.from_npz(path)
            except Exception as e:  # noqa: BLE001
                print(f"[engine] skip preset {name}: {e}")
        if mock:
            self.backend = B.MockBackend()
            if not self.presets:
                self.presets = {n: self.backend.seed_from_preset(n, 0).cond
                                for n in self.backend.list_presets()}
            print("[engine] MOCK backend (no model, synthetic audio)")
        else:
            self.backend = self._load_real()
        self.lock = threading.Lock()     # one generation at a time on the GPU
        try:
            from backend import STYLE_AXIS_RANDOM, style_axis_labels
            self.style_axes = [x for x in style_axis_labels() if x != STYLE_AXIS_RANDOM]
        except Exception:  # noqa: BLE001
            self.style_axes = []

    def _load_real(self):
        nmg = self.nmg
        os.environ["STYLE_CKPT"] = _resolve_ckpt(nmg, "STYLE_CKPT", "style.pt")
        # CLAP modality centroids for text style (same files app_hf.py uses)
        for cand in ("clap_nopre_clap_modality_train.npz", "clap_nopre_clap_modality_val.npz"):
            if os.path.exists(os.path.join(nmg, cand)) and "MODALITY_NPZ" not in os.environ:
                os.environ["MODALITY_NPZ"] = os.path.join(nmg, cand)
        from real_backend import RealBackend
        # live mode is stage 2 only: don't download or load stage 1 unless asked
        ckpts = {2: _resolve_ckpt(nmg, "STAGE2_CKPT", "stage2.pt")}
        if os.environ.get("LOAD_STAGE1", "0") == "1":
            ckpts[1] = _resolve_ckpt(nmg, "STAGE1_CKPT", "stage1.pt")
        be = RealBackend(
            model_ckpts=ckpts, default_stage=2,
            tokenizer_ckpt=_resolve_ckpt(nmg, "TOKENIZER_CKPT", "tokenizer.pt"),
            adapter_ckpt=_resolve_ckpt(nmg, "ADAPTER_CKPT", "adapter.pt"),
            preset_cond=self.presets,
            default_cfg_mode=os.environ.get("CFG_MODE", "joint"),
            default_guidance=float(os.environ.get("GUIDANCE", "3.0")),
            default_t_dist=os.environ.get("T_DIST", "logit"),
        )
        t0 = time.time()
        be.to_gpu()
        print(f"[engine] RealBackend on {be.device} ({time.time() - t0:.1f}s)",
              flush=True)
        return be

    def info(self):
        return {"backend": "mock" if self.mock else "real",
                "device": getattr(self.backend, "device", "cpu"),
                "presets": list(self.presets),
                "defaults": DEFAULT_PARAMS, "sr": self.B.SR,
                "style_axes": self.style_axes,
                "n_chunks": self.B.N_CHUNKS}

    def style_vector(self, preset):
        c = self.presets.get(preset)
        if c is None:
            return None
        v = np.asarray(c.style, dtype=np.float32).mean(axis=0)
        return v / (float(np.linalg.norm(v)) + 1e-9)


class Session:
    """One run, online or offline. `send(obj_or_bytes)` must be thread-safe.

    ONLINE keeps a constant brain-to-ear delay D instead of a small buffer:

      * the client starts its speaker exactly D after pressing Start and
        reports an ANCHOR -- stream sample s plays at client time t -- which
        it re-sends if playback ever has to re-buffer;
      * a measure that will play at wall time p is authored from the EEG at
        p - D ("locked" mode), so the delay is the same for every measure;
      * the server therefore WAITS until the EEG for the next take's last
        measure exists, then generates. Audio for that take is ready
        `compute` seconds later and starts playing D - take_len after the
        EEG it needs was recorded, so the slack is D - take_len - compute.
        D below take_len + compute cannot be smooth; the UI shows that floor.
      * if slack runs low anyway (a slow take, a GPU hiccup) the server stops
        waiting for EEG, uses the newest it has (flagged "early"), and with
        auto_draft halves the steps for that take.

    OFFLINE has no clock to keep: the recording's frames are loaded whole,
    song time maps to recording time by `time_scale`, and takes run back to
    back until `duration_s` of music exists.
    """

    def __init__(self, engine: Engine, send):
        self.E, self.B, self.send = engine, engine.B, send
        self.params = dict(DEFAULT_PARAMS)
        if self.params["preset"] is None and engine.presets:
            self.params["preset"] = next(iter(engine.presets))
        self.cond = LiveConditioner(snap_bpm=self.B.snap_bpm)
        self.mode = "online"
        self.anchor = None            # (stream sample, client time)
        self.t_start = None
        self.played = 0
        self.emitted = 0
        self.seq = 0
        self.compute_est = None
        self._style_key = None
        self._thread = None
        self._stop = threading.Event()
        self.state = "idle"

    # ---- control --------------------------------------------------------------
    def set_params(self, p: dict):
        for k, v in (p or {}).items():
            if k in DEFAULT_PARAMS:
                self.params[k] = v
        self._apply_cond_cfg()

    def _apply_cond_cfg(self):
        from octomusic.model_space import bands_from_json
        p = self.params
        cfg = {k: p[k] for k in _COND_KEYS}
        cfg["bands"] = bands_from_json(p.get("bands"))
        self.cond.configure(**cfg)
        key = (p.get("preset"), p.get("style_b"), p.get("style_text"),
               p.get("style_axis"), p.get("style_mode"))
        if key == self._style_key:
            return
        self._style_key = key
        be = self.E.backend
        self.cond.style_a = self.E.style_vector(p.get("preset"))
        self.cond.style_b = self.E.style_vector(p.get("style_b")) if p.get("style_b") else None
        if self.cond.style_b is None and p.get("style_text") and p.get("style_mode") == "morph":
            try:
                self.cond.style_b = be.embed_style_text(p["style_text"])
            except Exception as e:  # noqa: BLE001
                self.send({"type": "status", "state": self.state,
                           "msg": f"style text unavailable: {e}"})
        self.cond.style_axis = None
        if p.get("style_mode") == "axis" and p.get("style_axis"):
            try:
                self.cond.style_axis = be.style_axis_vector(p["style_axis"])
            except Exception as e:  # noqa: BLE001
                self.send({"type": "status", "state": self.state,
                           "msg": f"style axis unavailable: {e}"})
        c = self.E.presets.get(p.get("preset"))
        if c is not None:
            med = float(np.median(np.asarray(c.bpm, float)))
            while med > 300:
                med /= 2.0                     # beat-tracker doubling
            self.cond.bpm_fixed = self.B.snap_bpm(med)

    def start(self, p=None, mode="online", t_start=None, frames=None):
        self.stop()
        self.set_params(p or {})
        self.mode = mode
        self.t_start = float(t_start) if t_start else None
        self.anchor = None
        if mode == "offline":
            self.cond.load(frames or [])
        else:
            self.cond.keep_s = 900.0
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=120)
        self._thread = None

    def on_playback(self, played: int):
        self.played = max(self.played, int(played))

    def on_anchor(self, sample: int, t: float):
        self.anchor = (int(sample), float(t))

    # ---- clocks -----------------------------------------------------------------
    def play_time(self, song_s):
        """Client wall time at which song second `song_s` reaches the speaker."""
        a_s, a_t = self.anchor
        return a_t + (song_s * self.B.SR - a_s) / self.B.SR

    def eeg_time_online(self, song_s):
        return self.play_time(song_s) - float(self.params["delay_s"])

    # ---- the loop -------------------------------------------------------------
    def _status(self, state, msg="", **kw):
        self.state = state
        self.send({"type": "status", "state": state, "msg": msg, "mode": self.mode, **kw})

    def _configure_track(self, tr):
        p = self.params
        tr.guidance = float(p["guidance"])
        tr.hist_guidance = float(p["hist_guidance"])
        tr.ctx_noise = float(p["ctx_noise"])
        tr.style_off = p["style_mode"] == "off"
        tr.bpm_follow = False
        tr.bpm_step = False
        tr.style_walk = "off"
        tr.style_target = None
        return tr

    def _run(self):
        try:
            if self.mode == "offline":
                self._loop_offline()
            else:
                self._loop_online()
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self._status("error", f"{type(e).__name__}: {e}")
        else:
            if self.state != "done":
                self._status("idle", "stopped")

    def _new_track(self):
        B, p = self.B, self.params
        name = p["preset"] if p["preset"] in self.E.presets else next(iter(self.E.presets))
        cond = self.E.presets[name].copy()
        tr = B.Track(cond=cond, seed=int(p["seed"]), n_chunks=cond.T,
                     label=f"live: {name}", style_preset=name)
        tr = self.E.backend.set_stage(tr, 2)
        tr = self._configure_track(tr)
        tr.cond_hook = self.cond
        return tr

    def _render_opening(self, tr):
        tr.cond = self.cond(tr.cond, 0)
        t0 = time.time()
        with self.E.lock:
            tr = self.E.backend.render(tr, n_steps=int(self.params["n_steps"]))
        return tr, time.time() - t0

    def _take(self, tr, k, steps):
        be, p, T = self.E.backend, self.params, tr.cond.T
        n_new = int(np.clip(p["n_new"], 1, T - 1))
        tr = self._configure_track(tr)
        t0 = time.time()
        with self.E.lock:
            if hasattr(be, "_continue_step"):
                tr = be._continue_step(tr, n_new, int(p["seed"]) + k, int(steps), decode="new")
            else:                                   # MockBackend
                tr = be.continue_track(tr, n_new, int(p["seed"]) + k, int(steps))
        compute = time.time() - t0
        self.compute_est = compute if self.compute_est is None else \
            0.7 * self.compute_est + 0.3 * compute
        first = int(tr.prefix_measures)
        return tr, list(range(first + T - n_new, first + T)), compute

    @staticmethod
    def _song_end(tr):
        b = list(tr.bank_bounds or tr.window_bounds or [0.0])
        return float(b[-1])

    def _loop_online(self):
        B, p = self.B, self.params
        sr = B.SR
        self.cond.reset()
        self.seq, self.played, self.emitted = 0, 0, 0
        self._status("starting", "rendering opening window")
        t_wait = time.time()
        while not self.cond.has_frames() and time.time() - t_wait < 5.0:
            if self._stop.is_set():
                return
            time.sleep(0.05)

        tr = self._new_track()
        self.cond.plan = {"start_s": 0.0, "eeg_time": None}     # newest EEG
        tr, compute = self._render_opening(tr)
        self.compute_est = compute * 0.6        # a take decodes fewer measures
        T = tr.cond.T
        lead = int(np.clip(p["lead_in_measures"], 1, T))
        bounds = list(tr.bank_bounds or tr.window_bounds)
        start = int(round(bounds[T - lead] * sr))
        self.emitted = self.played = start
        if self.anchor is None:
            t0 = self.t_start or self.cond.latest_t() or time.time()
            self.anchor = (start, t0 + float(p["delay_s"]))      # the plan
        self._emit(tr, list(range(T - lead, T)), compute, first=True)
        self._status("running", f"opening window {compute:.1f}s")

        k = 0
        while not self._stop.is_set():
            T = tr.cond.T
            n_new = int(np.clip(p["n_new"], 1, T - 1))
            start_s = self._song_end(tr)
            est_len = n_new * B.TARGET_SIG * 60.0 / float(tr.cond.bpm[-1])
            now_c = self.cond.client_now() or time.time()
            latest = self.cond.latest_t() or now_c
            need = self.eeg_time_online(start_s + est_len)
            slack = self.play_time(self.emitted / sr) - now_c
            ce = self.compute_est or 5.0
            urgent = slack < ce * 1.25 + 1.0
            if latest < need and not urgent:
                time.sleep(0.03)
                continue
            steps = int(p["n_steps"])
            degraded = bool(urgent and p.get("auto_draft") and steps > 8)
            if degraded:
                steps = max(8, steps // 2)
            self.cond.plan = {"start_s": start_s, "eeg_time": self.eeg_time_online}
            k += 1
            tr, measures, compute = self._take(tr, k, steps)
            now_c = self.cond.client_now() or time.time()
            first_new = self.cond.record(measures[0]) or {}
            margin = self.play_time(first_new.get("song_s", start_s)) - now_c
            self._emit(tr, measures, compute, slack_s=margin, degraded=degraded,
                       steps=steps, waited_for_eeg=latest >= need)
            self.cond.forget_before(int(tr.prefix_measures) - 64)

    def _loop_offline(self):
        B, p = self.B, self.params
        sr = B.SR
        self.cond.reset()
        self.seq, self.emitted = 0, 0
        if not self.cond.has_frames():
            raise RuntimeError("no EEG frames supplied for offline generation")
        t0 = self.cond.first_t() + float(p.get("offset_s") or 0.0)
        scale = max(1e-3, float(p.get("time_scale") or 1.0))
        rec_len = (self.cond.latest_t() - t0) / scale
        dur = float(p["duration_s"]) if p.get("duration_s") else rec_len
        eeg_time = lambda song_s: t0 + song_s * scale            # noqa: E731
        self._status("starting", f"offline: {dur:.0f}s of music from "
                                 f"{rec_len * scale:.0f}s of EEG", progress=0.0)
        tr = self._new_track()
        self.cond.plan = {"start_s": 0.0, "eeg_time": eeg_time}
        tr, compute = self._render_opening(tr)
        T = tr.cond.T
        self._emit(tr, list(range(T)), compute, first=True)      # all of it
        k = 0
        while not self._stop.is_set() and self.emitted / sr < dur:
            self.cond.plan = {"start_s": self._song_end(tr), "eeg_time": eeg_time}
            k += 1
            tr, measures, compute = self._take(tr, k, int(p["n_steps"]))
            self._emit(tr, measures, compute)
            self._status("running", f"offline {self.emitted / sr:.0f}/{dur:.0f}s",
                         progress=min(1.0, self.emitted / sr / max(dur, 1e-6)))
            self.cond.forget_before(int(tr.prefix_measures) - 64)
        if not self._stop.is_set():
            self._status("done", f"offline finished: {self.emitted / sr:.0f}s", progress=1.0)

    def _emit(self, tr, measures, compute, first=False, **extra):
        """Send the audio between `emitted` and the crossfade hold-back."""
        B, sr = self.B, self.B.SR
        hold = 0 if self.E.mock else B.CROSSFADE_SAMPLES
        cut = len(tr.audio) - hold
        chunk = np.asarray(tr.audio[self.emitted:cut], dtype=np.float32)
        start = self.emitted
        self.emitted = cut
        self.seq += 1
        bounds = list(tr.bank_bounds or [])
        rows = []
        for m in measures:
            rec = self.cond.record(m) or {}
            ms = int(round(bounds[m] * sr)) if m < len(bounds) else None
            me = int(round(bounds[m + 1] * sr)) if m + 1 < len(bounds) else None
            ch = rec.get("chroma")
            rows.append({"m": m, "start": ms, "end": me,
                         "dials": rec.get("dials", {}), "vals": rec.get("vals", {}),
                         "t": rec.get("t"), "early": rec.get("early", False),
                         "chroma": None if ch is None else np.round(ch, 3)})
        self.send({
            "type": "take", "mode": self.mode, "seq": self.seq,
            "start_sample": start, "n_samples": len(chunk), "measures": rows,
            "compute_s": compute, "audio_s": len(chunk) / sr, "first": first,
            "active": list(self.params["active"]),
            "diag": (getattr(tr, "diag", "") or "")[-600:],
            "anchor": self.anchor, **extra,
        })
        self.send(("audio", self.seq, start, chunk, sr))
        if self.params.get("analyze_output") and not self.E.mock:
            self._analyze(tr, measures)

    def _analyze(self, tr, measures):
        """Re-measure the generated audio with the training DSP so the UI can
        plot requested vs played for every signal. CPU, after the send."""
        try:
            from preset_extractor import analyze_measures
            first = int(tr.prefix_measures)
            sig = analyze_measures(tr.window_audio, tr.window_bounds,
                                   sr=self.B.SR, n_chunks=tr.cond.T,
                                   with_style=False, label="live")
            rows = []
            for m in measures:
                i = m - first
                if 0 <= i < tr.cond.T:
                    rows.append({"m": m, **{s: float(sig[s][i]) for s in
                                            ("rms", "density", "zcr", "flatness")}})
            self.send({"type": "measured", "mode": self.mode, "seq": self.seq,
                       "measures": rows})
        except Exception as e:  # noqa: BLE001
            self.send({"type": "status", "state": self.state,
                       "msg": f"output analysis failed: {e}"})
