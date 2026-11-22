"""The laptop process: EEG source -> DSP -> mapping -> GPU link -> speaker,
plus a local web UI (http://localhost:8080) to watch and steer all of it.

Two modes, one GPU session at a time:

  ONLINE   live EEG in, live audio out, at a fixed brain-to-ear delay. The
           speaker starts `delay_s` after Start and never waits on the model
           again unless the GPU genuinely can't keep up (see server Session).
  OFFLINE  a recording in, a whole piece out, generated as fast as the GPU
           allows and written next to the recording -- for finding mappings
           and normalization stats before trusting them live.

One asyncio loop runs everything except the sounddevice callback.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections import deque

import numpy as np
from aiohttp import WSMsgType, web

from octomusic import protocol
from octomusic.client import offline as OFF
from octomusic.client.audio_out import AudioOut
from octomusic.client.dsp import DSP, FEATURES
from octomusic.client.gpu_link import GPULink
from octomusic.client.mapping import NORMS, PRESETS, SOURCES, Mapper
from octomusic.client import protocols as PROTO
from octomusic.client import sessions as SESS
from octomusic.client.recorder import Recorder
from octomusic.client.sources import EEGBuffer, OctopusBLE, OctopusUSB, Replay, Synthetic
from octomusic.model_space import (DEFAULT_BANDS, NOTE_NAMES, SCALAR_TARGETS,
                                   TARGET_LABELS, TARGETS, bands_from_json,
                                   dial_to_value)

STATIC = os.path.join(os.path.dirname(__file__), "static")
CONFIG_DIR = "configs"
REC_DIR = "recordings"

# Generation parameters mirrored to the server (see server/engine.py
# DEFAULT_PARAMS for meanings). Kept here so the UI works before connecting.
GEN_DEFAULTS = {
    "preset": None, "style_b": None, "style_text": "", "style_axis": "",
    "style_axis_deg": 60.0, "seed": 0,
    "n_new": 8, "n_steps": 32, "guidance_on": True, "guidance": 3.0,
    "history_on": True, "hist_guidance_on": False, "hist_guidance": 1.5, "ctx_noise": 0.2, "delay_s": 20.0, "auto_draft": True, "lead_in_measures": 6,
    "analyze_output": True,
    "mode": "locked", "smooth_s": 2.0, "glide": 0.5, "bpm_jitter": 0.0,
    "time_scale": 1.0, "offset_s": 0.0, "duration_s": None,
    "bands": {k: list(v) for k, v in DEFAULT_BANDS.items()},
}
FEATURE_HOP_S = 0.25


class OfflineJob:
    """Collects one offline generation's stream into files."""

    def __init__(self, rec_path, meta):
        import soundfile as sf
        stamp = time.strftime("%Y%m%d-%H%M%S")
        base = rec_path if os.path.isdir(rec_path) else os.path.dirname(rec_path)
        self.dir = os.path.join(base, "offline", stamp)
        os.makedirs(self.dir, exist_ok=True)
        self.wav = sf.SoundFile(os.path.join(self.dir, "output.wav"), "w",
                                samplerate=16000, channels=1, subtype="PCM_16")
        self.takes = open(os.path.join(self.dir, "takes.jsonl"), "w")
        with open(os.path.join(self.dir, "config.json"), "w") as f:
            f.write(protocol.dumps(meta))
        self.next = 0
        self.done = False

    def audio(self, start, x):
        if start > self.next:
            self.wav.write(np.zeros(start - self.next, np.float32))
            self.next = start
        elif start < self.next:
            x = x[self.next - start:]
        self.wav.write(x)
        self.next += len(x)

    def close(self):
        if not self.done:
            self.done = True
            self.wav.close()
            self.takes.close()


class App:
    def __init__(self, server_url: str, audio_device=None, source="synthetic",
                 replay_path=None, ble_address=None):
        self.buf = EEGBuffer()
        self.dsp = DSP(self.buf.fs)
        self.mapper = Mapper()
        self.audio = AudioOut(audio_device)
        self.rec = Recorder(REC_DIR)
        self.link = GPULink(server_url, self._on_server, self._on_audio)
        self.gen = dict(GEN_DEFAULTS)
        self.source = None
        self._initial_source = (source, replay_path, ble_address)
        self.ui_clients: dict = {}       # ws -> outgoing queue
        self._ui_drops: dict = {}
        self.server_state = "idle"
        self.server_msg = ""
        self.server_progress = None
        self.run_mode = None           # None | "online" | "offline"
        self.takes = deque(maxlen=64)
        self.waves = deque(maxlen=400)
        self.feat_hist = deque(maxlen=int(120 / FEATURE_HOP_S))   # replayed to new pages
        self._pending_lat = deque()    # (start_sample, eeg_t)
        self.latency_s = None
        self.rtf = None
        self.compute_s = None
        self.take_audio_s = None
        self.margin_s = None
        self.degraded = 0
        self._eeg_out = []
        self._cursor = 0
        self.calibrating = None
        self.offline = None            # last analysis {path, ex, summary}
        self.job: OfflineJob | None = None
        self.last_offline = None       # {"dir", "url"}
        self.off_msgs = []             # last offline run's takes/waves, for page reloads
        self.session_meta = {"subject": "", "placement": "Pz", "notes": ""}
        self.protocol_state = None     # live guided-protocol state for the UI
        self._protocol_task = None
        self._sessions = {}            # name -> SessionAnalysis (cached for baseline)
        self.buf.listeners.append(self.rec.eeg)
        self.audio.listeners.append(self.rec.audio)

    # ---- lifecycle --------------------------------------------------------------
    async def startup(self, _app=None):
        try:
            self.audio.open()
        except Exception as e:  # noqa: BLE001
            print(f"[client] audio: {e}")
        self.link.start()
        name, path, addr = self._initial_source
        await self.set_source(name, path=path, address=addr)
        for coro in (self._dsp_loop(), self._ui_loop(), self._playback_loop()):
            asyncio.create_task(coro)

    async def shutdown(self, _app=None):
        if self.rec.active:
            self.rec.stop()
        if self.job:
            self.job.close()
        if self.source:
            await self.source.stop()
        await self.link.stop()
        self.audio.close()

    async def set_source(self, name, path=None, address=None):
        if self.source:
            await self.source.stop()
        if name == "octopus":
            self.source = OctopusBLE(self.buf, address=address or None)
        elif name == "octopus-usb":
            self.source = OctopusUSB(self.buf, port=address or None)
        elif name == "replay":
            if not path:
                raise ValueError("replay needs a recording path")
            self.source = Replay(self.buf, path)
        else:
            self.source = Synthetic(self.buf)
        self.buf.reset_stats()
        self.dsp.reset()
        self.source.start()

    # ---- server traffic -----------------------------------------------------------
    def _server_params(self):
        p = dict(self.gen)
        p["active"] = self.mapper.active_targets()
        p["style_mode"] = self.mapper.routes["style"]["source"]
        return p

    async def push_params(self):
        await self.link.send({"type": "params", "params": self._server_params()})

    def _on_server(self, m):
        t = m.get("type")
        mode = m.get("mode")
        if t == "server_info":
            presets = m.get("presets") or []
            if self.gen["preset"] not in presets and presets:
                self.gen["preset"] = presets[0]
            asyncio.create_task(self.push_params())
            asyncio.create_task(self.broadcast({"type": "config", **self.config()}))
        elif t == "status":
            self.server_state = m.get("state", self.server_state)
            self.server_msg = m.get("msg", "")
            self.server_progress = m.get("progress")
            if mode == "offline" and m.get("state") in ("done", "idle", "error") and self.job:
                self._finish_job()
        elif t == "take":
            if mode == "offline" and self.job:
                self.job.takes.write(protocol.dumps(m) + "\n")
                self.off_msgs.append(m)
            elif mode == "online":
                self.takes.append(m)
                self.rec.take(m)
                rows = m.get("measures") or []
                if rows and rows[0].get("t"):
                    self._pending_lat.append((m["start_sample"], rows[0]["t"][1]))
            self.compute_s = m.get("compute_s")
            if m.get("audio_s") and not m.get("first"):
                self.take_audio_s = m["audio_s"]
                # compute / audio for this take: must stay < 1 to keep up
                self.rtf = m["compute_s"] / max(1e-6, m["audio_s"])
            if m.get("slack_s") is not None:
                self.margin_s = m["slack_s"]
            if m.get("degraded"):
                self.degraded += 1
        elif t == "measured":
            if mode == "offline" and self.job:
                self.job.takes.write(protocol.dumps(m) + "\n")
                self.off_msgs.append(m)
            else:
                self.rec.take(m)
        elif t == "link":
            self.server_state = "disconnected"
            if self.run_mode:
                # the server ends the session when its client disconnects
                self.run_mode = None
                if self.job:
                    self._finish_job()
        asyncio.create_task(self.broadcast(m))

    def _finish_job(self):
        job, self.job = self.job, None
        job.close()
        rel = os.path.relpath(os.path.join(job.dir, "output.wav"), REC_DIR)
        self.last_offline = {"dir": job.dir, "url": "/files/" + rel.replace(os.sep, "/")}
        self.run_mode = None
        asyncio.create_task(self.broadcast({"type": "offline_done", **self.last_offline}))

    def _on_audio(self, seq, start, sr, pcm):
        mode = self.run_mode
        if mode == "offline":
            if self.job:
                self.job.audio(start, pcm)
        elif mode == "online":
            self.audio.push(start, pcm)
        else:
            return
        k = 160                                    # envelope for the UI waveform
        n = len(pcm) // k
        if n:
            env = np.abs(pcm[:n * k]).reshape(n, k).max(axis=1)
            w = {"type": "wave", "mode": mode, "start": start, "step": k,
                 "env": np.round(env, 3).tolist()}
            (self.waves if mode == "online" else self.off_msgs).append(w)
            asyncio.create_task(self.broadcast(w))

    # ---- loops --------------------------------------------------------------------
    async def _dsp_loop(self):
        next_feat = time.time()
        while True:
            await asyncio.sleep(0.04)
            raw, self._cursor = self.buf.read_since(self._cursor)
            if len(raw):
                disp = self.dsp.process(raw)
                self._eeg_out.append(disp)
            now = time.time()
            if now < next_feat:
                continue
            next_feat = now + FEATURE_HOP_S
            self.dsp.contact()
            res = self.dsp.features()
            feats, spec12, art, nch = res if res else (None, None, 0.0, 0)
            artifact = art > 0.5
            dials, chroma = self.mapper.step(feats, artifact, now, spec12)
            bands = bands_from_json(self.gen.get("bands"))
            vals = {t: dial_to_value(t, dials[t], bands) for t in SCALAR_TARGETS}
            row = {"type": "feat", "t": now, "feats": feats, "artifact": artifact,
                   "art_frac": art, "nch": nch, "dials": dials, "z": self.mapper.z,
                   "vals": vals, "chroma": None if chroma is None else np.round(chroma, 3)}
            if feats:
                self.feat_hist.append(row)
                await self.link.send({"type": "dials", "t": now, "values": dials,
                                      "chroma": None if chroma is None
                                      else [round(float(c), 4) for c in chroma]})
                self.rec.features(row)
            await self.broadcast(row)

    def _note_source_events(self):
        """Mirror the source's disconnects into the session recording."""
        src = self.source
        if not src:
            return
        seen = getattr(self, "_events_seen", (None, 0))
        if seen[0] is not src:
            seen = (src, 0)
        ev = list(src.events)
        for t, what in ev[seen[1]:]:
            self.rec.event("source_error", source=src.name, error=what, at=t)
            print(f"[client] EEG source {src.name}: {what} — "
                  f"{'reconnecting' if src.reconnect else 'stopped'}", flush=True)
        self._events_seen = (src, len(ev))

    async def _playback_loop(self):
        while True:
            await asyncio.sleep(0.1)
            self._note_source_events()
            played = self.audio.played
            while self.audio.anchors:
                s, t = self.audio.anchors.pop(0)
                await self.link.send({"type": "anchor", "sample": s, "t": t})
                self.rec.event("anchor", sample=s, t=t)
            if self.run_mode == "online":
                await self.link.send({"type": "playback", "played": played, "t": time.time()})
            while self._pending_lat and self._pending_lat[0][0] <= played:
                _, t_eeg = self._pending_lat.popleft()
                self.latency_s = time.time() - t_eeg
            await self.broadcast({"type": "playhead", "played": played})

    async def _ui_loop(self):
        k = 0
        while True:
            await asyncio.sleep(0.05)
            k += 1
            if self._eeg_out:
                blk = np.concatenate(self._eeg_out)
                self._eeg_out = []
                await self.broadcast({"type": "eeg", "fs": self.buf.fs,
                                      "data": np.round(blk, 1).tolist()})
            if k % 10 == 0:
                sp = self.dsp.spectrum()
                if sp is not None:
                    await self.broadcast({"type": "spectrum", "f": sp[0].tolist(),
                                          "p": np.round(sp[1], 3).tolist()})
                await self.broadcast(self.status())

    # ---- status / config ----------------------------------------------------------
    def min_delay(self):
        """The smallest delay that can be smooth with the measured take length
        and compute time: a take's EEG is complete take_len before its end is
        heard at the delay, then it needs compute seconds, plus margin."""
        if self.take_audio_s is None or self.compute_s is None:
            return None
        return self.take_audio_s + self.compute_s + 2.0

    def status(self):
        rec = self.buf.received
        return {
            "type": "status",
            "source": self.source.info() if self.source else None,
            "fs": self.buf.measured_fs, "lost": self.buf.lost,
            "loss_pct": 100.0 * self.buf.lost / max(1, rec + self.buf.lost),
            "contact": np.round(self.dsp.contact_rms, 1).tolist(),
            "good": self.dsp.good.tolist(), "used": self.dsp.used_channels().tolist(),
            "quality": None if self.dsp.quality is None else {
                "status": self.dsp.quality["status"].tolist(),
                "pp": np.round(self.dsp.quality["pp"], 1).tolist(),
                "line": np.round(self.dsp.quality["line"], 2).tolist(),
                "snr": np.round(self.dsp.quality["snr"], 1).tolist()},
            "link": self.link.info(), "audio": self.audio.info(),
            "server_state": self.server_state, "server_msg": self.server_msg,
            "progress": self.server_progress, "run_mode": self.run_mode,
            "music_on": self.run_mode == "online", "latency_s": self.latency_s,
            "rtf": self.rtf, "compute_s": self.compute_s, "margin_s": self.margin_s,
            "min_delay_s": self.min_delay(), "degraded": self.degraded,
            "recording": self.rec.dir, "calibrating": self.calibrating,
            "label": self.rec.current_label,
            "label_summary": self.rec.label_summary() if self.rec.active else None,
            "protocol": self.protocol_state,
        }

    def config(self):
        info = self.link.server_info or {}
        recs = sorted(d for d in os.listdir(REC_DIR)
                      if os.path.isfile(os.path.join(REC_DIR, d, "eeg.csv"))) \
            if os.path.isdir(REC_DIR) else []
        return {
            "routes": self.mapper.routes, "dsp": self.dsp.cfg, "gen": self.gen,
            "synth": self.source.knobs if isinstance(self.source, Synthetic) else None,
            "features": FEATURES, "targets": list(TARGETS),
            "target_labels": TARGET_LABELS, "sources": SOURCES, "norms": NORMS,
            "presets_map": list(PRESETS), "notes": NOTE_NAMES,
            "presets": info.get("presets", []), "style_axes": info.get("style_axes", []),
            "backend": info.get("backend"), "device": info.get("device"),
            "calib": self.mapper.state()["calib"],
            "configs": sorted(f[:-5] for f in os.listdir(CONFIG_DIR) if f.endswith(".json"))
            if os.path.isdir(CONFIG_DIR) else [],
            "recordings": recs[-50:],
            "offline_loaded": self.offline["path"] if self.offline else None,
            "protocols": PROTO.listing(), "sessions": SESS.list_sessions(REC_DIR)[:40],
            "session_meta": self.session_meta,
            "last_offline": self.last_offline,
        }

    async def broadcast(self, obj):
        """Queue a message for every open page. NEVER awaits the network: the
        DSP loop calls this, and a page that stops reading (background tab,
        closing browser) used to stall EEG processing and the dials sent to
        the GPU for as long as its socket was backed up. A page that falls
        behind loses frames; one that stays stuck is dropped."""
        if not self.ui_clients:
            return
        s = protocol.dumps(obj)
        for ws, q in list(self.ui_clients.items()):
            try:
                q.put_nowait(s)
            except asyncio.QueueFull:
                self._ui_drops[ws] = self._ui_drops.get(ws, 0) + 1
                if self._ui_drops[ws] > 2000:          # stuck for minutes: let it go
                    self.ui_clients.pop(ws, None)
                    asyncio.create_task(ws.close())

    async def _ui_writer(self, ws, q):
        try:
            while True:
                await ws.send_str(await q.get())
        except (ConnectionResetError, RuntimeError, asyncio.CancelledError):
            pass

    # ---- online ---------------------------------------------------------------------
    def _require_gpu(self):
        if self.link.state != "connected":
            raise RuntimeError(
                f"not connected to the GPU server at {self.link.url} — is the SSH tunnel "
                f"running? (ssh -N -L 8765:localhost:8765 you@gpu-box)  [{self.link.error}]")

    async def start_online(self):
        if self.run_mode == "offline":
            raise RuntimeError("an offline generation is running")
        self._require_gpu()
        self.audio.reset()
        self._pending_lat.clear()
        self.takes.clear()
        self.waves.clear()
        self.degraded = 0
        self.margin_s = None
        t_start = time.time()
        delay = float(self.gen["delay_s"])
        self.audio.schedule(t_start + delay, rebuffer_s=min(6.0, max(1.0, delay / 3)))
        self.run_mode = "online"
        if not await self.link.send({"type": "start", "t_start": t_start,
                                     "params": self._server_params()}):
            self.run_mode = None
            self.audio.reset()
            self._require_gpu()
            raise RuntimeError("the GPU server connection dropped while starting")
        self.rec.event("music_start", params=self._server_params())

    async def stop_run(self):
        mode, self.run_mode = self.run_mode, None
        await self.link.send({"type": "stop"})
        self.audio.reset()
        if mode == "offline" and self.job:
            self._finish_job()
        self.rec.event("music_stop")

    # ---- offline ----------------------------------------------------------------------
    def _rec_path(self, name):
        p = os.path.join(REC_DIR, name)
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        return p

    async def offline_analyze(self, name, start_s=0.0, duration_s=None):
        path = self._rec_path(name)
        await self.broadcast({"type": "offline_busy", "msg": f"analyzing {name}…"})

        def work():
            t, x, fs = OFF.load_recording(path)
            ex = OFF.extract(t, x, fs, self.dsp.cfg, start_s, duration_s)
            if not len(ex["times"]):
                raise RuntimeError("recording too short for the feature window")
            return t, fs, ex

        t, fs, ex = await asyncio.to_thread(work)
        series = OFF.session_series(ex, self.mapper.routes)
        # whole-session stats -> session_z / session_quantile, offline AND live
        self.mapper.set_session_stats(series)
        self.offline = {"path": name, "ex": ex, "fs": fs,
                        "summary": OFF.summarize(series),
                        "rec_len": float(t[-1] - t[0])}
        await self.broadcast(self._analysis_msg())

    def _analysis_msg(self):
        o = self.offline
        ex = o["ex"]
        frames = OFF.dial_frames(ex, self.mapper.clone_for_offline())
        times = ex["times"] - ex["times"][0]
        dials = {}
        for tgt in TARGETS:
            arr = np.asarray([f["values"].get(tgt, 0.5) for f in frames])
            _, dials[tgt] = OFF.decimate_series(times, arr)
        feats, tt = {}, []
        for n, arr in ex["feats"].items():
            if n.startswith("ch"):
                continue
            tt, feats[n] = OFF.decimate_series(times, arr)
        art = OFF.decimate_series(times, ex["artifact"].astype(float))[1]
        return {"type": "offline_analysis", "path": o["path"], "fs": o["fs"],
                "duration": float(times[-1]) if len(times) else 0.0,
                "rec_len": o["rec_len"], "summary": o["summary"], "t": tt,
                "dials": dials, "feats": feats, "artifact": art,
                "artifact_pct": 100.0 * float(ex["artifact"].mean())}

    async def offline_generate(self):
        if not self.offline:
            raise RuntimeError("analyze a recording first")
        self._require_gpu()
        if self.run_mode == "online":
            await self.stop_run()
        o = self.offline
        frames = OFF.dial_frames(o["ex"], self.mapper.clone_for_offline())
        params = self._server_params()
        self.job = OfflineJob(self._rec_path(o["path"]), {
            "recording": o["path"], "params": params, "routes": self.mapper.routes,
            "dsp": self.dsp.cfg, "summary": o["summary"]})
        self.run_mode = "offline"
        self.off_msgs = []
        await self.link.send({"type": "offline_start", "params": params, "frames": frames})
        await self.broadcast({"type": "offline_started", "dir": self.job.dir})

    # ---- UI commands -----------------------------------------------------------------
    async def handle(self, m):
        t = m.get("type")
        if t == "set_route":
            self.mapper.set_route(m["target"], **{m["key"]: m["value"]})
            self.rec.event("route", target=m["target"], key=m["key"], value=m["value"])
            if m["key"] == "source":
                await self.push_params()
        elif t == "map_preset":
            self.mapper.apply_preset(m["name"])
            self.rec.event("map_preset", name=m["name"])
            await self.push_params()
        elif t == "set_dsp":
            v = m["value"]
            if m["key"] == "channel":
                ch = list(self.dsp.cfg["channels"])
                ch[int(v[0])] = bool(v[1])
                self.dsp.configure(channels=ch)
            else:
                self.dsp.configure(**{m["key"]: v})
            self.rec.event("dsp", key=m["key"], value=v)
        elif t == "set_gen":
            k, v = m["key"], m["value"]
            if k == "band":
                tgt, lo, hi = v
                self.gen["bands"][tgt] = [float(lo), float(hi), DEFAULT_BANDS[tgt][2]]
            else:
                self.gen[k] = v
            self.rec.event("gen", key=k, value=v)
            await self.push_params()
        elif t == "set_synth" and isinstance(self.source, Synthetic):
            self.source.knobs[m["key"]] = float(m["value"])
        elif t == "source":
            await self.set_source(m["name"], path=m.get("path"), address=m.get("address"))
        elif t == "music":
            if m["action"] == "start":
                await self.start_online()
            else:
                await self.stop_run()
        elif t == "offline_analyze":
            await self.offline_analyze(m["path"], float(m.get("start_s") or 0),
                                       float(m["duration_s"]) if m.get("duration_s") else None)
        elif t == "offline_refresh" and self.offline:
            await self.broadcast(self._analysis_msg())   # re-map with current routes
        elif t == "offline_generate":
            await self.offline_generate()
        elif t == "calibrate":
            asyncio.create_task(self._calibrate(m["kind"], float(m.get("seconds", 30))))
        elif t == "reset_norm":
            for st in self.mapper.stats.values():
                st.mean = st.var = None
        elif t == "record":
            if m["action"] == "start" and not self.rec.active:
                self._start_recording("free")
            elif m["action"] == "stop" and self.rec.active:
                await self._stop_protocol()
                self.rec.stop()
        elif t == "session_meta":
            self.session_meta.update({k: str(v) for k, v in (m.get("meta") or {}).items()
                                      if k in ("subject", "placement", "notes")})
            if self.rec.active:
                self.rec.update_meta(**self.session_meta)
        elif t == "rec_label":
            if not self.rec.active:
                self._start_recording("free")
            self.rec.label(m.get("label") or None)
        elif t == "protocol_start":
            await self._stop_protocol()
            self._protocol_task = asyncio.create_task(self._run_protocol(
                m["name"], m.get("block_s"), m.get("reps"), bool(m.get("shuffle", True))))
        elif t == "protocol_stop":
            await self._stop_protocol()
        elif t == "session_analyze":
            await self.broadcast({"type": "session_busy", "name": m["name"]})
            path = os.path.join(REC_DIR, m["name"])
            sa = await asyncio.to_thread(SESS.SessionAnalysis, path, dict(self.dsp.cfg))
            self._sessions[m["name"]] = sa
            await self.broadcast(await asyncio.to_thread(sa.report))
        elif t == "session_baseline":
            sa = self._sessions.get(m["name"])
            if sa is None:
                sa = await asyncio.to_thread(SESS.SessionAnalysis, os.path.join(REC_DIR, m["name"]),
                                             dict(self.dsp.cfg))
                self._sessions[m["name"]] = sa
            done = sa.apply_baseline(self.mapper, m.get("baseline"), m.get("low"), m.get("high"))
            self.rec.event("baseline_applied", session=m["name"], **done)
            await self.broadcast({"type": "info", "msg":
                f"baseline from {m['name']}: baseline='{done['baseline']}', range "
                f"'{done['low']}'→'{done['high']}', session stats for {done['features']} features"})
        elif t == "note":
            self.rec.event("note", text=m.get("text", ""))
        elif t == "audio":
            if "volume" in m:
                self.audio.volume = float(m["volume"])
            if "muted" in m:
                self.audio.muted = bool(m["muted"])
        elif t == "save_config":
            os.makedirs(CONFIG_DIR, exist_ok=True)
            with open(os.path.join(CONFIG_DIR, f"{m['name']}.json"), "w") as f:
                f.write(protocol.dumps({"mapper": self.mapper.state(),
                                        "dsp": self.dsp.cfg, "gen": self.gen}))
        elif t == "load_config":
            with open(os.path.join(CONFIG_DIR, f"{m['name']}.json")) as f:
                d = json.load(f)
            self.mapper.load(d.get("mapper", {}))
            self.dsp.configure(**d.get("dsp", {}))
            self.gen.update(d.get("gen", {}))
            await self.push_params()
        await self.broadcast({"type": "config", **self.config()})

    # ---- recording / protocols ---------------------------------------------------------
    def _start_recording(self, protocol_name, extra=None):
        q = self.dsp.quality
        meta = {**self.session_meta, "protocol": protocol_name,
                "source": self.source.info() if self.source else None,
                "measured_fs": self.buf.measured_fs,
                "quality_at_start": None if q is None else {
                    "status": q["status"].tolist(), "snr_db": np.round(q["snr"], 1).tolist(),
                    "pp_uv": np.round(q["pp"], 1).tolist()},
                **(extra or {})}
        return self.rec.start({"routes": self.mapper.routes, "dsp": self.dsp.cfg, "gen": self.gen},
                              self.buf.measured_fs or self.buf.fs, meta)

    async def _stop_protocol(self):
        if self._protocol_task and not self._protocol_task.done():
            self._protocol_task.cancel()
            try:
                await self._protocol_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self._protocol_task = None

    async def _run_protocol(self, name, block_s=None, reps=None, shuffle=True):
        steps = PROTO.schedule(name, block_s, reps, shuffle)
        total = sum(st["seconds"] for st in steps)
        started_here = not self.rec.active
        if started_here:
            self._start_recording(name, {"protocol_steps": steps})
        else:
            self.rec.update_meta(protocol=name, protocol_steps=steps)
        d = self.rec.dir
        t_begin = time.time()
        cancelled = False
        try:
            for i, st in enumerate(steps):
                self.rec.label(st["label"])
                t_end = time.time() + st["seconds"]
                while True:
                    now = time.time()
                    if now >= t_end:
                        break
                    self.protocol_state = {
                        "name": name, "title": PROTO.PROTOCOLS[name]["title"], "step": i,
                        "n_steps": len(steps), "kind": st["kind"], "label": st["label"],
                        "next": st.get("next"), "instruction": st["instruction"],
                        "left": round(t_end - now, 1), "total_left": round(total - (now - t_begin), 0),
                        "progress": min(1.0, (now - t_begin) / total), "dir": d}
                    await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            cancelled = True
        self.rec.label(None)
        if started_here and self.rec.active:
            self.rec.stop()
        self.protocol_state = {"name": name, "done": True, "cancelled": cancelled, "dir": d}
        await self.broadcast({"type": "config", **self.config()})
        if not cancelled and d:
            name_d = os.path.basename(d)
            sa = await asyncio.to_thread(SESS.SessionAnalysis, d, dict(self.dsp.cfg))
            self._sessions[name_d] = sa
            await self.broadcast(await asyncio.to_thread(sa.report))

    async def _calibrate(self, kind, seconds):
        """Hold a state for `seconds`, then summarize it (see Mapper.calibrate)."""
        t0 = time.time()
        self.rec.event("calibrate_begin", kind=kind, seconds=seconds)
        while time.time() - t0 < seconds:
            self.calibrating = {"kind": kind, "left": round(seconds - (time.time() - t0), 1)}
            await asyncio.sleep(0.25)
        self.calibrating = None
        res = self.mapper.calibrate(kind, seconds)
        self.rec.event("calibrate_end", kind=kind, summary=res)
        await self.broadcast({"type": "config", **self.config()})

    # ---- web ------------------------------------------------------------------------------
    async def ws_ui(self, request):
        ws = web.WebSocketResponse(heartbeat=20, max_msg_size=64 << 20)
        await ws.prepare(request)
        await ws.send_str(protocol.dumps({"type": "config", **self.config()}))
        # a reloaded page gets the recent output back instead of a blank timeline
        for m in list(self.takes) + list(self.waves) + list(self.off_msgs):
            await ws.send_str(protocol.dumps(m))
        for row in list(self.feat_hist):
            await ws.send_str(protocol.dumps(row))
        q = asyncio.Queue(maxsize=600)                  # ~15 s of traffic
        self.ui_clients[ws] = q
        writer = asyncio.create_task(self._ui_writer(ws, q))
        if self.offline:
            await ws.send_str(protocol.dumps(self._analysis_msg()))
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        await self.handle(json.loads(msg.data))
                    except Exception as e:  # noqa: BLE001
                        await ws.send_str(protocol.dumps(
                            {"type": "error", "msg": f"{type(e).__name__}: {e}"}))
        finally:
            self.ui_clients.pop(ws, None)
            self._ui_drops.pop(ws, None)
            writer.cancel()
        return ws


def build_web(app: App):
    w = web.Application()
    w.on_startup.append(app.startup)
    w.on_cleanup.append(app.shutdown)
    w.router.add_get("/ws", app.ws_ui)

    async def index(_):
        return web.FileResponse(os.path.join(STATIC, "index.html"))

    w.router.add_get("/", index)
    w.router.add_static("/static", STATIC)
    os.makedirs(REC_DIR, exist_ok=True)
    w.router.add_static("/files", REC_DIR)
    return w
