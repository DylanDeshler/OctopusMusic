// OctopusMusic UI. Plain JS, no build step. Talks to the local client over /ws.
"use strict";

const COLORS = { rms: "#ff8c5a", bpm: "#7aa2ff", density: "#4ccf8a", zcr: "#f5c451",
                 flatness: "#c38bff", chroma: "#ff7ac8", style: "#5fd3e0" };
const SCALARS = ["rms", "bpm", "density", "zcr", "flatness"];
const $ = (id) => document.getElementById(id);
let ws, cfg = null, built = false, lastStatus = null, uiMode = "online";
const N_CH = 16;
let sr = 16000;

// ---------------------------------------------------------------- state buffers
const eeg = { fs: 250, cap: 250 * 20, data: [], n: 0 };
for (let c = 0; c < N_CH; c++) eeg.data.push(new Float32Array(eeg.cap));
let spectrum = null;
const dialHist = [];
const views = {                         // output timeline data, per mode
  online: { takes: new Map(), measured: new Map(), waves: [] },
  offline: { takes: new Map(), measured: new Map(), waves: [] },
};
let played = 0;
let analysis = null;                    // offline_analysis message

// ---------------------------------------------------------------- socket
function connect() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onmessage = (e) => onMsg(JSON.parse(e.data));
  ws.onclose = () => { $("msg").textContent = "lost connection to the local client — retrying…"; setTimeout(connect, 1500); };
  ws.onopen = () => { $("msg").textContent = ""; };
}
function send(o) { if (ws && ws.readyState === 1) ws.send(JSON.stringify(o)); }
function flash(text) { $("msg").textContent = text; setTimeout(() => { if ($("msg").textContent === text) $("msg").textContent = ""; }, 7000); }

function onMsg(m) {
  const v = views[m.mode === "offline" ? "offline" : "online"];
  switch (m.type) {
    case "eeg":
      eeg.fs = m.fs;
      for (const row of m.data) {
        const i = eeg.n % eeg.cap;
        for (let c = 0; c < N_CH; c++) eeg.data[c][i] = row[c];
        eeg.n++;
      }
      break;
    case "spectrum": spectrum = m; break;
    case "feat":
      if (m.feats) dialHist.push({ t: m.t, dials: m.dials, artifact: m.artifact, feats: m.feats });
      while (dialHist.length && m.t - dialHist[0].t > 120) dialHist.shift();
      updateMeters(m);
      break;
    case "status": lastStatus = m; updateStatus(m); break;
    case "config": cfg = m; if (!built) build(); else refresh(); break;
    case "take":
      if (m.first) { v.takes.clear(); v.measured.clear(); v.waves.length = 0; }
      for (const r of m.measures) v.takes.set(r.m, { ...r, seq: m.seq, active: m.active });
      if (m.diag) $("diag").textContent = m.diag;
      prune(v);
      break;
    case "measured": for (const r of m.measures) v.measured.set(r.m, r); break;
    case "wave": v.waves.push(m); if (m.mode !== "offline") while (v.waves.length > 400) v.waves.shift(); break;
    case "playhead": played = m.played; break;
    case "server_info": sr = m.sr || 16000; break;
    case "offline_busy": $("off-status").textContent = m.msg; break;
    case "offline_analysis":
      analysis = m;
      $("off-status").textContent = `${m.path}: ${m.duration.toFixed(0)} s analyzed · ${m.artifact_pct.toFixed(1)}% artifact`;
      renderStats();
      break;
    case "offline_started":
      for (const k of ["takes", "measured"]) views.offline[k].clear();
      views.offline.waves.length = 0;
      $("off-status").textContent = `generating → ${m.dir}`;
      break;
    case "offline_done": setOfflineResult(m); break;
    case "session_report": renderReport(m); break;
    case "session_busy": $("session-report").innerHTML = `<div class="small">analyzing ${m.name}…</div>`; break;
    case "info": flash(m.msg); break;
    case "error": flash(m.msg); break;
  }
}
function prune(v) {
  if (v === views.offline) return;
  const lim = played - 150 * sr;
  for (const [k, r] of v.takes) if (r.end != null && r.end < lim) { v.takes.delete(k); v.measured.delete(k); }
}
function setOfflineResult(m) {
  if (!m) return;
  const a = $("off-audio");
  if (!a.src.includes(m.url)) a.src = m.url + "?t=" + Date.now();
  $("off-link").textContent = "output.wav"; $("off-link").href = m.url;
  $("off-status").textContent = `done → ${m.dir}`;
}

// ---------------------------------------------------------------- control helpers
function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v; else if (k.startsWith("on")) e[k] = v; else e.setAttribute(k, v);
  }
  for (const k of kids) e.append(k instanceof Node ? k : document.createTextNode(k));
  return e;
}
const bindings = [];   // keep inputs in sync with cfg without fighting the user
const visRules = [];   // show/hide controls depending on cfg
function fmtNum(v) { return Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2); }
function vis(node, fn) { visRules.push({ el: node, fn }); return node; }
function slider(label, min, max, step, get, onset, fmt = fmtNum, tip = "") {
  const inp = el("input", { type: "range", min, max, step });
  const out = el("span", {});
  inp.value = get(); out.textContent = fmt(+inp.value);
  inp.oninput = () => { out.textContent = fmt(+inp.value); onset(+inp.value); };
  bindings.push({ el: inp, get, out, fmt });
  return el("div", { class: "ctl", title: tip }, el("span", {}, label), inp, out);
}
function select(label, options, get, onset, tip = "") {
  const s = el("select", {});
  for (const o of options) { const [v, t] = Array.isArray(o) ? o : [o, o]; s.append(el("option", { value: v }, t)); }
  s.value = get(); s.onchange = () => onset(s.value);
  bindings.push({ el: s, get });
  return el("div", { class: "ctl", title: tip }, el("span", {}, label), s, el("span", {}));
}
function check(label, get, onset, tip = "") {
  const c = el("input", { type: "checkbox" });
  c.checked = !!get(); c.onchange = () => onset(c.checked);
  bindings.push({ el: c, get, isCheck: true });
  return el("div", { class: "ctl", title: tip }, el("span", {}, label), c, el("span", {}));
}
function text(label, get, onset, attrs = {}, tip = "") {
  const i = el("input", { type: "text", ...attrs });
  i.value = get() ?? ""; i.onchange = () => onset(i.value);
  bindings.push({ el: i, get: () => get() ?? "" });
  return el("div", { class: "ctl", title: tip }, el("span", {}, label), i, el("span", {}));
}
function fillSelect(s, opts) {
  const cur = s.value, sig = JSON.stringify(opts);
  if (s.dataset.sig === sig) return;
  s.dataset.sig = sig; s.innerHTML = "";
  for (const o of opts) { const [v, t] = Array.isArray(o) ? o : [o, o]; s.append(el("option", { value: v }, t)); }
  if ([...s.options].some((o) => o.value === cur)) s.value = cur;
}
function refresh() {
  for (const b of bindings) {
    if (document.activeElement === b.el) continue;
    const v = b.get();
    if (b.isCheck) b.el.checked = !!v;
    else if (v !== undefined && v !== null && String(b.el.value) !== String(v)) {
      b.el.value = v; if (b.out) b.out.textContent = b.fmt(+v);
    }
  }
  for (const r of visRules) r.el.classList.toggle("hidden", !r.fn());
  fillSelect($("cfg-list"), cfg.configs);
  fillSelect($("src-path"), cfg.recordings.map((r) => ["recordings/" + r, r]));
  fillSelect($("off-rec"), cfg.recordings);
  if (cfg.offline_loaded && !$("off-rec").dataset.touched) $("off-rec").value = cfg.offline_loaded;
  fillSelect($("map-preset"), cfg.presets_map);
  for (const s of document.querySelectorAll("select[data-presets]")) {
    fillSelect(s, s.dataset.presets === "opt" ? [["", "—"], ...cfg.presets.map((p) => [p, p])] : cfg.presets);
    if (document.activeElement !== s) s.value = cfg.gen[s.dataset.key] || "";
  }
  for (const s of document.querySelectorAll("select[data-axes]")) {
    fillSelect(s, [["", "—"], ...cfg.style_axes.map((a) => [a, a])]);
    if (document.activeElement !== s) s.value = cfg.gen.style_axis || "";
  }
  fillSelect($("off-feat"), [["", "none"], ...Object.keys(cfg.features).filter((f) => !f.startsWith("ch")).map((f) => [f, f])]);
  $("synth-card").classList.toggle("hidden", !cfg.synth);
  $("backend").textContent = cfg.backend ? `${cfg.backend} on ${cfg.device}` : "server not connected";
  if (cfg.last_offline && !$("off-audio").getAttribute("src")) setOfflineResult(cfg.last_offline);
  renderCal(); renderStats(); refreshRecord();
}

// ---------------------------------------------------------------- build UI
function build() {
  built = true;
  bindings.length = 0; visRules.length = 0;
  const dl = $("feat-list"); dl.innerHTML = "";
  for (const [k, d] of Object.entries(cfg.features)) dl.append(el("option", { value: k }, d));
  buildChannels(); buildDSP(); buildRoutes(); buildGen(); buildBands(); buildSynth(); wireButtons(); buildRecord();
  refresh();
}

function buildChannels() {
  const box = $("chan-list"); box.innerHTML = "";
  for (let c = 0; c < N_CH; c++) {
    const d = el("div", { class: "chan", id: `ch-${c}`, title: "click to exclude/include" },
      el("span", { class: "dot", id: `dot-${c}` }), `${c + 1}`);
    d.onclick = () => send({ type: "set_dsp", key: "channel", value: [c, !cfg.dsp.channels[c]] });
    box.append(d);
  }
}
function buildDSP() {
  const box = $("dsp-controls"); box.innerHTML = "";
  const d = () => cfg.dsp;
  const set = (k) => (v) => send({ type: "set_dsp", key: k, value: v });
  const mk = (label, node, tip = "") => el("span", { class: "row-inline", title: tip }, label, node);
  const num = (k, w = "3.6em") => {
    const i = el("input", { type: "number", step: "any", style: `width:${w}` });
    i.value = d()[k]; i.onchange = () => set(k)(+i.value);
    bindings.push({ el: i, get: () => d()[k] }); return i;
  };
  const chk = (k) => {
    const c = el("input", { type: "checkbox" }); c.checked = d()[k];
    c.onchange = () => set(k)(c.checked);
    bindings.push({ el: c, get: () => d()[k], isCheck: true }); return c;
  };
  const notch = el("select", {}, el("option", { value: 0 }, "off"), el("option", { value: 50 }, "50 Hz"), el("option", { value: 60 }, "60 Hz"));
  notch.value = d().notch_hz; notch.onchange = () => set("notch_hz")(+notch.value);
  bindings.push({ el: notch, get: () => d().notch_hz });
  box.append(
    mk("Hampel", chk("hampel"), "spike removal (PiEEG default: window 5, 3σ)"),
    mk("band-pass", el("span", {}, num("bp_lo"), "–", num("bp_hi"), " Hz"), "PiEEG default 1–40 Hz, order 5"),
    mk("order", num("bp_order", "2.6em")),
    mk("notch", notch, "mains + first harmonic, Q 30"),
    mk("feature window s", num("window_s")),
    mk("artifact µV", num("artifact_uv", "4.4em")),
    mk("use good/fair pins only", chk("auto_exclude")));
}

function driveOf(t, r) {
  if (SCALARS.includes(t)) return r.source;
  if (t === "chroma") return r.source === "dial_key" ? r.drive : "off";
  return (r.source === "morph" || r.source === "axis") ? r.drive : "off";
}
const SRC_LABEL = { off: "off (unconditional)", eeg: "EEG", manual: "manual", key: "fixed key",
  spectrum: "EEG spectrum", dial_key: "key from dial", preset: "preset A", morph: "morph A→B", axis: "axis" };

function buildRoutes() {
  const box = $("routes"); box.innerHTML = "";
  for (const t of cfg.targets) {
    const r = () => cfg.routes[t];
    const set = (k) => (v) => {
      send({ type: "set_route", target: t, key: k, value: v });
      if (uiMode === "offline" && $("off-auto").checked && analysis) setTimeout(() => send({ type: "offline_refresh" }), 150);
    };
    const setG = (k) => (v) => send({ type: "set_gen", key: k, value: v });
    const srcSel = el("select", {}, ...cfg.sources[t].map((s) => el("option", { value: s }, SRC_LABEL[s] || s)));
    srcSel.value = r().source; srcSel.onchange = () => set("source")(srcSel.value);
    bindings.push({ el: srcSel, get: () => r().source });
    const meter = t === "chroma"
      ? el("div", { class: "chroma-mini", id: "cm-chroma" }, ...Array.from({ length: 12 }, () => el("i", {})))
      : el("div", { class: "meter" }, el("i", { id: `m-${t}` }), el("b", {}));
    const head = el("div", { class: "route-h" },
      el("span", { class: "swatch", style: `background:${COLORS[t]}` }),
      el("span", { class: "name" }, cfg.target_labels[t]), srcSel, meter,
      el("span", { class: "val", id: `v-${t}` }, "—"));
    const dv = () => driveOf(t, r());
    const isEEG = () => dv() === "eeg", isMan = () => dv() === "manual";
    const featInput = el("input", { list: "feat-list", placeholder: "feature or 0.5*a - b" });
    featInput.value = r().feature; featInput.onchange = () => set("feature")(featInput.value.trim());
    bindings.push({ el: featInput, get: () => r().feature });
    const notes = cfg.notes.map((n, i) => [i, n]);
    const body = el("div", { class: "route-body" },
      vis(select("drive", [["eeg", "EEG"], ["manual", "manual"]], () => r().drive, set("drive"),
        "what moves this input's dial"), () => !SCALARS.includes(t) && ["dial_key", "morph", "axis"].includes(r().source)),
      vis(el("div", { class: "ctl", title: Object.entries(cfg.features).map(([k, d]) => `${k}: ${d}`).join("\n") },
        el("span", {}, "feature"), featInput, el("span", {})), isEEG),
      vis(select("normalize", cfg.norms, () => r().norm, set("norm")), isEEG),
      vis(slider("center", 0, 1, 0.01, () => r().center, set("center"), fmtNum, "dial at your baseline"), isEEG),
      vis(slider("gain", 0, 6, 0.05, () => r().gain, set("gain"), fmtNum, "1 = one σ moves the dial 15%"), isEEG),
      vis(slider("smooth s", 0.1, 20, 0.1, () => r().smooth_s, set("smooth_s")), () => isEEG() || isMan()),
      vis(slider("adapt τ s", 5, 600, 5, () => r().tau_s, set("tau_s")), () => isEEG() && r().norm === "adaptive"),
      vis(slider("window s", 10, 600, 10, () => r().window_s, set("window_s")), () => isEEG() && r().norm === "quantile"),
      vis(check("invert", () => r().invert, set("invert")), isEEG),
      vis(slider("fixed lo", -5, 5, 0.01, () => r().fixed_lo, set("fixed_lo")), () => isEEG() && r().norm === "fixed"),
      vis(slider("fixed hi", -5, 5, 0.01, () => r().fixed_hi, set("fixed_hi")), () => isEEG() && r().norm === "fixed"),
      vis(slider("manual", 0, 1, 0.01, () => r().manual, set("manual")), isMan),
    );
    if (t === "chroma") {
      body.append(
        vis(select("key root", notes, () => r().key_root, (v) => set("key_root")(+v)), () => r().source === "key"),
        vis(check("minor", () => r().key_minor, set("key_minor")), () => ["key", "dial_key"].includes(r().source)),
        vis(slider("contrast", 0.2, 4, 0.1, () => r().spec_gamma, set("spec_gamma"), fmtNum, "exponent on the 12 EEG bands"), () => r().source === "spectrum"),
        vis(slider("hysteresis", 0, 1, 0.05, () => r().key_hysteresis, set("key_hysteresis"), fmtNum, "how far past a key boundary before changing key"), () => r().source === "dial_key"),
      );
    }
    if (t === "style") {
      const bSel = el("select", { "data-presets": "opt", "data-key": "style_b" });
      bSel.onchange = () => setG("style_b")(bSel.value || null);
      const axSel = el("select", { "data-axes": "1" });
      axSel.onchange = () => setG("style_axis")(axSel.value);
      body.append(
        vis(el("div", { class: "ctl" }, el("span", {}, "B preset"), bSel, el("span", {})), () => r().source === "morph"),
        vis(text("…or B text", () => cfg.gen.style_text, setG("style_text"), { placeholder: "if no B preset" },
          "morph toward a CLAP text embedding (needs the text tower on the server)"), () => r().source === "morph"),
        vis(el("div", { class: "ctl" }, el("span", {}, "axis"), axSel, el("span", {})), () => r().source === "axis"),
        vis(slider("max angle", 5, 120, 5, () => cfg.gen.style_axis_deg, setG("style_axis_deg"), (v) => v.toFixed(0) + "°"), () => r().source === "axis"),
      );
    }
    const det = el("details", {}, el("summary", {}, "settings"), body);
    box.append(el("div", { class: "route" }, head, det));
  }
}

function buildGen() {
  const box = $("gen"); box.innerHTML = "";
  const g = () => cfg.gen;
  const set = (k) => (v) => send({ type: "set_gen", key: k, value: v });
  const presetSel = el("select", { "data-presets": "req", "data-key": "preset" });
  presetSel.onchange = () => set("preset")(presetSel.value);
  const row = (label, node, tip = "") => el("div", { class: "ctl", title: tip }, el("span", {}, label), node, el("span", {}));
  const on = () => uiMode === "online";
  box.append(
    row("preset", presetSel, "seeds the opening window, supplies style A and the fixed tempo when bpm is off"),
    vis(slider("delay s", 4, 60, 0.5, () => g().delay_s, set("delay_s"), (v) => v.toFixed(1),
      "fixed brain→ear delay. Applies at the next Start. Must exceed 'min delay'."), on),
    vis(check("auto draft", () => g().auto_draft, set("auto_draft"), "halve steps for a take that would otherwise arrive late"), on),
    vis(slider("lead-in meas.", 1, 24, 1, () => g().lead_in_measures, set("lead_in_measures"), (v) => v.toFixed(0),
      "how much of the opening window to play"), on),
    slider("measures/take", 1, 16, 1, () => g().n_new, set("n_new"), (v) => v.toFixed(0),
      "smaller = finer time resolution of control, but more compute per second of audio"),
    select("steps", [["32", "32 (reference)"], ["16", "16 (draft)"], ["8", "8 (rough)"]], () => String(g().n_steps), (v) => set("n_steps")(+v)),
    check("signal guidance", () => g().guidance_on, set("guidance_on"), "CFG on the EEG/manual signals; off = one plain conditional pass"),
    vis(slider("  strength", 1, 8, 0.1, () => g().guidance, set("guidance")), () => cfg.gen.guidance_on),
    check("history", () => g().history_on, set("history_on"),
      "read the previous measures as context. Off: each take comes from the signals alone (previous generations don't carry forward; joins between takes get looser)"),
    vis(check("history guidance", () => g().hist_guidance_on, set("hist_guidance_on"), "amplify the history; extra DiT passes"), () => cfg.gen.history_on),
    vis(slider("  strength", 1, 3, 0.05, () => g().hist_guidance, set("hist_guidance")), () => cfg.gen.history_on && cfg.gen.hist_guidance_on),
    vis(slider("context noise", 0, 0.6, 0.01, () => g().ctx_noise, set("ctx_noise"), fmtNum, "noise the context is held at; higher = looser"), () => cfg.gen.history_on),
    select("authoring", [["locked", "locked (each measure its own EEG)"], ["hold", "hold (one value per take)"]], () => g().mode, set("mode")),
    slider("EEG avg s", 0.25, 15, 0.25, () => g().smooth_s, set("smooth_s"), fmtNum, "seconds of EEG averaged into each measure"),
    vis(slider("glide", 0, 1, 0.05, () => g().glide, set("glide")), () => cfg.gen.mode === "hold"),
    slider("bpm jitter", 0, 0.05, 0.001, () => g().bpm_jitter, set("bpm_jitter"), (v) => (v * 100).toFixed(1) + "%"),
    slider("seed", 0, 999, 1, () => g().seed, set("seed"), (v) => v.toFixed(0)),
    check("analyze output", () => g().analyze_output, set("analyze_output"), "re-measure the generated audio (the dots)"),
  );
}

function buildBands() {
  const box = $("bands"); box.innerHTML = "";
  for (const t of SCALARS) {
    const b = () => cfg.gen.bands[t];
    const lo = el("input", { type: "number", step: "any" }), hi = el("input", { type: "number", step: "any" });
    lo.value = b()[0]; hi.value = b()[1];
    const push = () => send({ type: "set_gen", key: "band", value: [t, +lo.value, +hi.value] });
    lo.onchange = push; hi.onchange = push;
    bindings.push({ el: lo, get: () => b()[0] }, { el: hi, get: () => b()[1] });
    box.append(el("div", { class: "band" }, el("span", {}, cfg.target_labels[t] + (b()[2] ? " (log)" : "")), lo, hi));
  }
}

function buildSynth() {
  const box = $("synth"); box.innerHTML = "";
  if (!cfg.synth) return;
  const ranges = { theta: 40, alpha: 60, beta: 30, emg: 40, noise: 30, line: 20, blink: 10 };
  for (const k of Object.keys(cfg.synth)) {
    box.append(slider(k, 0, ranges[k] || 50, 0.5, () => (cfg.synth || {})[k] ?? 0, (v) => send({ type: "set_synth", key: k, value: v })));
  }
}

function setMode(m) {
  uiMode = m;
  if (location.hash !== "#" + m) history.replaceState(null, "", "#" + m);
  document.body.className = "mode-" + m;
  document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b.dataset.mode === m));
  if (cfg) refresh();
}

function wireButtons() {
  document.querySelectorAll(".tab").forEach((b) => (b.onclick = () => setMode(b.dataset.mode)));
  $("b-music").onclick = () => send({ type: "music", action: lastStatus && lastStatus.music_on ? "stop" : "start" });
  $("b-rec").onclick = () => send({ type: "record", action: lastStatus && lastStatus.recording ? "stop" : "start" });
  $("vol").oninput = () => { send({ type: "audio", volume: +$("vol").value }); $("off-audio").volume = +$("vol").value; };
  $("src-name").onchange = () => $("src-path").classList.toggle("hidden", $("src-name").value !== "replay");
  $("b-src").onclick = () => { send({ type: "source", name: $("src-name").value, path: $("src-path").value }); setTimeout(() => { built = false; }, 400); };
  document.querySelectorAll("[data-cal]").forEach((b) => {
    b.onclick = () => send({ type: "calibrate", kind: b.dataset.cal, seconds: +$("cal-sec").value });
  });
  $("b-resetnorm").onclick = () => send({ type: "reset_norm" });
  $("b-note").onclick = () => { send({ type: "note", text: $("note").value }); $("note").value = ""; };
  $("b-save").onclick = () => $("cfg-name").value && send({ type: "save_config", name: $("cfg-name").value });
  $("b-load").onclick = () => $("cfg-list").value && send({ type: "load_config", name: $("cfg-list").value });
  $("b-map-preset").onclick = () => {
    send({ type: "map_preset", name: $("map-preset").value });
    if (uiMode === "offline" && analysis) setTimeout(() => send({ type: "offline_refresh" }), 200);
  };
  $("off-rec").onchange = () => { $("off-rec").dataset.touched = "1"; };
  $("b-off-analyze").onclick = () => send({ type: "offline_analyze", path: $("off-rec").value,
    start_s: +$("off-start").value || 0, duration_s: $("off-dur").value ? +$("off-dur").value : null });
  $("b-off-remap").onclick = () => send({ type: "offline_refresh" });
  $("b-off-gen").onclick = () => {
    send({ type: "set_gen", key: "time_scale", value: +$("off-scale").value || 1 });
    send({ type: "set_gen", key: "duration_s", value: $("off-gendur").value ? +$("off-gendur").value : null });
    setTimeout(() => send({ type: "offline_generate" }), 100);
  };
  $("b-off-stop").onclick = () => send({ type: "music", action: "stop" });
}

function renderCal() {
  const rows = Object.entries(cfg.calib || {}).filter(([, c]) => c.base || Object.keys(c.anchors || {}).length);
  if (!rows.length) { $("cal-table").textContent = "no calibration yet"; return; }
  const t = el("table", { class: "cal" }, el("tr", {}, el("td", {}, "feature"), el("td", {}, "baseline μ±σ"), el("td", {}, "low"), el("td", {}, "high")));
  for (const [n, c] of rows) {
    t.append(el("tr", {}, el("td", {}, n),
      el("td", {}, c.base ? `${fmtNum(c.base[0])} ± ${fmtNum(c.base[1])}` : "—"),
      el("td", {}, c.anchors.low != null ? fmtNum(c.anchors.low) : "—"),
      el("td", {}, c.anchors.high != null ? fmtNum(c.anchors.high) : "—")));
  }
  $("cal-table").innerHTML = ""; $("cal-table").append(t);
}
function renderStats() {
  if (!analysis || !cfg) return;
  const used = new Set(Object.values(cfg.routes).map((r) => r.feature));
  const t = el("table", { class: "stats" }, el("tr", {}, ...["feature", "mean", "std", "p5", "p50", "p95"].map((h) => el("th", {}, h))));
  const names = Object.keys(analysis.summary).sort((a, b) => (used.has(b) - used.has(a)) || a.localeCompare(b));
  for (const n of names) {
    const s = analysis.summary[n];
    t.append(el("tr", { class: used.has(n) ? "used" : "" }, el("td", {}, n),
      ...["mean", "std", "p5", "p50", "p95"].map((k) => el("td", {}, fmtNum(s[k])))));
  }
  $("off-stats").innerHTML = ""; $("off-stats").append(t);
}

// ---------------------------------------------------------------- record tab
let customLabels = JSON.parse(localStorage.getItem("om-labels") || "[]");
let lastStep = null, audioCtx = null, selectedSession = null, lastReport = null;
function beep(freq = 880, ms = 160, times = 1) {
  if (!$("proto-beep").checked) return;
  try {
    audioCtx = audioCtx || new AudioContext();
    for (let k = 0; k < times; k++) {
      const o = audioCtx.createOscillator(), g = audioCtx.createGain(), t = audioCtx.currentTime + k * (ms / 1000 + 0.08);
      o.frequency.value = freq; o.connect(g); g.connect(audioCtx.destination);
      g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(0.25, t + 0.01);
      g.gain.exponentialRampToValueAtTime(0.0001, t + ms / 1000);
      o.start(t); o.stop(t + ms / 1000 + 0.02);
    }
  } catch (e) { /* audio is optional */ }
}
function allLabels() {
  const s = new Set();
  for (const p of Object.values(cfg.protocols || {})) for (const l of p.labels) s.add(l);
  for (const l of customLabels) s.add(l);
  return [...s];
}
function buildRecord() {
  const sel = $("proto-name");
  sel.innerHTML = "";
  for (const [k, p] of Object.entries(cfg.protocols || {})) sel.append(el("option", { value: k }, p.title));
  const fill = () => {
    const p = cfg.protocols[sel.value]; if (!p) return;
    $("proto-block").value = p.block_s; $("proto-reps").value = p.reps; $("proto-note").textContent = p.note; updLen();
  };
  const updLen = () => {
    const p = cfg.protocols[sel.value]; if (!p) return;
    const sec = 8 + (+$("proto-reps").value) * p.labels.length * ((+$("proto-block").value) + 3);
    $("proto-len").textContent = `≈ ${(sec / 60).toFixed(1)} min · ${p.labels.join(" / ")}`;
  };
  sel.onchange = fill; $("proto-block").oninput = updLen; $("proto-reps").oninput = updLen;
  if (!sel.dataset.init) { sel.value = "alpha"; sel.dataset.init = "1"; }
  fill();
  $("b-proto").onclick = () => {
    const running = lastStatus && lastStatus.protocol && !lastStatus.protocol.done;
    if (running) { send({ type: "protocol_stop" }); return; }
    audioCtx = audioCtx || new AudioContext();       // user gesture: allow beeps
    send({ type: "protocol_start", name: sel.value, block_s: +$("proto-block").value,
           reps: +$("proto-reps").value, shuffle: $("proto-shuffle").checked });
  };
  $("b-lab-add").onclick = () => {
    const v = $("lab-custom").value.trim().toLowerCase(); if (!v) return;
    if (!customLabels.includes(v)) customLabels.push(v);
    localStorage.setItem("om-labels", JSON.stringify(customLabels)); $("lab-custom").value = ""; renderLabelButtons();
  };
  $("b-lab-stop").onclick = () => send({ type: "record", action: "stop" });
  const m = cfg.session_meta || {};
  for (const k of ["subject", "placement", "notes"]) {
    const i = $("meta-" + k); i.value = m[k] || (k === "placement" ? "Pz" : "");
    i.onchange = () => send({ type: "session_meta", meta: { [k]: i.value } });
  }
  renderLabelButtons();
}
function renderLabelButtons() {
  const box = $("label-btns"); box.innerHTML = "";
  const none = el("button", { "data-label": "" }, el("kbd", {}, "0"), "unlabelled");
  none.onclick = () => send({ type: "rec_label", label: null });
  box.append(none);
  allLabels().forEach((l, i) => {
    const b = el("button", { "data-label": l }, el("kbd", {}, i < 9 ? String(i + 1) : ""), l);
    b.onclick = () => send({ type: "rec_label", label: l });
    box.append(b);
  });
}
document.addEventListener("keydown", (e) => {
  if (uiMode !== "record" || ["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName)) return;
  if (!/^[0-9]$/.test(e.key)) return;
  const k = +e.key;
  if (k === 0) send({ type: "rec_label", label: null });
  else { const l = allLabels()[k - 1]; if (l) send({ type: "rec_label", label: l }); }
});
function refreshRecord() {
  if (!cfg || !cfg.sessions) return;
  const box = $("sessions");
  const t = el("table", { class: "sess" }, el("tr", {}, ...["session", "protocol", "subject · placement", "length", "labelled", ""].map((h) => el("th", {}, h))));
  for (const ss of cfg.sessions) {
    const labs = Object.entries(ss.labels).map(([k, v]) => `${k} ${v.seconds.toFixed(0)}s×${v.segments}`).join(", ") || "—";
    const an = el("button", {}, "Analyze"); an.onclick = () => { selectedSession = ss.name; send({ type: "session_analyze", name: ss.name }); };
    const off = el("button", { title: "open in the Offline tab" }, "Offline"); off.onclick = () => {
      $("off-rec").value = ss.name; $("off-rec").dataset.touched = "1"; setMode("offline");
      send({ type: "offline_analyze", path: ss.name });
    };
    t.append(el("tr", { class: ss.name === selectedSession ? "sel" : "" }, el("td", {}, ss.name), el("td", {}, ss.protocol || "—"),
      el("td", {}, [ss.subject, ss.placement].filter(Boolean).join(" · ") || "—"), el("td", {}, `${Math.round(ss.duration_s)}s`),
      el("td", { class: "small" }, labs), el("td", {}, an, off)));
  }
  box.innerHTML = ""; box.append(cfg.sessions.length ? t : el("div", { class: "small" }, "No recordings yet."));
}
function renderReport(r) {
  lastReport = r;
  const box = $("session-report"); box.innerHTML = "";
  const div = el("div", { class: "report" });
  const dec = r.decode && r.decode.norm != null
    ? `states decodable from features: <b>${(r.decode.bacc * 100).toFixed(0)}%</b> balanced accuracy (chance ${(r.decode.chance * 100).toFixed(0)}%, best lag ${r.decode.lag_s}s)`
    : "states not decodable yet (need ≥ 2 labelled segments per state)";
  div.innerHTML = `<h3>${r.name} — ${Math.round(r.duration_s)}s, ${r.artifact_pct}% artifacts</h3><div class="small">${dec}</div>` +
    (r.warnings.length ? `<div class="small warn">⚠ ${r.warnings.join("<br>⚠ ")}</div>` : "");
  // per label
  const pl = el("table", { class: "sess" }, el("tr", {}, ...["state", "seconds", "segments", "artifacts"].map((h) => el("th", {}, h))));
  for (const [c, d] of Object.entries(r.per_label)) pl.append(el("tr", {}, el("td", {}, c), el("td", {}, d.seconds.toFixed(0)),
    el("td", {}, d.segments), el("td", { class: d.artifact_pct > 30 ? "warn" : "" }, d.artifact_pct + "%")));
  div.append(el("h3", {}, "States"), pl);
  // features
  const ft = el("table", { class: "sess" }, el("tr", {}, el("th", {}, "feature"), el("th", { title: "share of the feature's variance explained by the state" }, "η²"),
    el("th", {}, "AUC"), el("th", {}, "best pair"), el("th", { title: "Cohen's d for that pair" }, "d"), ...r.classes.map((c) => el("th", {}, c))));
  for (const f of r.features.slice(0, 14)) {
    ft.append(el("tr", {}, el("td", { title: (cfg.features || {})[f.feature] || "" }, f.feature), el("td", {}, f.eta2.toFixed(2)),
      el("td", {}, f.auc.toFixed(2)), el("td", { class: "small" }, f.pair || "—"), el("td", {}, f.d.toFixed(1)),
      ...r.classes.map((c) => el("td", {}, r.per_label[c].medians[f.feature] != null ? fmtNum(r.per_label[c].medians[f.feature]) : "—"))));
  }
  div.append(el("h3", {}, "Which features respond (most first)"), ft);
  // baseline
  const opts = [["", "—"], ...r.classes.map((c) => [c, c])];
  const mk = (id, label, guess) => {
    const s = el("select", { id }); for (const [v, t] of opts) s.append(el("option", { value: v }, t)); s.value = guess || "";
    return el("div", { class: "ctl" }, el("span", {}, label), s, el("span", {}));
  };
  const guess = (want) => r.classes.find((c) => want.some((w) => c.includes(w))) || "";
  const apply = el("button", { class: "primary" }, "Use as baseline");
  apply.onclick = () => send({ type: "session_baseline", name: r.name, baseline: $("bl-base").value || null,
                               low: $("bl-low").value || null, high: $("bl-high").value || null });
  div.append(el("h3", {}, "Use as baseline"),
    el("div", { class: "small" }, "Sets the 'baseline' normalizer (mean/std of one state), the 'range' normalizer's anchors (low → high), and the reference for session_z / session_quantile — for every feature."),
    mk("bl-base", "baseline state", guess(["rest", "open"])), mk("bl-low", "range low", guess(["open", "rest"])),
    mk("bl-high", "range high", guess(["closed", "clench"])), apply);
  box.append(div);
}
function updateRecordStatus(s) {
  const p = s.protocol, box = $("instr");
  const running = p && !p.done;
  $("b-proto").textContent = running ? "■ Stop protocol" : "▶ Start protocol";
  $("b-proto").classList.toggle("on", !!running);
  if (running) {
    box.className = "instr " + p.kind;
    $("instr-label").textContent = p.kind === "block" ? p.label : p.kind === "setup" ? "get ready" : `next: ${p.next}`;
    $("instr-text").textContent = p.instruction;
    $("instr-count").textContent = Math.ceil(p.left);
    $("instr-prog").style.width = `${(p.progress * 100).toFixed(1)}%`;
    $("instr-next").textContent = `step ${p.step + 1}/${p.n_steps} · ${Math.round(p.total_left)}s left · recording → ${p.dir}`;
    if (lastStep !== p.step) { if (lastStep !== null) beep(p.kind === "block" ? 880 : 660); lastStep = p.step; }
  } else if (p && p.done && lastStep !== null) {
    beep(990, 140, 2); lastStep = null;
    box.className = "instr";
    $("instr-label").textContent = p.cancelled ? "stopped" : "done";
    $("instr-text").textContent = p.cancelled ? "Protocol stopped; what was recorded is saved." : "Saved. The analysis appears under Sessions.";
    $("instr-count").textContent = ""; $("instr-prog").style.width = "100%"; $("instr-next").textContent = p.dir || "";
  }
  $("lab-now").textContent = s.recording ? `● recording · current: ${s.label || "unlabelled"}` : "not recording";
  for (const b of document.querySelectorAll("#label-btns button")) b.classList.toggle("on", (b.dataset.label || null) === (s.label || null) && !!s.recording);
  $("lab-summary").textContent = s.label_summary ? Object.entries(s.label_summary).map(([k, v]) => `${k}: ${v.seconds.toFixed(0)}s × ${v.segments}`).join(" · ") : "";
}

// ---------------------------------------------------------------- live readouts
function unitFmt(t, v) {
  if (t === "bpm") return `${v.toFixed(1)} bpm`;
  if (t === "flatness") return v.toExponential(1);
  if (t === "density") return v.toFixed(2) + "/s";
  return v.toFixed(3);
}
function updateMeters(m) {
  if (!cfg) return;
  for (const t of cfg.targets) {
    const r = cfg.routes[t], val = $(`v-${t}`);
    if (!val) continue;
    if (t === "chroma") {
      const bars = $("cm-chroma").children;
      for (let i = 0; i < 12; i++) bars[i].style.height = m.chroma && r.source !== "off" ? `${(m.chroma[i] * 100).toFixed(0)}%` : "1px";
      val.textContent = r.source === "off" ? "off" : r.source === "dial_key" && m.chroma
        ? cfg.notes[m.chroma.indexOf(Math.max(...m.chroma))] + (r.key_minor ? "m" : "") : SRC_LABEL[r.source];
      continue;
    }
    const bar = $(`m-${t}`);
    const off = r.source === "off", driven = driveOf(t, r) !== "off";
    bar.style.width = `${(m.dials[t] * 100).toFixed(1)}%`;
    bar.style.background = off || !driven ? "#444" : COLORS[t];
    val.textContent = off ? "off" : t === "style" ? (driven ? m.dials[t].toFixed(2) : SRC_LABEL[r.source]) : unitFmt(t, m.vals[t]);
    val.title = `dial ${m.dials[t].toFixed(3)}  z ${(m.z[t] || 0).toFixed(2)}`;
  }
  $("nch").textContent = m.feats ? `${m.nch} pins used${m.artifact ? " · ARTIFACT (holding)" : ""}` : "warming up…";
}
function pill(id, txt, cls) {
  const p = $(id), oo = p.classList.contains("online-only");
  p.textContent = txt; p.className = "pill " + (cls || "") + (oo ? " online-only" : "");
}
function updateStatus(s) {
  const src = s.source || {};
  const fs = s.fs ? s.fs.toFixed(1) + " Hz" : "—";
  pill("p-eeg", `EEG ${src.name || ""}: ${src.status || "—"} · ${fs} · loss ${s.loss_pct.toFixed(1)}%` +
    (src.reconnects ? ` · ${src.reconnects} reconnect${src.reconnects > 1 ? "s" : ""}` : ""),
    src.status === "streaming" ? (s.loss_pct > 2 || src.reconnects ? "warn" : "ok")
      : src.status === "error" ? "bad" : (src.status || "").startsWith("reconnecting") ? "warn" : "");
  $("p-eeg").title = [src.error && "last error: " + src.error,
    src.reset_reasons && src.reset_reasons.length && "board boots: " + src.reset_reasons.join(", "),
    src.last_hb].filter(Boolean).join("\n") || "EEG source · measured sample rate · packet loss";
  if (src.status === "error" && src.error) flash(src.error);
  if (src.name && document.activeElement !== $("src-name") && $("src-name").value !== src.name) {
    $("src-name").value = src.name; $("src-path").classList.toggle("hidden", src.name !== "replay");
  }
  const L = s.link;
  pill("p-link", `GPU ${L.state}${L.rtt_ms != null ? " · " + L.rtt_ms.toFixed(0) + " ms" : ""}`, L.state === "connected" ? "ok" : "bad");
  pill("p-server", `server ${s.server_state}${s.server_msg ? ": " + s.server_msg : ""}`,
    s.server_state === "error" ? "bad" : s.server_state === "idle" ? "" : "ok");
  const a = s.audio;
  const delay = cfg ? cfg.gen.delay_s : 0;
  pill("p-play", a.preroll_s ? `pre-roll ${a.preroll_s.toFixed(1)} s` : a.rebuffering ? "RE-BUFFERING"
    : `buffer ${a.buffered_s.toFixed(1)} s · ${a.rebuffers} rebuf · ${a.glitches} glitch · ${a.join_jumps} seam`,
    a.rebuffering ? "bad" : (a.rebuffers || a.glitches || a.join_jumps) ? "warn" : s.music_on ? "ok" : "");
  $("p-play").title = `re-buffers: queue ran dry\nglitches: audio callback ran late (CoreAudio underflow)\n` +
    `seams: chunk joins with a click-sized step (largest ${a.max_join})\nnear full scale in last chunk: ${(a.clipped * 100).toFixed(2)}% (soft-limited)`;
  pill("p-margin", `margin ${s.margin_s != null ? s.margin_s.toFixed(1) + " s" : "—"}${s.degraded ? " · " + s.degraded + " draft" : ""}`,
    s.margin_s == null ? "" : s.margin_s > 2 ? "ok" : s.margin_s > 0 ? "warn" : "bad");
  pill("p-mindelay", `min delay ${s.min_delay_s != null ? s.min_delay_s.toFixed(1) + " s" : "—"} (set ${delay})`,
    s.min_delay_s == null ? "" : s.min_delay_s < delay ? "ok" : "bad");
  pill("p-rtf", `RTF ${s.rtf != null ? s.rtf.toFixed(2) : "—"}${s.compute_s != null ? " (" + s.compute_s.toFixed(1) + " s/take)" : ""}`,
    s.rtf == null ? "" : s.rtf < 0.8 ? "ok" : s.rtf < 1 ? "warn" : "bad");
  pill("p-lat", `brain→ear ${s.latency_s != null ? s.latency_s.toFixed(1) + " s" : "—"}`, "");
  $("b-music").textContent = s.music_on ? "■ Stop" : "▶ Start";
  $("b-music").classList.toggle("on", s.music_on);
  $("b-rec").textContent = s.recording ? "■ Stop recording" : "● Record";
  $("b-rec").classList.toggle("on", !!s.recording);
  $("rec-dir").textContent = s.recording ? `→ ${s.recording}` : "";
  if (cfg) updateRecordStatus(s);
  $("cal-status").textContent = s.calibrating ? `recording ${s.calibrating.kind}… ${s.calibrating.left}s — hold the state` : "";
  if (s.run_mode === "offline" && s.progress != null) $("off-prog").style.width = `${(s.progress * 100).toFixed(1)}%`;
  if (cfg) {
    const q = s.quality, counts = {};
    for (let c = 0; c < N_CH; c++) {
      const d = $(`dot-${c}`), ch = $(`ch-${c}`);
      if (!d) continue;
      const st = q ? q.status[c] : "";
      counts[st] = (counts[st] || 0) + 1;
      d.className = "dot q-" + (st || "none").replace(" ", "-");
      ch.classList.toggle("off", !cfg.dsp.channels[c]);
      ch.title = q ? `ch ${c + 1}: ${st}\np-p ${q.pp[c]} µV · LINE ${q.line[c]} µV · SNR ${q.snr[c]} dB` +
        (s.used[c] ? "" : "\nnot used for features") : "warming up";
    }
    $("q-summary").textContent = q ? ["good", "fair", "poor", "railed", "no contact"]
      .filter((k) => counts[k]).map((k) => `${counts[k]} ${k}`).join(" · ") : "";
  }
}

// ---------------------------------------------------------------- drawing
function fit(c) {
  // pin CSS height to the markup's height attribute once; otherwise a
  // width:100% canvas rescales with its own backing store every frame
  if (!c.dataset.h) { c.dataset.h = c.getAttribute("height"); c.style.height = c.dataset.h + "px"; }
  const dpr = window.devicePixelRatio || 1, w = c.clientWidth, h = +c.dataset.h;
  if (c.width !== Math.round(w * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
  const g = c.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); return [g, w, h];
}
function drawEEG() {
  const [g, w, h] = fit($("c-eeg"));
  g.clearRect(0, 0, w, h);
  const win = +$("eeg-win").value, scale = +$("eeg-scale").value;
  const N = Math.floor(win * eeg.fs), n = Math.min(eeg.n, N, eeg.cap);
  const lane = h / N_CH;
  g.strokeStyle = "#1c2029"; g.lineWidth = 1;
  for (let s = 1; s < win; s++) { const x = (s / win) * w; g.beginPath(); g.moveTo(x, 0); g.lineTo(x, h); g.stroke(); }
  if (n < 2) return;
  const per = N / w, X = (k) => w - ((n - 1 - k) / N) * w;
  for (let c = 0; c < N_CH; c++) {
    const y0 = lane * (c + 0.5), d = eeg.data[c];
    const at = (k) => d[(eeg.n - n + k) % eeg.cap];
    g.strokeStyle = !lastStatus || lastStatus.used[c] ? "#9fb3d9" : "#4a4f5c"; g.beginPath();
    if (per <= 3) {
      for (let k = 0; k < n; k++) { const y = y0 - (at(k) / scale) * lane * 0.5; k ? g.lineTo(X(k), y) : g.moveTo(X(k), y); }
    } else {
      const x0 = Math.floor(X(0));
      for (let x = Math.max(0, x0); x < w; x++) {
        const a = Math.max(0, Math.floor((x - x0) * per)), b = Math.min(n, Math.max(a + 1, Math.floor((x + 1 - x0) * per)));
        let lo = Infinity, hi = -Infinity;
        for (let k = a; k < b; k++) { const v = at(k); if (v < lo) lo = v; if (v > hi) hi = v; }
        if (lo === Infinity) continue;
        const ya = y0 - (hi / scale) * lane * 0.5, yb = y0 - (lo / scale) * lane * 0.5;
        if (x === Math.max(0, x0)) g.moveTo(x, ya);
        g.lineTo(x, ya); g.lineTo(x, yb);
      }
    }
    g.stroke();
  }
  const ov = $("eeg-overlay").value;
  if (ov !== "none" && cfg) {
    g.fillStyle = "#0c0e1288"; g.fillRect(0, 0, w, h);
    drawInOut(g, 0, w, 0, h, win, { stack: false, inp: ov !== "out", out: ov !== "in" });
    g.font = "11px sans-serif"; let lx = 6;
    for (const t of activeRoutes()) {
      g.fillStyle = COLORS[t]; g.fillRect(lx, 6, 10, 3); g.fillStyle = "#c8ccd6"; g.fillText(t, lx + 13, 11); lx += 22 + g.measureText(t).width;
    }
    g.fillStyle = "#8a91a2"; g.fillText("solid = dial out · dashed = feature in", lx + 6, 11);
  }
}
const BAND_SHADE = [["δ", 1, 4], ["θ", 4, 8], ["α", 8, 13], ["β", 13, 30], ["γ", 30, 45], ["EMG", 65, 95]];
function drawSpec() {
  const [g, w, h] = fit($("c-spec"));
  g.clearRect(0, 0, w, h);
  const X = (f) => (f / 100) * w;
  BAND_SHADE.forEach(([n, a, b], i) => {
    g.fillStyle = i % 2 ? "#151922" : "#12161e"; g.fillRect(X(a), 0, X(b) - X(a), h);
    g.fillStyle = "#59617a"; g.fillText(n, X(a) + 3, 12);
  });
  if (!spectrum) return;
  const p = spectrum.p, lo = Math.min(...p), hi = Math.max(...p);
  const Y = (v) => h - 6 - ((v - lo) / Math.max(1e-6, hi - lo)) * (h - 22);
  g.strokeStyle = "#7aa2ff"; g.lineWidth = 1.5; g.beginPath();
  spectrum.f.forEach((f, i) => (i ? g.lineTo(X(f), Y(p[i])) : g.moveTo(X(f), Y(p[i]))));
  g.stroke();
}
// ---- what comes in (features) and what goes out (dials) --------------------
function activeRoutes() {
  return cfg.targets.filter((t) => driveOf(t, cfg.routes[t]) !== "off");
}
function featVal(d, expr) {
  if (!d.feats) return null;
  if (expr in d.feats) return d.feats[expr];
  let tot = 0;
  for (const m of expr.replace(/\s+/g, "").matchAll(/([+-]?)(\d*\.?\d+)?\*?([A-Za-z_][A-Za-z0-9_]*)/g)) {
    if (!(m[3] in d.feats)) return null;
    tot += (m[1] === "-" ? -1 : 1) * (m[2] ? +m[2] : 1) * d.feats[m[3]];
  }
  return tot;
}
function drawInOut(g, x0, w, y0, h, span, opts) {
  // one band per driven route: dashed = its feature (in, min-max scaled over the
  // visible window), solid = the dial it produced (out, 0..1)
  const routes = activeRoutes();
  if (!routes.length || !dialHist.length) return;
  const tN = Date.now() / 1000, X = (t) => x0 + w - ((tN - t) / span) * w;
  const vis = dialHist.filter((d) => tN - d.t <= span);
  const lane = opts.stack ? h / routes.length : h;
  routes.forEach((t, i) => {
    const r = cfg.routes[t], top = y0 + (opts.stack ? i * lane : 0);
    const Y = (u) => top + lane - 4 - u * (lane - 8);
    if (opts.stack) {
      g.fillStyle = i % 2 ? "#10131a" : "#0d1016"; g.fillRect(x0, top, w, lane);
      g.fillStyle = "#8a91a2"; g.font = "11px sans-serif";
      g.fillText(`${r.source === "manual" || r.drive === "manual" && !SCALARS.includes(t) ? "manual" : r.feature}  →  ${t}`, x0 + 4, top + 12);
    }
    if (opts.inp && driveOf(t, r) === "eeg") {
      const vals = vis.map((d) => featVal(d, r.feature)).filter((v) => v != null && isFinite(v));
      if (vals.length > 1) {
        const lo = Math.min(...vals), hi = Math.max(...vals);
        g.save(); g.setLineDash([4, 3]); g.strokeStyle = COLORS[t] + "aa"; g.lineWidth = 1.2; g.beginPath();
        let prevT = null;
        for (const d of vis) {
          const v = featVal(d, r.feature); if (v == null) continue;
          const y = Y((v - lo) / Math.max(1e-12, hi - lo));
          (prevT === null || d.t - prevT > 1.5) ? g.moveTo(X(d.t), y) : g.lineTo(X(d.t), y);
          prevT = d.t;
        }
        g.stroke(); g.restore();
        if (opts.stack) { g.fillStyle = "#6b7183"; g.fillText(`${fmtNum(lo)} … ${fmtNum(hi)}`, x0 + w - 90, top + 12); }
      }
    }
    if (opts.out) {
      g.strokeStyle = COLORS[t]; g.lineWidth = opts.stack ? 2 : 2.2; g.beginPath();
      vis.forEach((d, k) => ((k && d.t - vis[k - 1].t <= 1.5) ? g.lineTo(X(d.t), Y(d.dials[t])) : g.moveTo(X(d.t), Y(d.dials[t]))));
      g.stroke();
    }
  });
  g.fillStyle = "#3a1a2066";
  for (const d of vis) if (d.artifact) g.fillRect(X(d.t) - 1, y0, 3, h);
}
function drawDials() {
  const [g, w, h] = fit($("c-dials"));
  g.clearRect(0, 0, w, h);
  if (!cfg) return;
  drawInOut(g, 0, w, 0, h, +$("io-span").value, { stack: true, inp: true, out: true });
}

function valueToDial(t, v) {
  const [lo, hi, log] = cfg.gen.bands[t];
  const u = log ? (Math.log(v) - Math.log(lo)) / (Math.log(hi) - Math.log(lo)) : (v - lo) / (hi - lo);
  return Math.min(1, Math.max(0, u));
}
function hatch(g, x, y, w, h) {
  g.save(); g.beginPath(); g.rect(x, y, w, h); g.clip();
  g.strokeStyle = "#2a2f3b"; g.lineWidth = 1;
  for (let k = -h; k < w; k += 6) { g.beginPath(); g.moveTo(x + k, y + h); g.lineTo(x + k + h, y); g.stroke(); }
  g.restore();
}
function drawOut() {
  const [g, w, h] = fit($("c-out"));
  g.clearRect(0, 0, w, h);
  if (!cfg) return;
  const offline = uiMode === "offline", V = views[offline ? "offline" : "online"];
  const head = offline ? $("off-audio").currentTime * sr : played;
  const past = offline ? 30 : 40, future = offline ? 30 : (cfg.gen.delay_s + 8), span = past + future;
  const t0 = head / sr - past, X = (s) => ((s / sr - t0) / span) * w;
  const lanes = ["rms", "bpm", "density", "zcr", "flatness", "chroma", "style"];
  const waveH = 56, laneH = (h - waveH - 12) / lanes.length;
  g.fillStyle = "#3b4766";
  for (const wv of V.waves) {
    const x0 = X(wv.start), x1 = X(wv.start + wv.env.length * wv.step);
    if (x1 < 0 || x0 > w) continue;
    for (let i = 0; i < wv.env.length; i++) {
      const x = X(wv.start + i * wv.step);
      if (x < -2 || x > w + 2) continue;
      const a = wv.env[i] * waveH * 0.95;
      g.fillRect(x, waveH / 2 - a / 2, Math.max(1, (wv.step / sr / span) * w), a);
    }
  }
  lanes.forEach((t, li) => {
    const y0 = waveH + 10 + li * laneH, lh = laneH - 2;
    g.fillStyle = "#12151c"; g.fillRect(0, y0, w, lh);
    for (const [m, r] of V.takes) {
      if (r.start == null || r.end == null) continue;
      const xa = X(r.start), xb = X(r.end);
      if (xb < 0 || xa > w) continue;
      const act = (r.active || []).includes(t);
      if (!act) { hatch(g, xa, y0, xb - xa, lh); continue; }
      if (t === "chroma") {
        if (!r.chroma) { hatch(g, xa, y0, xb - xa, lh); continue; }
        for (let k = 0; k < 12; k++) {
          const c = Math.round(40 + r.chroma[k] * 200);
          g.fillStyle = `rgb(${c},${Math.round(c * 0.45)},${Math.round(c * 0.75)})`;
          g.fillRect(xa + 0.5, y0 + lh - (k + 1) * (lh / 12), Math.max(1, xb - xa - 1), lh / 12);
        }
        continue;
      }
      const u = t === "style" ? (r.dials.style ?? 0) : valueToDial(t, r.vals[t] ?? 0);
      g.fillStyle = COLORS[t] + (r.seq % 2 ? "cc" : "88");
      g.fillRect(xa + 0.5, y0 + lh - u * (lh - 2), Math.max(1, xb - xa - 1), u * (lh - 2));
      if (r.early) { g.fillStyle = "#ff5a6b"; g.fillRect(xa, y0, Math.max(1, xb - xa), 2); }
      const mm = V.measured.get(m);
      if (mm && mm[t] != null) {
        g.fillStyle = "#fff"; g.beginPath();
        g.arc((xa + xb) / 2, y0 + lh - valueToDial(t, mm[t]) * (lh - 2), 2.5, 0, 7); g.fill();
      }
    }
    g.fillStyle = "#8a91a2"; g.fillText(t, 4, y0 + 11);
  });
  const xp = X(head);
  g.strokeStyle = "#ff5a6b"; g.lineWidth = 2; g.beginPath(); g.moveTo(xp, 0); g.lineTo(xp, h); g.stroke();
  g.fillStyle = "#59617a"; g.fillText(`−${past}s`, 2, h - 2); g.fillText(`+${future.toFixed(0)}s`, w - 30, h - 2);
}
function drawOffline() {
  const [g, w, h] = fit($("c-off"));
  g.clearRect(0, 0, w, h);
  if (!analysis || !cfg) { g.fillStyle = "#59617a"; g.fillText("Analyze a recording to see its whole session here.", 12, 20); return; }
  const T = analysis.t, dur = Math.max(1e-6, analysis.duration);
  const X = (t) => (t / dur) * w, Y = (u) => h - 14 - u * (h - 24);
  g.fillStyle = "#3a1a20";
  analysis.artifact.forEach((a, i) => { if (a > 0.5) g.fillRect(X(T[i]) - 1, 0, 2, h - 12); });
  const f = $("off-feat").value;
  if (f && analysis.feats[f]) {
    const v = analysis.feats[f], lo = Math.min(...v), hi = Math.max(...v);
    g.strokeStyle = "#6b7183"; g.lineWidth = 1; g.beginPath();
    v.forEach((x, i) => { const y = Y((x - lo) / Math.max(1e-9, hi - lo)); i ? g.lineTo(X(T[i]), y) : g.moveTo(X(T[i]), y); });
    g.stroke();
  }
  for (const t of cfg.targets) {
    if (driveOf(t, cfg.routes[t]) === "off") continue;
    const v = analysis.dials[t];
    g.strokeStyle = COLORS[t]; g.lineWidth = 1.4; g.beginPath();
    v.forEach((u, i) => (i ? g.lineTo(X(T[i]), Y(u)) : g.moveTo(X(T[i]), Y(u))));
    g.stroke();
  }
  const a = $("off-audio");
  if (a.getAttribute("src") && a.currentTime > 0) {
    const x = X(a.currentTime * (cfg.gen.time_scale || 1) + (cfg.gen.offset_s || 0));
    g.strokeStyle = "#ff5a6b"; g.lineWidth = 2; g.beginPath(); g.moveTo(x, 0); g.lineTo(x, h - 12); g.stroke();
  }
  g.fillStyle = "#59617a";
  const step = Math.max(10, Math.ceil(dur / 10 / 10) * 10);
  for (let s = 0; s <= dur; s += step) g.fillText(`${s.toFixed(0)}s`, X(s) + 2, h - 2);
}
function frame() {
  try {
    if (uiMode === "offline") drawOffline(); else { drawEEG(); drawSpec(); drawDials(); }
    if (uiMode !== "record") drawOut();
  } catch (e) { console.error(e); }
  if (!built && cfg) build();
  requestAnimationFrame(frame);
}
const modeFromHash = () => (["#offline", "#record"].includes(location.hash) ? location.hash.slice(1) : "online");
setMode(modeFromHash());
window.addEventListener("hashchange", () => setMode(modeFromHash()));
connect();
requestAnimationFrame(frame);
