"""
preset_extractor.py
===================

Builds real per-measure `Conditioning` objects (for the demo's presets) from
audio files, mirroring the DSP in generate_chroma_dataset.py EXACTLY so the
values land on the same manifold the model was trained on.

Per measure (bounded by downbeats), at sr=16000, hop_length=1024, n_fft=2048:
    chroma   (12,) = mean over time of librosa.feature.chroma_cqt
    rms       ()   = mean of librosa.feature.rms
    density   ()   = onsets_in_measure / measure_duration_sec
    zcr       ()   = mean of librosa.feature.zero_crossing_rate
    flatness  ()   = mean of librosa.feature.spectral_flatness
    bpm       ()   = 60 * TARGET_SIG / measure_duration_sec  (4 beats/measure)
    style    (512,) = CLAP audio tower, centered ~10s window (SEAM 1)

Downbeats come from <your source>  (SEAM 2 — see _get_downbeats).

The output Conditioning has exactly N_CHUNKS measures: if a song yields more we
take the first N_CHUNKS; if fewer we tile/pad (see _fit_to_n_chunks).

Usage
-----
    from preset_extractor import extract_conditioning, build_preset_cond
    cond = extract_conditioning("song.wav")                    # one Conditioning
    presets = build_preset_cond({"lofi": "a.wav", "edm": "b.wav"})  # dict

    # same DSP over audio whose measure boundaries are already known -- used by
    # RealBackend to read the conditioning back off its own renders
    from preset_extractor import analyze_measures
    signals = analyze_measures(wav, downbeat_times, n_chunks=24)

This module imports librosa lazily (only when you actually extract), so importing
it never disturbs the mock/GPU demo path.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from backend import (
    Conditioning, N_CHUNKS, STYLE_DIM, TARGET_SIG, SR,
)

# DSP config — MUST match generate_chroma_dataset.py
HOP_LENGTH = 1024
N_FFT = 2048


# =============================================================================
# SEAM 1: style embedding — now CLAP (clap.py), not contrast.py.
#
# The DiT is conditioned on the 512-d CLAP space, which holds BOTH modalities:
#   audio: raw waveform [B, 1, 163830]   --encode_audio--> [B, 512], L2-normed
#   text : T5 tokens    [B, 256, 1024]   --encode_text -->  [B, 512], L2-normed
#
# Per-measure AUDIO style mirrors generate_clap_dataset.build_audio EXACTLY:
# one ~10s raw window CENTERED on the measure's span (clamped to the song, zero-
# padded if the song is shorter), pushed through the raw-audio tower. Adjacent
# measures share most of their window, so the per-measure style is smooth — that
# is the intended behavior, not an artifact.
#
# NOTE (vs. the old contrast.py path): this is NOT whole-song chunking with an
# overlap-weighted average anymore. The audio tower is a full raw-waveform
# encoder trained on ~10s crops; centered windows are what the DiT's style bin
# was built from.
# =============================================================================
STYLE_N_SAMPLES = 16383 * 10          # 163830 samples (~10.24s) — train_clap n_samples
STYLE_BATCH = 64                      # centered windows per forward chunk
_CLAP_MODEL = None
_STYLE_CKPT = os.environ.get("STYLE_CKPT", "style.pt")


def _aux_device():
    """Device for the CONDITIONING-side models (CLAP, T5) — not the DiT stack.

    On ZeroGPU there is no usable CUDA device outside a @spaces.GPU call, and
    these run during preset extraction / prompt entry, which are CPU-side. They
    are also small (a 12-layer ViT over ~240 patches; a T5 encoder over 256
    tokens), so CPU costs a second or two per call. Off ZeroGPU (a real GPU box)
    we still take CUDA. Force either way with AUX_DEVICE."""
    import torch
    env = os.environ.get("AUX_DEVICE")
    if env:
        return env
    try:
        import spaces  # noqa: F401  -- only importable inside a ZeroGPU Space
        return "cpu"
    except Exception:  # noqa: BLE001
        return "cuda" if torch.cuda.is_available() else "cpu"


def _load_clap():
    """Load the trained CLAP model from style.pt. Cached across calls."""
    global _CLAP_MODEL
    if _CLAP_MODEL is not None:
        return _CLAP_MODEL
    import torch
    from clap import CLAP
    ck = torch.load(_STYLE_CKPT, map_location="cpu", weights_only=False)
    model = CLAP(**ck["model_args"])
    # ALWAYS the raw 'model' weights, never EMA — mirrors
    # generate_clap_dataset.load_clap, which is what actually produced the style
    # vectors the DiT was conditioned on.
    if ck.get("ema") is not None:
        print("[preset_extractor] CLAP: using raw model weights "
              "(ignoring EMA — matches generate_clap_dataset.py)")
    sd = dict(ck["model"])
    # train_clap.py compiles the TOWERS, not the whole model, so torch.compile's
    # '_orig_mod.' lands MID-key ('audio_tower._orig_mod.blocks...'). A
    # startswith() strip silently misses those and load_state_dict then fails
    # with every tower key missing. Replace every occurrence.
    for k in list(sd.keys()):
        if "_orig_mod." in k:
            sd[k.replace("_orig_mod.", "")] = sd.pop(k)
    model.load_state_dict(sd)
    # eval() is load-bearing: it disables SpecAugment, DropPath and the text-side
    # dropout, all of which clap.py gates on self.training.
    model.eval()
    dev = _aux_device()
    model.to(dev)
    args = ck["model_args"]
    if int(args.get("proj_dim", STYLE_DIM)) != STYLE_DIM:
        raise ValueError(
            f"{_STYLE_CKPT}: CLAP proj_dim={args.get('proj_dim')} but "
            f"backend.STYLE_DIM={STYLE_DIM}; the DiT would be fed the wrong "
            f"style width.")
    _CLAP_TEXT_DIM[0] = int(args.get("text_dim", 1024))
    print(f"[preset_extractor] CLAP loaded on {dev}: proj_dim={STYLE_DIM}, "
          f"text_dim={_CLAP_TEXT_DIM[0]}, n_text_tokens={_N_TEXT_TOKENS}")
    _CLAP_MODEL = model
    return model


def _extract_style(wav: np.ndarray, measure_times: list[tuple[float, float]],
                   sr: int = SR) -> np.ndarray:
    """Return (T, STYLE_DIM) CLAP audio embeddings, one per measure.

    measure_times: (t_start, t_end) in SECONDS for exactly the measures the DSP
    loop kept. Reproduces generate_clap_dataset.build_audio's centered-window
    construction sample-for-sample."""
    import torch
    model = _load_clap()
    dev = next(model.parameters()).device

    n = STYLE_N_SAMPLES
    half = n // 2
    windows = np.empty((len(measure_times), n), dtype=np.float32)
    for i, (t_start, t_end) in enumerate(measure_times):
        fs, fe = int(t_start * sr), int(t_end * sr)
        center = (fs + fe) // 2
        lo = center - half
        hi = lo + n
        if lo < 0:
            lo, hi = 0, n
        if hi > len(wav):
            hi, lo = len(wav), max(0, len(wav) - n)
        seg = wav[lo:hi]
        if len(seg) < n:
            seg = np.pad(seg, (0, n - len(seg)))
        windows[i] = seg

    out = np.empty((len(measure_times), STYLE_DIM), dtype=np.float32)
    with torch.no_grad():
        for c in range(0, len(windows), STYLE_BATCH):
            chunk = torch.from_numpy(windows[c:c + STYLE_BATCH]).unsqueeze(1).to(dev)
            emb = model.encode_audio(chunk)          # (b, 512), already L2-normed
            out[c:c + chunk.shape[0]] = emb.float().cpu().numpy()
    return out


# =============================================================================
# SEAM 1b: TEXT conditioning.
#
# clap.TextTower consumes FROZEN PRECOMPUTED T5 token features, not strings:
# (B, 256, 1024) `last_hidden_state`, zero-padded to max_length. The caption bin
# the tower trained on (caption_embeddings_expanded_shuffled_*.bin, shape
# (N, 256, 1024)) was produced by generate_captions_dataset.py with
# T5EncoderModel("t5-large") — d_model=1024. T5-v1.1-xxl is d_model=4096 and
# cannot produce these features, so `t5-large` is the default here; override
# with T5_MODEL if the checkpoint you upload was built with a different
# 1024-wide encoder (e.g. google/t5-v1_1-large). The width is asserted against
# the CLAP checkpoint's text_dim on first use, so a mismatch fails loudly.
# =============================================================================
_N_TEXT_TOKENS = 256                  # generate_captions_dataset.py max_tokens
_CLAP_TEXT_DIM = [1024]               # filled in from the checkpoint by _load_clap
_T5_MODEL_ID = os.environ.get("T5_MODEL", "t5-large")
_T5 = None


def _load_t5():
    """Load (and cache) the frozen T5 encoder + tokenizer. Lazy: this pulls a
    ~3 GB checkpoint, so it only happens the first time text is actually used."""
    global _T5
    if _T5 is not None:
        return _T5
    import torch
    from transformers import AutoTokenizer, T5EncoderModel
    print(f"[preset_extractor] loading text encoder {_T5_MODEL_ID!r} …")
    # use_fast=False -> the sentencepiece T5Tokenizer, which is what
    # generate_captions_dataset.py used (T5Tokenizer.from_pretrained). The fast
    # tokenizer agrees on ordinary prose, but there is no reason to risk a
    # tokenization difference on the one path that has to match training.
    try:
        tok = AutoTokenizer.from_pretrained(_T5_MODEL_ID, use_fast=False)
    except Exception as e:  # noqa: BLE001  -- sentencepiece missing
        print(f"[preset_extractor] slow T5 tokenizer unavailable ({e}); "
              f"falling back to the fast tokenizer")
        tok = AutoTokenizer.from_pretrained(_T5_MODEL_ID)
    enc = T5EncoderModel.from_pretrained(_T5_MODEL_ID)
    enc.eval()
    want = _CLAP_TEXT_DIM[0]
    if int(enc.config.d_model) != want:
        raise ValueError(
            f"text encoder {_T5_MODEL_ID!r} has d_model={enc.config.d_model} but "
            f"the CLAP checkpoint expects text_dim={want}. Set T5_MODEL to the "
            f"encoder the caption bin was built with.")
    enc.to(_aux_device())
    _T5 = (tok, enc)
    return _T5


def embed_text(prompt: str) -> np.ndarray:
    """Return the (STYLE_DIM,) CLAP TEXT embedding of `prompt`.

    Reproduces generate_captions_dataset.py's tokenization (max_length padding
    to 256, truncation) and generate_clap_dataset.build_text's masking (valid ==
    any nonzero feature, first token forced valid), so the vector lands in the
    same place a training caption would.

    On that mask: generate_captions_dataset.py writes `last_hidden_state`
    VERBATIM, and HF's T5 does not zero pad positions (the attention mask hides
    pads as keys; as queries they still attend to real tokens and emit nonzero
    vectors). So `abs().sum(-1) > 0` is effectively all-True, and the tower
    trained attending over all 256 positions. Reproducing the same "wrong" mask
    is the point — do not tighten it to the tokenizer's attention_mask."""
    import torch
    model = _load_clap()                      # also fixes _CLAP_TEXT_DIM
    tok, enc = _load_t5()
    p0 = next(model.parameters())
    dev, dtype = p0.device, p0.dtype

    with torch.no_grad():
        inputs = tok([prompt], padding="max_length", truncation=True,
                     max_length=_N_TEXT_TOKENS, return_tensors="pt")
        inputs = {k: v.to(next(enc.parameters()).device) for k, v in inputs.items()}
        feats = enc(**inputs).last_hidden_state          # (1, 256, 1024)
        # The bin the tower trained on is float16, so round-trip through fp16 to
        # reproduce that precision -- then cast BACK to the model's dtype. In
        # training, fp16 features went straight into an autocast(bf16) region and
        # the matmul dtype was resolved for us; here the tower is plain fp32 on
        # CPU (ZeroGPU: no CUDA outside @spaces.GPU) and F.linear will not mix
        # Half inputs with float weights.
        feats = feats.to(torch.float16).to(device=dev, dtype=dtype)
        mask = (feats.abs().sum(-1) > 0)
        mask[:, 0] = True
        emb = model.encode_text(feats, mask)             # (1, 512), L2-normed
    v = emb.float().cpu().numpy()[0]
    return _apply_modality_shift(v)


# ---- text->audio centroid correction -----------------------------------------
# Both towers L2-normalize, but CLIP-family models still park the two modalities
# in separate cones: on this checkpoint the audio and text centroids sit at
# cosine 0.64 (L2 gap 0.36) — nowhere near coincident. The DiT was trained ONLY
# on audio-tower vectors (train_modern_fancy_dit_measures.py reads the clap index
# for window bounds and caption logging, never to swap in a text embedding), so a
# raw text vector is off-manifold for the style channel.
#
# generate_clap_dataset.modality_stats saves mean_audio/mean_text for exactly
# this. The correction translates the text cone onto the audio cone:
#     v_audio ≈ normalize(v_text - mean_text + mean_audio)
# The centroids are means of unit vectors, so they are NOT unit length
# (|mean_audio| ≈ 0.35, |mean_text| ≈ 0.46) — that is expected, not a bug.
#
# TRAIN is the default: 4.49M measures vs val's 99k, and the two splits agree to
# ~1%, so it is the same estimate with less noise. Override with MODALITY_NPZ.
_MODALITY_NPZ = os.environ.get("MODALITY_NPZ",
                               "clap_nopre_clap_modality_train.npz")
_MODALITY = None


def _apply_modality_shift(v: np.ndarray) -> np.ndarray:
    global _MODALITY
    if _MODALITY is None:
        if _MODALITY_NPZ and os.path.exists(_MODALITY_NPZ):
            with np.load(_MODALITY_NPZ) as z:
                _MODALITY = (np.asarray(z["mean_audio"], np.float32),
                             np.asarray(z["mean_text"], np.float32))
                gap = float(z["centroid_l2_gap"])
                cos = float(z["centroid_cosine"])
            print(f"[preset_extractor] text->audio centroid shift enabled from "
                  f"{_MODALITY_NPZ} (centroid cosine {cos:.4f}, L2 gap {gap:.4f})")
        else:
            _MODALITY = False
            print(f"[preset_extractor] {_MODALITY_NPZ} not found; using raw CLAP "
                  f"text embeddings (they sit in the text cone, which the DiT "
                  f"never saw — expect weaker adherence than audio conditioning)")
    if _MODALITY is False:
        return v.astype(np.float32)
    mean_audio, mean_text = _MODALITY
    v = v - mean_text + mean_audio
    return (v / (np.linalg.norm(v) + 1e-9)).astype(np.float32)


# Real style embeddings are available (clap.py + style.pt wired in above).
STYLE_AVAILABLE = True
# Text conditioning additionally needs `transformers` + the T5 encoder; probed
# lazily so a missing dep degrades to "audio-only" rather than breaking startup.
def text_available() -> bool:
    try:
        import transformers  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


# =============================================================================
# SEAM 2: downbeats.  Paste your downbeat source here (beat_this parse, a beat
# file per preset, or librosa.beat).  Must return a sorted list of downbeat
# TIMES in seconds.
# =============================================================================
# Cache the beat_this model across calls (it's expensive to construct).
_BEAT_MODEL = None


def _get_downbeats(wav: np.ndarray, path: str, sr: int = SR) -> list[float]:
    """Return downbeat times (seconds), one per measure boundary.

    Uses the SAME source as training (annotate_beats.py): beat_this's
    File2Beats, which returns (beats, downbeats). We use `downbeats` directly —
    this matches generate_chroma_dataset.py, which kept indices where beat == 1
    (i.e. downbeats). Falls back to librosa beat tracking if beat_this isn't
    installed, so the module is still runnable in a lightweight environment.

    Precedence:
      1. a sidecar beat file at <path>.beats (TSV, beat_this format) if present
      2. beat_this File2Beats on the audio
      3. librosa.beat.beat_track (every TARGET_SIG-th beat) as a last resort
    """
    # 1. sidecar beat file (same TSV format save_beat_tsv writes)
    beat_file = path + ".beats"
    if os.path.exists(beat_file):
        return _parse_downbeats_tsv(beat_file)

    # 2. beat_this (training-accurate)
    try:
        global _BEAT_MODEL
        if _BEAT_MODEL is None:
            from beat_this.inference import Audio2Beats
            _BEAT_MODEL = Audio2Beats(checkpoint_path="final0",
                                      device=_aux_device(), dbn=False)
        beats, downbeats = _BEAT_MODEL(wav, sr)
        if len(downbeats) >= 2:
            return list(map(float, downbeats))
    except Exception as e:  # noqa: BLE001
        print(f"[preset_extractor] beat_this unavailable ({e}); "
              f"falling back to librosa beat tracking")

    # 3. librosa fallback
    import librosa
    _, beats = librosa.beat.beat_track(y=wav, sr=sr, hop_length=HOP_LENGTH,
                                       units="time")
    return list(beats[::TARGET_SIG])


def _parse_downbeats_tsv(beat_file: str) -> list[float]:
    """Parse a beat_this TSV (save_beat_tsv format): '<time>\\t<beat_number>'.
    Keep times where beat == 1 (downbeats), mirroring generate_chroma_dataset."""
    downs = []
    with open(beat_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    ts = float(parts[0])
                    bn = int(float(parts[1]))
                    if bn > 0:
                        bn = ((bn - 1) % 4) + 1   # 8/4 -> 4/4 fix, as in training
                    if bn == 1:
                        downs.append(ts)
                except ValueError:
                    continue
    return downs


# =============================================================================
# Core DSP — mirrors generate_chroma_dataset.py measure loop exactly
# =============================================================================
def _extract_measures(wav: np.ndarray, downbeat_times: list[float],
                      sr: int = SR):
    """Returns (measures_list, measure_bounds). Each measure is the 16-d vector
    [chroma(12), rms, density, zcr, flatness] plus a computed bpm; bounds are
    (frame_start, frame_end) for style extraction."""
    import librosa

    chroma = librosa.feature.chroma_cqt(y=wav, sr=sr, hop_length=HOP_LENGTH)
    rms = librosa.feature.rms(y=wav, hop_length=HOP_LENGTH)[0]
    onset_env = librosa.onset.onset_strength(y=wav, sr=sr, hop_length=HOP_LENGTH)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr,
                                              hop_length=HOP_LENGTH)
    zcr = librosa.feature.zero_crossing_rate(wav, hop_length=HOP_LENGTH)[0]
    flatness = librosa.feature.spectral_flatness(y=wav, n_fft=N_FFT,
                                                 hop_length=HOP_LENGTH)[0]

    rows = []
    times = []
    for i in range(len(downbeat_times) - 1):
        t_start, t_end = downbeat_times[i], downbeat_times[i + 1]
        f_start = librosa.time_to_frames(t_start, sr=sr, hop_length=HOP_LENGTH)
        f_end = librosa.time_to_frames(t_end, sr=sr, hop_length=HOP_LENGTH)
        if f_end > chroma.shape[1]:
            break
        f_end = min(f_end, chroma.shape[1])
        if f_end <= f_start:
            continue

        m_chroma = np.mean(chroma[:, f_start:f_end], axis=1)
        m_rms = float(np.mean(rms[f_start:f_end]))
        n_onsets = int(np.sum((onset_frames >= f_start) & (onset_frames < f_end)))
        dur = (f_end - f_start) / (sr / HOP_LENGTH)
        m_density = n_onsets / dur if dur > 0 else 0.0
        m_zcr = float(np.mean(zcr[f_start:f_end]))
        m_flat = float(np.mean(flatness[f_start:f_end]))
        # bpm from measure duration: TARGET_SIG beats per measure
        m_bpm = (60.0 * TARGET_SIG / dur) if dur > 0 else 120.0

        rows.append((m_chroma, m_rms, m_density, m_zcr, m_flat, m_bpm))
        times.append((t_start, t_end))

    return rows, times


def _fit_to_n_chunks(arr: np.ndarray, n: int = N_CHUNKS) -> np.ndarray:
    """Trim to first n rows, or tile to reach n if the song is too short."""
    if arr.shape[0] == 0:
        raise ValueError("no measures extracted — check downbeats/audio length")
    if arr.shape[0] >= n:
        return arr[:n]
    reps = int(np.ceil(n / arr.shape[0]))
    return np.concatenate([arr] * reps, axis=0)[:n]


def analyze_measures(wav: np.ndarray, downbeat_times, sr: int = SR,
                     n_chunks: int = N_CHUNKS, with_style: bool = True,
                     label: str = "audio") -> dict:
    """Measure every conditioning signal of `wav` given its measure boundaries.

    `downbeat_times` are seconds, one more entry than measures (measure i spans
    [t[i], t[i+1])). Returns a dict keyed exactly like Conditioning's fields,
    every array already fitted to `n_chunks` rows. `style` is omitted when
    with_style is False (the CLAP tower is 1.5 GB; the caller may not need it).

    Split out of extract_conditioning so the SAME DSP can be run over a
    GENERATED render, whose downbeats are known exactly from the bpm curve and
    therefore don't need (or want) the beat tracker."""
    rows, times = _extract_measures(wav, list(downbeat_times), sr=sr)
    if not rows:
        raise ValueError(f"{label}: no measures extracted")

    out = {
        "chroma": _fit_to_n_chunks(
            np.stack([r[0] for r in rows], axis=0).astype(np.float32), n_chunks),
    }
    for j, name in enumerate(("rms", "density", "zcr", "flatness", "bpm"),
                             start=1):
        col = np.array([r[j] for r in rows], dtype=np.float32)[:, None]
        out[name] = _fit_to_n_chunks(col, n_chunks)[:, 0]
    if with_style:
        out["style"] = _fit_to_n_chunks(
            _extract_style(wav, times, sr=sr).astype(np.float32), n_chunks)
    return out


def extract_conditioning(path: str, sr: int = SR,
                         n_chunks: int = N_CHUNKS) -> Conditioning:
    """Extract a full N_CHUNKS-measure Conditioning from an audio file."""
    import librosa
    wav, _ = librosa.load(path, sr=sr)

    downbeats = _get_downbeats(wav, path, sr=sr)
    if len(downbeats) < 2:
        raise ValueError(f"{path}: fewer than 2 downbeats found")

    sig = analyze_measures(wav, downbeats, sr=sr, n_chunks=n_chunks,
                           with_style=True, label=path)
    return Conditioning(**sig)


def build_preset_cond(name_to_path: dict[str, str],
                      n_chunks: int = N_CHUNKS) -> dict[str, Conditioning]:
    """Build the {preset_name: Conditioning} dict app_hf.py feeds RealBackend."""
    out = {}
    for name, path in name_to_path.items():
        if not os.path.exists(path):
            print(f"[preset_extractor] skip {name!r}: {path} not found")
            continue
        try:
            out[name] = extract_conditioning(path, n_chunks=n_chunks)
            print(f"[preset_extractor] built {name!r} from {os.path.basename(path)}")
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"[preset_extractor] FAILED {name!r}: {type(e).__name__}: {e}")
            traceback.print_exc()
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python preset_extractor.py <audio.wav> [more.wav ...]")
        raise SystemExit(1)
    for p in sys.argv[1:]:
        c = extract_conditioning(p)
        print(f"{p}: T={c.T} rms[:3]={c.rms[:3]} bpm[:3]={c.bpm[:3]} "
              f"chroma{c.chroma.shape} style{c.style.shape} "
              f"(STYLE_AVAILABLE={STYLE_AVAILABLE})")
