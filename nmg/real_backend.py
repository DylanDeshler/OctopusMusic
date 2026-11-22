"""
real_backend.py
===============

`RealBackend`: drives the actual trained model behind the same `ModelBackend`
interface the Gradio app already uses. Drop-in replacement for `MockBackend`:

    from real_backend import RealBackend
    from app import build_app
    build_app(RealBackend(...)).launch()

It has NO effect on the mock demo — `app.py` / `backend.py` don't import this
file. It only runs where PyTorch + a GPU + the model code are available.

--------------------------------------------------------------------------------
Mapping to your training pipeline (train_modern_fancy_dit_measures.py)
--------------------------------------------------------------------------------
This mirrors `save_samples` / `predict_measures` / `decode_latents` exactly:

  * Model      : diffusion_forcing.MetaConditionalModernDiTV2_smedium, loaded
                 from the EMA weights in model.pt (checkpoint['ema']), model_args
                 taken from the checkpoint. Wrapper exposes `.net` and
                 `.generate(...)`.
  * Tokenizer  : dito.DiToV5, loaded from tokenizer.pt (checkpoint['model']).
  * Adapter    : adapter.InvertibleAdapter, loaded from its own ckpt. REQUIRED
                 for decode (predict_measures does adapter.decode before
                 tokenizer.decode). Pass its checkpoint path in.
  * Normalization happens INSIDE the model forward (rms/density/zcr/flatness/
                 chroma are z-scored against registered buffers), so we feed the
                 UI's raw signal values straight through — matching the demo's
                 "write signals verbatim" contract. bpm is embedded via
                 nn.Embedding(350) after round+clamp, also inside forward.

Latent geometry (the key simplification for inpainting)
  Generation shape is x.shape == (B, T, N, C) with T == n_chunks == measures and
  N == spatial_window. So **one measure == one latent row along T**. Region
  inpainting over measures [s, e) is just holding rows [0:s) and [e:T) fixed and
  resampling [s:e) — no sub-measure ratio bookkeeping.

Conditioning tensors expected by the model forward (see forward signature:
  forward(x, t, bpm, rms, density, zcr, flatness, chroma, style, ...)):
    bpm, rms, density, zcr, flatness : (B, T)
    chroma                            : (B, T, 12)
    style                             : (B, T, style_dim)

Sampler parity with save_samples (checked line by line against the train script)
  Matched: EMA for the DiT and raw 'model' for tokenizer/adapter; bf16 autocast
  around generate and decode; n_steps=32 for BOTH samplers (SAMPLE_STEPS below);
  t_dist='logit'; cfg_mode='joint'; guidance 3.0; rescale_phi=0;
  memory_efficient=False; the (B,T,1) unconditional_mask shapes, including
  chroma's; gen_shape (B, T, spatial_window, in_channels); target_samples /
  max_len / the tokenizer decode length, and the 20 ms crossfade.

  Deliberately different:
    - B=1 (one track) instead of the val batch of 10. That is what forced the
      one real divergence below: predict_measures sizes the adapter's decode by
      the batch's slowest measure, which only looks like a general rule because
      its batch of 10 songs always makes it big enough. See _decode.
    - the adapter decodes at its full query length (max_seq_len-1) rather than at
      max_latent_len, then the result is sliced back. Not cosmetic -- at one
      track's worth of fast measures the unsliced version loses ~45% of the
      latent. _decode has the measurements.
    - decoder noise is drawn fresh instead of being a fixed tensor shared across
      variants. save_samples fixes it only so its five variants are comparable
      to each other; the draw is the same distribution.
    - the unconditional_mask is built per measure from the UI's `active` flags,
      where save_samples uses all-on (its `cfg`) or a whole signal off (its
      `text_only`). Those are both reachable as special cases; per-measure
      coverage is in-distribution because training dropped signals per token.
    - guidance is passed as a 1-element list, not [3]*7. In joint mode
      get_prediction reads guidance_list[0] and ignores the rest.
    - after decoding, the preset DSP is re-run over the render and written back
      into every measure whose signal was OFF (see _measure_render). Purely a UI
      affordance -- it happens after the last tensor op and cannot affect the
      audio -- so that the curves the user edits from describe the take they
      just heard rather than a placeholder. save_samples has no editing loop and
      needs none of this.

What you still need present on the machine that runs this:
    diffusion_forcing.py, dito.py, adapter.py, and dito's imports
    (seanet.py, consistency_decoder.py, fm.py); model.pt, tokenizer.pt, and the
    adapter checkpoint. Presets need real conditioning arrays (see PRESET_COND).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from backend import (
    ModelBackend, Track, Conditioning,
    ALL_SIGNALS, CONTINUE_CTX_NOISE, CONTINUE_MEASURES, EDITABLE_SIGNALS,
    REGION_CTX_NOISE, REGEN_STRENGTH,
    CROSSFADE_MS, CROSSFADE_SAMPLES,
    FF_MAX_MEASURES, HG_MODE, HISTORY_GUIDANCE, HISTORY_GUIDANCE_FREQ,
    N_CHUNKS, STREAM_LEG_MEASURES,
    STYLE_DIM, TARGET_SIG, SR,
    apply_cond_hook, apply_loop_escape_style, apply_style_steering,
    arm_idle_stream,
    bank_bounds, bank_sync, bpm_curve,
    bpm_grid_error, snap_bpm, ensure_tail,
    eval_shape, fold_bpm, key_template, loop_escape, note_loop_check,
    plan_fast_forward, record_style_trail,
    resolve_axis_prompts, shift_window, slerp, unit,
)

# Torch is only imported lazily inside methods/__init__ so this module can be
# imported (for reading/inspection) even where torch isn't installed.

# Sampler steps. train_modern_fancy_dit_measures.save_samples hardcodes
# `n_steps = 32` and passes it to predict_measures, which spends it TWICE: once
# on the DiT's FMEulerSampler and again on tokenizer.decode (also an
# FMEulerSampler). The training-run wavs are the reference for how this model is
# supposed to sound, so this is the operating point to reproduce, not a knob to
# turn up. app.py's draft toggle halves it; nothing should raise it.
SAMPLE_STEPS = 32


def _crossfade_segments(segment_a, segment_b, sample_rate, crossfade_ms=15):
    """Linear crossfade of two 1D numpy audio arrays (prevents boundary clicks).
    Inlined from train_modern_fancy_dit_measures.crossfade_segments so this
    module does not import the train script (which pulls in fad/pyrubberband)."""
    crossfade_samples = int(sample_rate * (crossfade_ms / 1000.0))
    if len(segment_a) < crossfade_samples or len(segment_b) < crossfade_samples:
        return np.concatenate((segment_a, segment_b))
    fade_out = np.linspace(1.0, 0.0, crossfade_samples)
    fade_in = np.linspace(0.0, 1.0, crossfade_samples)
    overlap_a = segment_a[-crossfade_samples:] * fade_out
    overlap_b = segment_b[:crossfade_samples] * fade_in
    mixed_overlap = overlap_a + overlap_b
    return np.concatenate((
        segment_a[:-crossfade_samples],
        mixed_overlap,
        segment_b[crossfade_samples:],
    ))


# =============================================================================
# Masked sampler, DIFFUSION-FORCING style. Standalone so diffusion_forcing.py
# stays untouched; uses the same FM diffusion + get_prediction/CFG machinery.
# =============================================================================
def fm_df_sample(
    diffusion,          # FM instance (net wrapper's .diffusion)
    net,                # the raw DiT (wrapper.net)
    known_x0,           # (B, T, N, C) clean latents to preserve where kept
    keep_mask,          # (B, T, 1, 1) bool: True where latents are FROZEN (kept)
    n_steps,
    net_kwargs=None,
    uncond_net_kwargs=None,
    guidance=1.0,
    noise=None,
    memory_efficient=False,
    rescale_phi=0.0,
    cfg_mode="independent",
    t_dist="uniform",
    ctx_noise=0.0,          # noise level to hold the context at; see below
    history_guidance=1.0,   # omega on the history axis; 1.0 == off
    hg_freq_scale=None,     # HG-f: level the AMPLIFIED leg reads history at
    hg_mode="nested",       # "nested" (ours) or "dfot" (flat weight table)
    joint_net_kwargs=None,  # the all-signals conditional; dfot mode needs it
    strength=1.0,           # where the FREE measures start; 1.0 == pure noise
):
    """Masked FM sampling with a PER-MEASURE noise level.

    This model is a diffusion-forcing model, and that is the whole point of it.
    FM.loss draws

        t = torch.sigmoid(torch.randn(B, T))        # (B, T) -- one per measure

    so every measure was trained at its OWN independently-sampled noise level,
    and MetaConditionalModernDiTV2 modulates per measure to match (`t = t +
    style + bpm`, then `t[:, :, None]` broadcast across that measure's latent
    positions). Attention is fully bidirectional, so a clean measure and a
    pure-noise measure can sit in the same sequence and the clean one is read as
    clean. Conditioning on the past is therefore NATIVE: hand the context
    measures over at t~0 with their real latents and denoise only the rest.

    That is what this does. The context is pinned to `known_x0` UNCHANGED at
    every Euler step and reported at noise level `ctx_noise`, while the free
    measures walk the schedule from t=1 down to 0.

    Why not RePaint (which is what this function used to do): RePaint exists
    because ordinary diffusion models accept a single scalar t, so the only way
    to express "this part is known" is to forward-noise the known part to the
    same level as the part being generated and splice it in. Here that is both
    unnecessary and actively destructive -- at the first step (t=1) the context
    gets noised to t=1, i.e. handed over as pure noise, so the model has no past
    to condition on exactly when the global structure is being decided. It also
    redraws that noise every step, so what context does survive flickers. Uniform
    t is in-distribution for training (all-equal is one draw of (B,T) among
    many), but it throws away the one capability that makes rollout work.

    ctx_noise: the level the context is held at, and it is held there for the
    whole denoise -- the context is something the model READS, not something
    being denoised, so there is nothing for it to decay toward. (What is graded
    in DF's pyramid schedules is position among the tokens being GENERATED, not
    the history.) 0.0 is the default and matches the DF reference implementation;
    see CONTINUE_CTX_NOISE in backend.py for when to raise it and why the useful
    range starts around 0.1.

    At 0.0 the kept LATENTS come back bit-exact. The decoded waveform still
    won't be, because _decode draws fresh decoder noise every call, so a
    re-decoded context measure differs slightly from its previous decode --
    that is what the 20 ms crossfade at the prefix join absorbs.

    strength: WHERE THE FREE MEASURES START. This is the partial-noise restart
    (SDEdit, in flow-matching form), and it is the other half of the answer to
    "rewrite these measures and have them still belong here". ctx_noise is about
    what the model READS; this is about what it BEGINS FROM.

    At 1.0 the free measures begin at pure noise and the schedule runs the whole
    way from t=1, which is what this function has always done. Below 1.0 they
    begin at their OWN existing latents, noised to t=s:

        x_s = alpha(s) * known_x0 + sigma(s) * noise

    and the schedule is truncated to its tail at or below s. The identity of the
    passage is then carried by the INITIALISATION rather than by the surrounding
    context, which matters because context is the thing in short supply: a wide
    selection leaves few frozen neighbours, and at the ends of a song it leaves
    them on one side only. Start from the take you have and neither shortage
    bites.

    Both ends are exact rather than approximate, and nothing special-cases them:

      s = 1.0   alpha(1) = 0 and sigma(1) = 1 exactly, so the initialisation is
                `0*known_x0 + 1*noise` == noise; and t_steps[0] is already 1.0,
                so the truncation keeps every step and rewrites t_steps[0] with
                the value it already held. Bit-for-bit the old behaviour.
      s = 0.0   alpha(0) = 1, sigma(0) = 1e-5, and the truncated schedule has no
                steps left to take, so this returns the existing latents (plus
                1e-5 of noise, which is nothing). "Change nothing" is reachable.

    Fewer steps survive as s falls, which is deliberate and is what img2img
    does: the step SIZE stays what the schedule says, so per-step Euler error is
    unchanged and a small edit is proportionally cheaper. It cannot underflow --
    the truncation index is at most n_steps, i.e. at worst zero steps and the
    original back.

    Why this is in distribution here, and only awkwardly so elsewhere. Plain
    SDEdit noises everything to t=s and asks a single-scalar-t model to denoise
    the lot; the context has to be re-invented along with the edit. This model
    draws t = sigmoid(randn(B, T)) in training -- one INDEPENDENT level per
    measure -- so holding the guards at ctx_noise while the middle descends from
    s is an ordinary training draw, not a configuration it has to tolerate.

    history_guidance: omega for VANILLA HISTORY GUIDANCE (HG-v, Song & Chen et
    al., "History-Guided Video Diffusion"). Ordinary CFG steers toward the
    conditioning signals. This steers toward THE PAST, composed with the signal
    axis in the usual nested way -- signal CFG relative to the with-history
    base, history relative to the bare unconditional:

        u_n + omega * (u_h - u_n) + sum_i g * (c_i^h - u_h)
      = v_history + (omega - 1) * (u_h - u_n)

    where u_h / u_n are the unconditional predictions with and without history.
    omega=1.0 cancels the correction exactly, so it is free to leave at 1. Do
    NOT implement this as `v_none + omega*(v_h - v_none)` with both sides fully
    CFG'd -- see the note at the call site for why that overdrives by (1-n*g).

    The trick that makes this cheap on a diffusion-forcing model is how you
    express "no history". You do not need a null token or a mask embedding: a
    measure at pure noise is, by construction, a measure the model knows nothing
    about. So the negative branch puts the context at t=1.0 -- PINNED THERE FOR
    THE WHOLE TRAJECTORY, not tracking the schedule -- and the difference between
    the two branches is then precisely "what knowing the past buys you", with
    omega scaling it.

    THE PIN IS LOAD-BEARING AND THIS FUNCTION USED TO GET IT WRONG. The negative
    branch used to sit at the current schedule level t_cur, on the reasoning that
    a measure at the same level as the ones being generated is one the model
    cannot read. That reasoning holds at high t and fails at low t, because t_cur
    descends through ctx_noise and out the bottom. Once t_cur < ctx_noise the
    "no history" leg is holding a strictly CLEANER context than the "with
    history" leg, so (u_h - u_n) has inverted and omega > 1 is steering AWAY from
    the past -- at least the last ~20% of steps at the 0.2 default, and in
    practice earlier, since the declared level shifts the balance before the
    content does. The term decays across the whole descent rather than holding
    strength, so omega never meant what the number said. The reference
    implementation has no such decay: DFoT's HistoryGuidanceManager seeds its
    weight table with `hist_to_weights[(1.0,)*hist_len + (...)] = 1.0` and
    comments it `# unconditional score` -- noise level 1, fixed, for every step.
    Pinning also means the negative context is a constant, so it is drawn once
    outside the loop rather than resampled per step.

    (Amusingly, a negative branch of this shape is what this function used to
    compute and return as the answer, back when it was RePaint.)

    This matters most when the new measures are UNDERSPECIFIED. Continuation
    frees chroma/rms/density/zcr/flatness on the new slots -- nobody knows what
    an unwritten bar contains -- so the frozen context is nearly the only thing
    determining them, and it is worth making the model lean on it. DFoT reports
    best FVD at a small omega=1.5; 3.0 was visibly over-driven there. Their own
    200-frame rollout recipe runs omega=4.0, so the useful range is wider than
    the FVD number suggests once the task is length rather than fidelity.

    hg_freq_scale: FRACTIONAL history guidance (HG-f), the paper's "advanced
    variant" -- the one its abstract credits with motion dynamics and extended
    rollout, as against vanilla's quality and temporal consistency. None leaves
    this at plain HG-v.

    HG-v amplifies the direction from "no history" to "the history exactly as
    the model is reading it", which is a direction that says COPY. HG-f amplifies
    the direction from "no history" to "the history at an INTERMEDIATE noise
    level" instead, so what gets boosted is the coarse content of the past and
    not its detail. Here that is the difference between "stay in this groove,
    register and harmony" and "play those notes again" -- which is the whole
    argument for expecting it to help a rollout that collapses into a riff.

    In the reference this is a three-segment composition, `hist_segments=[full(),
    partial_constant(freq_scale, 1.0)]` at weights `[1, w-1]`, resolving to

        s(x | hist@0, cond) + (w-1) * [ s(x | hist@freq_scale, cond) - s(x | 0) ]

    Because we factor the signal axis out separately (see the nested-composition
    note at the call site below), our base term v already plays the role of their
    weight-1 `s(x | hist@0, cond)`, and the only thing HG-f changes here is which
    level the POSITIVE leg of the history difference reads the context at:

        v + (omega - 1) * (u_f - u_n)        u_f = uncond, history at freq_scale
                                             u_n = uncond, history at 1.0

    so it costs nothing extra -- freq_scale == ctx_noise is exactly HG-v again,
    and is reused bit-for-bit rather than recomputed when it is.

    Cost: two extra unconditional passes per step whenever omega != 1.0 (decode
    unchanged) -- 4 DiT passes instead of 2 in the shipped joint cfg_mode, or 10
    instead of 8 under CFG_MODE=independent at the usual 7 signals.

    hg_mode: WHICH COMPOSITION. "nested" is everything described above -- signal
    CFG at `guidance` measured against the with-history base, history guidance at
    `omega` measured against the bare unconditional, one scale each, applied in
    sequence. "dfot" is the reference implementation's composition instead, and
    the two are genuinely different formulations rather than different tunings.

    DFoT has ONE guidance axis. HistoryGuidanceManager.__enter__ builds a flat
    table keyed by (history noise level per token, whether external conditioning
    is masked) -> weight, seeds it with `hist_to_weights[(1.0,)*hist_len +
    (use_external_cond_guidance,)] = 1.0`, then `+= weight` on each segment's
    start key (conditioning KEPT) and `-= weight` on its end key (conditioning
    masked, since use_external_cond_guidance defaults True for HG-v and HG-f).
    The final velocity is the weighted sum of one net call per surviving key, and
    the weights sum to 1. History and external conditioning are therefore dropped
    TOGETHER, on the same leg, under the same scale -- there is no leg holding the
    history without the conditioning. Their omega=4 is doing the work our
    `guidance` and `omega` do separately, which is why their published scales do
    not transfer to our 1.5.

    Ported here with one substitution: their clean leg reads history at 0.0 and
    ours reads it at ctx_noise, because our floor exists for a reason that has not
    changed (see ctx_noise above; their nearest equivalent, stabilization_level,
    is 0.02 and applies only to already-generated history). Everything else is
    theirs. The tables, merged:

        vanilla     [(ctx_noise, cond, omega), (1.0, no cond, 1 - omega)]
        fractional  [(ctx_noise, cond, 1), (freq, cond, omega - 1),
                     (1.0, no cond, 1 - omega)]

    Consequences worth knowing before reading an A/B: `guidance` STOPS DOING
    ANYTHING in this mode -- omega is the only scale, and at omega=1.0 the whole
    thing collapses to a single plain conditional pass with no CFG at all.
    rescale_phi is ignored; the reference has no equivalent. Cost is 2 passes for
    vanilla and 3 for fractional, against nested's 4, so it is also cheaper.

    Returns x0-space latents (B, T, N, C)."""
    import torch

    dev = next(net.parameters()).device
    B, T, N, C = known_x0.shape
    x_t = torch.randn_like(known_x0) if noise is None else noise

    if t_dist == "uniform":
        t_steps = torch.linspace(1, 0, n_steps + 1, device=dev)
    else:  # 'logit'
        u = torch.linspace(1.0 - 1e-5, 1e-5, n_steps + 1, device=dev)
        z = math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)
        t_steps = torch.sigmoid(z)
        t_steps[0] = 1.0
        t_steps[-1] = 0.0

    # Partial-noise restart: keep the TAIL of whichever schedule was chosen
    # rather than building a new one between s and 0. The shape of the descent
    # is a property of t_dist -- the logit schedule spends its steps where the
    # model's training draws put their mass -- and re-deriving it over a shorter
    # interval would change the step sizes as well as their number. Taking the
    # suffix keeps every surviving step exactly where it was and drops only the
    # ones above s. See `strength` in the docstring for why the two ends come
    # out exact.
    strength = float(strength)
    i0 = int(torch.nonzero(t_steps <= strength)[0])
    t_steps = t_steps[i0:].clone()
    t_steps[0] = strength
    n_steps = len(t_steps) - 1

    keep = keep_mask.to(dev)                       # (B, T, 1, 1)
    keep_row = keep[:, :, 0, 0]                    # (B, T) -- for the t vector
    ctx_noise = float(ctx_noise)

    def _hist_at(level):
        """The context held at noise level `level`. Drawn ONCE, not per step:
        the context is a fixed thing the model is reading, not a quantity being
        resampled. At 0.0 this is exactly known_x0 (alpha(0)=1, sigma(0)=1e-5);
        at 1.0 it is pure noise (alpha(1)=0, sigma(1)=1)."""
        if level <= 0.0:
            return known_x0
        tt = torch.full((B, T, N, 1), level, device=dev, dtype=known_x0.dtype)
        return (diffusion.alpha(tt) * known_x0
                + diffusion.sigma(tt) * torch.randn_like(known_x0))

    ctx_x = _hist_at(ctx_noise)

    # Levels are drawn once and shared, so two legs asking for the same level get
    # the SAME realisation -- otherwise the difference between them would carry a
    # noise delta that has nothing to do with history.
    _hist_cache = {ctx_noise: ctx_x}

    def _hist_for(level):
        if level not in _hist_cache:
            _hist_cache[level] = _hist_at(level)
        return _hist_cache[level]

    omega = float(history_guidance)
    hg_on = abs(omega - 1.0) > 1e-6 and bool(keep_row.any())

    # HG-f's intermediate level, and the two extra history tensors the guidance
    # legs read. Both are step-independent now that the negative branch is
    # pinned, so both are hoisted out of the loop. `hg_f_on` false means
    # freq_scale lands on ctx_noise, i.e. plain HG-v, and the positive leg reuses
    # x_t/t_hist unchanged so that path stays bit-exact.
    freq = ctx_noise if hg_freq_scale is None else float(hg_freq_scale)
    hg_f_on = hg_on and abs(freq - ctx_noise) > 1e-9
    hist_f = _hist_for(freq) if hg_f_on else None
    hist_none = _hist_for(1.0) if hg_on else None

    # DFoT's merged weight table, as (history level, keep conditioning, weight).
    # Zero-weight rows are dropped rather than evaluated: at omega == 1.0 this
    # leaves a single plain conditional leg, which is what their `conditional`
    # scheme is. See hg_mode in the docstring.
    dfot_on = hg_mode == "dfot" and bool(keep_row.any())
    if dfot_on:
        if hg_f_on:
            legs = [(ctx_noise, True, 1.0), (freq, True, omega - 1.0),
                    (1.0, False, 1.0 - omega)]
        else:
            legs = [(ctx_noise, True, omega), (1.0, False, 1.0 - omega)]
        legs = [(lv, kc, w) for lv, kc, w in legs if abs(w) > 1e-9]
        for lv, _, _ in legs:
            _hist_for(lv)
        joint_kwargs = (net_kwargs[0] if joint_net_kwargs is None
                        else joint_net_kwargs)

    def _predict(x, t_vec):
        return diffusion.get_prediction(
            net, x, t_vec,
            net_kwargs=net_kwargs,
            uncond_net_kwargs=uncond_net_kwargs,
            guidance=guidance,
            memory_efficient=memory_efficient,
            rescale_phi=rescale_phi,
            cfg_mode=cfg_mode,
        )

    def _uncond(x, t_vec):
        """The bare unconditional leg -- every signal nulled. guidance=1.0 with a
        single dict makes get_prediction's is_cfg False, so this is one pass."""
        return diffusion.get_prediction(
            net, x, t_vec,
            net_kwargs=uncond_net_kwargs,
            uncond_net_kwargs=uncond_net_kwargs,
            guidance=1.0,
            memory_efficient=memory_efficient,
            rescale_phi=0.0,
            cfg_mode=cfg_mode,
        )

    def _leg(level, keep_cond, t_cur_c, x_t):
        """One row of the DFoT weight table: the history at `level`, with the
        external conditioning either kept whole or dropped whole. Always a single
        pass -- guidance=1.0 with one dict -- because in this composition the
        scale lives in the table's weights, not inside get_prediction."""
        x = torch.where(keep, _hist_for(level), x_t)
        tv = torch.where(keep_row,
                         torch.full_like(keep_row, level, dtype=x_t.dtype),
                         t_cur_c)
        return diffusion.get_prediction(
            net, x, tv,
            net_kwargs=joint_kwargs if keep_cond else uncond_net_kwargs,
            uncond_net_kwargs=uncond_net_kwargs,
            guidance=1.0,
            memory_efficient=memory_efficient,
            rescale_phi=0.0,
            cfg_mode=cfg_mode,
        )

    with torch.no_grad():
        # The free measures start from their own existing latents, noised to
        # `strength`, instead of from pure noise. At strength 1.0 alpha is 0 and
        # sigma is 1, so this is `x_t` unchanged -- no branch, and no way for the
        # old path to drift. The kept measures are overwritten on the next line
        # regardless, so this is written over the whole tensor.
        x_t = (diffusion.alpha(strength) * known_x0
               + diffusion.sigma(strength) * x_t)
        x_t = torch.where(keep, ctx_x, x_t)
        for i in range(n_steps):
            t_cur, t_next = t_steps[i], t_steps[i + 1]
            t_cur_c = t_cur.to(x_t.dtype)

            # per-measure noise level: context sits at ctx_noise, free measures
            # at wherever the schedule currently is.
            t_hist = torch.where(
                keep_row,
                torch.full_like(keep_row, ctx_noise, dtype=x_t.dtype),
                t_cur_c)
            if dfot_on:
                # One weighted sum over the merged table and nothing else: no
                # nesting, no separate signal axis, `guidance` unused. The weights
                # already sum to 1 so this is a convex-style extrapolation in the
                # same sense CFG is, just with the conditioning and the history
                # riding the single omega together. See hg_mode in the docstring.
                v = sum(w * _leg(lv, kc, t_cur_c, x_t) for lv, kc, w in legs)
            else:
                v = _predict(x_t, t_hist)

            if hg_on and not dfot_on:
                # Negative branch: the context at t=1.0, which is how a
                # diffusion-forcing model spells "I have no history here". Only
                # the HISTORY slots differ from the positive branch -- the free
                # measures are the tokens being generated and stay where the
                # schedule has them. Both the declared level and the content
                # have to move together or the model is handed an off-manifold
                # token. See the docstring for why this is pinned rather than
                # tracking t_cur, and what the old version did below ctx_noise.
                x_none = torch.where(keep, hist_none, x_t)
                t_none = torch.where(
                    keep_row,
                    torch.full_like(keep_row, 1.0, dtype=x_t.dtype),
                    t_cur_c)

                # Positive leg of the history difference. HG-v reads the context
                # exactly as the base prediction did; HG-f reads it blurrier.
                if hg_f_on:
                    x_pos = torch.where(keep, hist_f, x_t)
                    t_pos = torch.where(
                        keep_row,
                        torch.full_like(keep_row, freq, dtype=x_t.dtype),
                        t_cur_c)
                else:
                    x_pos, t_pos = x_t, t_hist

                # COMPOSE THE TWO GUIDANCE AXES NESTED, NOT BY DIFFERENCING TWO
                # FULLY-GUIDED VELOCITIES. get_prediction's independent mode
                # returns  u + sum_i g*(c_i - u), so the uncond leg carries
                # coefficient (1 - n*g). Extrapolating v_h against a v_none built
                # the same way gives
                #     v_h - v_none = (u_h - u_n)(1 - n*g) + g*sum_i(c_i^h - c_i^n)
                # so with n=7 signals at g=3 the history term arrives at
                # (omega-1)*(1-21) = -10x instead of +0.5x -- twenty times too
                # strong and sign-flipped. That inflates the velocity on exactly
                # the free measures (the frozen ones are pinned), which is heard
                # as the new bars coming back at absurd volume.
                #
                # The right composition is the standard nested one: signal CFG
                # measured against the WITH-history base, history guidance
                # measured against the bare unconditional.
                #     u_n + omega*(u_h - u_n) + sum_i g*(c_i^h - u_h)
                #   = v_h + (omega - 1)*(u_h - u_n)
                # Each axis now enters at its own scale. It is also cheaper:
                # two extra passes flat, rather than doubling all n_sig+1.
                v = v + (omega - 1.0) * (_uncond(x_pos, t_pos)
                                         - _uncond(x_none, t_none))

            # (t_cur - t_next), NOT (t_next - t_cur). The net predicts -dx_t/dt
            # (FM.loss's target is A(t)x + B(t)noise == x - (1-sigma_min)noise,
            # commented "-dxt/dt"), and t runs 1 -> 0, so the descending step
            # dt = t_next - t_cur is negative and the two negatives cancel:
            #     dx = (dx_t/dt) * dt = (-v) * (t_next - t_cur) = v * (t_cur - t_next)
            # This is exactly what FMEulerSampler.sample does. Getting it
            # backwards integrates the flow the wrong way and returns noise.
            x_t = x_t + v * (t_cur - t_next)
            # the Euler step moved the context too; put it back untouched.
            x_t = torch.where(keep, ctx_x, x_t)

    # Hand back the CLEAN context, not ctx_x. ctx_x is an input the model reads;
    # it is not an answer. The caller stores this whole tensor as tr.latents and
    # the next continuation slices its context out of it, so returning the noised
    # copy would re-noise the same measures on every press: after k continues the
    # oldest surviving measure has been scaled by alpha(ctx_noise)^k with k
    # independent noise draws stacked on it (0.95^k at the 0.05 default -- 0.77
    # by the fifth press) and has drifted well off the latent manifold. That
    # decays into mush no matter how good the sampler is. It also means the
    # re-decoded context measures are the real ones rather than degraded ones.
    return torch.where(keep, known_x0, x_t)


def _check_stage(m, stage, path, model_args):
    """Shout if the weights we just loaded aren't the stage we asked for.

    Stage 2 is the only DiT that hears chroma/rms/density/zcr/flatness: the
    forward adds them as `x = x + local_embedder(c)`. Stage-1 checkpoints ship
    that block at its ControlNet identity init -- every tensor in it is exactly
    zero -- and stage 1 freezes it (requires_grad=False), so it is still exactly
    zero at the end of a 300k-step run. Load stage-1 weights under stage=2 and
    set_training_stage(2) happily runs the block, which adds 0. Generation is
    then bit-identical to stage 1: style + bpm only, every per-measure DSP signal
    inert. Nothing raises, nothing looks wrong, the audio is just quietly much
    worse than the training run's `{i}_cfg.wav`.

    That is not hypothetical -- it is what a missing stage2.pt in the model repo
    plus a STAGE2_CKPT=stage1.pt fallback produces. Cheap to detect, so detect it.
    """
    import torch
    declared = model_args.get("stage")
    if declared is not None and int(declared) != stage:
        print(f"[real_backend] WARNING: {path} was trained as stage {declared} "
              f"but is registered as stage {stage}.")
    if stage != 2:
        return
    with torch.no_grad():
        mag = sum(float(p.abs().sum()) for p in m.net.local_embedder.parameters())
    if mag == 0.0:
        print("=" * 72)
        print(f"[real_backend] WARNING: stage 2 loaded from {path}, but its "
              f"local_embedder is all zeros.")
        print("[real_backend] These are stage-1 weights. chroma/rms/density/zcr/"
              "flatness will be IGNORED;")
        print("[real_backend] stage 2 will produce exactly the same audio as "
              "stage 1. Point STAGE2_CKPT at")
        print("[real_backend] a real stage-2 checkpoint.")
        print("=" * 72)


# =============================================================================
# RealBackend
# =============================================================================
class RealBackend(ModelBackend):
    def __init__(
        self,
        model_ckpt: Optional[str] = None,      # back-compat: stage-2 DiT
        tokenizer_ckpt: str = "tokenizer.pt",
        adapter_ckpt: str = "adapter.pt",
        model_ckpts: Optional[dict] = None,    # {stage:int -> ckpt path}; both DiTs
        default_stage: int = 2,
        device: Optional[str] = None,
        dtype: str = "bfloat16",
        use_ema: bool = True,
        preset_cond: Optional[dict] = None,   # {name: Conditioning}
        meta_model=None,                       # optional text->conditioning model
        default_cfg_mode: str = "joint",
        default_guidance: float = 3.0,
        default_t_dist: str = "logit",
        drop_style: bool = False,      # True => style always masked (ignored),
                                       # for when style embeddings aren't real
    ):
        import torch
        from diffusion_forcing import MetaConditionalModernDiTV2_smedium
        from dito import DiToV5
        from adapter import InvertibleAdapter

        self.torch = torch
        # ZeroGPU rule: DO NOT touch CUDA at construction time. Load everything
        # on CPU here; move to GPU lazily inside the @spaces.GPU-decorated call
        # (see to_gpu()). `device` stays None until then so no CUDA context is
        # created at import/startup.
        self._requested_device = device        # None -> auto ('cuda' if avail at call time)
        self.device = "cpu"                     # current device of the modules
        self.ptdtype = {"float32": torch.float32,
                        "bfloat16": torch.bfloat16,
                        "float16": torch.float16}[dtype]
        self.cfg_mode = default_cfg_mode
        self.guidance = default_guidance
        self.t_dist = default_t_dist
        self.meta_model = meta_model
        self.drop_style = drop_style
        self.default_stage = int(default_stage)

        def _strip(sd):
            pre = "_orig_mod."
            for k in list(sd.keys()):
                if k.startswith(pre):
                    sd[k[len(pre):]] = sd.pop(k)
            return sd

        def _dit_weights(ck, label):
            """DiT weights: prefer EMA when present (and use_ema), else raw.

            Mirrors training's sampling path, which generates from
            `ema.ema_model` — i.e. the EMA copy of the DiT."""
            if use_ema and "ema" in ck and ck["ema"] is not None:
                print(f"[real_backend] {label}: using EMA weights")
                return ck["ema"]
            print(f"[real_backend] {label}: using raw model weights "
                  f"({'no ema key' if 'ema' not in ck else 'use_ema=False'})")
            return ck["model"]

        def _raw_weights(ck, label):
            """Tokenizer / adapter weights: ALWAYS the raw 'model' state dict,
            NEVER EMA. This matches train_modern_fancy_dit_measures.py, which
            loads the tokenizer and adapter from `checkpoint['model']` and only
            ever uses EMA for the DiT. Decoding with an EMA copy of the tokenizer
            or adapter that training never used degrades audio quality even
            though the generated latents are identical — the bug this avoids."""
            if "ema" in ck and ck["ema"] is not None:
                print(f"[real_backend] {label}: using raw model weights "
                      f"(ignoring EMA — matches training decode path)")
            else:
                print(f"[real_backend] {label}: using raw model weights")
            return ck["model"]

        # ---- DiT(s) ---- loaded on CPU
        # We support one or two stage checkpoints. Stage 2 is the full model (all
        # 7 conditioning signals); stage 1 conditions on style + bpm only (better
        # for coarse / unconditional generation). Both share the SAME tokenizer
        # and adapter, so the only extra cost of loading both is one more DiT
        # (~2.4 GB) — comfortable on an A10G, and it makes the toggle instant.
        if model_ckpts is None:
            # back-compat: a single ckpt registered under the default stage.
            model_ckpts = {self.default_stage: model_ckpt or "model.pt"}
        self.models = {}          # {stage:int -> wrapper module}
        self.spatial_window = None
        self.vae_embed_dim = None
        for stage, path in sorted(model_ckpts.items()):
            stage = int(stage)
            ck = torch.load(path, map_location="cpu", weights_only=False)
            m = MetaConditionalModernDiTV2_smedium(**ck["model_args"])
            m.load_state_dict(_strip(dict(_dit_weights(ck, f"stage{stage} model"))))
            # Force the DiT into the requested inference stage: stage 1 skips the
            # local_embedder (DSP signals), stage 2 uses it. This is the switch
            # that changes which signals actually condition the model.
            m.net.set_training_stage(stage)
            m.eval()
            _check_stage(m, stage, path, ck["model_args"])
            self.models[stage] = m
            # geometry is identical across stages; capture from whichever loads.
            self.spatial_window = ck["model_args"].get("spatial_window", 64)
            self.vae_embed_dim = ck["model_args"].get("in_channels", 16)
            del ck
        if self.default_stage not in self.models:
            self.default_stage = min(self.models)
        # `self.model` / `self.net` / `self.diffusion` always point at the ACTIVE
        # stage's modules; _select_stage() re-points them. Start on the default.
        self.model = self.models[self.default_stage]
        self.net = self.model.net
        self.diffusion = self.model.diffusion

        # ---- tokenizer (DiToV5) ---- loaded on CPU
        tck = torch.load(tokenizer_ckpt, map_location="cpu", weights_only=False)
        self.tokenizer = DiToV5(**tck["model_args"])
        self.tokenizer.load_state_dict(_strip(dict(_raw_weights(tck, "tokenizer"))))
        self.tokenizer.eval()
        self.encoder_ratios = math.prod(self.tokenizer.encoder.ratios)
        del tck

        # ---- adapter (InvertibleAdapter) ---- loaded on CPU
        ack = torch.load(adapter_ckpt, map_location="cpu", weights_only=False)
        self.adapter = InvertibleAdapter(**ack["model_args"])
        self.adapter.load_state_dict(_strip(dict(_raw_weights(ack, "adapter"))))
        self.adapter.eval()
        self.max_adapter_len = self.adapter.max_seq_len
        del ack

        self.preset_cond = preset_cond or {}
        self.active_stage = self.default_stage
        self._on_gpu = False
        self._text_style_cache = {}

    # ---- stage selection ----------------------------------------------------
    def _select_stage(self, stage) -> int:
        """Point self.model/net/diffusion at the requested stage's DiT. Falls
        back to the default stage if the requested one wasn't loaded. Returns the
        stage actually selected."""
        stage = int(stage) if stage is not None else self.default_stage
        if stage not in self.models:
            stage = self.default_stage
        if stage != self.active_stage or self.model is not self.models[stage]:
            self.model = self.models[stage]
            self.net = self.model.net
            self.diffusion = self.model.diffusion
            self.active_stage = stage
        return stage

    # ---- device management (ZeroGPU-safe) -----------------------------------
    def to_gpu(self):
        """Move all modules onto CUDA. MUST be called inside the @spaces.GPU
        function (that's the only place ZeroGPU exposes a live CUDA device).
        Idempotent and safe to call every request."""
        torch = self.torch
        if self._requested_device is not None:
            dev = self._requested_device
        else:
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        if dev != self.device:
            for m in self.models.values():   # move BOTH stage DiTs
                m.to(dev)
            self.tokenizer.to(dev)
            self.adapter.to(dev)
            # re-point active refs (they moved with the module)
            self.net = self.model.net
            self.diffusion = self.model.diffusion
            self.device = dev
            self._on_gpu = "cuda" in dev
        return self

    # ---- amp context helper -------------------------------------------------
    def _ctx(self):
        import torch
        if "cuda" in self.device:
            return torch.amp.autocast(device_type="cuda", dtype=self.ptdtype)
        from contextlib import nullcontext
        return nullcontext()

    # =====================================================================
    # entry points
    # =====================================================================
    def list_presets(self) -> list[str]:
        return list(self.preset_cond.keys())

    def seed_from_preset(self, name: str, seed: int) -> Track:
        cond = self.preset_cond[name].copy()
        return Track(cond=cond, seed=int(seed), n_chunks=cond.T, label=name,
                     style_preset=name)

    def _neutral_cond(self, n_chunks: int = N_CHUNKS,
                      bpm: float = 120.0, bpm_variance: float = 2.0,
                      seed: int = 0) -> Conditioning:
        """A shape-valid conditioning with plausible values in every channel.

        Used as the carrier for entry points that don't supply every signal
        (text, unconditional): the model still READS all seven tensors, the mask
        is what decides whether they're listened to, so they only need to be the
        right shape and not absurd."""
        T = int(n_chunks)
        return Conditioning(
            style=np.zeros((T, STYLE_DIM), np.float32),
            chroma=np.full((T, 12), 1.0 / 12, np.float32),
            rms=np.full(T, 5.0, np.float32),
            density=np.full(T, 2.5, np.float32),
            zcr=np.full(T, 0.1, np.float32),
            flatness=np.full(T, 0.01, np.float32),
            bpm=bpm_curve(bpm, bpm_variance, T, seed))

    def seed_from_text(self, prompt: str, seed: int,
                       bpm: float = 120.0, bpm_variance: float = 2.0,
                       n_chunks: int = N_CHUNKS) -> Track:
        """Text conditioning via CLAP.

        CLAP aligns its text and audio towers in ONE 512-d space, and that space
        is exactly what the DiT's `style` channel was trained on (per-measure
        AUDIO embeddings). So a caption's text embedding is a drop-in style
        vector — no separate text->conditioning meta-model needed.

        A prompt says nothing about the per-measure DSP signals, so those start
        DROPPED (their learned/zeroed null path) and only style + bpm condition
        the model. That's in-distribution: stage-2 trained with independent
        per-signal dropout at p=0.5. Turn the others on from the coverage panel
        once you've heard what the prompt gives you.

        `meta_model`, if one is wired in, still wins — it produces a full
        trajectory rather than style alone."""
        if self.meta_model is not None:
            cond = self.meta_model.predict_conditioning(
                prompt, n_chunks=n_chunks, seed=int(seed))
            return Track(cond=cond, seed=int(seed), n_chunks=cond.T,
                         label=f"text: {prompt[:40]}", style_text=prompt)

        from preset_extractor import embed_text
        vec = embed_text(prompt)                      # (STYLE_DIM,), L2-normed
        cond = self._neutral_cond(n_chunks, bpm, bpm_variance, int(seed))
        cond.style[:] = vec
        for s in ("chroma", "rms", "density", "zcr", "flatness"):
            cond.active[s][:] = False
        return Track(cond=cond, seed=int(seed), n_chunks=cond.T,
                     label=f"text: {prompt[:40]}", style_text=prompt)

    def seed_from_audio(self, path: str, seed: int,
                        n_chunks: int = N_CHUNKS) -> Track:
        """Full conditioning extracted from an uploaded audio file: the seven
        per-measure signals from the training DSP, style from the CLAP audio
        tower (centered ~10s windows). Same code path the presets/ folder uses,
        so an upload and a preset are indistinguishable downstream."""
        import os
        from preset_extractor import extract_conditioning
        cond = extract_conditioning(path, n_chunks=n_chunks)
        return Track(cond=cond, seed=int(seed), n_chunks=cond.T,
                     label=f"upload: {os.path.basename(path)}")

    def set_style_from_text(self, track, prompt, start=0, end=None):
        """Overwrite the style rows in [start, end) with a CLAP text embedding.
        Everything else (chroma/rms/.../bpm and their active flags) is untouched,
        so this composes with an audio-derived trajectory: keep a real song's
        rhythm and harmony, swap its character for a described one."""
        from preset_extractor import embed_text
        tr = track.copy()
        c = tr.cond
        e = c.T if end is None else end
        s, e = self._range(start, e, c.T)
        c.style[s:e] = embed_text(prompt)
        c.active["style"][s:e] = True
        tr.style_text = str(prompt)
        tr.latents = None
        return tr

    def set_bpm_global(self, track, bpm, variance, seed=0, start=0, end=None):
        """Global tempo + per-measure Gaussian jitter (see backend.bpm_curve).
        Written into cond.bpm verbatim, so the curve stays visible in the plot
        and editable by apply_shape afterwards."""
        tr = track.copy()
        c = tr.cond
        e = c.T if end is None else end
        s, e = self._range(start, e, c.T)
        c.bpm[s:e] = bpm_curve(bpm, variance, e - s, int(seed))
        tr.latents = None
        return tr

    def seed_unconditional(self, seed: int, n_chunks: int = N_CHUNKS) -> Track:
        """Pure init-from-noise: all signals inactive. The conditioning values
        still need valid shapes (the model reads them but the mask drops them),
        so we borrow a neutral template and flip every active flag off."""
        if self.preset_cond:
            cond = next(iter(self.preset_cond.values())).copy()
        else:
            cond = self._neutral_cond(n_chunks, seed=int(seed))
        for s in cond.active:
            cond.active[s][:] = False
        return Track(cond=cond, seed=int(seed), n_chunks=cond.T,
                     label="unconditional")

    def set_active(self, track, signal, start, end, on):
        tr = track.copy()
        s, e = self._range(start, end, tr.cond.T)
        tr.cond.set_active(signal, s, e, bool(on))
        tr.latents = None   # conditioning coverage changed; cached latents stale
        return tr

    def set_stage(self, track, stage):
        """Choose which DiT (stage 1 or 2) future renders use. Stored on the
        Track.

        Keeps tr.latents, for the reason set_guidance spells out -- the two had
        the same bug and it is the same argument."""
        tr = track.copy()
        st = int(stage)
        tr.stage = st if st in self.models else self.default_stage
        return tr

    def set_guidance(self, track, guidance):
        """Set the CFG guidance scale future renders use. Stored on the Track
        (survives the ZeroGPU pickle boundary). 1.0 == plain conditional
        (training's `_cond`); ~3.0 == joint CFG (training's `_cfg`).

        KEEPS tr.latents, and that is load-bearing. This used to drop it, on the
        reasoning that "a different scale produces different latents, so drop
        the cache". The reasoning confuses two things. `tr.latents` is not a
        derived cache of the conditioning that a new scale invalidates; it is
        the record of the latents the audio in `tr.window_audio` was actually
        decoded from. Guidance changes what FUTURE sampling does. It cannot
        retroactively change what the existing bars are made of, and the
        existing bars are precisely what a continuation needs to condition on.

        Dropping it did not even buy the re-render it was aiming at: `render()`
        regenerates unconditionally and never reads `tr.latents`. The only code
        that reads the field is continuation, and there a None means something
        entirely different -- `continue_track`/`fast_forward`/`stream_forward`
        all guard with

            if tr.latents is None or tr.window_bounds is None:
                tr = self.render(tr, n_steps=n_steps)

        which is meant for a track that was never rendered. On a track that is
        96 measures in, it samples a brand-new window from PURE NOISE (
        `model.generate`, no context, no conditioning on the past) while
        `_finish` keeps `prefix_audio` -- so the committed measures survive and
        the 24 measures the listener just heard are replaced by unrelated music
        spliced onto the end of them.

        That made it a live bug on the hottest path in the app, not a corner
        case: `app._steering` runs `set_guidance` on EVERY Continue and EVERY
        Fast-forward press, with whatever the slider already said. So every
        press after the first threw away the song's context and jumped. It hid
        because the first press on a fresh track has `prefix_measures == 0` and
        a deterministic seed, so the re-render reproduces the same window and
        nothing sounds wrong; the jump only appears once there is committed
        audio to jump away from."""
        tr = track.copy()
        tr.guidance = float(guidance)
        return tr

    # =====================================================================
    # conditioning -> model kwargs
    # =====================================================================
    def _net_kwargs(self, cond: Conditioning):
        """Build the (B=1) conditioning kwargs the model.forward expects.
        Raw values; the model z-scores internally."""
        torch = self.torch
        dev = self.device

        def f(a):  # (T,) -> (1, T) float
            return torch.from_numpy(np.asarray(a, dtype=np.float32))[None].to(dev)

        bpm = f(cond.bpm)
        rms = f(cond.rms)
        density = f(cond.density)
        zcr = f(cond.zcr)
        flatness = f(cond.flatness)
        chroma = torch.from_numpy(
            np.asarray(cond.chroma, dtype=np.float32))[None].to(dev)   # (1,T,12)
        style = torch.from_numpy(
            np.asarray(cond.style, dtype=np.float32))[None].to(dev)    # (1,T,512)

        net_kwargs = {"bpm": bpm, "rms": rms, "density": density, "zcr": zcr,
                      "flatness": flatness, "chroma": chroma, "style": style}

        T = cond.T
        # Per-signal shapes of the unconditional_mask entries. The trailing dim
        # MUST be 1 for every signal, chroma included.
        #
        # Two things in the model's forward consume these:
        #   1. torch.where(mask, null, value) -- broadcasting (1,T,1) over
        #      (B,T,12) chroma / (B,T,768) style|bpm masks per measure.
        #   2. local_drop[k] |= mask[k].squeeze(-1) -- which then feeds the
        #      PRESENCE CHANNELS the stage-2 local_embedder takes as input
        #      (21 channels = 12 chroma + 4 scalars + 5 presence flags).
        # (2) needs a (B,T) result, so a (1,T,12) chroma mask would broadcast
        # into (1,T,12) and blow up the presence stack. Keep the trailing 1.
        shapes = {
            "bpm": (1, T, 1), "rms": (1, T, 1), "density": (1, T, 1),
            "zcr": (1, T, 1), "flatness": (1, T, 1),
            "chroma": (1, T, 1), "style": (1, T, 1),
        }

        def _all_dropped():
            return {k: torch.ones(s, device=dev).bool() for k, s in shapes.items()}

        # unconditional mask (all True == everything dropped): the uncond CFG leg.
        uncond_net_kwargs = net_kwargs | {"unconditional_mask": _all_dropped()}

        # Per-measure "active" flags from the UI. active[sig] is (T,) bool; a
        # signal is CONDITIONED (mask False) only where active AND (for style)
        # not globally dropped. Everywhere else it's dropped (mask True).
        #
        # PARTIAL PATTERNS ARE OFF-DISTRIBUTION. The mechanism exists in the
        # model -- the mask is per measure and the forward handles it -- but it
        # was never exercised: both stages call multi_token_drop with
        # p_ind_low = p_ind_high = 0, so `token_noise < sample_drop_rates` is
        # uniformly False and the per-measure drop branch is dead code. Sampling
        # 28,672 windows of each stage's real schedule found a signal partially
        # dropped 0 times; every window was all-on or all-off per signal. The
        # damage shows up through the stage-2 presence channels, which enter via
        # a kernel-3 conv over the measure axis and an additive skip, so an
        # on->off step inside the window pushes the measures next to it off the
        # latent manifold -- audible as those measures alone coming back at
        # clipping volume. Continue therefore drops the DSP signals window-wide
        # rather than only on its new measures (see backend.shift_window); the
        # coverage grid can still produce partial patterns and this is why they
        # sound wrong.
        def _drop_from_active(sig):
            a = np.asarray(cond.active[sig], dtype=bool)              # (T,)
            drop = ~a                                                 # True == dropped
            if self.drop_style and sig == "style":
                drop = np.ones_like(drop)                            # force dropped
            t = torch.from_numpy(drop).to(dev).view(1, T, 1)
            return t

        # a signal contributes to conditioning iff it is active on >=1 measure
        # (and not force-dropped)
        def _has_cond(sig):
            if self.drop_style and sig == "style":
                return False
            return bool(np.asarray(cond.active[sig], dtype=bool).any())

        # The joint-conditional pass: every signal masked per its active measures
        # in ONE kwargs entry. This is training's dict-form `net_kwargs` (used for
        # `{i}_cond.wav`, guidance 1.0) and also the joint-CFG conditional leg.
        cond_mask = {k: _drop_from_active(k) for k in shapes}
        joint_net_kwargs = net_kwargs | {"unconditional_mask": cond_mask}

        if self.cfg_mode == "joint":
            cfg_net_kwargs = [joint_net_kwargs]
        else:  # independent / compositional: one pass per contributing signal
            cfg_net_kwargs = []
            for k in shapes:
                if not _has_cond(k):
                    continue  # skip signals that are off everywhere
                m = _all_dropped()
                m[k] = _drop_from_active(k)   # condition only this signal, per-measure
                cfg_net_kwargs.append(net_kwargs | {"unconditional_mask": m})

        return cfg_net_kwargs, uncond_net_kwargs, joint_net_kwargs, len(cfg_net_kwargs)

    def _sampling_args(self, cond: Conditioning, guidance: Optional[float] = None):
        """Assemble the sampler args (net_kwargs list, uncond kwargs, per-pass
        guidances, cfg_mode, joint kwargs). Handles the fully-unconditional case:
        when no signal is active on any measure we do a single plain pass
        (guidance 1, no CFG steering) so generation is driven purely by the noise
        seed.

        The joint conditional comes back alongside the list because the DFoT
        history-guidance composition needs the all-signals-at-once leg by itself,
        and in independent cfg_mode the list holds one-signal-each legs instead.
        In every other mode it is already the list's only entry.

        `guidance` overrides the backend default (self.guidance) when provided —
        the app passes the per-Track slider value here. A guidance of 1.0 gives
        a single plain conditional pass (matches training's `{i}_cond.wav`);
        >1 does CFG (3.0 matches `{i}_cfg.wav`)."""
        g = self.guidance if guidance is None else float(guidance)
        cfg_net_kwargs, uncond_net_kwargs, joint_net_kwargs, n_sig = \
            self._net_kwargs(cond)
        if cond.is_unconditional() or n_sig == 0:
            # one unconditional pass; guidance 1.0 == just that prediction.
            return [uncond_net_kwargs], uncond_net_kwargs, [1.0], "joint", \
                uncond_net_kwargs
        if g == 1.0:
            # No CFG: a single plain conditional pass in joint mode. This is
            # training's `{i}_cond.wav` path (predict_measures with guidance=1.0
            # and the dict-form net_kwargs). get_prediction sees guidance 1.0
            # and one kwargs entry, so it skips the uncond leg entirely — even in
            # independent cfg_mode, guidance 1 means "no steering", so the single
            # joint conditional pass is the right (and cheapest) choice.
            return [joint_net_kwargs], uncond_net_kwargs, [1.0], "joint", \
                joint_net_kwargs
        return cfg_net_kwargs, uncond_net_kwargs, [g] * n_sig, \
            self.cfg_mode, joint_net_kwargs

    # =====================================================================
    # decode latents -> waveform  (mirrors predict_measures tail exactly)
    # =====================================================================
    def _bpm_row(self, cond):
        """cond.bpm as the (1, T) tensor `_decode` wants.

        Also the one place that can notice a bpm write site forgot to snap.
        Every producer in backend.py goes through snap_bpm, so an off-grid
        value arriving here means a NEW write site was added without it, and
        the consequence -- the DiT reading a row of nn.Embedding(350) that
        training never visited -- is otherwise completely silent.

        Warns; does NOT raise. Off-grid conditioning is degraded, not invalid:
        it renders, it just renders worse. Anything stronger than a log line
        here turns a quality problem into an outage.

        The threshold is relative because the grid quantum grows as bpm^2, so
        no absolute tolerance is right at both ends. 1e-3 is comfortably above
        float32 round-trip noise (the presets land within 2.8e-6) and well
        below any real miss."""
        import torch
        b = np.asarray(cond.bpm, dtype=np.float64)
        err = bpm_grid_error(b)
        if err > 1e-3 * max(1.0, float(np.max(b))):
            worst = int(np.argmax(np.abs(b - np.asarray(snap_bpm(b)))))
            print(f"[RealBackend] bpm off the tempo grid by {err:.3f} BPM "
                  f"(measure {worst}: {b[worst]:.3f}, nearest grid tempo "
                  f"{snap_bpm(float(b[worst])):.3f}) -- some write site is "
                  f"missing snap_bpm; the model reads an embedding row "
                  f"training never visited. Rendering anyway.")
        return torch.from_numpy(b.astype(np.float32))[None].to(self.device)

    def _decode(self, y, bpm_row, n_steps, decoder_noise=None,
                return_bounds=False, max_len=None):
        """y: (B, T, N, C) latents.  bpm_row: (B, T) tensor.  Returns np audio (mono).

        With return_bounds=True returns (audio, downbeats) where `downbeats` is
        a list of n_chunks+1 times IN SECONDS marking where each measure landed
        in the stitched output -- see the crossfade note in the join loop.

        `max_len` overrides the length the tokenizer decodes every measure at,
        which is otherwise the batch maximum over `target_samples`. That default
        makes a measure's audio depend on the SLOWEST measure it happens to be
        batched with, so decoding a subset of a window would not reproduce what
        the whole window produced. Callers decoding a subset (see
        `_continue_step(decode="new")`) pass the whole window's value to keep
        the two identical. Costs nothing relative to a full decode, which runs
        at that length for every measure anyway."""
        import torch
        from einops import rearrange
        crossfade_segments = _crossfade_segments  # inlined; avoids importing train script (fad/pyrubberband)

        dev = self.device
        n_chunks = y.shape[1]
        seconds_per_beat = 60.0 / bpm_row
        measure_duration_sec = seconds_per_beat * TARGET_SIG
        target_samples = (measure_duration_sec * SR).long()

        max_len = min(int(max_len) if max_len is not None
                      else target_samples.max().item(),
                      self.encoder_ratios * (self.max_adapter_len - 1))
        max_len = self.encoder_ratios * math.ceil(max_len / self.encoder_ratios)
        max_latent_len = max_len // self.encoder_ratios

        # The adapter's decode ALWAYS runs at its full query length, never at
        # this batch's max_latent_len.
        #
        # adapter.SequenceDecoder.forward lays out `max_latent_len` query slots,
        # cross-attends them to the 64 DiT tokens (kv/q masked, fine) and then
        # runs SELF-attention across the slots -- Block.forward calls
        # self.sa(...) with no mask at all. So every real slot mixes with every
        # padding slot, and the result depends on how many slots there are. The
        # adapter was trained with the slot count set by the batch max over many
        # random measures, which pins it at max_seq_len-1 essentially always, so
        # that is the only regime it is any good in. Measured on real audio
        # (latent recon NMSE, 24 measures through encode -> adapter -> compare):
        #
        #     slots  34 -> 0.446      slots  64 -> 0.0039
        #     slots  48 -> 0.184      slots  96 -> 0.00016
        #
        # predict_measures gets away with `max_latent_len` because it renders 10
        # val songs at once, so one slow song drags the batch max up into the
        # good regime. This demo renders ONE track, and a preset whose measures
        # are all the same fast tempo lands at 34 slots -- the adapter throws
        # away a third of the signal before the tokenizer ever sees it, which is
        # audible as across-the-board mush no amount of sampler steps can fix.
        #
        # Slice back to max_latent_len afterwards: the tokenizer wants the SHORT
        # length. Decoding zero-padded latent slots out to the full 65024 samples
        # measurably hurts the valid region (NMSE vs source 0.339 -> 0.582) and
        # costs ~4x the decoder compute.
        adapter_len = self.max_adapter_len - 1

        indices = torch.arange(adapter_len, device=dev).view(1, 1, -1)
        lengths = ((target_samples + self.encoder_ratios - 1)
                   // self.encoder_ratios).unsqueeze(-1)
        mask = (indices < lengths).view(y.shape[0] * n_chunks, adapter_len)
        shape = (y.shape[0] * n_chunks, self.vae_embed_dim, adapter_len)

        with self._ctx():
            yy = rearrange(y, "b t n c -> (b t) c n")
            yy = self.adapter.decode(yy, shape, mask=mask)[..., :max_latent_len]
            yy = self.tokenizer.decode(
                yy, shape=(1, max_len), n_steps=n_steps,
                noise=decoder_noise[:, :, :max_len] if decoder_noise is not None else None)

        ts = target_samples.flatten().cpu().numpy().reshape(y.shape[0], n_chunks)
        yy = yy.squeeze().float().cpu().numpy()

        out, downbeats = [], []
        for i in range(y.shape[0]):
            segs = [yy[i * n_chunks + j][:min(int(ts[i][j]), max_len)]
                    for j in range(n_chunks)]
            temp = segs[0]
            starts = [0]
            for j in range(1, n_chunks):
                temp = crossfade_segments(temp, segs[j], sample_rate=SR,
                                          crossfade_ms=CROSSFADE_MS)
                # Where measure j begins in the stitched signal. The crossfade
                # overlaps the tail of `temp` with the head of segs[j], so the
                # total grows by len(segs[j]) - crossfade_samples -- but it
                # degrades to a plain concatenation when either side is shorter
                # than the overlap. Reading the start back out of the *actual*
                # new length covers both cases without duplicating that rule.
                starts.append(len(temp) - len(segs[j]))
            out.append(temp.astype(np.float32))
            downbeats.append([s / SR for s in starts] + [len(temp) / SR])
        if return_bounds:
            return out[0], downbeats[0]
        return out[0]

    def _decoder_noise(self, n_chunks, first_measure, base_seed):
        """Initial noise for the tokenizer's sampler, keyed per SONG measure.

        Returns (n_chunks, 1, cap) laid out to match _decode's flattened
        (b t) batch at B=1, generated at the longest measure this decoder can
        produce so _decode's `[:, :, :max_len]` slice is a prefix of a fixed
        draw -- the same measure therefore gets the same noise whatever the
        window's tempo makes max_len on any given call.

        Keyed on `first_measure + slot`, i.e. the measure's index in the SONG,
        not its slot in the window. After a Continue the window slides by 8 and
        every surviving measure keeps its key, so its decode is unchanged. That
        is what makes the re-decoded context match the audio already playing
        (see Track.decode_seed), and it is a precondition for streaming, where
        the join between what has played and what comes next has to be sample-
        continuous rather than merely musically continuous."""
        import torch
        cap = self.encoder_ratios * (self.max_adapter_len - 1)
        out = torch.empty((int(n_chunks), 1, cap), device=self.device)
        for j in range(int(n_chunks)):
            key = (int(base_seed) * 1_000_003 + int(first_measure) + j)
            g = torch.Generator(device=self.device)
            g.manual_seed(key & 0x7FFF_FFFF)
            out[j] = torch.randn((1, cap), device=self.device, generator=g)
        return out

    def _measure_render(self, tr, downbeats, audio=None):
        """Fill in the conditioning of everything the model was NOT told, from
        what it actually generated.

        `audio` is the WINDOW's audio, which is what `downbeats` indexes and
        what tr.cond describes -- after a Continue that is not the same thing as
        tr.audio, which also carries the banked prefix.

        A text seed carries a style vector and a tempo and nothing else, so its
        chroma/rms/density/zcr/flatness sit at neutral placeholders; an
        unconditional seed has no meaningful values at all. Those are the curves
        the UI draws, so before this the user was looking at (and would be
        editing from) numbers that describe nothing they can hear. Re-running
        the preset DSP over the render replaces them with the real trajectory of
        the audio in the player -- a starting point that is on-manifold and
        already coherent with the take, so turning a signal ON is a nudge rather
        than a jump.

        Only measures where a signal is OFF are touched. Where it is on, the
        model followed the number the user put there and that number is the
        record of their edit; overwriting it with a measurement of how well the
        model complied would silently destroy edits and make every regenerate
        drift. Per-measure, so partial coverage behaves the same way.

        bpm is never touched: it does not describe the render, it DEFINES it
        (_decode turns each bpm into that measure's sample count), so the
        measured value is the requested one by construction."""
        c = tr.cond
        free = {s: ~np.asarray(c.active[s], dtype=bool)
                for s in ALL_SIGNALS if s != "bpm"}
        if not any(m.any() for m in free.values()):
            return                       # fully conditioned: nothing to learn
        # Style is inactive on every measure whenever Track.style_off is set,
        # so this re-measures it from the render on every take of a styled-off
        # run -- one CLAP audio-tower forward per Continue, which is worth
        # paying: it is what makes turning style back on resume from the style
        # the music actually has rather than from whatever it was pinned to
        # before it went off. `self.drop_style` is the different, backend-wide
        # case where style.pt never loaded and there is no tower to ask.
        want_style = bool(free["style"].any()) and not self.drop_style
        try:
            from preset_extractor import analyze_measures
            sig = analyze_measures(tr.audio if audio is None else audio,
                                   downbeats, sr=SR, n_chunks=c.T,
                                   with_style=want_style, label="render")
        except Exception as e:  # noqa: BLE001
            # Best-effort: librosa/CLAP failing here must not lose a good render.
            # On tr.diag, not stdout: this runs in the forked @spaces.GPU
            # worker, whose stdout does not reach the Space log (see
            # Track.diag). A print here is a failure nobody can see, and the
            # symptom it leaves behind -- the curves still showing
            # _neutral_cond's constants instead of the render -- looks like a
            # plotting bug rather than an analysis that never happened.
            tr.diag = (f"render analysis failed ({type(e).__name__}: {e}); "
                       f"conditioning curves left as they were\n"
                       + (tr.diag or ""))
            return
        for s, m in free.items():
            if m.any() and s in sig:
                getattr(c, s)[m] = sig[s][m]

    # =====================================================================
    # render
    # =====================================================================
    def render(self, track: Track, n_steps: int = SAMPLE_STEPS) -> Track:
        import torch
        self.to_gpu()
        tr = track.copy()
        self._select_stage(getattr(tr, "stage", self.default_stage))
        c = tr.cond
        gen_shape = (1, c.T, self.spatial_window, self.vae_embed_dim)

        g = torch.Generator(device=self.device).manual_seed(int(tr.seed))
        gen_noise = torch.randn(gen_shape, device=self.device, generator=g)

        cfg_net_kwargs, uncond_net_kwargs, guidances, cfg_mode, \
            joint_net_kwargs = \
            self._sampling_args(c, getattr(tr, "guidance", None))

        with self._ctx():
            y = self.model.generate(
                gen_shape, net_kwargs=cfg_net_kwargs,
                uncond_net_kwargs=uncond_net_kwargs, n_steps=n_steps,
                guidance=guidances, noise=gen_noise, memory_efficient=False,
                cfg_mode=cfg_mode, t_dist=self.t_dist)

        bpm_row = self._bpm_row(c)
        window_audio, bounds = self._decode(
            y, bpm_row, n_steps, return_bounds=True,
            decoder_noise=self._decoder_noise(c.T, tr.prefix_measures,
                                              tr.decode_seed))
        tr.sr = SR
        tr.latents = y.float().cpu().numpy()   # cache for inpainting
        self._finish(tr, window_audio, bounds)
        self._measure_render(tr, bounds, window_audio)
        return tr

    # =====================================================================
    # editing (identical semantics to MockBackend; verbatim, no clamping)
    # =====================================================================
    def _range(self, start, end, T):
        s = max(0, min(int(start), T))
        e = max(s, min(int(end), T))
        if e == s:
            e = min(T, s + 1)
        return s, e

    def apply_shape(self, track, signal, start, end, shape, params, mode="replace"):
        assert signal in EDITABLE_SIGNALS, f"unknown signal {signal}"
        tr = track.copy()
        c = tr.cond
        s, e = self._range(start, end, c.T)
        vals = eval_shape(shape, e - s, params)
        arr = getattr(c, signal)
        arr[s:e] = (arr[s:e] + vals) if mode == "add" else vals
        if signal == "bpm":
            # bpm is the one signal with a representable SET rather than a
            # range, so a drawn curve gets projected onto it. Not a clamp; see
            # snap_bpm, and MockBackend.apply_shape, which does the same.
            arr[s:e] = snap_bpm(arr[s:e])
        tr.latents = None   # conditioning changed; cached latents are stale
        return tr

    def set_key(self, track, root, minor, start=0, end=None):
        tr = track.copy()
        c = tr.cond
        e = c.T if end is None else end
        s, e = self._range(start, e, c.T)
        c.chroma[s:e] = key_template(root, minor)
        tr.latents = None
        return tr

    def morph_style(self, track, style_name, amount, start=0, end=None):
        tr = track.copy()
        c = tr.cond
        e = c.T if end is None else end
        s, e = self._range(start, e, c.T)
        if style_name in self.preset_cond:
            tgt = self.preset_cond[style_name].style[0]
            for i in range(s, e):
                c.style[i] = self._slerp(c.style[i], tgt, float(amount))
            tr.style_preset = str(style_name)
        tr.latents = None
        return tr

    # One implementation, in backend.py, shared with MockBackend and with the
    # steering path -- morph_style and a Continue steer have to agree on what a
    # half-way-between style is or the two controls fight each other.
    _slerp = staticmethod(slerp)

    # ---- style steering ------------------------------------------------------
    def embed_style_text(self, prompt):
        """Unit CLAP text embedding, cached by prompt.

        Cached because a Continue re-resolves the steer text on every press and
        the text tower is a full T5-large forward -- seconds on CPU, which is
        where this deliberately runs (see ModelBackend.embed_style_text)."""
        key = str(prompt).strip()
        if not key:
            return None
        hit = self._text_style_cache.get(key)
        if hit is None:
            from preset_extractor import embed_text
            hit = unit(embed_text(key))
            self._text_style_cache[key] = hit
        return hit

    def style_points(self):
        """Every MEASURE of every preset, deduped — real CLAP audio-tower
        embeddings of real recordings.

        Per measure, not per preset, because a per-preset pool would be as
        small as the preset list and make `wander` close to a coin flip. The
        style bin is per-measure (see the training loader) and a song's intro
        really does embed differently from its peak, so `n_presets x 24` are
        legitimate points rather than an inflation of one. Near-duplicates are
        dropped at cos > 0.995 so the walk cannot pick a "target" it is already
        sitting on; measures are labelled `preset#measure` so the progress
        readout can name the nearest one, and so `style_bpm` can find the
        preset a point came from.

        HOW MUCH DIVERSITY THIS ACTUALLY BUYS, measured on the five shipped
        presets: 88 points survive dedup, and the median angle BETWEEN presets
        is 81 degrees -- against STYLE_FULL_CHANGE_DEG's 90 for "an unrelated
        piece". So the clusters are far apart even if each is tight, and a
        wander between them is a real journey. (An earlier version of this note
        claimed "three tight clusters ... buys texture, not diversity" from a
        three-preset era; it was measured and is wrong for the current set.)

        More presets is the only real fix. A named drift axis (style_axis_vector)
        sidesteps the pool entirely and is the better tool for going somewhere
        genuinely new."""
        out = []
        for name, c in self.preset_cond.items():
            try:
                st = np.asarray(c.style, dtype=np.float32)
            except Exception:
                continue
            for i in range(st.shape[0]):
                v = unit(st[i])
                if any(float(v @ p) > 0.995 for _, p in out):
                    continue
                out.append((f"{name}#{i}", v))
        return out

    def style_bpm(self):
        """{point_name: bpm} for every point `style_points` offers.

        THE PRESET'S TEMPO, NOT THE MEASURE'S, and every point of a preset gets
        the same number even though the points themselves are per measure. Two
        reasons, and the second is the load-bearing one.

        First, semantics: `shift_window` already treats bpm as a property of the
        piece rather than a description of a bar, and a style destination stands
        for "somewhere that sounds like this", which is a piece.

        Second, the per-measure values are BEAT-TRACKED and the tracker doubles.
        Preset 4 reads ~273 for eighteen measures and then 522-571 for the last
        six; preset 5 scatters 571s among 286s; preset 8 has 462-480 among 235s.
        Those are octave errors, not tempo changes -- the model itself cannot
        even represent them apart, since bpm goes through nn.Embedding(350)
        after round+clamp, so 522 and 571 are both just row 349. Steering the
        tempo toward a per-measure value would therefore aim at a beat-tracking
        artefact roughly a fifth of the time and be heard as a 2x lurch.

        The median is taken AFTER an octave fold rather than before, so the
        doubled measures are pulled back onto the piece's tempo instead of just
        being out-voted -- which matters for a preset where they are common.
        Measured over the shipped set this collapses the per-measure spread from
        std 118 / 98 / 88 BPM to 4.4 / 17 / 2.8, leaving folded medians of 176,
        231, 235, 273 and 286: a 1.6x range, which is a real journey to steer
        through rather than noise to be smoothed away."""
        out = {}
        for name, c in self.preset_cond.items():
            try:
                b = np.asarray(c.bpm, dtype=np.float64)
                b = b[np.isfinite(b) & (b > 0)]
                if not b.size:
                    continue
                # Fold against the raw median first (a robust starting guess),
                # then re-take the median of the folded values.
                m = float(np.median(b))
                folded = np.array([fold_bpm(x, m) for x in b])
                out[name] = float(np.median(folded))
            except Exception:
                continue
        # style_points labels every point `preset#measure`; give each the tempo
        # of the preset it came from, so lookups by destination name just work.
        return {f"{name}#{i}": bpm
                for name, bpm in out.items()
                for i in range(int(np.asarray(
                    self.preset_cond[name].style).shape[0]))}

    def style_axis_vector(self, label):
        """unit(embed(plus) - embed(minus)) for a named axis; None for random.

        Two text-tower forwards, cached, on CPU. First use of a given axis pays
        for them; after that it is free. Runs outside the @spaces.GPU call for
        the same reason embed_style_text does."""
        pair = resolve_axis_prompts(label)
        if pair is None:
            return None
        key = f"axis::{pair[0]}||{pair[1]}"
        hit = self._text_style_cache.get(key)
        if hit is None:
            hit = unit(self.embed_style_text(pair[0])
                       - self.embed_style_text(pair[1]))
            self._text_style_cache[key] = hit
        return hit

    # =====================================================================
    # region regenerate == masked resample of measure rows [start, end)
    # =====================================================================
    def regenerate_region(self, track, start, end, seed, n_steps=SAMPLE_STEPS,
                          strength=REGEN_STRENGTH):
        """Resample measures [start, end); hold the rest at their real latents.

        The ONLY sampler the Edit tab's Regenerate reaches, at every width. It
        used to share the button with `render` for a selection that filled the
        window, which was not a wider version of this but a different operation
        -- pure noise, no frozen neighbours, no past -- and it is why an
        untouched full-width Regenerate came back as unrelated music.
        EDIT_MAX_MEASURES now keeps a selection below the window width so there
        is always something to freeze, and `strength` covers the rest: identity
        can come from where the denoise STARTS as well as from what surrounds
        it. See REGEN_STRENGTH and fm_df_sample's `strength`."""
        import torch
        self.to_gpu()
        tr = track.copy()
        self._select_stage(getattr(tr, "stage", self.default_stage))
        c = tr.cond
        s, e = self._range(start, end, c.T)

        # need clean latents to keep for the frozen region; render once if absent
        if tr.latents is None:
            tr = self._ensure_window(tr, n_steps)
        known_x0 = torch.from_numpy(tr.latents).to(self.device)

        keep = torch.ones(1, c.T, 1, 1, device=self.device).bool()
        keep[:, s:e] = False   # regenerate this range, keep the rest

        g = torch.Generator(device=self.device).manual_seed(int(seed))
        noise = torch.randn(known_x0.shape, device=self.device, generator=g)

        cfg_net_kwargs, uncond_net_kwargs, guidances, cfg_mode, \
            joint_net_kwargs = \
            self._sampling_args(c, getattr(tr, "guidance", None))

        with self._ctx():
            y = fm_df_sample(
                self.diffusion, self.net, known_x0, keep, n_steps,
                net_kwargs=cfg_net_kwargs, uncond_net_kwargs=uncond_net_kwargs,
                guidance=guidances, noise=noise, memory_efficient=False,
                cfg_mode=cfg_mode, t_dist=self.t_dist,
                ctx_noise=REGION_CTX_NOISE,
                # off here: a region being regenerated normally still has its
                # conditioning curves on, so it is not underspecified the way a
                # continuation is, and this keeps region-regen behaviour as it was.
                history_guidance=1.0,
                strength=float(strength))

        bpm_row = self._bpm_row(c)
        window_audio, bounds = self._decode(
            y, bpm_row, n_steps, return_bounds=True,
            decoder_noise=self._decoder_noise(c.T, tr.prefix_measures,
                                              tr.decode_seed))
        tr.latents = y.float().cpu().numpy()
        tr.seed = int(seed)
        self._finish(tr, window_audio, bounds)
        self._measure_render(tr, bounds, window_audio)
        return tr

    # =====================================================================
    # continuation
    # =====================================================================
    def _ensure_window(self, tr, n_steps):
        """Render a window for a track that has none. The three continuation
        entry points' `if tr.latents is None` guard, in one place.

        That guard is for a track that was never rendered. Reaching it with
        COMMITTED audio in `prefix_audio` means something cleared the latent
        cache mid-song, and the render that follows cannot continue anything:
        `render` samples from pure noise and never reads the past, so the
        committed measures keep playing and then jump into unrelated music.

        There is no repair available here -- the latents that described those
        bars are gone and nothing else can reconstruct them. What IS available
        is saying so, on `tr.diag`, the only channel out of the ZeroGPU worker
        (see Track.diag). Silence is what let `set_guidance` on an untouched
        slider do this on every press for as long as it did."""
        committed = int(getattr(tr, "prefix_measures", 0) or 0)
        tr = self.render(tr, n_steps=n_steps)
        if committed:
            tr.diag = (
                f"⚠ no latent cache {committed} measures into the song, so the "
                f"window was RE-RENDERED rather than continued: the last "
                f"{tr.cond.T} measures are new music spliced onto the committed "
                f"ones, not a continuation of them. Something cleared "
                f"tr.latents mid-song — see set_guidance.\n" + (tr.diag or ""))
        return tr

    def continue_track(self, track, n_new=CONTINUE_MEASURES, seed=0,
                       n_steps=SAMPLE_STEPS):
        """Write the next `n_new` measures against the ones already there.

        Classic fixed-window continuation. T is pinned at the trained 24 (see
        CONTINUE_MEASURES for why that is not a knob), so this slides the window
        rather than growing it:

            window before   [ 0 .. 7 | 8 ................ 23 ]
                              banked   re-presented as context
            window after    [ 8 ............... 23 | new 8  ]
                              FROZEN latents         sampled

        The 24-n_new context measures keep their exact latents from the previous
        render and are handed to the model CLEAN, at noise level ~0, while the
        new measures denoise from t=1 -- diffusion forcing's per-measure noise
        level, which is what this model was trained for (see fm_df_sample). Same
        machinery region-regen uses, just with the free region pinned to the end
        of the window. So the new bars are conditioned on the actual audio that
        precedes them, not merely on a similar-looking conditioning curve.

        Only the newly sampled tail is appended to the song; the context
        measures are already in the mix, and their audio is carried forward
        rather than re-decoded, even though the model emits the whole window."""
        self.to_gpu()
        tr = track.copy()
        self._select_stage(getattr(tr, "stage", self.default_stage))

        # Need real latents (and the bounds that go with them) to freeze.
        if tr.latents is None or tr.window_bounds is None:
            tr = self._ensure_window(tr, n_steps)

        # decode="new", not "full". The full-window decode was the default here
        # on two grounds: a lone Continue has no context audio worth protecting,
        # and re-decoding everything keeps the output byte-comparable with
        # render(). Neither survives the seam contract. A freshly decoded window
        # genuinely is a different tokenizer decode from the prefix, so `_finish`
        # genuinely has to crossfade the two -- and that crossfade consumes 20 ms
        # of prefix the listener has ALREADY heard. Every Continue therefore
        # deletes 320 samples mid-song and slides everything past the seam 20 ms
        # earlier: once per press, on a bar line, compounding (120 ms gone after
        # six). It is the "skip" the Track.audio_seam note warns about, arriving
        # through the other door -- not a mismatched join, but a join that is
        # correct and still lossy because it is being asked to happen at all.
        #
        # The incremental path has no seam to crossfade: `_bank_window` cuts the
        # prefix out of the very array `_join_window` keeps the rest of, so
        # prefix ⌢ window is sample-continuous by construction and `_finish`
        # butts them. It is also a third of the decode work. Fast-forward and
        # Stream have always run this way; this is what makes one Continue press
        # and one Fast-forward take produce the same samples, which is what
        # fast_forward's "musically identical to pressing Continue N times"
        # claims and, until now, was not quite true of the audio.
        tr = self._continue_step(tr, n_new, int(seed), n_steps, decode="new")
        self._measure_render(tr, tr.window_bounds, tr.window_audio)
        return tr

    def fast_forward(self, track, seed=0, n_steps=SAMPLE_STEPS,
                     max_measures=FF_MAX_MEASURES, idle="refuse"):
        """Run continuations back to back until the active steer is spent.

        Musically identical to pressing Continue N times; the reason it is its
        own entry point is cost. Three savings, in the order they matter:

        1. **One GPU reservation.** On ZeroGPU each @spaces.GPU call forks a
           worker, and the Track crosses that boundary by pickle -- including
           `prefix_audio`, which is the whole song so far and grows every take.
           N presses pay N forks and N round trips of an expanding array. This
           pays one.
        2. **Decode only the new measures.** `_decode`'s cost is per measure in
           the window, but 16 of every 24 are context whose audio is already
           rendered and carried forward, so a take re-decodes them for nothing.
           Fast-forward decodes the 8 new slots and splices them onto the
           carried context with the same 20 ms crossfade `_decode` uses
           internally -- a third of the decode work per take. This is only
           correct because `_decoder_noise` is keyed by absolute song measure
           (not by position in the batch), so a measure gets the same tokenizer
           noise whether it is decoded alone or inside a full window; without
           that determinism the seam would move.
        There used to be a third saving here -- analyse once, at the end --
        and it was wrong. The argument was that `_measure_render`'s librosa
        pass over an intermediate window is waste, because `shift_window` turns
        every DSP signal off window-wide at the top of the next take and the
        back-filled values are dropped before the model ever sees them. True of
        the MODEL, false of the UI: `_continue_step` banks the whole live
        window on its way out (`_bank_window` -> `bank_sync`), and the bank is
        what **Select** reads. Skipping the analysis meant every window that
        scrolled past mid-run was banked holding the flat constants
        `shift_window` seeds new slots with, so a 48-measure fast-forward drew
        one good window at the start, one at the end, and a dead-flat plateau
        between them. So it is analysed per take now, and the saving is gone:
        one librosa pass per take rather than per press.

        Sampling itself is untouched -- the DiT still runs its full schedule per
        take, and that is the floor on how fast this can be.

        `idle` means what it means in ModelBackend.fast_forward: whether an
        unsteered track is refused or run for the full cap."""
        self.to_gpu()
        tr = track.copy()
        self._select_stage(getattr(tr, "stage", self.default_stage))
        if tr.latents is None or tr.window_bounds is None:
            tr = self._ensure_window(tr, n_steps)

        measures, note = plan_fast_forward(tr, int(max_measures), idle=idle)
        if measures <= 0:
            tr.diag = note
            return tr
        for k in range(measures // CONTINUE_MEASURES):
            tr = self._continue_step(tr, CONTINUE_MEASURES, int(seed) + k,
                                     n_steps, decode="new")
            # Per take, not once at the end: the next `_continue_step` banks
            # this window before it shifts, and the bank is what Select plots.
            # See the docstring.
            self._measure_render(tr, tr.window_bounds, tr.window_audio)
        tr.diag = f"{note}\n{tr.diag}"
        return tr

    def stream_forward(self, track, seed=0, n_steps=SAMPLE_STEPS,
                       max_measures=STREAM_LEG_MEASURES, lead_in=True,
                       lead_in_s=0.0):
        """fast_forward, but yielding each take's audio the moment it exists.

        See ModelBackend.stream_forward for the protocol and for what a LEG is.
        Same single GPU reservation, same incremental decode, same per-take
        analysis -- the only difference is that the audio leaves as it is
        made instead of at the end, which is what turns a 60 s wait into music
        that starts playing immediately. At 120 BPM a take is 16 s of audio for
        ~7 s of compute, so generation outruns playback and the stream never
        has to stall.

        THE HOLD-BACK. `_join_window` crossfades the new measures onto the
        carried context, and that rewrites the last CROSSFADE_MS of audio the
        previous take produced. So every chunk stops one crossfade short of the
        end of `tr.audio` and the next chunk picks up from there. That applies
        to the last take of a leg too, precisely because a leg is not
        necessarily the end of anything: the next leg continues from this
        Track, its first take rewrites those samples, and it must be free to
        send the rewritten version. The cost is that the final CROSSFADE_MS
        (20 ms) of a whole press never goes out over the stream; it is in the
        returned Track, which is what gets saved.

        Chunks are therefore disjoint, gapless, and reassemble bit-for-bit into
        `tr.audio[:-CROSSFADE_SAMPLES]` -- across leg boundaries as well as
        within a leg, which is exactly the property the deterministic
        per-song-measure `_decoder_noise` was built to make possible."""
        tr = track.copy()
        if tr.latents is None or tr.window_bounds is None or tr.audio is None:
            self.to_gpu()
            self._select_stage(getattr(tr, "stage", self.default_stage))
            tr = self._ensure_window(tr, n_steps)

        # Lead-in BEFORE to_gpu(): it is already-rendered audio, and on ZeroGPU
        # the checkpoint load behind to_gpu() is the longest silence in the run.
        # Getting the window out first means the listener never hears it.
        emitted = max(0, len(tr.audio) - CROSSFADE_SAMPLES)
        if lead_in:
            lead = len(tr.audio) - len(tr.window_audio)
            # A press after the first runs against a warm worker and only needs
            # enough already-heard audio to cover one take, not a whole window
            # -- see LEAD_IN_WARM_S.
            if float(lead_in_s) > 0:
                lead = max(lead, len(tr.audio) - int(float(lead_in_s) * tr.sr))
            lead = min(lead, emitted)
            emitted = max(lead, emitted)
            yield "audio", np.array(tr.audio[lead:emitted])

        self.to_gpu()
        self._select_stage(getattr(tr, "stage", self.default_stage))
        arm_idle_stream(tr)
        measures, note = plan_fast_forward(tr, int(max_measures), idle="run")
        for k in range(measures // CONTINUE_MEASURES):
            tr = self._continue_step(tr, CONTINUE_MEASURES, int(seed) + k,
                                     n_steps, decode="new")
            cut = len(tr.audio) - CROSSFADE_SAMPLES
            yield "audio", np.array(tr.audio[emitted:cut])
            emitted = cut
            # After the yield, so the listener waits on sampling and decode and
            # not on librosa. Per take for the reason fast_forward gives: the
            # next step banks this window, and the bank is what Select plots.
            self._measure_render(tr, tr.window_bounds, tr.window_audio)
        tr.diag = f"{note}\n{tr.diag}"
        yield "track", tr

    def _continue_step(self, tr, n_new, seed, n_steps, decode="full"):
        """One window slide, in place on `tr`. See continue_track for the shape.

        `decode="new"` decodes only the newly sampled measures and reuses the
        context audio the previous step already rendered; `decode="full"`
        re-decodes the whole window. Both produce the same window_audio to
        within the tokenizer's own numerics -- see fast_forward for why that
        holds.

        Every caller now passes "new". "full" is kept because it is the honest
        description of what the model emits and because render()/region-regen
        still produce a whole window that way, but it is not a live path through
        here: re-decoding the context forces `_finish` into a crossfade that
        deletes 20 ms of already-committed audio. See continue_track, which used
        to be the one caller that asked for it."""
        import torch
        # The single choke point for Continue, Fast-forward and Stream, so the
        # single place that has to insist the window is the end of the song --
        # everything below writes past it and then banks off its front. A no-op
        # unless Edit has seated the window mid-track. See backend.ensure_tail.
        tr = ensure_tail(tr)
        T = tr.cond.T
        n_new = max(1, min(int(n_new), T - 1))
        n_ctx = T - n_new

        # Audio of the measures that survive the slide, cut BEFORE the bank
        # moves the split. Rebased to zero so the bounds are window-relative.
        ctx_audio = ctx_bounds = None
        if decode == "new":
            b0 = tr.window_bounds[n_new]
            ctx_audio = tr.window_audio[int(round(b0 * tr.sr)):]
            ctx_bounds = [b - b0 for b in tr.window_bounds[n_new:]]

        # 1. bank the measures about to fall out of the window, then slide.
        tr = self._bank_window(tr, n_new)
        tr.cond = shift_window(tr.cond, n_new)
        # 1b. steer/walk the style of the new slots only -- after the slide (it
        #     needs the post-shift context) and before the latents are read (it
        #     must not touch the frozen ones). See backend.steer_style.
        steer_diag = apply_style_steering(
            tr, n_ctx, int(seed), self.style_points(),
            axis=self.style_axis_vector(getattr(tr, "style_axis", "")),
            bpms=self.style_bpm())
        # 1c. loop escape, if the last few takes said the music has stopped
        #     moving (see backend.note_loop_check). This reads the level the
        #     PREVIOUS take set and spends it here, so a collapse detected at
        #     take k is broken at take k+1 -- there is nothing to be done about
        #     a window that already exists. The style jump goes last because it
        #     overwrites what the walk just wrote; the two sampler overrides are
        #     applied at the fm_df_sample call below.
        esc_cn, esc_hg, tr.loop_jump_deg = loop_escape(tr)
        jump_diag = apply_loop_escape_style(tr, n_ctx, int(seed),
                                            self.style_points())
        tr.loop_jump_deg = 0.0
        if jump_diag:
            steer_diag = f"{steer_diag}; {jump_diag}" if steer_diag \
                else jump_diag
        # 1d. and last, a Track that authors its own per-measure conditioning
        #     writes the whole window. See backend.apply_cond_hook.
        apply_cond_hook(tr)
        tr.seed = int(seed)
        c = tr.cond

        # 2. the context slots take the OLD window's tail latents verbatim; the
        #    newly exposed slots are placeholders that `keep` marks as free.
        old = torch.from_numpy(tr.latents).to(self.device)
        known_x0 = old.clone()
        known_x0[:, :n_ctx] = old[:, n_new:]
        keep = torch.zeros(1, T, 1, 1, device=self.device).bool()
        # `n_hist` is how much of the context the model may READ; the rest of the
        # window still holds the true latents but is marked free, so the model
        # denoises stand-ins for it from pure noise. Dropped from the OLDEST end so
        # the surviving history stays adjacent to the measures being written. See
        # backend.CONTINUE_HISTORY for why shortening history looks like this
        # rather than like a shorter window.
        n_hist = getattr(tr, "hist_measures", None) or n_ctx
        n_hist = max(1, min(int(n_hist), n_ctx))
        keep[:, n_ctx - n_hist:n_ctx] = True

        g = torch.Generator(device=self.device).manual_seed(int(seed))
        noise = torch.randn(known_x0.shape, device=self.device, generator=g)

        cfg_net_kwargs, uncond_net_kwargs, guidances, cfg_mode, \
            joint_net_kwargs = \
            self._sampling_args(c, getattr(tr, "guidance", None))

        with self._ctx():
            y = fm_df_sample(
                self.diffusion, self.net, known_x0, keep, n_steps,
                net_kwargs=cfg_net_kwargs, uncond_net_kwargs=uncond_net_kwargs,
                guidance=guidances, noise=noise, memory_efficient=False,
                cfg_mode=cfg_mode, t_dist=self.t_dist,
                # `esc_*` are the Track's own ctx_noise/hist_guidance at escape
                # level 0, so this is the plain path unless a loop was caught.
                # hg_freq_scale is deliberately NOT on the escape ladder: it is
                # a standing choice about what omega amplifies, not a rescue.
                ctx_noise=esc_cn, history_guidance=esc_hg,
                hg_freq_scale=getattr(tr, "hist_freq", HISTORY_GUIDANCE_FREQ),
                # Which composition. Continuation only: it is the only path that
                # leans on history guidance at all.
                hg_mode=getattr(tr, "hg_mode", None) or HG_MODE,
                joint_net_kwargs=joint_net_kwargs)

        # Throw the stand-ins away. With n_hist < n_ctx the leading slots came
        # back generated rather than verbatim, and those measures are ALREADY
        # banked and already rendered -- keeping the sampler's version would drift
        # tr.latents away from the audio and feed the drift to the next slide.
        # A no-op when the whole context was kept, since fm_df_sample returns kept
        # slots bit-exact; done before the diagnostic so its ctx in->out check
        # still means what it says.
        y[:, :n_ctx] = known_x0[:, :n_ctx]

        bpm_row = self._bpm_row(c)
        if decode == "new":
            # max_len from the WHOLE window, not the 8 slots being decoded:
            # that is what makes these measures byte-identical to the ones a
            # full-window decode would have produced. See _decode's max_len.
            new_audio, new_bounds = self._decode(
                y[:, n_ctx:], bpm_row[:, n_ctx:], n_steps, return_bounds=True,
                max_len=(60.0 / bpm_row * TARGET_SIG * SR).long().max().item(),
                decoder_noise=self._decoder_noise(
                    n_new, tr.prefix_measures + n_ctx, tr.decode_seed))
            window_audio, bounds = self._join_window(
                ctx_audio, ctx_bounds, new_audio, new_bounds)
        else:
            window_audio, bounds = self._decode(
                y, bpm_row, n_steps, return_bounds=True,
                decoder_noise=self._decoder_noise(c.T, tr.prefix_measures,
                                                  tr.decode_seed))
        tr.latents = y.float().cpu().numpy()
        tr.sr = SR
        # Score THIS take for collapse and arm (or unwind) the escape the next
        # one will spend. Deliberately after tr.latents is set, on the numpy
        # copy rather than the tensor: it is the same numbers, and keeping the
        # detector numpy-only is what lets it be tested without a GPU.
        loop_diag = note_loop_check(tr, tr.latents)
        self._continue_diagnostic(known_x0, y, window_audio, bounds, n_ctx, tr,
                                  "; ".join(d for d in (steer_diag, loop_diag)
                                            if d))
        self._finish(tr, window_audio, bounds,
                     seam="butt" if decode == "new" else "crossfade")
        return tr

    @staticmethod
    def _join_window(ctx_audio, ctx_bounds, new_audio, new_bounds):
        """Splice freshly decoded measures onto carried context audio.

        The same 20 ms crossfade `_decode` puts on every measure join, applied
        at the one seam an incremental decode creates. `ctx_bounds` last entry
        is the end of the context, which the crossfade overwrites -- the first
        new measure starts a crossfade-width BEFORE it, exactly as it would
        have inside `_decode`'s own stitch loop, so dropping it is the point."""
        joined = _crossfade_segments(ctx_audio, new_audio, sample_rate=SR,
                                     crossfade_ms=CROSSFADE_MS)
        off = (len(joined) - len(new_audio)) / SR
        bounds = list(ctx_bounds[:-1]) + [off + b for b in new_bounds]
        return joined.astype(np.float32), bounds

    @staticmethod
    def _continue_diagnostic(known_x0, y, window_audio, bounds, n_ctx, tr,
                             steer_diag=""):
        """Record where a continuation's magnitude blows up, if it does.

        The new measures coming back at clipping volume is ambiguous from the
        outside: it could be inflated LATENTS (a sampler/guidance problem) or a
        sane latent the decoder mishandles. These numbers separate those.
        `ctx` is the frozen prefix and `new` the generated tail; the context
        latents are returned verbatim, so ctx in->out is a free correctness
        check and must read exactly 1.000.

        The result goes on tr.diag rather than stdout. On ZeroGPU this runs
        inside a forked @spaces.GPU worker and its stdout does not reach the
        Space log; the Track is pickled back to the main process, so that is the
        only channel out of here that actually works."""
        try:
            import torch
            with torch.no_grad():
                per = [y[0, i].float().pow(2).mean().sqrt().item()
                       for i in range(y.shape[1])]
                ci = known_x0[0, :n_ctx].float().pow(2).mean().sqrt().item()
                co = float(np.sqrt(np.mean(np.square(per[:n_ctx]))))
                no = float(np.sqrt(np.mean(np.square(per[n_ctx:]))))
                ymax = y[0, n_ctx:].abs().max().item()

            def apk(i):
                seg = window_audio[int(round(bounds[i] * SR)):
                                   int(round(bounds[i + 1] * SR))]
                return float(np.abs(seg).max()) if seg.size else 0.0
            ctx_pk = max(apk(i) for i in range(n_ctx))
            new_pk = max(apk(i) for i in range(n_ctx, len(bounds) - 1))

            # Conditioning shape. A signal that is on for some measures and off
            # for others is off-distribution for this checkpoint (see
            # _net_kwargs) and is the known cause of loud/clipped new measures,
            # so name any signal that is partial rather than just counting.
            on, partial = [], []
            for s in ALL_SIGNALS:
                a = np.asarray(tr.cond.active[s], dtype=bool)
                if a.all():
                    on.append(s)
                elif a.any():
                    partial.append(f"{s}[{int(a.sum())}/{a.size}]")
            cond_line = "cond: on=" + (",".join(on) or "none")
            cond_line += ("  ⚠ PARTIAL (off-distribution): "
                          + ",".join(partial) if partial
                          else "  (uniform ✓)")
            # Where the style trajectory ended up. `move` is the angle from the
            # last CONTEXT measure to the last NEW one, so it reads 0.0° when
            # nothing steered and is the only place a walk is visible at all --
            # a walk leaves no trace in the UI, it just quietly changes what
            # the next eight bars are being asked for.
            st = tr.cond.style
            move = math.degrees(math.acos(float(np.clip(
                unit(st[max(0, n_ctx - 1)]) @ unit(st[-1]), -1.0, 1.0))))
            if steer_diag:
                cond_line += f"\n{steer_diag}"
            cond_line += f"  ·  style ctx→end {move:.1f}°"

            profile = " ".join(f"{v:.2f}" for v in per)
            tr.diag = (
                f"ctx_noise={getattr(tr, 'ctx_noise', None)} "
                f"hist_g={getattr(tr, 'hist_guidance', None)} "
                f"cfg={getattr(tr, 'guidance', None)} n_ctx={n_ctx}\n"
                f"{cond_line}\n"
                f"latent RMS: ctx {ci:.4f}→{co:.4f} "
                f"(×{co / (ci + 1e-9):.3f} — MUST be 1.000) · "
                f"new {no:.4f} (×{no / (co + 1e-9):.2f} vs ctx, "
                f"max|z| {ymax:.2f})\n"
                f"per-measure latent RMS: {profile}\n"
                f"audio peak: ctx {ctx_pk:.3f} new {new_pk:.3f} "
                f"(×{new_pk / (ctx_pk + 1e-9):.2f})"
                + ("  ⚠ CLIPPING" if new_pk >= 0.999 else ""))
        except Exception as e:  # noqa: BLE001
            tr.diag = f"diagnostic failed: {type(e).__name__}: {e}"

    # ---- window bookkeeping -------------------------------------------------
    @staticmethod
    def _bank_window(tr, n_new):
        """Move the first n_new measures of the window into the frozen prefix.

        Cuts at the downbeat the DECODE actually produced (tr.window_bounds),
        not at a recomputed one, so the cut lands exactly on a measure edge even
        though the crossfades make the offsets non-obvious.

        The join has to REPEAT whatever join built the current `tr.audio`,
        which is why it reads `tr.audio_seam` instead of always crossfading.
        The prefix and the head being spliced here meet at exactly the same
        sample as the prefix and the window do in `tr.audio`, so:

          audio was crossfaded -> crossfade(prefix, head) is literally a prefix
            of `tr.audio`, because head starts with the very samples the
            window did.
          audio was butted     -> so must this be, or the prefix grows a
            crossfade shorter than the audio it was cut from.

        Get it wrong and each take blends 20 ms of a continuous signal against
        the 20 ms directly before it -- a flam on the bar line, plus a 20 ms
        drift per take between the audio and everything indexed against it.
        With it right, `tr.audio` is append-only across takes apart from its
        final crossfade, which is the invariant `stream_forward` rides on."""
        # Same first lines as ModelBackend._bank_window, and for the same
        # reason: this is the last moment these measures are live. The trail
        # records where the style went; the bank keeps everything a later edit
        # would need to seat the window back over them.
        record_style_trail(tr, n_new)
        bank_sync(tr)
        split = int(round(tr.window_bounds[n_new] * tr.sr))
        head = tr.window_audio[:split]
        if tr.prefix_audio is None:
            tr.prefix_audio = head
        elif getattr(tr, "audio_seam", "crossfade") == "butt":
            tr.prefix_audio = np.concatenate([tr.prefix_audio, head])
        else:
            tr.prefix_audio = _crossfade_segments(tr.prefix_audio, head,
                                                  sample_rate=SR,
                                                  crossfade_ms=CROSSFADE_MS)
        tr.prefix_measures += n_new
        return tr

    @staticmethod
    def _finish(tr, window_audio, bounds, seam="crossfade"):
        """Store a fresh window render and rebuild the full-song audio.

        The prefix/window seam gets the same 20 ms crossfade as every measure
        join. It needs one: the two sides are the same music but come from
        different tokenizer decodes (its sampler draws fresh noise), so butting
        them together would put a phase discontinuity on a bar line.

        `seam="butt"` is for the incremental decode path only, where the two
        sides are not different decodes at all: `_bank_window` cut the prefix
        out of this very array and `_join_window` kept the rest, so
        prefix ⌢ window is sample-continuous by construction. Crossfading there
        would blend a 20 ms region that has already been blended once, which is
        a small smear bought for nothing."""
        tr.window_audio = window_audio
        tr.window_bounds = bounds
        if tr.prefix_audio is None:
            tr.audio = window_audio
        elif seam == "butt":
            tr.audio = np.concatenate([tr.prefix_audio, window_audio])
        else:
            tr.audio = _crossfade_segments(tr.prefix_audio, window_audio,
                                           sample_rate=SR,
                                           crossfade_ms=CROSSFADE_MS)
        # Where the window and the suffix actually START in the assembled array,
        # read back off it rather than predicted: _crossfade_segments falls back
        # to a plain concatenation when either side is shorter than the fade, so
        # subtracting CROSSFADE_MS would be wrong in exactly that case. See
        # backend.bank_bounds.
        w_off = (len(tr.audio) - len(window_audio)) / float(tr.sr)
        s_off = 0.0
        if tr.suffix_audio is not None:
            # The window is a fresh decode and the suffix is not, so this seam
            # needs the crossfade for the same reason the prefix one does.
            tr.audio = _crossfade_segments(tr.audio, tr.suffix_audio,
                                           sample_rate=SR,
                                           crossfade_ms=CROSSFADE_MS)
            s_off = (len(tr.audio) - len(tr.suffix_audio)) / float(tr.sr)
        bank_bounds(tr, bounds, w_off, s_off)
        # Recorded so the NEXT _bank_window can reproduce this join. See there.
        tr.audio_seam = seam
        return tr

    def vary(self, track, seed, n_steps=SAMPLE_STEPS):
        tr = track.copy()
        tr.seed = int(seed)
        return self.render(tr, n_steps=n_steps)
