"""
backend.py
==========

Model-agnostic backend interface for the interactive music-editing demo, plus a
MockBackend that runs with no GPU / no trained model so the whole UI is
demoable today.

DESIGN
------
The demo's core invariant (from the model's behavior): audio quality is high
exactly when the per-measure conditioning vector is *on-manifold* -- i.e. the
kind of (style, chroma, rms, density, zcr, flatness, bpm) tuple that real audio
would have produced. Every backend method is therefore organized around
producing and editing whole *conditioning trajectories* (T = n_chunks measures),
and only ever calling the low-level model to render audio from a trajectory.

A "Track" is the single source of truth passed between UI and backend:

    Track:
        cond        : Conditioning         # the per-measure signals (T measures)
        seed        : int                  # noise seed -> reproducible render
        bpm_global  : float                # convenience mirror of cond.bpm mean
        n_chunks    : int
        audio       : np.ndarray | None    # last rendered waveform (mono, sr)
        sr          : int

Conditioning holds one row per measure for each signal, matching the training
tensors in MetaConditionalModernDiTV2.forward:

        style     : (T, style_dim)         # 512-d CLAP embedding (audio or text)
        chroma    : (T, 12)
        rms       : (T,)
        density   : (T,)
        zcr       : (T,)
        flatness  : (T,)
        bpm       : (T,)                    # per-measure BPM

To wire up the real model, implement RealBackend (skeleton at the bottom) by
mapping these methods onto your existing functions:

    seed_from_preset / seed_from_text -> build a Conditioning, then render()
    render()                          -> predict_measures / decode_latents
    regenerate_region()               -> NEW inpainting sampler (see notes)
    apply_macros()                    -> move signals along on-manifold dirs
    vary()                            -> re-render with perturbed noise

Nothing in the UI layer imports torch; only RealBackend does. That keeps the
MockBackend importable and runnable anywhere.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import io
import math
import os
import wave
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

# -----------------------------------------------------------------------------
# Constants that mirror the training/model config
# -----------------------------------------------------------------------------
N_CHUNKS = 24            # fixed sequence length (measures) -- matches training

# How many measures "Continue" adds per press, and therefore how many of the
# window's 24 measures are handed to the model as fixed context (24 - 8 = 16).
#
# T IS NOT NEGOTIABLE. get_batch slices `data[idx:idx+n_chunks]` with n_chunks=24
# on every training step, so the DiT has only ever seen a 24-measure sequence.
# The RoPE table is registered at spatial_window*n_chunks = 1536 while the real
# flattened length is T*(spatial_window/patch_size) = T*32 = 768, so a longer T
# would not even raise until T>48 -- it would just quietly extrapolate RoPE into
# positions the model never trained on. Continuation therefore SLIDES a fixed
# 24-measure window instead of growing it.
CONTINUE_MEASURES = 8

# How wide an EDIT may be, and how much untouched music has to sit either side of
# it. The window is still 24 -- that is the model's whole sequence -- but the
# editable span inside it is capped so that the measures being rewritten are
# never the whole window.
#
# The point is what the model reads while it writes. `fm_df_sample` freezes the
# unselected measures at their real latents and denoises only the selected ones,
# with attention running both ways, so frozen neighbours are the entire mechanism
# by which a rewrite stays part of the piece. A selection that filled all 24
# slots would leave none, and the "inpaint" would be an unconditioned render
# wearing an inpaint's name -- which is exactly the bug this cap removes: the
# Edit tab used to hand a full-width selection to `render`, a different sampler
# that samples from pure noise and never reads the past.
#
# Four each side rather than one or two: a measure is the unit the conditioning
# is written in and the unit attention is modulated in, so one measure of guard
# is one token of context. Four gives the model a phrase to join onto at each
# end. It is also half of CONTINUE_MEASURES, so an edit is never narrower in
# context than a continuation.
#
# At the START or END of a song the guard is whatever there is room for -- the
# window clamps and the padding lands on the side that has any. That is not a
# compromise: before measure 0 there is nothing to be coherent with.
EDIT_GUARD = 4
EDIT_MAX_MEASURES = N_CHUNKS - 2 * EDIT_GUARD    # 16

# How many of those 16 context measures the model is actually allowed to READ as
# history. None == all of them, which is the shipped behaviour; 8 keeps only the
# half nearest the new measures.
#
# The window does not shrink -- T is not negotiable (above) -- so the dropped
# measures do not leave the sequence, they stop being history and become tokens to
# generate. That is the reference implementation's own semantics, not a fudge:
# DFoT's HistorySegment.to_noise_levels sets every unselected token to noise level
# 1.0 rather than deleting it, so "shorter history" and "these tokens are pure
# noise" are the same statement. They are freed from the OLDEST end, so the history
# that remains is the half adjacent to what is being written and the RoPE distance
# from history to new measures is unchanged.
#
# Free, as experiments go: same window, same sequence length, same number of DiT
# passes. The generated stand-ins for the dropped measures are thrown away -- they
# are already banked and already rendered, and _continue_step writes the true
# latents back over them before anything downstream sees the window, so neither the
# audio nor the next slide's context can drift from what was committed.
_ch = os.environ.get("CONTINUE_HISTORY", "").strip()
CONTINUE_HISTORY = int(_ch) if _ch else None

# Noise level the already-written measures are held at while the new ones are
# sampled.
#
# NOT 0.0, even though clean history is what the Diffusion Forcing reference
# implementation ships. That implementation trains on DISCRETE noise indices
# drawn uniformly over [0, K), so index 0 -- perfectly clean -- is an ordinary
# training input for it. This model does not: FM.loss draws
#
#     t = torch.sigmoid(torch.randn(B, T))
#
# a continuous logit-normal, which has ZERO density at 0 and never gets near it.
# Over a 300k-step run (grad_accum 4 x batch 32 x 24 measures ~= 9.2e8 draws) the
# smallest t the model ever saw is about 0.0025, and t < 0.001 never occurred at
# all. t=0.0 is therefore off the end of the training support, and t is not a
# passive tag: it feeds the per-measure AdaLN modulation (`t = t + style + bpm`),
# so an out-of-distribution value on the 16 context measures corrupts their
# features and, through attention, the free measures with them.
#
# 0.05 is the smallest value that is comfortably supported -- roughly 1.5e6
# training draws landed below it -- while leaving the context essentially intact
# (alpha=0.95, sigma=0.05). Draws below each level over that run:
#
#     t < 0.001              0        t < 0.05      1,490,876
#     t < 0.01           1,993        t < 0.10     12,904,433
#
# Raising it toward 0.1-0.2 is also well supported and is the knob to reach for
# if repeated Continues drift or harden into a loop (Diffusion Forcing's
# stabilization result: a model that half-distrusts its own past compounds error
# more slowly). Above ~0.3 the context genuinely washes out.
#
# BUT: the per-measure counts above are MARGINAL, and the marginal is the easy
# half of the question. A continuation pins 16 of 24 measures near-clean while
# the other 8 sit near t=1, and because FM.loss draws every measure's t
# INDEPENDENTLY that joint configuration has effectively never been trained on:
#
#     P(measure t < x)      share of a 24-measure window at that level
#       x=0.05   0.161%                0.04 measures
#       x=0.1    1.400%                0.34
#       x=0.2    8.289%                1.99
#       x=0.3   19.852%                4.76
#       x=0.5   50.007%               12.00
#
# so P(16 measures below 0.05 in one window) ~ 1e-45. The model has seen plenty
# of INDIVIDUAL near-clean measures and essentially zero windows that are mostly
# near-clean. Sampling logit-normal is standard for flow matching, but it is a
# poor fit for diffusion-forcing rollout, which is exactly why the original DF
# work samples noise levels per token from a distribution that routinely
# produces clean-past/noisy-future windows. Whether this model generalizes to
# the configuration anyway is an empirical question -- hence the UI slider.
#
# MEASURED, so do not re-derive it: sweeping the slider end to end (0.05 through
# 0.5) changed nothing about the "new measures come back at clipping volume"
# failure. At 0.5 the context is half noise, which is squarely in-distribution
# even jointly, so the joint-configuration argument above is real but was NOT
# what made continuations bad -- that was partial per-measure CONDITIONING (see
# shift_window). Treat this knob as what DF says it is: a drift/robustness dial.
#
# THE DEFAULT IS 0.2, NOT THE 0.05 THAT LOOKS "SAFEST". Listening to long runs
# is what settled it: 0.05 holds the context so rigidly that repeated takes
# harden into a loop, and 0.2 keeps the take recognisably continuous while
# giving each one enough freedom not to restate the last. That is also the
# direction the table above predicts -- at 0.05 a context measure sits where
# only 0.16% of training draws landed, so a 16-measure context held there is a
# window the model effectively never saw, while 0.2 is 8.3% per measure and
# puts about 2 measures of a real training window at or below it. Above
# ~0.3 the context genuinely does wash out and continuity goes with it.
CONTINUE_CTX_NOISE = 0.2

# What region-regeneration holds ITS context at, deliberately not the same
# number. Everything above is an argument about ROLLOUT: a continuation feeds
# on its own output, so a context it half-distrusts compounds error more slowly
# and takes stop restating each other. Regenerating a region in place is not a
# rollout -- the context is real material the user wants kept, there is no
# feedback loop to damp, and raising the noise buys nothing while blurring the
# measures that were supposed to survive. So this stayed at the old default
# when CONTINUE_CTX_NOISE moved to 0.2, and region-regen behaves as it did.
REGION_CTX_NOISE = 0.05

# How far a Regenerate re-rolls the measures it rewrites: 0 is the take you
# already have, 1 is a take written from nothing but the conditioning and the
# frozen neighbours.
#
# This is the partial-noise restart (SDEdit, in the flow-matching form). Instead
# of starting the selected measures at pure noise, `fm_df_sample` starts them at
#
#     x_s = alpha(s) * their_existing_latents + sigma(s) * noise
#
# and runs the schedule from t=s down to 0 rather than from t=1. What carries
# the identity of the passage is then the INITIALISATION, not the context -- so
# it keeps working when the selection is wide and the guards are thin, which is
# where "regenerate this and have it still be this bit of the song" used to fall
# apart. The dial is continuous and means what it says: at 0.2 you get the same
# phrase played slightly differently, at 0.9 a different phrase in the same slot.
#
# The maths makes the two ends exact rather than approximate. alpha(1)=0 and
# sigma(1)=1, so s=1 is bit-for-bit the old pure-noise behaviour and the
# truncated schedule is the whole schedule. alpha(0)=1 and sigma(0)=1e-5 with
# zero steps left to take, so s=0 returns the existing latents. Nothing special-
# cases either end.
#
# In distribution, too, which is the part that makes it more than a trick here.
# Training draws t = sigmoid(randn(B, T)) -- one INDEPENDENT noise level per
# measure -- so a window reading (0.05, 0.05, 0.05, 0.05, 0.7, ..., 0.05) is an
# ordinary draw. Plain SDEdit has to denoise everything at one scalar t; this
# model was built to hold the guards clean while the middle descends.
#
# 0.7 is a starting value chosen by argument, not by listening: below ~0.5 a low
# start fights a large conditioning change (the old audio pulls back toward the
# old curves), above ~0.85 it is close enough to 1.0 not to be worth the dial.
# It wants ears on a GPU.
REGEN_STRENGTH = 0.7

# Vanilla history guidance (HG-v) weight for Continue. CFG steers toward the
# conditioning signals; this steers toward THE PAST. It composes with the signal
# axis in the usual NESTED way -- signal CFG relative to the with-history base,
# history relative to the bare unconditional:
#
#     u_n + omega*(u_h - u_n) + sum_i g*(c_i^h - u_h)
#       = v(with history, full CFG) + (omega - 1) * (u_h - u_n)
#
# and NOT `v_none + omega*(v_h - v_none)` with both legs fully CFG'd, which
# looks equivalent and is off by a factor of (1 - n*g). See fm_df_sample.
#
# 1.0 is exactly off. It is on by default here because continuation is
# UNDERSPECIFIED by construction: shift_window drops chroma/rms/density/zcr/
# flatness, since nobody knows what an unwritten bar contains, which leaves the
# frozen context as very nearly the only thing determining the new measures.
# Worth making the model actually lean on it.
#
# 1.5 is where DFoT ("History-Guided Video Diffusion") reports best FVD; they
# show 3.0 as visibly over-driven. Costs one extra DiT pass per sampling step.
# See real_backend.fm_df_sample for how "no history" is expressed.
HISTORY_GUIDANCE = 1.5

# FRACTIONAL history guidance (HG-f): the noise level the AMPLIFIED leg reads
# the context at. None == plain HG-v, which is the shipped default.
#
# HG-v amplifies the direction from "no past" to "the past exactly as the model
# is reading it". That direction says COPY, and a rollout that already collapses
# into a riff is being pushed along it once per take. HG-f amplifies the
# direction to a BLURRED past instead, so what gets boosted is the coarse shape
# of the last 16 measures -- groove, register, harmony -- and not the notes.
# DFoT's abstract splits the credit exactly this way: vanilla buys quality and
# temporal consistency, the advanced variant buys motion dynamics and long
# rollout. Low dynamics over a long rollout is the failure we have.
#
# NOT the same lever as raising CONTINUE_CTX_NOISE, and this is the point. That
# blurs the context the base prediction reads, so it buys freedom by spending
# continuity -- which is why LOOP_CTX_NOISE has to unwind. This leaves the base
# prediction reading the context at CONTINUE_CTX_NOISE exactly as before and
# blurs it only inside the guidance difference. The join stays tight; only the
# copy pressure comes off.
#
# 0.4 is the value to try first, and it is the paper's, not a guess. DFoT C.7 on
# sliding-window rollout -- the one setting that is our Continue/Stream rather
# than their one-shot benchmark -- says the default scheme there is HG-f "which we
# find to be extremely stable during long rollouts", at omega = 4 with k_H = 0.4,
# "chosen to ensure optimal stability". Their sweeps put the useful band at
# 0.3-0.8: 0.8 is the Kinetics setting, picked for "sufficient dynamics" rather
# than stability, and k_H = 0 is by definition vanilla. Pair it with the omega it
# was measured at -- HG-f at 0.4 with our 1.5 is a weaker lever than either.
#
# Their k_H is a fraction of a DDIM/cosine schedule and ours is flow-matching t
# with alpha = 1-t, so the number does not transfer to the decimal; both mean
# "this far toward pure noise", which is enough for the band to carry. 0.4 also
# sits just under the 0.5 median of the sigmoid(randn) draw the model trained on,
# so the blurred context is squarely in-distribution, and it is far enough above
# the 0.2 CONTINUE_CTX_NOISE default for the two legs to actually differ.
#
# Off by default because pinning the negative branch at t=1.0 (see fm_df_sample)
# already changed what omega does, and the two want hearing one at a time.
_hgf = os.environ.get("HISTORY_GUIDANCE_FREQ", "").strip()
HISTORY_GUIDANCE_FREQ = float(_hgf) if _hgf else None

# WHICH history-guidance composition to use. "nested" is ours and the default;
# "dfot" is the reference implementation's flat weight table, ported so the two
# formulations can be A/B'd rather than argued about. Continue/Stream only.
#
# They are different formulations, not different tunings. Ours has two axes:
# signal CFG at `guidance` against the with-history base, then history guidance at
# HISTORY_GUIDANCE against the bare unconditional. Theirs has ONE -- the history
# and the external conditioning drop together on the same leg, under one omega --
# so their published omega = 4 is doing the work our 3.0 and 1.5 do separately.
#
# Consequences when it is on: the guidance slider stops doing anything, omega is
# the only scale, and omega = 1.0 degenerates to a single plain conditional pass
# with no guidance at all. It is also cheaper -- 2 passes, or 3 with HG-f,
# against nested's 4. See fm_df_sample's hg_mode docstring for the tables.
HG_MODE = os.environ.get("HG_MODE", "nested").strip() or "nested"

# ---- loop collapse ----------------------------------------------------------
# The failure a long radio press actually has. Not divergence, not drift into
# noise -- COLLAPSE: after a few minutes the model settles onto a short riff and
# repeats it verbatim, forever. See `loop_score` for the detector and
# `loop_escape` for what is done about it.
#
# Why it happens is structural rather than a bug. During a stream the ONLY
# time-varying conditioning is the style vector (shift_window turns every DSP
# signal off window-wide, and bpm is carried forward), history guidance is
# explicitly steering TOWARD the frozen past, and 16 of the window's 24 measures
# are frozen context -- so a loop of period <= 16 fits entirely inside material
# the model is being asked to be consistent with. It cannot see that it is
# repeating, and everything in the sampler is pushing it to keep doing so.
#
# Centered cosine similarity between two measures' latents: 1.0 is "the same
# measure again", and centering across the window first removes the constant
# this-is-one-song component that would otherwise put a high floor under every
# pair. Distinct measures of real music land well under this even when they
# share a groove. Deliberately set high -- a false positive audibly disrupts
# music that was fine, a false negative just leaves the status quo.
LOOP_COS = float(os.environ.get("LOOP_COS", "0.90"))

# Consecutive takes that must score over LOOP_COS before anything is done. One
# window scoring high is ordinary music: a four-bar phrase repeated three times
# IS a loop, and the correct response to it is to enjoy it. A riff still
# repeating two takes (16 measures) later is the pathology.
LOOP_SUSTAIN = int(os.environ.get("LOOP_SUSTAIN", "2"))

# Takes below threshold before the escape unwinds one level. Asymmetric with
# LOOP_SUSTAIN on purpose: slow to intervene, slower still to let go, because
# relaxing the instant the score dips lands straight back in the same basin.
LOOP_RELAX = int(os.environ.get("LOOP_RELAX", "3"))

# How hard to noise the frozen context while escaping, per level. THE PRIMARY
# LEVER, and it goes most of the way on the first level rather than creeping.
#
# The reason it can be pushed this hard is the opposite of the intuition. This
# value is a LITERAL NOISE LEVEL t on the context measures (fm_df_sample holds
# them there for the whole run while the free measures walk t=1->0), and
# training draws t per measure as sigmoid(randn) -- MEDIAN 0.5. So 0.6 sits in
# the fat middle of what the model was trained on, while the 0.2 default is out
# at P(t<0.2) ~ 8% and a bit-exact context is P(t<0.05) ~ 0.16%. Raising this
# moves the sampler TOWARD the training distribution. The cost is continuity,
# not stability: the new bars are following a blurrier past, so the seam is the
# thing to listen for, and it is why this unwinds rather than staying put.
LOOP_CTX_NOISE = (0.45, 0.60, 0.60)

# History guidance while escaping. Left alone at level 1 so the first
# intervention changes exactly one thing; pulled to 1.0 from level 2 because
# steering hard toward a context that has just been heavily noised is
# self-contradictory -- omega scales "what knowing the past buys you", and at
# t=0.6 the past is buying much less. 1.0 cancels the term exactly and saves
# two DiT passes per step, so it is a real configuration, not a hack value.
LOOP_HIST_GUIDANCE = (None, 1.0, 1.0)

# Style embedding width. The DiT is now conditioned on CLAP embeddings
# (clap.CLAP proj_dim=512), NOT the old 128-d contrast.py vectors. Both stage
# checkpoints carry style_dim=512 in their model_args; keep this in sync.
STYLE_DIM = 512

# ---- style steering ---------------------------------------------------------
# THE ONE HARD CONSTRAINT. A Continue writes CONTINUE_MEASURES (8) measures and
# everything before them is a FROZEN latent whose audio has already played, so
# no style change can begin earlier than 8 measures from now. That bounds
# LATENCY -- how soon a change starts -- and nothing else. It does NOT bound
# DURATION: a change spread over 64 measures is just a change that advances an
# eighth of the way on each of eight consecutive Continues, and the only thing
# needed to express that is somewhere to remember how far along it is. Track
# carries `style_anchor` / `style_done` / `style_span` for exactly that, so a
# steer survives across Continues (and across Undo, which snapshots them).
#
# The unit of a "full" style change. Two unrelated recordings sit roughly 40-70
# degrees apart in CLAP space, so 90 degrees is a safe stand-in for "somewhere
# else entirely" -- deliberately a slight over-estimate, so a control calibrated
# in these units under-promises. Every user-facing pace number is measures per
# this much arc, which is the only phrasing of it that is checkable by ear.
STYLE_FULL_CHANGE_DEG = 90.0

# Default measures for a typed steer to complete in. Two Continues: long enough
# to hear it arrive rather than cut, short enough that you find out whether the
# prompt did anything without committing a minute of audio to it.
STYLE_SPAN = 16

# Default measures for one LEG of an automatic walk -- one journey to one
# destination, worth about STYLE_FULL_CHANGE_DEG of arc. 96 is
# twelve Continues, ~3 minutes at 120 BPM -- style enters through
# `t = t + style + bpm`, modulating every block's AdaLN, so it is the
# highest-leverage channel in the model and a big jump between adjacent windows
# reads as an edit rather than a development.
STYLE_PACE = 96

STYLE_WALK_MODES = ["off", "wander", "drift"]

# How many of the walk's most recent destination SONGS a new leg refuses to go
# back to, when refusing still leaves it somewhere to go.
#
# `pick_reference`'s `avoid` already blocks the previous destination, but only
# the POINT: the pool is 88 measures drawn from five recordings, and a song's
# own measures sit ~12 degrees apart, so "not that point" leaves the whole song
# it came from available on the very next leg. Combined with `cross_cluster`
# (which only bars the song the walk is standing IN), the reachable set after
# arriving at B is "anything but B" -- and A is in it. A<->A<->B is a legal
# walk that visits two of five songs forever.
#
# Measured over 400 seeds, distinct songs reached in 8 legs (of 5 possible):
#
#     leg seed advancing, point-weighted draw      4.35
#     + uniform over clusters                      4.46
#     + this instead, at 2                         4.64
#     + both                                       4.72
#
# 2 rather than 3 because at five songs a memory of 3 empties the filter often
# enough that the "only if something survives" fallback makes most of the
# choices anyway, which would make the knob read as stronger than it is.
WANDER_RECENT = 2

# Hard ceiling on what ONE Fast-forward press will generate. The binding
# constraint is the ZeroGPU wall clock, not anything musical: the whole run
# happens inside a single @spaces.GPU call (that is the point of it -- see
# RealBackend.fast_forward), and that call has one duration budget.
#
# This is a CEILING, not the length. The length comes from whatever is being
# run out -- the steer's span, or the leg's pace -- and plan_fast_forward takes
# the smaller of the two. The ceiling is set to STYLE_PACE deliberately: a leg
# at the default pace is the single most common thing anyone asks Fast-forward
# for, and at 64 it could not finish one, so the default settings guaranteed a
# press that stopped short of the only finish line on offer. 96 measures is 12
# takes; at the measured ~10 s for a full Continue, less the two thirds of the
# decode a fast-forward take skips, that is ~90 s inside a 240 s reservation --
# still most of a reservation again in margin for a cold model load. A longer
# pace just needs a second press, and the status line says so rather than
# silently stopping short.
FF_MAX_MEASURES = STYLE_PACE

# ---- streaming ------------------------------------------------------------
# Stream is open-ended: it runs until the listener stops it, not until a
# measure count runs out. It gets there by generating in LEGS rather than in
# one long call, because the GPU reservation is finite and no single call can
# outlive it. Each leg is one reservation's worth of takes; the app loop
# (app.on_stream) starts the next leg as soon as the buffer needs it, so
# "forever" is a property of the loop and not of any one allocation.
#
# One leg = 4 takes = ~64 s of music at 120 BPM for ~28 s of compute, which
# leaves the 240 s reservation almost entirely to spare -- deliberately, since
# the first leg of a cold worker pays for the checkpoint load as well. Bigger
# legs would amortise the per-leg re-allocation better; smaller ones would
# track the headroom more tightly and waste less when the listener stops
# mid-leg. This is the knee.
STREAM_LEG_MEASURES = 32

# Seconds of generated-but-unplayed audio to keep in hand. Between legs the app
# sleeps until the buffer has drained back to this, which is what stops
# generation racing ~2.3x ahead of playback and burning GPU on music nobody
# reaches. It has to cover the gap where nothing is being produced: the
# re-allocation at the start of the next leg plus that leg's first take. 45 s
# is roughly double the measured worst case for the two together, and is also
# about what the opening window buys (24 measures ~ 48 s at 120 BPM), so the
# steady state feels the same as the start.
STREAM_HEADROOM_S = 45.0

# Safety ceiling on one Stream press, in NEW measures. Not a musical limit: the
# Track carries the whole song as raw audio twice over (`audio` and
# `prefix_audio`), it is deep-copied per leg and pickled across the ZeroGPU
# process boundary in both directions, and all of that grows linearly with the
# run. 1024 measures is ~34 minutes at 120 BPM and ~260 MB of float32 -- past
# any plausible session and short of anything that would take the Space down.
# Reaching it ends the press cleanly, with the take kept, rather than dying.
STREAM_MAX_MEASURES = 1024

# Seconds of already-heard audio a SECOND-or-later press replays before it has
# anything new. The first press sends the whole current window (see
# stream_forward's `lead_in`), because a cold ZeroGPU worker has to load
# multi-GB checkpoints before its first take and ~48 s of music is what covers
# that. Every press after it is running against a warm process -- the checkpoint
# is already resident, the first take is back in ~7-15 s -- so the full window
# is 30-odd seconds of insurance nobody needs, and it is not free: a stream is a
# new media element that starts at 0:00, so replaying 48 s of music the listener
# just heard is indistinguishable from the radio having restarted. 16 s covers
# the warm first take with margin and is short enough to read as a join.
LEAD_IN_WARM_S = 16.0

# Named directions for `drift`, as CONTRASTING PROMPT PAIRS rather than single
# prompts. The axis is `unit(embed(plus) - embed(minus))`, and the subtraction
# is the entire point: CLAP's text and audio towers are offset from each other
# (the "modality gap"), the offset is approximately a constant vector, so it
# cancels in a difference. A single text embedding therefore cannot be used as a
# heading for an audio-derived point, but the difference of two can -- this is
# the same trick StyleCLIP's global directions and CLIP-space attribute editing
# run on. "Approximately" is doing real work in that sentence; if a named axis
# moves the music somewhere unrelated to its label, that assumption failing is
# the first suspect.
#
# Each entry names BOTH poles, so the UI can offer a signed direction ("→
# sparser" and "→ denser") instead of an axis plus a sign toggle.
STYLE_AXES = [
    ("sparser", "denser",
     "sparse minimal arrangement, few instruments, lots of space",
     "dense busy arrangement, many layered instruments playing at once"),
    ("darker", "brighter",
     "dark muffled low-end heavy recording",
     "bright airy treble-forward recording"),
    ("calmer", "more intense",
     "calm gentle relaxed quiet playing",
     "intense aggressive loud energetic playing"),
    ("more acoustic", "more electronic",
     "acoustic instruments, upright bass, brushed drums, live room",
     "electronic synthesizers, drum machine, programmed beat"),
    ("cleaner", "more distorted",
     "clean clear undistorted hi-fi recording",
     "distorted saturated lo-fi noisy recording"),
    ("warmer", "colder",
     "warm rich analog tape sound",
     "cold clinical digital sound"),
]

# What the drift axis dropdown offers. "" == pick a random axis between two
# presets, which needs no text tower but is limited by how many presets there
# are (see ModelBackend.style_points).
STYLE_AXIS_RANDOM = "random (between presets)"


def style_axis_labels() -> list[str]:
    """Signed direction names for the drift dropdown, both poles of each axis."""
    out = [STYLE_AXIS_RANDOM]
    for plus, minus, _, _ in STYLE_AXES:
        out += [f"→ {plus}", f"→ {minus}"]
    return out


def resolve_axis_prompts(label: str):
    """('→ sparser') -> (plus_prompt, minus_prompt), or None for the random axis."""
    label = (label or "").strip()
    if not label.startswith("→"):
        return None
    want = label[1:].strip()
    for plus, minus, pp, mp in STYLE_AXES:
        if want == plus:
            return pp, mp
        if want == minus:
            return mp, pp
    return None

TARGET_SIG = 4           # beats per measure (from train.py TARGET_SIG)
SR = 16000               # render sample rate (tokenizer decodes at 16k)

# Every audio join in the decode path -- measure to measure inside _decode,
# window to window across a Continue, context to new measures in an incremental
# decode -- uses this one crossfade. It is a shared constant rather than a
# literal at each call site because streaming depends on the exact number:
# a crossfade is the ONLY thing that ever rewrites audio that has already been
# produced, and it only ever touches the last CROSSFADE_MS of it, so the stream
# holds back exactly that much and no more. Change it in one place or the
# hold-back stops matching what gets rewritten.
CROSSFADE_MS = 20
CROSSFADE_SAMPLES = int(SR * CROSSFADE_MS / 1000)

# BPM bounds. The model embeds bpm with nn.Embedding(350) after round+clamp(0,349),
# and the decoder turns bpm into a measure duration, so a degenerate value makes
# unusably long/short measures.
#
# The ceiling is the TRAINING range, not a judgement about what a person should
# be dragging a slider to, and those came apart once: this was 220, while the
# shipped presets sit at folded medians of 176, 231, 235, 273 and 286 -- three
# of five above the old cap and so unreachable by hand, and unreachable by a
# tempo steer that clamped to the control's range instead of the model's. One
# number for both. Note these are per-MEASURE rates (60 * TARGET_SIG /
# measure_duration), not tapped-foot tempo, which is why they run high.
BPM_MIN, BPM_MAX = 40.0, 350.0

# ---- the tempo GRID ---------------------------------------------------------
# Only some tempi exist as far as this model is concerned, and the ones that do
# are not evenly spaced. Two independent reasons, one grid.
#
# 1. The conditioning is a LOOKUP TABLE. diffusion_forcing.py:1978 is
#    `nn.Embedding(350, hidden)` indexed by `clamp(round(bpm), 0, 349)` -- 350
#    separately-learned rows with no continuity prior between row 176 and row
#    177, and line 2042 excludes the table from weight decay, so a row training
#    never visited keeps its random init forever. An unvisited row is not a
#    neutral vector, it is a full-magnitude noise vector. Which rows were
#    visited is readable off `stage2.pt` rather than guessed -- init is
#    std=0.02, so an untouched row has norm 0.02*sqrt(768) = 0.554, and over
#    rows 40..349:
#
#        reachable by this grid (145 rows)   mean norm  1.229
#        not reachable         (165 rows)    mean norm  0.566   <- init, to 2dp
#
#    For scale, over the same rows `mean ||row(b+1) - row(b)|| = 1.102` while
#    `mean ||row(b) - null_bpm|| = 1.002` -- stepping the tempo by one BPM moves
#    the conditioning token further than deleting the tempo condition entirely.
#
# 2. The labels were quantised before the table ever saw them, which is WHY
#    only some rows got visited. The analysis pipeline that produced the
#    training bpm measured each measure in whole 320-sample frames (0.02 s at
#    16 kHz), so it could only ever emit
#
#        bpm = 60 * TARGET_SIG / (320/SR * k) = BPM_GRID_NUM / k
#
# DO NOT re-derive this from the tokenizer. The WavTokenizer hop is 512, not
# 320, and an earlier version of this comment claimed the two were the same
# number and added an assertion to that effect, which took the Space down. They
# are unrelated: 320 is the ANALYSIS hop baked into the labels, 512 is what the
# decoder happens to use, and a measure's `target_samples` (= 320k for on-grid
# bpm) is an exact integer either way. The grid is a fact about the training
# data, so it is verified against the training data, twice over:
#
#   - all 20 distinct bpm values across the five demo presets sit on
#     BPM_GRID_NUM/k to within 2.8e-6 (float32 storage noise), while the two
#     next-most-plausible numerators, 7500 and 6000, are off by half a frame;
#   - of stage2.pt's 182 on-grid rows 170 are trained, and of its 168 off-grid
#     rows 3 are. Nothing but the grid explains that split.
#
# The quantum grows as the SQUARE of the tempo -- 0.3 BPM at 60, 1.2 at
# 120, 2.6 at 176, 4.0 at 220 -- so HOW MUCH THIS MATTERS DEPENDS ENTIRELY ON
# THE TEMPO. Fraction of integers in a band that hit a trained row:
#
#     bpm  70-140    90%        bpm 150-220    28%
#     bpm  60-150    88%        bpm  40-220    66%
#
# Slow music barely notices; the presets, which sit at 176-286, are the problem.
# See snap_bpm for what snapping costs and buys at each end.
BPM_ANALYSIS_HOP = 320      # the LABEL pipeline's hop. Not the tokenizer's.
BPM_GRID_NUM = 60.0 * TARGET_SIG * SR / BPM_ANALYSIS_HOP     # 12000.0
# k is frames per measure, so the bounds invert: the SMALLEST k is the FASTEST
# tempo. Rounded inwards, so a snapped value is always inside [BPM_MIN, BPM_MAX]
# and the clip that used to do that job has nothing left to do.
BPM_K_MIN = int(math.ceil(BPM_GRID_NUM / BPM_MAX))           # 35 -> 342.9 BPM
BPM_K_MAX = int(math.floor(BPM_GRID_NUM / BPM_MIN))          # 300 -> 40.0 BPM

# Tempo destinations are octave-folded into [ref/FOLD, ref*FOLD) before use.
# Two jobs, one rule. The presets' per-measure bpm is beat-TRACKED, and the
# tracker doubles: preset 4 reads 273 for eighteen measures then 522-571 for the
# last six, preset 5 scatters 571s among 286s, preset 8 has 462-480 among 235s.
# Folding collapses their per-measure spread from std 118/98/88 to 4.4/17/2.8,
# which is the entire difference between tempo data and noise. The same fold
# then bounds a LEG: whatever it is aiming at ends up within a factor of two of
# where the music currently is, so a wander can change the tempo but cannot
# teleport it, and no destination survives as a 2x lurch.
BPM_STEER_FOLD = 2.0

# ---- the OTHER tempo-follow, the one shaped like the training data ----------
# `steer_bpm` ramps, and its docstring records the measurement that says a ramp
# is the wrong shape: real per-measure tempo is white jitter about a constant,
# with a linear trend explaining 0.0-1.8% of the variation. `steer_bpm_stepped`
# is the in-distribution alternative -- ONE constant per wander leg, plus white
# jitter, with the constant stepping at the leg boundary.
#
# Relative std of the jitter. Measured on the five demo presets' folded
# per-measure bpm: deviation-from-median rel-std 1.18 / 1.46 / 1.58 / 1.61 /
# 7.25 % (median 1.58), and a white-noise std estimate from successive
# differences of 1.51 %. 1.5 % is that number, not a taste.
BPM_STEP_JITTER = float(os.environ.get("BPM_STEP_JITTER", "0.015"))

# Most a leg may move the tempo, as a ratio. A step cannot be made
# in-distribution -- the window straddling it fits a trend almost perfectly
# whatever its size -- so the choice is how MUCH tempo the two straddling
# windows carry and how often they happen. Averaged over 400 jitter draws, the
# step window's |drift| across 24 measures is 12.7 BPM at 1.05, 25.5 at 1.10,
# 38.3 at 1.15 and 158.6 at 1.62 (the whole journey at once), against real
# songs' 6.4 BPM worst case. The other side of it: with the default 96-measure
# leg only 2 of every 12 windows straddle a step, and the remaining 10 score
# 4.6% var-by-drift with a jitter index of 1.03 -- indistinguishable from a real
# song. 1.10 is the compromise, env-tunable because the right answer here is a
# listening question and this exists to be swept.
BPM_STEP_MAX = float(os.environ.get("BPM_STEP_MAX", "1.10"))

# How long one constant is HELD, in measures. This used to be "one wander leg",
# and that was wrong for a reason worth recording, because it looked right:
# a leg is the unit the style travels in, so it seemed like the unit the tempo
# should travel in too. It made the tempo movement per leg independent of how
# long the leg was, so the LONGER the leg the further behind the tempo arrived
# -- 48% off the destination's tempo at the default 96-measure pace against 36%
# at 16, i.e. exactly backwards. Arriving at a 70-BPM ambient reference at 103
# BPM is not "taking the tempo along" by any reading.
#
# Tying the cadence to measures fixes the direction: a long leg gets more steps
# and therefore arrives. One window is the floor, and the reason is structural
# -- a constant held for less than N_CHUNKS measures is in no window on its own,
# so EVERY window straddles a step and the in-distribution shape this mode
# exists for is gone. Measured at pace 16 (a leg shorter than a window) that is
# already visible: var-by-drift 66% against 12% at pace 96.
#
# The corollary is a limit, not a bug: a pace below N_CHUNKS cannot have both.
# There is no way to move a tempo 1.6x inside two takes without the straddling
# window being a ramp, and the ramp mode is the one that does that honestly.
BPM_STEP_HOLD = int(os.environ.get("BPM_STEP_HOLD", str(N_CHUNKS)))

# The 1-D conditioning signals the user controls directly (one control each).
# chroma (12-d) and style (128-d) are not scalar curves and are handled
# separately (key/mode and preset-morph respectively).
EDITABLE_SIGNALS = ["rms", "density", "zcr", "flatness", "bpm"]

# Human-facing labels + a sensible default value used when a shape needs a
# baseline (e.g. so a fresh "ramp" spans a musical range for that signal).
SIGNAL_LABELS = {
    "rms": "energy (rms)",
    "density": "density",
    "zcr": "brightness (zcr)",
    "flatness": "flatness",
    "bpm": "tempo (bpm)",
}

# The parametric shapes a user can assign to a signal over a measure range.
# Each fills all measures in the range from a handful of parameters, so the user
# never edits per-measure values by hand.
SHAPES = ["constant", "ramp", "envelope", "periodic", "step", "random"]


# -----------------------------------------------------------------------------
# Data model
# -----------------------------------------------------------------------------
# All conditioning channels the model accepts, in the order the model's
# unconditional_mask expects them. Each can be toggled on/off PER MEASURE.
ALL_SIGNALS = ["style", "chroma", "bpm", "rms", "density", "zcr", "flatness"]

# What the STAGE-1 DiT actually consumes. Stage 1 keeps `local_embedder` frozen
# at its ControlNet zero-init, and that block is the ONLY path the 21-channel
# presence-augmented signals (chroma + the four DSP scalars) take into the
# network -- `x = x + local_embedder(c)` adds exactly zero. Style and bpm go in
# through their own embedders, so they are the only two that do anything. Their
# per-measure `active` flags still matter; the other five are inert regardless
# of what the coverage grid says, which is why the UI hides them in stage 1.
STAGE1_SIGNALS = ["style", "bpm"]


def shift_window(c: "Conditioning", n_new: int) -> "Conditioning":
    """Slide a conditioning window forward by `n_new` measures.

    The first T-n_new slots become the OLD window's tail -- those are the
    measures handed to the model as fixed context, so their conditioning has to
    travel with their latents or the model is told one thing and shown another.
    Values and per-measure `active` flags both shift.

    The n_new slots exposed at the end are the ones being WRITTEN, and what goes
    in them is a real decision:

      style, bpm  -- carried forward from the last context measure, active flag
                     included. These are properties of the piece, not
                     descriptions of a particular bar; a continuation should
                     stay in the same style and the same tempo unless asked
                     otherwise, and bpm additionally has to hold a real number
                     because it sets the decoded measure duration.

      chroma, rms, density, zcr, flatness  -- turned off, but for the WHOLE
                     window, not just the new slots. These describe what a bar
                     actually contains, and nobody knows that for a bar that has
                     not been written yet. Copying the previous eight bars'
                     values forward would assert the new bars have the same
                     chords, energy and texture as the ones they follow, which
                     is both a claim we cannot make and a direct instruction to
                     repeat.

    WHY THE WHOLE WINDOW. Turning a signal off on only the new slots looks
    obviously right and is the single worst thing you can do to this checkpoint.
    Both stages call multi_token_drop with p_ind_low = p_ind_high = 0, which
    makes `token_noise < sample_drop_rates` uniformly False -- the per-measure
    dropping path is dead code. Every signal in every training window was
    ALL-ON across all 24 measures or ALL-OFF across all 24; a partial pattern
    occurred 0 times in 28,672 sampled windows. So a window with rms present on
    measures 0-15 and absent on 16-23 is a conditioning configuration the model
    has never once seen. The stage-2 local_embedder reads the presence channels
    through a kernel-3 conv over the measure axis, so the step lands directly on
    the free measures, and the additive `x = x + local_embedder(c)` path pushes
    them off-manifold -- heard as the new bars arriving at clipping volume,
    identically at every history-noise level, because the noise level has
    nothing to do with it.

    Dropping the signal window-wide instead is exactly what p_ind_uncond=0.5 did
    half the time, per signal, throughout stage-2 training. The context measures
    lose their DSP conditioning, but they are frozen latents -- they are shown to
    the model, not inferred by it, so they do not need describing. Style and bpm
    stay on everywhere, which is also uniform.

    The freed values still get back-filled from the render by
    `_measure_render`, so after a Continue the curves show what the model
    actually played and are immediately editable -- shape them and hit
    Regenerate to push the new bars somewhere else. Because the drop is now
    window-wide, that back-fill covers the context measures too: their curves
    become a measurement of the audio already in the player, which is what they
    in fact describe, so any earlier hand-edit to them is replaced by what came
    out. Nothing is lost that the render did not already answer.

    KEEP CONDITIONING UNIFORM ACROSS THE WINDOW. The same constraint applies to
    the per-measure coverage grid in the UI: any partial pattern is off the
    training distribution for this checkpoint, however reasonable it looks."""
    T = c.T
    n_new = max(1, min(int(n_new), T - 1))
    n_ctx = T - n_new
    last = n_ctx - 1                     # last context measure, post-shift
    carry = ("style", "bpm")
    new = c.copy()
    for s in Conditioning._NPZ_SIGNALS:
        arr, old = getattr(new, s), getattr(c, s)
        arr[:n_ctx] = old[n_new:]
        # seed every new slot from the last context measure. For style/bpm that
        # IS the conditioning; for the freed signals it is just a placeholder so
        # the arrays hold something plausible until the render overwrites them.
        arr[n_ctx:] = arr[last]
    for s in ALL_SIGNALS:
        a = new.active[s]
        if s in carry:
            a[:n_ctx] = c.active[s][n_new:]
            a[n_ctx:] = a[last]
        else:
            a[:] = False     # window-wide, NOT just the new slots -- see above
    return new


# -----------------------------------------------------------------------------
# Style steering
# -----------------------------------------------------------------------------
# Why steering is worth having at all: from the first Continue onward the ONLY
# things describing the future are style, bpm and the frozen latents --
# shift_window drops the five DSP signals window-wide because nobody knows what
# an unwritten bar contains. Style is therefore the one channel through which a
# user can say where the piece should go next, and the one lever against the
# failure mode long rollouts actually have, which is not divergence but
# collapse into a loop.
#
# WHY A PER-MEASURE STYLE TRAJECTORY IS LEGAL. It looks like the same mistake as
# partial per-measure conditioning (see shift_window) and is not. The style bin
# in training holds ONE CLAP AUDIO-TOWER EMBEDDING PER MEASURE and a batch takes
# a contiguous `style[idx:idx+n_chunks]` slice, so every audio-conditioned
# training window already had style varying measure to measure, tracking
# whatever the song was doing. What is uniform in training is the PRESENCE flag,
# not the value -- so `active["style"]` must stay all-on or all-off across the
# window, while the values underneath are free to move.
#
# The one caveat is the modality gap: `_swap_in_text_style` broadcasts a single
# TEXT-tower vector across the whole window (`emb[:, None, :]`), so a window
# that is audio-tower at one end and text-tower at the other interpolates
# through a region of CLAP space where neither tower puts anything. Slerp keeps
# it on the unit sphere, which is the part the model is sensitive to, but a text
# steer is a bigger ask than a walk between two audio-derived points. If a text
# steer sounds mushy in the middle and fine at the end, that gap is the suspect.
def unit(v) -> np.ndarray:
    """L2-normalize. CLAP embeddings are unit norm coming out of both towers."""
    v = np.asarray(v, dtype=np.float32)
    return (v / (float(np.linalg.norm(v)) + 1e-9)).astype(np.float32)


def slerp(a, b, t: float) -> np.ndarray:
    """Spherical interpolation on the unit sphere, renormalized.

    CLAP embeddings are unit norm (both towers end in F.normalize) and the DiT
    only ever saw unit vectors, so a plain lerp -- which shortens the vector
    toward the midpoint, by up to 30% for the ~70-degree separations typical of
    two unrelated tracks -- walks off the manifold and reads as the style
    channel weakening rather than changing. Falls back to a lerp for
    near-parallel inputs, where the sines underflow."""
    na, nb = unit(a), unit(b)
    dot = float(np.clip(na @ nb, -1.0, 1.0))
    omega = math.acos(dot)
    if omega < 1e-6:
        out = (1.0 - t) * na + t * nb
    else:
        so = math.sin(omega)
        out = (math.sin((1.0 - t) * omega) / so) * na + \
              (math.sin(t * omega) / so) * nb
    return unit(out)

def steer_style(cond: "Conditioning", n_ctx: int, anchor, target,
                f_start: float, f_end: float) -> "Conditioning":
    """Write a slerp segment into the NEW measures, in place.

    The steer as a whole runs from `anchor` to `target` along one geodesic; this
    call fills the window's free slots with the piece of it between fractions
    `f_start` and `f_end`. A steer that spans more measures than one Continue
    writes is just consecutive calls with advancing fractions, which is why the
    anchor is passed in rather than read off the context: re-anchoring at the
    current position every Continue would make each step a fraction of the
    REMAINING distance, so the steer would approach its target and never arrive
    (Zeno, in CLAP space). Anchor + absolute fraction reaches it exactly.

    Only slots >= n_ctx are touched, and that is not a stylistic choice: the
    context slots are FROZEN LATENTS. Their audio has already been decoded and,
    after the first Continue, already played. Rewriting their style would
    describe them as something they demonstrably are not, while the model is
    looking straight at the real thing.

    `active["style"]` is forced on across the WHOLE window (uniform, per
    shift_window's constraint) -- steering a channel the model has been told to
    ignore would do nothing, silently."""
    T = cond.T
    n_ctx = max(0, min(int(n_ctx), T))
    if target is None or anchor is None or n_ctx >= T:
        return cond
    a, b = unit(anchor), unit(target)
    n_new = T - n_ctx
    for j in range(n_new):
        # (j+1)/n_new, not j/n_new: the first new measure has already moved. A
        # steer that spends its opening bar sounding exactly like the context
        # wastes an eighth of the only runway it has.
        u = (j + 1) / float(n_new)
        f = float(np.clip(f_start + (f_end - f_start) * u, 0.0, 1.0))
        cond.style[n_ctx + j] = slerp(a, b, f)
    cond.active["style"][:] = True
    return cond


def snap_bpm(bpm):
    """The nearest tempo the model actually has, given `bpm`. See BPM_GRID_NUM.

    Scalar in, float out; array in, array out -- the write sites need both, and
    the two must not drift apart, which is why there is one function.

    Nearest in FRAMES, not in BPM: `k = round(BPM_GRID_NUM / bpm)` rounds the
    measure's length to a whole number of latent frames, which is the thing
    that is actually quantised. Nearest-in-BPM would be a different (and, at
    the top of the range, worse) projection of the same constraint.

    What it costs, as absolute error: 0.00 BPM at 60 and 120, 0.23 at 90, 0.47
    at 176, 1.82 at 220, 0.29 at 286. Half a quantum is the bound, EXCEPT at
    the ends, where the clamp makes the projection one-sided: the top grid
    tempo is BPM_GRID_NUM/35 = 342.857 because 12000/34 = 352.94 is outside the
    trained range, so a slider dragged to 350 lands 7.1 BPM low. That is the
    right answer -- 352.94 would index row 353, which the model's own
    `clamp(round(bpm), 0, 349)` folds back to an off-grid row -- but it means
    the top of the range is coarse in a way the bpm^2 quantum alone does not
    predict. What it buys, scored on fraction of
    measures on-grid / mean embedding row norm / distinct rows / median
    cos(row(m), row(m+1)) over a 96-measure journey:

        176->286 ramp,    as written    24%   0.636   95   -0.005
        176->286 ramp,    snapped      100%   0.844   27   +1.000
        176 const+jitter, as written    34%   0.662   10   -0.004
        176 const+jitter, snapped      100%   0.869    6   +0.785

         90->130 ramp,    as written    94%   0.892   41   +1.000
         90->130 ramp,    snapped      100%   0.913   38   +1.000

    Read the top row as: on three quarters of measures the fast ramp handed the
    DiT a random untrained vector, and consecutive measures' tempo tokens were
    mutually ORTHOGONAL. Read the bottom pair as the honest limit of this fix --
    at ordinary tempi the ramp was already 94% on-grid and snapping is close to
    a no-op. It earns its keep in the range the presets actually occupy.

    Note also what the third column says about the fast ramp: snapped, 96
    measures collapse to 27 levels held 3-4 measures each. The grid turns a ramp
    into a staircase on its own, and picks the step size while it is there. That
    is not a defect of snapping -- those 27 tempi are the only ones the decoder
    can render, so the unsnapped ramp was a staircase too, just one whose treads
    the conditioning lied about.

    Non-finite and non-positive input is returned as BPM_MIN rather than
    raising: this sits under every bpm write in the app, and a NaN reaching
    `_decode` is a measure length, i.e. a crash in the tokenizer instead of a
    wrong number in a plot."""
    scalar = np.isscalar(bpm) or np.asarray(bpm).ndim == 0
    b = np.asarray(bpm, dtype=np.float64).reshape(-1)
    ok = np.isfinite(b) & (b > 0)
    k = np.rint(np.divide(BPM_GRID_NUM, b, out=np.full_like(b, BPM_K_MAX),
                          where=ok))
    k = np.clip(k, BPM_K_MIN, BPM_K_MAX)
    out = np.where(ok, BPM_GRID_NUM / k, BPM_MIN)
    if scalar:
        return float(out[0])
    return out.reshape(np.asarray(bpm).shape).astype(
        np.asarray(bpm).dtype if np.asarray(bpm).dtype.kind == "f"
        else np.float32)


def bpm_grid_error(bpm):
    """Largest distance from `bpm` to the grid, in BPM. For diagnostics -- see
    real_backend._bpm_row, which is the one place that gets to find out a write
    site forgot to snap."""
    b = np.asarray(bpm, dtype=np.float64).reshape(-1)
    if b.size == 0:
        return 0.0
    return float(np.max(np.abs(b - np.asarray(snap_bpm(b), dtype=np.float64))))


def fold_bpm(target, ref, window: float = BPM_STEER_FOLD) -> float:
    """`target` moved by octaves until it is within a factor of `window` of
    `ref` AND inside the trained range. See BPM_STEER_FOLD.

    The two conditions can disagree: the window is relative to `ref`, the range
    is absolute. Target 571 against ref 286 is inside the window (571 < 2*286)
    and outside the range, and there halving beats clamping -- an octave keeps
    the destination musically the tempo that was asked for, where a clamp to 350
    invents a tempo unrelated to either end of the leg.

    Snapped on the way out, because folding is itself a way OFF the grid:
    halving `BPM_GRID_NUM/k` gives `BPM_GRID_NUM/2k` and stays on it, but
    doubling gives `BPM_GRID_NUM/(k/2)`, which only exists when k is even."""
    t, r = float(target), float(ref)
    if not (t > 0 and r > 0 and math.isfinite(t) and math.isfinite(r)):
        return snap_bpm(t if math.isfinite(t) else BPM_MIN)
    for _ in range(8):                    # bounded: 8 octaves covers any input
        if t >= r * window:
            t /= 2.0
        elif t < r / window:
            t *= 2.0
        else:
            break
    # Then the absolute range, which wins. Bounded for the same reason, and it
    # cannot fight the loop above: [BPM_MIN, BPM_MAX] spans 8.75x, wider than
    # any sane `window`, so a value can't be pushed out one side by the other.
    for _ in range(8):
        if t > BPM_MAX:
            t /= 2.0
        elif t < BPM_MIN:
            t *= 2.0
        else:
            break
    return snap_bpm(float(np.clip(t, BPM_MIN, BPM_MAX)))


def steer_bpm(cond: "Conditioning", n_ctx: int, anchor, target,
              f_start: float, f_end: float) -> "Conditioning":
    """Write a tempo ramp into the NEW measures, in place. Mirrors steer_style.

    Same slots, same fractions, same anchor-plus-absolute-fraction rule, and
    that is the point: called with the fractions `steer_style` just used, the
    tempo and the style arrive together instead of the music reaching a new
    style at the old speed. Only slots >= n_ctx are touched, for the reason
    given in steer_style -- the context measures are frozen latents whose audio
    has already played, and bpm is worse than style to lie about, because
    `_decode` turns each measure's bpm into that measure's SAMPLE COUNT. Rewrite
    a context measure's tempo and the window's bounds stop matching its audio.

    GEOMETRIC, NOT LINEAR. Tempo is a ratio scale -- 120->180 and 180->270 are
    the same musical distance -- so a constant-ratio ramp is heard as an even
    accelerando where a constant-BPM-per-measure ramp drags at the top. It also
    matches what slerp does on the style side: constant rate along the natural
    metric of the space, so the two really do stay in step.

    THE RAMP SHAPE IS WRONG, AND MEASURING IT SAID SO. This docstring used to
    claim a ramp was "strictly closer to the manifold than the flat line it
    replaces", reasoning from `bpm_curve`'s note that real recordings wander a
    few BPM per measure. That reasoning only established that the flat line was
    missing JITTER; it said nothing about TREND, and it was the trend that was
    invented. Scored on the statistics
    Jazz/neural/analyze_bpm_variation.py uses, over the five demo presets'
    measured per-measure bpm:

                              real songs          this ramp
        variance from drift   0.0 - 1.8 %         100 %
        drift over 24 meas.   -1.2 .. +6.4 BPM    +22.6 BPM
        jitter index          0.75 - 1.57 (~white) 0.00
        lag-1 autocorrelation -0.66 .. +0.25      +1.00
        steps in one direction 26 - 48 %          100 %
        better per-measure
          predictor           song median        fitted trend

    Real tempo is white jitter about a constant; every song is predicted better
    by its own median than by a fitted trend. A ramp is the exact inverse on
    every one of those, and roughly 4x the largest real drift in magnitude.
    That is why `bpm_follow` now defaults to OFF: the flat line this replaces is
    wrong only in lacking jitter, whereas the ramp is wrong in shape, and bpm is
    the one signal the model cannot soften -- `_decode` turns it straight into
    each measure's length. The in-distribution construction is a constant plus
    white jitter whose CONSTANT moves between legs, which is a different
    function from this one; see the audit note in the app's checkbox."""
    T = cond.T
    n_ctx = max(0, min(int(n_ctx), T))
    if target is None or anchor is None or n_ctx >= T:
        return cond
    a = float(anchor)
    b = fold_bpm(target, a)
    if not (a > 0 and b > 0):
        return cond
    n_new = T - n_ctx
    la, lb = math.log(a), math.log(b)
    for j in range(n_new):
        # (j+1)/n_new, matching steer_style: the first new measure has already
        # moved, so the two channels stay aligned measure for measure.
        u = (j + 1) / float(n_new)
        f = float(np.clip(f_start + (f_end - f_start) * u, 0.0, 1.0))
        # Snapped per measure, which is what turns this ramp into the staircase
        # snap_bpm's table describes: at preset tempi, 96 measures over 27
        # levels, each held for as long as the grid is flat there. The geometric
        # spacing is still doing the work -- it decides WHERE the treads fall --
        # the grid only decides that there are treads.
        cond.bpm[n_ctx + j] = snap_bpm(math.exp(la + (lb - la) * f))
    cond.active["bpm"][:] = True
    return cond


def step_bpm_level(here, target, steps_left: int = 1,
                   max_ratio: float = BPM_STEP_MAX):
    """The next tempo constant to hold: one of `steps_left` equal geometric
    steps from `here` to `target`, capped at a factor of `max_ratio`. None if
    there is nothing to move.

    Geometric, for the reason `steer_bpm` gives -- tempo is a ratio scale, so
    "at most 10% faster or slower" is one bound rather than two, and equal
    ratios are what makes a sequence of steps read as one journey rather than a
    big move followed by small ones. `target` is octave-folded against `here`
    first, so a destination is never approached the long way round.

    `steps_left` is what makes the tempo ARRIVE. It is recomputed at each step
    from the measures actually remaining in the leg, so the cap costs less than
    it looks: a step that clamps leaves a shortfall, and the next step divides
    that shortfall by a smaller count and so asks for a larger ratio, spending
    the cap it has rather than inheriting the last step's deficit. Over the
    default 96-measure leg that is 4 steps and a journey of 1.62x lands about
    10% short, against the 48% that one bounded step per leg left behind.

    Landing short at all is deliberate. Raising BPM_STEP_MAX to 1.15 would
    close it, at 38.3 BPM of apparent drift in each straddling window instead
    of 25.5 -- a listening trade, which is why it is an env var."""
    if here is None or target is None:
        return None
    a = float(here)
    if not (a > 0 and math.isfinite(a)):
        return None
    b = fold_bpm(target, a)
    if not (b > 0):
        return None
    n = max(1, int(steps_left))
    r = max(1.0 + 1e-9, float(max_ratio))
    # The whole remaining journey is b/a; one step of n equal ones is its n-th
    # root. Done in log space so n=1 is exactly b/a with no rounding drift.
    ratio = math.exp(math.log(b / a) / n)
    ratio = float(np.clip(ratio, 1.0 / r, r))
    # Snapped here and not only at the write, because the LEVEL is what gets
    # carried between takes (tr.bpm_level) and compared against to decide
    # whether the leg has arrived. An unsnapped level would ask for a tempo the
    # written measures then quietly round away from, so "arrived" would be
    # measured against a tempo nothing ever played.
    return snap_bpm(float(np.clip(a * ratio, BPM_MIN, BPM_MAX)))


def steer_bpm_stepped(cond: "Conditioning", n_ctx: int, level, seed: int = 0):
    """Write ONE tempo constant plus white jitter into the new measures.

    The in-distribution counterpart to `steer_bpm`, and the shape the audit in
    that function's docstring points at: real per-measure bpm is white noise
    about a fixed value, so that is what this writes. No trend within the
    window, no trend within the leg -- the leg's constant is chosen once by
    `step_bpm_level` and every take of that leg writes the same one.

    WHAT IT COSTS, STATED PLAINLY. The tempo still has to change some time, and
    the window that straddles the change holds 16 context measures at the old
    constant and 8 at the new one, which fits a trend almost perfectly. That
    window is off-manifold and no arrangement of these measures avoids it; the
    trade is that it happens twice per leg instead of in every window. On the
    default 96-measure leg, 10 of 12 windows score 4.6% var-by-drift with a
    jitter index of 1.03 -- real songs' own range -- and 2 do not. See
    BPM_STEP_MAX for how big the ones that do not are allowed to be.

    The jitter is drawn from `seed`, which callers advance per take, so it is
    reproducible with the rest of the take and never identical between takes.
    It is multiplicative because BPM_STEP_JITTER was measured as a relative
    std, and because the same absolute wobble means less at 286 than at 176."""
    T = cond.T
    n_ctx = max(0, min(int(n_ctx), T))
    if level is None or n_ctx >= T:
        return cond
    lv = float(level)
    if not (lv > 0 and math.isfinite(lv)):
        return cond
    n_new = T - n_ctx
    rng = np.random.default_rng(_seed_int("bpm-step", str(int(seed))))
    j = rng.standard_normal(n_new) * float(BPM_STEP_JITTER)
    # Snapped after the jitter, not instead of it. 1.5% of 176 is 2.6 BPM and
    # the grid quantum there is 2.6 BPM, so the two are the same size: the
    # jitter survives as a walk over the two or three grid levels either side
    # of the constant rather than as a continuum, which is what the training
    # data's own per-measure tempo looks like for the same reason.
    cond.bpm[n_ctx:] = snap_bpm(np.clip(lv * (1.0 + j), BPM_MIN, BPM_MAX))
    cond.active["bpm"][:] = True
    return cond


def point_cluster(name) -> str:
    """The song a reference point came from.

    `style_points` labels every point `preset#measure`, so a cluster IS a
    preset IS a song -- the three words mean the same thing here, and the
    per-measure points of one song are a tight blob in style space rather than
    separate places to go (measured: every pair of presets separates by
    15-55x the spread within one). The mock backend labels points with the bare
    preset name and no `#`, which this handles by returning the whole string."""
    return str(name).split("#")[0]


def pick_reference(current, named, seed, avoid=None, cross_cluster=False,
                   here_name=None, recent=()):
    """`wander`'s destination for the next leg: (name, unit vec), or None.

    Split out of `walk_step` because a destination outlives the take that chose
    it. Choosing one per take and stepping a fraction of the way toward it is a
    random walk however far each step goes -- see Track.style_leg_key for what
    that measured like -- so the choice happens once per LEG and the travelling
    is done by `steer_style`, the same interpolator a typed steer uses.

    Excludes anything `current` is effectively already sitting on (a zero-length
    leg would stall the walk) and, via `avoid`, the point the last leg was
    heading for, so the radio cannot pick the same journey twice in a row.

    `cross_cluster` additionally refuses any point from the SONG the walk is
    currently in. It exists because the reference pool is 88 points drawn from
    only five recordings, so most of it is not 88 places -- it is five places
    listed many times over. Without this, most legs are a hop between two
    measures of the piece already playing: a real journey in CLAP space and an
    inaudible one. With it, every leg crosses to a different song and the walk
    actually wanders. This is a small-data workaround and says so in the UI;
    with a large enough reference set, within-song motion becomes worth having
    again and the flag should go back to off.

    `here_name` is the label of the point the walk is sitting on, when the
    caller knows it. When it does not, the current cluster is inferred as the
    nearest point's -- which is what "where am I" means for a position that is
    mid-slerp between two songs and therefore has no label of its own.

    `recent` is the songs the last few legs went to (WANDER_RECENT of them),
    refused when refusing leaves anywhere to go. `avoid` is not enough on its
    own: it blocks the previous POINT, and a song's own measures sit ~12 deg
    apart, so the song just left is fully available again on the next leg.

    THE DRAW IS UNIFORM OVER SONGS, THEN OVER THAT SONG'S POINTS, not uniform
    over points. The pool's five songs contribute 23/23/18/16/8 surviving
    measures, so a flat draw visits the biggest ~3x as often as the smallest --
    and that ratio is an artefact of how internally varied each recording is
    (how many of its measures cleared the 0.995 dedup), which is not a reason
    to go there more. Two-stage keeps every measure a legitimate destination,
    which is the point the pool is built on, while making the CHOICE OF SONG
    say what it means."""
    cur = unit(current)
    av = None if avoid is None else unit(avoid)
    pool = [(str(n), unit(v)) for n, v in (named or []) if v is not None]
    far = [(n, v) for n, v in pool
           if float(cur @ v) < 0.995 and (av is None or float(av @ v) < 0.995)]
    if not far:
        return None
    if cross_cluster:
        home = (point_cluster(here_name) if here_name else
                point_cluster(max(pool, key=lambda nv: float(cur @ nv[1]))[0]))
        # Only if something survives. A pool with one song in it, or a walk
        # already excluded down to its own cluster, gets the ordinary choice
        # rather than a stall -- the flag is a preference for variety, and
        # stopping the music is not a more varied outcome.
        away = [(n, v) for n, v in far if point_cluster(n) != home]
        if away:
            far = away
    if recent:
        # Same "only if something survives" rule as cross_cluster above, and for
        # the same reason: variety is a preference, and stopping the music is
        # not a more varied outcome.
        rs = {str(r) for r in recent}
        fresh = [(n, v) for n, v in far if point_cluster(n) not in rs]
        if fresh:
            far = fresh
    rng = np.random.default_rng(int(seed) & 0x7FFF_FFFF)
    # Song first, then a measure of it. See the docstring for why this is not
    # one flat draw over `far`.
    groups = {}
    for n, v in far:
        groups.setdefault(point_cluster(n), []).append((n, v))
    keys = sorted(groups)
    g = groups[keys[int(rng.integers(len(keys)))]]
    return g[int(rng.integers(len(g)))]


def _recent_clusters(tr, n: int) -> list:
    """The songs the walk's last `n` legs went to, oldest first.

    Read off `style_leg_log` rather than kept in a field of its own: the log
    already holds one entry per leg with the destination's name, and
    `stamp_new_measures` writes it AFTER the steer runs, so at the moment
    `pick_reference` is called its tail is exactly "where the walk has just
    been" and does not yet include the leg being chosen."""
    out = []
    for e in list(getattr(tr, "style_leg_log", None) or [])[-max(0, int(n)):]:
        c = point_cluster(e.get("name") or "")
        if c:
            out.append(c)
    return out


def walk_step(current, points, mode: str, step_deg: float, seed: int,
              axis=None, direction=None):
    """`drift`'s per-take step along a held heading. Returns (target, heading).

    `step_deg` is how far to travel THIS Continue, in degrees of arc, so drift
    and wander are calibrated in the same unit and the pace control means the
    same thing whichever is selected. `direction` is the heading the caller
    stored on the Track last take and hands back now.

    `axis` is an ambient vector (a named contrast, or the difference between two
    reference points); it is projected into the tangent space at the current
    position and the step is a rotation, so the path is a great circle rather
    than a chord and every point on it stays unit-norm. Because the heading
    persists, this genuinely goes somewhere: a walk that re-picks a direction
    every step has expected displacement growing only as sqrt(n), since it
    spends most of its time undoing itself.

    Speculative as MUSIC. Nothing here says the path between two plausible
    styles is itself plausible; that is the experiment. `wander` at least lands
    on embeddings of real recordings -- see pick_reference."""
    cur = unit(current)
    pts = [unit(p) for p in (points or []) if p is not None]
    step = math.radians(max(0.0, float(step_deg)))
    if step <= 0.0 or mode == "off":
        return None, direction
    rng = np.random.default_rng(int(seed) & 0x7FFF_FFFF)

    if mode == "drift":
        ax = None
        if axis is not None:
            ax = np.asarray(axis, np.float32)
        elif direction is not None:
            ax = np.asarray(direction, np.float32)
        for _ in range(8):
            if ax is None:
                if len(pts) < 2:
                    return None, direction
                i, j = rng.choice(len(pts), size=2, replace=False)
                ax = pts[j] - pts[i]
            # Project the ambient axis into the tangent space at `cur`. Redoing
            # this every step IS the parallel transport -- holding the ambient
            # axis fixed and re-projecting keeps the heading semantically
            # constant, which is the only reason to hold a direction at all.
            tan = ax - float(cur @ ax) * cur
            if float(np.linalg.norm(tan)) > 1e-4:
                tan = unit(tan)
                tgt = unit(math.cos(step) * cur + math.sin(step) * tan)
                return tgt, ax.astype(np.float32)
            # `cur` has drifted parallel to the axis; there is nowhere left to
            # go along it. A named axis is a user instruction, so give up rather
            # than silently substituting a different heading.
            if axis is not None:
                return None, direction
            ax = None
        return None, None

    return None, direction


def angle_deg(a, b) -> float:
    """Angle between two style vectors, in degrees. The distance unit for
    everything user-facing (see STYLE_FULL_CHANGE_DEG)."""
    return math.degrees(math.acos(float(np.clip(unit(a) @ unit(b), -1.0, 1.0))))


def _bar(frac: float, width: int = 12) -> str:
    n = int(round(max(0.0, min(1.0, frac)) * width))
    return "▓" * n + "░" * (width - n)


def style_is_off(tr) -> bool:
    """Will this Track's style channel actually be dropped on the next Continue?

    `Track.style_off` is a REQUEST, and stage 1 cannot honour it (see
    _style_off_gate). Three places have to agree about what the next take will
    do -- the gate that performs it, `plan_fast_forward` which sizes the run,
    and `arm_idle_stream` which decides whether the run needs a walk -- and
    when any two of those re-derive a rule separately they eventually disagree.
    """
    return (bool(getattr(tr, "style_off", False))
            and int(getattr(tr, "stage", 2)) != 1)


def _style_off_gate(tr, c) -> tuple[bool, str]:
    """Resolve `Track.style_off` against the window. (suppressed, warning).

    Suppressing style is one line -- `active["style"][:] = False`, window-wide,
    which makes `_drop_from_active` swap in the model's learned `null_style` on
    every measure and makes `_has_cond` skip the style CFG pass entirely. The
    rest of this is the two edges around that line.

    STAGE 1 CANNOT DO THIS. See Track.style_off: stage 2 dropped style for the
    whole sequence in roughly half its training windows and did so independently
    of the other signals, so style-off-with-bpm-on is ordinary for it; stage 1
    ran p_ind_uncond=0, where the only windows without style were the 10% that
    had nothing at all. Rather than silently sounding out a configuration that
    stage has never seen, this refuses and says so, and the steer/walk that was
    already set runs instead.

    TURNING IT BACK ON has to be explicit, because `shift_window` carries the
    active flags forward: without the restore branch, one styled-off Continue
    would leave the style channel dead for the rest of the track, and only a
    typed steer (which forces the flags on) would ever revive it. The values
    underneath stayed truthful while it was off -- `_measure_render` back-fills
    style from the audio precisely on the measures where it is inactive -- so
    coming back on resumes from the style the music actually has rather than
    from whatever it was pinned to before.
    """
    if bool(getattr(tr, "style_off", False)) and not style_is_off(tr):
        return False, ("⚠️ Style off ignored — stage 1 trains with "
                       "p_ind_uncond=0, so a window with style dropped and "
                       "tempo kept is one it has never seen. Switch the model "
                       "to stage 2 to use it.")
    if style_is_off(tr):
        c.active["style"][:] = False
        tr._style_forced_off = True
        tr.style_note = (
            "🚫 Style off — every measure gets the model's learned null style "
            "token, so nothing is describing where this should go except the "
            "bars it has already written. About half of stage 2's training "
            "windows looked like this, so it is ordinary input, not a hack. "
            "Tempo is still conditioned; steer, wander and drift do nothing "
            "until you turn style back on.")
        return True, ""
    if getattr(tr, "_style_forced_off", False):
        c.active["style"][:] = True
        tr._style_forced_off = False
    return False, ""


def style_destination(tr):
    """(name, unit vector) of where the style is headed right now, or ("", None).

    There are two destinations in a Track and they are not the same field: a
    typed steer keeps its target in `style_target` under the name `style_key`,
    and an automatic walk keeps this leg's in `style_dest`/`style_dest_name`.
    The precedence between them is not a choice made here -- it is the one
    `apply_style_steering` already makes, where the typed branch returns before
    the walk branch is reached, so a prompt in the box wins. Reading only one of
    the two (which is what the first version of the trail recorder did) tags
    every typed steer with whatever walk ran before it, which is worse than
    untagged: the map would draw a confident line to the wrong place.
    """
    if tr is None or style_is_off(tr):
        return "off", None
    tgt = getattr(tr, "style_target", None)
    if tgt is not None:
        return str(getattr(tr, "style_key", "") or "typed"), unit(tgt)
    dest = getattr(tr, "style_dest", None)
    if dest is not None:
        return str(getattr(tr, "style_dest_name", "") or ""), unit(dest)
    return "", None


def stamp_new_measures(tr, n_ctx: int):
    """Record which destination the measures just written were aimed at.

    WHY THIS CANNOT BE DONE LATER, which is the whole point of the function.
    A measure is written into window slots n_ctx..T, and it is not banked until
    it has slid all the way down to slot 0 -- two Continues later, at the
    default eight measures a take. Asking "what is the destination?" at bank
    time therefore answers for a leg chosen SIXTEEN MEASURES after the measure
    was written, and at a leg boundary that is simply the wrong leg. Measured
    before this existed: with pace 24 the recorded boundaries came out at
    0/8/32/56 when the real ones were 0/24/48/72 -- every segment shifted early
    by exactly n_ctx, so the trajectory panel sliced each leg's approach across
    two legs and drew both of them wrong.

    So the stamp happens here, at the one moment the answer is correct: right
    after the steer wrote those slots, with the destination that steered them
    still in force.

    Two records, because they answer different questions. `style_window_tag`
    is per slot and rides the window down into `style_trail_tag`, which is what
    gives the map its leg boundaries. `style_leg_log` is per leg and keeps the
    destination VECTOR -- without it a finished leg's target is unrecoverable
    (style_dest holds only the current one), and the panel would have to guess
    it from where the leg ended, which makes every leg look like it arrived
    perfectly whether it did or not.
    """
    if tr is None or getattr(tr, "cond", None) is None:
        return tr
    T = int(tr.cond.T)
    n_ctx = max(0, min(int(n_ctx), T))
    name, dest = style_destination(tr)
    tags = list(getattr(tr, "style_window_tag", None) or [])
    # Pad at the FRONT: a short list means slots that predate the stamping (a
    # Track seeded before this existed, or the very first window), and those
    # are the OLD end of the window, not the new one.
    if len(tags) < n_ctx:
        tags = [""] * (n_ctx - len(tags)) + tags
    tr.style_window_tag = tags[:n_ctx] + [name] * (T - n_ctx)

    log = list(getattr(tr, "style_leg_log", None) or [])
    if not log or log[-1].get("name") != name:
        log.append({"start": int(getattr(tr, "prefix_measures", 0)) + n_ctx,
                    "name": name,
                    "dest": None if dest is None
                    else np.asarray(dest, dtype=np.float32)})
    elif log[-1].get("dest") is None and dest is not None:
        # Same name, and we now know the vector we did not before. Can happen
        # when a leg's first stamp lands on an early-return path.
        log[-1]["dest"] = np.asarray(dest, dtype=np.float32)
    tr.style_leg_log = log
    return tr


def record_style_trail(tr, n_new: int):
    """Bank the style of the `n_new` measures about to leave the window.

    Called from both backends' `_bank_window`, which is the one moment a
    measure's conditioning stops being editable: after the slide it is gone
    from `cond.style` for good (see Track.style_trail). Recording it anywhere
    later would miss the measures the first slide banks; recording it earlier
    would bank measures a re-render can still change.

    The tags come out of `style_window_tag`, which `stamp_new_measures` wrote
    when these measures were written. They are NOT re-derived from the Track's
    current destination here: by the time a measure reaches slot 0 the walk has
    picked a new destination twice over, and reading it now attributes the
    measure to a leg that did not exist when it was written. That was a real
    bug and it is the reason the stamp exists at all -- see there.
    """
    if tr is None or getattr(tr, "cond", None) is None:
        return tr
    n = max(0, min(int(n_new), int(tr.cond.T)))
    if n <= 0:
        return tr
    new = np.asarray(tr.cond.style[:n], dtype=np.float32).copy()
    old = getattr(tr, "style_trail", None)
    tr.style_trail = new if old is None or len(old) == 0 else \
        np.concatenate([np.asarray(old, np.float32), new])
    # "off" is a state, not a destination, and it has to be distinguishable
    # from "nothing steering": with style off the vector in cond.style is
    # whatever was last written and the model is not being shown it at all, so
    # a map that drew those measures as a path would be drawing a journey
    # nothing took. style_destination returns "off" for exactly that case and
    # "" for unsteered-but-listening, which is the distinction the stamp keeps.
    win = list(getattr(tr, "style_window_tag", None) or [])
    if len(win) < n:
        # Only reachable for measures written before the window was ever
        # stamped (a seeded Track's first window). Untagged, not mis-tagged.
        win = [""] * (n - len(win)) + win
    tags = list(getattr(tr, "style_trail_tag", None) or [])
    tags.extend(win[:n])
    tr.style_trail_tag = tags
    # Drop what we just banked; the rest slides with the window, which is what
    # keeps slot i of `style_window_tag` describing slot i of `cond.style`.
    tr.style_window_tag = win[n:]
    return tr


def apply_cond_hook(tr) -> None:
    """Run `Track.cond_hook`, if the track has one. In place, no-op if not.

    LAST, after the slide and after the style steer, so a hook that writes bpm
    beats `steer_bpm` rather than racing it -- a track that authors its own
    tempo per measure has said something more specific than "ramp toward the
    destination's tempo", and the more specific statement should win.

    Both backends call this from their continuation step so the mock and the
    real model can disagree about the model but never about the conditioning,
    which is the same reason `apply_style_steering` is shared."""
    hook = getattr(tr, "cond_hook", None)
    if hook is not None:
        tr.cond = hook(tr.cond, int(tr.prefix_measures))


def apply_style_steering(tr, n_ctx: int, seed: int, points=None,
                         axis=None, bpms=None) -> str:
    """`_apply_style_steering`, then stamp the measures it just wrote.

    A wrapper and not an extra line at the bottom of the real function, because
    that function has eleven return statements -- every early exit (style off,
    a stalled walk, no reference points) is a case the stamp still has to
    happen for, and the version of this that tried to remember them all missed
    three. The stamp is the only record of which destination a measure was
    written toward; see `stamp_new_measures` for why that cannot be recovered
    later."""
    out = _apply_style_steering(tr, n_ctx, seed, points=points, axis=axis,
                                bpms=bpms)
    stamp_new_measures(tr, n_ctx)
    return out


def _apply_style_steering(tr, n_ctx: int, seed: int, points=None,
                          axis=None, bpms=None) -> str:
    """Resolve a Track's steer/walk settings into its new measures, in place.

    Shared by both backends' continue_track, so the mock and the real model can
    disagree about the model but never about the conditioning. A typed target
    beats the automatic walk. Also writes `tr.style_note`, the persistent
    progress readout the UI shows under the player -- without it neither a
    long steer nor a walk leaves any visible trace, and "am I nearly there?"
    has no answer.

    `Track.style_off` outranks both, and outranks them by making them
    meaningless rather than by preference: with the style channel dropped
    window-wide there is no vector left for a steer or a walk to move, so
    running one would produce a progress readout for a journey nothing is
    taking. `plan_fast_forward` and `arm_idle_stream` make the same call, and
    all three have to keep agreeing.

    `points` is [(name, vec), ...]; names are only used for the readout.
    `bpms` is {name: bpm} over the same names, and is what lets `wander` take
    the tempo with it (see steer_bpm). Optional: a backend that cannot say what
    tempo a style point came from passes nothing and tempo following is simply
    inert, rather than every caller having to know about it."""
    c = tr.cond
    n_new = c.T - n_ctx
    if n_new <= 0:
        return ""
    off, warn = _style_off_gate(tr, c)
    if off:
        # Tempo follows the STYLE destination, so with no style there is no
        # destination to read a tempo off and the walk is not running anyway.
        return "steer[style-off]: window-wide null, no style CFG pass"
    diag = _steer_or_walk(tr, c, n_ctx, n_new, seed, points, axis, bpms)
    if warn:
        tr.style_note = f"{warn}\n\n{tr.style_note}".strip()
        diag = f"style-off refused (stage 1); {diag}" if diag else \
            "style-off refused (stage 1)"
    return diag


def _steer_or_walk(tr, c, n_ctx: int, n_new: int, seed: int, points, axis,
                   bpms=None) -> str:
    """The typed-steer / wander / drift half of `apply_style_steering`.

    Split out only so the style-off gate above can sit in front of it without
    being threaded through every one of this function's return points."""
    here = unit(c.style[n_ctx - 1] if n_ctx else c.style[0])
    named = [(str(n), unit(v)) for n, v in (points or [])]
    vecs = [v for _, v in named]

    def nearest(at):
        """Named reference point closest to `at`. Called with the position the
        window ENDS at, not the one it started from, or the readout describes
        where the walk was one Continue ago."""
        if not named:
            return ""
        n, v = min(named, key=lambda nv: angle_deg(at, nv[1]))
        return f"  ·  nearest reference: {n} ({angle_deg(at, v):.0f}° away)"

    # ---- typed target: a fixed destination, reached over style_span measures -
    target = getattr(tr, "style_target", None)
    if target is not None:
        span = max(1, int(getattr(tr, "style_span", STYLE_SPAN)))
        key = str(getattr(tr, "style_key", ""))
        # A new prompt (or a changed span) restarts the journey from wherever we
        # currently are; the same one continues the journey already in flight.
        if (key, span) != getattr(tr, "_style_journey", None) or \
                tr.style_anchor is None:
            tr.style_anchor = here
            tr.style_done = 0
            tr._style_journey = (key, span)
        done = int(getattr(tr, "style_done", 0))
        f0, f1 = done / span, min(1.0, (done + n_new) / span)
        steer_style(c, n_ctx, tr.style_anchor, target, f0, f1)
        tr.style_done = min(span, done + n_new)
        left = max(0, span - tr.style_done)
        total = angle_deg(tr.style_anchor, target)
        if left:
            tr.style_note = (
                f"🎯 steering to “{key}”  {_bar(f1)} {f1 * 100:.0f}%  ·  "
                f"{left} measures to go ({math.ceil(left / n_new)} more "
                f"Continue{'s' if math.ceil(left / n_new) != 1 else ''})  ·  "
                f"{total * (1 - f1):.0f}° of {total:.0f}° left")
        else:
            tr.style_note = (f"🎯 arrived at “{key}” — moved {total:.0f}°. "
                             f"Clear the box to stop holding it.")
        return f"steer[text] {f0:.2f}→{f1:.2f} of {total:.0f}°"

    # ---- automatic walk: a SEQUENCE of legs, each `pace` measures long --------
    # The walk has no single destination, so it invents one, travels there over
    # a leg, and on arrival invents the next. That loop is the point: a walk
    # that re-decided every take never arrived anywhere (Track.style_leg_key),
    # and one that committed to a single destination forever would arrive and
    # then hold static conditioning for the rest of the press -- the exact
    # failure arm_idle_stream exists to avoid, just further from home.
    mode = str(getattr(tr, "style_walk", "off") or "off")
    if mode == "off":
        tr.style_note = ""
        tr.style_origin = None
        _end_walk_leg(tr)
        return ""
    pace = max(1, int(getattr(tr, "style_pace", STYLE_PACE)))
    if tr.style_origin is None:
        tr.style_origin = here
    # Restart rule, deliberately the same shape as the typed steer's: a leg
    # belongs to one (mode, pace, axis), and changing any of them mid-walk
    # abandons the journey in flight rather than finishing an old one at a new
    # pace. Reaching `pace` also ends the leg -- that is arrival.
    key = (mode, pace, str(getattr(tr, "style_axis", "") or ""))
    done = int(getattr(tr, "style_leg_done", 0))
    new_leg = (key != getattr(tr, "style_leg_key", None)
               or getattr(tr, "style_leg_from", None) is None
               or done >= pace)
    if new_leg:
        arrived_at = str(getattr(tr, "style_dest_name", ""))
        tr.style_leg_from, tr.style_leg_done, tr.style_leg_key = here, 0, key
        tr.style_legs = int(getattr(tr, "style_legs", 0)) + 1
        done = 0
        if mode == "wander":
            # `style_dest_name` is still the leg that just FINISHED, so it is
            # the label of where the walk is standing -- which is exactly what
            # cross_cluster needs and is better than inferring it by nearest
            # neighbour. On the first leg it is empty and the inference runs.
            # KEYED ON THE LEG INDEX, not on `seed` alone, and that is the
            # whole difference between a walk and a two-song loop.
            #
            # `seed` here is the take's seed, which the app takes from a Number
            # the user sets and no handler ever writes back (cont_seed / s_seed
            # are wired as inputs only). A leg begins on the FIRST take of a
            # press, so at the default pace -- where one press is exactly one
            # leg -- every leg was choosing its destination from an identical
            # RNG state. The choice is then a pure function of where the walk is
            # standing, so the walk becomes a deterministic orbit on five songs,
            # and orbits that short are almost always 2-cycles: measured over
            # 400 seed values, 99% of runs reached exactly two songs in eight
            # legs, and the mean was 2.01 of 5. Mixing the leg counter in takes
            # that to 4.35, and to 4.72 with the cluster-uniform draw and
            # `recent` below.
            #
            # `style_legs` was incremented just above, so it identifies THIS
            # leg. Hashing rather than adding keeps consecutive legs from
            # drawing adjacent states, and keeps the same seed reproducing the
            # same journey -- which is what the seed box promises.
            nxt = pick_reference(here, named,
                                 _seed_int("wander-leg", int(seed),
                                           int(tr.style_legs)),
                                 avoid=getattr(tr, "style_dest", None),
                                 cross_cluster=bool(getattr(
                                     tr, "walk_cross_cluster", True)),
                                 here_name=arrived_at or None,
                                 recent=_recent_clusters(tr, WANDER_RECENT))
            if nxt is None:
                _end_walk_leg(tr)
                tr.style_note = (
                    "🧭 wander stalled — every reference point is one this is "
                    "already sitting on. More presets is the only fix; drift "
                    "or a typed steer will still move.")
                return ""
            tr.style_dest_name, tr.style_dest = nxt
            # Tempo leg, chosen with the style leg and only here: `here_bpm` is
            # the last CONTEXT measure's tempo, i.e. what the music is actually
            # doing right now, so the ramp starts from the audio rather than
            # from whatever the leg before it was aiming at.
            tr.bpm_leg_from = tr.bpm_dest = tr.bpm_level = None
            tr.bpm_step_idx = None
            want = (bpms or {}).get(tr.style_dest_name)
            if want and bool(getattr(tr, "bpm_follow", False)):
                here_bpm = float(c.bpm[n_ctx - 1] if n_ctx else c.bpm[0])
                tr.bpm_leg_from = here_bpm
                tr.bpm_dest = fold_bpm(want, here_bpm)
            # The stepped mode chooses nothing here on purpose. Its cadence is
            # BPM_STEP_HOLD measures, not one leg, so the choice belongs where
            # the leg's progress is known -- below, per take.
        elif not str(getattr(tr, "style_axis", "") or ""):
            # A random heading is re-rolled per leg for the same reason wander
            # re-targets: held forever it walks off into space and never comes
            # back to anything. A NAMED axis is a standing instruction from the
            # user and is left alone -- see walk_step.
            tr.style_dir = None
    else:
        arrived_at = ""

    if mode == "wander":
        # The travelling itself is `steer_style`, i.e. exactly what a typed
        # steer does; the only difference is who chose the destination.
        f0, f1 = done / pace, min(1.0, (done + n_new) / pace)
        steer_style(c, n_ctx, tr.style_leg_from, tr.style_dest, f0, f1)
        # The SAME fractions, so style and tempo arrive on the same measure.
        # Passing f0/f1 rather than re-deriving from the current tempo is the
        # anchor-plus-absolute-fraction rule steer_style's docstring argues for,
        # and it matters more here: re-anchoring each take would make every step
        # a fraction of the REMAINING tempo difference and the ramp would
        # approach the destination without ever reaching it.
        bpm_from, bpm_to = getattr(tr, "bpm_leg_from", None), \
            getattr(tr, "bpm_dest", None)
        # The stepped mode is checked FIRST and the ramp is an `elif`, because
        # both write c.bpm[n_ctx:] and running both would just mean "whichever
        # ran last" -- see the Track.bpm_step comment.
        level, want = getattr(tr, "bpm_level", None), None
        if bool(getattr(tr, "bpm_step", False)):
            want = (bpms or {}).get(tr.style_dest_name)
            hold = max(1, int(getattr(tr, "bpm_hold", BPM_STEP_HOLD)
                              or BPM_STEP_HOLD))
            # Which BPM_STEP_HOLD-measure block of this leg the measures about
            # to be written fall in. `done` moves by n_new per take, so with the
            # default hold of one window this changes every third take and a
            # take's 8 new measures are never split across two constants.
            idx = int(done) // hold
            if want and (level is None
                         or idx != getattr(tr, "bpm_step_idx", None)):
                # Step from the constant being HELD, not from c.bpm, so the
                # jitter written into the last measure does not accumulate into
                # the journey. At the start of a leg there is no constant yet
                # and the audio's own tempo is the honest starting point.
                src = level or float(c.bpm[n_ctx - 1] if n_ctx else c.bpm[0])
                left = max(1, math.ceil((pace - done) / hold))
                level = step_bpm_level(src, want, left)
                tr.bpm_level, tr.bpm_step_idx = level, idx
            # Rewritten every take, not only the take that chose it:
            # shift_window carries the last context measure forward into the
            # new slots, so without this they would settle onto that one
            # jittered sample instead of holding the constant.
            if level:
                steer_bpm_stepped(c, n_ctx, level, seed)
        elif bool(getattr(tr, "bpm_follow", False)) and bpm_to:
            steer_bpm(c, n_ctx, bpm_from, bpm_to, f0, f1)
        tr.style_leg_done = min(pace, done + n_new)
        left = max(0, pace - tr.style_leg_done)
        leg_deg = angle_deg(tr.style_leg_from, tr.style_dest)
        gone = angle_deg(tr.style_origin, c.style[-1])
        head = (f"🧭 wander · leg {tr.style_legs} → “{tr.style_dest_name}”  "
                f"{_bar(f1)} {f1 * 100:.0f}%")
        tail = (f"{left} measures to go" if left else
                "arrived — a new reference is chosen next Continue")
        # Tempo gets its own clause rather than a separate line: it is one leg
        # with two channels, and splitting the readout would suggest otherwise.
        tempo, tempo_diag = "", ""
        if bool(getattr(tr, "bpm_step", False)) and level:
            # The stepped mode has no in-flight fraction to report -- the tempo
            # is AT this block's value and stays there -- so the readout says
            # where it is, where the style's own tempo is, and how much of that
            # gap is still open. Reporting the gap rather than a progress bar
            # is the point: this mode is capped per step and can finish a leg
            # short, and a bar that read 100% while the tempo was 10% out would
            # be describing the style's arrival, not the tempo's.
            aim = (bpms or {}).get(tr.style_dest_name)
            aim = fold_bpm(aim, level) if aim else None
            far = ("" if not aim or abs(aim - level) < 1.0 else
                   f", {aim:.0f} at this style "
                   f"({(level / aim - 1) * 100:+.0f}%)")
            tempo = f"  ·  ♩ holding {level:.0f} BPM{far}"
            tempo_diag = (f", bpm step {level:.0f}"
                          + (f"/{aim:.0f}" if aim else ""))
        elif bpm_to and bpm_from and abs(bpm_to - bpm_from) >= 1.0:
            tempo = (f"  ·  ♩ {c.bpm[-1]:.0f} BPM, heading for "
                     f"{bpm_to:.0f} (from {bpm_from:.0f})")
            tempo_diag = (f", bpm {bpm_from:.0f}→{bpm_to:.0f} "
                          f"(now {c.bpm[-1]:.0f})")
        tr.style_note = (
            f"{head}  ·  {tail}  ·  {leg_deg * (1 - f1):.0f}° of "
            f"{leg_deg:.0f}° left on this leg  ·  {gone:.0f}° from where the "
            f"walk started ({STYLE_FULL_CHANGE_DEG:.0f}° ≈ an unrelated piece)"
            + tempo
            + (f"\n\n*Arrived at “{arrived_at}”. Now heading for "
               f"“{tr.style_dest_name}”.*" if arrived_at else ""))
        return (f"steer[wander] leg {tr.style_legs} "
                f"{f0:.2f}→{f1:.2f} of {leg_deg:.0f}° to "
                f"{tr.style_dest_name}, {gone:.0f}° total" + tempo_diag)

    step_deg = STYLE_FULL_CHANGE_DEG * n_new / pace
    nxt, tr.style_dir = walk_step(here, vecs, mode, step_deg, seed,
                                  axis=axis,
                                  direction=getattr(tr, "style_dir", None))
    if nxt is None:
        tr.style_note = (f"🧭 {mode} stalled — nowhere left to go "
                         f"(too few reference points, or the heading ran out).")
        return ""
    steer_style(c, n_ctx, here, nxt, 0.0, 1.0)
    tr.style_leg_done = min(pace, done + n_new)
    gone = angle_deg(tr.style_origin, c.style[-1])
    frac = gone / STYLE_FULL_CHANGE_DEG
    held = str(getattr(tr, "style_axis", "") or "")
    tr.style_note = (
        f"🧭 {mode} · leg {tr.style_legs}"
        + (f" along “{held}”" if held else "")
        + f"  {_bar(frac)} {gone:.0f}° from where the walk started "
        f"({STYLE_FULL_CHANGE_DEG:.0f}° ≈ an unrelated piece)  ·  "
        f"{step_deg:.1f}°/Continue  ·  "
        + (f"{max(0, pace - tr.style_leg_done)} measures until it picks a new "
           f"heading" if not held else "holding that heading")
        + nearest(c.style[-1]))
    return (f"steer[{mode}] leg {tr.style_legs}: +{step_deg:.1f}°, "
            f"{gone:.0f}° total")


def _end_walk_leg(tr) -> None:
    """Forget the leg in flight, so the next walk starts a fresh journey."""
    tr.style_dest = None
    tr.style_dest_name = ""
    tr.style_leg_from = None
    tr.style_leg_done = 0
    tr.style_leg_key = None
    # The tempo leg rides the style leg exactly (same fractions, same arrival),
    # so it has to be forgotten with it -- a stale bpm_leg_from would have the
    # next leg's ramp start from a tempo the music left several legs ago.
    tr.bpm_leg_from = None
    tr.bpm_dest = None
    # The stepped mode's staircase belongs to the leg too. Left behind, the next
    # leg would hold a tempo chosen for a destination it is no longer going to,
    # and a stale block index would make the first take of that leg look like a
    # continuation of the old staircase and skip choosing a new step at all.
    tr.bpm_level = None
    tr.bpm_step_idx = None


# ---- loop collapse: detection -----------------------------------------------

def loop_score(latents, min_pairs: int = 8):
    """Strongest periodic self-similarity in one window. -> (period, score).

    `latents` is the (B=1, T, N, C) array a render or continuation leaves on
    Track.latents. Each measure is flattened to one vector, the window mean is
    subtracted, and for every lag p the mean cosine between measures p apart is
    taken; the answer is the lag that scores highest.

    THE CENTERING IS THE WHOLE TRICK. Raw latents from one song share a large
    common component -- same instrumentation, same key, same room -- so raw
    cosine between two entirely different measures already reads ~0.8 and the
    threshold that separates "loop" from "song" would have to be found
    empirically per checkpoint. Subtracting the window mean leaves only how each
    measure DIFFERS from the others, which is scale-free and portable: two
    measures that are the same event score ~1.0 by construction, and two that
    are different events score near 0 no matter how alike the song makes them.

    Lags run up to T - min_pairs rather than T//2, because the loops that matter
    are exactly the ones the model cannot see -- period <= 16 fits inside the 16
    frozen context measures (see LOOP_COS) -- and a half-window cap would stop
    at 12 and miss the top of that range. `min_pairs` is what keeps the long
    lags honest: at lag 16 there are 8 pairs, and fewer than that is a statistic
    thin enough to fire on coincidence.

    Returns (0, 0.0) when there is not enough window to say anything. Pure and
    numpy-only so it can be tested on synthetic latents, which is the only way
    it CAN be tested off-GPU -- MockBackend produces no latents at all."""
    a = np.asarray(latents, dtype=np.float32)
    if a.ndim == 4:
        a = a[0]
    if a.ndim < 2:
        return 0, 0.0
    T = a.shape[0]
    max_lag = T - int(min_pairs)
    if T < 2 or max_lag < 1:
        return 0, 0.0

    v = a.reshape(T, -1)
    v = v - v.mean(axis=0, keepdims=True)
    n = np.linalg.norm(v, axis=1)
    # Every measure sitting ON the window mean means every measure is the same
    # measure -- the most complete collapse there is, and centering maps it to
    # all-zeros, where cosine is 0/0. Answered explicitly, because the arithmetic
    # would otherwise report the worst case as the cleanest.
    if not np.any(n > 1e-6 * max(1.0, float(np.abs(a).mean()))):
        return 1, 1.0
    # An individual measure that lands on the mean has no deviation to compare;
    # its similarity to anything is genuinely undefined, so it contributes 0.
    v = v / np.where(n > 1e-8, n, 1.0)[:, None]
    v[n <= 1e-8] = 0.0

    scores = [float((v[:-p] * v[p:]).sum(axis=1).mean())
              for p in range(1, max_lag + 1)]
    best_s = max(scores)
    # Smallest lag within a hair of the best, not the argmax. Every MULTIPLE of
    # a true period scores the same in expectation, so with any jitter at all
    # the argmax lands on whichever multiple the noise happened to favour and a
    # 4-measure loop gets reported as a 12-measure one. The score is what fires
    # the escape either way; this is about the number the listener is shown.
    best_p = 1 + next(i for i, s in enumerate(scores) if s >= best_s - 0.01)
    return best_p, best_s


def note_loop_check(tr, latents) -> str:
    """Score a freshly sampled window and advance the escape state machine.

    Called once per take, AFTER sampling, so what it measures is what the take
    actually produced. The escape it arms is applied to the NEXT take -- there
    is nothing to be done about a window that already exists, and the listener
    is going to hear it either way.

    Three counters on the Track, all of which have to survive the ZeroGPU pickle
    boundary and Undo's snapshot, which is why they live there rather than here:

      loop_hits   consecutive takes at or over LOOP_COS
      loop_miss   consecutive takes under it
      loop_level  how hard `loop_escape` is currently pushing, 0..3

    Escalation is one level per LOOP_SUSTAIN sustained hits, so a riff that
    survives the first intervention gets a stronger one rather than the same one
    forever. Relaxation is one level per LOOP_RELAX clean takes, and it unwinds
    rather than resetting: dropping straight back to the user's settings the
    moment the score dips is how a detector oscillates."""
    period, score = loop_score(latents)
    tr.loop_period, tr.loop_cos = int(period), float(score)
    if period <= 0:
        return ""

    if score >= LOOP_COS:
        tr.loop_hits = int(getattr(tr, "loop_hits", 0)) + 1
        tr.loop_miss = 0
        if tr.loop_hits >= LOOP_SUSTAIN:
            tr.loop_hits = 0
            tr.loop_level = min(3, int(getattr(tr, "loop_level", 0)) + 1)
    else:
        tr.loop_hits = 0
        tr.loop_miss = int(getattr(tr, "loop_miss", 0)) + 1
        if tr.loop_miss >= LOOP_RELAX:
            tr.loop_miss = 0
            tr.loop_level = max(0, int(getattr(tr, "loop_level", 0)) - 1)

    lvl = int(getattr(tr, "loop_level", 0))
    tr.loop_note = "" if not lvl else (
        f"🔁 caught a {period}-measure loop (self-similarity "
        f"{score:.2f}) — breaking out of it (level {lvl}/3). "
        f"This backs off on its own once the music moves again.")
    return (f"loop: period {period} cos {score:.3f} "
            f"(fires at {LOOP_COS:.2f}) hits {int(getattr(tr, 'loop_hits', 0))}"
            f" miss {int(getattr(tr, 'loop_miss', 0))} level {lvl}")


# ---- loop collapse: escape --------------------------------------------------

def loop_escape(tr):
    """The next take's sampler overrides, given the current escape level.

    -> (ctx_noise, hist_guidance, jump_deg). At level 0 these are exactly the
    user's own settings and nothing about the run changes.

    THE LADDER:

      1. context noise to 0.45. One change, and it goes most of the way
         immediately rather than creeping -- see LOOP_CTX_NOISE for why a big
         step here is the conservative option and not the bold one. The frozen
         context is what the loop is being copied FROM, so this is the most
         direct possible attack on the copy, and it is inaudible except as a
         slightly looser join.
      2. context noise to 0.60 and history guidance to 1.0. See
         LOOP_HIST_GUIDANCE: the two belong together, because omega is a
         multiplier on information the noised context no longer carries.
      3. the same, plus a style JUMP. Last, and separated by a whole level from
         the rest, because it is the only rung a listener will definitely
         notice -- it moves the music somewhere else on purpose.

    So the first two rungs are the "just noise the history harder" hypothesis,
    taken seriously and given two takes at full strength to work. The jump only
    happens to a riff that survived a context at t=0.6 with no history steering
    at all, which is a riff that is not being held in place by the sampler --
    it is being held in place by the CONDITIONING, and moving the conditioning
    is then the only thing left that addresses it.

    RE-ROLLING THE SEED IS NOT ON THIS LADDER, on purpose. stream_forward
    already passes `seed + k` per take, so every take of a collapsed radio
    ALREADY had fresh noise and looped anyway -- the attractor is in the
    conditioning and the guidance, not in the noise draw, and demonstrably so.
    Adding a seed re-roll would look like an intervention while doing nothing.

    A jump is discontinuous where the walk's slerp is smooth, and that is the
    entire reason it works when the walk does not: a contracting map tracks a
    smooth perturbation adiabatically -- the riff simply gets played in a
    slightly different colour -- but it cannot track a discontinuity, because
    there is no nearby state to slide to."""
    base_cn = float(getattr(tr, "ctx_noise", CONTINUE_CTX_NOISE))
    base_hg = float(getattr(tr, "hist_guidance", HISTORY_GUIDANCE))
    lvl = max(0, min(len(LOOP_CTX_NOISE), int(getattr(tr, "loop_level", 0))))
    if lvl <= 0:
        return base_cn, base_hg, 0.0

    # max/min rather than assignment: the rung is a FLOOR on the escape, not an
    # override of the user. Someone who has already dragged context noise to
    # 0.6 and history guidance to 1.1 by hand has pre-empted rungs 1 and 2, and
    # the rescue must not undo that by dragging either back toward the default.
    cn = max(base_cn, float(LOOP_CTX_NOISE[lvl - 1]))
    hg_target = LOOP_HIST_GUIDANCE[lvl - 1]
    hg = base_hg if hg_target is None else min(base_hg, float(hg_target))
    jump = STYLE_FULL_CHANGE_DEG if lvl >= len(LOOP_CTX_NOISE) else 0.0
    return cn, hg, jump


def apply_loop_escape_style(tr, n_ctx: int, seed: int, points=None) -> str:
    """Overwrite the new measures' style with a jump, at escape level 3.

    Runs AFTER `apply_style_steering` and deliberately clobbers what it just
    wrote, rather than being folded into it. Two reasons, both about blast
    radius: the walk's bookkeeping (`style_leg_done`, the progress readout, the
    origin the "N° from where the walk started" number is measured against) all
    stays correct and untouched, and this works identically when the walk is
    off, when a typed steer is running, and when neither is -- none of which
    share a code path inside `_steer_or_walk`.

    `_end_walk_leg` afterwards is what makes the jump stick: without it the walk
    would resume next take from `style_leg_from`, which is where the loop was,
    and slerp straight back into it.

    Silent no-op when style is dropped window-wide -- with no style channel
    there is no vector to jump, and pretending otherwise would write a
    conditioning signal the model has been told to ignore. The check is
    `style_is_off`, the pure predicate, and NOT `_style_off_gate`, which
    `apply_style_steering` has already run this take and which mutates
    (`active["style"]`, `_style_forced_off`, `style_note`). Running it twice
    happens to be idempotent today; depending on that would be a trap for
    whoever edits it next."""
    if float(getattr(tr, "loop_jump_deg", 0.0)) <= 0.0:
        return ""
    c = tr.cond
    if n_ctx >= c.T or style_is_off(tr):
        return ""
    here = unit(c.style[n_ctx - 1] if n_ctx else c.style[0])
    named = [(str(n), unit(v)) for n, v in (points or [])]
    # `avoid=here` rather than the last destination: the point of a jump is to
    # get AWAY from where the music currently is, and pick_reference's own
    # already-sitting-on-it filter is a 0.995 cosine (~6°), which is far too
    # generous to guarantee that.
    far = [(n, v) for n, v in named
           if angle_deg(here, v) >= 0.5 * float(getattr(tr, "loop_jump_deg",
                                                        STYLE_FULL_CHANGE_DEG))]
    pool = far or [(n, v) for n, v in named if angle_deg(here, v) > 10.0]
    if not pool:
        return "loop-escape: no reference point far enough to jump to"
    rng = np.random.default_rng((int(seed) * 2654435761) & 0x7FFF_FFFF)
    name, dest = pool[int(rng.integers(len(pool)))]
    steer_style(c, n_ctx, here, dest, 1.0, 1.0)
    _end_walk_leg(tr)
    tr.style_dest_name = str(name)
    return (f"loop-escape: jumped {angle_deg(here, dest):.0f}° to {name} "
            f"(in one take, not slerped)")


def steer_measures_left(tr):
    """Measures remaining on a typed steer, or None if none is set.

    The restart rule lives here and nowhere else: a prompt or span the Track
    has not started yet is a full span, not a resumption. `plan_fast_forward`,
    `arm_idle_stream` and `apply_style_steering` all have to agree about the
    finish line, and when they drifted apart the plan and the run disagreed
    about how far there was to go.
    """
    if getattr(tr, "style_target", None) is None:
        return None
    span = max(1, int(getattr(tr, "style_span", STYLE_SPAN)))
    started = ((str(getattr(tr, "style_key", "")), span)
               == getattr(tr, "_style_journey", None)
               and getattr(tr, "style_anchor", None) is not None)
    return span - int(getattr(tr, "style_done", 0)) if started else span


def walk_measures_left(tr) -> int:
    """Measures left on the walk LEG in flight; a whole leg if none is.

    Mirrors `steer_measures_left`, and exists for the same reason: a walk now
    HAS a finish line -- the end of the current journey -- and plan_fast_forward
    has to agree with apply_style_steering about where it is. The restart rule
    is the one in `_steer_or_walk`, so a (mode, pace, axis) the Track has not
    started under is a whole leg rather than a resumption. A leg that has
    already run out reads as a whole one too: the next Continue starts the next
    journey, so that is genuinely what is ahead."""
    pace = max(1, int(getattr(tr, "style_pace", STYLE_PACE)))
    key = (str(getattr(tr, "style_walk", "off") or "off"), pace,
           str(getattr(tr, "style_axis", "") or ""))
    started = (key == getattr(tr, "style_leg_key", None)
               and getattr(tr, "style_leg_from", None) is not None)
    left = pace - int(getattr(tr, "style_leg_done", 0)) if started else pace
    return left if left > 0 else pace


def plan_fast_forward(tr, cap: int = FF_MAX_MEASURES,
                      n_new: int = CONTINUE_MEASURES,
                      idle: str = "refuse") -> tuple[int, str]:
    """How many measures Fast-forward should generate, and what to tell the user.

    Returns (measures, note). measures is a whole number of takes and is 0 when
    there is nothing to fast-forward to -- in that case `note` explains why
    rather than the button quietly doing nothing.

    The two steering modes have genuinely different endpoints:

    * A typed steer HAS one. It finishes at `style_span` measures from the
      anchor, so the remaining distance is whatever the in-flight journey has
      left. The restart rule here has to match `apply_style_steering`'s exactly
      -- a prompt or span the Track has not started yet is a full span, not a
      resumption -- or the plan and the run disagree about the finish line.
    * A walk has one too, and only since it grew legs: a leg is `style_pace`
      measures spent getting to one place, so the remaining distance is what is
      left of the leg in flight (`walk_measures_left`). Before that a walk was
      an endless stroll and one press ran a whole pace-worth from wherever the
      track happened to be, which meant a press could end anywhere.

    `idle="run"` returns a full `cap` when nothing is steering instead of
    refusing. Fast-forward refuses -- it is named after the steer and has no
    finish line without one. Stream does not: it is asked for continuous music,
    and it would rather run unsteered than not run. An *arrived* steer counts
    as idle for that purpose too, which is not a detail: Stream's own steer
    finishes mid-run more often than not, and returning 0 there stopped the
    stream dead after the lead-in.
    """
    n_new = max(1, int(n_new))
    key = str(getattr(tr, "style_key", ""))
    if style_is_off(tr):
        # Style off makes a steer and a walk equally inert (see
        # apply_style_steering), so there is no finish line to run out and no
        # pace that means anything -- this is the idle case whatever the
        # controls say. Deciding it here as well as there is what keeps the
        # plan and the run agreeing about how far there is to go.
        if idle != "run":
            return 0, ("Style is off, so there is no steer to run out — "
                       "Fast-forward is named after the steer. Turn style "
                       "back on to steer, or use Stream, which is happy to "
                       "run without one.")
        left, what = int(cap), "style off — the model's prior, nothing steering"
    elif getattr(tr, "style_target", None) is not None:
        left = steer_measures_left(tr)
        what = f"steer to “{key}”"
    elif str(getattr(tr, "style_walk", "off") or "off") != "off":
        left = walk_measures_left(tr)
        # Both modes now travel to something and arrive, so a leg-worth of
        # measures really is a leg-worth of distance in both -- what differs is
        # only where they are going. Wander names its destination because it
        # has one; drift's is wherever the heading leads.
        # Only name a destination the run will actually keep. A changed pace or
        # axis abandons the leg in flight (see walk_measures_left), and the name
        # still on the Track then belongs to a journey nobody is taking.
        inflight = (str(getattr(tr, "style_walk", "")),
                    max(1, int(getattr(tr, "style_pace", STYLE_PACE))),
                    str(getattr(tr, "style_axis", "") or "")
                    ) == getattr(tr, "style_leg_key", None)
        dest = str(getattr(tr, "style_dest_name", "")) if inflight else ""
        what = (f"{tr.style_walk} — the rest of this leg"
                + (f", to “{dest}”" if dest and tr.style_walk == "wander"
                   else ""))
    elif idle == "run":
        left, what = int(cap), "plain continuation, nothing steering"
    else:
        return 0, ("Nothing to fast-forward to — Fast-forward runs out the "
                   "steer you asked for, so type a direction in the box above "
                   "or turn on wander/drift first. Plain Continue works fine "
                   "without one.")
    if left <= 0:
        if idle != "run":
            return 0, ("That steer has already arrived — nothing left to run. "
                       "Type somewhere new, or widen the span.")
        left, what = int(cap), f"“{key}” already arrived, holding it"
    take = math.ceil(min(left, cap) / n_new) * n_new
    note = f"⏭ {take} measures ({take // n_new} takes) — {what}"
    if take < left:
        note += (f"; {left - take} measures of it will still be left over "
                 f"(one press caps at {cap} to fit the GPU budget)")
    return take, note


def arm_idle_stream(tr, mode: str = "drift") -> bool:
    """Give an otherwise-unsteered Stream something to move toward, in place.

    Returns True if it armed anything, False if the Track was already steering
    and was left alone.

    A Stream press with nothing typed and no walk hands the model **bit-
    identical conditioning on every take**: style is one constant vector, bpm
    is constant, and `shift_window` drops the DSP signals window-wide, so the
    only things that differ between takes are the seed and the 16 measures of
    frozen context. Measured over a 64-measure press, per-measure style
    movement was 0.0000. Asking a diffusion-forcing model the same question
    over and over while feeding it its own last answer is the textbook way to
    make it converge, and it is audible as the take looping.

    Stream is where that bites hardest -- it is the one control that runs take
    after take with no chance to intervene, and since it now runs until it is
    stopped there is no take count at which it stops mattering -- so an idle
    Stream drifts instead of standing still. It uses the pace already in the
    walk controls (STYLE_PACE = a full change per 96 measures, i.e. 7.5 deg per
    take), which is slow enough to stay recognisably the same piece across a
    long press and fast enough that no two takes are asked for the same thing.
    Anyone who wants the static behaviour still has Continue and Fast-forward.

    This is a mitigation, not a cure. It only moves *style*, and style is a
    single CLAP vector per measure: it can change what the music sounds like
    without making it any less repetitive. The conditioning that would actually
    say "denser here, these chords next" is the DSP signals, and continuation
    currently drops all of them, because `shift_window` has no values to put in
    the new measures and a partially-present signal is off-distribution for
    this model (see the p_ind_low/p_ind_high note there). Authoring those
    values instead of dropping them -- extending the context's measured curves
    forward, all-on across the whole window, which IS in-distribution -- is the
    real fix, and a much larger change than this one.
    """
    if style_is_off(tr):
        # Nothing to arm: a walk moves the style vector, and the style vector
        # is not being read. Style off is its own answer to the same problem,
        # and arguably a more direct one -- a walk keeps the conditioning
        # moving so no two takes are asked for exactly the same thing, whereas
        # this stops asking. The null token IS constant across takes, so this
        # is not "varied conditioning"; the claim is only that an absent
        # instruction constrains a rollout less than a fixed one repeated
        # sixteen times. Which of the two actually loops less is a listening
        # question, which is why both are exposed rather than one being wired
        # in as the default.
        return False
    # THE STEER IS CHECKED BEFORE THE WALK, and that order is the whole fix.
    # A typed target outranks the walk in `_steer_or_walk`, so a Track carrying
    # both is not "already walking" -- the walk branch is unreachable. Guarding
    # on the walk first meant an arrived steer was never released whenever a
    # walk mode happened to be selected, and the run held one vector for the
    # rest of the press: measured style movement of exactly 0.0 per take, which
    # is the static conditioning this function exists to prevent, arrived at by
    # choosing the options that sound like they would prevent it.
    left = steer_measures_left(tr)
    if left is not None:
        if left > 0:
            return False        # a steer still in flight is already movement
        # Arrived. `apply_style_steering` gives a typed target priority over
        # the walk and would pin the style to it for every remaining take --
        # static conditioning again, just at the destination instead of at the
        # start. Release it and drift on from where it landed; the style is
        # already the target, so nothing jumps. The steer box is unchanged, so
        # the next Continue re-applies it.
        #
        # Releasing does not violate the steer/walk exclusivity the UI enforces:
        # exclusivity holds while the steer is IN FLIGHT. Once it has arrived
        # there is no steer left to be exclusive with, so if a walk is somehow
        # still set -- a Track built by API or restored from a pickle, since the
        # UI forces "off" alongside text -- letting it take over from here beats
        # overriding it with `mode`.
        tr.style_target = None
    if str(getattr(tr, "style_walk", "off") or "off") != "off":
        return False            # a walk really is already movement
    tr.style_walk = str(mode)
    return True


def signals_for_stage(stage: int) -> list[str]:
    """The subset of ALL_SIGNALS that `stage` actually reads (order preserved)."""
    if int(stage) == 1:
        return [s for s in ALL_SIGNALS if s in STAGE1_SIGNALS]
    return list(ALL_SIGNALS)


# ---- the chroma channel as a key --------------------------------------------
# One 12-d template per (root, mode), defined once here: both backends' set_key
# write it and the Edit tab reads it back to fill the key control. A key is not
# something a Track stores -- chroma is twelve continuous numbers per measure,
# which is strictly more than a key name -- so "what key is this in" can only be
# answered by asking which template the measures most look like.
def key_template(root: int, minor: bool) -> np.ndarray:
    """The chroma row that stands for a key: scale degrees lifted above the
    rest, the tonic lifted again."""
    scale = [0, 2, 3, 5, 7, 8, 10] if minor else [0, 2, 4, 5, 7, 9, 11]
    t = np.full(12, 0.35)
    for i in scale:
        t[(int(root) + i) % 12] += 0.4
    t[int(root) % 12] += 0.2
    return np.clip(t, 0, 1)


def key_from_chroma(chroma) -> tuple[int, bool]:
    """Best-fitting (root, minor) for one or more chroma rows.

    Correlation against all 24 templates, not distance: chroma extracted from
    real audio has its own level and contrast, and only the shape -- which
    degrees sit above the mean -- carries the key. Exact for anything set_key
    wrote; a reasonable reading of anything else. A relative major/minor pair
    has the same seven degrees and is told apart only by the tonic's extra
    lift, which is the one thing that distinguishes them in what set_key
    writes too. A flat row has no key and falls out as C major."""
    v = np.asarray(chroma, dtype=float).reshape(-1, 12).mean(axis=0)
    v = v - v.mean()
    best, out = -np.inf, (0, False)
    for minor in (False, True):
        for root in range(12):
            t = key_template(root, minor)
            t = t - t.mean()
            score = float(v @ t / (np.linalg.norm(v) * np.linalg.norm(t) + 1e-9))
            if score > best:
                best, out = score, (root, bool(minor))
    return out


@dataclass
class Conditioning:
    """Per-measure conditioning signals. One row per measure (length T)."""
    style: np.ndarray      # (T, STYLE_DIM)
    chroma: np.ndarray     # (T, 12)
    rms: np.ndarray        # (T,)
    density: np.ndarray    # (T,)
    zcr: np.ndarray        # (T,)
    flatness: np.ndarray   # (T,)
    bpm: np.ndarray        # (T,)
    # Per-measure, per-signal activation. active[sig] is a (T,) bool array:
    # True  -> that signal is CONDITIONED on for that measure
    # False -> that signal is DROPPED (model sees its null token) for that measure
    # This maps directly onto the model's `unconditional_mask` (which is the
    # negation) and is exactly the token-level dropping the model was trained
    # with (multi_token_drop, independent mode). Defaults to all-True (every
    # signal conditioned everywhere) so existing behavior is unchanged.
    active: Optional[dict] = None

    def __post_init__(self):
        if self.active is None:
            T = self.rms.shape[0]
            self.active = {s: np.ones(T, dtype=bool) for s in ALL_SIGNALS}
        else:
            # ensure every signal has a full-length bool array
            T = self.rms.shape[0]
            for s in ALL_SIGNALS:
                a = self.active.get(s)
                if a is None:
                    self.active[s] = np.ones(T, dtype=bool)
                else:
                    self.active[s] = np.asarray(a, dtype=bool)

    @property
    def T(self) -> int:
        return self.rms.shape[0]

    def is_unconditional(self) -> bool:
        """True if NO signal is active in ANY measure (pure init from noise)."""
        return not any(bool(self.active[s].any()) for s in ALL_SIGNALS)

    def set_active(self, signal: str, start: int, end: int, on: bool):
        """Toggle a signal on/off over measures [start, end)."""
        assert signal in ALL_SIGNALS, f"unknown signal {signal}"
        self.active[signal][start:end] = on

    def copy(self) -> "Conditioning":
        return copy.deepcopy(self)

    # Display order for as_curves() -- deliberately not EDITABLE_SIGNALS' order
    # (rms/zcr read as a pair, then the two texture signals, then tempo).
    _CURVE_ORDER = ("rms", "zcr", "density", "flatness", "bpm")

    def as_curves(self, signals: Optional[list] = None) -> dict:
        """Signals a user can see/edit as 1-D curves (style/chroma are not 1-D).

        `signals` restricts the result to a subset -- pass signals_for_stage()
        so a stage-1 track only shows the curves that stage actually reads."""
        return {SIGNAL_LABELS[s]: getattr(self, s) for s in self._CURVE_ORDER
                if signals is None or s in signals}

    # ---- (de)serialization ---------------------------------------------------
    # The seven signal arrays fully describe a conditioning trajectory. Persist
    # them (plus the per-measure active flags) to a single .npz so a trajectory
    # can be saved, shared, and reloaded — and, most usefully, so a known-good
    # conditioning array can be dropped in as a preset to sanity-check the whole
    # generate/decode pipeline end to end. Field names in the archive match this
    # dataclass exactly; active flags are stored as active_<signal>.
    _NPZ_SIGNALS = ("style", "chroma", "rms", "density", "zcr", "flatness", "bpm")

    def to_npz(self, path: str) -> str:
        """Write this conditioning to `path` (a .npz). Returns the path."""
        arrays = {s: np.asarray(getattr(self, s), dtype=np.float32)
                  for s in self._NPZ_SIGNALS}
        for s in ALL_SIGNALS:
            arrays[f"active_{s}"] = np.asarray(self.active[s], dtype=bool)
        np.savez(path, **arrays)
        return path

    @classmethod
    def from_npz(cls, path: str) -> "Conditioning":
        """Build a Conditioning from a .npz holding the seven signal arrays.

        Required keys: style (T,STYLE_DIM), chroma (T,12), and rms/density/zcr/
        flatness/bpm (T,). Optional active_<signal> (T,) bool keys restore the
        per-measure on/off flags; when absent every signal defaults to on.
        Raises ValueError on a missing signal or a shape that doesn't line up."""
        with np.load(path, allow_pickle=False) as d:
            missing = [s for s in cls._NPZ_SIGNALS if s not in d]
            if missing:
                raise ValueError(
                    f"{path}: missing conditioning signal(s) {missing}; "
                    f"expected keys {list(cls._NPZ_SIGNALS)}")
            sig = {s: np.asarray(d[s], dtype=np.float32) for s in cls._NPZ_SIGNALS}
            active = None
            act_keys = {s: f"active_{s}" for s in ALL_SIGNALS
                        if f"active_{s}" in d}
            if act_keys:
                active = {s: np.asarray(d[k], dtype=bool)
                          for s, k in act_keys.items()}

        T = sig["rms"].shape[0]
        # Validate the (T, ...) alignment so a malformed file fails loudly here
        # rather than deep inside the model forward.
        expected = {"style": (T, STYLE_DIM), "chroma": (T, 12),
                    "density": (T,), "zcr": (T,), "flatness": (T,), "bpm": (T,)}
        for s, want in expected.items():
            if sig[s].shape != want:
                raise ValueError(
                    f"{path}: signal {s!r} has shape {sig[s].shape}, "
                    f"expected {want}")
        return cls(active=active, **sig)


@dataclass
class Track:
    cond: Conditioning
    # The seed of the LAST sampling call, not an identity for the track: render,
    # continue_track and vary each overwrite it on their way out (both backends).
    # stream_forward runs continue_track once per CONTINUE_MEASURES, passing
    # seed+k, so the radio walks this up by one per take -- a track begun at 42
    # holds 45 after one 32-measure leg. It is what would reproduce the newest
    # take given the same context; nothing here records what began the track, and
    # after any continuation no single seed reproduces the whole of it. Label it
    # accordingly wherever it reaches a user.
    seed: int = 0
    n_chunks: int = N_CHUNKS
    audio: Optional[np.ndarray] = None
    sr: int = SR
    label: str = ""
    # What the style channel was last asked for, in the words it was asked in:
    # `style_text` a free-text prompt, `style_preset` the name of a preset. No
    # backend reads either -- style is the 512-d embedding, and these are only
    # how it was requested -- but the Edit tab fills its two style controls from
    # them, so coming back to a track shows what was written rather than an
    # empty box. Separate fields because they feed separate controls and a
    # preset name is not a prompt: typing one into the text box gets the CLAP
    # embedding of those words, not the preset.
    style_text: str = ""
    style_preset: str = ""
    # Which DiT to sample with: 1 = stage-1 (style + bpm only, better for
    # unconditional/coarse generation), 2 = stage-2 (all 7 signals; the full
    # controllable model). Carried on the Track so it survives the ZeroGPU
    # process boundary (which pickles the Track, not backend state).
    stage: int = 2
    # CFG guidance scale for the next render. Carried on the Track (like `stage`)
    # so it survives the ZeroGPU pickle boundary. 1.0 == a single plain
    # conditional pass (matches training's `{i}_cond.wav`); >1 does joint CFG
    # (guidance 3.0 matches training's `{i}_cfg.wav`). MockBackend ignores it.
    guidance: float = 3.0
    # History-guidance weight for Continue (see HISTORY_GUIDANCE). A separate
    # axis from `guidance`: that one steers toward the conditioning signals,
    # this one toward the frozen context measures. 1.0 == off.
    hist_guidance: float = HISTORY_GUIDANCE
    # Noise level the frozen context is held at during Continue (see
    # CONTINUE_CTX_NOISE). Exposed because the training schedule makes the
    # right value an open empirical question, not because it is a normal knob.
    ctx_noise: float = CONTINUE_CTX_NOISE

    # ---- tempo following (see steer_bpm) -----------------------------------
    # When `wander` heads for a reference point, the tempo heads for that
    # point's piece as well, over the same leg and at the same fractions. A
    # style destination is an embedding of a real recording and that recording
    # had a tempo; arriving at its style while still running at the old speed
    # is arriving half way. Off by setting `bpm_follow` False, which is what
    # someone who has pinned a tempo with the global-BPM control wants.
    #
    # Only `wander` does this, and only because it is the only mode with a
    # NAMED destination to read a tempo off. A typed steer aims at a CLAP text
    # embedding, which has no recording and so no tempo; `drift` aims along an
    # axis rather than at a point. Both leave the tempo alone -- see
    # _steer_or_walk. (Giving text a tempo means inferring one from its nearest
    # style neighbours, which wants a bigger pool than five presets.)
    bpm_follow: bool = False
    bpm_leg_from: Optional[float] = None   # tempo when the current leg began
    bpm_dest: Optional[float] = None       # folded tempo of the leg's target

    # The other tempo-follow: a staircase of constants, each held for
    # BPM_STEP_HOLD measures with white jitter on top, which is the shape real
    # per-measure bpm actually has (see steer_bpm_stepped). Exclusive with
    # `bpm_follow` and outranks it, because the two write the same measures and
    # "both" is not a third behaviour -- it is whichever ran last. Same
    # reasoning as the steer/walk exclusivity in app._steering.
    bpm_step: bool = False
    bpm_hold: int = BPM_STEP_HOLD          # measures one constant is held for
    bpm_level: Optional[float] = None      # the constant being held right now
    bpm_step_idx: Optional[int] = None     # which block of the leg it belongs to

    # ---- loop-collapse escape (see LOOP_COS / note_loop_check) -------------
    # State for the detector that watches a long radio press for the moment it
    # settles onto one riff and starts repeating it verbatim. All of it lives on
    # the Track rather than in the backend because a stream is not one call: it
    # is a chain of @spaces.GPU reservations, each a forked worker that gets the
    # Track by pickle and returns it the same way, so the Track is the only
    # thing that persists across a press -- let alone across a token handover.
    # Undo snapshots it too, which is correct: rewinding to before the collapse
    # should also rewind the rescue.
    loop_level: int = 0       # 0 = nothing wrong, 1..3 = escape strength
    loop_hits: int = 0        # consecutive takes at or over LOOP_COS
    loop_miss: int = 0        # consecutive takes under it
    loop_period: int = 0      # lag of the strongest repeat in the last window
    loop_cos: float = 0.0     # its centered cosine -- ALWAYS recorded, even at
    #                           level 0, because the threshold is the one number
    #                           here that cannot be calibrated without a GPU and
    #                           this is how it gets calibrated from a real run.
    loop_jump_deg: float = 0.0  # one-shot: armed by the sampler, spent by
    #                             apply_loop_escape_style on the same take.
    # User-facing line for when the escape is engaged; rendered like style_note.
    loop_note: str = ""

    # Free-text diagnostic from the last GPU-side operation, surfaced in the UI
    # status line. On ZeroGPU the heavy methods run in a forked @spaces.GPU
    # worker whose stdout does not reach the Space log, so print() is not a
    # usable channel from in there -- the Track is, because it gets pickled back.
    diag: str = ""
    # Base seed for the TOKENIZER's noise, keyed per absolute song measure (see
    # RealBackend._decoder_noise). The decoder is itself a flow-matching model
    # sampling from noise, so re-decoding identical latents used to give a
    # slightly different waveform every time -- which meant Undo/redo did not
    # round-trip, two renders of one edit were never bit-comparable, and every
    # Continue's re-decode of the context disagreed with the audio already
    # playing. Seeding it per SONG measure (not per window slot) fixes all
    # three: a measure decodes the same way no matter which slot it currently
    # occupies or how many times it has been re-rendered. Change this int for a
    # different-but-still-stable decode.
    decode_seed: int = 0

    # ---- style steering (read by continue_track; see steer_style/walk_step) -
    # These live on the TRACK rather than in continue_track's signature on
    # purpose: on ZeroGPU the call goes through a module-level @spaces.GPU
    # wrapper (app_hf._gpu_continue) and the Track is what crosses the fork
    # boundary, so adding a field costs nothing while adding an argument means
    # touching the wrapper too.
    #
    # ---- typed steer ----
    # `style_target` is a unit CLAP vector or None; the UI recomputes it from
    # the steer textbox on every press, so clearing the box clears the steer.
    # The other three are the JOURNEY, and they are what let a steer outlive the
    # 8 measures a single Continue writes: anchor is where it set off from,
    # span is how many measures it should take in total, done is how many of
    # those have been written. `style_key` identifies the steer (the prompt +
    # span) so that changing either restarts the journey while pressing
    # Continue again advances the one in flight. Undo snapshots all of it, so
    # stepping back mid-steer resumes correctly rather than re-cutting.
    style_target: Optional[np.ndarray] = None
    style_anchor: Optional[np.ndarray] = None
    style_span: int = STYLE_SPAN
    style_done: int = 0
    style_key: str = ""

    # ---- automatic walk ----
    # "off" | "wander" | "drift". Ignored while style_target is set -- an
    # explicit destination beats an automatic one.
    style_walk: str = "off"
    # How long ONE LEG of the walk is, in measures. A leg is a single journey
    # with a single destination, and it is worth about STYLE_FULL_CHANGE_DEG of
    # arc: `wander` spends it travelling to one chosen reference point, `drift`
    # spends it holding one heading. When it runs out the walk picks a new
    # destination and starts another -- which is what makes this a radio rather
    # than a track that ends up somewhere and stays. Not a step size: the
    # per-Continue step is derived from it, so the number means the same thing
    # in both modes and stays meaningful if CONTINUE_MEASURES ever moves.
    style_pace: int = STYLE_PACE
    # Named drift direction (see STYLE_AXES); "" == a random axis between two
    # reference points. A NAMED axis is a standing instruction and is held for
    # every leg; a random one is re-rolled at each leg boundary.
    style_axis: str = ""
    # Opaque persisted heading for drift. Persisting it is the whole difference
    # between a heading and a random walk.
    style_dir: Optional[np.ndarray] = None
    # Where the current walk began, so "how far have I come" has an origin.
    style_origin: Optional[np.ndarray] = None

    # ---- the leg of the walk currently in flight ----
    # Same shape as the typed steer's anchor/span/done, and for the same reason:
    # a journey has to outlive the 8 measures one Continue writes, or every
    # Continue re-picks a destination, takes one step toward it and throws it
    # away -- which is a random walk, and a random walk's displacement grows as
    # sqrt(n), so it oscillates around a fixed distance from where it started
    # and keeps coming home. Measured before this existed: 24 takes of `wander`
    # sat between 22 deg and 68 deg of the origin for the whole run and never
    # arrived anywhere. `style_leg_key` is (mode, pace, axis); changing any of
    # them abandons the leg in flight and starts a fresh one, exactly as a
    # changed prompt or span restarts a typed steer.
    style_dest: Optional[np.ndarray] = None   # wander's destination this leg
    style_dest_name: str = ""                 # its label, for the readout
    style_leg_from: Optional[np.ndarray] = None
    style_leg_done: int = 0
    style_leg_key: Optional[tuple] = None
    style_legs: int = 0                       # legs finished + the one in flight
    # Make every wander leg cross to a DIFFERENT song. Defaults on, which is a
    # statement about the current reference set rather than about the feature:
    # 88 points from 5 recordings is five places listed many times, so a free
    # choice spends most legs travelling between two measures of the piece
    # already playing -- real motion in CLAP space, inaudible as music. Turn it
    # off to see the unfiltered behaviour, and turn it off for good once the
    # reference set is large enough that within-song motion goes somewhere.
    # See pick_reference.
    walk_cross_cluster: bool = True

    # ---- where the style has actually BEEN ----------------------------------
    # The conditioning is a 24-measure sliding window, so `cond.style` is the
    # PRESENT, not the past: by measure 56 the style of measures 0-31 has been
    # overwritten and is gone. Nothing else keeps it, which meant the one
    # question worth asking about a walk -- did it actually go where it said,
    # or did it wander in a circle -- had no answer after the second take.
    #
    # So the committed measures are banked here as they fall out of the window
    # (record_style_trail, called from both backends' _bank_window). One row per
    # measure, in order, plus the destination in force when it was written so
    # the map can mark where each leg began. The live window is NOT in here --
    # it is still in `cond.style` and is appended at draw time; duplicating it
    # would double-count the eight measures that are about to be banked.
    #
    # Cost: 512 floats per measure. A 1024-measure radio run is ~2 MB against a
    # prefix_audio of ~130 MB for the same run, so it is not what makes the
    # Track expensive to pickle across the ZeroGPU boundary.
    style_trail: Optional[np.ndarray] = None   # (n_committed, STYLE_DIM)
    style_trail_tag: list = field(default_factory=list)  # dest name per measure
    # Per WINDOW SLOT, the destination that slot was written toward. Stamped by
    # stamp_new_measures at write time and banked into style_trail_tag as the
    # slot falls out; it exists because the answer is only correct at the
    # moment of writing (see there). One leg's destination VECTOR, which the
    # per-measure names cannot carry, is kept in style_leg_log.
    style_window_tag: list = field(default_factory=list)
    style_leg_log: list = field(default_factory=list)

    # ---- style off (unconditional style) -----------------------------------
    # Drop the style channel window-wide on every Continue: `active["style"]`
    # goes all-False, so the model gets its learned `null_style` on every
    # measure and the independent-CFG pass for style disappears. What is left
    # driving the music is the frozen context, the bpm channel and the model's
    # own prior -- nothing is describing what it should sound like.
    #
    # IN-DISTRIBUTION ON STAGE 2, NOT ON STAGE 1, and that difference is the
    # whole reason this is a flag and not just an empty style box. Stage 2
    # calls multi_token_drop(p_joint_uncond=0.1, p_joint_full=0.1,
    # p_ind_uncond=0.5): 80% of the batch takes the independent branch, and
    # inside it EVERY signal, style included, is dropped for the whole sequence
    # with p=0.5, independently of the others. So ~50% of stage-2 training
    # windows carried no style at all and ~20% carried no style while bpm was
    # still present -- which is exactly the configuration this produces. Stage 1
    # runs p_ind_uncond=0, where the only windows without style were the 10%
    # joint-uncond ones that had nothing at all; style-off-with-bpm-on is a
    # combination it has never seen, so style_is_off() refuses it there.
    #
    # All-off across all 24 measures, never partial. Same uniformity rule
    # shift_window is built around: p_ind_low = p_ind_high = 0 in both stages
    # makes the per-measure drop path dead code, so a half-styled window is off
    # the distribution however reasonable it looks.
    #
    # It also changes what the guidance slider does, which is easy to miss.
    # Independent CFG is pred = uncond + Σᵢ g·(condᵢ − uncond), one term per
    # signal that is on anywhere; a continuation with style off has exactly one
    # term left (bpm), where it had two. So the same slider value is a weaker
    # total push, and what it pushes toward is the tempo alone. That is the
    # honest consequence of removing a condition rather than a bug, but if a
    # styled-off run sounds limp before it sounds free, the guidance is the
    # first thing to raise.
    style_off: bool = False

    # Optional author for the conditioning of a window that has just slid.
    # Called as `hook(cond, first_measure)` -> cond, after `shift_window` and
    # after the style steer, by `apply_cond_hook`. None on every track the app
    # makes on its own, which is the whole of its footprint here.
    #
    # It exists because the style channel has an author for its new measures
    # (`apply_style_steering`, which is why a steer or a drift survives the
    # slide) and the five DSP channels have none -- so `shift_window` drops
    # them window-wide rather than invent values nobody knows. A caller that
    # DOES know them for every measure, which is what the Weather tab is, needs
    # somewhere to say so, and post-shift is the only moment where "every
    # measure of this window" is a question with an answer.
    cond_hook: Optional[Callable] = None

    # Has the radio ever run on this track? Only thing it decides is how much
    # already-heard audio the next press leads in with (LEAD_IN_WARM_S): the
    # first press may be talking to a cold worker, later ones are not. On the
    # Track rather than in the app because it has to survive Undo -- stepping
    # back to a snapshot from before the first press should lead in long again.
    streamed: bool = False

    # Human-readable progress line for the steer/walk, rebuilt on every
    # Continue and rendered under the player. Lives on the Track for the same
    # reason `diag` does: it is computed inside the ZeroGPU worker.
    style_note: str = ""

    # Cached model latents (B=1, T, N, C) from the last render. Used by
    # RealBackend to freeze measures during region regen / continuation.
    latents: Optional[np.ndarray] = None

    # ---- continuation ------------------------------------------------------
    # `cond`/`latents` are always exactly one 24-measure WINDOW (see
    # CONTINUE_MEASURES). Continuation slides that window forward, so a track
    # can be longer than its window; these carry the part that has scrolled out.
    #
    #   audio  ==  prefix_audio  ⌢  window_audio      (what you hear/export)
    #
    # Keeping the two halves separately is what lets Regenerate and region-regen
    # keep working after a Continue: they re-render the WINDOW and re-attach the
    # frozen prefix, instead of silently throwing the earlier material away.
    prefix_audio: Optional[np.ndarray] = None
    window_audio: Optional[np.ndarray] = None
    # Downbeats of window_audio, in SECONDS, n_measures+1 of them. Produced by
    # the decode itself (RealBackend._decode) rather than recomputed, so the
    # split point a Continue cuts at is exactly where the measure really landed.
    window_bounds: Optional[list] = None
    prefix_measures: int = 0
    # How the prefix⌢window seam in `audio` was joined: "crossfade" (the two
    # sides are separate tokenizer decodes and need a 20 ms blend) or "butt"
    # (they are an exact split of one array, so blending would smear it).
    # RealBackend._bank_window has to repeat whatever join built the current
    # `audio` when it cuts the prefix back out of it, or the two disagree by a
    # crossfade width -- which is invisible in a single Continue and shows up
    # as a flam once per take under Fast-forward, and as a skip under Stream.
    audio_seam: str = "crossfade"
    # What comes AFTER the window. Empty for a track that has only ever been
    # continued -- the window is the end of the song and there is nothing past
    # it. Non-empty only once `seat_window` has moved the window back into the
    # middle of a track to edit it, which is the whole reason this exists:
    #
    #   audio  ==  prefix_audio  ⌢  window_audio  ⌢  suffix_audio
    #
    # Continuation refuses to run with a suffix in play -- you cannot extend
    # from the middle -- so `_continue_step` re-seats to the tail first and the
    # suffix goes back to being empty. See `ensure_tail`.
    suffix_audio: Optional[np.ndarray] = None
    suffix_measures: int = 0

    # ---- the measure bank ---------------------------------------------------
    # Every measure of the track, not just the 24 in the window. `cond` and
    # `latents` are one window and nothing kept the rest, so a measure that had
    # scrolled out could not be edited at all: its latents were gone and its
    # conditioning with them, leaving only audio. This is that missing store,
    # and it is what lets `seat_window` move the window anywhere in the song by
    # pure slicing rather than by decoding audio back into latents (which would
    # be lossy, would cost GPU time, and would have to reconstruct measure
    # boundaries that nothing records).
    #
    # Filled by `bank_sync`, which both backends call from `_bank_window` -- the
    # moment measures leave the window -- and which `seat_window` calls again on
    # the way past so a window edited since it was banked writes back first.
    # `bank_bounds` is the exception: it is rebuilt by `_finish`, because that
    # is the one place `audio` is assembled and so the one place the seams are
    # known exactly.
    #
    # Cost is ~6 KB a measure, about 2.4 MB over a 400-measure run, against the
    # ~130 MB of prefix_audio already crossing the ZeroGPU pickle boundary on
    # every take. Same argument as style_trail above.
    bank_cond: Optional["Conditioning"] = None       # M measures
    bank_latents: Optional[np.ndarray] = None        # (M, N, C); None on Mock
    # M+1 measure downbeats, in SECONDS, absolute into `audio`. Not into any one
    # segment: the segments are cut FROM this array, so an absolute offset is
    # the only kind that survives the window moving.
    bank_bounds: Optional[list] = None

    @property
    def bpm_global(self) -> float:
        return float(np.mean(self.cond.bpm))

    @property
    def total_measures(self) -> int:
        """Measures of audio, counting what has scrolled out of the window --
        on BOTH sides of it once the window has been seated mid-track."""
        return (int(self.prefix_measures) + self.cond.T
                + int(self.suffix_measures))

    def copy(self) -> "Track":
        return copy.deepcopy(self)


# -----------------------------------------------------------------------------
# The measure bank: every measure of the track, so the window can move
# -----------------------------------------------------------------------------
# Three entry points, and the split between them is not arbitrary -- each one
# runs at the only moment its data is both correct and complete:
#
#   bank_sync(tr)       conditioning + latents for the live window. Called from
#                       both backends' _bank_window (the last moment the
#                       outgoing measures are live) and again from seat_window
#                       (so a window edited since it was banked writes back).
#   bank_bounds(tr,..)  measure downbeats. Rebuilt by _finish, because that is
#                       the one place `audio` is assembled and therefore the one
#                       place the crossfade seams are known rather than guessed.
#   seat_window(tr, f)  move the window to measure f. Pure slicing of arrays
#                       that already exist -- no decode, no re-encode, nothing
#                       lossy, no GPU.
#
# Backends do not subclass any of this; both call the same functions, so Mock
# exercises the whole bookkeeping path on CPU. Mock has no latents at all, which
# is handled by carrying None through rather than by special-casing it.
_COND_ROWS = ("style", "chroma", "rms", "density", "zcr", "flatness", "bpm")


def _cond_blank(n: int, like: "Conditioning") -> "Conditioning":
    """n measures of nothing, to grow the bank into. Never read as
    conditioning -- bank_sync overwrites every row it claims before anyone
    slices it back out.

    Shaped and TYPED from `like`, which is not housekeeping: the two backends
    build their curves differently and the mock's arrive float64, so a bank
    fixed at float32 would round every value on the way in. Seating the window
    away and back would then come back changed in the seventh decimal -- still
    inaudible, but it would make "moving the window is lossless" false, and a
    guarantee that is nearly true is one nobody can reason with."""
    return Conditioning(
        **{k: np.zeros((n,) + np.asarray(getattr(like, k)).shape[1:],
                       dtype=np.asarray(getattr(like, k)).dtype)
           for k in _COND_ROWS},
        active={s: np.ones(n, dtype=bool) for s in ALL_SIGNALS})


def cond_slice(c: "Conditioning", a: int, b: int) -> "Conditioning":
    """Measures [a, b) as a Conditioning of their own. Copies, so the slice and
    the bank cannot alias -- editing a seated window must not silently rewrite
    the bank behind bank_sync's back."""
    return Conditioning(
        **{k: np.asarray(getattr(c, k))[a:b].copy() for k in _COND_ROWS},
        active={s: np.asarray(c.active[s])[a:b].copy() for s in ALL_SIGNALS})


def bank_sync(tr: Track) -> Track:
    """Write the live window's conditioning and latents into the bank.

    Idempotent and cheap, and deliberately writes the WHOLE window rather than
    just the measures about to scroll out: at the moment it is called every one
    of them is current, and "all of it" has no edge cases where "the first
    n_new" has several."""
    if tr is None or getattr(tr, "cond", None) is None:
        return tr
    pre, T = int(tr.prefix_measures), tr.cond.T
    need = pre + T + int(tr.suffix_measures)

    have = 0 if tr.bank_cond is None else tr.bank_cond.T
    if have < need:
        pad = _cond_blank(need - have, tr.cond)
        tr.bank_cond = pad if tr.bank_cond is None else Conditioning(
            **{k: np.concatenate([getattr(tr.bank_cond, k), getattr(pad, k)])
               for k in _COND_ROWS},
            active={s: np.concatenate([tr.bank_cond.active[s], pad.active[s]])
                    for s in ALL_SIGNALS})

    for k in _COND_ROWS:
        getattr(tr.bank_cond, k)[pre:pre + T] = getattr(tr.cond, k)
    for s in ALL_SIGNALS:
        tr.bank_cond.active[s][pre:pre + T] = tr.cond.active[s]

    lat = getattr(tr, "latents", None)
    if lat is not None:
        lat = np.asarray(lat)
        # (1, T, N, C) from the sampler; the bank drops the batch axis because
        # it is not a batch, it is a track.
        lat = lat[0] if lat.ndim == 4 else lat
        if tr.bank_latents is None or len(tr.bank_latents) < need:
            # lat.dtype, not float32, for the reason in _cond_blank: the bank
            # stores what it is given, it does not decide the precision.
            grown = np.zeros((need,) + lat.shape[1:], dtype=lat.dtype)
            if tr.bank_latents is not None:
                grown[:len(tr.bank_latents)] = tr.bank_latents
            tr.bank_latents = grown
        tr.bank_latents[pre:pre + T] = lat
    return tr


def bank_bounds(tr: Track, window_bounds, w_off: float, s_off: float) -> Track:
    """Rebuild `tr.bank_bounds` after `_finish` has reassembled `tr.audio`.

    Only from `prefix_measures` on. The prefix's own measure starts never move:
    `audio` is built prefix-first, so every offset into the prefix is also an
    offset into the song, whichever way this render happened to join the seam.

    `w_off`/`s_off` are where the window and the suffix actually begin, measured
    from the assembled array rather than predicted from CROSSFADE_MS -- the
    crossfade silently degrades to a plain concatenation when either side is
    shorter than the fade, and a predicted offset would be 20 ms wrong exactly
    then."""
    pre, T = int(tr.prefix_measures), len(window_bounds) - 1
    old = list(tr.bank_bounds or [])
    new = old[:pre] + [w_off + float(b) for b in window_bounds[:T]]
    if tr.suffix_audio is None:
        new.append(w_off + float(window_bounds[T]))
    else:
        # The suffix array itself is untouched, so its internal spacing is the
        # old absolute bounds re-based on where it used to start.
        base = old[pre + T]
        new += [s_off + (float(b) - base) for b in old[pre + T:]]
    tr.bank_bounds = new
    return tr


def seat_window(tr: Track, first: int) -> Track:
    """Move the 24-measure window so it starts at song measure `first`.

    A re-partition of arrays that already exist, not a regeneration: `audio` is
    cut at two banked downbeats and the three pieces become prefix / window /
    suffix, while `cond` and `latents` are sliced straight out of the bank. So
    seating away and back is the identity, to the sample and to the bit.

    Clamped to the track: `first` past the last full window lands on the tail,
    which is the seating every continuation needs (see `ensure_tail`)."""
    if tr is None or tr.bank_bounds is None or tr.audio is None:
        return tr
    bank_sync(tr)
    T = tr.cond.T
    total = tr.total_measures
    f = max(0, min(int(first), total - T))
    if f == int(tr.prefix_measures):
        return tr

    b = tr.bank_bounds
    s0 = int(round(b[f] * tr.sr))
    s1 = int(round(b[f + T] * tr.sr))
    tr.prefix_audio = tr.audio[:s0].copy() if s0 > 0 else None
    tr.window_audio = tr.audio[s0:s1].copy()
    tr.suffix_audio = tr.audio[s1:].copy() if s1 < len(tr.audio) else None
    tr.window_bounds = [float(x) - float(b[f]) for x in b[f:f + T + 1]]
    tr.prefix_measures = f
    tr.suffix_measures = total - f - T
    tr.cond = cond_slice(tr.bank_cond, f, f + T)
    tr.latents = (None if tr.bank_latents is None
                  else tr.bank_latents[f:f + T][None].copy())
    # The three pieces are exact cuts of one array, so putting them back
    # together is a butt join. Crossfading would blend 20 ms of a signal
    # against itself and shorten the song every time the window moved.
    tr.audio_seam = "butt"
    return tr


def ensure_tail(tr: Track) -> Track:
    """Seat the window at the end of the track, where continuation needs it.

    A no-op unless the window has been moved back to edit something. It has to
    exist because continuation writes measures AFTER the window and then banks
    the ones that fall off the front -- both of which assume the window is the
    end of the song. Run from the middle it would append the new measures to a
    prefix that already has a suffix behind it, i.e. write the future into the
    middle of the past."""
    if tr is None or int(getattr(tr, "suffix_measures", 0)) <= 0:
        return tr
    return seat_window(tr, tr.total_measures - tr.cond.T)


# -----------------------------------------------------------------------------
# Backend interface
# -----------------------------------------------------------------------------
class ModelBackend:
    """Interface the Gradio app talks to. Implement for real; mock provided."""

    # ---- entry points (produce an on-manifold seed trajectory, then render) --
    def list_presets(self) -> list[str]:
        raise NotImplementedError

    def seed_from_preset(self, name: str, seed: int) -> Track:
        raise NotImplementedError

    def seed_from_text(self, prompt: str, seed: int, bpm: float = 120.0,
                       bpm_variance: float = 2.0) -> Track:
        """Seed from a text prompt. The real backend encodes it with the CLAP
        text tower into the same 512-d space the style channel was trained on;
        bpm/bpm_variance supply the tempo the prompt can't (see bpm_curve)."""
        raise NotImplementedError

    def seed_unconditional(self, seed: int) -> Track:
        """Seed a track with ALL signals inactive -> pure unconditional
        generation from noise. Used to initialize/explore before conditioning."""
        raise NotImplementedError

    # ---- rendering -----------------------------------------------------------
    def render(self, track: Track, n_steps: int = 100) -> Track:
        """Render (or re-render) full audio for the track's conditioning."""
        raise NotImplementedError

    # ---- conditioning activation --------------------------------------------
    def set_active(self, track: Track, signal: str, start: int, end: int,
                   on: bool) -> Track:
        """Turn a signal on/off over measures [start, end). Returns a new Track
        (conditioning changed; not re-rendered)."""
        raise NotImplementedError

    # ---- editing -------------------------------------------------------------
    def apply_shape(
        self,
        track: Track,
        signal: str,
        start: int,
        end: int,
        shape: str,
        params: dict,
        mode: str = "replace",
    ) -> Track:
        """Write a parametric shape into one signal over measures [start, end).

        signal : one of EDITABLE_SIGNALS.
        shape  : one of SHAPES; evaluated to fill every measure in the range.
        params : shape parameters (see eval_shape) -- absolute values in the
                 signal's own units.
        mode   : "replace" overwrites the range with the shape; "add" layers the
                 shape on top of the existing values (for LFO/humanize stacking).

        No clamping or manifold projection is applied -- the signal is written
        exactly as the shape produces it. Returns a new Track (not re-rendered)."""
        raise NotImplementedError

    def set_key(self, track: Track, root: int, minor: bool,
                start: int = 0, end: Optional[int] = None) -> Track:
        """Rewrite the chroma template over a range from key + mode."""
        raise NotImplementedError

    def morph_style(self, track: Track, style_name: str, amount: float,
                    start: int = 0, end: Optional[int] = None) -> Track:
        """Slerp the style embedding over a range toward another preset's style."""
        raise NotImplementedError

    # ---- style steering ------------------------------------------------------
    def embed_style_text(self, prompt: str) -> np.ndarray:
        """A unit CLAP text embedding for `prompt`, for Track.style_target.

        Separate from set_style_from_text because the caller here is the UI
        thread deciding where the NEXT measures should head, not an edit to the
        current window -- and because on ZeroGPU it has to run before the
        @spaces.GPU call, in the CPU process, so that no GPU time is spent
        loading a text tower."""
        raise NotImplementedError

    def style_points(self) -> list:
        """[(name, vec)] — the reference points `wander` aims at and `drift`
        draws its random axes from. Real CLAP audio-tower embeddings of real
        recordings, so every point on a wander geodesic sits between two things
        the model has actually heard.

        DIVERSITY IS THE LIMIT HERE, and it is worth being blunt about it: the
        deployed Space ships three preset songs. Using one vector per preset
        would give a pool of 3 and, for drift, only 3 distinct axes. So the
        pool is every MEASURE of every preset instead (3 x 24, deduped), which
        is legitimate -- the style bin is per-measure and a song's intro and its
        peak really are different points -- but they are three tight clusters,
        not 72 independent styles. A wander confined to them will circle. The
        honest fix for real diversity is more presets; the available one is
        drift along a NAMED axis (see style_axis_vector), which does not depend
        on this pool at all."""
        return []

    def style_bpm(self) -> dict:
        """{name: bpm} over `style_points`' names — the tempo of the recording
        each reference point came from, so `wander` can take the tempo with it
        (see Track.bpm_follow and steer_bpm).

        Empty by default rather than abstract: a backend that has style points
        but cannot say what tempo they came from is a coherent thing to be, and
        tempo following just stays inert for it.

        A name that style_points offers and this does not is fine and means
        exactly "no tempo known for that destination" -- the leg still runs, it
        simply does not move the tempo."""
        return {}

    def style_axis_vector(self, label: str):
        """Ambient direction for a named drift axis, or None for the random one.

        Computed as unit(embed(plus) - embed(minus)) from a contrasting prompt
        pair (see STYLE_AXES). The subtraction is load-bearing: it cancels
        CLAP's roughly-constant text/audio modality offset, which is what makes
        a text-derived heading usable at an audio-derived point."""
        return None

    def regenerate_region(
        self, track: Track, start: int, end: int, seed: int, n_steps: int = 100,
        strength: float = REGEN_STRENGTH
    ) -> Track:
        """Inpaint: resample only measures [start, end), keep the rest fixed.

        `strength` is how far [start, end) is re-rolled -- 0 returns what is
        already there, 1 writes it from noise. See REGEN_STRENGTH."""
        raise NotImplementedError

    def continue_track(self, track: Track, n_new: int = CONTINUE_MEASURES,
                       seed: int = 0, n_steps: int = 100) -> Track:
        """Extend the song by `n_new` measures that follow on from what's there.

        Slides the fixed 24-measure window forward by n_new: the last 24-n_new
        measures are re-presented to the model as FIXED context and only the
        n_new newly exposed slots are sampled, so the continuation is written
        against the real latents of the bars before it rather than starting
        cold. The audio of the measures that fall out of the window is banked in
        prefix_audio; track.audio stays the whole song."""
        raise NotImplementedError

    def fast_forward(self, track: Track, seed: int = 0, n_steps: int = 100,
                     max_measures: int = FF_MAX_MEASURES,
                     idle: str = "refuse") -> Track:
        """Run Continue repeatedly until the active steer reaches its end.

        Length comes from `plan_fast_forward`. The generic implementation here
        is just the loop a user would drive by hand; RealBackend overrides it
        with a version that does the same arithmetic in one GPU call and skips
        the work on intermediate windows that gets thrown away anyway.

        `idle` is passed straight through, and is what lets one button mean two
        things. Advanced's Fast-forward refuses when nothing is steering,
        because it is named after the steer. Simple's runs anyway: its default
        move is "stay close", which is the idle case, so a button that refused
        there would be dead on the tab's own default setting -- the same
        argument `plan_fast_forward` already makes for Stream."""
        tr = track.copy()
        measures, note = plan_fast_forward(tr, max_measures, idle=idle)
        if measures <= 0:
            tr.diag = note
            return tr
        for k in range(measures // CONTINUE_MEASURES):
            tr = self.continue_track(tr, CONTINUE_MEASURES, int(seed) + k,
                                     n_steps)
        tr.diag = f"{note}\n{tr.diag}"
        return tr

    def stream_forward(self, track: Track, seed: int = 0, n_steps: int = 100,
                       max_measures: int = STREAM_LEG_MEASURES,
                       lead_in: bool = True, lead_in_s: float = 0.0):
        """One LEG of a stream: hand back audio as it is produced.

        Yields ``("audio", np.ndarray)`` zero or more times and then exactly one
        ``("track", Track)``. Two kinds of thing come down one channel because
        the ZeroGPU boundary is a single pickled stream; the tag is what lets
        the UI tell a chunk of sound from the finished state.

        A leg is bounded because a GPU reservation is (see
        STREAM_LEG_MEASURES). Open-ended streaming is the caller's job: feed
        the Track this leg returns straight into the next call with
        `lead_in=False` and the two runs join seamlessly. app.on_stream does
        exactly that, and decides when to start the next leg.

        `lead_in` emits already-rendered audio from the end of the track before
        touching the model. It costs nothing and it buys the listener time to
        cover the worst case on ZeroGPU -- a cold worker loading multi-GB
        checkpoints before the first take can start. Only the first leg of a
        press wants it; a later leg would be re-sending audio the listener
        already has. `lead_in_s` caps how much: 0 means the whole current window
        (~48 s at 120 BPM, the cold-start figure), and a positive value takes
        only that many seconds off the end. Presses after the first pass
        LEAD_IN_WARM_S, because on a warm worker the rest of the window is
        insurance that reads as the radio having started over.

        Chunks are disjoint and in order: concatenating them reproduces the
        final `track.audio` from the lead-in point onward, exactly. That holds
        because `audio` only ever grows at the end (see
        RealBackend._bank_window) -- if a backend rewrote earlier audio the
        stream would have already sent the old version.

        The Track arrives only at the end of the leg, so abandoning one mid-leg
        loses that leg. That is the deliberate trade: shipping the Track every
        chunk means pickling the whole song across the process boundary every
        time, which is most of what makes N separate Continues expensive. The
        leg boundary is therefore also the commit point the UI saves at.

        An unsteered press is armed with a slow drift first (`arm_idle_stream`)
        because static conditioning over consecutive takes is what makes this
        model loop."""
        tr = track.copy()
        if tr.audio is None:
            tr = self.render(tr, n_steps=n_steps)
        lead = (len(tr.audio) - len(tr.window_audio)
                if tr.window_audio is not None else 0)
        if float(lead_in_s) > 0:
            lead = max(lead, len(tr.audio) - int(float(lead_in_s) * tr.sr))
        if lead_in:
            yield "audio", np.array(tr.audio[lead:])
        emitted = len(tr.audio)
        arm_idle_stream(tr)
        measures, note = plan_fast_forward(tr, max_measures, idle="run")
        for k in range(measures // CONTINUE_MEASURES):
            tr = self.continue_track(tr, CONTINUE_MEASURES, int(seed) + k,
                                     n_steps)
            yield "audio", np.array(tr.audio[emitted:])
            emitted = len(tr.audio)
        tr.diag = f"{note}\n{getattr(tr, 'diag', '')}"
        yield "track", tr

    def vary(self, track: Track, seed: int, n_steps: int = 100) -> Track:
        """Re-render with a different noise seed (near-neighbor variation)."""
        raise NotImplementedError

    # ---- moving the window ---------------------------------------------------
    def seat_window(self, track: Track, first: int) -> Track:
        """Put the editable window over measures [first, first + T).

        Concrete rather than abstract, and shared by both backends, because it
        is pure array bookkeeping -- no model, no decode, no GPU. See
        backend.seat_window for what it costs (nothing) and why it is exact."""
        return seat_window(track.copy(), int(first))

    def set_stage(self, track: Track, stage: int) -> Track:
        """Select which DiT to sample with (1 or 2). Returns a new Track
        (conditioning unchanged; not re-rendered). Stage 1 conditions on
        style + bpm only; stage 2 uses all signals."""
        raise NotImplementedError

    def set_guidance(self, track: Track, guidance: float) -> Track:
        """Set the CFG guidance scale for future renders. Returns a new Track
        (conditioning unchanged; not re-rendered). 1.0 == plain conditional
        (training's `_cond`); ~3.0 == joint CFG (training's `_cfg`)."""
        raise NotImplementedError

    # ---- global tempo --------------------------------------------------------
    def set_bpm_global(self, track: Track, bpm: float, variance: float,
                       seed: int = 0, start: int = 0,
                       end: Optional[int] = None) -> Track:
        """Write a global tempo + per-measure Gaussian jitter into cond.bpm over
        measures [start, end). See backend.bpm_curve. The result is an ordinary
        signal — apply_shape can still edit it afterwards."""
        raise NotImplementedError

    # ---- CLAP entry points ---------------------------------------------------
    def seed_from_audio(self, path: str, seed: int) -> Track:
        """Extract a full conditioning trajectory from an uploaded audio file
        (all seven signals, style from the CLAP audio tower)."""
        raise NotImplementedError

    def set_style_from_text(self, track: Track, prompt: str, start: int = 0,
                            end: Optional[int] = None) -> Track:
        """Replace the style embedding over [start, end) with the CLAP TEXT
        embedding of `prompt`. Text and audio share one 512-d space, so this is
        the text-conditioning path: everything else keeps whatever it had."""
        raise NotImplementedError


# -----------------------------------------------------------------------------
# Helpers shared by mock (and usable by real for macro directions)
# -----------------------------------------------------------------------------
def _rng(*parts) -> np.random.Generator:
    """Deterministic RNG keyed by arbitrary parts (names, seeds)."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return np.random.default_rng(int(h[:16], 16))


def _smooth(x: np.ndarray, k: int = 3) -> np.ndarray:
    """Light moving-average so trajectories look/behave temporally coherent.

    Always returns an array the same length as ``x`` (for even ``k`` an
    asymmetric pad is used, since ``k//2`` on each side would over-produce by
    one sample under ``np.convolve(..., 'valid')``)."""
    if k <= 1 or x.shape[0] < k:
        return x
    left = k // 2
    right = k - 1 - left  # total pad = k - 1, so 'valid' output length == len(x)
    xp = np.pad(x, (left, right), mode="edge")
    ker = np.ones(k) / k
    return np.convolve(xp, ker, mode="valid")


def bpm_curve(mean: float, variance: float, T: int = N_CHUNKS,
              seed: int = 0) -> np.ndarray:
    """Per-measure BPM from a global tempo plus Gaussian jitter.

    Real recordings don't hold a single tempo — the per-measure BPM the training
    data carries (60 * TARGET_SIG / measure_duration) wanders by a few BPM from
    measure to measure. A flat line is therefore off-manifold. `variance` is the
    standard deviation in BPM (0 == a perfectly flat tempo); the draw is
    deterministic in `seed` so the same controls always give the same curve, and
    the result is clipped to [BPM_MIN, BPM_MAX] so the decoder can't be handed a
    degenerate measure length. Returned verbatim — it stays editable by
    apply_shape like any other signal.

    Snapped to the tempo grid on the way out (see snap_bpm), which is also what
    keeps `variance` honest: below one grid quantum it rounds away entirely
    rather than producing a wobble the model cannot be told about."""
    T = int(T)
    rng = np.random.default_rng(_seed_int("bpm", mean, variance, seed))
    jitter = rng.normal(0.0, max(0.0, float(variance)), T)
    return snap_bpm(np.clip(float(mean) + jitter,
                            BPM_MIN, BPM_MAX)).astype(np.float32)


def _seed_int(*parts) -> int:
    """Stable 64-bit seed from arbitrary parts (so callers can pass floats)."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:16], 16)


# -----------------------------------------------------------------------------
# Shape-function library
# -----------------------------------------------------------------------------
# Each shape maps a measure range of length n -> an (n,) array, from a few
# parameters given in the signal's own absolute units. Nothing here clamps or
# projects; the caller writes the output verbatim into the signal.
#
# Parameter reference (all optional; sensible fallbacks used):
#   constant : value
#   ramp     : start, end, curve   (curve>0 exponent: 1=linear, >1 ease-in,
#                                    <1 ease-out)
#   envelope : low, peak, attack, release  (attack/release are fractions 0..1 of
#                                           the range marking where the peak
#                                           plateau begins/ends)
#   periodic : center, amp, cycles, phase, wave in {sine,triangle,saw}
#   step     : levels (list), or low/high with n_steps
#   random   : center, amount (std), smoothness (>=1 smoothing window), seed
def eval_shape(shape: str, n: int, params: dict) -> np.ndarray:
    n = int(n)
    if n <= 0:
        return np.zeros(0, dtype=float)
    p = params or {}
    tau = np.linspace(0.0, 1.0, n) if n > 1 else np.zeros(1)

    if shape == "constant":
        return np.full(n, float(p.get("value", 0.0)))

    if shape == "ramp":
        a = float(p.get("start", 0.0))
        b = float(p.get("end", 1.0))
        curve = max(1e-3, float(p.get("curve", 1.0)))
        return a + (b - a) * (tau ** curve)

    if shape == "envelope":
        low = float(p.get("low", 0.0))
        peak = float(p.get("peak", 1.0))
        atk = float(np.clip(p.get("attack", 0.3), 0.0, 1.0))
        rel = float(np.clip(p.get("release", 0.7), 0.0, 1.0))
        if rel < atk:
            atk, rel = rel, atk
        y = np.empty(n)
        for i, t in enumerate(tau):
            if t <= atk:
                frac = t / atk if atk > 0 else 1.0
                y[i] = low + (peak - low) * frac
            elif t >= rel:
                frac = (1.0 - t) / (1.0 - rel) if rel < 1.0 else 1.0
                y[i] = low + (peak - low) * frac
            else:
                y[i] = peak
        return y

    if shape == "periodic":
        center = float(p.get("center", 0.0))
        amp = float(p.get("amp", 1.0))
        cycles = float(p.get("cycles", 1.0))
        phase = float(p.get("phase", 0.0))
        wave = p.get("wave", "sine")
        ph = 2 * np.pi * (cycles * tau) + phase
        if wave == "triangle":
            osc = 2.0 / np.pi * np.arcsin(np.sin(ph))
        elif wave == "saw":
            frac = (cycles * tau + phase / (2 * np.pi)) % 1.0
            osc = 2.0 * frac - 1.0
        else:  # sine
            osc = np.sin(ph)
        return center + amp * osc

    if shape == "step":
        levels = p.get("levels")
        if levels is None:
            low = float(p.get("low", 0.0))
            high = float(p.get("high", 1.0))
            n_steps = max(1, int(p.get("n_steps", 2)))
            levels = list(np.linspace(low, high, n_steps))
        levels = [float(v) for v in levels]
        idx = np.minimum((tau * len(levels)).astype(int), len(levels) - 1)
        return np.array([levels[i] for i in idx])

    if shape == "random":
        center = float(p.get("center", 0.0))
        amount = float(p.get("amount", 1.0))
        smooth = max(1, int(p.get("smoothness", 3)))
        r = _rng("shape-random", p.get("seed", 0), n).normal(0, 1, n)
        r = _smooth(r, smooth)
        sd = r.std() + 1e-9
        return center + amount * (r / sd)

    raise ValueError(f"unknown shape: {shape}")


# -----------------------------------------------------------------------------
# Mock backend
# -----------------------------------------------------------------------------
class MockBackend(ModelBackend):
    """No-GPU stand-in. Synthesizes audio *from* the conditioning so that edits
    are actually audible: energy -> loudness, brightness -> filter cutoff,
    density -> event rate, flatness -> noisiness, chroma -> pitch, bpm -> rate.

    This lets you feel the entire interaction loop before the real model is
    attached, and doubles as a spec of what each signal is supposed to do."""

    _PRESETS = {
        "Flamenco (real song)": dict(bpm=120, rms=6.5, density=3.2, zcr=0.14, flat=0.010, root=9, minor=True),
        "Lo-fi boom bap (real song)": dict(bpm=86, rms=4.0, density=2.0, zcr=0.08, flat=0.006, root=0, minor=True),
        "Ambient pad (real song)": dict(bpm=70, rms=2.6, density=1.2, zcr=0.05, flat=0.020, root=7, minor=False),
        "Uptempo house (real song)": dict(bpm=126, rms=7.5, density=4.2, zcr=0.16, flat=0.008, root=2, minor=False),
        "Acoustic ballad (real song)": dict(bpm=76, rms=3.4, density=1.8, zcr=0.09, flat=0.005, root=4, minor=False),
    }

    def list_presets(self) -> list[str]:
        return list(self._PRESETS.keys())

    # ---- conditioning construction ------------------------------------------
    def _cond_from_profile(self, p: dict, key: str) -> Conditioning:
        """Build a full T-measure trajectory with mild, coherent arcs."""
        rng = _rng("profile", key)
        T = N_CHUNKS
        arc = np.sin(np.linspace(0, np.pi, T))  # gentle build-and-release

        rms = _smooth(p["rms"] * (0.7 + 0.5 * arc) + rng.normal(0, 0.15, T))
        density = _smooth(p["density"] * (0.8 + 0.35 * arc) + rng.normal(0, 0.1, T))
        zcr = _smooth(p["zcr"] * (0.9 + 0.2 * arc) + rng.normal(0, 0.004, T))
        flatness = _smooth(np.full(T, p["flat"]) + rng.normal(0, 0.001, T))
        # real songs drift; a flat tempo line is off-manifold (see bpm_curve)
        bpm = bpm_curve(float(p["bpm"]), 1.5, T, seed=_seed_int("bpm", key))

        # chroma: emphasize a scale around the root; one-hot-ish per measure
        chroma = np.full((T, 12), 0.35) + rng.normal(0, 0.05, (T, 12))
        scale = [0, 2, 3, 5, 7, 8, 10] if p["minor"] else [0, 2, 4, 5, 7, 9, 11]
        for i in scale:
            chroma[:, (p["root"] + i) % 12] += 0.4
        chroma = np.clip(chroma, 0, 1)

        # style: fixed random embedding per preset (stands in for the real CLAP vector)
        style_vec = _rng("style", key).normal(0, 1, STYLE_DIM)
        style_vec /= np.linalg.norm(style_vec)
        style = np.tile(style_vec, (T, 1))

        return Conditioning(style, chroma, rms, density, zcr, flatness, bpm)

    # ---- entry points --------------------------------------------------------
    def seed_from_preset(self, name: str, seed: int) -> Track:
        p = self._PRESETS[name]
        cond = self._cond_from_profile(p, key=name)
        tr = Track(cond=cond, seed=seed, label=name, style_preset=name)
        return self.render(tr)

    def seed_from_text(self, prompt: str, seed: int, bpm: float = 120.0,
                       bpm_variance: float = 2.0) -> Track:
        """Mock meta-model: hash the prompt into a plausible profile. (Real
        backend runs the prompt through the CLAP text tower here.) The tempo
        comes from the UI, not the prompt, matching the real backend."""
        rng = _rng("text", prompt.lower().strip())
        p = dict(
            bpm=float(bpm),
            rms=float(rng.uniform(3, 8)),
            density=float(rng.uniform(1.2, 4.5)),
            zcr=float(rng.uniform(0.05, 0.17)),
            flat=float(rng.uniform(0.004, 0.02)),
            root=int(rng.integers(0, 12)),
            minor=bool(rng.integers(0, 2)),
        )
        cond = self._cond_from_profile(p, key="text:" + prompt)
        cond.bpm[:] = bpm_curve(float(bpm), float(bpm_variance), cond.T, seed)
        # Match RealBackend: a prompt only supplies style, and the UI supplies
        # tempo, so the five DSP signals start dropped. The mock still renders
        # from their values (it has no model); the flags drive the coverage plot.
        for s in ("chroma", "rms", "density", "zcr", "flatness"):
            cond.active[s][:] = False
        tr = Track(cond=cond, seed=seed, label=f'text: "{prompt}"',
                   style_text=prompt)
        return self.render(tr)

    def seed_unconditional(self, seed: int) -> Track:
        """All signals inactive -> the model would generate purely from noise.
        The mock still needs *some* values to synthesize plausible audio, so we
        build a neutral trajectory but flip every measure's active flag off."""
        rng = _rng("uncond", seed)
        p = dict(bpm=int(rng.integers(90, 130)), rms=5.0, density=2.5,
                 zcr=0.1, flat=0.01, root=int(rng.integers(0, 12)),
                 minor=bool(rng.integers(0, 2)))
        cond = self._cond_from_profile(p, key=f"uncond:{seed}")
        for s in ALL_SIGNALS:
            cond.active[s][:] = False
        tr = Track(cond=cond, seed=seed, label="unconditional")
        return self.render(tr)

    def set_active(self, track, signal, start, end, on):
        tr = track.copy()
        s, e = self._range(start, end, tr.cond.T)
        tr.cond.set_active(signal, s, e, bool(on))
        return tr

    def set_stage(self, track, stage):
        # Mock has no real model; just record the choice so the UI round-trips.
        tr = track.copy()
        tr.stage = int(stage)
        return tr

    def set_guidance(self, track, guidance):
        # Mock ignores guidance; record it so the UI round-trips.
        tr = track.copy()
        tr.guidance = float(guidance)
        return tr

    def set_bpm_global(self, track, bpm, variance, seed=0, start=0, end=None):
        tr = track.copy()
        c = tr.cond
        e = c.T if end is None else end
        s, e = self._range(start, e, c.T)
        c.bpm[s:e] = bpm_curve(bpm, variance, e - s, seed)
        return tr

    def seed_from_audio(self, path, seed):
        """Mock has no extractor; hash the filename into a profile so the UI
        flow (upload -> track) is exercisable without the real model."""
        import os
        name = os.path.basename(str(path))
        tr = self.seed_from_text(f"upload:{name}", int(seed))
        tr.label = f"upload: {name}"
        return tr

    def set_style_from_text(self, track, prompt, start=0, end=None):
        """Mock text style: a deterministic unit vector from the prompt."""
        tr = track.copy()
        c = tr.cond
        e = c.T if end is None else end
        s, e = self._range(start, e, c.T)
        v = _rng("clap-text", prompt.lower().strip()).normal(0, 1, STYLE_DIM)
        v /= np.linalg.norm(v) + 1e-9
        c.style[s:e] = v
        tr.style_text = str(prompt)
        return tr

    def embed_style_text(self, prompt):
        """Same deterministic prompt->vector map set_style_from_text uses, so
        a mock steer and a mock text-style edit agree on where "warmer" is."""
        v = _rng("clap-text", str(prompt).lower().strip()).normal(0, 1, STYLE_DIM)
        return unit(v)

    def style_points(self):
        # the same per-preset vector _cond_from_profile plants, so `wander`
        # aims at points the mock's own presets actually occupy. The mock has no
        # per-measure style variation to widen the pool with.
        return [(n, unit(_rng("style", n).normal(0, 1, STYLE_DIM)))
                for n in self.list_presets()]

    def style_bpm(self):
        # The same tempo `_cond_from_profile` builds each preset's bpm curve
        # around, so the mock's destinations and its audio agree.
        return {n: float(p["bpm"]) for n, p in self._PRESETS.items()}

    def style_axis_vector(self, label):
        pair = resolve_axis_prompts(label)
        if pair is None:
            return None
        return unit(self.embed_style_text(pair[0])
                    - self.embed_style_text(pair[1]))

    # ---- editing -------------------------------------------------------------
    def apply_shape(self, track, signal, start, end, shape, params, mode="replace"):
        assert signal in EDITABLE_SIGNALS, f"unknown signal {signal}"
        tr = track.copy()
        c = tr.cond
        s, e = self._range(start, end, c.T)
        vals = eval_shape(shape, e - s, params)   # written verbatim, no clamping
        arr = getattr(c, signal)
        if mode == "add":
            arr[s:e] = arr[s:e] + vals
        else:  # replace
            arr[s:e] = vals
        if signal == "bpm":
            # The one signal with a representable set rather than a range. Not
            # a clamp -- an off-grid tempo is not out of bounds, it is a row of
            # nn.Embedding(350) that training never visited, so a drawn ramp
            # through the preset range would hand the model noise on three
            # measures in four. See snap_bpm. Every other signal is z-scored
            # and continuous and keeps the verbatim contract above.
            arr[s:e] = snap_bpm(arr[s:e])
        return tr

    def set_key(self, track, root, minor, start=0, end=None):
        tr = track.copy()
        c = tr.cond
        e = c.T if end is None else end
        s, e = self._range(start, e, c.T)
        c.chroma[s:e] = key_template(root, minor)
        return tr

    def morph_style(self, track, style_name, amount, start=0, end=None):
        tr = track.copy()
        c = tr.cond
        e = c.T if end is None else end
        s, e = self._range(start, e, c.T)
        if style_name in self._PRESETS:
            tgt = self._cond_from_profile(self._PRESETS[style_name], key=style_name).style[0]
            for i in range(s, e):
                c.style[i] = self._slerp(c.style[i], tgt, float(amount))
            tr.style_preset = str(style_name)
        return tr

    def regenerate_region(self, track, start, end, seed, n_steps=100,
                          strength=REGEN_STRENGTH):
        """Mock inpaint: keep audio outside [start,end), re-render inside with a
        new seed. The real backend resamples only those latents, holding the
        kept measures clean at noise level ~0 -- see RealBackend notes.

        `strength` is mixed in the only way a backend with no latents can mix
        it: linearly, in the audio, between what is there and the fresh take.
        That is NOT what RealBackend does -- there the dial moves where the
        denoise starts, and the result is a different performance rather than a
        blend of two. It matches at the ends, which is what can be tested here:
        0 returns the existing audio untouched and 1 is the full re-render."""
        tr = track.copy()
        s, e = self._range(start, end, tr.cond.T)
        full, bounds = self._synthesize(tr.cond, seed=seed)   # fresh noise render
        base = tr.window_audio
        if base is None:
            base, _ = self._synthesize(tr.cond, tr.seed)
        spm = len(full) // tr.cond.T                          # samples per measure
        out = base.copy()
        a, b = s * spm, e * spm
        w = max(0.0, min(1.0, float(strength)))
        if w < 1.0:
            n = min(len(full), len(out))
            full = full[:n] * w + out[:n] * (1.0 - w)
        # crossfade the regenerated region back in to avoid clicks
        out = self._splice(out, full, a, b, spm // 20)
        return self._finish(tr, out, bounds)

    def vary(self, track, seed, n_steps=100):
        tr = track.copy()
        tr.seed = seed
        return self._finish(tr, *self._synthesize(tr.cond, seed=seed))

    def render(self, track, n_steps=100):
        tr = track.copy()
        return self._finish(tr, *self._synthesize(tr.cond, seed=tr.seed))

    def continue_track(self, track, n_new=CONTINUE_MEASURES, seed=0,
                       n_steps=100):
        """Mock continuation: no latents to freeze, so this only exercises the
        window bookkeeping (bank the outgoing measures, slide the conditioning,
        re-render). RealBackend.continue_track is where the past actually
        conditions the future."""
        tr = track.copy()
        n_new = max(1, min(int(n_new), tr.cond.T - 1))
        if tr.window_audio is None:
            tr = self.render(tr, n_steps=n_steps)
        # Continuation writes past the end of the song, so the window has to BE
        # the end of the song. See ensure_tail.
        tr = ensure_tail(tr)
        tr = self._bank_window(tr, n_new)
        tr.cond = shift_window(tr.cond, n_new)
        tr.diag = apply_style_steering(
            tr, tr.cond.T - n_new, int(seed), self.style_points(),
            axis=self.style_axis_vector(getattr(tr, "style_axis", "")),
            bpms=self.style_bpm())
        apply_cond_hook(tr)
        tr.seed = int(seed)
        return self.render(tr, n_steps=n_steps)

    # ---- window bookkeeping (shared shape with RealBackend) ------------------
    @staticmethod
    def _bank_window(tr, n_new):
        """Move the first n_new measures of the window into the frozen prefix."""
        record_style_trail(tr, n_new)
        bank_sync(tr)
        split = int(round(tr.window_bounds[n_new] * tr.sr))
        head = tr.window_audio[:split]
        tr.prefix_audio = (head if tr.prefix_audio is None
                           else np.concatenate([tr.prefix_audio, head]))
        tr.prefix_measures += n_new
        return tr

    @staticmethod
    def _finish(tr, window_audio, bounds):
        """Store a fresh window render and rebuild the full-song audio."""
        tr.window_audio = window_audio
        tr.window_bounds = bounds
        if tr.prefix_audio is None:
            tr.audio, w_off = window_audio, 0.0
        else:
            tr.audio = np.concatenate([tr.prefix_audio, window_audio])
            w_off = len(tr.prefix_audio) / float(tr.sr)
        s_off = 0.0
        if tr.suffix_audio is not None:
            s_off = len(tr.audio) / float(tr.sr)
            tr.audio = np.concatenate([tr.audio, tr.suffix_audio])
        bank_bounds(tr, bounds, w_off, s_off)
        return tr

    # ---- utilities -----------------------------------------------------------
    @staticmethod
    def _range(start, end, T):
        s = max(0, min(int(start), T - 1))
        e = max(s + 1, min(int(end), T))
        return s, e

    _slerp = staticmethod(slerp)

    @staticmethod
    def _splice(base, new, a, b, xf):
        out = base.copy()
        xf = max(1, min(xf, (b - a) // 2))
        out[a + xf:b - xf] = new[a + xf:b - xf]
        fo = np.linspace(1, 0, xf)
        fi = np.linspace(0, 1, xf)
        out[a:a + xf] = base[a:a + xf] * fo + new[a:a + xf] * fi
        out[b - xf:b] = base[b - xf:b] * fi + new[b - xf:b] * fo
        return out

    def _synthesize(self, c: Conditioning, seed: int):
        """Turn conditioning into audio so edits are audible. Purely illustrative
        DSP -- NOT the real model. Each signal maps to a perceptual property.

        Returns (audio, downbeats) to match RealBackend._decode's contract; the
        measures are plain-concatenated here (no crossfade), so the downbeats
        are just the cumulative measure lengths."""
        rng = np.random.default_rng(seed)
        pieces = []
        for i in range(c.T):
            bpm = float(c.bpm[i])
            dur = 60.0 / bpm * TARGET_SIG                     # seconds per measure
            n = int(dur * SR)
            t = np.arange(n) / SR

            # pitch from chroma argmax (root of the measure)
            root = int(np.argmax(c.chroma[i]))
            f0 = 220.0 * (2 ** (root / 12.0))

            # density -> number of note onsets in the measure
            n_notes = max(1, int(round(c.density[i])))
            seg = np.zeros(n)
            onset_idx = np.linspace(0, n, n_notes, endpoint=False).astype(int)
            for oi, onset in enumerate(onset_idx):
                note_len = n // n_notes
                tt = np.arange(note_len) / SR
                # brightness (zcr) -> add higher harmonics / noise
                bright = c.zcr[i] / 0.30
                tone = np.sin(2 * np.pi * f0 * tt)
                tone += 0.5 * bright * np.sin(2 * np.pi * 2 * f0 * tt)
                tone += 0.25 * bright * np.sin(2 * np.pi * 3 * f0 * tt)
                # flatness -> broadband noise mix (noisier = more flat spectrum)
                noise = rng.normal(0, 1, note_len)
                mix = (1 - c.flatness[i] * 6) * tone + (c.flatness[i] * 6) * noise
                # per-note percussive envelope
                env = np.exp(-tt * (6.0 + 4.0 * (c.density[i] / 3.0)))
                end = min(onset + note_len, n)
                seg[onset:end] += (mix * env)[: end - onset]

            # energy (rms) -> measure loudness
            cur = np.sqrt(np.mean(seg ** 2)) + 1e-9
            target = c.rms[i] / 18.0 * 0.5                    # map to a sane level
            seg = seg * (target / cur)
            pieces.append(seg.astype(np.float32))

        audio = np.concatenate(pieces)
        peak = np.max(np.abs(audio)) + 1e-9
        offsets = np.cumsum([0] + [len(p) for p in pieces])
        return ((audio / peak * 0.9).astype(np.float32),
                [float(o) / SR for o in offsets])


# -----------------------------------------------------------------------------
# WAV encoding helper (so the app has zero heavy deps for playback/export)
# -----------------------------------------------------------------------------
def to_wav_bytes(audio: np.ndarray, sr: int = SR) -> bytes:
    audio = np.clip(audio, -1, 1)
    pcm = (audio * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


# -----------------------------------------------------------------------------
# RealBackend skeleton -- fill these in on the box with the model/GPU.
# -----------------------------------------------------------------------------
class RealBackend(ModelBackend):
    """Skeleton wiring the interface onto the trained stack. Import torch etc.
    HERE (not at module top) so MockBackend stays dependency-free.

    Expected collaborators (from your train.py):
        ema.ema_model    : MetaConditionalModernDiTV2 wrapper (has .generate)
        tokenizer, adapter
        predict_measures(...)   -> full render from conditioning
        decode_latents(...)     -> render from given latents (GT path)
    """

    def __init__(self, model, tokenizer, adapter, device="cuda"):
        raise NotImplementedError(
            "Implement on the GPU box. Mapping notes:\n"
            "  seed_from_preset: load precomputed Conditioning for the real song\n"
            "                    (style embedding + extracted per-measure signals),\n"
            "                    then render().\n"
            "  seed_from_text:   run the (fixed) text->conditioning meta-model to\n"
            "                    produce a Conditioning, then render().\n"
            "  render:           build net_kwargs {style,chroma,bpm,rms,density,\n"
            "                    zcr,flatness}; call predict_measures with full-\n"
            "                    conditioning CFG (cfg_mode='joint', tuned scale,\n"
            "                    t_dist='logit'); return waveform.\n"
            "  apply_macros:     move raw signals along DATA-DERIVED directions\n"
            "                    (PCA / paired low-vs-high real measures), then\n"
            "                    project back onto the feature manifold (clamp by\n"
            "                    Mahalanobis distance) before re-render.\n"
            "  regenerate_region:INPAINT. In FMEulerSampler.sample, at each step\n"
            "                    overwrite x_t[:, keep_measures] with the noised\n"
            "                    known latents at that t (add_noise on the fixed\n"
            "                    measures' clean latents), so only [start,end) is\n"
            "                    freely sampled and stays coherent with neighbors.\n"
            "                    Give the DSP signals LEARNED null tokens first so\n"
            "                    masking != mean-imputation.\n"
            "  vary:             same conditioning, new gen_noise/decoder_noise."
        )
