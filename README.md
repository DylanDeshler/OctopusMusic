# OctopusMusic

Live EEG from a **PiEEG Octopus 16** conditions the Neural Music Generator in real time.
It is the Weather tab's idea with your brain in place of a city: the same per-measure
conditioning seam (`Track.cond_hook`), the same five signals, and the same two-term
normalizer. The difference is that it runs on your own GPU rather than in a Hugging Face
Space, and the music is played on your laptop.

```
 Octopus 16 ──BLE──▶ laptop: octomusic.client ──WebSocket (SSH tunnel)──▶ GPU: octomusic.server
  16 ch @ 250 Hz      filter → features → dials                         RealBackend._continue_step
                      web UI on :8080          ◀──── int16 audio chunks ── + LiveConditioner
                      speaker (sounddevice)
```

## Two modes

| | **Online** | **Offline** |
|---|---|---|
| input | live EEG (headset, synthetic or live replay) | a recording made with **● Record** |
| output | your speaker, continuous | `recordings/<rec>/offline/<stamp>/output.wav` plus per-measure conditioning, playable in the UI |
| timing | a **fixed brain→ear delay** (default 20 s); the speaker never waits on the model | as fast as the GPU goes; recording time maps onto song time by *time scale* |
| for | performing and listening | finding mappings and normalization stats you trust |

**Online is a fixed-delay playout, not a race.** A measure that will play at wall time
`p` is authored from the EEG at `p − delay` (the default **locked** authoring), so every
measure is heard exactly *delay* seconds after the brain activity that shaped it. The
client holds the speaker silent until `Start + delay`, then plays continuously. It
reports an **anchor** (stream sample *s* plays at time *t*) so the server knows when
every future measure will be heard. The server waits until the EEG for a take's last
measure exists, generates it, and sends it ahead of need.

- **margin** pill: how early each take arrived.
- **min delay** pill: the floor, `take length + compute + 2 s`. Keep **delay** above
  it, and the stream is gapless as long as the GPU keeps up (**RTF** < 1).
- **If a take runs late anyway:** the server stops waiting for EEG, authors from the
  newest it has (a red tick on that measure in the Output panel), and with **auto
  draft** halves the steps for that take.
- **If the queue still empties:** the speaker goes silent once, refills a few seconds,
  and re-anchors. You get one gap, not a stutter on every take.

**Offline** runs the same filters, features and mapper over the recording at the same
0.25 s hop, so a mapping that works offline behaves the same live. It can also do one
thing live can't: **Analyze** computes whole-session statistics for every feature. These
feed the `session_z` / `session_quantile` normalizers, non-causally offline, and as a
fixed reference in later live runs. That's how offline exploration hands its numbers to
a live performance.

## Setup

### GPU box

The repo is self-contained: the model code and presets are bundled in `nmg/`, copied
unmodified from the Neural Music Generator Space (commit in `nmg/SOURCE_COMMIT`).
Only the checkpoints come from elsewhere.

```bash
rsync -av --exclude .venv --exclude recordings --exclude firmware OctopusMusic/ you@gpu-box:~/OctopusMusic/
# on the GPU box
cd ~/OctopusMusic && python3 -m venv .venv && source .venv/bin/activate
pip install torch torchaudio            # the build matching the box's CUDA
pip install -r requirements-server.txt
python -m octomusic.server              # in tmux
```

Checkpoints (`stage2.pt`, `tokenizer.pt`, `adapter.pt`, `style.pt`, about 6.5 GB): copy
them into `~/OctopusMusic/nmg/`, or let the server download them from `MODEL_REPO`
(default `ddeshler/NeuralMusic`; set `HF_TOKEN` if the repo is private).

| env | default | |
|---|---|---|
| `NMG_PATH` | the bundled `nmg/` | point at a Neural_Music_Generator checkout to use live model code instead |
| `STAGE2_CKPT`, `TOKENIZER_CKPT`, `ADAPTER_CKPT`, `STYLE_CKPT` | `stage2.pt`, … | local path (or a filename inside `nmg/`), else downloaded |
| `LOAD_STAGE1` | `0` | live mode uses stage 2 only, so stage 1 is neither downloaded nor loaded |
| `CFG_MODE`, `T_DIST` | `joint`, `logit` | same as the Space |

The server binds to `127.0.0.1:8765`, so nothing is publicly exposed. `--mock` runs the
mock backend (no torch, no GPU) to test the plumbing.

### Laptop

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements-client.txt
ssh -N -L 8765:localhost:8765 you@gpu-box          # leave this running
.venv/bin/python -m octomusic.client --source octopus
```

This opens `http://localhost:8080`. Other source options:

- `--source synthetic`: fake EEG with slider-controlled rhythms. It exercises the whole
  chain without the headset.
- `--source replay --replay recordings/<stamp>`: plays a recorded session back
  through the *online* path in real time.
- `python -m octomusic.client.make_test_recording --minutes 4`: writes a synthetic
  recording with known structure (30 s eyes-open/closed alpha blocks, jaw clenches
  every 20 s, blinks), so offline mode can be tried before any real session.
- `--ble-address <addr>`: skips the Bluetooth scan.
- `--source octopus-usb [--usb-port /dev/cu.usbmodem101]`: the headset over USB-C. It
  needs the OctopusMusic firmware (see **Firmware** below); the port is auto-detected
  by Espressif's USB vendor ID.

On macOS, the terminal (or IDE) running the client needs Bluetooth permission: System
Settings → Privacy & Security → Bluetooth.

**Safety:** only run the headset on its battery while it is on your head. Don't wear it
while it's plugged into USB, because the board has no isolation.

## Firmware and connecting the headset

The XIAO ESP32-S3 shipped with **Seeed's factory camera demo**, which doesn't touch the
ADCs. It has to be flashed. Two images exist:

| image | EEG over | notes |
|---|---|---|
| official `octopus16.bin` v1.0.0 (octoBridge, `octobridge.pieeg.com/flash`) | **BLE only**, one notification per sample | macOS dropped ~50% of samples in testing (125 of 250 Hz arrived) |
| **`firmware/octopus_usb`** (this repo) | **USB-C serial** (send `S`) **and BLE**, batched 4 samples per notification | same packets, UUIDs and name format, so official tools still work |

USB is strictly better for data: no loss, no pairing, and nothing to tune. Bluetooth
remains for untethered use.

**⚠ USB safety:** the board has no electrical isolation. While the headset is worn over
USB, the computer must be **on battery, charger unplugged, nothing mains-powered
attached** (no monitor or dock). Bluetooth avoids the issue.

What was measured on this board, with the board off the head and connected to the laptop:

| | official v1.0.0 | OctopusMusic firmware |
|---|---|---|
| boot | ADC2 found, 16 ch, 250 SPS, CLKIN 8 MHz on D6 | same |
| BLE (macOS) | 125 Hz received, ~51% lost | **249.8 Hz, 0 lost** (20 s) |
| USB-C | no EEG (log text only) | **249.6 Hz, 0 lost** (20 s) |

The full client on `--source octopus-usb` ran at 249.9 Hz with 0% loss and computed
features.

Flashing (from the repo root):

```bash
brew install arduino-cli
arduino-cli config add board_manager.additional_urls https://espressif.github.io/arduino-esp32/package_esp32_index.json
arduino-cli core update-index && arduino-cli core install esp32:esp32@3.3.13   # 3.3.x = ESP-IDF 5.5, like the official build; 4.0 changes the NimBLE API
arduino-cli lib install NimBLE-Arduino
FQBN=esp32:esp32:XIAO_ESP32S3:USBMode=hwcdc,CDCOnBoot=default
arduino-cli compile --fqbn $FQBN firmware/octopus_usb
arduino-cli upload  --fqbn $FQBN -p /dev/cu.usbmodem101 firmware/octopus_usb
```

`firmware/backup/` holds the original Seeed demo (`seeed_demo_20261122.bin`) and the
official image (`official_octopus16_v1.0.0.bin`). Either can be restored with
`esptool --chip esp32s3 -p /dev/cu.usbmodem101 write-flash 0x0 <file>`. If the port
ever disappears, hold **BOOT**, tap **RESET**, release **BOOT**, then flash again.

Serial protocol of the OctopusMusic firmware (115200 baud; the rate is ignored on USB
CDC):

- `S`: start streaming 51-byte packets.
- `X`: stop.
- `I`: print info.
- When not streaming, it prints a heartbeat line once a second
  (`[HB] … rate=250/s usbdrop=…`).

Opening the port resets the chip. The client waits for boot, then sends `S`.

## What's where

| file | role |
|---|---|
| `octomusic/model_space.py` | Signal bands and tempo grid, copied from `weather.py`/`backend.py` so the client doesn't need torch. |
| `octomusic/protocol.py` | Wire format: JSON control/telemetry plus binary int16 audio frames carrying absolute stream sample indices. |
| `octomusic/server/conditioner.py` | **LiveConditioner**: the `cond_hook` (the EEG equivalent of `WeatherTrajectory`). |
| `octomusic/server/engine.py` | Loads `RealBackend`; the online fixed-delay scheduler and the offline loop. |
| `octomusic/client/sources.py` | Octopus BLE reader, synthetic EEG, replay. |
| `octomusic/client/dsp.py` | Filters, contact quality and the feature catalogue. |
| `octomusic/client/mapping.py` | One route per model input: sources, expressions, normalizers, presets. **The open question lives here.** |
| `octomusic/client/offline.py` | Recording → features → session stats → dial timeline. |
| `octomusic/client/audio_out.py` | Scheduled playout (pre-roll, re-buffer, anchors) and speaker output. |
| `octomusic/client/recorder.py` | Session recording: raw EEG, features, dials, takes and output wav. |
| `octomusic/client/app.py`, `static/` | Orchestrator and web UI. |

### Signal processing (PiEEG's published defaults)

PiEEG's browser SDK and dashboard (`pieeg-club/PiEEG-docs`, `javascript-sdk.mdx`)
filter with **Hampel spike removal (window 5, 3σ) → Butterworth band-pass 1–40 Hz,
order 5 → notch at mains (Q 30)**. That is the client's *clean* stream: the traces and
every EEG feature use it. A separate *wide* stream (1 Hz high-pass + notch only) keeps
the EMG (65–95 Hz) and gamma features, which the 40 Hz cut and the spike removal
would otherwise destroy. Live and offline run the identical filter bank; the output
was bit-for-bit equal in testing. All stages are adjustable under the EEG traces.

**Channel quality** follows PiEEG's *Contact Calibration*:
- Each channel is rated by peak-to-peak µV, LINE (mains pickup) and SNR (dB), giving
  a status of good · fair · poor · railed · no contact. PiEEG doesn't publish its
  thresholds, so ours are constants at the top of `dsp.py`.
- Hover a channel number for its numbers.
- Only good and fair channels feed the features (median PSD across them).

**Their placement advice:**
- Put the board over **Pz** or the back of the head for alpha.
- Part the hair and press the board straight down.
- Power it from a power bank only; a strong 50/60 Hz hum means something is on mains.

**New features:** PiEEG's ratios `focus_idx` β/(β+θ), `relax_idx` α/(α+β) and
`meditation_idx` (θ+α)/total. All are bounded 0–1, which makes them easy to map. Their
docs are explicit that these are plain band-power ratios, not validated measures.

### Seeing what goes in and what comes out

- **In → Out panel:** one lane per driven model input. The dashed line is the feature
  coming in (scaled to its range over the window, which is printed on the lane). The
  solid line is the dial going out to the model.
- **Overlay on the EEG traces** (*overlay: in + out / features in / dials out*): the
  same curves drawn over the raw signals on the same time axis, so a blink or clench
  in the traces lines up with what it did to each feature and dial.
- **Output panel:** per measure, the conditioning actually sent versus what the model
  played.

### Octopus 16 facts the code relies on

These come from `pieeg-club/Octopus_16` and `hello_world_octopus_16`:

- **Hardware:** 2× ADS131M08 ADCs, 24-bit, gain 4, Vref 1.2 V, so
  `µV = raw · 1.2/4/8388607 · 1e6`. Sample rate is about 250 SPS; some boards run at
  about 244, so the UI shows the measured rate.
- **Bluetooth:**
  - Controller: XIAO ESP32-S3, BLE only.
  - Service `4fafc201-…914b`, notify characteristic `beb5483e-…26a8`.
  - Packets are 51 bytes: `A0, ctr, 16×int24 BE, C0`, and one notification can carry
    several. The counter is used to detect packet loss.
- **Pin layout:** all 16 pins sit in one 26 mm cluster, so there is **no montage**.
  Channels are redundant views of one site. Features use the median PSD across channels
  with good contact (2–150 µV RMS, the official rule). Asymmetry-style features are
  meaningless here.

## How the conditioning works (and the two rules it keeps)

`LiveConditioner` is installed as `Track.cond_hook`. `_continue_step` calls it last,
after the window slides, so it owns the whole 24-measure window:

1. **Presence is uniform across the window.** Every signal is on for all 24 measures or
   off for all 24. A partial pattern is outside this checkpoint's training distribution:
   `multi_token_drop`'s per-measure path is dead code, and a partial signal comes back
   at clipping volume. Toggling a target's source to `off` therefore drops it
   window-wide at the next take.
2. **Context measures keep the conditioning they were generated with.** Weather got this
   for free from a precomputed table. EEG is only known up to "now", so the conditioner
   keeps a **ledger**: the first time a measure index appears it is *authored* from the
   live dials, and after that it is read back verbatim. This matters most for bpm,
   because a measure's bpm sets its decoded length.

New measures are authored in one of two modes:

- **locked** (default): each measure is authored from the EEG at *its own* mapped time.
  Online that's "when it will play, minus the delay"; offline it's "recording start +
  song time × time scale". *EEG avg s* is averaged back from that point. Because
  measure durations depend on their own bpm, the walk is sequential.
- **hold**: every new measure in a take gets one value. The first *glide* fraction of
  the take ramps from the previous measure. It produces a staircase and is kept for
  comparison.

## Model inputs: every combination

All seven model inputs are configurable in the **Model inputs** panel. **off** means the
input is left unconditional (window-wide), so the model makes it up:

| input | sources |
|---|---|
| rms, density, zcr, flatness | off · **EEG** (a feature or an expression) · manual slider |
| bpm | off (the model gets no tempo token; the decoder still needs a measure length, so it plays at the preset's tempo) · EEG · manual |
| chroma | off · fixed key · **EEG spectrum** (12 log-spaced bands, 2–40 Hz, folded into the 12 chroma bins) · key from a dial (walks the circle of fifths, with hysteresis) |
| style | off · preset A · morph A→B (B is a preset or a text prompt) · axis (a named CLAP direction such as "→ brighter", rotated up to N° by a dial) |

**Features** include:

- band powers (absolute and relative), engagement, relax, θ/β ratio, SEF90, entropy,
  IAF, EMG;
- per-channel amplitude `ch1_rms`…`ch16_rms`;
- **raw signal descriptors**: the EEG waveform measured like audio (`sig_rms`,
  `sig_zcr`, `sig_flatness`, `sig_density`). This is the literal "feed it the raw
  signal" option, at the only resolution the model accepts (one value per measure).

A route's feature can also be a **linear expression**, e.g. `0.5*alpha_rel - beta_rel`.

**Presets** (one click):

- **features**: the argued defaults below.
- **raw signal (EEG as audio)**: the `sig_*` descriptors plus spectrum chroma.
- **brightness only**: one input driven, everything else unconditional.
- **style morph only**.
- **all manual**.
- **unconditional**.

Chroma turns on only once every measure in the window has a real chroma value. After
switching it on mid-run, it takes effect within one window (24 measures), never as a
partial window.

## Normalization: what's built in

Every target is a route **feature → normalizer → dial (0..1) → band → model units**. The
normalizer uses weather's two terms:

```
u = clip(center + gain · 0.15 · z, 0, 1)        then EMA-smoothed (smooth s)
```

`center` is where the music sits when you're at your own baseline (weather's
`G(city)`). `z` is the deviation from that baseline, which is what makes anything move.
At `gain` 1, one standard deviation moves the dial 15% of the band. What differs between
normalizers is how `z` is computed:

| normalizer | z from | good for | catch |
|---|---|---|---|
| `adaptive` | running mean/std, time constant τ | exploring; robust to electrode drift | adapts away sustained states: hold your eyes shut for 2τ and the music drifts back to center |
| `baseline` | mean/std captured with **Calibrate → Baseline** | sustained states stay audible | drift isn't removed |
| `quantile` | rank within the last N s | outlier-proof and bounded | relative only |
| `session_z` | mean/std of the whole recording you last **Analyzed** offline | consistent across sessions; offline it sees the future | needs an analysis first (it falls back to adaptive) |
| `session_quantile` | rank within that whole recording | bounded, uses the full distribution | same |
| `range` | linear between the **Low** and **High** anchors you record | demos ("eyes open = 0, eyes closed = 1") | needs two calibrations |
| `fixed` | linear between hand-entered lo/hi | after you know the numbers | — |

The initial routes are argued guesses, in the same spirit as `weather.MAPPING`:

- brightness ← alpha (inverted): "close your eyes and it goes dark"
- energy ← engagement index, β/(α+θ)
- density ← EMG (65–95 Hz): a jaw clench is a deliberate, unmistakable control, and
  density is the one signal that literally counts events
- tempo ← spectral edge frequency, with a slow EMA so tempo moves least
- flatness ← spectral entropy

During artifacts (more than half the channels over *artifact µV*), routes hold their
value and the normalizer stats freeze. The exception is routes reading `emg_log`, which
is the intended signal there.

**Suggested first session:**

1. **Online**, music off, **● Record** on. Check that the contact dots are green, and
   set the notch to your mains frequency (60 Hz US, 50 Hz EU).
2. Do a few minutes of structured states: eyes open, eyes closed, jaw clenches,
   reading, relaxing. Use **Mark** to note each change.
3. **Offline** tab: pick the recording and press **Analyze**.
   - The stats table and whole-session plot show which features actually separated
     your states.
   - The plot overlays any feature against the dials it produces. Changing a route
     re-maps the preview immediately.
4. Press **Generate** to hear it. The Output panel shows bars (conditioning sent) and
   dots (what the model played, re-measured with the training DSP), so you can see
   which inputs the model really follows.
5. Back **online**, the `session_*` normalizers now use that recording's statistics.
   Set **manual** on an input to hear what it does on its own.

## Audio quality checks

- **Clipping:** the model's output peaks slightly past full scale on about 0.2% of
  10 ms windows. Audio is **soft-limited** (a tanh shoulder above 0.85) before the
  16-bit encode, rather than hard-clipped, which was audible as clicks. This happens
  in `protocol.encode_audio`, on the GPU server.
- **Playout pill** counts three things separately, so any remaining clicks can be
  attributed:
  - *rebuf*: the queue ran dry.
  - *glitch*: the audio callback ran late (CoreAudio underflow).
  - *seam*: a click-sized step where two chunks join.
  
  Hover it for the largest join step and how much of the audio is near full scale.
  The output stream uses a large device buffer; with a 20 s delay, its ~100 ms
  latency is irrelevant.
- **Artifact hold:** dials freeze through artifacts for at most 3 s
  (`Mapper.artifact_hold_s`). A longer run is treated as signal, so noisy pins can't
  leave the music on one conditioning value indefinitely.

## Latency (online)

The delay is a setting, not a side effect: brain→ear equals **delay** for every measure.
The pill measures it, and it should read the delay plus a few tens of milliseconds of
audio-device latency. The constraint is

```
delay  ≥  take length + compute + margin        (shown as "min delay")
```

Take length is `measures/take × measure duration`, about 8 s at the default 8 measures
and roughly one measure per second. On the A10G Space a take took about 7 s, so 20 s is
comfortable there. A faster GPU lowers the floor. Settings that lower it further:

- **measures/take**: fewer measures per take. Each take still costs a full
  24-measure DiT pass, so **RTF** rises; keep it under about 0.8.
- **steps** 16 (draft) and **history guidance** off: each skips DiT passes. With all
  DSP signals on, the window is well specified, so history matters less than it did in
  the Space's continuation mode.

**Conditioning toggles** (Generation panel): *signal guidance* (CFG, with its strength),
*history*, and with history on *history guidance* (with its strength) and *context
noise*. A toggle that is off keeps its strength for when it comes back on. **History is
on and history guidance off by default**: the model reads the previous measures as
context, but doesn't amplify them with history guidance's extra passes. Turning history
off holds the context measures at t = 1.0 (pure noise) for the whole denoise, which is
how the model represents "nothing known here", so every take is generated from the
signals and the seed alone; the cost is looser joins between takes.

- **EEG avg s**, the feature window (2 s) and dial smoothing don't add delay in locked
  mode; they set how much EEG history each measure summarizes.

## Not yet verified

- **The real-model path hasn't run yet.** Everything was exercised end to end against
  NMG's `MockBackend` on this laptop (no torch here). `Engine._load_real` mirrors
  `app_hf.build_backend` without the ZeroGPU wrapper and calls
  `RealBackend._continue_step(..., decode="new")` directly, the same call
  `stream_forward` makes. The first GPU run is where this gets proven.
- **Headset I/O is verified, but not yet on a head.** USB and BLE both stream at
  250 Hz with no loss (see **Firmware**), but all signals so far are from a board on a
  desk.
- **Measured so far** (mock backend with a simulated 2 s compute): brain→ear held at
  exactly the delay for every measure with 0 re-buffers. A too-short delay degraded as
  designed (early authoring plus draft steps). Offline mapped EEG time to song time with
  0.00 s error across 168 measures.
- **Long sessions:** `Track` keeps the whole song's audio in memory (about 0.5 GB per
  hour at 16 kHz × 2 copies). Restart the music for multi-hour runs.
