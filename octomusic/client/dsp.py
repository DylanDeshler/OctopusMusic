"""Streaming filtering, contact quality and EEG feature extraction.

Filtering follows the official Octopus GUI / docs: a 4th-order Butterworth
high-pass (the ADCs' own DC-block is ~1 Hz; raw values sit on a large offset)
plus an IIR notch at mains (Q=30). Features are computed from those filtered
signals; a separate low-pass is applied for DISPLAY only, so the EMG band
(65-95 Hz) stays available as a feature.

THE ELECTRODES ARE ALL IN ONE SPOT. Octopus 16 is a 26 mm board with its 16
pins in one cluster, so there is no montage: hemispheric asymmetry, frontal vs
occipital, etc. are meaningless. Channels are redundant views of one site,
which makes the right move a robust AVERAGE over good channels (median PSD),
and it makes contact quality the main reason to look at individual channels.
Where on the head the board sits decides what the features mean: over the
occiput, eyes-closed alpha is huge; on the forehead, blinks and frontalis EMG
dominate.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from octomusic.client.sources import N_CH

BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
    "emg": (65.0, 95.0),       # muscle; avoids 50/60 Hz and the 100 Hz harmonic
}

# Feature catalogue: name -> human description. The mapping UI lists these, and
# a route may also use any arithmetic combination ("0.5*alpha_rel - beta_rel").
FEATURES = {
    # spectral band features (median PSD over good channels)
    "alpha_rel": "alpha / total power (eyes closed, relaxed ↑)",
    "beta_rel": "beta / total power (active thinking ↑)",
    "theta_rel": "theta / total power (drowsy, meditative ↑)",
    "delta_rel": "delta / total power (eye movement, drowsiness ↑)",
    "gamma_rel": "gamma / total power (often muscle-contaminated)",
    "alpha_log": "log10 alpha power (µV²)",
    "beta_log": "log10 beta power (µV²)",
    "theta_log": "log10 theta power (µV²)",
    "delta_log": "log10 delta power (µV²)",
    "total_log": "log10 total 1–45 Hz power",
    "emg_log": "log10 65–95 Hz power (jaw clench / muscle ↑)",
    "relax": "log10 alpha/beta",
    "engage": "log10 beta/(alpha+theta) (engagement index)",
    "theta_beta": "log10 theta/beta",
    "sef90": "spectral edge frequency 90% (Hz)",
    "entropy": "spectral entropy 1–45 Hz (0..1)",
    "iaf": "peak alpha frequency 7–14 Hz",
    # RAW SIGNAL DESCRIPTORS: the EEG waveform treated as if it were audio and
    # measured the way the music's own conditioning is measured. The literal
    # "feed it the raw signal" option, at the only resolution the model takes.
    "sig_rms": "log10 RMS amplitude of the filtered EEG (µV) — like music rms",
    "sig_zcr": "zero crossings per second of the EEG — like music zcr",
    "sig_flatness": "spectral flatness of the EEG 1–45 Hz — like music flatness",
    "sig_density": "amplitude events per second (>2.5σ bursts) — like onset density",
}
for _c in range(N_CH):
    FEATURES[f"ch{_c + 1}_rms"] = f"log10 RMS of channel {_c + 1} alone (µV)"

# 12 log-spaced EEG bands, 2–40 Hz, for the "EEG spectrum → chroma" source.
SPEC12_EDGES = np.geomspace(2.0, 40.0, 13)


def compute_features(seg: np.ndarray, fs: float, use: np.ndarray, artifact_uv: float):
    """Features for one window of FILTERED uV data, seg (n, 16).
    Returns (feats dict, spec12 (12,), artifact fraction, f, psd)."""
    x = seg[:, use]
    art = float(np.mean(np.max(np.abs(x), axis=0) > artifact_uv))
    k = len(seg)
    nper = min(k, int(fs))
    f, p = signal.welch(x, fs=fs, nperseg=nper, noverlap=nper // 2, axis=0)
    psd = np.median(p, axis=1)             # robust average over channels

    def bp(lo, hi):
        m = (f >= lo) & (f < hi)
        return float(np.trapezoid(psd[m], f[m])) if m.sum() > 1 else \
            float(psd[m].sum() * (f[1] - f[0])) if m.any() else 1e-12

    pw = {b: max(bp(*r), 1e-12) for b, r in BANDS.items()}
    total = max(bp(1.0, 45.0), 1e-12)
    m = (f >= 1.0) & (f <= 45.0)
    pm = psd[m] / max(psd[m].sum(), 1e-20)
    cum = np.cumsum(pm)
    sef90 = float(np.interp(0.9, cum, f[m]))       # interpolated, not bin-snapped
    ent = float(-(pm * np.log(pm + 1e-20)).sum() / np.log(len(pm)))
    ma = (f >= 7.0) & (f <= 14.0)
    iaf = float(f[ma][np.argmax(psd[ma])]) if ma.any() else 10.0
    flat = float(np.exp(np.mean(np.log(psd[m] + 1e-20))) / max(np.mean(psd[m]), 1e-20))
    xc = x - x.mean(axis=0)
    rms_ch = np.sqrt(np.mean(seg ** 2, axis=0)) + 1e-6
    zc = np.median(np.mean(np.abs(np.diff(np.signbit(xc).astype(np.int8), axis=0)), axis=0)) * fs
    sd = 1.4826 * np.median(np.abs(xc), axis=0) + 1e-6
    above = np.abs(xc) > 2.5 * sd
    events = np.median(np.sum(above[1:] & ~above[:-1], axis=0)) / (k / fs)
    feats = {
        "alpha_rel": pw["alpha"] / total, "beta_rel": pw["beta"] / total,
        "theta_rel": pw["theta"] / total, "delta_rel": pw["delta"] / total,
        "gamma_rel": pw["gamma"] / total,
        "alpha_log": np.log10(pw["alpha"]), "beta_log": np.log10(pw["beta"]),
        "theta_log": np.log10(pw["theta"]), "delta_log": np.log10(pw["delta"]),
        "total_log": np.log10(total), "emg_log": np.log10(pw["emg"]),
        "relax": np.log10(pw["alpha"] / pw["beta"]),
        "engage": np.log10(pw["beta"] / (pw["alpha"] + pw["theta"])),
        "theta_beta": np.log10(pw["theta"] / pw["beta"]),
        "sef90": sef90, "entropy": ent, "iaf": iaf,
        "sig_rms": np.log10(np.median(rms_ch[use])), "sig_zcr": zc,
        "sig_flatness": flat, "sig_density": events,
    }
    for c in range(N_CH):
        feats[f"ch{c + 1}_rms"] = np.log10(rms_ch[c])
    sp = np.array([max(bp(SPEC12_EDGES[i], SPEC12_EDGES[i + 1]), 1e-12) for i in range(12)])
    sp = np.log10(sp)
    sp = (sp - sp.min()) / max(sp.max() - sp.min(), 1e-9)
    return {k: float(v) for k, v in feats.items()}, sp, art, f, psd


class DSP:
    def __init__(self, fs: float = 250.0):
        self.fs = fs
        self.cfg = {
            "hp_hz": 1.0,
            "notch_hz": 60.0,          # 50 in Europe; 0 = off
            "display_lp_hz": 40.0,     # display only
            "window_s": 2.0,           # feature window
            "channels": [True] * N_CH,
            "auto_exclude": True,      # drop channels failing the contact test
            "artifact_uv": 150.0,      # |x| above this on most channels = artifact
            "contact_lo": 2.0,         # official quickstart's "good pin" RMS range
            "contact_hi": 150.0,
        }
        self._build()
        self.hist = np.zeros((int(10 * fs), N_CH), np.float32)   # filtered
        self.n = 0
        self.contact_rms = np.zeros(N_CH)
        self.good = np.ones(N_CH, bool)
        self.last_psd = None

    def configure(self, **kw):
        rebuild = any(k in kw and kw[k] != self.cfg.get(k)
                      for k in ("hp_hz", "notch_hz", "display_lp_hz"))
        self.cfg.update(kw)
        if rebuild:
            self._build()

    def _build(self):
        fs, c = self.fs, self.cfg
        secs = []
        if c["hp_hz"] > 0:
            secs.append(signal.butter(4, c["hp_hz"], "highpass", fs=fs, output="sos"))
        if c["notch_hz"] and c["notch_hz"] < fs / 2:
            b, a = signal.iirnotch(c["notch_hz"], 30.0, fs=fs)
            secs.append(signal.tf2sos(b, a))
        self.sos = np.concatenate(secs) if secs else None
        self.zi = None
        self.sos_lp = signal.butter(4, c["display_lp_hz"], "lowpass", fs=fs, output="sos") \
            if 0 < c["display_lp_hz"] < fs / 2 else None
        self.zi_lp = None

    def process(self, raw: np.ndarray):
        """raw (n, 16) uV -> (filtered, display) blocks; updates history."""
        x = raw.astype(np.float64)
        if self.sos is not None:
            if self.zi is None:
                self.zi = signal.sosfilt_zi(self.sos)[:, :, None] * x[0][None, None, :]
            y, self.zi = signal.sosfilt(self.sos, x, axis=0, zi=self.zi)
        else:
            y = x - x.mean(axis=0)
        if self.sos_lp is not None:
            if self.zi_lp is None:
                self.zi_lp = np.zeros((self.sos_lp.shape[0], 2, N_CH))
            d, self.zi_lp = signal.sosfilt(self.sos_lp, y, axis=0, zi=self.zi_lp)
        else:
            d = y
        y = y.astype(np.float32)
        n = len(y)
        if n >= len(self.hist):
            self.hist[:] = y[-len(self.hist):]
        else:
            self.hist = np.roll(self.hist, -n, axis=0)
            self.hist[-n:] = y
        self.n += n
        return y, d.astype(np.float32)

    def contact(self):
        """RMS of the last 0.5 s per channel (after the start-up transient)."""
        k = int(0.5 * self.fs)
        if self.n < 2 * self.fs:
            return self.contact_rms, self.good
        seg = self.hist[-k:]
        self.contact_rms = np.sqrt(np.mean((seg - seg.mean(axis=0)) ** 2, axis=0))
        c = self.cfg
        ok = (self.contact_rms >= c["contact_lo"]) & (self.contact_rms <= c["contact_hi"])
        self.good = ok
        return self.contact_rms, ok

    def used_channels(self):
        en = np.asarray(self.cfg["channels"], bool)
        use = en & self.good if self.cfg["auto_exclude"] else en
        if not use.any():
            use = en if en.any() else np.ones(N_CH, bool)
        return use

    def features(self):
        """(feats, spec12, artifact fraction, n channels) for the latest
        window, or None before enough data."""
        fs = self.fs
        k = int(self.cfg["window_s"] * fs)
        if self.n < k + fs:                    # skip the filter warm-up
            return None
        use = self.used_channels()
        feats, spec12, art, f, psd = compute_features(self.hist[-k:], fs, use,
                                                      self.cfg["artifact_uv"])
        self.last_psd = (f, psd)
        return feats, spec12, art, int(use.sum())

    def spectrum(self, fmax=100.0):
        if self.last_psd is None:
            return None
        f, p = self.last_psd
        m = f <= fmax
        return f[m], np.log10(p[m] + 1e-12)
