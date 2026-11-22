"""Streaming filtering, per-channel quality and EEG feature extraction.

THE FILTER CHAIN IS PiEEG's OWN. Their browser SDK (pieeg.js) and dashboard
default to

    Hampel spike removal (window 5, 3 sigma) -> Butterworth band-pass 1-40 Hz
    (order 5) -> IIR notch at mains (Q 30)

(docs: PiEEG-docs pages/software/api/javascript-sdk.mdx). That is the CLEAN
stream: it is what the traces show and what every EEG feature is computed
from. A second, WIDE stream (1 Hz high-pass + notch, no Hampel, no 40 Hz
cut) exists only for the features that live above 40 Hz -- EMG (65-95 Hz) and
gamma -- which the band-pass would otherwise delete, and which Hampel would
partly clip, since muscle bursts are exactly the spiky signal it removes.

QUALITY follows PiEEG's "Contact Calibration" screen (pages/hardware/
signal-quality/Calibration.mdx): per channel, peak-to-peak uV, LINE (mains
pickup) and SNR in dB, with a status of good / fair / poor / railed / no
contact. The front-end can't measure impedance, so -- like theirs -- this
judges contact from the signal. Their exact thresholds are not published; the
ones below are ours and are constants you can tune.

THE ELECTRODES ARE ALL IN ONE SPOT. Octopus 16's 16 pins sit in one 26 mm
cluster, so there is no montage: channels are redundant views of one site, and
the right move is a robust average over the good ones (median PSD). Placement
decides what features mean: over Pz / the back of the head, eyes-closed alpha
is large; on the forehead, blinks and frontalis EMG dominate.
"""

from __future__ import annotations

import numpy as np
from scipy import signal
from numpy.lib.stride_tricks import sliding_window_view

from octomusic.client.sources import N_CH

BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),     # from the WIDE stream
    "emg": (65.0, 95.0),       # from the WIDE stream; avoids 50/60 Hz and 100 Hz
}

# Feature catalogue: name -> description. A route may also use any linear
# combination ("0.5*alpha_rel - beta_rel").
FEATURES = {
    "alpha_rel": "alpha / total power (eyes closed, relaxed ↑)",
    "beta_rel": "beta / total power (active thinking ↑)",
    "theta_rel": "theta / total power (drowsy, meditative ↑)",
    "delta_rel": "delta / total power (eye movement, drowsiness ↑)",
    "gamma_rel": "gamma / total power (often muscle-contaminated)",
    "alpha_log": "log10 alpha power (µV²)",
    "beta_log": "log10 beta power (µV²)",
    "theta_log": "log10 theta power (µV²)",
    "delta_log": "log10 delta power (µV²)",
    "total_log": "log10 total 1–40 Hz power",
    "emg_log": "log10 65–95 Hz power (jaw clench / muscle ↑)",
    "relax": "log10 alpha/beta",
    "engage": "log10 beta/(alpha+theta) (engagement index)",
    "theta_beta": "log10 theta/beta",
    # PiEEG SDK's ratios (bounded 0..1). Their docs: simple band-power ratios,
    # "not a validated" measure of focus/relaxation -- names are shorthand.
    "focus_idx": "PiEEG focus index β/(β+θ), 0–1",
    "relax_idx": "PiEEG relaxation index α/(α+β), 0–1",
    "meditation_idx": "PiEEG meditation index (θ+α)/total, 0–1",
    "sef90": "spectral edge frequency 90% (Hz)",
    "entropy": "spectral entropy 1–40 Hz (0..1)",
    "iaf": "peak alpha frequency 7–14 Hz",
    # RAW SIGNAL DESCRIPTORS: the EEG measured like audio.
    "sig_rms": "log10 RMS amplitude of the clean EEG (µV) — like music rms",
    "sig_zcr": "zero crossings per second of the EEG — like music zcr",
    "sig_flatness": "spectral flatness of the EEG 1–40 Hz — like music flatness",
    "sig_density": "amplitude events per second (>2.5σ bursts) — like onset density",
}
for _c in range(N_CH):
    FEATURES[f"ch{_c + 1}_rms"] = f"log10 RMS of channel {_c + 1} alone (µV)"

SPEC12_EDGES = np.geomspace(2.0, 40.0, 13)    # "EEG spectrum → chroma"

# Quality thresholds (ours; PiEEG doesn't publish theirs).
FULL_SCALE_UV = 1.2 / 4.0 * 1e6               # ±300 mV at gain 4
Q_RAILED_FRAC = 0.95                          # |raw| beyond this x full scale
Q_FLAT_PP = 1.0                               # µV p-p below this: no contact
Q_POOR_PP = 300.0                             # µV p-p above this: poor
Q_GOOD_SNR = 6.0                              # dB
Q_FAIR_SNR = 0.0                              # dB

DEFAULT_DSP = {
    "hampel": True, "hampel_w": 5, "hampel_sigma": 3.0,
    "bp_lo": 1.0, "bp_hi": 40.0, "bp_order": 5,
    "notch_hz": 60.0,          # 50 in Europe; 0 = off
    "window_s": 2.0,           # feature window
    "channels": [True] * N_CH,
    "auto_exclude": True,      # use only channels whose status is good/fair
    "artifact_uv": 150.0,      # |clean x| above this on most channels = artifact
}


class Hampel:
    """Streaming Hampel filter: a sample further than n_sigma * MAD from its
    window's median is replaced by the median. Centered window, so the output
    lags the input by w//2 samples (8 ms at w=5, 250 Hz)."""

    def __init__(self, w=5, n_sigma=3.0):
        self.w = int(w) | 1
        self.k = float(n_sigma)
        self.hist = None

    def __call__(self, x):
        if self.hist is None:
            self.hist = np.repeat(x[:1], self.w - 1, axis=0)
        xx = np.concatenate([self.hist, x], axis=0)
        win = sliding_window_view(xx, self.w, axis=0)            # (n, ch, w)
        med = np.median(win, axis=-1)
        mad = 1.4826 * np.median(np.abs(win - med[..., None]), axis=-1)
        h = self.w // 2
        center = xx[h:h + len(x)]
        out = np.where(np.abs(center - med) > self.k * mad + 1e-9, med, center)
        self.hist = xx[-(self.w - 1):]
        return out


class _SOS:
    """sosfilt with persistent per-channel state."""

    def __init__(self, sos):
        self.sos, self.zi = sos, None

    def __call__(self, x):
        if self.sos is None:
            return x
        if self.zi is None:
            self.zi = signal.sosfilt_zi(self.sos)[:, :, None] * x[0][None, None, :]
        y, self.zi = signal.sosfilt(self.sos, x, axis=0, zi=self.zi)
        return y


def _notch_sos(f0, fs):
    secs = []
    for f in (f0, 2 * f0):                    # mains and its first harmonic
        if f and f < fs / 2 - 2:
            b, a = signal.iirnotch(f, 30.0, fs=fs)
            secs.append(signal.tf2sos(b, a))
    return np.concatenate(secs) if secs else None


class FilterBank:
    """The three streams, with state. Used live (block by block) and offline
    (one block = the whole recording) so both see identical filtering."""

    def __init__(self, fs, cfg):
        c = cfg
        self.hampel = Hampel(c["hampel_w"], c["hampel_sigma"]) if c["hampel"] else None
        hi = min(c["bp_hi"], fs / 2 - 1)
        bp = signal.butter(int(c["bp_order"]), [c["bp_lo"], hi], "bandpass", fs=fs, output="sos")
        notch = _notch_sos(c["notch_hz"], fs)
        cat = lambda *s: np.concatenate([x for x in s if x is not None])  # noqa: E731
        self.clean = _SOS(cat(bp, notch))
        hp = signal.butter(4, c["bp_lo"], "highpass", fs=fs, output="sos")
        self.wide = _SOS(cat(hp, notch))
        self.hp = _SOS(hp)                    # no notch: for the LINE metric

    def __call__(self, raw):
        x = raw.astype(np.float64)
        xc = self.hampel(x) if self.hampel is not None else x
        return (self.clean(xc).astype(np.float32), self.wide(x).astype(np.float32),
                self.hp(x).astype(np.float32))


def channel_quality(clean, hp, raw, fs, notch_hz):
    """PiEEG-style contact estimates for one window. Returns dict of arrays."""
    pp = np.percentile(clean, 99, axis=0) - np.percentile(clean, 1, axis=0)
    nper = min(len(hp), int(fs))
    f, p = signal.welch(hp, fs=fs, nperseg=nper, axis=0)            # (F, ch)
    df = f[1] - f[0]

    def bandp(lo, hi):
        m = (f >= lo) & (f < hi)
        return p[m].sum(axis=0) * df if m.any() else np.zeros(p.shape[1])

    mains = notch_hz or 60.0
    line = bandp(mains - 1.5, mains + 1.5)
    sig = bandp(1.0, 40.0)
    noise = line + bandp(45.0, mains - 2) + bandp(mains + 2, fs / 2 - 1) + 1e-12
    snr = 10 * np.log10(sig / noise + 1e-12)
    railed = np.max(np.abs(raw), axis=0) > Q_RAILED_FRAC * FULL_SCALE_UV
    status = np.where(railed, "railed",
             np.where(pp < Q_FLAT_PP, "no contact",
             np.where((pp > Q_POOR_PP) | (snr < Q_FAIR_SNR), "poor",
             np.where(snr < Q_GOOD_SNR, "fair", "good"))))
    return {"pp": pp, "line": np.sqrt(line), "snr": snr, "status": status,
            "good": np.isin(status, ("good", "fair"))}


def compute_features(clean, wide, fs, use, artifact_uv):
    """Features for one window. Returns (feats, spec12, artifact frac, f, psd)."""
    x = clean[:, use]
    art = float(np.mean(np.max(np.abs(x), axis=0) > artifact_uv))
    k = len(clean)
    nper = min(k, int(fs))

    def med_psd(seg):
        f, p = signal.welch(seg, fs=fs, nperseg=nper, noverlap=nper // 2, axis=0)
        return f, np.median(p, axis=1)              # robust average over channels

    f, psd = med_psd(x)
    _, psd_w = med_psd(wide[:, use])

    def bp(ps, lo, hi):
        m = (f >= lo) & (f < hi)
        if m.sum() > 1:
            return max(float(np.trapezoid(ps[m], f[m])), 1e-12)
        return max(float(ps[m].sum() * (f[1] - f[0])), 1e-12) if m.any() else 1e-12

    pw = {b: bp(psd, *r) for b, r in BANDS.items() if b not in ("gamma", "emg")}
    pw["gamma"] = bp(psd_w, *BANDS["gamma"])
    pw["emg"] = bp(psd_w, *BANDS["emg"])
    total = bp(psd, 1.0, 40.0)
    m = (f >= 1.0) & (f <= 40.0)
    pm = psd[m] / max(psd[m].sum(), 1e-20)
    cum = np.cumsum(pm)
    sef90 = float(np.interp(0.9, cum, f[m]))
    ent = float(-(pm * np.log(pm + 1e-20)).sum() / np.log(len(pm)))
    ma = (f >= 7.0) & (f <= 14.0)
    iaf = float(f[ma][np.argmax(psd[ma])]) if ma.any() else 10.0
    flat = float(np.exp(np.mean(np.log(psd[m] + 1e-20))) / max(np.mean(psd[m]), 1e-20))
    xc = x - x.mean(axis=0)
    rms_ch = np.sqrt(np.mean(clean ** 2, axis=0)) + 1e-6
    zc = np.median(np.mean(np.abs(np.diff(np.signbit(xc).astype(np.int8), axis=0)), axis=0)) * fs
    sd = 1.4826 * np.median(np.abs(xc), axis=0) + 1e-6
    above = np.abs(xc) > 2.5 * sd
    events = np.median(np.sum(above[1:] & ~above[:-1], axis=0)) / (k / fs)
    a, b, t, d = pw["alpha"], pw["beta"], pw["theta"], pw["delta"]
    feats = {
        "alpha_rel": a / total, "beta_rel": b / total, "theta_rel": t / total,
        "delta_rel": d / total, "gamma_rel": pw["gamma"] / (total + pw["gamma"]),
        "alpha_log": np.log10(a), "beta_log": np.log10(b), "theta_log": np.log10(t),
        "delta_log": np.log10(d), "total_log": np.log10(total), "emg_log": np.log10(pw["emg"]),
        "relax": np.log10(a / b), "engage": np.log10(b / (a + t)), "theta_beta": np.log10(t / b),
        "focus_idx": b / (b + t), "relax_idx": a / (a + b), "meditation_idx": (t + a) / total,
        "sef90": sef90, "entropy": ent, "iaf": iaf,
        "sig_rms": np.log10(np.median(rms_ch[use])), "sig_zcr": zc,
        "sig_flatness": flat, "sig_density": events,
    }
    for c in range(N_CH):
        feats[f"ch{c + 1}_rms"] = np.log10(rms_ch[c])
    sp = np.log10(np.array([bp(psd, SPEC12_EDGES[i], SPEC12_EDGES[i + 1]) for i in range(12)]))
    sp = (sp - sp.min()) / max(sp.max() - sp.min(), 1e-9)
    # spectrum for display: clean below 40 Hz, wide above (so EMG is visible)
    disp = np.where(f <= 40.0, psd, psd_w)
    return {k: float(v) for k, v in feats.items()}, sp, art, f, disp


def used_channels(cfg, good):
    en = np.asarray(cfg["channels"], bool)
    use = en & good if cfg["auto_exclude"] else en
    if not use.any():
        use = en if en.any() else np.ones(N_CH, bool)
    return use


class DSP:
    HIST_S = 10.0
    _FILTER_KEYS = ("hampel", "hampel_w", "hampel_sigma", "bp_lo", "bp_hi", "bp_order", "notch_hz")

    def __init__(self, fs: float = 250.0):
        self.fs = fs
        self.cfg = {**DEFAULT_DSP, "channels": [True] * N_CH}
        self.bank = FilterBank(fs, self.cfg)
        n = int(self.HIST_S * fs)
        self.clean = np.zeros((n, N_CH), np.float32)
        self.wide = np.zeros((n, N_CH), np.float32)
        self.hp = np.zeros((n, N_CH), np.float32)
        self.raw = np.zeros((n, N_CH), np.float32)
        self.n = 0
        self.quality = None
        self.good = np.ones(N_CH, bool)
        self.contact_rms = np.zeros(N_CH)       # p-p µV per channel (status API)
        self.last_psd = None

    def configure(self, **kw):
        kw = {k: v for k, v in kw.items() if k in self.cfg}
        rebuild = any(k in self._FILTER_KEYS and kw[k] != self.cfg.get(k) for k in kw)
        self.cfg.update(kw)
        if rebuild:
            self.reset()

    def reset(self):
        """Fresh filter state (new source, new settings)."""
        self.bank = FilterBank(self.fs, self.cfg)
        self.n = 0

    @staticmethod
    def _push(buf, y):
        n = len(y)
        if n >= len(buf):
            buf[:] = y[-len(buf):]
        else:
            buf[:-n] = buf[n:]
            buf[-n:] = y

    def process(self, raw: np.ndarray):
        """raw (n, 16) uV -> clean block (also what the traces display)."""
        c, w, h = self.bank(raw)
        for buf, y in ((self.clean, c), (self.wide, w), (self.hp, h), (self.raw, raw)):
            self._push(buf, y)
        self.n += len(raw)
        return c

    def contact(self):
        if self.n < 3 * self.fs:                  # filter warm-up
            return self.contact_rms, self.good
        k = int(self.fs)
        q = channel_quality(self.clean[-k:], self.hp[-2 * k:], self.raw[-k:], self.fs,
                            self.cfg["notch_hz"])
        self.quality = q
        self.good = q["good"]
        self.contact_rms = q["pp"]
        return self.contact_rms, self.good

    def used_channels(self):
        return used_channels(self.cfg, self.good)

    def features(self):
        fs = self.fs
        k = int(self.cfg["window_s"] * fs)
        if self.n < k + 2 * fs:
            return None
        use = self.used_channels()
        feats, spec12, art, f, psd = compute_features(self.clean[-k:], self.wide[-k:], fs, use,
                                                      self.cfg["artifact_uv"])
        self.last_psd = (f, psd)
        return feats, spec12, art, int(use.sum())

    def spectrum(self, fmax=100.0):
        if self.last_psd is None:
            return None
        f, p = self.last_psd
        m = f <= fmax
        return f[m], np.log10(p[m] + 1e-12)
