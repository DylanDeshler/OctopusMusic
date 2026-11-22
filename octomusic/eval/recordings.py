"""Labelled EEG recordings for evaluation.

A Recording is raw µV (n, 16) + timestamps + labelled segments. Sources:

  * this app's recorder: recordings/<stamp>/eeg.csv plus either
      labels.json  [[t0_rel_s, t1_rel_s, "label"], ...]   (explicit), or
      events.jsonl "note" marks: each note's text labels the time until the next
                   note (notes starting with "#" are comments, "-"/"none" = no label).
    So: press Mark with "eyes open", "eyes closed", "clench", "rest" ... as you go.
  * PiEEG xlsx exports (their recorder's format: "Raw data" sheet, header
    `Time (s), Counter, CH1..CH16`, trailing MIN/MAX/MEAN/STD rows). Raw ADC
    counts are detected and converted to µV.
  * `python -m octomusic.eval.recordings fetch` downloads PiEEG's public files and
    writes them in this app's format under eval_data/:
      octopus16_pz_open_closed   Octopus 16, 10 s eyes open + 10 s eyes closed
                                 at Pz (two files concatenated; NB their closed
                                 file shows no alpha increase -- smoke test only)
      octopus16_fp_blink         Octopus 16, 10 s blinking on the forehead
      pieeg16_alpha              PiEEG-16 (ADS1299, dry, different board),
                                 eyes closed/open alternating 5 s; labelled
                                 "block_a"/"block_b" since the docs don't say
                                 which comes first -- separability doesn't care
      pieeg16_chew_blink_1/2     PiEEG-16 chewing then blinking (unlabelled:
                                 the protocol timing isn't published) -- used
                                 for artifact robustness
"""

from __future__ import annotations

import csv
import json
import os
import urllib.request
from dataclasses import dataclass, field

import numpy as np

UV_PER_COUNT = 1.2 / 4.0 / 8388607.0 * 1e6
FS = 250.0


@dataclass
class Recording:
    name: str
    t: np.ndarray               # absolute seconds (any epoch; only differences matter)
    x: np.ndarray               # (n, 16) µV
    fs: float
    labels: list = field(default_factory=list)   # [(t0_abs, t1_abs, label)]

    @property
    def duration(self):
        return float(self.t[-1] - self.t[0])

    def label_at(self, tq):
        """Label per query time (None where unlabelled)."""
        tq = np.atleast_1d(np.asarray(tq, float))
        out = np.array([None] * len(tq), dtype=object)
        for t0, t1, lab in self.labels:
            out[(tq >= t0) & (tq < t1)] = lab
        return out


def _labels_from_events(path, t0, t_end):
    ev = []
    with open(path) as f:
        for line in f:
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("kind") == "note":
                ev.append((float(e["t"]), str(e.get("text", "")).strip()))
    ev.sort()
    labels = []
    for i, (t, text) in enumerate(ev):
        if not text or text.startswith("#") or text.lower() in ("-", "none"):
            continue
        t1 = ev[i + 1][0] if i + 1 < len(ev) else t_end
        labels.append((max(t, t0), min(t1, t_end), text.lower()))
    return labels


def load_dir(path):
    p = os.path.join(path, "eeg.csv") if os.path.isdir(path) else path
    with open(p) as f:
        r = csv.reader(f)
        header = next(r)
        rows = [row for row in r if row]
    a = np.asarray(rows, dtype=np.float64)
    chans = [i for i, h in enumerate(header) if h.startswith("ch")]
    t = a[:, header.index("t")] if "t" in header else np.arange(len(a)) / FS
    x = a[:, chans].astype(np.float32)
    dur = t[-1] - t[0]
    fs = (len(t) - 1) / dur if dur > 0 else FS
    base = path if os.path.isdir(path) else os.path.dirname(path)
    labels = []
    lj = os.path.join(base, "labels.json")
    ej = os.path.join(base, "events.jsonl")
    if os.path.exists(lj):
        with open(lj) as f:
            labels = [(t[0] + float(a0), t[0] + float(a1), str(lab)) for a0, a1, lab in json.load(f)]
    elif os.path.exists(ej):
        labels = _labels_from_events(ej, t[0], t[-1])
    return Recording(os.path.basename(base.rstrip("/")), t, x, fs, labels)


def load_xlsx(path):
    """PiEEG xlsx export -> (t_rel, x µV).

    Two layouts exist: the Octopus recorder ("Raw data" sheet, `Time (s),
    Counter, CH1..CH16`, raw 24-bit COUNTS, trailing MIN/MAX/MEAN/STD rows) and
    the PiEEG-16 one (`data_1ch_test..data_16ch_test`, already µV). The Counter
    column is what tells them apart -- not magnitude, since both sit on large
    DC offsets."""
    import re
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb["Raw data"] if "Raw data" in wb.sheetnames else wb.worksheets[0]
    rows = list(ws.iter_rows(values_only=True))
    is_ch = lambda h: bool(re.match(r"(?i)^(ch\d+|data_\d+ch)", str(h or "")))  # noqa: E731
    hi = next(i for i, r in enumerate(rows) if r and sum(is_ch(c) for c in r) >= 4)
    header = [str(c or "") for c in rows[hi]]
    ch = [i for i, h in enumerate(header) if is_ch(h)][:16]
    counts = any(h.strip().lower() == "counter" for h in header)
    data = []
    for r in rows[hi + 1:]:
        if r is None or not isinstance(r[ch[0]], (int, float)):
            continue                      # MIN/MAX/MEAN/STD summary rows
        data.append([float(r[i]) for i in ch])
    x = np.asarray(data, np.float64)
    if counts:
        x = x * UV_PER_COUNT
    t = np.arange(len(x)) / FS           # the Time column is BLE arrival, bursty
    return t, x.astype(np.float32)


def write_dir(out, t, x, labels_rel):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "eeg.csv"), "w") as f:
        f.write("t," + ",".join(f"ch{i + 1}" for i in range(x.shape[1])) + "\n")
        for i in range(len(t)):
            f.write(f"{t[i]:.4f}," + ",".join(f"{v:.2f}" for v in x[i]) + "\n")
    with open(os.path.join(out, "labels.json"), "w") as f:
        json.dump(labels_rel, f)


def load(path):
    if path.lower().endswith(".xlsx"):
        t, x = load_xlsx(path)
        return Recording(os.path.basename(path), t, x, FS, [])
    return load_dir(path)


# ---- fetch PiEEG's public data --------------------------------------------------

OCTO = "https://raw.githubusercontent.com/pieeg-club/Octopus_16/main/Dataset/"
P16 = "https://raw.githubusercontent.com/pieeg-club/PiEEG-16/main/Dataset/"


def _get(url, dst):
    if not os.path.exists(dst):
        urllib.request.urlretrieve(url, dst)
    return dst


def fetch(root="eval_data"):
    raw = os.path.join(root, "_raw")
    os.makedirs(raw, exist_ok=True)
    made = []
    # Octopus 16 at Pz: open then closed, concatenated
    to, xo = load_xlsx(_get(OCTO + "open_yeys_Pz.xlsx", os.path.join(raw, "octo_open.xlsx")))
    tc, xc = load_xlsx(_get(OCTO + "closed_eys_Pz.xlsx", os.path.join(raw, "octo_closed.xlsx")))
    # remove each file's own DC so the join isn't a step
    x = np.concatenate([xo - np.median(xo, 0), xc - np.median(xc, 0)])
    t = np.arange(len(x)) / FS
    d = os.path.join(root, "octopus16_pz_open_closed")
    write_dir(d, t, x, [[0, len(xo) / FS, "eyes open"], [len(xo) / FS, len(x) / FS, "eyes closed"]])
    made.append(d)
    tb, xb = load_xlsx(_get(OCTO + "Blinking_Fp.xlsx", os.path.join(raw, "octo_blink.xlsx")))
    d = os.path.join(root, "octopus16_fp_blink")
    write_dir(d, tb, xb, [[0, len(xb) / FS, "blinking"]])
    made.append(d)
    # PiEEG-16 (different board): alpha test, alternating 5 s blocks
    for name, fn, lab in [("pieeg16_alpha", "3.Alpha_test.xlsx", "alpha"),
                          ("pieeg16_chew_blink_1", "1.Chewing_Blinking.xlsx", None),
                          ("pieeg16_chew_blink_2", "2.Chewing_Blinking.xlsx", None)]:
        try:
            t, x = load_xlsx(_get(P16 + urllib.request.quote(fn), os.path.join(raw, fn)))
        except Exception as e:  # noqa: BLE001
            print(f"[fetch] skip {fn}: {e}")
            continue
        labels = []
        if lab == "alpha":
            n = int(len(t) / FS // 5)
            labels = [[5 * k, 5 * (k + 1), "block_a" if k % 2 == 0 else "block_b"] for k in range(n)]
        d = os.path.join(root, name)
        write_dir(d, t, x, labels)
        made.append(d)
    for d in made:
        r = load_dir(d)
        print(f"[fetch] {d}: {r.duration:.0f}s, {r.x.shape[1]} ch, labels "
              f"{sorted(set(l for *_, l in r.labels))}")
    return made


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "fetch":
        fetch(sys.argv[2] if len(sys.argv) > 2 else "eval_data")
    else:
        print("usage: python -m octomusic.eval.recordings fetch [dir]")
