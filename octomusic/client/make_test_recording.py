"""Write a synthetic recording in the recorder's format, with known structure,
for exercising offline mode before real sessions exist.

    python -m octomusic.client.make_test_recording --minutes 4

Structure (repeats): 30 s eyes open (low alpha), 30 s eyes closed (high
alpha), with a 3 s jaw clench (EMG burst) every 20 s and occasional blinks.
If a mapping works, brightness should follow the alpha blocks and density
should spike on the clenches -- visible in the offline analysis and audible
in the generated piece.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

FS = 250.0


def synth(minutes: float, seed=0):
    rng = np.random.default_rng(seed)
    n = int(minutes * 60 * FS)
    t = np.arange(n) / FS
    gains = 0.7 + 0.6 * rng.random(16)
    pink = np.zeros((n, 16))
    acc = np.zeros(16)
    white = rng.normal(size=(n, 16))
    for i in range(n):
        acc = 0.97 * acc + white[i]
        pink[i] = acc
    x = pink * 0.25 * 6.0
    eyes_closed = (t % 60) >= 30
    alpha_amp = np.where(eyes_closed, 30.0, 6.0)
    alpha_amp = np.convolve(alpha_amp, np.ones(250) / 250, mode="same")    # 1 s ramps
    ph = 2 * np.pi * np.cumsum(10.0 + 0.5 * np.sin(2 * np.pi * 0.05 * t)) / FS
    x += (alpha_amp * np.sin(ph))[:, None] * gains
    x += 3.0 * np.sin(2 * np.pi * 20.0 * t[:, None] + rng.uniform(0, 6, 16)[None, :])
    x += (4.0 * np.sin(2 * np.pi * 6.0 * t))[:, None] * gains
    clench = ((t % 20) >= 10) & ((t % 20) < 13)
    x += clench[:, None] * rng.normal(size=(n, 16)) * 25.0
    for tb in np.arange(5, t[-1], 7.3):                                    # blinks
        i = int(tb * FS)
        w = np.hanning(int(0.3 * FS))
        x[i:i + len(w)] += 80.0 * w[: max(0, min(len(w), n - i))][:, None]
    x += 2.0 * np.sin(2 * np.pi * 60.0 * t)[:, None]
    x += 2000.0
    return t, x.astype(np.float32), eyes_closed, clench


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=4.0)
    ap.add_argument("--out", default="recordings")
    args = ap.parse_args()
    t, x, closed, clench = synth(args.minutes)
    d = os.path.join(args.out, "synthetic-" + time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(d, exist_ok=True)
    t0 = time.time()
    with open(os.path.join(d, "eeg.csv"), "w") as f:
        f.write("t," + ",".join(f"ch{i + 1}" for i in range(16)) + "\n")
        for i in range(len(t)):
            f.write(f"{t0 + t[i]:.4f}," + ",".join(f"{v:.2f}" for v in x[i]) + "\n")
    with open(os.path.join(d, "events.jsonl"), "w") as f:
        for k in range(int(args.minutes * 2)):
            f.write(json.dumps({"t": t0 + 30 * k, "kind": "note",
                                "text": "eyes closed" if k % 2 else "eyes open"}) + "\n")
    print(d)


if __name__ == "__main__":
    main()
