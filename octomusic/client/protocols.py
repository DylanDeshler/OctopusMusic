"""Guided recording protocols: labelled blocks for baselining and the search.

Design choices that matter for the data:
  * blocks >= 10-20 s: the feature window (2 s) + smoothing put ~3 s of lag on
    everything, and PiEEG's 5 s alpha blocks were too short to separate;
  * several repetitions of each state: leave-one-segment-out evaluation needs
    >= 2 segments per label (3+ is better);
  * an unlabelled 3 s transition before each block (instruction shown, label
    off), so the moment of switching -- eyes opening, jaw relaxing -- doesn't
    smear into either state;
  * optional shuffling per repetition (never the same state twice in a row), so
    a slow drift (electrode settling, fatigue) can't masquerade as a state.
"""

from __future__ import annotations

import random

PROTOCOLS = {
    "alpha": {
        "title": "Alpha: eyes open / eyes closed",
        "note": "Board over Pz or the back of the head. The most reliable EEG effect.",
        "blocks": [("eyes open", "Eyes OPEN. Look at one point, relax, stay still."),
                   ("eyes closed", "Eyes CLOSED. Relax, stay still.")],
        "block_s": 20, "reps": 4,
    },
    "muscle": {
        "title": "Muscle: rest / jaw clench",
        "note": "A deliberate control signal. Clench firmly but comfortably.",
        "blocks": [("rest", "Relax your jaw and face."),
                   ("clench", "Clench your jaw and HOLD it.")],
        "block_s": 10, "reps": 5,
    },
    "blink": {
        "title": "Artifacts: rest / blinking",
        "note": "Teaches the artifact detector what blinks look like on your setup.",
        "blocks": [("rest", "Eyes open, relaxed, try not to blink."),
                   ("blinking", "Blink about once per second.")],
        "block_s": 10, "reps": 4,
    },
    "cognitive": {
        "title": "Cognitive: rest / mental arithmetic",
        "note": "Subtle on dry electrodes; needs longer blocks.",
        "blocks": [("rest", "Eyes open, relax, let your mind wander."),
                   ("arithmetic", "Count backwards from 1000 in steps of 7, in your head.")],
        "block_s": 30, "reps": 3,
    },
    "baseline": {
        "title": "Full baseline (all states)",
        "note": "One session covering everything. About 6 minutes at the defaults.",
        "blocks": [("eyes open", "Eyes OPEN. Look at one point, relax, stay still."),
                   ("eyes closed", "Eyes CLOSED. Relax, stay still."),
                   ("clench", "Clench your jaw and HOLD it."),
                   ("arithmetic", "Count backwards from 1000 in steps of 7, in your head."),
                   ("blinking", "Blink about once per second.")],
        "block_s": 20, "reps": 2,
    },
}

SETUP_S = 8.0
TRANSITION_S = 3.0


def schedule(name, block_s=None, reps=None, shuffle=True, seed=None):
    """-> list of steps {label|None, instruction, seconds, kind}."""
    p = PROTOCOLS[name]
    block_s = float(block_s or p["block_s"])
    reps = int(reps or p["reps"])
    rng = random.Random(seed)
    steps = [{"label": None, "kind": "setup", "seconds": SETUP_S,
              "instruction": "Get settled. Sit still, relax your face. Starting soon."}]
    prev = None
    for _ in range(reps):
        order = list(p["blocks"])
        if shuffle:
            for _ in range(20):
                rng.shuffle(order)
                if order[0][0] != prev:
                    break
        for lab, text in order:
            steps.append({"label": None, "kind": "transition", "seconds": TRANSITION_S,
                          "instruction": f"Next: {text}", "next": lab})
            steps.append({"label": lab, "kind": "block", "seconds": block_s, "instruction": text})
            prev = lab
    return steps


def listing():
    return {k: {"title": v["title"], "note": v["note"], "block_s": v["block_s"], "reps": v["reps"],
                "labels": [b[0] for b in v["blocks"]],
                "minutes": round((SETUP_S + v["reps"] * len(v["blocks"]) * (v["block_s"] + TRANSITION_S)) / 60, 1)}
            for k, v in PROTOCOLS.items()}
