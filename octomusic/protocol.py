"""Laptop <-> GPU wire format. One WebSocket, two kinds of frame.

TEXT frames are JSON objects with a "type":

  client -> server
    hello     {token?}                         first message
    dials     {t, values: {target: 0..1}}      ~10 Hz, the live EEG stream
    params    {params: {...}}                  any subset; applied next take
    start     {params: {...}}                  build a track and start the loop
    stop      {}
    playback  {played: int, t}                 stream sample the speaker is at
    ping      {t}

  server -> client
    server_info {backend, presets, defaults, sr}
    status      {state, msg, ...}
    take        {seq, start_sample, n_samples, measures: [...], compute_s, ...}
    measured    {seq, measures: [{m, rms, ...}]}  what the output actually did
    pong        {t, server_t}
    error       {msg}

BINARY frames are audio, always one per `take` with the same seq:

    "<4sIQII"  magic b"OMA1", seq, start_sample, n_samples, sr
    then n_samples little-endian int16 mono PCM

`start_sample` is the index into the session's single continuous stream, so
chunks are disjoint and gapless and the client can place them exactly. The
client reports back how far its speaker has got in the SAME index (`played`),
which is the only thing the server needs to keep a fixed amount of audio in
flight -- the knob that trades smoothness against brain-to-ear latency.

Why int16 and not float32: half the bytes over an SSH tunnel and the model's
decoder output is nowhere near 16-bit precision limits.
"""

from __future__ import annotations

import json
import struct

import numpy as np

MAGIC = b"OMA1"
HEADER = struct.Struct("<4sIQII")
DEFAULT_PORT = 8765


def encode_audio(seq: int, start_sample: int, audio: np.ndarray, sr: int) -> bytes:
    a = np.asarray(audio, dtype=np.float32).reshape(-1)
    pcm = (np.clip(a, -1.0, 1.0) * 32767.0).astype("<i2")
    return HEADER.pack(MAGIC, int(seq), int(start_sample), int(pcm.size), int(sr)) + pcm.tobytes()


def decode_audio(buf: bytes):
    magic, seq, start, n, sr = HEADER.unpack_from(buf, 0)
    if magic != MAGIC:
        raise ValueError("not an OctopusMusic audio frame")
    pcm = np.frombuffer(buf, dtype="<i2", count=n, offset=HEADER.size)
    return seq, start, sr, pcm.astype(np.float32) / 32767.0


def dumps(obj) -> str:
    return json.dumps(obj, default=_default, separators=(",", ":"))


def _default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    raise TypeError(f"not JSON serializable: {type(o).__name__}")
