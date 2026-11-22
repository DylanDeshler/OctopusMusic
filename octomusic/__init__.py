"""OctopusMusic: live EEG (PiEEG Octopus 16) -> per-measure conditioning ->
the Neural Music Generator DiT on a remote GPU -> audio back on the laptop.

    octomusic.model_space   what the model accepts (bands, tempo grid). Shared.
    octomusic.protocol      the laptop <-> GPU wire format. Shared.
    octomusic.server        runs on the GPU box, wraps RealBackend.
    octomusic.client        runs on the laptop: BLE, DSP, mapping, audio, UI.
"""

__version__ = "0.1.0"
