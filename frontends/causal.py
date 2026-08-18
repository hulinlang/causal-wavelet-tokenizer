"""Time-causal wavelet frontend (Lindeberg), built on causal_wavelet.core.

Channels: [lowpass_residual, W_1, ..., W_L] where W_k are causal bandpass
channels (differences of adjacent scale-space levels). Unlike Haar this
frontend is fully causal and time-recursive (O(1) memory per channel),
at the cost of full-length channels (no dyadic downsampling).
"""

from __future__ import annotations

import numpy as np

from causal_wavelet.core import wavelet_decompose, wavelet_reconstruct

from .base import Frontend


class CausalWaveletFrontend:
    name = "causal_wavelet"

    def __init__(self, levels: int = 4, tau0: float = 1.0, c: float = np.sqrt(2.0), K: int = 8):
        self.levels = levels
        self.tau0 = tau0
        self.c = c
        self.K = K

    def forward(self, x: np.ndarray) -> list[np.ndarray]:
        bandpass, lowpass = wavelet_decompose(x, self.levels, self.tau0, self.c, self.K)
        return [lowpass] + bandpass

    def inverse(self, channels: list[np.ndarray]) -> np.ndarray:
        return wavelet_reconstruct(channels[1:], channels[0])
