"""Time-causal wavelet frontend (Lindeberg), built on causal_wavelet.core.

Channels: [lowpass_residual, W_1, ..., W_L] where W_k are causal bandpass
channels (differences of adjacent scale-space levels). Unlike Haar this
frontend is fully causal and time-recursive (O(1) memory per channel).

Two rate modes:

- rate_matched=False: full-length channels, total (L+1)*n coefficients,
  exactly invertible (telescoping identity).
- rate_matched=True: each channel decimated to the Haar-matched rate
  (lowpass -> n/2^L, W_k -> n/2^(k+1)), total = n coefficients, matching
  the L-level Haar scalar budget exactly. Reconstruction interpolates the
  decimated channels back to full length (linear), so the round trip is
  lossy even at 100% keep — full-density PSNR is reported as the ceiling.

Decimation preserves causality (no future samples are read), so the
streaming claim survives rate matching.
"""

from __future__ import annotations

import numpy as np

from causal_wavelet.core import wavelet_decompose, wavelet_reconstruct

from .base import Frontend


def _decimate(ch: np.ndarray, target_len: int) -> np.ndarray:
    """Uniform decimation to target_len samples (keeps first sample: causal)."""
    idx = np.linspace(0, ch.shape[-1] - 1, target_len).round().astype(int)
    return ch[idx]


def _interpolate(ch: np.ndarray, target_len: int) -> np.ndarray:
    src = np.linspace(0.0, 1.0, ch.shape[-1])
    dst = np.linspace(0.0, 1.0, target_len)
    return np.interp(dst, src, ch)


class CausalWaveletFrontend:
    name = "causal_wavelet"

    def __init__(
        self,
        levels: int = 4,
        tau0: float = 1.0,
        c: float = np.sqrt(2.0),
        K: int = 8,
        rate_matched: bool = False,
    ):
        self.levels = levels
        self.tau0 = tau0
        self.c = c
        self.K = K
        self.rate_matched = rate_matched
        self._n: int | None = None
        if rate_matched:
            self.name = "causal_wavelet_rm"

    def _target_lens(self, n: int) -> list[int]:
        """Haar-matched channel lengths: lowpass n/2^L, W_k n/2^(k+1)."""
        lens = [max(1, n >> self.levels)]
        lens += [max(1, n >> (k + 1)) for k in range(self.levels)]
        return lens

    def forward(self, x: np.ndarray) -> list[np.ndarray]:
        n = x.shape[-1]
        self._n = n
        bandpass, lowpass = wavelet_decompose(x, self.levels, self.tau0, self.c, self.K)
        channels = [lowpass] + bandpass
        if self.rate_matched:
            channels = [_decimate(ch, m) for ch, m in zip(channels, self._target_lens(n))]
        return channels

    def inverse(self, channels: list[np.ndarray]) -> np.ndarray:
        assert self._n is not None, "forward must be called before inverse"
        if self.rate_matched:
            channels = [_interpolate(ch, self._n) for ch in channels]
        return wavelet_reconstruct(channels[1:], channels[0])
