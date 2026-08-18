"""One-dimensional multi-level Haar DWT frontend (baseline, non-causal).

Mirrors the WAT (arXiv:2606.02631) one-level Haar frontend, extended to
multiple levels for a fair comparison with the multi-scale causal wavelet.
Implemented manually to avoid the PyWavelets dependency at inference.
"""

from __future__ import annotations

import numpy as np

from .base import Frontend

_S = np.sqrt(0.5)


def _dwt_step(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One Haar level: returns (approximation, detail), each length n/2."""
    n = x.shape[-1] - (x.shape[-1] % 2)
    ev, od = x[:n:2], x[1:n:2]
    return _S * (ev + od), _S * (ev - od)


def _idwt_step(a: np.ndarray, d: np.ndarray) -> np.ndarray:
    n = a.shape[-1]
    out = np.empty(2 * n, dtype=np.float64)
    out[0::2] = _S * (a + d)
    out[1::2] = _S * (a - d)
    return out


class HaarFrontend:
    """Multi-level Haar DWT. Channels: [cA_L, cD_L, cD_{L-1}, ..., cD_1]."""

    name = "haar"

    def __init__(self, levels: int = 4):
        self.levels = levels

    def forward(self, x: np.ndarray) -> list[np.ndarray]:
        details: list[np.ndarray] = []
        a = x.astype(np.float64)
        for _ in range(self.levels):
            a, d = _dwt_step(a)
            details.append(d)
        return [a] + details  # coarse approximation first

    def inverse(self, channels: list[np.ndarray]) -> np.ndarray:
        a = channels[0]
        for d in reversed(channels[1:]):  # coarsest detail last in forward order
            a = _idwt_step(a, d)
        return a
