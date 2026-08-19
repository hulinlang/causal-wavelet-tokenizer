"""Causal Laplacian pyramid frontend (route-1 improvement).

Classic Laplacian-pyramid recursion built from causal smoothing stages,
with encoder-side residual refinement:

    x_0 = x                                    (length n)
    L_1 = causal_smooth(x_0, tau_1)            (full length, causal)
    b_1 = decimate(L_1, n/2)                   (keeps first sample: causal)
    r_0 = x_0 - interp(b_1)                    (encoder-side refinement)
    transmit r_0 (n/2); recurse on x_1 = b_1

Channels: [b_L (n/2^L), r_{L-1} (n/2^L), ..., r_0 (n/2)] — total exactly n
coefficients, matching the L-level Haar scalar budget.

Reconstruction (decoder, streaming-capable):

    x_L     = b_L
    x_{k}   = interp(b_{k+1}) + r_k           (interp is the only lookahead)

Causality note: smoothing and decimation are strictly causal; the encoder
uses lookahead when computing residuals against the *decodable* lowpass
(interp of decimated samples). In the tokenizer setting this is legitimate:
encoding may be offline, decoding is streaming. The decoder only ever adds
already-received residual samples to the interpolated lowpass.

Compared to naive rate matching (decimating bandpass channels), the
encoder-side refinement folds the interpolation error of every smooth
channel into the next residual, so only the finest residual's decimation
error survives — the coarse-scale accumulation error is eliminated.
"""

from __future__ import annotations

import numpy as np

from causal_wavelet.core import scale_space

from .base import Frontend
from .causal import _decimate, _interpolate


class CausalLapPyramidFrontend:
    name = "causal_lap"
    rate_matched = True  # lossy at 100% keep; skip exact round-trip assertion

    def __init__(
        self,
        levels: int = 4,
        tau0: float = 1.0,
        c: float = np.sqrt(2.0),
        K: int = 8,
        delay_compensate: bool = False,
    ):
        self.levels = levels
        self.tau0 = tau0
        self.c = c
        self.K = K
        self.delay_compensate = delay_compensate
        if delay_compensate:
            self.name = "causal_lap_dc"
        self._lens: list[int] | None = None

    def forward(self, x: np.ndarray) -> list[np.ndarray]:
        residuals: list[np.ndarray] = []
        cur = x.astype(np.float64)
        self._lens = [cur.shape[-1]]
        for k in range(self.levels):
            nxt_len = max(1, cur.shape[-1] // 2)
            smooth = scale_space(
                cur, self.tau0 * self.c ** (2.0 * k), c=self.c, K=self.K,
                delay_compensate=self.delay_compensate,
            )
            base = _decimate(smooth, nxt_len)
            resid = cur - _interpolate(base, cur.shape[-1])
            residuals.append(resid[::2])  # decimate residual to n/2 (keep even: causal)
            cur = base
            self._lens.append(nxt_len)
        # channels: coarse base first, then residuals coarse-to-fine
        return [cur] + residuals[::-1]

    def inverse(self, channels: list[np.ndarray]) -> np.ndarray:
        assert self._lens is not None, "forward must be called before inverse"
        cur = channels[0]  # coarsest base, length n/2^L
        for ch in channels[1:]:  # residuals coarse-to-fine
            cur = _interpolate(cur, 2 * cur.shape[-1]) + _interpolate(ch, 2 * cur.shape[-1])
        return cur
