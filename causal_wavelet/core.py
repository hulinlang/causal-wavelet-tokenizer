"""Time-causal scale-space via cascaded first-order recursive filters.

Implementation of the discrete approximation of Lindeberg's time-causal
limit kernel (arXiv:2510.05834): the limit kernel Psi(t; tau, c) is the
composition of infinitely many truncated exponential kernels in cascade,
which in discrete form becomes a cascade of first-order recursive (IIR)
filters.

Given a target temporal scale tau (variance of the limit kernel) and a
distribution parameter c > 1, the variance budget is distributed over K
cascade stages with time constants

    mu_k^2 = tau * (1 - c^-2) * c^(-2(k-1)),   k = 1..K

so that sum_k mu_k^2 -> tau as K -> infinity. Each stage is a first-order
recursive filter

    y[t] = (1 - a_k) * x[t] + a_k * y[t-1],   a_k = exp(-1 / mu_k)

which is fully time-causal and time-recursive: no future samples are used
and no temporal memory is needed beyond the filter states themselves.

NOTE (stage-0 validation item): the exact mapping from (tau, c) to the
per-stage time constants follows Lindeberg's canonical discretization;
experiment 0.2 verifies scaling properties numerically.
"""

from __future__ import annotations

import numpy as np


def first_order_recursive(x: np.ndarray, mu: float) -> np.ndarray:
    """Apply a single first-order recursive smoothing filter (causal).

    y[t] = (1 - a) * x[t] + a * y[t-1],  a = exp(-1/mu)
    """
    a = np.exp(-1.0 / mu)
    b = 1.0 - a
    y = np.empty_like(x, dtype=np.float64)
    prev = 0.0
    for t in range(x.shape[-1]):
        prev = b * x[t] + a * prev
        y[t] = prev
    return y


def time_constants(tau: float, c: float, K: int) -> np.ndarray:
    """Per-stage time constants mu_k for the cascade approximation."""
    ks = np.arange(K)
    var_k = tau * (1.0 - c**-2) * c ** (-2.0 * ks)
    return np.sqrt(var_k)


def scale_space(x: np.ndarray, tau: float, c: float = np.sqrt(2.0), K: int = 8) -> np.ndarray:
    """Causal temporal scale-space representation L(t; tau, c)."""
    y = x.astype(np.float64)
    for mu in time_constants(tau, c, K):
        y = first_order_recursive(y, mu)
    return y


def wavelet_decompose(
    x: np.ndarray, n_scales: int, tau0: float = 1.0, c: float = np.sqrt(2.0), K: int = 8
) -> tuple[list[np.ndarray], np.ndarray]:
    """Time-causal bandpass wavelet decomposition over discrete scale levels.

    Scale levels tau_k = tau0 * c^(2k). Bandpass channel k is the difference
    of adjacent scale-space representations:

        W_k(t) = L(t; tau_k) - L(t; tau_{k-1})

    which (Lindeberg, Sec. 2-3) corresponds up to a scaling factor to the
    first-order temporal derivative of the time-causal limit kernel.

    Returns (bandpass_channels, lowpass_residual). Reconstruction is by
    the telescoping identity:

        x_hat = lowpass_residual - sum_k W_k   (exact: x = L_0)
    """
    levels = [x.astype(np.float64)]
    for k in range(n_scales):
        levels.append(scale_space(x, tau0 * c ** (2.0 * (k + 1)), c=c, K=K))
    bandpass = [levels[k + 1] - levels[k] for k in range(n_scales)]
    lowpass = levels[-1]
    return bandpass, lowpass


def wavelet_reconstruct(bandpass: list[np.ndarray], lowpass: np.ndarray) -> np.ndarray:
    """Reconstruct from bandpass channels and lowpass residual.

    Since W_k = L_{k+1} - L_k with L_0 = x, the telescoping sum gives
    x = lowpass_residual - sum_k W_k (exact up to floating point).
    """
    out = lowpass.copy()
    for w in bandpass:
        out = out - w
    return out
