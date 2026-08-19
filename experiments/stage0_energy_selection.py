"""Stage 0, experiment 0.1/0.2: non-parametric fixed-rate energy selection.

Protocol follows WAT (arXiv:2606.02631) Table 4:
- decompose a signal with a frontend,
- keep a fixed ratio rho of coefficients by global energy top-k
  (vs uniform-stride and random baselines),
- zero the dropped coefficients, reconstruct, measure PSNR.

Frontends: Haar baseline, causal wavelet (full-length channels), and
rate-matched causal wavelet (channels decimated to the Haar scalar budget;
100%-keep PSNR reported as the frontend ceiling).

Signals: synthetic probes for the smoke test (chirp, transient bursts,
piecewise-smooth). Real datasets (Speech Commands / DAVIS audio track)
plug in via --data once downloaded.

No model training is involved; runs on CPU in seconds.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from frontends.causal import CausalWaveletFrontend
from frontends.haar import HaarFrontend

KEEP_RATIOS = [0.50, 0.25, 0.10, 0.05, 0.01]
DATA_RANGE = 1.0


# ---------------------------------------------------------------- signals

def probe_signals(n: int = 4096) -> dict[str, np.ndarray]:
    t = np.arange(n) / n
    sigs = {
        # smooth + varying frequency content
        "chirp": np.sin(2 * np.pi * (3 * t + 40 * t**2)),
        # localized transients at multiple durations
        "bursts": sum(
            np.exp(-0.5 * ((t - c0) / w) ** 2) * np.sin(2 * np.pi * f * (t - c0))
            for c0, w, f in [(0.2, 0.002, 200), (0.5, 0.01, 60), (0.8, 0.05, 15)]
        ),
        # piecewise smooth with edges
        "edges": np.where(t < 0.33, 0.2, np.where(t < 0.66, 0.9, 0.4))
        + 0.05 * np.sin(2 * np.pi * 8 * t),
    }
    return {k: v / (np.abs(v).max() + 1e-12) for k, v in sigs.items()}


# ------------------------------------------------------------- selection

def select(channels: list[np.ndarray], ratio: float, method: str, rng: np.random.Generator):
    """Return masked copies of channels under a fixed keep ratio.

    Energy of coefficient i in channel k is c^2; 'energy_global' pools all
    coefficients and keeps the global top-k, mirroring WAT's selector.
    """
    flat = np.concatenate([ch.ravel() for ch in channels])
    total = flat.size
    k = max(1, int(round(ratio * total)))

    if method == "energy_global":
        keep = np.zeros(total, bool)
        keep[np.argpartition(-flat**2, k - 1)[:k]] = True
    elif method == "uniform":
        idx = np.linspace(0, total - 1, k).round().astype(int)
        keep = np.zeros(total, bool)
        keep[idx] = True
    elif method == "random":
        keep = np.zeros(total, bool)
        keep[rng.choice(total, k, replace=False)] = True
    else:
        raise ValueError(method)

    masked, off = [], 0
    for ch in channels:
        m = keep[off : off + ch.size].reshape(ch.shape)
        masked.append(ch * m)
        off += ch.size
    return masked


# --------------------------------------------------------------- metrics

def psnr(x: np.ndarray, y: np.ndarray) -> float:
    mse = float(np.mean((x - y) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(DATA_RANGE**2 / mse)


# ------------------------------------------------------------------ main

def run(frontend, x: np.ndarray, rng: np.random.Generator) -> dict:
    channels = frontend.forward(x)
    ceiling = psnr(x, frontend.inverse(channels))
    if not getattr(frontend, "rate_matched", False):
        assert ceiling > 120, f"{frontend.name}: round-trip not exact ({ceiling:.1f} dB)"

    out: dict[str, dict[float, float]] = {}
    for method in ("energy_global", "uniform", "random"):
        out[method] = {
            r: psnr(x, frontend.inverse(select(channels, r, method, rng))) for r in KEEP_RATIOS
        }
    return {"ceiling": ceiling, "selectors": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, default=4)
    ap.add_argument("--c", type=float, default=np.sqrt(2.0), help="causal wavelet distribution parameter")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    frontends = [
        HaarFrontend(levels=args.levels),
        CausalWaveletFrontend(levels=args.levels, c=args.c),
        CausalWaveletFrontend(levels=args.levels, c=args.c, rate_matched=True),
    ]

    header = (
        f"{'signal':<10}{'frontend':<20}{'selector':<15}{'100%':>8}"
        + "".join(f"{r:>8.0%}" for r in KEEP_RATIOS)
    )
    print(header)
    print("-" * len(header))
    for sname, x in probe_signals().items():
        for fe in frontends:
            rows = [run(fe, x, np.random.default_rng(s)) for s in range(args.seeds)]
            ceiling = np.mean([row["ceiling"] for row in rows])
            for method in ("energy_global", "uniform", "random"):
                vals = [np.mean([row["selectors"][method][r] for row in rows]) for r in KEEP_RATIOS]
                print(
                    f"{sname:<10}{fe.name:<20}{method:<15}{ceiling:>8.2f}"
                    + "".join(f"{v:>8.2f}" for v in vals)
                )


if __name__ == "__main__":
    main()
