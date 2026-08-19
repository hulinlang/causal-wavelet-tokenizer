"""Experiment 0.6 analysis: exact-coefficient ceiling + error localization.

Separates "coefficients lack the information" from "trunk lacks capacity":
run each frontend's own numpy inverse on the EXACT coefficients of the 64
fixed validation trajectories (no learned trunk). Haar is exactly invertible
(ceiling = inf); causal_lap's ceiling is finite because decimated residuals
discard odd-frame deviations.

Also verifies the error-localization claim (exp03 structural observation):
for causal_lap, even time positions should reconstruct exactly
(x_hat[2j] = b_j + r_j = x[2j]), with all error on odd positions.

And reports the static-trajectory share of the validation set (flat
background pixels inflate PSNR for both frontends equally).

Run:  python experiments/exp06_frontend_ceiling.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stage1_shared_trunk import make_frontend, psnr  # noqa: E402
from exp06_video_temporal import build_gray_cache, DavisSampler  # noqa: E402

DATA = Path(r"F:\小波变换\data2\DAVIS")
CACHE = Path(r"F:\小波变换\data2\davis_gray_cache")
N = 64


def main():
    seqs = build_gray_cache(DATA, CACHE, N)
    sampler = DavisSampler(seqs, N)
    val_x = sampler.batch(64, np.random.default_rng(10_000))  # same val set as exp06

    # static share: near-flat trajectories (peak-to-peak < 0.01 of [-1,1] units)
    ptp = val_x.max(axis=1) - val_x.min(axis=1)
    print(f"val trajectories: 64; static (ptp<0.01): {(ptp < 0.01).sum()}; "
          f"ptp median {np.median(ptp):.3f}, mean {ptp.mean():.3f}")

    for fe_name in ("haar", "causal_lap"):
        for levels in (1, 2, 4):
            fe = make_frontend(fe_name, levels=levels)
            psnrs, even_err, odd_err = [], [], []
            for x in val_x:
                ch = fe.forward(x)
                xh = fe.inverse(ch)[:N]
                psnrs.append(psnr(x, xh))
                even_err.append(float(np.max(np.abs(xh[0::2] - x[0::2]))))
                odd_err.append(float(np.max(np.abs(xh[1::2] - x[1::2]))))
            finite = [p for p in psnrs if np.isfinite(p)]
            exact = len(psnrs) - len(finite)
            msg = f"{fe_name:>10} L={levels}: exact-coeff ceiling "
            if exact:
                msg += f"{exact}/64 clips exact (inf); "
            if finite:
                msg += f"finite mean {np.mean(finite):6.2f} dB, min {np.min(finite):6.2f}; "
            msg += f"max|err| even {max(even_err):.2e}, odd {max(odd_err):.3f}"
            print(msg, flush=True)


if __name__ == "__main__":
    main()
