"""Stage 1: shared token trunk reconstruction, Haar vs causal frontends.

Research question (positioning A): can a LEARNED token-wise decoder close
the structural reconstruction gap that the causal frontends showed in the
non-parametric stage-0 protocol — while buying streaming capability?

Protocol mirrors WAT (arXiv:2606.02631) Appendix A.2:
token width 32, latent 16, hidden 64, AdamW lr 1e-3, batch 2, 300 steps,
signal-space MSE loss, validation PSNR (data range 2.0 for [-1,1] audio)
every 50 steps.

Token layout (one-level, identical across frontends):
  Haar L=1       -> channels [cA (n/2), cD (n/2)]  -> tokens (N=n/2, C=2)
  causal_lap L=1 -> channels [base (n/2), r0 (n/2)] -> tokens (N=n/2, C=2)

Data: synthetic audio-like probes until Speech Commands is downloaded
(--data points at the dataset root later).

Run (GPU):
  python experiments/stage1_shared_trunk.py --frontend haar
  python experiments/stage1_shared_trunk.py --frontend causal_lap
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from frontends.causal_lap import CausalLapPyramidFrontend
from frontends.haar import HaarFrontend
from models.token_trunk import TokenTrunkAE

DATA_RANGE = 2.0  # audio normalized to [-1, 1]


# ------------------------------------------------------------------- data

def synth_batch(batch: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Random audio-like probes: mixture of chirps, bursts, smooth segments."""
    t = np.arange(n) / n
    out = []
    for _ in range(batch):
        kind = rng.integers(3)
        if kind == 0:
            f0, f1 = rng.uniform(1, 5), rng.uniform(20, 80)
            x = np.sin(2 * np.pi * (f0 * t + (f1 - f0) * t**2))
        elif kind == 1:
            x = np.zeros(n)
            for _ in range(rng.integers(1, 4)):
                c0, w, f = rng.uniform(0.1, 0.9), 10 ** rng.uniform(-3, -1.3), rng.uniform(10, 200)
                x = x + np.exp(-0.5 * ((t - c0) / w) ** 2) * np.sin(2 * np.pi * f * (t - c0))
        else:
            bp = np.sort(rng.uniform(0, 1, rng.integers(2, 5)))
            x = np.interp(t, np.concatenate([[0], bp, [1]]),
                          rng.uniform(-1, 1, len(bp) + 2))
        out.append(x / (np.abs(x).max() + 1e-12))
    return np.stack(out)


# -------------------------------------------------------------- frontends

def make_frontend(name: str, levels: int = 1):
    if name == "haar":
        return HaarFrontend(levels=levels)
    if name == "causal_lap":
        return CausalLapPyramidFrontend(levels=levels)
    raise ValueError(name)


def to_tokens(fe, x: np.ndarray) -> np.ndarray:
    """(n,) -> tokens (N=n/2, C=2) for one-level frontends."""
    ch = fe.forward(x)
    assert len(ch) == 2 and ch[0].shape == ch[1].shape
    return np.stack([ch[0], ch[1]], axis=-1)


_S = float(np.sqrt(0.5))


def _interp_torch(y: torch.Tensor, n: int) -> torch.Tensor:
    """Linear upsample (B, m) -> (B, n≈2m); matches frontends' np.interp."""
    idx = torch.arange(n, device=y.device, dtype=y.dtype) * (y.shape[-1] - 1) / (n - 1)
    i0 = idx.long().clamp(max=y.shape[-1] - 2)
    w = idx - i0.to(y.dtype)  # (n,)
    return y[:, i0] * (1 - w) + y[:, i0 + 1] * w


def inverse_torch(frontend: str, tok: torch.Tensor, n: int) -> torch.Tensor:
    """Differentiable 1-level inverse. tok: (B, N, 2) -> signal (B, n)."""
    a, d = tok[..., 0], tok[..., 1]
    if frontend == "haar":
        out = torch.stack([_S * (a + d), _S * (a - d)], dim=-1)  # (B, N, 2)
        return out.reshape(a.shape[0], -1)[..., :n]
    # causal_lap: x = interp(base) + interp(residual)
    return _interp_torch(a, n) + _interp_torch(d, n)


def psnr(x: np.ndarray, y: np.ndarray) -> float:
    mse = float(np.mean((x - y) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(DATA_RANGE**2 / mse)


# ------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontend", default="haar", choices=["haar", "causal_lap"])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--n", type=int, default=16384, help="samples per clip (WAT audio protocol)")
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--latent", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--value-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    fe = make_frontend(args.frontend)
    model = TokenTrunkAE(2, args.width, args.latent, args.hidden, args.value_scale).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    t0 = time.time()
    for step in range(1, args.steps + 1):
        xb = torch.from_numpy(synth_batch(args.batch, args.n, rng)).float().to(args.device)
        tok = torch.from_numpy(
            np.stack([to_tokens(fe, x) for x in xb.cpu().numpy().astype(np.float64)])
        ).float().to(args.device)
        rec = model(tok)
        x_hat = inverse_torch(args.frontend, rec, args.n)          # differentiable
        loss = torch.nn.functional.mse_loss(xb, x_hat)             # signal-space MSE (WAT)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 50 == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                xv = synth_batch(4, args.n, np.random.default_rng(10_000))
                tokv = torch.from_numpy(np.stack([to_tokens(fe, x) for x in xv])).float().to(args.device)
                xhv = inverse_torch(args.frontend, model(tokv), args.n).cpu().numpy()
                psnrs = [psnr(x, xh) for x, xh in zip(xv, xhv)]
            model.train()
            mem = (torch.cuda.max_memory_allocated() / 2**30) if args.device == "cuda" else 0.0
            print(f"step {step:>4}  loss {loss.item():.3e}  val PSNR {np.mean(psnrs):6.2f} dB"
                  f"  ({time.time()-t0:5.1f}s, peak {mem:.2f} GB)")

    # final summary line for the report
    print(f"FINAL frontend={args.frontend} latent={args.latent} "
          f"val_PSNR={np.mean(psnrs):.2f} time={time.time()-t0:.1f}s peak_GB={mem:.2f}")


if __name__ == "__main__":
    main()
