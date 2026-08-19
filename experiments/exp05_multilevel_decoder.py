"""Experiment 0.5: multi-level decomposition + learned decoder.

Question (F11 follow-up): at L=1 the converged learned-decoder gap is
structural (~13-17 dB, causal_lap saturates ~50.6 dB because odd samples
only enter tokens through the causal smoothing tail). Does a multi-level
causal Laplacian pyramid — whose encoder-side residual refinement folds
each level's interpolation error into the next residual — close the
converged gap?

Token layout (identical for both frontends, generalizes stage 1):
block of 2^L samples -> one token; channels reordered coarse->fine,
token j carries ch_0[j] and ch_i[2^(i-1) j : 2^(i-1)(j+1)], so
  N = n/2^L tokens, C = 2^L values per token (N*C = n exactly).
L=1 reduces to the stage-1 layout (N=n/2, C=2).

Rate axis: latent d per token => d / 2^L latent values per sample.

Trunk: TokenTrunkAE with dec_ln=False for ALL runs (F9: decoder-input
LayerNorm(d) destroys magnitude information at small d; the L=1 baselines
at d in {8,16} without LN are completed here for a clean grid).

Protocol: 3000 steps, batch 2, AdamW 1e-3, signal MSE, 64 fixed val clips
(val-seed 10000), eval at 750/1500/3000. Resumable via the output JSON.

Run (GPU):
  python experiments/exp05_multilevel_decoder.py --frontends haar --levels 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stage1_shared_trunk import (  # noqa: E402
    DATA_RANGE,
    list_speech_commands,
    make_frontend,
    psnr,
    real_batch,
)
from models.token_trunk import TokenTrunkAE  # noqa: E402

_S = float(np.sqrt(0.5))


# ------------------------------------------------------- token layout (numpy)

def channels_coarse_to_fine(fe_name: str, channels: list[np.ndarray]) -> list[np.ndarray]:
    """Unify channel order to coarse->fine: [base_L, ch_L, ..., ch_1]."""
    if fe_name == "haar":
        # HaarFrontend.forward returns [a_L, d_1(finest), ..., d_L(coarsest)]
        return [channels[0]] + channels[1:][::-1]
    return channels  # causal_lap already returns [base, r_{L-1}, ..., r_0]


def to_tokens_ml(fe_name: str, fe, x: np.ndarray, levels: int) -> np.ndarray:
    """(n,) -> tokens (N=n/2^L, C=2^L), block layout (see module docstring)."""
    ch = channels_coarse_to_fine(fe_name, fe.forward(x))
    N = ch[0].shape[-1]
    tok = np.empty((N, 2**levels), dtype=np.float64)
    tok[:, 0] = ch[0]
    for i in range(1, levels + 1):
        w = 2 ** (i - 1)
        tok[:, w : 2 * w] = ch[i].reshape(N, w)
    return tok


# ------------------------------------------------------- torch inverses

def split_tokens_ml(tok: torch.Tensor, levels: int) -> list[torch.Tensor]:
    """(B, N, 2^L) -> channel list coarse->fine: [ch_0 (B,N), ch_i (B, 2^(i-1)N)]."""
    B, N, _ = tok.shape
    chs = [tok[..., 0]]
    for i in range(1, levels + 1):
        w = 2 ** (i - 1)
        chs.append(tok[..., w : 2 * w].reshape(B, w * N))
    return chs


def _interp_torch(y: torch.Tensor, n: int) -> torch.Tensor:
    """Linear upsample (B, m) -> (B, n); matches frontends' np.interp grids."""
    idx = torch.arange(n, device=y.device, dtype=y.dtype) * (y.shape[-1] - 1) / (n - 1)
    i0 = idx.long().clamp(max=y.shape[-1] - 2)
    w = idx - i0.to(y.dtype)
    return y[:, i0] * (1 - w) + y[:, i0 + 1] * w


def _idwt_step_torch(a: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    out = torch.stack([_S * (a + d), _S * (a - d)], dim=-1)
    return out.reshape(a.shape[0], -1)


def inverse_torch_ml(frontend: str, tok: torch.Tensor, levels: int, n: int) -> torch.Tensor:
    """Differentiable multi-level inverse. tok: (B, N, 2^L) -> signal (B, ~n)."""
    chs = split_tokens_ml(tok, levels)
    cur = chs[0]
    if frontend == "haar":
        for i in range(1, levels + 1):
            cur = _idwt_step_torch(cur, chs[i])
    else:  # causal_lap: x_k = interp(x_{k+1}) + interp(r_k)
        for i in range(1, levels + 1):
            m = 2 * cur.shape[-1]
            cur = _interp_torch(cur, m) + _interp_torch(chs[i], m)
    return cur[..., :n]


# ------------------------------------------------------- training

def train_one(frontend: str, levels: int, latent: int, seed: int, args, files,
              val_tokens, val_x, device):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    fe = make_frontend(frontend, levels=levels)
    C = 2**levels
    model = TokenTrunkAE(C, args.width, latent, args.hidden, dec_ln=False).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    curve = {}
    t0 = time.time()
    for step in range(1, args.steps + 1):
        xb_np = real_batch(files, args.batch, args.n, rng)
        tok = torch.from_numpy(
            np.stack([to_tokens_ml(frontend, fe, x, levels) for x in xb_np.astype(np.float64)])
        ).float().to(device)
        xb = torch.from_numpy(xb_np).float().to(device)
        x_hat = inverse_torch_ml(frontend, model(tok), levels, args.n)
        loss = torch.nn.functional.mse_loss(xb, x_hat)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step in args.eval_steps:
            model.eval()
            with torch.no_grad():
                xhv = inverse_torch_ml(frontend, model(val_tokens), levels, args.n).cpu().numpy()
                psnrs = [psnr(x, xh) for x, xh in zip(val_x, xhv)]
            model.train()
            curve[str(step)] = {"mean": float(np.mean(psnrs)), "std": float(np.std(psnrs))}

    final = curve[str(args.steps)]
    return {
        "frontend": frontend,
        "levels": levels,
        "latent": latent,
        "rate_vals_per_sample": round(latent / 2**levels, 4),
        "seed": seed,
        "val_psnr_mean": round(final["mean"], 3),
        "val_psnr_std": round(final["std"], 3),
        "curve": {k: {"mean": round(v["mean"], 3), "std": round(v["std"], 3)} for k, v in curve.items()},
        "train_seconds": round(time.time() - t0, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontends", nargs="+", default=["haar", "causal_lap"])
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--latents", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--eval-steps", type=int, nargs="+", default=[750, 1500, 3000])
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--n", type=int, default=16384)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-clips", type=int, default=64)
    ap.add_argument("--val-seed", type=int, default=10_000)
    ap.add_argument("--data", default=r"F:\小波变换\data")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "reports" / "exp05_multilevel.json"))
    args = ap.parse_args()

    out = Path(args.out)
    results = {"runs": [], "config": {}}
    if out.exists():
        results = json.loads(out.read_text(encoding="utf-8"))
    done = {(r["frontend"], r["levels"], r["latent"], r["seed"]) for r in results["runs"]}

    files = list_speech_commands(args.data)
    print(f"real data: {len(files)} wav files; device {args.device}", flush=True)

    val_x = real_batch(files, args.val_clips, args.n, np.random.default_rng(args.val_seed))
    for frontend in args.frontends:
        for levels in args.levels:
            fe = make_frontend(frontend, levels=levels)
            val_tokens = torch.from_numpy(
                np.stack([to_tokens_ml(frontend, fe, x, levels) for x in val_x])
            ).float().to(args.device)
            for latent in args.latents:
                for seed in args.seeds:
                    if (frontend, levels, latent, seed) in done:
                        print(f"skip {frontend} L={levels} d={latent} s={seed}", flush=True)
                        continue
                    t0 = time.time()
                    r = train_one(frontend, levels, latent, seed, args, files,
                                  val_tokens, val_x, args.device)
                    results["runs"].append(r)
                    print(
                        f"{frontend:>10} L={levels} d={latent:>2} s={seed}  "
                        f"val PSNR {r['val_psnr_mean']:6.2f} ± {r['val_psnr_std']:.2f} dB  "
                        f"({time.time()-t0:.0f}s)",
                        flush=True,
                    )
                    results["config"] = {
                        "levels": args.levels,
                        "latents": args.latents,
                        "seeds": args.seeds,
                        "steps": args.steps,
                        "eval_steps": args.eval_steps,
                        "batch": args.batch,
                        "n": args.n,
                        "width": args.width,
                        "hidden": args.hidden,
                        "lr": args.lr,
                        "dec_ln": False,
                        "val_clips": args.val_clips,
                        "val_seed": args.val_seed,
                        "data_range": DATA_RANGE,
                        "device": args.device,
                    }
                    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== summary (val PSNR dB @ final step, mean ± seed-std) ===")
    for frontend in args.frontends:
        for levels in args.levels:
            row = []
            for latent in args.latents:
                vals = [r["val_psnr_mean"] for r in results["runs"]
                        if r["frontend"] == frontend and r["levels"] == levels
                        and r["latent"] == latent]
                if vals:
                    row.append(f"d={latent:>2}: {np.mean(vals):6.2f} ± {np.std(vals):.2f}")
            if row:
                print(f"{frontend:>10} L={levels}  " + "  ".join(row), flush=True)
    print(f"[done] results -> {out}", flush=True)


if __name__ == "__main__":
    main()
