"""Experiment 0.4 (stage-1 follow-up): latent rate-distortion sweep.

Question: does the learned-decoder gap between Haar and causal_lap (F5:
1.85 dB at latent 16) depend on the rate? Sweep latent d in {1,2,4,8,16}
for both frontends, 3 seeds, real Speech Commands data.

Protocol identical to stage 1 (WAT arXiv:2606.02631 Appendix A.2):
TokenTrunkAE(C=2, width 32, hidden 64), AdamW lr 1e-3, batch 2, 300 steps,
signal-space MSE. Two changes versus stage 1, both noise-reduction:
validation uses 64 fixed clips (val-seed 10000) instead of 4, and the
validation PSNR is reported at steps 150 and 300.

The latent dimension is the rate proxy: each of the N = n/2 tokens carries
d latent values, i.e. d/2 latent values per input sample (no quantization
yet — this is the continuous-bottleneck rate-distortion frontier).

Skips configs already present in the output JSON (resumable).

Run (GPU):
  python experiments/exp04_latent_sweep.py --frontends haar causal_lap
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
    inverse_torch,
    list_speech_commands,
    make_frontend,
    psnr,
    real_batch,
    to_tokens,
)
from models.token_trunk import TokenTrunkAE  # noqa: E402


def train_one(frontend: str, latent: int, seed: int, args, files, val_tokens, val_x, device):
    """One (frontend, latent, seed) run of the stage-1 protocol."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    fe = make_frontend(frontend)
    model = TokenTrunkAE(2, args.width, latent, args.hidden, dec_ln=not args.no_dec_ln).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    curve = {}
    t0 = time.time()
    for step in range(1, args.steps + 1):
        xb_np = real_batch(files, args.batch, args.n, rng)
        tok = torch.from_numpy(
            np.stack([to_tokens(fe, x) for x in xb_np.astype(np.float64)])
        ).float().to(device)
        xb = torch.from_numpy(xb_np).float().to(device)
        rec = model(tok)
        x_hat = inverse_torch(frontend, rec, args.n)
        loss = torch.nn.functional.mse_loss(xb, x_hat)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step in args.eval_steps:
            model.eval()
            with torch.no_grad():
                xhv = inverse_torch(frontend, model(val_tokens), args.n).cpu().numpy()
                psnrs = [psnr(x, xh) for x, xh in zip(val_x, xhv)]
            model.train()
            curve[str(step)] = {"mean": float(np.mean(psnrs)), "std": float(np.std(psnrs))}

    final = curve[str(args.steps)]
    return {
        "frontend": frontend,
        "latent": latent,
        "seed": seed,
        "val_psnr_mean": round(final["mean"], 3),
        "val_psnr_std": round(final["std"], 3),
        "curve": {k: {"mean": round(v["mean"], 3), "std": round(v["std"], 3)} for k, v in curve.items()},
        "train_seconds": round(time.time() - t0, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontends", nargs="+", default=["haar", "causal_lap"])
    ap.add_argument("--latents", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--eval-steps", type=int, nargs="+", default=[150, 300])
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--n", type=int, default=16384)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--no-dec-ln", action="store_true",
                    help="remove decoder-input LayerNorm(latent) (low-rate control)")
    ap.add_argument("--val-clips", type=int, default=64)
    ap.add_argument("--val-seed", type=int, default=10_000)
    ap.add_argument("--data", default=r"F:\小波变换\data")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "reports" / "exp04_latent_sweep.json"))
    args = ap.parse_args()

    out = Path(args.out)
    results = {"runs": [], "config": {}}
    if out.exists():
        results = json.loads(out.read_text(encoding="utf-8"))
    done = {(r["frontend"], r["latent"], r["seed"]) for r in results["runs"]}

    files = list_speech_commands(args.data)
    print(f"real data: {len(files)} wav files; device {args.device}", flush=True)

    # fixed validation set, tokenized once per frontend
    val_x = real_batch(files, args.val_clips, args.n, np.random.default_rng(args.val_seed))
    for frontend in args.frontends:
        fe = make_frontend(frontend)
        val_tokens = torch.from_numpy(
            np.stack([to_tokens(fe, x) for x in val_x])
        ).float().to(args.device)

        for latent in args.latents:
            for seed in args.seeds:
                if (frontend, latent, seed) in done:
                    print(f"skip {frontend} d={latent} s={seed} (done)", flush=True)
                    continue
                t0 = time.time()
                r = train_one(frontend, latent, seed, args, files, val_tokens, val_x, args.device)
                results["runs"].append(r)
                print(
                    f"{frontend:>10} d={latent:>2} s={seed}  "
                    f"val PSNR {r['val_psnr_mean']:6.2f} ± {r['val_psnr_std']:.2f} dB  "
                    f"({time.time()-t0:.0f}s)",
                    flush=True,
                )
                results["config"] = {
                    "latents": args.latents,
                    "seeds": args.seeds,
                    "steps": args.steps,
                    "eval_steps": args.eval_steps,
                    "batch": args.batch,
                    "n": args.n,
                    "width": args.width,
                    "hidden": args.hidden,
                    "lr": args.lr,
                    "val_clips": args.val_clips,
                    "val_seed": args.val_seed,
                    "data_range": DATA_RANGE,
                    "device": args.device,
                }
                out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    # summary matrix: mean ± std over seeds, per (frontend, latent)
    print("\n=== summary (val PSNR dB @ step 300, mean ± seed-std) ===")
    for frontend in args.frontends:
        row = []
        for latent in args.latents:
            vals = [r["val_psnr_mean"] for r in results["runs"]
                    if r["frontend"] == frontend and r["latent"] == latent]
            if vals:
                row.append(f"d={latent:>2}: {np.mean(vals):6.2f} ± {np.std(vals):.2f}")
        print(f"{frontend:>10}  " + "  ".join(row), flush=True)
    print(f"[done] results -> {out}", flush=True)


if __name__ == "__main__":
    main()
