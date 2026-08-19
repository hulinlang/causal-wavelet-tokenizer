"""Experiment 0.6: video temporal axis — causal vs Haar frontend on DAVIS 2017.

Question: do the audio findings transfer to the temporal axis of natural
video? (F10: causal saturates structurally at L=1; F13: deeper levels hurt
both frontends, causal more; F11: converged gap is structural.)

Signal model: per-pixel temporal trajectories. One clip = the gray value of
one pixel over n=64 consecutive frames, normalized to [-1, 1] via
v/127.5 - 1 (DATA_RANGE = 2.0 — identical PSNR convention to the audio
experiments, so dB numbers are directly comparable across modalities).

Why per-pixel trajectories: the project's causal innovation lives on the
time axis only; spatial handling (patches / spatial wavelets) is orthogonal
and intentionally factored out. Each trajectory is a natural 1D video-time
signal containing static background, motion edges, and texture.

Token layout / rate axis: identical to exp05 — block layout, N = n/2^L
tokens, C = 2^L values per token (N*C = n exactly), rate = d / 2^L latent
values per frame-sample.

Protocol mirrors exp05: TokenTrunkAE(width=32, hidden=64, dec_ln=False for
all runs, F9), AdamW 1e-3, 3000 steps, eval at 750/1500/3000, 64 fixed
validation trajectories (val-seed 10000), 3 seeds. Batch = 32
trajectories/step (a trajectory is 256x shorter than an audio clip; batch
raised to keep per-step diversity comparable).

Data: DAVIS 2017 trainval 480p (90 sequences; 59 with T >= 64 frames).
Frames are converted to grayscale once and cached as .npy (uint8) under
<data2>/davis_gray_cache/, then preloaded into RAM (~1.9 GB) per process.

Batched frontends: vectorized re-implementations of HaarFrontend /
CausalLapPyramidFrontend forward over the batch axis (lfilter operates on
the last axis; decimate/interp replicated exactly). A startup self-check
validates the batched path against the per-sample ground-truth frontend
(max abs diff must be < 1e-10) before any training run.

Run (GPU, resumable; --max-runs bounds each invocation's wall time):
  python experiments/exp06_video_temporal.py --max-runs 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stage1_shared_trunk import DATA_RANGE, make_frontend, psnr  # noqa: E402
from exp05_multilevel_decoder import inverse_torch_ml  # noqa: E402
from models.token_trunk import TokenTrunkAE  # noqa: E402

_S = float(np.sqrt(0.5))


# ------------------------------------------------------------- data (DAVIS)

def build_gray_cache(root: Path, cache_dir: Path, n: int) -> list[tuple[str, int, Path]]:
    """Convert usable sequences (T >= n) to grayscale uint8 .npy caches.

    Returns [(name, T, npy_path)] sorted by name.
    """
    seq_root = root / "JPEGImages" / "480p"
    cache_dir.mkdir(parents=True, exist_ok=True)
    seqs = []
    for p in sorted(seq_root.iterdir()):
        if not p.is_dir():
            continue
        frames = sorted(p.glob("*.jpg"))
        if len(frames) < n:
            continue
        out = cache_dir / f"{p.name}.npy"
        if not out.exists():
            arr = np.stack(
                [np.asarray(Image.open(f).convert("L")) for f in frames]
            )  # (T, 480, 854) uint8, ITU-R 601-2 luma
            np.save(out, arr)
            print(f"  cached {p.name}: {arr.shape}", flush=True)
        seqs.append((p.name, len(frames), out))
    return seqs


class DavisSampler:
    """Random per-pixel temporal trajectories; all sequences preloaded to RAM.

    59 usable sequences = ~1.9 GB uint8 — loaded once per process so the
    training loop never touches disk (mmap was I/O-bound per step).
    """

    def __init__(self, seqs: list[tuple[str, int, Path]], n: int):
        self.seqs = seqs
        self.n = n
        t0 = time.time()
        self._data = [np.load(path) for _, _, path in seqs]
        gb = sum(a.nbytes for a in self._data) / 2**30
        print(f"[setup] preloaded {len(self._data)} sequences to RAM: {gb:.2f} GB ({time.time()-t0:.0f}s)", flush=True)

    def trajectory(self, rng: np.random.Generator) -> np.ndarray:
        i = int(rng.integers(len(self.seqs)))
        arr = self._data[i]
        T = self.seqs[i][1]
        t0 = int(rng.integers(0, T - self.n + 1))
        y = int(rng.integers(arr.shape[1]))
        x = int(rng.integers(arr.shape[2]))
        return np.asarray(arr[t0 : t0 + self.n, y, x], dtype=np.float64) / 127.5 - 1.0

    def batch(self, batch: int, rng: np.random.Generator) -> np.ndarray:
        return np.stack([self.trajectory(rng) for _ in range(batch)])


# --------------------------------------------- batched frontends (validated)

def _decimate_batch(ch: np.ndarray, target_len: int) -> np.ndarray:
    idx = np.linspace(0, ch.shape[-1] - 1, target_len).round().astype(int)
    return ch[..., idx]


def _interp_batch(ch: np.ndarray, target_len: int) -> np.ndarray:
    """Batched copy of frontends' np.interp linear upsample (last axis)."""
    m = ch.shape[-1]
    pos = np.linspace(0.0, m - 1, target_len)
    i0 = np.floor(pos).astype(int).clip(max=m - 2)
    w = pos - i0
    return ch[..., i0] * (1 - w) + ch[..., i0 + 1] * w


def haar_forward_batch(X: np.ndarray, levels: int) -> list[np.ndarray]:
    """Batched copy of HaarFrontend.forward. X: (B, n) -> [a_L, d_1, ..., d_L]."""
    details: list[np.ndarray] = []
    a = X.astype(np.float64)
    for _ in range(levels):
        m = a.shape[-1] - (a.shape[-1] % 2)
        ev, od = a[..., :m:2], a[..., 1:m:2]
        a, d = _S * (ev + od), _S * (ev - od)
        details.append(d)
    return [a] + details


def causal_lap_forward_batch(fe, X: np.ndarray) -> list[np.ndarray]:
    """Batched copy of CausalLapPyramidFrontend.forward. X: (B, n)."""
    from causal_wavelet.core import scale_space

    residuals: list[np.ndarray] = []
    cur = X.astype(np.float64)
    for k in range(fe.levels):
        nxt_len = max(1, cur.shape[-1] // 2)
        smooth = scale_space(
            cur, fe.tau0 * fe.c ** (2.0 * k), c=fe.c, K=fe.K,
            delay_compensate=fe.delay_compensate,
        )
        base = _decimate_batch(smooth, nxt_len)
        resid = cur - _interp_batch(base, cur.shape[-1])
        residuals.append(resid[..., ::2])
        cur = base
    return [cur] + residuals[::-1]


def forward_batch(fe_name: str, fe, X: np.ndarray) -> list[np.ndarray]:
    if fe_name == "haar":
        return haar_forward_batch(X, fe.levels)
    return causal_lap_forward_batch(fe, X)


def channels_coarse_to_fine(fe_name: str, channels: list[np.ndarray]) -> list[np.ndarray]:
    if fe_name == "haar":
        return [channels[0]] + channels[1:][::-1]
    return channels


def tokens_batch(fe_name: str, fe, X: np.ndarray, levels: int) -> np.ndarray:
    """(B, n) -> tokens (B, N=n/2^L, C=2^L), exp05 block layout."""
    ch = channels_coarse_to_fine(fe_name, forward_batch(fe_name, fe, X))
    B, N = ch[0].shape[0], ch[0].shape[-1]
    tok = np.empty((B, N, 2**levels), dtype=np.float64)
    tok[..., 0] = ch[0]
    for i in range(1, levels + 1):
        w = 2 ** (i - 1)
        tok[..., w : 2 * w] = ch[i].reshape(B, N, w)
    return tok


def self_check(n: int) -> None:
    """Batched frontends + token layout must match the per-sample originals."""
    from exp05_multilevel_decoder import to_tokens_ml

    rng = np.random.default_rng(123)
    X = rng.normal(size=(3, n))
    for fe_name in ("haar", "causal_lap"):
        for levels in (1, 2, 4):
            fe = make_frontend(fe_name, levels=levels)
            ref_ch = [fe.forward(x) for x in X]
            bat_ch = forward_batch(fe_name, fe, X)
            d_ch = max(
                float(np.max(np.abs(np.stack([r[i] for r in ref_ch]) - bat_ch[i])))
                for i in range(len(bat_ch))
            )
            ref_tok = np.stack(
                [to_tokens_ml(fe_name, fe, x, levels) for x in X]
            )
            bat_tok = tokens_batch(fe_name, fe, X, levels)
            d_tok = float(np.max(np.abs(ref_tok - bat_tok)))
            assert max(d_ch, d_tok) < 1e-10, (fe_name, levels, d_ch, d_tok)
            print(f"  self-check {fe_name:>10} L={levels}: max diff {max(d_ch, d_tok):.2e}", flush=True)


# ------------------------------------------------------------- training

def train_one(frontend: str, levels: int, latent: int, seed: int, args,
              sampler: DavisSampler, val_tokens, val_x, device):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    fe = make_frontend(frontend, levels=levels)
    C = 2**levels
    model = TokenTrunkAE(C, args.width, latent, args.hidden, dec_ln=False).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    curve = {}
    t0 = time.time()
    for step in range(1, args.steps + 1):
        xb_np = sampler.batch(args.batch, rng)
        tok = torch.from_numpy(tokens_batch(frontend, fe, xb_np, levels)).float().to(device)
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
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-clips", type=int, default=64)
    ap.add_argument("--val-seed", type=int, default=10_000)
    ap.add_argument("--data", default=r"F:\小波变换\data2\DAVIS")
    ap.add_argument("--cache", default=r"F:\小波变换\data2\davis_gray_cache")
    ap.add_argument("--max-runs", type=int, default=0, help="0 = no limit")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "reports" / "exp06_video_temporal.json"))
    args = ap.parse_args()

    out = Path(args.out)
    results = {"runs": [], "config": {}}
    if out.exists():
        results = json.loads(out.read_text(encoding="utf-8"))
    done = {(r["frontend"], r["levels"], r["latent"], r["seed"]) for r in results["runs"]}

    print("[setup] building/loading DAVIS grayscale cache ...", flush=True)
    t0 = time.time()
    seqs = build_gray_cache(Path(args.data), Path(args.cache), args.n)
    print(f"[setup] {len(seqs)} usable sequences (T>={args.n}), {sum(t for _, t, _ in seqs)} frames "
          f"({time.time()-t0:.0f}s)", flush=True)

    print("[setup] batched-frontend self-check ...", flush=True)
    self_check(args.n)

    sampler = DavisSampler(seqs, args.n)
    val_rng = np.random.default_rng(args.val_seed)
    val_x = sampler.batch(args.val_clips, val_rng)
    print(f"[setup] {args.val_clips} fixed validation trajectories; device {args.device}", flush=True)

    new_runs = 0
    for frontend in args.frontends:
        for levels in args.levels:
            fe = make_frontend(frontend, levels=levels)
            val_tokens = torch.from_numpy(
                tokens_batch(frontend, fe, val_x, levels)
            ).float().to(args.device)
            for latent in args.latents:
                for seed in args.seeds:
                    if (frontend, levels, latent, seed) in done:
                        continue
                    if args.max_runs and new_runs >= args.max_runs:
                        print(f"[pause] max-runs {args.max_runs} reached; resume with same command", flush=True)
                        return
                    t0 = time.time()
                    r = train_one(frontend, levels, latent, seed, args, sampler,
                                  val_tokens, val_x, args.device)
                    results["runs"].append(r)
                    new_runs += 1
                    print(
                        f"{frontend:>10} L={levels} d={latent:>2} s={seed}  "
                        f"val PSNR {r['val_psnr_mean']:6.2f} ± {r['val_psnr_std']:.2f} dB  "
                        f"({time.time()-t0:.0f}s)  [{len(results['runs'])}/90]",
                        flush=True,
                    )
                    results["config"] = {
                        "modality": "video_temporal",
                        "dataset": "DAVIS-2017-trainval-480p",
                        "signal": "per-pixel gray trajectory, n frames, v/127.5-1",
                        "usable_sequences": len(seqs),
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
