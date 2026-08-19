"""Experiment 0.3: streaming benchmark — causal recursive frontend vs Haar frame buffering.

Research question: the "streaming" claim of the causal frontend has so far been
narrative-only. This experiment quantifies it with four head-to-head metrics:

  (a) first-token latency   — samples consumed before the first token exists
  (b) resident memory       — persistent state floats: O(K*L) vs O(F) frame buffer
  (c) long-stream throughput— ns/sample on a 2^20-sample stream (chunked callbacks)
  (d) chunk-boundary error  — state-reset transient vs persistent state vs Haar frames

Streaming causal encoder (StreamingCausalLap): a strictly causal Laplacian
pyramid. Per level k (scale tau0*c^(2k), K-stage first-order IIR cascade):
every input sample updates K filter states; every even-positioned input sample
emits a base sample (fed to the next level) and a residual token r = x - base.
Total coefficients = n, matching the L-level Haar scalar budget. Decoding needs
no future tokens at even positions; odd positions need one more token
(1-token lookahead per level; analytic worst case 2^L - 1 samples, see report).

CPU only. Run:
  python experiments/exp03_streaming_benchmark.py --data F:/小波变换/data
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from frontends.causal_lap import CausalLapPyramidFrontend
from frontends.haar import HaarFrontend

DATA_RANGE = 2.0  # audio normalized to [-1, 1]


# ----------------------------------------------------------------- data

def list_speech_commands(root: str) -> list[str]:
    files = []
    for p in sorted(Path(root).iterdir()):
        if p.is_dir() and not p.name.startswith("_"):
            files.extend(str(f) for f in sorted(p.glob("*.wav")))
    return files


def load_wav(path: str, n: int) -> np.ndarray:
    from scipy.io import wavfile

    sr, x = wavfile.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float64) / 32768.0
    if x.shape[0] < n:
        x = np.pad(x, (0, n - x.shape[0]))
    return x[:n]


def psnr(x: np.ndarray, y: np.ndarray) -> float:
    mse = float(np.mean((x - y) ** 2))
    if mse <= 0.0:
        return float("inf")
    return 10.0 * np.log10(DATA_RANGE**2 / mse)


# ------------------------------------------------- streaming causal frontend

class StreamingCausalLap:
    """Stateful, strictly causal Laplacian-pyramid streaming encoder.

    Equivalent rate budget to CausalLapPyramidFrontend (channels: final base
    n/2^L + L residuals n/2^{k+1}, total n), but decimation is anchored at
    even positions and interpolation at position-anchored midpoints, so that
    encoding needs ZERO lookahead (residuals are kept only at even positions,
    where the interpolated base equals the base sample itself).
    """

    def __init__(self, levels: int = 4, tau0: float = 1.0, c: float = np.sqrt(2.0), K: int = 8):
        self.levels, self.tau0, self.c, self.K = levels, tau0, c, K
        self.reset()

    def reset(self) -> None:
        self._st = []
        for k in range(self.levels):
            tau = self.tau0 * self.c ** (2.0 * k)
            ks = np.arange(self.K)
            mus = np.sqrt(tau * (1.0 - self.c**-2) * self.c ** (-2.0 * ks))
            a = np.exp(-1.0 / mus)
            self._st.append(
                {
                    "b": list(1.0 - a),
                    "a": list(a),
                    "zi": [np.zeros(1) for _ in range(self.K)],
                    "phase": 0,
                }
            )
        self.residuals: list[list[np.ndarray]] = [[] for _ in range(self.levels)]
        self.final_base: list[np.ndarray] = []

    @property
    def state_floats(self) -> int:
        """Persistent per-sample state: K IIR states + 1 phase flag per level."""
        return self.levels * self.K + self.levels

    def group_delays(self) -> list[float]:
        """Per-level IIR cascade group delay D_k = sum(mu), in input samples."""
        out = []
        for k in range(self.levels):
            tau = self.tau0 * self.c ** (2.0 * k)
            ks = np.arange(self.K)
            mus = np.sqrt(tau * (1.0 - self.c**-2) * self.c ** (-2.0 * ks))
            out.append(float(mus.sum() * (2**k)))  # convert level-k samples to input samples
        return out

    def process(self, chunk: np.ndarray) -> np.ndarray:
        """Consume a chunk; append emitted tokens to internal streams."""
        x = np.asarray(chunk, dtype=np.float64)
        for k in range(self.levels):
            st = self._st[k]
            y = x
            for j in range(self.K):
                y, st["zi"][j] = lfilter([st["b"][j]], [1.0, -st["a"][j]], y, zi=st["zi"][j])
            idx = np.arange(st["phase"], y.shape[0], 2)
            base = y[idx]
            resid = x[idx] - base
            self.residuals[k].append(resid)
            st["phase"] = (st["phase"] + y.shape[0]) % 2
            x = base
        self.final_base.append(x)
        return x

    def streams(self) -> tuple[list[np.ndarray], np.ndarray]:
        return [np.concatenate(r) for r in self.residuals], np.concatenate(self.final_base)


def lap_decode(residuals: list[np.ndarray], final_base: np.ndarray) -> np.ndarray:
    """Decode token streams: even positions exact (b+r), odd positions midpoint
    interpolation (np.interp-consistent edge rule: hold last value)."""
    s = final_base.astype(np.float64)
    for r in reversed(residuals):
        m = len(r)
        up = np.empty(2 * m)
        ru = np.empty(2 * m)
        up[0::2] = s
        ru[0::2] = r
        if m > 1:
            up[1:-1:2] = 0.5 * (s[:-1] + s[1:])
            ru[1:-1:2] = 0.5 * (r[:-1] + r[1:])
        up[-1] = s[-1]
        ru[-1] = r[-1]
        s = up + ru
    return s


# ------------------------------------------------- (a) first-token latency

def exp_a(levels_list: list[int], K: int, rng: np.random.Generator) -> list[dict]:
    """Feed samples one at a time; record samples consumed until the first
    token appears at each level. Haar frame: first token after a full frame."""
    rows = []
    for L in levels_list:
        enc = StreamingCausalLap(levels=L, K=K)
        counts = [0] * L
        first: dict[int, int] = {}
        for t in range(1, 2**L * 4 + 1):
            enc.process(np.array([rng.standard_normal()]))
            for k in range(L):
                new = len(enc.residuals[k][-1])
                counts[k] += new
                if k not in first and counts[k] > 0:
                    first[k] = t
        rows.append(
            {
                "levels": L,
                # first token at t=1 for every level: the kept first sample
                # propagates through the whole decimation chain immediately;
                # afterwards level k emits every 2^k input samples.
                "causal_first_token_samples": {str(k): first[k] for k in sorted(first)},
                "causal_emission_period_samples": {str(k): 2**k for k in range(L)},
                "haar_frame_first_token_samples_minimal": 2**L,
                "decoder_lookahead_bound_samples": 2**L - 1,
            }
        )
    return rows


# ------------------------------------------------- (b) resident memory

def exp_b(levels_list: list[int], K: int, frame_sizes: list[int]) -> list[dict]:
    rows = []
    for L in levels_list:
        enc = StreamingCausalLap(levels=L, K=K)
        causal_bytes = enc.state_floats * 8
        for F in frame_sizes:
            rows.append(
                {
                    "levels": L,
                    "causal_state_floats": enc.state_floats,
                    "causal_state_bytes": causal_bytes,
                    "haar_frame_samples": F,
                    "haar_frame_bytes": F * 8,
                    "memory_ratio": round(F / enc.state_floats, 1),
                }
            )
    return rows


# ------------------------------------------------- (c) throughput

def _time_causal(x: np.ndarray, L: int, K: int, chunk: int, reps: int) -> float:
    best = float("inf")
    for _ in range(reps):
        enc = StreamingCausalLap(levels=L, K=K)
        t0 = time.perf_counter()
        for s in range(0, len(x), chunk):
            enc.process(x[s : s + chunk])
        best = min(best, time.perf_counter() - t0)
    return best


def _time_haar(x: np.ndarray, L: int, frame: int, reps: int) -> float:
    best = float("inf")
    for _ in range(reps):
        fe = HaarFrontend(levels=L)
        t0 = time.perf_counter()
        for s in range(0, len(x), frame):
            fe.forward(x[s : s + frame])
        best = min(best, time.perf_counter() - t0)
    return best


def exp_c(x: np.ndarray, levels_list: list[int], chunk_sizes: list[int], K: int, reps: int) -> list[dict]:
    rows = []
    n = len(x)
    for L in levels_list:
        for C in chunk_sizes:
            tc = _time_causal(x, L, K, C, reps)
            th = _time_haar(x, L, C, reps)
            rows.append(
                {
                    "levels": L,
                    "chunk": C,
                    "causal_s": round(tc, 4),
                    "haar_s": round(th, 4),
                    "causal_ns_per_sample": round(tc / n * 1e9, 1),
                    "haar_ns_per_sample": round(th / n * 1e9, 1),
                    "cost_ratio": round(tc / th, 2),
                    "causal_rtf_16khz": round(16e3 * tc / n, 4),  # <1 => faster than real time
                    "haar_rtf_16khz": round(16e3 * th / n, 4),
                }
            )
    return rows


# ------------------------------------------------- (d) chunk-boundary error

def _encode_chunked(x: np.ndarray, L: int, K: int, B: int, reset: bool):
    enc = StreamingCausalLap(levels=L, K=K)
    for s in range(0, len(x), B):
        if reset:
            enc.reset()
        enc.process(x[s : s + B])
    return enc.streams()


def _decode_chunked_reset(x: np.ndarray, L: int, K: int, B: int) -> np.ndarray:
    out = np.empty_like(x)
    for s in range(0, len(x), B):
        enc = StreamingCausalLap(levels=L, K=K)
        enc.process(x[s : s + B])
        res, base = enc.streams()
        out[s : s + B] = lap_decode(res, base)
    return out


def _haar_chunked(x: np.ndarray, L: int, B: int) -> np.ndarray:
    out = np.empty_like(x)
    fe = HaarFrontend(levels=L)
    for s in range(0, len(x), B):
        ch = fe.forward(x[s : s + B])
        rec = fe.inverse(ch)
        out[s : s + B] = rec[: B]
    return out


def exp_d(x: np.ndarray, levels_list: list[int], chunk_sizes: list[int], K: int) -> list[dict]:
    n = len(x)
    W = 64  # boundary window
    rows = []
    for L in levels_list:
        # one-shot reference (persistent state, single chunk)
        enc = StreamingCausalLap(levels=L, K=K)
        enc.process(x)
        res, base = enc.streams()
        xh_one = lap_decode(res, base)[:n]
        psnr_one = psnr(x, xh_one)
        mse_one = float(np.mean((x - xh_one) ** 2))

        # sanity: one-shot frontend class on the same signal
        fe = CausalLapPyramidFrontend(levels=L, K=K)
        ch = fe.forward(x)
        psnr_fe = psnr(x, fe.inverse(ch)[:n])

        delays = StreamingCausalLap(levels=L, K=K).group_delays()

        for B in chunk_sizes:
            if B % (2**L) or B > n:
                continue
            bounds = np.arange(B, n, B)  # chunk starts (skip 0)

            # persistent state across chunks: must reproduce one-shot exactly
            res_p, base_p = _encode_chunked(x, L, K, B, reset=False)
            max_diff = max(
                float(np.max(np.abs(a - b))) if len(a) else 0.0
                for a, b in zip(res_p, res)
            )
            xh_per = lap_decode(res_p, base_p)[:n]
            psnr_per = psnr(x, xh_per)

            # state reset per chunk
            xh_rst = _decode_chunked_reset(x, L, K, B)
            psnr_rst = psnr(x, xh_rst)

            # Haar frames of size B
            xh_haar = _haar_chunked(x, L, B)
            psnr_haar = psnr(x, xh_haar)

            # boundary-localized error profiles (causal: one-sided warm-up expected)
            taus = np.arange(0, min(256, B))
            err_rst = np.mean((xh_rst[bounds[:, None] + taus] - x[bounds[:, None] + taus]) ** 2, axis=0)
            err_one = np.mean((xh_one[bounds[:, None] + taus] - x[bounds[:, None] + taus]) ** 2, axis=0)
            post_rst = float(np.mean(err_rst[:W]) / mse_one)
            post_one = float(np.mean(err_one[:W]) / mse_one)

            # reset transient isolated: delta = x_hat_reset - x_hat_persistent
            # (both share the intrinsic interpolation error, which cancels out)
            delta = xh_rst - xh_per
            d_post = np.mean(delta[bounds[:, None] + taus] ** 2, axis=0)
            d_rms = float(np.sqrt(np.mean(delta**2)))
            # post-boundary transient support: last offset with power > 1% of peak
            peak = float(np.max(d_post))
            sig = np.where(d_post > 0.01 * max(peak, 1e-30))[0]
            d_post_support = int(sig[-1] + 1) if len(sig) else 0
            # pre-boundary contamination support (edge-rule effect at chunk end):
            # trailing run of samples with |delta| > 1e-6 of the global max
            dabs = np.abs(delta)
            thr = 1e-6 * max(float(dabs.max()), 1e-30)
            d_pre_support = 0
            for b0 in bounds:
                w = 0
                t = int(b0) - 1
                while t >= 0 and dabs[t] > thr:
                    w += 1
                    t -= 1
                d_pre_support = max(d_pre_support, w)
            profile_taus = [0, 1, 2, 4, 8, 16, 32, 64, 128, 255]
            profile = {
                str(t): f"{float(d_post[t]):.3e}" for t in profile_taus if t < len(d_post)
            }

            rows.append(
                {
                    "levels": L,
                    "chunk": B,
                    "psnr_oneshot": round(psnr_one, 2),
                    "psnr_frontend_class": round(psnr_fe, 2),
                    "psnr_persistent": round(psnr_per, 2),
                    "persist_max_stream_diff": max_diff,
                    "psnr_reset": round(psnr_rst, 2),
                    "reset_penalty_db": round(psnr_per - psnr_rst, 3),
                    "psnr_haar_frame": round(psnr_haar, 2),
                    "boundary_post_ratio_reset": round(post_rst, 2),
                    "boundary_post_ratio_oneshot": round(post_one, 2),
                    "delta_rms": f"{d_rms:.3e}",
                    "delta_post_support_samples": d_post_support,
                    "delta_pre_support_samples": d_pre_support,
                    "delta_post_profile": profile,
                    "group_delay_max_input_samples": round(max(delays), 1),
                }
            )
    return rows


# ----------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"F:\小波变换\data")
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--n-files", type=int, default=8)
    ap.add_argument("--n", type=int, default=16384, help="per-file samples")
    ap.add_argument("--chunks", type=int, nargs="+", default=[256, 1024, 4096, 16384])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "reports" / "exp03_results.json"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    files = list_speech_commands(args.data)
    picks = rng.choice(len(files), size=args.n_files, replace=False)
    x = np.concatenate([load_wav(files[int(i)], args.n) for i in picks])
    n = len(x)
    print(f"[data] {args.n_files} wav files -> {n} samples", flush=True)

    x_long = np.tile(x, int(np.ceil(2**20 / n)))[: 2**20]

    print("[a] first-token latency ...", flush=True)
    res_a = exp_a(args.levels, args.K, rng)

    print("[b] resident memory ...", flush=True)
    res_b = exp_b(args.levels, args.K, args.chunks)

    print("[c] throughput (2^20 samples) ...", flush=True)
    res_c = exp_c(x_long, args.levels, args.chunks, args.K, args.reps)

    print("[d] chunk-boundary error ...", flush=True)
    res_d = exp_d(x, args.levels, args.chunks, args.K)

    results = {
        "config": {
            "levels": args.levels,
            "K": args.K,
            "tau0": 1.0,
            "c": float(np.sqrt(2.0)),
            "n_files": args.n_files,
            "n_per_file": args.n,
            "chunks": args.chunks,
            "seed": args.seed,
            "data_range": DATA_RANGE,
            "machine": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "processor": os.environ.get("PROCESSOR_IDENTIFIER", platform.processor()),
            },
        },
        "a_first_token_latency": res_a,
        "b_resident_memory": res_b,
        "c_throughput": res_c,
        "d_chunk_boundary": res_d,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] results -> {out}", flush=True)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
