# Causal Wavelet Tokenizer

Toward a unified wavelet token schema for native multimodal LLMs:
text via BPE embedding lookup, images via Haar DWT, audio/video via
**time-causal wavelets** (Lindeberg 2026), unified as float token sequences
feeding a shared Transformer trunk.

Theoretical basis:

1. Lindeberg, *Time-causal and time-recursive wavelets* (arXiv:2510.05834) — causal frontend theory
2. Ding, *Wavelet as Tokenizer* (arXiv:2606.02631) — shared wavelet token schema (Haar)
3. Amiri et al., *Harmonizer* (Mathematics 2025, 13, 1819) — signal-to-token pipeline for LLMs

## Stage 0: frontend go/no-go (no training)

Compare causal-wavelet vs Haar frontends under non-parametric fixed-rate
energy selection (WAT Table 4 protocol), on synthetic and real signals.

```
python experiments/stage0_energy_selection.py
```

## Layout

- `causal_wavelet/` — time-causal scale-space + wavelet decomposition via cascaded first-order recursive filters
- `frontends/` — Haar DWT baseline, causal wavelet frontend
- `experiments/` — staged experiment scripts
- `reports/` — per-stage experiment reports (math, protocol, results)
