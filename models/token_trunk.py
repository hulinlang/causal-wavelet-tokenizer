"""Stage-1 shared token trunk (WAT-style, 1-D audio first).

Architecture follows WAT (arXiv:2606.02631) Sec. 4.3 exactly:
- per-modality value adapter A_in: Linear(C -> width), optional value scale s
- shared token-wise trunk: LayerNorm-MLP encoder to latent dim, decoder back
- output adapter A_out: Linear(width -> C), divide by s
- no attention, no codebook: isolates the schema/frontend question

Tokens: one-level decomposition gives N = n/2 tokens with C = 2 values
each (Haar: [cA, cD]; causal_lap: [base, residual]), so both frontends
share the identical token layout and differ ONLY in the transform.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TokenTrunkAE(nn.Module):
    def __init__(self, channels: int = 2, width: int = 32, latent: int = 16, hidden: int = 64,
                 value_scale: float = 1.0):
        super().__init__()
        self.value_scale = value_scale
        self.a_in = nn.Linear(channels, width)
        self.enc = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, hidden), nn.GELU(),
                                 nn.Linear(hidden, latent))
        self.dec = nn.Sequential(nn.LayerNorm(latent), nn.Linear(latent, hidden), nn.GELU(),
                                 nn.Linear(hidden, width))
        self.a_out = nn.Linear(width, channels)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, N, C) -> reconstructed tokens (B, N, C)."""
        e = self.a_in(tokens * self.value_scale)
        z = self.enc(e)
        h = self.dec(z)
        return self.a_out(h) / self.value_scale
