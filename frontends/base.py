"""Unified frontend interface.

A frontend maps a 1-D signal to a list of coefficient channels and back.
Both frontends are exactly invertible when no coefficients are dropped,
which stage 0 verifies numerically before any selection experiment.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Frontend(Protocol):
    name: str

    def forward(self, x: np.ndarray) -> list[np.ndarray]: ...

    def inverse(self, channels: list[np.ndarray]) -> np.ndarray: ...
