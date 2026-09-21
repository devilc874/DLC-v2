"""Shared types for SOTA baseline compressors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class BaselineResult:
    name: str
    compressed_size: int
    recovered: Optional[np.ndarray]
    compress_seconds: float
    decompress_seconds: float
    lossless: bool
    backend: str  # e.g. "pysz", "zfpy", "pure-python"
    notes: str = ""

    @property
    def max_error(self) -> float:
        return float("nan") if self.recovered is None else float("nan")


def abs_error_stats(original: np.ndarray, recovered: np.ndarray):
    """Return (max_abs, mae, rmse)."""
    diff = np.abs(original.astype(np.float64) - recovered.astype(np.float64))
    max_abs = float(np.max(diff)) if len(diff) else 0.0
    mae = float(np.mean(diff)) if len(diff) else 0.0
    rmse = float(np.sqrt(np.mean(diff ** 2))) if len(diff) else 0.0
    return max_abs, mae, rmse
