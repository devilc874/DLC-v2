"""Tests for adaptive precision: default 16-bit floor + ablation 8–20."""

import numpy as np

from dlc.encoders import (
    MAX_ERROR_BOUND,
    precision_floor_for_error_bound,
    select_precision,
)


class TestAdaptivePrecision:
    def test_production_floor_is_16(self):
        assert precision_floor_for_error_bound(MAX_ERROR_BOUND) == 16

    def test_default_clamps_low_preq_to_floor(self):
        residuals = np.array([1e-4, -2e-4, 5e-5])
        for preq in (8, 10, 12, 14, 16):
            p = select_precision(residuals, preq, enforce_error_bound=True)
            assert p == 16, f"preq={preq} produced p*={p}, expected 16"

    def test_default_still_adapts_above_floor(self):
        # Very tiny residuals → adaptive wants high bits; cap by request
        residuals = np.array([1e-4, -2e-4, 5e-5])
        p20 = select_precision(residuals, 20, enforce_error_bound=True)
        assert 16 <= p20 <= 20

    def test_ablation_allows_full_8_to_20(self):
        residuals = np.array([1e-4, -2e-4, 5e-5])
        assert select_precision(residuals, 8, enforce_error_bound=False) == 8
        assert select_precision(residuals, 12, enforce_error_bound=False) == 12
        p20 = select_precision(residuals, 20, enforce_error_bound=False)
        assert 8 <= p20 <= 20

    def test_near_zero_escape_to_8(self):
        residuals = np.array([0.0, 0.0, 1e-20])
        assert select_precision(residuals, 16, enforce_error_bound=True) == 8
        assert select_precision(residuals, 16, enforce_error_bound=False) == 8
