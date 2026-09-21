"""
Tests for predictive models (models.py).

Validates model fitting, selection logic, reconstruction error,
and edge cases (NaN, Inf, constant data).
"""

import numpy as np
import pytest

from dlc.models import (
    fit_linear, fit_quadratic, fit_xor_delta, reconstruct_xor_delta,
    select_model, MODEL_LINEAR, MODEL_QUADRATIC, MODEL_XOR_DELTA,
    _float64_to_bits, _bits_to_float64,
)


class TestFloat64Bitwise:
    def test_round_trip_normal(self):
        for v in [0.0, 1.0, -1.0, 3.14159, 1e-300, 1e300]:
            assert _bits_to_float64(_float64_to_bits(v)) == v

    def test_round_trip_special(self):
        bits = _float64_to_bits(float('inf'))
        assert _bits_to_float64(bits) == float('inf')


class TestLinear:
    def test_perfect_linear(self):
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        m, c, res = fit_linear(y)
        assert abs(m - 2.0) < 1e-10
        assert abs(c - 2.0) < 1e-10
        assert np.max(np.abs(res)) < 1e-10

    def test_single_point(self):
        y = np.array([42.0])
        m, c, res = fit_linear(y)
        assert c == 42.0


class TestQuadratic:
    def test_perfect_quadratic(self):
        x = np.arange(20, dtype=np.float64)
        y = 0.5 * x ** 2 - 3.0 * x + 1.0
        a, b, c, res = fit_quadratic(y)
        assert abs(a - 0.5) < 1e-8
        assert abs(b - (-3.0)) < 1e-8
        assert abs(c - 1.0) < 1e-8
        assert np.max(np.abs(res)) < 1e-8


class TestXorDelta:
    def test_round_trip(self):
        y = np.array([1.0, 2.0, 3.0, 1.5, -0.5])
        anchor, xor_res = fit_xor_delta(y)
        reconstructed = reconstruct_xor_delta(anchor, xor_res)
        np.testing.assert_array_equal(reconstructed, y)

    def test_lossless_for_arbitrary_data(self):
        rng = np.random.default_rng(42)
        y = rng.standard_normal(100)
        anchor, xor_res = fit_xor_delta(y)
        reconstructed = reconstruct_xor_delta(anchor, xor_res)
        np.testing.assert_array_equal(reconstructed, y)

    def test_single_point(self):
        y = np.array([99.9])
        anchor, xor_res = fit_xor_delta(y)
        assert anchor == 99.9
        assert xor_res == []


class TestModelSelection:
    def test_perfect_linear_selects_linear_or_xor(self):
        """Perfect linear data should select Linear or XOR-Delta (both have ~0 SSR)."""
        y = np.arange(100, dtype=np.float64) * 3.0 + 1.0
        model_id, params, res = select_model(y)
        # XOR-Delta is lossless (0 SSR) and gets 5% preference, so it may win
        assert model_id in (MODEL_LINEAR, MODEL_XOR_DELTA)

    def test_noisy_quadratic_selects_quadratic_or_better(self):
        x = np.arange(50, dtype=np.float64)
        y = 0.1 * x ** 2 + x + 5.0 + np.random.default_rng(42).normal(0, 0.1, 50)
        model_id, params, res = select_model(y)
        # Should pick quadratic or XOR-delta
        assert model_id in (MODEL_QUADRATIC, MODEL_XOR_DELTA)

    def test_random_data_selects_xor_delta(self):
        """Highly random data benefits from XOR-Delta."""
        rng = np.random.default_rng(42)
        y = rng.standard_normal(100)
        model_id, params, res = select_model(y)
        # XOR-Delta is lossless → should be preferred
        assert model_id == MODEL_XOR_DELTA

    def test_nan_data_uses_xor_delta(self):
        y = np.array([1.0, np.nan, 3.0, np.inf, -1.0])
        model_id, params, res = select_model(y)
        assert model_id == MODEL_XOR_DELTA

    def test_empty_data(self):
        y = np.array([], dtype=np.float64)
        model_id, params, res = select_model(y)
        assert model_id == MODEL_LINEAR
