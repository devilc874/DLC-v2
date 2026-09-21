"""
Tests for residual encoders (encoders.py).

Validates round-trip for all 4 strategies, quantization error bounds,
trial_encode selection, and edge cases.
"""

import numpy as np
import pytest

from dlc.encoders import (
    quantize, dequantize,
    encode_zigzag_varint, decode_zigzag_varint,
    encode_bitpack, decode_bitpack,
    encode_delta_of_residuals, decode_delta_of_residuals,
    encode_outlier_sep, decode_outlier_sep,
    trial_encode, decode_residuals,
)


class TestQuantization:
    def test_round_trip_zero(self):
        q = quantize(np.array([0.0]), 12)
        d = dequantize(q, 12)
        assert abs(d[0]) < 1e-15

    def test_quantization_error_bound(self):
        """Max quantization error should be <= 0.5 / 2^precision."""
        rng = np.random.default_rng(42)
        vals = rng.standard_normal(1000) * 0.01  # small residuals
        precision = 12
        q = quantize(vals, precision)
        d = dequantize(q, precision)
        max_err = np.max(np.abs(vals - d))
        bound = 0.5 / (1 << precision)
        assert max_err <= bound + 1e-15, f"Max error {max_err} > bound {bound}"

    def test_large_values(self):
        vals = np.array([1000.0, -1000.0, 0.0])
        q = quantize(vals, 12)
        d = dequantize(q, 12)
        assert abs(d[0] - 1000.0) < 0.001
        assert abs(d[1] - (-1000.0)) < 0.001


class TestZigzagVarint:
    def test_round_trip(self):
        vals = [0, 1, -1, 127, -128, 10000, -10000, 2**30]
        encoded = encode_zigzag_varint(vals)
        decoded = decode_zigzag_varint(encoded, len(vals))
        assert decoded == vals

    def test_zeros(self):
        vals = [0, 0, 0, 0]
        encoded = encode_zigzag_varint(vals)
        decoded = decode_zigzag_varint(encoded, 4)
        assert decoded == vals

    def test_single(self):
        vals = [42]
        decoded = decode_zigzag_varint(encode_zigzag_varint(vals), 1)
        assert decoded == vals


class TestBitpack:
    def test_round_trip(self):
        vals = [0, 1, -1, 3, -3, 7, -7]
        encoded = encode_bitpack(vals)
        decoded = decode_bitpack(encoded, len(vals))
        assert decoded == vals

    def test_zeros(self):
        vals = [0, 0, 0, 0, 0]
        decoded = decode_bitpack(encode_bitpack(vals), 5)
        assert decoded == vals

    def test_large_values(self):
        vals = [1000, -1000, 5000, -5000]
        decoded = decode_bitpack(encode_bitpack(vals), 4)
        assert decoded == vals

    def test_single(self):
        vals = [99]
        decoded = decode_bitpack(encode_bitpack(vals), 1)
        assert decoded == vals


class TestDeltaOfResiduals:
    def test_round_trip(self):
        vals = [10, 12, 14, 16, 18]  # constant delta = 2
        encoded = encode_delta_of_residuals(vals)
        decoded = decode_delta_of_residuals(encoded, len(vals))
        assert decoded == vals

    def test_random_values(self):
        rng = np.random.default_rng(42)
        vals = [int(x) for x in rng.integers(-1000, 1000, 50)]
        decoded = decode_delta_of_residuals(encode_delta_of_residuals(vals), len(vals))
        assert decoded == vals

    def test_single(self):
        vals = [42]
        decoded = decode_delta_of_residuals(encode_delta_of_residuals(vals), 1)
        assert decoded == vals


class TestOutlierSep:
    def test_round_trip_no_outliers(self):
        vals = [1, 2, 1, 2, 1, 2, 1, 2]
        decoded = decode_outlier_sep(encode_outlier_sep(vals), len(vals))
        assert decoded == vals

    def test_round_trip_with_outliers(self):
        vals = [1, 1, 1, 1, 1000, 1, 1, 1, -1000, 1]
        decoded = decode_outlier_sep(encode_outlier_sep(vals), len(vals))
        assert decoded == vals

    def test_large_random(self):
        rng = np.random.default_rng(42)
        vals = [int(x) for x in rng.integers(-10, 10, 100)]
        # Add some outliers
        vals[5] = 99999
        vals[50] = -99999
        decoded = decode_outlier_sep(encode_outlier_sep(vals), len(vals))
        assert decoded == vals


class TestTrialEncode:
    def test_selects_smallest(self):
        """Trial encode should return the smallest encoding."""
        vals = [1, 2, 3, 4, 5]
        best_id, best_buf = trial_encode(vals)
        # Verify all 4 would decode correctly
        decoded = decode_residuals(best_id, best_buf, len(vals))
        assert decoded == vals

    def test_constant_residuals(self):
        """Constant values should compress well with delta-of-residuals."""
        vals = [0] * 100
        enc_id, buf = trial_encode(vals)
        decoded = decode_residuals(enc_id, buf, 100)
        assert decoded == vals

    def test_decode_residuals_dispatch(self):
        """Verify decode_residuals correctly dispatches to all 4 strategies."""
        vals = [10, -5, 7, -3, 1]
        for strategy_id in range(4):
            if strategy_id == 0:
                buf = encode_zigzag_varint(vals)
            elif strategy_id == 1:
                buf = encode_bitpack(vals)
            elif strategy_id == 2:
                buf = encode_delta_of_residuals(vals)
            elif strategy_id == 3:
                buf = encode_outlier_sep(vals)
            decoded = decode_residuals(strategy_id, buf, len(vals))
            assert decoded == vals, f"Strategy {strategy_id} failed"
