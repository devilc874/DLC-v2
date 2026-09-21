"""
Hypothesis property-based fuzz tests.

Tests the engine with:
  - Arbitrary float64 values (NaN, Inf, subnormals)
  - Random realistic signals
  - Edge case lengths

Ensures no crashes and error bound < 8e-6 for finite values.
"""

import numpy as np
import pytest

try:
    from hypothesis import given, settings, HealthCheck
    from hypothesis.strategies import floats, integers
    from hypothesis.extra.numpy import arrays
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False

from dlc.pipeline import compress_parallel, decompress_payload


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
class TestFuzz:

    @given(arrays(np.float64, shape=integers(min_value=1, max_value=200),
                  elements=floats(allow_nan=True, allow_infinity=True,
                                  allow_subnormal=True)))
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow],
              deadline=None)
    def test_no_crash_on_any_input(self, data):
        """Engine must not crash on ANY float64 input."""
        payload = compress_parallel(data, precision_bits=12, chunk_size=500, num_workers=1)
        result = decompress_payload(payload, precision_bits=12)

        assert len(result) == len(data)

        # For finite values, check error bound
        mask = np.isfinite(data)
        if np.any(mask):
            finite_err = np.max(np.abs(data[mask] - result[mask]))
            assert finite_err < 8e-6, f"Error {finite_err} >= 8e-6"

    @given(arrays(np.float64, shape=integers(min_value=100, max_value=5000),
                  elements=floats(min_value=-1e6, max_value=1e6,
                                  allow_nan=False, allow_infinity=False)))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow],
              deadline=None)
    def test_error_bound_realistic(self, data):
        """Error bound must hold for realistic signals."""
        payload = compress_parallel(data, precision_bits=12, chunk_size=1000, num_workers=1)
        result = decompress_payload(payload, precision_bits=12)

        assert len(result) == len(data)
        max_err = np.max(np.abs(data - result))
        assert max_err < 8e-6, f"Max error {max_err} >= 8e-6"

    @given(arrays(np.float64, shape=integers(min_value=1, max_value=50),
                  elements=floats(min_value=-1.0, max_value=1.0,
                                  allow_nan=False, allow_infinity=False)))
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow],
              deadline=None)
    def test_small_arrays(self, data):
        """Small arrays must round-trip correctly."""
        payload = compress_parallel(data, precision_bits=12, chunk_size=100, num_workers=1)
        result = decompress_payload(payload, precision_bits=12)

        assert len(result) == len(data)
        max_err = np.max(np.abs(data - result))
        assert max_err < 8e-6
