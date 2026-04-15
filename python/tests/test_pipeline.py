"""
Tests for the parallel pipeline (pipeline.py).

Validates determinism (same output with different thread counts),
round-trip integrity, and the error bound.
"""

import numpy as np
import pytest

from dlc.pipeline import compress_parallel, decompress_payload, _process_block


class TestProcessBlock:
    def test_small_block_round_trip(self):
        data = np.sin(np.linspace(0, 2 * np.pi, 200))
        payload = _process_block(data, precision_bits=12)
        reconstructed = decompress_payload(payload, precision_bits=12)
        assert len(reconstructed) == len(data)
        # Check error bound
        max_err = np.max(np.abs(data - reconstructed))
        assert max_err < 8e-6, f"Max error {max_err} >= 8e-6"

    def test_constant_block(self):
        data = np.full(500, 42.0)
        payload = _process_block(data, precision_bits=12)
        reconstructed = decompress_payload(payload, precision_bits=12)
        max_err = np.max(np.abs(data - reconstructed))
        assert max_err < 8e-6


class TestCompressParallel:
    def test_determinism_across_thread_counts(self):
        """Output must be identical regardless of thread count."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(10_000)

        payload_1 = compress_parallel(data, precision_bits=12, chunk_size=2000, num_workers=1)
        payload_2 = compress_parallel(data, precision_bits=12, chunk_size=2000, num_workers=4)

        assert payload_1 == payload_2, "Output differs between 1 and 4 workers!"

    def test_round_trip_error_bound(self):
        t = np.linspace(0, 20 * np.pi, 50_000)
        data = np.sin(t) + 0.5 * np.cos(3 * t)

        payload = compress_parallel(data, precision_bits=12, chunk_size=10_000, num_workers=2)
        reconstructed = decompress_payload(payload, precision_bits=12)

        assert len(reconstructed) == len(data)
        max_err = np.max(np.abs(data - reconstructed))
        assert max_err < 8e-6, f"Max error {max_err} >= 8e-6"

    def test_empty_data(self):
        data = np.array([], dtype=np.float64)
        payload = compress_parallel(data)
        reconstructed = decompress_payload(payload)
        assert len(reconstructed) == 0

    def test_single_point(self):
        data = np.array([3.14])
        payload = compress_parallel(data)
        reconstructed = decompress_payload(payload)
        assert len(reconstructed) == 1
        assert abs(reconstructed[0] - 3.14) < 8e-6
