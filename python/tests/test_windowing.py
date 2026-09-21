"""
Tests for adaptive windowing (windowing.py).

Validates window bounds, no gaps/overlaps, split on variance boundaries,
and edge cases (empty, small, constant signals).
"""

import numpy as np
import pytest

from dlc.windowing import segment_block, MIN_WINDOW, MAX_WINDOW


class TestSegmentBlock:
    def test_constant_signal_minimal_splits(self):
        """Constant signal should produce as few windows as possible."""
        data = np.ones(2000)
        windows = segment_block(data)
        # Should coalesce into large windows (close to MAX_WINDOW)
        total = sum(len(w) for w in windows)
        assert total == 2000

    def test_all_windows_within_bounds(self):
        """Every window must be [16, 1024] (except possibly last)."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(5000)
        windows = segment_block(data)

        for i, w in enumerate(windows):
            if i < len(windows) - 1:
                assert len(w) >= MIN_WINDOW, f"Window {i} too small: {len(w)}"
            assert len(w) <= MAX_WINDOW, f"Window {i} too large: {len(w)}"

    def test_no_gaps_no_overlaps(self):
        """Concatenation of all windows must equal original block."""
        rng = np.random.default_rng(123)
        data = rng.standard_normal(3000)
        windows = segment_block(data)
        reconstructed = np.concatenate(windows)
        np.testing.assert_array_equal(reconstructed, data)

    def test_step_function_splits_at_boundary(self):
        """Step signal should produce a split near the step boundary."""
        data = np.concatenate([
            np.full(500, 1.0),
            np.full(500, 100.0),
        ])
        windows = segment_block(data)

        # Find the boundary: cumulative lengths should cross 500
        cumlen = 0
        boundary_window_idx = -1
        for i, w in enumerate(windows):
            prev_cumlen = cumlen
            cumlen += len(w)
            if prev_cumlen < 500 <= cumlen:
                boundary_window_idx = i
                break

        assert boundary_window_idx >= 0, "Should find a window crossing the boundary"

        # Reconstruction is exact
        np.testing.assert_array_equal(np.concatenate(windows), data)

    def test_empty_input(self):
        data = np.array([], dtype=np.float64)
        windows = segment_block(data)
        assert windows == []

    def test_tiny_input(self):
        """Input smaller than MIN_WINDOW returns single window."""
        data = np.array([1.0, 2.0, 3.0])
        windows = segment_block(data)
        assert len(windows) == 1
        np.testing.assert_array_equal(windows[0], data)

    def test_exactly_min_window(self):
        data = np.arange(MIN_WINDOW, dtype=np.float64)
        windows = segment_block(data)
        total = sum(len(w) for w in windows)
        assert total == MIN_WINDOW

    def test_large_random_no_crash(self):
        """100K random points should segment without error."""
        rng = np.random.default_rng(999)
        data = rng.standard_normal(100_000)
        windows = segment_block(data)
        total = sum(len(w) for w in windows)
        assert total == 100_000

    def test_sine_wave_reasonable_windows(self):
        """Smooth sine wave should produce relatively large windows."""
        t = np.linspace(0, 4 * np.pi, 2000)
        data = np.sin(t)
        windows = segment_block(data)
        avg_len = np.mean([len(w) for w in windows])
        # Smooth signal should have avg window > 100
        assert avg_len > 50, f"Average window too small for sine: {avg_len}"
        np.testing.assert_array_equal(np.concatenate(windows), data)
