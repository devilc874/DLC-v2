"""
Adaptive variance-guided window segmentation (IMPROVED).

Changes from original:
  1. Adaptive MAX_WINDOW: 512 (small), 1024 (medium), 4096 (large blocks)
  2. Stagnation guard: stops expansion when variance jumps > 1.5x
  3. Fast-path: constant block (variance ≈ 0) → single window
  4. Tuned factors: MEAN_VAR_FACTOR = 10.0 (was 13), GLOBAL_VAR_FACTOR = 0.10 (was 0.12)

Window bounds:
  MIN_WINDOW = 16
  MAX_WINDOW = adaptive (512 / 1024 / 4096)
"""

import numpy as np
from typing import List

# ── Constants ──────────────────────────────────────────────────────────────
MIN_WINDOW        = 16
MAX_WINDOW_SMALL  = 512     # blocks < 5,000 samples
MAX_WINDOW_MEDIUM = 1024    # blocks < 100,000 samples
MAX_WINDOW_LARGE  = 4096    # blocks >= 100,000 samples
MEAN_VAR_FACTOR   = 10.0    # was 13.0 — tighter = better local fit
GLOBAL_VAR_FACTOR = 0.10    # was 0.12
CONST_VAR_EPS     = 1e-20   # below this, block is treated as constant


def _variance(data: np.ndarray) -> float:
    """Compute population variance (Welford's for stability)."""
    n = len(data)
    if n < 2:
        return 0.0
    return float(np.var(data, ddof=0))


def _max_window_for_block(block_len: int) -> int:
    """Adaptive maximum window based on block size."""
    if block_len < 5_000:
        return min(MAX_WINDOW_SMALL, block_len)
    elif block_len < 100_000:
        return min(MAX_WINDOW_MEDIUM, block_len)
    else:
        return min(MAX_WINDOW_LARGE, block_len)


def segment_block(block: np.ndarray,
                  min_w: int = MIN_WINDOW,
                  max_w: int = None) -> List[np.ndarray]:
    """
    Segment a block of float64 data into variable-length windows.

    Uses variance-guided greedy expansion with binary-search boundary
    refinement.  Adaptive MAX_WINDOW is used if max_w is not provided.

    Args:
        block: 1D numpy array of float64 values.
        min_w: Minimum window size (default 16).
        max_w: Maximum window size (default: adaptive per block length).

    Returns:
        List of numpy arrays whose concatenation reproduces the block
        exactly (no gaps, no overlaps).
    """
    n = len(block)
    if n == 0:
        return []
    if n <= min_w:
        return [block.copy()]

    # Fast-path: entire block is essentially constant
    global_var = _variance(block)
    if global_var < CONST_VAR_EPS:
        return [block.copy()]

    # Adaptive MAX_WINDOW
    if max_w is None:
        max_w = _max_window_for_block(n)

    windows = []
    i = 0

    while i < n:
        remaining = n - i

        # Last fragment: just emit it
        if remaining <= min_w:
            windows.append(block[i:i + remaining].copy())
            break

        # Initial window of min_w
        win_size = min(min_w, remaining)
        initial_var = _variance(block[i:i + win_size])
        threshold = max(initial_var * MEAN_VAR_FACTOR,
                        global_var * GLOBAL_VAR_FACTOR)

        # Avoid zero threshold (constant signal) — allow maximum expansion
        if threshold == 0.0:
            threshold = 1e-30

        # Greedy doubling expansion with stagnation guard
        candidate = win_size
        prev_var = initial_var

        while candidate < max_w and i + candidate < n:
            next_size = min(candidate * 2, max_w, remaining)
            if next_size <= candidate:
                candidate = next_size
                break
            seg_var = _variance(block[i:i + next_size])

            if seg_var > threshold:
                break

            # Stagnation guard: big jump means signal changed character
            if prev_var > CONST_VAR_EPS and seg_var / prev_var > 1.5:
                break

            prev_var = seg_var
            candidate = next_size

        # Binary search for exact boundary in [win_size, candidate]
        lo = win_size
        hi = min(candidate, remaining)

        while lo < hi:
            mid = (lo + hi + 1) // 2
            if _variance(block[i:i + mid]) <= threshold:
                lo = mid
            else:
                hi = mid - 1

        win_size = lo
        windows.append(block[i:i + win_size].copy())
        i += win_size

    return windows
