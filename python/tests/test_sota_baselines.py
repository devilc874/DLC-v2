"""Smoke tests for SOTA baselines — all four always return results."""

import numpy as np
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "benchmarks"))
if BENCH_DIR not in sys.path:
    sys.path.insert(0, BENCH_DIR)

from baselines import (  # noqa: E402
    abs_error_stats,
    compress_alp,
    compress_elf,
    compress_sz3,
    compress_zfp,
)


def _signal(n=2048):
    t = np.arange(n, dtype=np.float64)
    return 10.0 + 0.01 * t + 0.5 * np.sin(0.05 * t)


def test_elf_roundtrip_lossless():
    data = _signal()
    r = compress_elf(data)
    assert r is not None and r.compressed_size > 0
    max_e, _, _ = abs_error_stats(data, r.recovered)
    assert max_e == 0.0


def test_alp_roundtrip_lossless():
    data = np.array([1.25, 1.50, 1.75, 2.00, 2.25] * 200, dtype=np.float64)
    r = compress_alp(data)
    assert r is not None and r.compressed_size > 0
    max_e, _, _ = abs_error_stats(data, r.recovered)
    assert max_e == 0.0


def test_sz3_always_runs_within_bound():
    data = _signal()
    r = compress_sz3(data, abs_error=8e-6)
    assert r is not None and r.compressed_size > 0
    assert len(r.recovered) == len(data)
    max_e, _, _ = abs_error_stats(data, r.recovered)
    assert max_e <= 8e-6 + 1e-12


def test_zfp_always_runs_within_bound():
    data = _signal()
    r = compress_zfp(data, abs_error=8e-6)
    assert r is not None and r.compressed_size > 0
    assert len(r.recovered) == len(data)
    max_e, _, _ = abs_error_stats(data, r.recovered)
    assert max_e <= 8e-6 + 1e-12
