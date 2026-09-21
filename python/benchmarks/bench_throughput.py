#!/usr/bin/env python3
"""
DLC Throughput Benchmarks.

Measures compression ratio, throughput (MB/s), and wall time
with varying thread counts using the Python engine.

Usage:
    python bench_throughput.py
"""

import time
import os
import sys
import tempfile
import numpy as np

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dlc.codec import compress, decompress


def run_benchmark(data: np.ndarray, name: str, worker_counts=(1, 2, 4, 8)):
    """Run compress/decompress benchmarks for different thread counts."""
    original_size = data.nbytes
    print(f"\n{'='*60}")
    print(f"Benchmark: {name}")
    print(f"  Samples: {len(data):,}  |  Raw size: {original_size / 1e6:.1f} MB")
    print(f"{'='*60}")
    print(f"  {'Workers':<10} {'Compress':<12} {'Decompress':<12} {'Ratio':<10} {'MB/s':<10}")
    print(f"  {'-'*54}")

    for workers in worker_counts:
        with tempfile.NamedTemporaryFile(suffix='.dlc', delete=False) as f:
            path = f.name

        try:
            # Compress
            t0 = time.perf_counter()
            compress(data, path, num_workers=workers)
            t_compress = time.perf_counter() - t0

            comp_size = os.path.getsize(path)
            ratio = original_size / comp_size if comp_size > 0 else float('inf')

            # Decompress
            t0 = time.perf_counter()
            _ = decompress(path)
            t_decompress = time.perf_counter() - t0

            throughput = (original_size / 1e6) / t_compress if t_compress > 0 else 0

            print(f"  {workers:<10} {t_compress:>8.3f}s   {t_decompress:>8.3f}s   "
                  f"{ratio:>6.1f}x   {throughput:>6.1f}")
        finally:
            os.unlink(path)


def main():
    print("DLC Throughput Benchmarks")
    print(f"NumPy dtype: float64 (8 bytes per sample)")

    rng = np.random.default_rng(42)

    # Benchmark 1: Sine wave (smooth, highly compressible)
    t = np.linspace(0, 200 * np.pi, 1_000_000)
    run_benchmark(np.sin(t), "Sine Wave (1M points)")

    # Benchmark 2: Noisy signal
    noisy = np.sin(t) + rng.normal(0, 0.01, len(t))
    run_benchmark(noisy, "Noisy Sine (1M points)")

    # Benchmark 3: Random walk (hard to compress)
    walk = np.cumsum(rng.standard_normal(1_000_000))
    run_benchmark(walk, "Random Walk (1M points)")

    # Benchmark 4: Ramp (linear)
    run_benchmark(np.linspace(0, 10000, 1_000_000), "Linear Ramp (1M points)")

    print(f"\n{'='*60}")
    print("Benchmarks complete.")


if __name__ == "__main__":
    main()
