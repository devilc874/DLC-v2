#!/usr/bin/env python3
"""
Automated C++ DLC Engine Benchmarks.

Executes the compiled C++ binary via subprocess to measure compression
ratio and throughput (MB/s) across different thread counts.

Dynamically locates .bin files in test_data/ and the C++ binary.

Usage:
    python python/benchmarks/bench_cpp.py
"""

import os
import sys
import time
import subprocess
import tempfile
import platform

# ── Locate the C++ binary ────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
TEST_DATA_DIR = os.path.join(PROJECT_ROOT, 'test_data')


def _find_cpp_binary():
    """Search common build locations for the compiled dlc binary."""
    candidates = [
        os.path.join(PROJECT_ROOT, 'cpp', 'build', 'dlc.exe'),
        os.path.join(PROJECT_ROOT, 'cpp', 'build', 'Release', 'dlc.exe'),
        os.path.join(PROJECT_ROOT, 'cpp', 'build', 'dlc'),
        os.path.join(PROJECT_ROOT, 'cpp', 'build', 'Release', 'dlc'),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def _find_test_files():
    """Discover all .bin files in test_data/, return {name: abs_path}."""
    bins = {}
    if not os.path.isdir(TEST_DATA_DIR):
        print(f"ERROR: test_data directory not found at {TEST_DATA_DIR}")
        sys.exit(1)

    for fname in sorted(os.listdir(TEST_DATA_DIR)):
        if fname.endswith('.bin'):
            name = fname[:-4]
            bins[name] = os.path.join(TEST_DATA_DIR, fname)
    return bins


# ── Benchmark Runner ─────────────────────────────────────────────────────────

def run_cpp_benchmark(binary: str, input_path: str, name: str,
                      worker_counts=(1, 2, 4, 8)):
    """
    Run C++ compress/decompress for each worker count.
    Measures wall-clock time, compression ratio, and throughput.
    """
    original_size = os.path.getsize(input_path)
    num_samples = original_size // 8  # float64 = 8 bytes

    print(f"\n{'=' * 70}")
    print(f"C++ Benchmark: {name}")
    print(f"  Samples: {num_samples:>12,}  |  Raw size: {original_size / 1e6:.1f} MB")
    print(f"{'=' * 70}")
    print(f"  {'Workers':<10} {'Compress':<12} {'Decompress':<12} "
          f"{'Ratio':<10} {'MB/s':<10} {'Status':<8}")
    print(f"  {'-' * 62}")

    # Add MSYS2 ucrt64 to PATH for DLL dependencies
    env = os.environ.copy()
    msys2_bin = r"C:\msys64\ucrt64\bin"
    if os.path.isdir(msys2_bin) and msys2_bin not in env.get("PATH", ""):
        env["PATH"] = msys2_bin + os.pathsep + env.get("PATH", "")

    for workers in worker_counts:
        dlc_path = tempfile.mktemp(suffix='.dlc')
        out_path = tempfile.mktemp(suffix='.bin')
        status = "OK"

        try:
            # ── Compress ─────────────────────────────────────────────────
            cmd_compress = [
                binary, "compress",
                "-i", input_path,
                "-o", dlc_path,
                "--workers", str(workers),
            ]
            t0 = time.perf_counter()
            result = subprocess.run(
                cmd_compress, capture_output=True, text=True,
                timeout=120, env=env,
            )
            t_compress = time.perf_counter() - t0

            if result.returncode != 0:
                status = "FAIL"
                print(f"  {workers:<10} {'ERROR':<12} {'—':<12} "
                      f"{'—':<10} {'—':<10} {status}")
                if result.stderr:
                    print(f"    stderr: {result.stderr.strip()[:100]}")
                continue

            comp_size = os.path.getsize(dlc_path) if os.path.isfile(dlc_path) else 0
            ratio = original_size / comp_size if comp_size > 0 else 0.0
            throughput = (original_size / 1e6) / t_compress if t_compress > 0 else 0.0

            # ── Decompress ───────────────────────────────────────────────
            cmd_decompress = [
                binary, "decompress",
                "-i", dlc_path,
                "-o", out_path,
            ]
            t0 = time.perf_counter()
            result_d = subprocess.run(
                cmd_decompress, capture_output=True, text=True,
                timeout=120, env=env,
            )
            t_decompress = time.perf_counter() - t0

            if result_d.returncode != 0:
                status = "D-FAIL"
                t_decompress = -1

            # ── Verify round-trip ────────────────────────────────────────
            if os.path.isfile(out_path) and status == "OK":
                recovered_size = os.path.getsize(out_path)
                if recovered_size != original_size:
                    status = f"SIZE!({recovered_size})"

            decomp_str = f"{t_decompress:>8.3f}s" if t_decompress >= 0 else "   ERROR"

            print(f"  {workers:<10} {t_compress:>8.3f}s   {decomp_str}   "
                  f"{ratio:>6.1f}x   {throughput:>6.1f}   {status}")

        except subprocess.TimeoutExpired:
            print(f"  {workers:<10} {'TIMEOUT':<12} {'—':<12} "
                  f"{'—':<10} {'—':<10} TIMEOUT")
        except Exception as e:
            print(f"  {workers:<10} {'ERR':<12} {'—':<12} "
                  f"{'—':<10} {'—':<10} {str(e)[:30]}")
        finally:
            for p in [dlc_path, out_path]:
                if os.path.isfile(p):
                    try:
                        os.unlink(p)
                    except OSError:
                        pass


def main():
    print("=" * 70)
    print("DLC C++ Engine — Automated Benchmarks")
    print("=" * 70)

    # Find binary
    binary = _find_cpp_binary()
    if binary is None:
        print("\nERROR: C++ binary not found!")
        print("Expected locations:")
        print(f"  {os.path.join(PROJECT_ROOT, 'cpp', 'build', 'dlc.exe')}")
        print(f"  {os.path.join(PROJECT_ROOT, 'cpp', 'build', 'dlc')}")
        print("\nBuild with:")
        print("  cd cpp/build && cmake .. -G \"MinGW Makefiles\" ... && mingw32-make -j8")
        sys.exit(1)

    print(f"\nBinary: {binary}")
    print(f"Data:   {TEST_DATA_DIR}")

    # Find test files
    all_bins = _find_test_files()
    if not all_bins:
        print(f"\nERROR: No .bin files found in {TEST_DATA_DIR}")
        print("Run: python test_data/generate_test_data.py")
        sys.exit(1)

    print(f"Found {len(all_bins)} datasets: {', '.join(all_bins.keys())}")

    # ── Primary benchmarks (large datasets, multi-worker) ────────────────
    primary = ['real_cpu_1m', 'ramp_1m', 'sine_1m']
    print(f"\n{'#' * 70}")
    print(f"# PRIMARY BENCHMARKS (1M-point datasets, multi-worker)")
    print(f"{'#' * 70}")

    for name in primary:
        if name in all_bins:
            run_cpp_benchmark(binary, all_bins[name], name,
                              worker_counts=(1, 2, 4, 8))
        else:
            print(f"\n  SKIP: {name}.bin not found — run generate_test_data.py first")

    # ── Secondary benchmarks (smaller datasets, single worker) ───────────
    secondary = [k for k in all_bins.keys()
                 if k not in primary and k not in ('edge_cases_100', 'small_200')]
    if secondary:
        print(f"\n{'#' * 70}")
        print(f"# SECONDARY BENCHMARKS (smaller datasets, 1 worker)")
        print(f"{'#' * 70}")

        for name in secondary:
            run_cpp_benchmark(binary, all_bins[name], name,
                              worker_counts=(1,))

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("All benchmarks complete.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
