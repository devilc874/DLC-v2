#!/usr/bin/env python3
"""
DLC vs GZIP vs ZLIB - Terminal Comparison Across ALL Datasets.

Compresses every .bin file in test_data/ with GZIP, ZLIB, and DLC C++,
printing a formatted comparison table.

Usage:
    python python/benchmarks/compare_all.py
"""

import os
import sys
import time
import gzip
import zlib
import tempfile
import subprocess

# Force UTF-8 output on Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
TEST_DATA_DIR = os.path.join(PROJECT_ROOT, 'test_data')

SEP = "=" * 115


def find_cpp_binary():
    for p in [
        os.path.join(PROJECT_ROOT, 'cpp', 'build', 'dlc.exe'),
        os.path.join(PROJECT_ROOT, 'cpp', 'build', 'Release', 'dlc.exe'),
        os.path.join(PROJECT_ROOT, 'cpp', 'build', 'dlc'),
    ]:
        if os.path.isfile(p):
            return p
    return None


def get_env():
    env = os.environ.copy()
    msys2 = r"C:\msys64\ucrt64\bin"
    if os.path.isdir(msys2):
        env["PATH"] = msys2 + os.pathsep + env.get("PATH", "")
    return env


def compress_gzip(raw_bytes):
    t0 = time.perf_counter()
    out = gzip.compress(raw_bytes, compresslevel=6)
    return len(out), time.perf_counter() - t0


def compress_zlib(raw_bytes):
    t0 = time.perf_counter()
    out = zlib.compress(raw_bytes, level=6)
    return len(out), time.perf_counter() - t0


def compress_dlc(binary, input_path, workers=4):
    dlc_path = tempfile.mktemp(suffix='.dlc')
    env = get_env()
    try:
        t0 = time.perf_counter()
        r = subprocess.run(
            [binary, "compress", "-i", input_path, "-o", dlc_path,
             "--workers", str(workers)],
            capture_output=True, text=True, timeout=120, env=env,
        )
        elapsed = time.perf_counter() - t0
        if r.returncode != 0:
            return None, elapsed
        size = os.path.getsize(dlc_path) if os.path.isfile(dlc_path) else 0
        return size, elapsed
    except subprocess.TimeoutExpired:
        return None, 120
    except Exception:
        return None, 0
    finally:
        if os.path.isfile(dlc_path):
            os.unlink(dlc_path)


def fmt_ratio(original, compressed):
    if compressed is None or compressed == 0:
        return "  ERR "
    return f"{original / compressed:>5.1f}x"


def fmt_time(t):
    if t < 0.001:
        return f"{t*1000:>5.1f}ms"
    return f"{t:>6.3f}s"


def fmt_mbps(original, t):
    if t <= 0:
        return "    --"
    return f"{(original / 1e6) / t:>6.1f}"


def main():
    binary = find_cpp_binary()

    print(SEP)
    print("  DLC v2 COMPRESSION COMPARISON -- ALL DATASETS")
    print(SEP)
    print(f"  DLC C++ Binary: {binary or 'NOT FOUND'}")
    print(f"  Data Directory: {TEST_DATA_DIR}")
    print(SEP)

    # Discover datasets
    files = []
    for f in sorted(os.listdir(TEST_DATA_DIR)):
        if f.endswith('.bin'):
            path = os.path.join(TEST_DATA_DIR, f)
            files.append((f[:-4], path))

    if not files:
        print("  ERROR: No .bin files found!")
        sys.exit(1)

    print(f"  Found {len(files)} datasets\n")

    # Header
    print(f"  {'Dataset':<35} {'Samples':>10} {'Raw':>6}"
          f" | {'GZIP':>6} {'Time':>7} {'MB/s':>6}"
          f" | {'ZLIB':>6} {'Time':>7} {'MB/s':>6}"
          f" | {'DLC':>6} {'Time':>7} {'MB/s':>6}"
          f" | {'Winner':<7}")
    print(f"  {'-'*35} {'-'*10} {'-'*6}"
          f"-+-{'-'*6}-{'-'*7}-{'-'*6}"
          f"-+-{'-'*6}-{'-'*7}-{'-'*6}"
          f"-+-{'-'*6}-{'-'*7}-{'-'*6}"
          f"-+-{'-'*7}")

    total_raw = 0
    total_gzip = 0
    total_zlib = 0
    total_dlc = 0
    dlc_wins = 0
    count = 0

    for name, path in files:
        raw_size = os.path.getsize(path)
        samples = raw_size // 8
        raw_mb = raw_size / 1e6

        # Skip tiny files
        if raw_size < 400:
            continue

        raw_bytes = open(path, 'rb').read()

        # GZIP
        gz_size, gz_time = compress_gzip(raw_bytes)

        # ZLIB
        zl_size, zl_time = compress_zlib(raw_bytes)

        # DLC
        if binary:
            dlc_size, dlc_time = compress_dlc(binary, path)
        else:
            dlc_size, dlc_time = None, 0

        # Determine winner by best ratio
        ratios = {}
        if gz_size and gz_size > 0:
            ratios['GZIP'] = raw_size / gz_size
        if zl_size and zl_size > 0:
            ratios['ZLIB'] = raw_size / zl_size
        if dlc_size and dlc_size > 0:
            ratios['DLC'] = raw_size / dlc_size

        winner = max(ratios, key=lambda k: ratios[k]) if ratios else "--"
        tag = f">> {winner}" if winner == 'DLC' else f"   {winner}"

        if winner == 'DLC':
            dlc_wins += 1

        total_raw += raw_size
        total_gzip += (gz_size or 0)
        total_zlib += (zl_size or 0)
        total_dlc += (dlc_size or raw_size)
        count += 1

        dlc_r = fmt_ratio(raw_size, dlc_size) if dlc_size else " N/A  "
        dlc_t = fmt_time(dlc_time) if dlc_size else "  N/A  "
        dlc_m = fmt_mbps(raw_size, dlc_time) if dlc_size else "   N/A"

        print(f"  {name:<35} {samples:>10,} {raw_mb:>5.2f}"
              f" | {fmt_ratio(raw_size, gz_size)} {fmt_time(gz_time)} {fmt_mbps(raw_size, gz_time)}"
              f" | {fmt_ratio(raw_size, zl_size)} {fmt_time(zl_time)} {fmt_mbps(raw_size, zl_time)}"
              f" | {dlc_r} {dlc_t} {dlc_m}"
              f" | {tag}")

    # Summary
    print(f"\n{SEP}")
    print("  SUMMARY")
    print(SEP)
    print(f"  Datasets tested:  {count}")
    print(f"  Total raw data:   {total_raw / 1e6:.1f} MB")
    if total_gzip > 0:
        print(f"  GZIP total:       {total_gzip / 1e6:.1f} MB  "
              f"(overall ratio: {total_raw / total_gzip:.2f}x)")
    if total_zlib > 0:
        print(f"  ZLIB total:       {total_zlib / 1e6:.1f} MB  "
              f"(overall ratio: {total_raw / total_zlib:.2f}x)")
    if total_dlc > 0:
        print(f"  DLC total:        {total_dlc / 1e6:.1f} MB  "
              f"(overall ratio: {total_raw / total_dlc:.2f}x)")
    print(f"  DLC wins:         {dlc_wins}/{count} datasets")
    print(SEP)


if __name__ == "__main__":
    main()
