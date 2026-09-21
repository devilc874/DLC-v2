"""
compare_sota.py — DLC v2 vs contemporary compressors (SZ3, ZFP, Elf, ALP).

Near-lossless methods (DLC, SZ3, ZFP) use the same absolute error tolerance
(default 8e-6, DLC's production contract). Elf and ALP are lossless baselines.

Usage:
    python python/benchmarks/compare_sota.py
    python python/benchmarks/compare_sota.py --abs-error 8e-6 --limit 5

Optional deps for native backends:
    pip install pysz zfpy
Elf/ALP always run via pure-Python references bundled in baselines/.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
PYTHON_ROOT = os.path.join(PROJECT_ROOT, "python")
TEST_DATA_DIR = os.path.join(PROJECT_ROOT, "test_data")

if PYTHON_ROOT not in sys.path:
    sys.path.insert(0, PYTHON_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from dlc.codec import compress as dlc_compress, decompress as dlc_decompress  # noqa: E402
from baselines import (  # noqa: E402
    abs_error_stats,
    compress_alp,
    compress_elf,
    compress_sz3,
    compress_zfp,
    probe_availability,
)


def _list_bins(limit: int | None = None):
    if not os.path.isdir(TEST_DATA_DIR):
        return []
    files = sorted(f for f in os.listdir(TEST_DATA_DIR) if f.endswith(".bin"))
    if limit is not None:
        files = files[:limit]
    return files


def _run_dlc(data: np.ndarray, precision_bits: int = 16):
    with tempfile.NamedTemporaryFile(suffix=".dlc", delete=False) as f:
        path = f.name
    try:
        t0 = time.perf_counter()
        dlc_compress(data, path, precision_bits=precision_bits, num_workers=1)
        t1 = time.perf_counter()
        size = os.path.getsize(path)
        recovered = dlc_decompress(path)
        t2 = time.perf_counter()
        return {
            "size": size,
            "recovered": recovered,
            "comp_s": t1 - t0,
            "decomp_s": t2 - t1,
        }
    finally:
        if os.path.isfile(path):
            os.unlink(path)


def _fmt_ratio(raw: int, size):
    if size is None or size <= 0:
        return "  N/A  "
    return f"{raw / size:6.2f}x"


def _fmt_err(err):
    if err is None or (isinstance(err, float) and (np.isnan(err) or err < 0)):
        return "   N/A   "
    if err == 0.0:
        return "  0 (lossless)"
    return f"{err:10.2e}"


def run(abs_error: float = 8e-6, precision_bits: int = 16, limit: int | None = None,
        csv_path: str | None = None):
    bins = _list_bins(limit)
    if not bins:
        print("ERROR: No .bin files in test_data/. Run: python test_data/generate_test_data.py")
        return 1

    avail = probe_availability()
    print()
    print("=" * 120)
    print("  DLC v2 vs SOTA FLOAT COMPRESSORS  (equal absolute error for near-lossless)")
    print("=" * 120)
    print(f"  Absolute error tolerance (DLC / SZ3 / ZFP): {abs_error:g}")
    print(f"  DLC precision request: {precision_bits} bits (production error floor ON)")
    print("  Backends:")
    print(f"    SZ3    READY ({'pysz' if avail.get('SZ3_native') else 'pure-python fallback'})")
    print(f"    ZFP    READY ({'zfpy' if avail.get('ZFP_native') else 'pure-python fallback'})")
    print("    Elf    READY (pure-python)")
    print("    ALP    READY (pure-python)")
    print("-" * 120)

    header = f"{'Dataset':<28} {'N':>8} {'DLC':>8} {'SZ3':>8} {'ZFP':>8} {'Elf':>8} {'ALP':>8} {'Best':>8}"
    print(header)
    print("-" * 120)

    rows = []
    wins = {"DLC": 0, "SZ3": 0, "ZFP": 0, "Elf": 0, "ALP": 0}

    for bin_file in bins:
        path = os.path.join(TEST_DATA_DIR, bin_file)
        data = np.fromfile(path, dtype=np.float64)
        raw = data.nbytes
        name = bin_file.replace(".bin", "")[:26]
        n = len(data)

        results = {}

        # DLC
        try:
            dlc = _run_dlc(data, precision_bits=precision_bits)
            max_e, mae, rmse = abs_error_stats(data, dlc["recovered"])
            results["DLC"] = {
                "size": dlc["size"], "max_err": max_e, "mae": mae, "rmse": rmse,
                "comp_s": dlc["comp_s"], "decomp_s": dlc["decomp_s"],
                "backend": "dlc-python", "lossless": False,
            }
        except Exception as exc:
            results["DLC"] = {"size": None, "max_err": None, "error": str(exc)}

        # SZ3 (always — native or pure-Python fallback)
        sz = compress_sz3(data, abs_error=abs_error)
        max_e, mae, rmse = abs_error_stats(data, sz.recovered)
        results["SZ3"] = {
            "size": sz.compressed_size, "max_err": max_e, "mae": mae, "rmse": rmse,
            "comp_s": sz.compress_seconds, "decomp_s": sz.decompress_seconds,
            "backend": sz.backend, "lossless": False,
        }

        # ZFP (always — native or pure-Python fallback)
        zfp = compress_zfp(data, abs_error=abs_error)
        max_e, mae, rmse = abs_error_stats(data, zfp.recovered)
        results["ZFP"] = {
            "size": zfp.compressed_size, "max_err": max_e, "mae": mae, "rmse": rmse,
            "comp_s": zfp.compress_seconds, "decomp_s": zfp.decompress_seconds,
            "backend": zfp.backend, "lossless": False,
        }

        # Elf (always)
        elf = compress_elf(data)
        max_e, mae, rmse = abs_error_stats(data, elf.recovered)
        results["Elf"] = {
            "size": elf.compressed_size, "max_err": max_e, "mae": mae, "rmse": rmse,
            "comp_s": elf.compress_seconds, "decomp_s": elf.decompress_seconds,
            "backend": elf.backend, "lossless": True,
        }

        # ALP (always)
        alp = compress_alp(data)
        max_e, mae, rmse = abs_error_stats(data, alp.recovered)
        results["ALP"] = {
            "size": alp.compressed_size, "max_err": max_e, "mae": mae, "rmse": rmse,
            "comp_s": alp.compress_seconds, "decomp_s": alp.decompress_seconds,
            "backend": alp.backend, "lossless": True,
        }

        candidates = {k: v["size"] for k, v in results.items() if v.get("size")}
        best = min(candidates, key=candidates.get) if candidates else "-"
        if best in wins:
            wins[best] += 1

        row_txt = f"{name:<28} {n:>8,}"
        for key in ("DLC", "SZ3", "ZFP", "Elf", "ALP"):
            row_txt += f" {_fmt_ratio(raw, results[key].get('size')):>8}"
        row_txt += f" {best:>8}"
        print(row_txt)

        for key, meta in results.items():
            rows.append({
                "dataset": name,
                "n": n,
                "raw_bytes": raw,
                "method": key,
                "compressed_bytes": meta.get("size"),
                "ratio": (raw / meta["size"]) if meta.get("size") else None,
                "max_error": meta.get("max_err"),
                "mae": meta.get("mae"),
                "rmse": meta.get("rmse"),
                "compress_s": meta.get("comp_s"),
                "decompress_s": meta.get("decomp_s"),
                "backend": meta.get("backend"),
                "lossless": meta.get("lossless"),
                "abs_error_tolerance": abs_error,
            })

    print("-" * 120)
    print()
    print("  Wins (smallest compressed size):")
    for k, v in sorted(wins.items(), key=lambda kv: -kv[1]):
        print(f"    {k:<6} {v}")
    print()

    # Error table for near-lossless trio
    print("=" * 120)
    print("  ROUND-TRIP MAX |error|  (near-lossless trio share abs tolerance; Elf/ALP are lossless)")
    print("=" * 120)
    err_h = f"{'Dataset':<28} {'DLC':>12} {'SZ3':>12} {'ZFP':>12} {'Elf':>14} {'ALP':>14}"
    print(err_h)
    print("-" * 120)
    by_ds = {}
    for r in rows:
        by_ds.setdefault(r["dataset"], {})[r["method"]] = r
    for ds, methods in by_ds.items():
        line = f"{ds:<28}"
        for m in ("DLC", "SZ3", "ZFP", "Elf", "ALP"):
            line += f" {_fmt_err(methods.get(m, {}).get('max_error')):>12}"
        print(line)
    print("-" * 120)
    print()
    print("  Note: SZ3/ZFP always run (pure-Python fallback if pysz/zfpy not installed).")
    print()

    out_csv = csv_path or os.path.join(PROJECT_ROOT, "paper", "sota_comparison_results.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote CSV: {out_csv}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="DLC vs SZ3/ZFP/Elf/ALP comparison")
    ap.add_argument("--abs-error", type=float, default=8e-6,
                    help="Absolute error bound for DLC/SZ3/ZFP (default 8e-6)")
    ap.add_argument("--precision", type=int, default=16,
                    help="DLC requested precision bits (default 16)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only first N datasets (smoke test)")
    ap.add_argument("--csv", type=str, default=None, help="Output CSV path")
    args = ap.parse_args()
    raise SystemExit(run(args.abs_error, args.precision, args.limit, args.csv))


if __name__ == "__main__":
    main()
