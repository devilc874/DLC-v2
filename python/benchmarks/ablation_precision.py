"""
ablation_precision.py — Precision sweep using --ablation-precision semantics.

Default engine keeps a 16-bit floor. This script turns the floor OFF
(enforce_error_bound=False) and sweeps p_req over {8,10,...,20}, plus a
production reference row at p_req=16 with the floor ON.

Usage:
    python python/benchmarks/ablation_precision.py
    python python/benchmarks/ablation_precision.py --limit 3
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
import time
from collections import Counter

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
PYTHON_ROOT = os.path.join(PROJECT_ROOT, "python")
TEST_DATA_DIR = os.path.join(PROJECT_ROOT, "test_data")

if PYTHON_ROOT not in sys.path:
    sys.path.insert(0, PYTHON_ROOT)

from dlc.codec import compress, decompress  # noqa: E402
from dlc.format import read_dlc_file, unpack_window_block  # noqa: E402
from dlc.encoders import (  # noqa: E402
    MAX_ERROR_BOUND,
    precision_floor_for_error_bound,
    select_precision,
)


PREC_SWEEP = [8, 10, 12, 14, 16, 18, 20]


def _list_bins(limit: int | None = None):
    if not os.path.isdir(TEST_DATA_DIR):
        return []
    files = sorted(f for f in os.listdir(TEST_DATA_DIR) if f.endswith(".bin"))
    if limit is not None:
        files = files[:limit]
    return files


def _precision_histogram(payload: bytes) -> Counter:
    hist: Counter = Counter()
    offset = 0
    while offset < len(payload):
        meta, _enc, consumed = unpack_window_block(payload, offset)
        offset += consumed
        hist[int(meta.precision_bits)] += 1
    return hist


def _hist_summary(hist: Counter) -> str:
    if not hist:
        return "-"
    return "{" + ", ".join(f"{p}:{c}" for p, c in sorted(hist.items())) + "}"


def _run_one(data: np.ndarray, precision_bits: int, enforce_error_bound: bool):
    with tempfile.NamedTemporaryFile(suffix=".dlc", delete=False) as f:
        path = f.name
    try:
        t0 = time.perf_counter()
        compress(
            data, path,
            precision_bits=precision_bits,
            num_workers=1,
            enforce_error_bound=enforce_error_bound,
        )
        t1 = time.perf_counter()
        size = os.path.getsize(path)
        recovered = decompress(path)
        t2 = time.perf_counter()
        _header, payload = read_dlc_file(path)
        hist = _precision_histogram(payload)
        diff = np.abs(data - recovered)
        max_err = float(np.max(diff)) if len(diff) else 0.0
        mae = float(np.mean(diff)) if len(diff) else 0.0
        rmse = float(np.sqrt(np.mean(diff ** 2))) if len(diff) else 0.0
        mb = data.nbytes / (1024 * 1024)
        return {
            "size": size,
            "ratio": data.nbytes / size if size else None,
            "max_err": max_err,
            "mae": mae,
            "rmse": rmse,
            "comp_s": t1 - t0,
            "decomp_s": t2 - t1,
            "throughput_MBps": mb / (t1 - t0) if (t1 - t0) > 0 else None,
            "hist_str": _hist_summary(hist),
            "mean_precision": (
                sum(p * c for p, c in hist.items()) / sum(hist.values())
                if hist else None
            ),
        }
    finally:
        if os.path.isfile(path):
            os.unlink(path)


def _demo_selector():
    residuals = np.array([1e-4, -2e-4, 5e-5])
    floor = precision_floor_for_error_bound(MAX_ERROR_BOUND)
    print()
    print("=" * 90)
    print("  SELECTOR DEMO")
    print("=" * 90)
    print(f"  Production floor for error {MAX_ERROR_BOUND:g}: {floor} bits")
    print(f"  {'p_req':>6} {'default p*':>12} {'ablation p*':>12}")
    for p in PREC_SWEEP:
        prod = select_precision(residuals, p, enforce_error_bound=True)
        abl = select_precision(residuals, p, enforce_error_bound=False)
        print(f"  {p:>6} {prod:>12} {abl:>12}")
    print()
    print("  Default = 16-bit floor + adaptive up to p_req.")
    print("  Ablation (--ablation-precision) = full 8–20 adaptive range.")
    print()


def run(limit: int | None = None, csv_path: str | None = None):
    _demo_selector()
    bins = _list_bins(limit)
    if not bins:
        print("ERROR: No .bin files in test_data/. Run: python test_data/generate_test_data.py")
        return 1

    rows = []
    print("=" * 120)
    print("  PRECISION SWEEP")
    print("=" * 120)

    for bin_file in bins:
        path = os.path.join(TEST_DATA_DIR, bin_file)
        data = np.fromfile(path, dtype=np.float64)
        name = bin_file.replace(".bin", "")[:26]
        print()
        print(f"  Dataset: {name}  (N={len(data):,})")
        print(f"  {'p_req':>6} {'mode':>10} {'ratio':>8} {'max_err':>10} "
              f"{'MAE':>10} {'MB/s':>8} {'mean_p':>8}  precision_hist")
        print("  " + "-" * 100)

        # Production reference
        try:
            prod = _run_one(data, 16, enforce_error_bound=True)
            print(
                f"  {16:>6} {'DEFAULT':>10} {prod['ratio']:8.2f}x "
                f"{prod['max_err']:10.2e} {prod['mae']:10.2e} "
                f"{prod['throughput_MBps']:8.1f} {prod['mean_precision']:8.2f}  "
                f"{prod['hist_str']}"
            )
            rows.append({
                "dataset": name, "n": len(data), "p_req": 16, "mode": "default",
                **{k: prod[k] for k in (
                    "size", "ratio", "max_err", "mae", "rmse",
                    "comp_s", "decomp_s", "throughput_MBps", "mean_precision",
                )},
                "precision_hist": prod["hist_str"],
            })
        except Exception as exc:
            print(f"  default@16 FAILED: {exc}")

        for p in PREC_SWEEP:
            try:
                r = _run_one(data, p, enforce_error_bound=False)
                print(
                    f"  {p:>6} {'ABLATION':>10} {r['ratio']:8.2f}x "
                    f"{r['max_err']:10.2e} {r['mae']:10.2e} "
                    f"{r['throughput_MBps']:8.1f} {r['mean_precision']:8.2f}  "
                    f"{r['hist_str']}"
                )
                rows.append({
                    "dataset": name, "n": len(data), "p_req": p, "mode": "ablation",
                    **{k: r[k] for k in (
                        "size", "ratio", "max_err", "mae", "rmse",
                        "comp_s", "decomp_s", "throughput_MBps", "mean_precision",
                    )},
                    "precision_hist": r["hist_str"],
                })
            except Exception as exc:
                print(f"  ablation@{p} FAILED: {exc}")

    out_csv = csv_path or os.path.join(PROJECT_ROOT, "paper", "precision_ablation_results.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    if rows:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print()
        print(f"  Wrote CSV: {out_csv}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="DLC precision ablation sweep")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--csv", type=str, default=None)
    args = ap.parse_args()
    raise SystemExit(run(args.limit, args.csv))


if __name__ == "__main__":
    main()
