#!/usr/bin/env python3
"""
Generate deterministic test data for DLC unit, cross-language, and benchmark tests.

Outputs raw little-endian float64 .bin files in the test_data/ directory.
All synthetic datasets use a fixed seed for reproducibility.

Downloads 6 real-world time-series datasets from the Numenta Anomaly Benchmark
(NAB) GitHub at their NATURAL length (no looping — avoids LZ77 dictionary exploits).

Also generates continuous (non-repeating) 1M-point synthetic datasets for fair
throughput benchmarking against gzip.

Usage:
    python generate_test_data.py
"""

import os
import csv
import urllib.request
import numpy as np

SEED = 42
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Numenta Anomaly Benchmark (NAB) dataset URLs ────────────────────────────
NAB_BASE = "https://raw.githubusercontent.com/numenta/NAB/master/data/"

NAB_DATASETS = {
    # name_suffix → (relative URL path, value column)
    # Saved at natural length — NO looping!
    "real_cpu":            ("realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv", "value"),
    "real_nyc_taxi":       ("realKnownCause/nyc_taxi.csv", "value"),
    "real_twitter_vol":    ("realTweets/Twitter_volume_AAPL.csv", "value"),
    "real_machine_temp":   ("realKnownCause/machine_temperature_system_failure.csv", "value"),
    "real_ambient_temp":   ("realKnownCause/ambient_temperature_system_failure.csv", "value"),
    "real_ec2_network":    ("realAWSCloudwatch/ec2_network_in_5abac7.csv", "value"),
}


def _download_nab_dataset(name: str, url_path: str, value_col: str) -> np.ndarray:
    """
    Download a NAB CSV from GitHub, extract float values.
    Returns the data at its NATURAL length — no looping.
    """
    full_url = NAB_BASE + url_path
    csv_path = os.path.join(OUTPUT_DIR, f"_nab_{name}_raw.csv")

    print(f"  Downloading {name}...")
    try:
        urllib.request.urlretrieve(full_url, csv_path)
    except Exception as e:
        print(f"    WARNING: Download failed: {e}")
        return None

    values = []
    try:
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    values.append(float(row[value_col]))
                except (ValueError, KeyError):
                    continue
    except Exception as e:
        print(f"    WARNING: Parse failed: {e}")
        return None
    finally:
        if os.path.exists(csv_path):
            os.unlink(csv_path)

    if not values:
        return None

    raw = np.array(values, dtype=np.float64)
    n = len(raw)
    suffix = f"{n // 1000}k" if n >= 1000 else str(n)
    print(f"    Got {n} points → {name}_{suffix}.bin  "
          f"(min={raw.min():.2f}, max={raw.max():.2f})")
    return raw


def generate_all():
    rng = np.random.default_rng(SEED)
    datasets = {}

    # ═════════════════════════════════════════════════════════════════════
    # SYNTHETIC DATASETS (unit tests + benchmarks)
    # ═════════════════════════════════════════════════════════════════════

    # 1. Small test — 200 points (fast unit tests)
    datasets["small_200"] = np.sin(np.linspace(0, 4 * np.pi, 200))

    # 2. Step function — 10K points (abrupt variance change)
    datasets["step_10k"] = np.concatenate([
        np.full(5000, 1.0),
        np.full(5000, 100.0),
    ])

    # 3. Noisy ramp — 100K points (linear trend + Gaussian noise)
    n = 100_000
    datasets["noisy_ramp_100k"] = np.linspace(0, 1000, n) + rng.normal(0, 0.01, n)

    # 4. Random walk — 50K points
    datasets["random_walk_50k"] = np.cumsum(rng.standard_normal(50_000))

    # 5. Edge cases — 100 points (NaN, ±Inf, subnormals)
    edge = np.zeros(100, dtype=np.float64)
    edge[0] = np.nan
    edge[1] = np.inf
    edge[2] = -np.inf
    edge[3] = np.finfo(np.float64).tiny
    edge[4] = np.nextafter(0.0, 1.0)
    edge[5] = -np.nextafter(0.0, 1.0)
    datasets["edge_cases_100"] = edge

    # ═════════════════════════════════════════════════════════════════════
    # CONTINUOUS 1M-POINT SYNTHETIC DATASETS (for throughput benchmarks)
    # NO np.tile / NO repetition — every point is unique
    # ═════════════════════════════════════════════════════════════════════

    # 6. Sine wave — 1M points (smooth, compressible)
    t = np.linspace(0, 100 * np.pi, 1_000_000)
    datasets["sine_1m"] = np.sin(t)

    # 7. Linear ramp — 1M points (perfectly linear)
    datasets["ramp_1m"] = np.linspace(0, 10_000, 1_000_000)

    # 8. Continuous random walk — 1M points (challenging, non-repeating)
    datasets["synthetic_random_walk_1m"] = np.cumsum(
        rng.standard_normal(1_000_000)
    )

    # 9. Continuous noisy sine — 1M points (realistic sensor-like)
    #    Multi-frequency sine + Gaussian noise + slow drift
    t2 = np.linspace(0, 500 * np.pi, 1_000_000)
    base = (50.0
            + 15.0 * np.sin(t2 * 0.003)        # slow oscillation
            + 5.0 * np.sin(t2 * 0.07)           # medium frequency
            + 1.5 * np.sin(t2 * 0.5))           # fast ripple
    drift = np.linspace(0, 10, 1_000_000)       # slow upward drift
    noise = rng.normal(0, 1.5, 1_000_000)       # Gaussian sensor noise
    datasets["synthetic_noisy_sine_1m"] = base + drift + noise

    # ═════════════════════════════════════════════════════════════════════
    # REAL-WORLD NAB DATASETS (saved at natural length — NO looping)
    # ═════════════════════════════════════════════════════════════════════

    print("\n  Downloading 6 NAB real-world datasets (natural length)...")
    for name, (url_path, value_col) in NAB_DATASETS.items():
        data = _download_nab_dataset(name, url_path, value_col)
        if data is not None:
            n = len(data)
            suffix = f"{n // 1000}k" if n >= 1000 else str(n)
            datasets[f"{name}_{suffix}"] = data

    # ═════════════════════════════════════════════════════════════════════
    # WRITE ALL DATASETS
    # ═════════════════════════════════════════════════════════════════════

    print(f"\nWriting {len(datasets)} datasets to {OUTPUT_DIR}/")
    for name, data in datasets.items():
        path = os.path.join(OUTPUT_DIR, f"{name}.bin")
        data.astype("<f8").tofile(path)
        size_mb = os.path.getsize(path) / 1e6
        print(f"  {name:35s}  samples={len(data):>10,}  "
              f"size={size_mb:>6.2f} MB  → {os.path.basename(path)}")


if __name__ == "__main__":
    print("=" * 60)
    print("DLC Test Data Generator (no looping)")
    print("=" * 60)
    generate_all()
    print(f"\nDone. All datasets generated.")
