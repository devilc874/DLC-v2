"""
download_large_datasets.py — Fetch real-world large time-series datasets
and generate massive synthetic sensor data for DLC v2 benchmarking.

Usage:
    python test_data/download_large_datasets.py

Downloads:
  1. UCI Household Electric Power Consumption  (~2.07M readings, 4 years)
  2. Jena Climate (full)                       (~420K readings, 8 years)
  3. NYC Taxi (extended)                        (~10K+ readings)

Generates:
  4. Massive sensor simulation (50M samples = 400 MB)
  5. Multi-frequency vibration sensor (10M samples = 80 MB)
  6. Industrial temperature ramp (10M samples = 80 MB)
  7. Noisy heartbeat signal (10M samples = 80 MB)
  8. Chaotic sensor (Lorenz attractor, 5M samples = 40 MB)

Total: ~800 MB+ of test data
"""

import os
import sys
import io
import struct
import zipfile
import urllib.request
import csv
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def save_float64_bin(data, filename):
    """Save numpy array as raw float64 binary."""
    path = os.path.join(SCRIPT_DIR, filename)
    arr = np.asarray(data, dtype=np.float64)
    arr.tofile(path)
    size_mb = os.path.getsize(path) / 1e6
    print(f"  [OK] Saved {filename}: {len(arr):,} samples, {size_mb:.1f} MB")
    return path


# ═══════════════════════════════════════════════════════════════════════════
# 1. UCI Household Electric Power Consumption
#    Source: https://archive.ics.uci.edu/dataset/235
#    2,075,259 minute-by-minute readings over ~4 years
#    Columns: global_active_power, voltage, sub_metering, etc.
# ═══════════════════════════════════════════════════════════════════════════

def download_household_power():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"
    print("\n[1/8] UCI Household Electric Power Consumption (~130 MB download)...")

    try:
        print("  Downloading...")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req, timeout=120)
        data = response.read()
        print(f"  Downloaded {len(data) / 1e6:.1f} MB")

        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            txt_name = [n for n in zf.namelist() if n.endswith('.txt')][0]
            with zf.open(txt_name) as f:
                content = f.read().decode('utf-8', errors='replace')

        lines = content.strip().split('\n')
        header = lines[0]
        print(f"  Parsing {len(lines)-1:,} rows...")

        # Extract: Global_active_power, Global_reactive_power, Voltage,
        #          Global_intensity, Sub_metering_1, Sub_metering_2, Sub_metering_3
        cols = {
            'power': [],       # Global_active_power (kW)
            'voltage': [],     # Voltage (V)
            'intensity': [],   # Global_intensity (A)
        }

        for line in lines[1:]:
            parts = line.strip().split(';')
            if len(parts) < 7:
                continue
            try:
                power = float(parts[2])
                voltage = float(parts[4])
                intensity = float(parts[5])
                cols['power'].append(power)
                cols['voltage'].append(voltage)
                cols['intensity'].append(intensity)
            except (ValueError, IndexError):
                cols['power'].append(0.0)
                cols['voltage'].append(0.0)
                cols['intensity'].append(0.0)

        save_float64_bin(cols['power'], 'real_household_power_2m.bin')
        save_float64_bin(cols['voltage'], 'real_household_voltage_2m.bin')
        save_float64_bin(cols['intensity'], 'real_household_intensity_2m.bin')

        # Combined: interleave all 3 channels for a ~6M sample file
        combined = np.empty(len(cols['power']) * 3, dtype=np.float64)
        combined[0::3] = cols['power']
        combined[1::3] = cols['voltage']
        combined[2::3] = cols['intensity']
        save_float64_bin(combined, 'real_household_combined_6m.bin')

        return True

    except Exception as e:
        print(f"  [FAIL] Failed: {e}")
        print(f"  -> Manual download: {url}")
        return False


# ═══════════════════════════════════════════════════════════════════════════
# 2. Jena Climate (Full — 8 years of weather data)
# ═══════════════════════════════════════════════════════════════════════════

def download_jena_climate():
    url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
    print("\n[2/8] Jena Climate Dataset (full, 8 years)...")

    try:
        print("  Downloading...")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req, timeout=60)
        data = response.read()

        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            csv_name = [n for n in zf.namelist() if n.endswith('.csv')][0]
            with zf.open(csv_name) as f:
                content = f.read().decode('utf-8')

        lines = content.strip().split('\n')
        reader = csv.reader(lines)
        header = next(reader)
        print(f"  Columns: {header}")

        # Extract multiple columns
        temp_col = header.index('T (degC)') if 'T (degC)' in header else 1
        pressure_col = header.index('p (mbar)') if 'p (mbar)' in header else 2

        temps = []
        pressures = []
        all_cols = []

        for row in reader:
            try:
                temps.append(float(row[temp_col]))
                pressures.append(float(row[pressure_col]))
                # All numeric columns interleaved
                for i in range(1, len(row)):
                    try:
                        all_cols.append(float(row[i]))
                    except ValueError:
                        all_cols.append(0.0)
            except (ValueError, IndexError):
                pass

        save_float64_bin(temps, 'real_jena_temperature_full_420k.bin')
        save_float64_bin(pressures, 'real_jena_pressure_full_420k.bin')
        save_float64_bin(all_cols, 'real_jena_all_channels_5m.bin')
        return True

    except Exception as e:
        print(f"  [FAIL] Failed: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════
# 3. NOAA Daily Temperature (large, multi-year)
# ═══════════════════════════════════════════════════════════════════════════

def download_noaa_temperature():
    """Download NOAA GHCN daily temperature data for a major station."""
    url = "https://www.ncei.noaa.gov/data/global-hourly/access/2023/72534014732.csv"
    print("\n[3/8] NOAA Hourly Weather Data (2023)...")

    try:
        print("  Downloading...")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req, timeout=60)
        content = response.read().decode('utf-8', errors='replace')

        lines = content.strip().split('\n')
        reader = csv.reader(lines)
        header = next(reader)
        print(f"  Rows: {len(lines)-1:,}")

        # Find temperature column (TMP)
        tmp_col = None
        for i, h in enumerate(header):
            if 'TMP' in h.upper():
                tmp_col = i
                break

        if tmp_col is None:
            print("  [FAIL] TMP column not found, skipping")
            return False

        temps = []
        for row in reader:
            try:
                raw = row[tmp_col]
                # NOAA format: "+0123,1" means 12.3°C with quality flag 1
                val = float(raw.split(',')[0]) / 10.0
                if -90 < val < 70:  # sanity check
                    temps.append(val)
            except (ValueError, IndexError):
                pass

        if len(temps) > 100:
            save_float64_bin(temps, f'real_noaa_hourly_temp_{len(temps)//1000}k.bin')
            return True
        else:
            print(f"  [FAIL] Only {len(temps)} valid readings, skipping")
            return False

    except Exception as e:
        print(f"  [FAIL] Failed: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════
# 4-8. Generate Massive Synthetic Sensor Data
# ═══════════════════════════════════════════════════════════════════════════

def generate_massive_sensor(n_samples=50_000_000):
    """
    50M samples = 400 MB.
    Simulates a multi-modal industrial sensor:
    - Slow sinusoidal drift (temperature)
    - Fast vibration superimposed
    - Random walk noise
    - Occasional spikes (anomalies)
    """
    print(f"\n[4/8] Massive sensor simulation ({n_samples//1_000_000}M samples, {n_samples*8/1e6:.0f} MB)...")
    print("  Generating (this takes ~30 seconds)...")

    # Generate in chunks to avoid memory issues
    chunk_size = 5_000_000
    chunks = []

    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        n = end - start
        t = np.arange(start, end, dtype=np.float64)

        # Multi-component sensor signal
        signal = (
            20.0 +                                          # DC offset (room temp)
            3.0 * np.sin(2 * np.pi * t / 86400) +         # Daily cycle
            0.5 * np.sin(2 * np.pi * t / 3600) +          # Hourly oscillation
            0.05 * np.sin(2 * np.pi * t / 60) +           # Minute vibration
            0.01 * np.cumsum(np.random.randn(n)) / np.sqrt(n) +  # Random walk
            np.random.randn(n) * 0.002                     # Thermal noise
        )

        # Add occasional spikes (0.01% of samples)
        spike_mask = np.random.rand(n) < 0.0001
        signal[spike_mask] += np.random.randn(np.sum(spike_mask)) * 5.0

        chunks.append(signal)
        pct = end / n_samples * 100
        print(f"  ... {pct:.0f}% ({end//1_000_000}M / {n_samples//1_000_000}M)")

    data = np.concatenate(chunks)
    save_float64_bin(data, f'massive_sensor_{n_samples//1_000_000}m.bin')


def generate_vibration_sensor(n_samples=10_000_000):
    """
    10M samples = 80 MB.
    Multi-frequency vibration sensor (bearing/motor diagnostic).
    """
    print(f"\n[5/8] Multi-frequency vibration sensor ({n_samples//1_000_000}M samples)...")

    t = np.arange(n_samples, dtype=np.float64)
    signal = (
        1.0 * np.sin(2 * np.pi * t / 1000) +          # Fundamental frequency
        0.3 * np.sin(2 * np.pi * t / 333) +            # 3rd harmonic
        0.1 * np.sin(2 * np.pi * t / 200) +            # 5th harmonic
        0.05 * np.sin(2 * np.pi * t / 143) +           # 7th harmonic
        np.random.randn(n_samples) * 0.01              # Noise floor
    )
    # Amplitude modulation (slow bearing wear)
    modulation = 1.0 + 0.3 * np.sin(2 * np.pi * t / n_samples)
    signal *= modulation

    save_float64_bin(signal, f'synthetic_vibration_{n_samples//1_000_000}m.bin')


def generate_industrial_ramp(n_samples=10_000_000):
    """
    10M samples = 80 MB.
    Industrial temperature ramp with controlled heating/cooling cycles.
    """
    print(f"\n[6/8] Industrial temperature ramp ({n_samples//1_000_000}M samples)...")

    t = np.arange(n_samples, dtype=np.float64)
    # Sawtooth ramp: heat up linearly, then cool down
    cycle_length = 100_000  # 100K samples per cycle
    phase = (t % cycle_length) / cycle_length
    ramp = np.where(phase < 0.7,
                    phase / 0.7,          # Heating (70% of cycle)
                    (1.0 - phase) / 0.3)  # Cooling (30% of cycle)
    signal = 25.0 + 175.0 * ramp + np.random.randn(n_samples) * 0.05
    # Add thermocouple drift
    signal += 0.001 * np.cumsum(np.random.randn(n_samples)) / np.sqrt(n_samples)

    save_float64_bin(signal, f'synthetic_industrial_ramp_{n_samples//1_000_000}m.bin')


def generate_heartbeat(n_samples=10_000_000):
    """
    10M samples = 80 MB.
    Simulated ECG-like heartbeat signal with noise.
    """
    print(f"\n[7/8] Noisy heartbeat signal ({n_samples//1_000_000}M samples)...")

    t = np.arange(n_samples, dtype=np.float64) / 1000.0  # 1 kHz sampling
    # Simplified ECG: sharp QRS complex + P wave + T wave
    heart_rate = 1.2  # Hz (72 BPM)
    phase = (t * heart_rate) % 1.0
    signal = np.zeros(n_samples, dtype=np.float64)

    # P wave
    p_center, p_width = 0.15, 0.04
    signal += 0.15 * np.exp(-0.5 * ((phase - p_center) / p_width) ** 2)

    # QRS complex
    qrs_center, qrs_width = 0.35, 0.008
    signal += 1.0 * np.exp(-0.5 * ((phase - qrs_center) / qrs_width) ** 2)
    signal -= 0.2 * np.exp(-0.5 * ((phase - 0.32) / 0.01) ** 2)

    # T wave
    t_center, t_width = 0.55, 0.06
    signal += 0.3 * np.exp(-0.5 * ((phase - t_center) / t_width) ** 2)

    # Add noise
    signal += np.random.randn(n_samples) * 0.02
    # Baseline wander
    signal += 0.05 * np.sin(2 * np.pi * t * 0.15)

    save_float64_bin(signal, f'synthetic_heartbeat_{n_samples//1_000_000}m.bin')


def generate_lorenz_attractor(n_samples=5_000_000):
    """
    5M samples = 40 MB.
    Lorenz chaotic attractor — represents truly complex, non-periodic data.
    Tests DLC's ability to compress non-linear dynamics.
    """
    print(f"\n[8/8] Lorenz chaotic attractor ({n_samples//1_000_000}M samples)...")

    dt = 0.001
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    x, y, z = 1.0, 1.0, 1.0

    # Use chunks for memory efficiency
    chunk_size = 1_000_000
    all_x = []

    for start in range(0, n_samples, chunk_size):
        n = min(chunk_size, n_samples - start)
        chunk = np.empty(n, dtype=np.float64)
        for i in range(n):
            dx = sigma * (y - x) * dt
            dy = (x * (rho - z) - y) * dt
            dz = (x * y - beta * z) * dt
            x += dx
            y += dy
            z += dz
            chunk[i] = x
        all_x.append(chunk)
        pct = (start + n) / n_samples * 100
        print(f"  ... {pct:.0f}%")

    data = np.concatenate(all_x)
    save_float64_bin(data, f'synthetic_lorenz_{n_samples//1_000_000}m.bin')


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  DLC v2 -- Large Dataset Downloader & Generator")
    print("=" * 70)
    print(f"  Output directory: {SCRIPT_DIR}")
    print()

    # Check what already exists
    existing = [f for f in os.listdir(SCRIPT_DIR) if f.endswith('.bin')]
    existing_size = sum(os.path.getsize(os.path.join(SCRIPT_DIR, f))
                        for f in existing)
    print(f"  Existing datasets: {len(existing)} files, {existing_size / 1e6:.0f} MB")
    print()

    # Ask what to generate
    print("  Options:")
    print("    [1] Download real datasets only (~50 MB)")
    print("    [2] Generate synthetic large datasets only (~680 MB)")
    print("    [3] Both downloads + synthetic (~730 MB)")
    print("    [4] Quick test (3M samples only, ~24 MB)")
    print()

    choice = input("  Choose (1-4, default=3): ").strip() or "3"

    if choice in ("1", "3"):
        download_household_power()
        download_jena_climate()
        download_noaa_temperature()

    if choice in ("2", "3"):
        generate_massive_sensor(50_000_000)  # 400 MB
        generate_vibration_sensor(10_000_000)  # 80 MB
        generate_industrial_ramp(10_000_000)   # 80 MB
        generate_heartbeat(10_000_000)         # 80 MB
        generate_lorenz_attractor(5_000_000)   # 40 MB

    if choice == "4":
        generate_massive_sensor(3_000_000)         # 24 MB
        generate_vibration_sensor(1_000_000)        # 8 MB
        generate_industrial_ramp(1_000_000)         # 8 MB
        generate_heartbeat(1_000_000)               # 8 MB
        generate_lorenz_attractor(500_000)          # 4 MB

    # Final summary
    print("\n" + "=" * 70)
    all_bins = sorted([f for f in os.listdir(SCRIPT_DIR) if f.endswith('.bin')])
    total_size = 0
    for f in all_bins:
        fp = os.path.join(SCRIPT_DIR, f)
        sz = os.path.getsize(fp)
        total_size += sz
        samples = sz // 8
        print(f"  {f:<50} {samples:>12,} samples  {sz/1e6:>8.1f} MB")

    print(f"\n  Total: {len(all_bins)} datasets, {total_size / 1e6:.0f} MB")
    print(f"\n  Now run:")
    print(f"    python python/benchmarks/compare_industry.py")
    print(f"    streamlit run python/app.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
