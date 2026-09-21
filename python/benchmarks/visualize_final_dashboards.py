#!/usr/bin/env python3
"""
Final Presentation Dashboards — DYNAMIC.

Runs GZIP, ZLIB, and DLC C++ on every .bin dataset, then generates:
  1. final_ratios.png    — Grouped bar chart of actual compression ratios
  2. final_throughput.png — Throughput scaling across 1/2/4/8 workers (DLC C++)

All numbers are measured live — nothing is hardcoded.

Usage:
    python python/benchmarks/visualize_final_dashboards.py
"""

import os
import sys
import time
import gzip
import zlib
import tempfile
import subprocess
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
TEST_DATA_DIR = os.path.join(PROJECT_ROOT, 'test_data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')


# ── Helpers ──────────────────────────────────────────────────────────────────

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
    elapsed = time.perf_counter() - t0
    return len(out), elapsed


def compress_zlib(raw_bytes):
    t0 = time.perf_counter()
    out = zlib.compress(raw_bytes, level=6)
    elapsed = time.perf_counter() - t0
    return len(out), elapsed


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
    except Exception:
        return None, 0
    finally:
        if os.path.isfile(dlc_path):
            os.unlink(dlc_path)


def discover_datasets():
    """Find all .bin files in test_data, return sorted list of (name, path, size)."""
    datasets = []
    if not os.path.isdir(TEST_DATA_DIR):
        return datasets
    for f in sorted(os.listdir(TEST_DATA_DIR)):
        if f.endswith('.bin'):
            path = os.path.join(TEST_DATA_DIR, f)
            size = os.path.getsize(path)
            if size >= 800:  # skip tiny edge-case files
                datasets.append((f[:-4], path, size))
    return datasets


# ── Chart 1: Compression Ratios ─────────────────────────────────────────────

def generate_ratios_chart(datasets, binary):
    """Run GZIP, ZLIB, DLC on all datasets and plot grouped bars."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.style.use('dark_background')

    names = []
    gzip_ratios = []
    zlib_ratios = []
    dlc_ratios = []

    print("\n  Benchmarking compression ratios...")
    for name, path, raw_size in datasets:
        raw_bytes = open(path, 'rb').read()
        samples = raw_size // 8

        # GZIP
        gz_size, _ = compress_gzip(raw_bytes)
        gz_ratio = raw_size / gz_size if gz_size > 0 else 1.0

        # ZLIB
        zl_size, _ = compress_zlib(raw_bytes)
        zl_ratio = raw_size / zl_size if zl_size > 0 else 1.0

        # DLC
        if binary:
            dlc_size, _ = compress_dlc(binary, path, workers=4)
            dlc_ratio = raw_size / dlc_size if dlc_size and dlc_size > 0 else 1.0
        else:
            dlc_ratio = 0

        # Format display name
        if samples >= 1_000_000:
            label = f"{name}\n({samples // 1_000_000}M pts)"
        elif samples >= 1_000:
            label = f"{name}\n({samples // 1_000}K pts)"
        else:
            label = f"{name}\n({samples} pts)"

        names.append(label)
        gzip_ratios.append(gz_ratio)
        zlib_ratios.append(zl_ratio)
        dlc_ratios.append(dlc_ratio)

        print(f"    {name:<35}  GZIP={gz_ratio:>6.1f}x  "
              f"ZLIB={zl_ratio:>6.1f}x  DLC={dlc_ratio:>6.1f}x")

    # Plot
    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(14, len(names) * 1.8), 8))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')

    bars1 = ax.bar(x - width, gzip_ratios, width, label='GZIP (level 6)',
                   color='#484f58', edgecolor='#6e7681', linewidth=0.8, zorder=3)
    bars2 = ax.bar(x, zlib_ratios, width, label='ZLIB (level 6)',
                   color='#3fb950', edgecolor='#56d364', linewidth=0.8, zorder=3)
    bars3 = ax.bar(x + width, dlc_ratios, width, label='DLC v2 (C++)',
                   color='#58a6ff', edgecolor='#79c0ff', linewidth=0.8, zorder=3)

    # Value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                # Position label above bar
                y_pos = h * 1.08 if ax.get_yscale() == 'log' else h + 0.1
                ax.text(bar.get_x() + bar.get_width() / 2., y_pos,
                        f'{h:.1f}x', ha='center', va='bottom',
                        fontsize=8, fontweight='bold', color='#f0f6fc')

    # Use log scale if range is large
    max_ratio = max(max(gzip_ratios), max(zlib_ratios), max(dlc_ratios))
    min_ratio = min(min(gzip_ratios), min(zlib_ratios), min(dlc_ratios))
    if max_ratio / max(min_ratio, 0.1) > 10:
        ax.set_yscale('log')
        ax.set_ylim(0.8, max_ratio * 2)

    ax.set_ylabel('Compression Ratio (higher = better)', fontsize=13, color='#c9d1d9')
    ax.set_xlabel('Dataset', fontsize=13, color='#c9d1d9')
    ax.set_title('Compression Ratio — GZIP vs ZLIB vs DLC v2 (Live Benchmark)',
                 fontsize=16, fontweight='bold', color='#f0f6fc', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7, color='#c9d1d9', rotation=30, ha='right')
    ax.tick_params(colors='#8b949e')
    ax.legend(fontsize=11, loc='upper left', fancybox=True,
              framealpha=0.7, edgecolor='#30363d')
    ax.grid(True, alpha=0.15, color='#30363d', axis='y')
    ax.set_axisbelow(True)

    # Highlight best DLC result
    best_idx = int(np.argmax(dlc_ratios))
    best_val = dlc_ratios[best_idx]
    if best_val > 2:
        ax.annotate(f'{best_val:.1f}x — Best DLC',
                    xy=(best_idx + width, best_val),
                    xytext=(best_idx + width + 0.5,
                            best_val * 0.7 if ax.get_yscale() == 'log' else best_val - 1),
                    fontsize=10, color='#f0883e', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#f0883e', lw=1.5),
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#21262d',
                              edgecolor='#f0883e', alpha=0.9))

    out_path = os.path.join(OUTPUT_DIR, 'final_ratios.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f"  -> Saved: {out_path}")
    return out_path


# ── Chart 2: Throughput Scaling ──────────────────────────────────────────────

def generate_throughput_chart(binary):
    """
    Measure DLC C++ throughput at 1, 2, 4, 8 workers on the largest dataset.
    Also measure Python gzip for comparison.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.style.use('dark_background')

    # Pick the largest dataset for throughput testing
    datasets = discover_datasets()
    if not datasets:
        print("  ERROR: No datasets found for throughput test!")
        return None

    largest = max(datasets, key=lambda d: d[2])
    name, path, raw_size = largest
    raw_bytes = open(path, 'rb').read()

    print(f"\n  Throughput test on: {name} ({raw_size / 1e6:.1f} MB)")

    worker_counts = [1, 2, 4, 8]

    # Measure gzip (single-threaded baseline — same for all worker counts)
    _, gz_time = compress_gzip(raw_bytes)
    gz_mbps = (raw_size / 1e6) / gz_time if gz_time > 0 else 0
    gzip_throughputs = [gz_mbps] * len(worker_counts)  # flat line

    # Measure DLC at each worker count
    dlc_throughputs = []
    if binary:
        for w in worker_counts:
            dlc_size, dlc_time = compress_dlc(binary, path, workers=w)
            if dlc_size and dlc_time > 0:
                mbps = (raw_size / 1e6) / dlc_time
            else:
                mbps = 0
            dlc_throughputs.append(mbps)
            print(f"    Workers={w}: DLC={mbps:.1f} MB/s")
    else:
        dlc_throughputs = [0] * len(worker_counts)

    print(f"    GZIP baseline: {gz_mbps:.1f} MB/s (single-threaded)")

    # Plot
    x = np.arange(len(worker_counts))
    labels = [f'{w} Worker{"s" if w > 1 else ""}' for w in worker_counts]
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')

    bars_gz = ax.bar(x - width/2, gzip_throughputs, width, label='GZIP (Python)',
                     color='#f0883e', edgecolor='#f5a623', linewidth=0.8, zorder=3)
    bars_dlc = ax.bar(x + width/2, dlc_throughputs, width, label='DLC v2 (C++)',
                      color='#58a6ff', edgecolor='#79c0ff', linewidth=0.8, zorder=3)

    # Value labels
    for bar in bars_gz:
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                f'{bar.get_height():.1f}', ha='center', va='bottom',
                fontsize=11, fontweight='bold', color='#f0883e')
    for bar in bars_dlc:
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                f'{bar.get_height():.1f}', ha='center', va='bottom',
                fontsize=11, fontweight='bold', color='#58a6ff')

    ax.set_ylabel('Throughput (MB/s)', fontsize=13, color='#c9d1d9')
    ax.set_xlabel('Thread Count', fontsize=13, color='#c9d1d9')
    ax.set_title(f'Throughput Scaling — GZIP vs DLC C++ ({name})',
                 fontsize=16, fontweight='bold', color='#f0f6fc', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12, color='#c9d1d9')
    ax.tick_params(colors='#8b949e')
    ax.legend(fontsize=13, loc='upper left', fancybox=True,
              framealpha=0.7, edgecolor='#30363d')
    ax.grid(True, alpha=0.15, color='#30363d', axis='y')
    ax.set_axisbelow(True)

    # Speedup annotation
    peak_dlc = max(dlc_throughputs) if dlc_throughputs else 0
    if peak_dlc > 0 and gz_mbps > 0:
        speedup = peak_dlc / gz_mbps
        peak_idx = dlc_throughputs.index(peak_dlc)
        ax.annotate(f'{speedup:.0f}x faster\nthan GZIP',
                    xy=(peak_idx + width/2, peak_dlc),
                    xytext=(peak_idx + 0.8, peak_dlc * 0.75),
                    fontsize=12, color='#3fb950', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#3fb950', lw=1.5),
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#21262d',
                              edgecolor='#3fb950', alpha=0.9))

    # GIL annotation on gzip flat line
    ax.annotate('GZIP: single-threaded\n(no scaling)',
                xy=(2 - width/2, gz_mbps),
                xytext=(0.3, max(dlc_throughputs) * 0.4 if dlc_throughputs else 20),
                fontsize=11, color='#f85149', fontstyle='italic',
                arrowprops=dict(arrowstyle='->', color='#f85149', lw=1.2),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#21262d',
                          edgecolor='#f85149', alpha=0.9))

    out_path = os.path.join(OUTPUT_DIR, 'final_throughput.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f"  -> Saved: {out_path}")
    return out_path


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    binary = find_cpp_binary()
    datasets = discover_datasets()

    print("=" * 60)
    print("Final Presentation Dashboards (DYNAMIC)")
    print("=" * 60)
    print(f"  C++ Binary: {binary or 'NOT FOUND'}")
    print(f"  Datasets:   {len(datasets)} found")

    p1 = generate_ratios_chart(datasets, binary)
    p2 = generate_throughput_chart(binary)

    print(f"\n{'=' * 60}")
    print(f"Done! Charts saved to: {OUTPUT_DIR}")
    if p1:
        print(f"  1. {os.path.basename(p1)}")
    if p2:
        print(f"  2. {os.path.basename(p2)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
