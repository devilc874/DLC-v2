#!/usr/bin/env python3
"""
DLC Real-World Data Visualization — Capstone Presentation Charts.

Generates two high-res images showing WHY structured real-world data
compresses dramatically better than synthetic noise:

  1. aws_cpu_overlay.png   — Original vs predicted signal + error bound
  2. aws_cpu_residuals.png — Residual histogram proving Laplacian peak

Usage:
    python python/benchmarks/visualize_real_data.py
"""

import os
import sys
import numpy as np

# Add parent to path for DLC imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..'))

from dlc.windowing import segment_block
from dlc.models import select_model, MODEL_LINEAR, MODEL_QUADRATIC, MODEL_XOR_DELTA
from dlc.models import fit_xor_delta, reconstruct_xor_delta
from dlc.encoders import quantize, dequantize

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
TEST_DATA_DIR = os.path.join(PROJECT_ROOT, 'test_data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')

DATA_FILE = os.path.join(TEST_DATA_DIR, 'real_cpu_4k.bin')
NUM_POINTS = 4032  # All points from the natural-length dataset
PRECISION_BITS = 12


def load_data():
    """Load the first NUM_POINTS from the real CPU dataset."""
    if not os.path.isfile(DATA_FILE):
        print(f"ERROR: {DATA_FILE} not found.")
        print("Run: python test_data/generate_test_data.py")
        sys.exit(1)

    raw = np.fromfile(DATA_FILE, dtype='<f8')
    data = raw[:NUM_POINTS]
    print(f"Loaded {len(data)} points from {os.path.basename(DATA_FILE)}")
    print(f"  Range: [{data.min():.2f}, {data.max():.2f}]  Mean: {data.mean():.2f}")
    return data


def run_pipeline(data):
    """
    Run the DLC pipeline on data, collecting per-window details
    for visualization.

    Also computes forced-Linear residuals for EVERY window so we
    can show the Laplacian distribution even when XOR-Delta wins.

    Returns:
        reconstructed: full reconstructed signal
        residuals_all: all quantized residuals from chosen model (ints)
        window_boundaries: list of (start, end) for each window
        model_names: list of model name strings per window
        predictions: full predicted signal
        linear_residuals: quantized residuals from forced-Linear fit (for histogram)
    """
    from dlc.models import fit_linear

    windows = segment_block(data)

    reconstructed = []
    predictions = []
    residuals_all = []
    linear_residuals = []  # forced-Linear residuals for histogram
    window_boundaries = []
    model_names = []

    offset = 0
    for window in windows:
        n = len(window)
        window_boundaries.append((offset, offset + n))

        model_id, params, residuals = select_model(window)

        # Always compute forced-Linear residuals for histogram
        try:
            _m, _c, lin_res = fit_linear(window)
            lin_q = quantize(lin_res, PRECISION_BITS)
            linear_residuals.extend(lin_q)
        except Exception:
            linear_residuals.extend([0] * n)

        # Determine model name
        if model_id == MODEL_LINEAR:
            model_names.append('Linear')
            m, c = params
            x = np.arange(n, dtype=np.float64)
            predicted = m * x + c
        elif model_id == MODEL_QUADRATIC:
            model_names.append('Quadratic')
            a, b, c = params
            x = np.arange(n, dtype=np.float64)
            predicted = a * x * x + b * x + c
        else:  # XOR-Delta
            model_names.append('XOR-Δ')
            anchor, xor_res = fit_xor_delta(window)
            rec = reconstruct_xor_delta(anchor, xor_res)
            predictions.extend(rec.tolist())
            reconstructed.extend(rec.tolist())
            residuals_all.extend([0] * n)
            offset += n
            continue

        # Quantize/dequantize residuals
        q = quantize(residuals, PRECISION_BITS)
        dq = dequantize(q, PRECISION_BITS)
        rec = predicted + dq

        predictions.extend(predicted.tolist())
        reconstructed.extend(rec.tolist())
        residuals_all.extend(q)
        offset += n

    return (
        np.array(reconstructed),
        residuals_all,
        window_boundaries,
        model_names,
        np.array(predictions),
        linear_residuals,
    )


def generate_overlay_chart(data, reconstructed, predictions,
                           window_boundaries, model_names):
    """
    Image 1: Two-panel chart.
    Top:    Original signal + prediction overlay + window split lines.
    Bottom: Reconstruction error with ±8e-6 bounds.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.style.use('dark_background')

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(18, 10), height_ratios=[3, 1],
        gridspec_kw={'hspace': 0.25},
    )
    fig.patch.set_facecolor('#0d1117')
    for ax in (ax_top, ax_bot):
        ax.set_facecolor('#161b22')

    x = np.arange(len(data))
    error = data - reconstructed

    # ── Top Panel: Signal + Prediction Overlay ───────────────────────────

    # Original signal
    ax_top.plot(x, data, color='#58a6ff', linewidth=0.6, alpha=0.85,
                label='Original Signal (AWS EC2 CPU %)')

    # Prediction overlay
    ax_top.plot(x, predictions, color='#f0883e', linewidth=0.5, alpha=0.7,
                label='DLC Prediction')

    # Window boundaries (vertical lines)
    model_colors = {
        'Linear': '#3fb950',
        'Quadratic': '#d2a8ff',
        'XOR-Δ': '#ff7b72',
    }
    for (start, end), mname in zip(window_boundaries, model_names):
        color = model_colors.get(mname, '#484f58')
        ax_top.axvline(x=start, color=color, linewidth=0.3, alpha=0.5)

    # Legend with model type markers
    custom_lines = [
        Line2D([0], [0], color='#58a6ff', lw=2),
        Line2D([0], [0], color='#f0883e', lw=2),
        Line2D([0], [0], color='#3fb950', lw=1.5, linestyle='--'),
        Line2D([0], [0], color='#d2a8ff', lw=1.5, linestyle='--'),
        Line2D([0], [0], color='#ff7b72', lw=1.5, linestyle='--'),
    ]
    model_counts = {}
    for m in model_names:
        model_counts[m] = model_counts.get(m, 0) + 1
    legend_labels = [
        'Original Signal',
        'DLC Prediction',
        f'Linear ({model_counts.get("Linear", 0)} wins)',
        f'Quadratic ({model_counts.get("Quadratic", 0)} wins)',
        f'XOR-Δ ({model_counts.get("XOR-Δ", 0)} wins)',
    ]
    ax_top.legend(custom_lines, legend_labels,
                  loc='upper right', fontsize=10, fancybox=True,
                  framealpha=0.7, edgecolor='#30363d')

    ax_top.set_ylabel('CPU Utilization (%)', fontsize=12, color='#c9d1d9')
    ax_top.set_title(
        'DLC Compression — AWS EC2 CPU Utilization (First 10K Samples)',
        fontsize=16, fontweight='bold', color='#f0f6fc', pad=15,
    )
    ax_top.tick_params(colors='#8b949e')
    ax_top.grid(True, alpha=0.15, color='#30363d')

    # Window count annotation
    ax_top.text(
        0.01, 0.95,
        f'{len(window_boundaries)} adaptive windows  |  '
        f'{len(data):,} samples  |  '
        f'Precision: {PRECISION_BITS}-bit',
        transform=ax_top.transAxes, fontsize=9, color='#8b949e',
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#21262d',
                  edgecolor='#30363d', alpha=0.8),
    )

    # ── Bottom Panel: Reconstruction Error ───────────────────────────────

    ax_bot.fill_between(x, error, 0, color='#58a6ff', alpha=0.4)
    ax_bot.plot(x, error, color='#58a6ff', linewidth=0.3, alpha=0.7)

    # Error bounds
    ax_bot.axhline(y=8e-6, color='#f85149', linewidth=1.2, linestyle='--',
                   alpha=0.9, label='Error Bound (±8×10⁻⁶)')
    ax_bot.axhline(y=-8e-6, color='#f85149', linewidth=1.2, linestyle='--',
                   alpha=0.9)
    ax_bot.axhline(y=0, color='#484f58', linewidth=0.5, alpha=0.5)

    max_err = np.max(np.abs(error))
    ax_bot.set_ylabel('Error', fontsize=12, color='#c9d1d9')
    ax_bot.set_xlabel('Sample Index', fontsize=12, color='#c9d1d9')
    ax_bot.set_title(
        f'Reconstruction Error  (max |e| = {max_err:.2e})',
        fontsize=13, color='#c9d1d9', pad=10,
    )
    ax_bot.legend(loc='upper right', fontsize=10, fancybox=True,
                  framealpha=0.7, edgecolor='#30363d')
    ax_bot.tick_params(colors='#8b949e')
    ax_bot.grid(True, alpha=0.15, color='#30363d')

    # Auto-scale Y to show the error bound clearly
    y_lim = max(max_err * 1.5, 1.2e-5)
    ax_bot.set_ylim(-y_lim, y_lim)

    # Save
    out_path = os.path.join(OUTPUT_DIR, 'aws_cpu_overlay.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f"\n  ✓ Saved: {out_path}")
    return out_path


def generate_residual_histogram(residuals_all, linear_residuals):
    """
    Image 2: Histogram of quantized residuals.

    When XOR-Delta dominates (all residuals are zero), we show the
    forced-Linear residuals instead — this demonstrates the Laplacian
    distribution that makes regression-based compression effective.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.stats import laplace, norm

    plt.style.use('dark_background')

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')

    chosen_res = np.array(residuals_all)
    lin_res = np.array(linear_residuals)

    # Decide which residuals to plot
    nonzero_chosen = chosen_res[chosen_res != 0]
    xor_pct = 100 * np.sum(chosen_res == 0) / len(chosen_res) if len(chosen_res) > 0 else 0

    if len(nonzero_chosen) < 50:
        # XOR-Delta dominates — use forced-Linear residuals for histogram
        plot_data = lin_res
        hist_label = 'Linear Model Residuals (quantized)'
        subtitle = ('XOR-Delta was selected for all windows (lossless).\n'
                    'Showing forced-Linear residuals to demonstrate Laplacian distribution.')
    else:
        plot_data = nonzero_chosen
        hist_label = f'Quantized Residuals (n={len(plot_data):,})'
        subtitle = ''

    # Remove exact zeros from Linear residuals for cleaner histogram
    plot_data_nz = plot_data[plot_data != 0]
    if len(plot_data_nz) < 20:
        plot_data_nz = plot_data  # fallback to include zeros

    # Compute stats
    mean_r = np.mean(plot_data_nz)
    std_r = np.std(plot_data_nz)
    median_r = np.median(plot_data_nz)
    p99 = np.percentile(np.abs(plot_data_nz), 99) if len(plot_data_nz) > 0 else 0

    # Histogram
    n_bins = min(200, max(50, int(np.sqrt(len(plot_data_nz)))))
    clip = max(p99 * 2, 5)
    clipped = plot_data_nz[(plot_data_nz >= -clip) & (plot_data_nz <= clip)]
    if len(clipped) < 20:
        clipped = plot_data_nz

    counts, bin_edges, patches = ax.hist(
        clipped, bins=n_bins, density=True,
        color='#58a6ff', alpha=0.7, edgecolor='#1f6feb', linewidth=0.3,
        label=hist_label,
    )

    # Fit Laplace distribution overlay
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    try:
        loc_l, scale_l = laplace.fit(clipped)
        laplace_pdf = laplace.pdf(bin_centers, loc=loc_l, scale=scale_l)
        ax.plot(bin_centers, laplace_pdf, color='#f0883e', linewidth=2.5,
                alpha=0.9, label=f'Laplace fit (μ={loc_l:.1f}, b={scale_l:.1f})')
    except Exception:
        pass

    # Fit Gaussian distribution overlay (for comparison)
    try:
        loc_n, scale_n = norm.fit(clipped)
        norm_pdf = norm.pdf(bin_centers, loc=loc_n, scale=scale_n)
        ax.plot(bin_centers, norm_pdf, color='#3fb950', linewidth=2.0,
                alpha=0.7, linestyle='--',
                label=f'Gaussian fit (μ={loc_n:.1f}, σ={scale_n:.1f})')
    except Exception:
        pass

    # Zero line
    ax.axvline(x=0, color='#f85149', linewidth=1.0, alpha=0.6,
               linestyle=':', label='Zero')

    # Stats annotation
    stats_text = (
        f'Statistics:\n'
        f'  Mean:   {mean_r:>10.2f}\n'
        f'  Median: {median_r:>10.2f}\n'
        f'  Std:    {std_r:>10.2f}\n'
        f'  P99:    {p99:>10.2f}\n'
        f'  XOR-Δ:  {xor_pct:.0f}% of windows'
    )
    ax.text(
        0.97, 0.95, stats_text,
        transform=ax.transAxes, fontsize=10, color='#c9d1d9',
        verticalalignment='top', horizontalalignment='right',
        fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#21262d',
                  edgecolor='#30363d', alpha=0.9),
    )

    # Compression insight annotation
    insight_text = (
        '↑ Sharp peak at zero = most residuals are tiny\n'
        '→ Fewer bits needed per sample → high compression'
    )
    ax.text(
        0.03, 0.95, insight_text,
        transform=ax.transAxes, fontsize=11, color='#f0883e',
        verticalalignment='top', fontfamily='sans-serif', fontstyle='italic',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#21262d',
                  edgecolor='#f0883e', alpha=0.8),
    )

    # Subtitle if XOR-Delta dominated
    if subtitle:
        ax.text(
            0.5, -0.10, subtitle,
            transform=ax.transAxes, fontsize=9, color='#8b949e',
            ha='center', fontstyle='italic',
        )

    ax.set_xlabel('Quantized Residual Value', fontsize=13, color='#c9d1d9')
    ax.set_ylabel('Probability Density', fontsize=13, color='#c9d1d9')
    ax.set_title(
        'DLC Residual Distribution — Why Real-World Data Compresses Well',
        fontsize=16, fontweight='bold', color='#f0f6fc', pad=15,
    )
    ax.legend(loc='upper left', fontsize=10, fancybox=True,
              framealpha=0.7, edgecolor='#30363d')
    ax.tick_params(colors='#8b949e')
    ax.grid(True, alpha=0.12, color='#30363d')

    # Save
    out_path = os.path.join(OUTPUT_DIR, 'aws_cpu_residuals.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f"  ✓ Saved: {out_path}")
    return out_path


def main():
    print("=" * 60)
    print("DLC Real-World Data Visualization")
    print("=" * 60)

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    data = load_data()

    # Run pipeline
    print("\nRunning DLC pipeline...")
    reconstructed, residuals_all, boundaries, model_names, predictions, \
        linear_residuals = run_pipeline(data)

    max_err = np.max(np.abs(data - reconstructed))
    print(f"  Windows: {len(boundaries)}")
    print(f"  Max error: {max_err:.2e}")
    print(f"  Models used: ", end='')
    model_counts = {}
    for m in model_names:
        model_counts[m] = model_counts.get(m, 0) + 1
    print(', '.join(f'{k}={v}' for k, v in model_counts.items()))

    nonzero_res = [r for r in residuals_all if r != 0]
    zero_pct = 100 * (1 - len(nonzero_res) / len(residuals_all)) if residuals_all else 0
    print(f"  Zero residuals: {zero_pct:.1f}%")
    nonzero_lin = [r for r in linear_residuals if r != 0]
    print(f"  Linear residuals (forced): {len(nonzero_lin)} non-zero")

    # Generate charts
    print("\nGenerating charts...")
    p1 = generate_overlay_chart(data, reconstructed, predictions,
                                boundaries, model_names)
    p2 = generate_residual_histogram(residuals_all, linear_residuals)

    print(f"\n{'=' * 60}")
    print(f"Done! Charts saved to: {OUTPUT_DIR}/")
    print(f"  1. {os.path.basename(p1)}")
    print(f"  2. {os.path.basename(p2)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
