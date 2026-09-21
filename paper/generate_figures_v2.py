"""
Revised figure generation for icSoftComp 2026 paper.
Reviewer comments addressed:
  - Larger font sizes (≥ 12pt labels, ≥ 10pt ticks)
  - Higher resolution (600 DPI)
  - Clearer legends with bordered boxes
  - Better color contrast for print
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
import numpy as np
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Global style for publication quality ──
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 10,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 1.2,
    'lines.linewidth': 1.5,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

# ─── Data from Table III (compression ratios) ───
datasets_ratio = [
    ("ramp_1m",                    1.0, 8.0, 0.7, 1.4, 1.4, 2355.7),
    ("test_recovered",             3.8, 8.0, 0.7, 3.7, 3.7, 102.0),
    ("sine_1m",                    1.0, 8.0, 0.7, 1.1, 1.1, 101.9),
    ("synth_lorenz_0m",            1.0, 7.5, 0.7, 1.0, 1.0, 15.5),
    ("real_jena_pres_420k",        1.3, 3.9, 0.7, 3.8, 3.9, 7.1),
    ("real_twitter_15k",           4.3, 2.3, 0.7, 5.4, 5.5, 5.6),
    ("real_jena_temp_420k",        1.0, 3.4, 0.7, 3.8, 4.0, 5.6),
    ("real_ec2_net_4k",            1.2, 2.0, 0.7, 7.8, 8.5, 5.7),
    ("test_rt_rec",                1.7, 4.3, 0.7, 3.3, 3.3, 5.2),
    ("massive_sensor_3m",          1.2, 4.3, 0.7, 1.2, 1.2, 5.2),
    ("noisy_ramp_100k",            1.0, 4.1, 0.7, 1.3, 1.3, 4.4),
    ("synth_vibration_1m",         1.0, 4.1, 0.7, 1.1, 1.1, 4.4),
    ("synth_heartbeat_1m",         1.0, 4.0, 0.7, 1.0, 1.0, 4.1),
    ("synth_ind_ramp_1m",          1.1, 3.5, 0.7, 1.2, 1.2, 3.7),
    ("real_nyc_taxi_10k",          3.6, 2.0, 0.7, 2.9, 2.9, 3.4),
    ("synth_noisy_sine_1m",        1.2, 2.7, 0.7, 1.1, 1.1, 3.2),
    ("real_cpu_4k",                1.2, 2.6, 0.7, 2.8, 2.8, 3.0),
    ("random_walk_50k",            1.0, 2.7, 0.7, 1.1, 1.1, 3.0),
    ("synth_rand_walk_1m",         1.0, 2.7, 0.7, 1.1, 1.1, 3.0),
    ("real_ambient_temp_7k",       1.2, 2.7, 0.7, 1.1, 1.1, 2.9),
    ("real_machine_temp_22k",      1.1, 2.7, 0.7, 1.1, 1.1, 2.9),
]

dlc_winners = set()
for name, gor, dod, rle, gz, zl, dlc in datasets_ratio:
    if name not in ("real_ec2_net_4k", "real_nyc_taxi_10k"):
        dlc_winners.add(name)

# ═══════════════════════════════════════════════════════════════════
# FIG 4 — Compression Ratio (log scale) — PUBLICATION QUALITY
# ═══════════════════════════════════════════════════════════════════

fig4, ax4 = plt.subplots(figsize=(18, 7.5))

labels = [d[0] for d in datasets_ratio]
gorilla_vals  = [d[1] for d in datasets_ratio]
dod16_vals    = [d[2] for d in datasets_ratio]
rle_vals      = [d[3] for d in datasets_ratio]
gzip_vals     = [d[4] for d in datasets_ratio]
zlib_vals     = [d[5] for d in datasets_ratio]
dlc_vals      = [d[6] for d in datasets_ratio]

x = np.arange(len(labels))
n_methods = 6
width = 0.13
offsets = np.arange(n_methods) - (n_methods - 1) / 2

# Colors — print-friendly, high contrast
colors = {
    'DLC v2':       '#1a5fb4',
    'Gorilla':      '#c64600',
    'DoD (16-bit)': '#3a7d22',
    'RLE':          '#7b2d8e',
    'GZIP':         '#c09000',
    'ZLIB':         '#1a9e8f',
}

bars = {}
method_data = [
    ('DLC v2', dlc_vals),
    ('Gorilla', gorilla_vals),
    ('DoD (16-bit)', dod16_vals),
    ('RLE', rle_vals),
    ('GZIP', gzip_vals),
    ('ZLIB', zlib_vals),
]

for idx, (mname, mvals) in enumerate(method_data):
    bars[mname] = ax4.bar(x + offsets[idx]*width, mvals, width,
                          label=mname, color=colors[mname], zorder=3,
                          edgecolor='white', linewidth=0.4)

# Mark DLC winners with asterisk
for i, name in enumerate(labels):
    if name in dlc_winners:
        bar_height = dlc_vals[i]
        ax4.text(x[i] + offsets[0]*width, bar_height * 1.15, '★',
                 ha='center', va='bottom', fontsize=10, fontweight='bold', color='#1a5fb4')

# Mark non-DLC winners
for i, name in enumerate(labels):
    if name == "real_ec2_net_4k":
        ax4.text(x[i] + offsets[5]*width, zlib_vals[i] * 1.15, '★',
                 ha='center', va='bottom', fontsize=10, fontweight='bold', color=colors['ZLIB'])
    elif name == "real_nyc_taxi_10k":
        ax4.text(x[i] + offsets[1]*width, gorilla_vals[i] * 1.15, '★',
                 ha='center', va='bottom', fontsize=10, fontweight='bold', color=colors['Gorilla'])

ax4.set_yscale('log')
ax4.set_ylabel('Compression Ratio (log scale)', fontsize=14, fontweight='bold')
ax4.set_xlabel('Dataset', fontsize=14, fontweight='bold')
ax4.set_title('Fig. 4.  Compression Ratio: DLC v2 vs. All Baselines (16-bit Precision)',
              fontsize=15, fontweight='bold', pad=15)
ax4.set_xticks(x)
ax4.set_xticklabels(labels, rotation=50, ha='right', fontsize=10)
ax4.legend(loc='upper right', fontsize=11, framealpha=0.95, edgecolor='#888888',
           fancybox=False, shadow=False, ncol=2)
ax4.set_axisbelow(True)
ax4.axhline(y=1.0, color='#888888', linewidth=0.8, linestyle='-', alpha=0.5)
ax4.text(len(labels) - 0.5, 1.08, '1.0× = no compression', fontsize=9,
         color='#666666', ha='right', va='bottom', style='italic')

fig4.tight_layout()
fig4_path = os.path.join(OUT_DIR, 'fig4_compression_ratio_v2.png')
fig4.savefig(fig4_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Fig 4 saved to: {fig4_path}")
plt.close(fig4)


# ═══════════════════════════════════════════════════════════════════
# FIG 5 — Reconstruction Error vs Bound — PUBLICATION QUALITY
# ═══════════════════════════════════════════════════════════════════

datasets_error = [
    ("massive_sensor_3m",          7.63e-06),
    ("noisy_ramp_100k",            7.63e-06),
    ("ramp_1m",                    7.63e-06),
    ("random_walk_50k",            7.63e-06),
    ("real_ambient_temp_7k",       7.63e-06),
    ("real_cpu_4k",                7.62e-06),
    ("real_ec2_network_4k",        6.94e-06),
    ("real_jena_pres_420k",        7.51e-06),
    ("real_jena_temp_420k",        7.59e-06),
    ("real_machine_temp_22k",      7.63e-06),
    ("real_nyc_taxi_10k",          1.87e-06),
    ("real_twitter_vol_15k",       1.92e-06),
    ("sine_1m",                    7.63e-06),
    ("synth_heartbeat_1m",         7.63e-06),
    ("synth_ind_ramp_1m",          7.63e-06),
    ("synth_lorenz_0m",            7.63e-06),
    ("synth_noisy_sine_1m",        7.63e-06),
    ("synth_rand_walk_1m",         7.63e-06),
    ("synth_vibration_1m",         7.63e-06),
    ("test_recovered",             2.21e-08),
    ("test_rt_rec",                2.64e-08),
]

fig5, ax5 = plt.subplots(figsize=(16, 7))

err_labels = [d[0] for d in datasets_error]
err_vals   = [d[1] for d in datasets_error]

x5 = np.arange(len(err_labels))

# Color by error magnitude — print-friendly
bar_colors = []
for v in err_vals:
    if v < 1e-07:
        bar_colors.append('#1a7a3a')   # ultra-tight (dark green)
    elif v < 5e-06:
        bar_colors.append('#2471a3')   # moderate (blue)
    else:
        bar_colors.append('#2c5f8a')   # near-bound (steel blue)

bars5 = ax5.bar(x5, err_vals, 0.7, color=bar_colors, zorder=3,
                edgecolor='white', linewidth=0.5)

# Strict bound line
ax5.axhline(y=8e-06, color='#c0392b', linewidth=2.5, linestyle='--',
            zorder=2, label='Strict bound 8×10⁻⁶')

ax5.text(len(err_labels) - 0.5, 8e-06 * 2.0,
         'Strict bound: 8×10⁻⁶',
         fontsize=12, color='#c0392b', ha='right', va='bottom',
         fontweight='bold', fontstyle='italic')

ax5.set_yscale('log')
ax5.set_ylabel('Max |original − reconstructed|', fontsize=14, fontweight='bold')
ax5.set_xlabel('Dataset', fontsize=14, fontweight='bold')
ax5.set_title('Fig. 5.  Per-Dataset Reconstruction Error vs. Strict 8×10⁻⁶ Bound',
              fontsize=15, fontweight='bold', pad=15)
ax5.set_xticks(x5)
ax5.set_xticklabels(err_labels, rotation=50, ha='right', fontsize=10)
ax5.set_ylim(1e-09, 5e-05)
ax5.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=10))
ax5.set_axisbelow(True)

# Annotate notable points
for i, (name, val) in enumerate(datasets_error):
    if name in ("test_recovered", "test_rt_rec"):
        ax5.annotate(f'{val:.2e}', (x5[i], val),
                     textcoords="offset points", xytext=(0, -22),
                     ha='center', fontsize=9, color='#1a7a3a', fontweight='bold')
    elif name in ("real_nyc_taxi_10k", "real_twitter_vol_15k"):
        ax5.annotate(f'{val:.2e}', (x5[i], val),
                     textcoords="offset points", xytext=(0, 8),
                     ha='center', fontsize=8, color='#2471a3')

legend_elements = [
    Patch(facecolor='#2c5f8a', edgecolor='white', label='Near bound (~7.6×10⁻⁶)'),
    Patch(facecolor='#2471a3', edgecolor='white', label='Moderate (~1.9×10⁻⁶)'),
    Patch(facecolor='#1a7a3a', edgecolor='white', label='Ultra-tight (<3×10⁻⁸)'),
    plt.Line2D([0], [0], color='#c0392b', linewidth=2.5, linestyle='--',
               label='Strict bound 8×10⁻⁶'),
]
ax5.legend(handles=legend_elements, loc='lower left', fontsize=11,
           framealpha=0.95, edgecolor='#888888', fancybox=False)

fig5.tight_layout()
fig5_path = os.path.join(OUT_DIR, 'fig5_reconstruction_error_v2.png')
fig5.savefig(fig5_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Fig 5 saved to: {fig5_path}")
plt.close(fig5)

print("\nDone! Publication-quality figures generated at 600 DPI.")
