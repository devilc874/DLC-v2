import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

# ─── Output directory ───
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Data from export.csv (compression ratios) ───
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

# Winner flags: DLC wins unless noted
# real_ec2_net_4k → ZLIB wins, real_nyc_taxi_10k → Gorilla wins
dlc_winners = set()
for name, gor, dod, rle, gz, zl, dlc in datasets_ratio:
    if name == "real_ec2_net_4k":
        continue  # ZLIB wins
    elif name == "real_nyc_taxi_10k":
        continue  # Gorilla wins
    else:
        dlc_winners.add(name)


# ═══════════════════════════════════════════════════════════════════
# FIG 4 — Compression Ratio (log scale) across all baselines
# ═══════════════════════════════════════════════════════════════════

fig4, ax4 = plt.subplots(figsize=(18, 7))

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

# Colors — curated palette
colors = {
    'DLC v2':     '#1a5fb4',
    'Gorilla':    '#e66100',
    'DoD (16-bit)': '#5e9c3b',
    'RLE':        '#9b59b6',
    'GZIP':       '#e6a817',
    'ZLIB':       '#2ec4b6',
}

bars_dlc = ax4.bar(x + offsets[0]*width, dlc_vals, width, label='DLC v2', color=colors['DLC v2'], zorder=3)
bars_gor = ax4.bar(x + offsets[1]*width, gorilla_vals, width, label='Gorilla', color=colors['Gorilla'], zorder=3)
bars_dod = ax4.bar(x + offsets[2]*width, dod16_vals, width, label='DoD (16-bit)', color=colors['DoD (16-bit)'], zorder=3)
bars_rle = ax4.bar(x + offsets[3]*width, rle_vals, width, label='RLE', color=colors['RLE'], zorder=3)
bars_gz  = ax4.bar(x + offsets[4]*width, gzip_vals, width, label='GZIP', color=colors['GZIP'], zorder=3)
bars_zl  = ax4.bar(x + offsets[5]*width, zlib_vals, width, label='ZLIB', color=colors['ZLIB'], zorder=3)

# Mark DLC winners with asterisk
for i, name in enumerate(labels):
    if name in dlc_winners:
        bar_height = dlc_vals[i]
        ax4.text(x[i] + offsets[0]*width, bar_height * 1.15, '*',
                 ha='center', va='bottom', fontsize=14, fontweight='bold', color='#1a5fb4')

# Mark non-DLC winners
for i, name in enumerate(labels):
    if name == "real_ec2_net_4k":
        ax4.text(x[i] + offsets[5]*width, zlib_vals[i] * 1.15, '★',
                 ha='center', va='bottom', fontsize=10, fontweight='bold', color=colors['ZLIB'])
    elif name == "real_nyc_taxi_10k":
        ax4.text(x[i] + offsets[1]*width, gorilla_vals[i] * 1.15, '★',
                 ha='center', va='bottom', fontsize=10, fontweight='bold', color=colors['Gorilla'])

ax4.set_yscale('log')
ax4.set_ylabel('Compression ratio (log scale)', fontsize=13, fontweight='bold')
ax4.set_title('Compression Ratio: DLC v2 vs All Baselines at Equal Precision (16-bit)',
              fontsize=15, fontweight='bold', pad=15)
ax4.set_xticks(x)
ax4.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
ax4.legend(loc='upper right', fontsize=10, framealpha=0.9, edgecolor='#cccccc')
ax4.set_axisbelow(True)
ax4.grid(axis='y', alpha=0.3, linestyle='--')
ax4.axhline(y=1.0, color='#888888', linewidth=0.8, linestyle='-', alpha=0.5)

# Add annotation for the 1.0x baseline
ax4.text(len(labels) - 0.5, 1.05, '1.0× = no compression', fontsize=8,
         color='#666666', ha='right', va='bottom', style='italic')

fig4.tight_layout()
fig4_path = os.path.join(OUT_DIR, 'fig4_compression_ratio.png')
fig4.savefig(fig4_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Fig 4 saved to: {fig4_path}")
plt.close(fig4)


# ═══════════════════════════════════════════════════════════════════
# FIG 5 — Per-Dataset Reconstruction Error vs 8×10⁻⁶ Bound
# ═══════════════════════════════════════════════════════════════════

# Data from export1.csv — DLC v2 error column
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

fig5, ax5 = plt.subplots(figsize=(16, 6.5))

err_labels = [d[0] for d in datasets_error]
err_vals   = [d[1] for d in datasets_error]

x5 = np.arange(len(err_labels))

# Color bars by error magnitude
bar_colors = []
for v in err_vals:
    if v < 1e-07:
        bar_colors.append('#27ae60')   # ultra-tight (green)
    elif v < 5e-06:
        bar_colors.append('#2980b9')   # moderate (blue)
    else:
        bar_colors.append('#3a6fb5')   # near-bound (darker blue)

bars5 = ax5.bar(x5, err_vals, 0.7, color=bar_colors, zorder=3, edgecolor='white', linewidth=0.3)

# Strict bound line
ax5.axhline(y=8e-06, color='#e74c3c', linewidth=2.0, linestyle='--', zorder=2, label='strict bound 8×10⁻⁶')

# Annotation for the bound
ax5.text(len(err_labels) - 0.5, 8e-06 * 1.8, 'strict bound 8×10⁻⁶',
         fontsize=11, color='#e74c3c', ha='right', va='bottom', fontweight='bold',
         fontstyle='italic')

ax5.set_yscale('log')
ax5.set_ylabel('max |original − reconstructed|', fontsize=13, fontweight='bold')
ax5.set_title('Per-Dataset Reconstruction Error vs Strict 8×10⁻⁶ Bound',
              fontsize=15, fontweight='bold', pad=15)
ax5.set_xticks(x5)
ax5.set_xticklabels(err_labels, rotation=45, ha='right', fontsize=9)

# Set y-axis range to show the full spread
ax5.set_ylim(1e-09, 1e-04)
ax5.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=10))

ax5.set_axisbelow(True)
ax5.grid(axis='y', alpha=0.3, linestyle='--')

# Add annotations for notable points
# test_recovered and test_rt_rec
for i, (name, val) in enumerate(datasets_error):
    if name == "test_recovered":
        ax5.annotate(f'{val:.2e}', (x5[i], val),
                     textcoords="offset points", xytext=(0, -20),
                     ha='center', fontsize=7.5, color='#27ae60', fontweight='bold')
    elif name == "test_rt_rec":
        ax5.annotate(f'{val:.2e}', (x5[i], val),
                     textcoords="offset points", xytext=(0, -20),
                     ha='center', fontsize=7.5, color='#27ae60', fontweight='bold')

# Add a legend entry for the color coding
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#3a6fb5', label='Near bound (~7.6×10⁻⁶)'),
    Patch(facecolor='#2980b9', label='Moderate (~1.9×10⁻⁶)'),
    Patch(facecolor='#27ae60', label='Ultra-tight (<3×10⁻⁸)'),
    plt.Line2D([0], [0], color='#e74c3c', linewidth=2, linestyle='--', label='Strict bound 8×10⁻⁶'),
]
ax5.legend(handles=legend_elements, loc='lower left', fontsize=9, framealpha=0.9, edgecolor='#cccccc')

fig5.tight_layout()
fig5_path = os.path.join(OUT_DIR, 'fig5_reconstruction_error.png')
fig5.savefig(fig5_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Fig 5 saved to: {fig5_path}")
plt.close(fig5)

print("\nDone! Both figures generated.")
