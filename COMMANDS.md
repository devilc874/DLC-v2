# DLC v2 — All Commands (Start to End)

Run every command from the project root unless noted:

```bash
cd /Users/admin/Desktop/DLC-v2
```

---

## 0. Prerequisites (once per machine)

- Python 3.10+
- CMake 3.16+
- A C++17 compiler (`clang++` / `g++`)
- `zlib` development library (usually already on macOS)

Check:

```bash
python3 --version
cmake --version
c++ --version
```

---

## 1. Install Python packages

```bash
cd /Users/admin/Desktop/DLC-v2

# Core (dashboard, tests, plots)
pip install -r python/requirements.txt

# Optional: official SZ3 / ZFP libraries
# (not required — built-in fallbacks always run)
pip install pysz zfpy
```

---

## 2. Build the C++ engine (real compressor)

```bash
cd /Users/admin/Desktop/DLC-v2

cmake -S cpp -B cpp/build
cmake --build cpp/build -j4
```

Confirm the binary exists:

```bash
ls -la cpp/build/dlc
./cpp/build/dlc
```

You should see usage text including `--precision` and `--ablation-precision`.

Rebuild later anytime with:

```bash
cmake --build cpp/build -j4
```

---

## 3. Generate test datasets

```bash
cd /Users/admin/Desktop/DLC-v2
python test_data/generate_test_data.py
```

Confirm:

```bash
ls test_data/*.bin | head
```

Optional larger downloads (if scripts exist / network available):

```bash
python test_data/download_large_datasets.py
python test_data/download_massive_data.py
```

---

## 4. Run unit tests (optional but recommended)

```bash
cd /Users/admin/Desktop/DLC-v2

# Python tests
PYTHONPATH=python:python/benchmarks python -m pytest python/tests -q

# C++ tests
./cpp/build/dlc_tests
```

---

## 5. Compress / decompress with C++ (CLI)

### Default mode (16-bit floor + adaptive up to `--precision`)

```bash
./cpp/build/dlc compress \
  -i test_data/sine_1m.bin \
  -o output/sine_1m.dlc \
  --precision 16 \
  --workers 4
```

### Ablation mode (full adaptive 8–20 range)

```bash
./cpp/build/dlc compress \
  -i test_data/sine_1m.bin \
  -o output/sine_1m_ablation.dlc \
  --precision 16 \
  --ablation-precision \
  --workers 4
```

### Decompress

```bash
mkdir -p output
./cpp/build/dlc decompress \
  -i output/sine_1m.dlc \
  -o output/sine_1m_recovered.bin
```

### Other useful flags

| Flag | Meaning |
|------|---------|
| `--precision N` | Cap bits (8–20). Default **16**. |
| `--ablation-precision` | Turn **off** the 16-bit floor → allow 8–20 adaptation |
| `--workers N` | Thread count |
| `--chunk-size N` | Block size (default 100000) |

---

## 6. Compress / decompress with Python CLI

```bash
cd /Users/admin/Desktop/DLC-v2
mkdir -p output

# Default (16-bit floor)
PYTHONPATH=python python -m dlc.cli compress \
  -i test_data/sine_1m.bin \
  -o output/sine_1m_py.dlc \
  --precision 16

# Ablation (full 8–20)
PYTHONPATH=python python -m dlc.cli compress \
  -i test_data/sine_1m.bin \
  -o output/sine_1m_py_ablation.dlc \
  --precision 16 \
  --ablation-precision

# Decompress
PYTHONPATH=python python -m dlc.cli decompress \
  -i output/sine_1m_py.dlc \
  -o output/sine_1m_py_recovered.bin
```

---

## 7. Start the Streamlit UI (main dashboard)

```bash
cd /Users/admin/Desktop/DLC-v2
streamlit run python/app.py
```

Browser usually opens at: `http://localhost:8501`

### What to click in the UI

1. Sidebar → pick a dataset  
2. Confirm sidebar shows **C++ engine found**  
3. Tab **Industry Comparison** → **Run Industry Comparison**  
   - Compares: Gorilla, DoD, RLE, GZIP, ZLIB, **SZ3, ZFP, Elf, ALP**, DLC  
4. Tab **Batch All Datasets** → **Run Full Batch Analysis**  
   - Full table across every `.bin` (export from the table if needed)  
5. Tab **Round-Trip Verification** → verify DLC error  

Stop the server: `Ctrl+C` in that terminal.

Restart after code changes:

```bash
streamlit run python/app.py
```

---

## 8. Terminal benchmarks / paper CSVs

Always from project root.

### Fast smoke (1 dataset)

```bash
PYTHONPATH=python:python/benchmarks python python/benchmarks/compare_sota.py --limit 1
PYTHONPATH=python:python/benchmarks python python/benchmarks/ablation_precision.py --limit 1
```

### Full SOTA comparison (DLC vs SZ3 / ZFP / Elf / ALP)

```bash
PYTHONPATH=python:python/benchmarks python python/benchmarks/compare_sota.py
```

Output: `paper/sota_comparison_results.csv`

### Fair industry comparison (Gorilla / DoD-16 / RLE / GZIP / ZLIB / DLC)

```bash
PYTHONPATH=python python python/benchmarks/compare_industry_fair.py
```

### Older industry comparison

```bash
PYTHONPATH=python python python/benchmarks/compare_industry.py
```

### Precision ablation sweep (default floor vs full 8–20)

```bash
PYTHONPATH=python:python/benchmarks python python/benchmarks/ablation_precision.py
```

Output: `paper/precision_ablation_results.csv`

### Throughput

```bash
PYTHONPATH=python python python/benchmarks/bench_throughput.py
PYTHONPATH=python python python/benchmarks/bench_cpp.py
```

### Other utility scripts

```bash
PYTHONPATH=python python python/benchmarks/compare_all.py
PYTHONPATH=python python python/benchmarks/visualize_real_data.py
PYTHONPATH=python python python/benchmarks/visualize_final_dashboards.py
```

---

## 9. Paper figure generation (if needed)

```bash
cd /Users/admin/Desktop/DLC-v2
python paper/generate_figures.py
# or
python paper/generate_figures_v2.py

python paper/benchmark_full_metrics.py
```

Compile LaTeX (if TeX is installed):

```bash
cd paper/latex
pdflatex DLC_Paper_Revised.tex
bibtex DLC_Paper_Revised
pdflatex DLC_Paper_Revised.tex
pdflatex DLC_Paper_Revised.tex
```

---

## 10. Typical “first day” full path (copy-paste)

```bash
cd /Users/admin/Desktop/DLC-v2

pip install -r python/requirements.txt
pip install pysz zfpy   # optional

cmake -S cpp -B cpp/build
cmake --build cpp/build -j4

python test_data/generate_test_data.py

PYTHONPATH=python:python/benchmarks python -m pytest python/tests -q

streamlit run python/app.py
```

In another terminal (while Streamlit runs), generate paper CSVs:

```bash
cd /Users/admin/Desktop/DLC-v2
PYTHONPATH=python:python/benchmarks python python/benchmarks/compare_sota.py
PYTHONPATH=python:python/benchmarks python python/benchmarks/ablation_precision.py
PYTHONPATH=python python python/benchmarks/compare_industry_fair.py
```

---

## 11. Precision modes cheat sheet

| Mode | How | What it does |
|------|-----|----------------|
| **Default** | omit flag | 16-bit floor + adaptive up to `--precision` |
| **Ablation** | add `--ablation-precision` | Floor off → adaptive **8–20** |

Examples:

```bash
# Safe / paper-default behaviour
./cpp/build/dlc compress -i test_data/sine_1m.bin -o out.dlc --precision 16

# Reviewer ablation demo
./cpp/build/dlc compress -i test_data/sine_1m.bin -o out_abl.dlc --precision 12 --ablation-precision
```

---

## 12. Troubleshooting

| Problem | Fix |
|---------|-----|
| `C++ binary not found` in Streamlit | `cmake --build cpp/build -j4` |
| No datasets in sidebar | `python test_data/generate_test_data.py` |
| `ModuleNotFoundError: dlc` | prefix with `PYTHONPATH=python` |
| `ModuleNotFoundError: baselines` | use `PYTHONPATH=python:python/benchmarks` |
| Streamlit stale after code change | stop (`Ctrl+C`) and rerun `streamlit run python/app.py` |
| SZ3/ZFP want official libs | `pip install pysz zfpy` (fallback already works without them) |
| Permission / sandbox install issues | install packages in your normal terminal / conda env |

---

## 13. What each major piece is

| Piece | Role |
|-------|------|
| `cpp/build/dlc` | Real fast compressor |
| `python/dlc/` | Same algorithm (demo / tests / scripts) |
| `python/app.py` | Streamlit UI (charts + batch tables) |
| `python/benchmarks/` | Terminal scoreboards → CSV |
| `test_data/*.bin` | Input datasets |

---

*Last updated for the adaptive-precision flag + SZ3/ZFP/Elf/ALP frontend comparisons.*
