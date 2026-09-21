# How to run all compression comparisons (simple checklist)

Do these steps **in order** from the project root:
`/Users/admin/Desktop/DLC-v2`

---

## 1. One-time setup

```bash
cd /Users/admin/Desktop/DLC-v2

# Python packages for the dashboard + optional SZ3/ZFP
pip install -r python/requirements.txt
pip install pysz zfpy          # optional but recommended for full SZ3/ZFP columns

# Build the real C++ compressor
cmake -S cpp -B cpp/build
cmake --build cpp/build -j4

# Create the .bin test datasets (needed for charts)
python test_data/generate_test_data.py
```

Check you now have:
- `cpp/build/dlc` (the engine)
- several files under `test_data/*.bin`

---

## 2. See everything on the frontend (same as Gorilla / GZIP)

This is the same Streamlit app you already used. Industry tab now also
includes **SZ3, ZFP, Elf, ALP** in the **same bar charts and tables**.

```bash
cd /Users/admin/Desktop/DLC-v2
streamlit run python/app.py
```

Then in the browser:

1. Left sidebar → pick a dataset  
2. Open tab **“Industry Comparison”**  
3. Click **“Run Industry Comparison”**  
4. You get the same style of ratio chart + size chart + results table  
   (Gorilla, DoD, RLE, GZIP, ZLIB, **SZ3, ZFP, Elf, ALP**, DLC)  
5. For *all* datasets at once → tab **“Batch All Datasets”** →
   **“Run Full Batch Analysis”**

If SZ3/ZFP show a warning about missing packages, go back to step 1 and
install `pysz` / `zfpy`, then refresh the page.

---

## 3. Optional: CSV numbers for the paper (terminal)

Same comparisons, written to CSV (no UI):

```bash
cd /Users/admin/Desktop/DLC-v2

# All newer methods vs DLC → paper/sota_comparison_results.csv
PYTHONPATH=python:python/benchmarks python python/benchmarks/compare_sota.py

# Old + fair industry script (Gorilla/DoD/GZIP…)
PYTHONPATH=python python python/benchmarks/compare_industry_fair.py

# Precision 8…20 trade-off table → paper/precision_ablation_results.csv
PYTHONPATH=python:python/benchmarks python python/benchmarks/ablation_precision.py
```

Smoke-test on 1 file first (faster):

```bash
PYTHONPATH=python:python/benchmarks python python/benchmarks/compare_sota.py --limit 1
```

---

## What each method is (one line)

| Name | Shown like | Notes |
|------|------------|--------|
| Gorilla / RLE / GZIP / ZLIB / Elf / ALP | Lossless | Error = 0 |
| DoD (16-bit) | Near-lossless | Fair match to DLC’s bit depth |
| SZ3 / ZFP | Near-lossless | Same error budget ≈ 8e-6 as DLC |
| DLC v2 | Near-lossless | Your C++ engine |

---

## If something is missing

| Symptom | Fix |
|---------|-----|
| Dashboard says C++ binary not found | Rebuild: `cmake --build cpp/build -j4` |
| No datasets in sidebar | `python test_data/generate_test_data.py` |
| SZ3 / ZFP columns were blank | Fixed: built-in fallbacks always run now — restart Streamlit and re-run batch |
| Want official SZ3/ZFP libs | Optional: `pip install pysz zfpy` (fallback still works without them) |
