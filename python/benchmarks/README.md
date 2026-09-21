# Benchmarks

## Older comparison (Gorilla / GZIP / etc.)
- `compare_industry.py` / `compare_industry_fair.py`

## Paper revision
- `compare_sota.py` — DLC vs SZ3, ZFP, Elf, ALP  
  Writes `paper/sota_comparison_results.csv`
- `ablation_precision.py` — precision sweep with `--ablation-precision` semantics
  (floor off) plus a default 16-bit-floor reference row.
  Writes `paper/precision_ablation_results.csv`.

**Precision modes (C++ and Python):**
- **Default:** 16-bit floor (error contract) + adaptive up to `--precision`
- **Ablation:** `--ablation-precision` → full adaptive 8–20 bit range


### Optional (for real SZ3 / ZFP numbers)
```bash
pip install pysz zfpy
```
Elf and ALP always run. SZ3/ZFP need the packages above.

### Quick smoke
```bash
PYTHONPATH=python:python/benchmarks python python/benchmarks/ablation_precision.py --limit 1
PYTHONPATH=python:python/benchmarks python python/benchmarks/compare_sota.py --limit 1
```

**Note:** These scripts *measure* the compressor. The real compressor is the
C++ engine in `cpp/` (Python can also compress for demos/tests). Both now use
adaptive 8–20-bit precision by default.
