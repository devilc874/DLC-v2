# DLC v2 — System Architecture & LLM Knowledge Base

> **Last updated:** 2026-04-15  
> **Status:** Production-complete (all 17 phases finalized)  
> **Context window optimization:** This document is designed for dense LLM ingestion. Upload this single file to any AI to obtain full project understanding.

---

## 1. Core Objective & Constraints

**DLC (Delta-Linear Compression)** is a domain-specific, near-lossless compression engine optimized for high-frequency time-series sensor data stored as IEEE 754 `float64` values.

### Key Constraints

| Constraint | Value |
|---|---|
| Target data type | `float64` (8-byte IEEE 754 little-endian) |
| Compression mode | Near-lossless (quantized residuals) |
| Max reconstruction error | `< 8x10^-6` (strict, per-sample at 16-bit; configurable via `--precision`) |
| Default precision | 16-bit quantization (`2^-16` step size), configurable 8-20 |
| File format | `.dlc` binary (cross-platform, little-endian) |
| Parallelism | Multi-threaded, capped at `cpu_count()` |
| Dependencies | Standard library only (no Boost, no external codecs) |

### Why not just use GZIP?

LZ77 dictionary compressors (GZIP, ZLIB, LZ4) operate on raw byte sequences. For `float64` sensor data, the **lower 40 bits of the mantissa contain thermal noise** — essentially random bytes. LZ77 cannot find repeating byte patterns in random noise, so it falls back to storing raw data plus overhead. This is the **"Entropy Trap"** — float64 sensor data is intrinsically incompressible to byte-level dictionary methods.

DLC solves this by operating in the **mathematical domain**: it fits predictive models to the signal, computes small residuals, quantizes them to discard noise below the precision threshold, then entropy-encodes the concentrated residual distribution.

---

## 2. The Dual Architecture

### Python Prototype (`python/dlc/`)

The Python implementation served as the reference prototype and is used for:
- Rapid algorithm iteration and unit testing
- Cross-language verification (Python-compressed ↔ C++-decompressed)
- Hypothesis-based fuzz testing

**GIL Bottleneck:** Python's `ThreadPoolExecutor` is used for parallel block dispatch, but the Global Interpreter Lock (GIL) prevents true parallel execution of CPU-bound code. Throughput plateaus at ~4-21 MB/s regardless of thread count.

### C++17 Production Engine (`cpp/`)

The C++ engine is a **bit-for-bit compatible** reimplementation using:
- `std::thread` + `std::future` for true parallel block compression
- `std::memcpy` for safe `float64 ↔ uint64` type punning (no `reinterpret_cast`)
- `zlib` for final deflate post-compression
- No external dependencies beyond the C++17 standard library and zlib

**Performance:** True multithreading bypasses the GIL entirely. On 8 MB datasets, the C++ engine achieves ~13 MB/s single-threaded. On larger datasets (100MB+), parallelism scales near-linearly across cores.

### Cross-Compatibility Guarantee

The `.dlc` binary format is identical between both engines:
- Python `zlib.crc32(payload) & 0xFFFFFFFF` matches C++ `crc32()` from `<zlib.h>`
- Python `struct.pack('<d', val)` matches C++ little-endian `memcpy`
- Python `struct.unpack('<Q', struct.pack('<d', val))[0]` matches C++ `uint64_t` casting via `memcpy`

---

## 3. The 5-Stage Compression Pipeline

```
Input: float64[] → [Block Chunking] → [Adaptive Windowing] → [Multi-Model Selection]
    → [Residual Encoding] → [Assembly + Zlib] → Output: .dlc file
```

### Stage 1: Block Chunking (Parallel Dispatch)

- Input array is split into fixed-size **blocks** (default: 100,000 samples)
- Each block is dispatched to a thread from the pool
- Results are collected in **strict original order** for deterministic output
- Thread pool capped at `os.cpu_count()` / `std::thread::hardware_concurrency()`

**Files:** `pipeline.py` → `compress_parallel()`, `pipeline.cpp`

### Stage 2: Adaptive Windowing (Variance-Guided Greedy Expansion)

Each block is segmented into variable-length **windows** using a variance-based algorithm with **adaptive MAX_WINDOW** based on block size:

| Block Size | MAX_WINDOW |
|---|---|
| < 5,000 samples | 512 |
| < 100,000 samples | 1,024 |
| ≥ 100,000 samples | 4,096 |

**Algorithm:**
1. **Initialize:** Start with `MIN_WINDOW = 16` samples
2. **Fast-path:** If block variance ≈ 0 → emit entire block as single window (constant data)
3. **Compute threshold:** `threshold = max(initial_var × 10.0, global_var × 0.10)`
4. **Greedy doubling:** Expand window size by 2× while `variance(window) ≤ threshold`
5. **Stagnation guard:** Stop expansion if variance jumps > 1.5× between doublings
6. **Binary search:** Find exact split boundary in `[min_size, candidate_size]`
7. **Emit window**, advance cursor, repeat

**Variance computation** uses Welford's online algorithm for numerical stability.

**Intuition:** Low-variance regions (smooth signal) get large windows → efficient model fitting. High-variance regions (spikes, transitions) get small windows → tight local fitting.

**Files:** `windowing.py` → `segment_block()`, `windowing.cpp`

### Stage 3: Multi-Model Selection

For each window, **six candidate models** are fitted and the best is selected by **lowest Sum of Squared Residuals (SSR)**:

| Model ID | Name | Formula | Parameters Stored | Best For |
|---|---|---|---|---|
| 0 | Linear | `y = mx + c` | `[m, c]` (2 × float64) | Ramps, steady trends |
| 1 | Quadratic | `y = ax² + bx + c` | `[a, b, c]` (3 × float64) | Curves, parabolic segments |
| 2 | XOR-Delta | `bits(yᵢ) ⊕ bits(yᵢ₋₁)` | `[anchor]` (1 × float64) | Noisy/random, non-finite values |
| 3 | **Constant** | `y = c` | `[mean]` (1 × float64) | Step functions, flat regions |
| 4 | **Sinusoidal** | `y = A·sin(ωx + φ) + dc` | `[A, ω, φ, dc]` (4 × float64) | Periodic sensor data |
| 5 | **Pred-XOR** | `bits(yᵢ) ⊕ bits(2yᵢ₋₁ − yᵢ₋₂)` | `[anchor0, anchor1]` (2 × float64) | Trending noisy data |

**Sinusoidal fitting** uses FFT for dominant frequency estimation, then Gauss-Newton / Levenberg-Marquardt refinement (C++ uses a custom O(n) DFT + 4×4 Gaussian elimination — no external FFTW dependency).

**Selection logic:**
1. If window contains `NaN`, `±Inf`, or subnormals → **force XOR-Delta** (only safe model)
2. Fit all 6 models, compute SSR for each
3. **Constant preference:** If Constant SSR is within **1%** of best → prefer Constant (1 param)
4. Pick best XOR variant (XOR-Delta vs Pred-XOR) by popcount SSR proxy
5. If best XOR SSR is within **2%** of best analytical → **prefer XOR** (lossless, fewer params)
6. Otherwise, pick the analytical model with lowest SSR

**Float64 bitwise conversion** (Python-safe, per addendum #2):
```python
bits = struct.unpack('<Q', struct.pack('<d', val))[0]  # float64 → uint64
val  = struct.unpack('<d', struct.pack('<Q', bits))[0]  # uint64 → float64
```

**Files:** `models.py` → `select_model()`, `models.cpp`

### Stage 4: Residual Encoding

For analytical models (Linear/Quadratic/Constant/Sinusoidal), the float residuals are:
1. **Adaptive precision selected:** 8–20 bits per window based on residual magnitude (smaller residuals → fewer bits → better compression)
2. **Quantized:** `q = round(residual × 2^precision_bits)` — converts to integers
3. **Trial-encoded** with **5 strategies** — the **smallest output wins**:

| Encoding ID | Strategy | Description |
|---|---|---|
| 0 | Zigzag + Varint | Signed→unsigned zigzag mapping, then variable-length byte encoding |
| 1 | Fixed-Width Bitpack | Uniform bitwidth for all values (1-byte header stores bit_width) |
| 2 | Delta-of-Residuals | First-order differences of quantized values, then zigzag+varint |
| 3 | MAD Outlier Separation | MAD-based inlier/outlier split; inliers bitpacked, outliers stored separately |
| 4 | **RLE + Varint** | Run-length encoding of repeated values (dominates on Constant model windows) |
| 5 | **Delta-of-Delta** | Second-order differencing of quantized values with zigzag-varint encoding |

**Block-level DoD Fast Path:** Before windowed compression, the pipeline evaluates encoding the entire block as a single Constant(mean) window with DoD encoding (subtracts block mean, quantizes, then applies second-order differencing). If this produces smaller output than windowed compression, it is used directly -- eliminating all per-window metadata overhead. This makes DLC a **functional superset** of standalone DoD.

**Zigzag encoding:** Maps signed integers to unsigned: `z = (n << 1) ^ (n >> 63)`. Concentrates small values near zero for efficient varint encoding.

**Varint encoding:** 7 bits per byte, MSB = continuation flag. Small residuals (common after good model fit) encode in 1-2 bytes.

**MAD Outlier Separation:**
- Computes Median Absolute Deviation of residuals
- Values beyond `4.5 × MAD` are classified as outliers (tuned from 6.0)
- Inliers bitpacked at reduced bitwidth; outliers stored as `(index, value)` pairs
- Wins when signal has occasional spikes in otherwise smooth data

**Adaptive Precision:** Each window independently selects precision bits (8–20) based on max |residual|. Stored as `uint8` in the window block header. Near-zero residuals (Constant model) use 8-bit precision → 1-byte quantized values → RLE encodes them in 2-3 bytes total.

**Dequantization:** `residual = q × 2^(-precision_bits)`. With 12-bit precision: step = `2⁻¹² ≈ 2.44×10⁻⁴`. Max error per sample: `< 1.22×10⁻⁴ ≪ 8×10⁻⁶` bound.

For XOR-based models (XOR-Delta, Pred-XOR): residuals are `uint64` XOR values with optional **delta-of-XOR** compression (prefix byte `0x00` = raw, `0x01` = delta-encoded). No quantization needed (lossless).

**Files:** `encoders.py` → `trial_encode()`, `encoders.cpp`

### Stage 5: Assembly & Zlib Post-Compression

The per-window encoded blocks are concatenated into a **raw payload**, then:

1. **Zlib deflate** compresses the entire payload (the quantized residuals have concentrated distributions that deflate efficiently)
2. **File assembly:**

```
┌──────────────────────────────────────────────────────────────┐
│ Header (64 bytes)                                            │
│   Magic: b'DLC\x02'  (4 bytes)                              │
│   Version: major(2B) + minor(2B)                             │
│   Precision bits: uint16 (file-level default)                │
│   Chunk size: uint32                                         │
│   Total samples: uint64                                      │
│   Reserved: 42 bytes (zero-padded)                           │
├──────────────────────────────────────────────────────────────┤
│ Uncompressed payload size (4 bytes, uint32 LE)               │
├──────────────────────────────────────────────────────────────┤
│ Zlib-compressed payload (N bytes)                            │
│   Window blocks packed sequentially:                         │
│     window_length (uint32),                                  │
│     model_id (uint8),                                        │
│     encoding_id (uint8),                                     │
│     precision_bits (uint8),  ← per-window adaptive precision │
│     model_params (N×float64),                                │
│     encoded_residuals_len (uint32),                          │
│     encoded_residuals (bytes)                                │
├──────────────────────────────────────────────────────────────┤
│ Footer (8 bytes)                                             │
│   Magic: 0xED 0xDC 0xBA 0x01 (4 bytes)                      │
│   CRC-32 of uncompressed payload (4 bytes, uint32 LE)        │
│   Python: zlib.crc32(payload) & 0xFFFFFFFF                   │
└──────────────────────────────────────────────────────────────┘
```

**Files:** `format.py` → `write_dlc_file()` / `read_dlc_file()`, `format.cpp`

---

## 4. The "Entropy Trap" & Benchmark Results

### Why DLC Beats GZIP on Structured Data

**IEEE 754 float64 layout:**
```
[Sign 1b][Exponent 11b][Mantissa 52b]
```

For sensor data like temperature readings (e.g., 23.456°C → 23.457°C), consecutive samples share the same sign and exponent bits, and their upper mantissa bits are nearly identical. The **lower ~40 bits** of the mantissa contain **thermal noise** — random bytes that defeat LZ77.

**DLC's approach:**
1. Fit a predictive model → captures the signal's mathematical structure
2. Compute residuals → small values centered around zero
3. Quantize → discard the noise below precision threshold
4. The quantized residuals follow a **Laplacian distribution** (peaked at zero)
5. Zigzag + varint encoding exploits this concentration → most values encode in 1-2 bytes

**GZIP's approach:**
1. Slide a 32KB dictionary window across raw bytes
2. Look for repeating byte sequences → **none exist** in float64 noise
3. Fall back to literal encoding → ratio ≈ 1.0x

### Live Benchmark Summary

All benchmarks are **dynamically measured** (no hardcoded values). Benchmark scripts:
- `python/benchmarks/compare_all.py` — GZIP vs ZLIB vs DLC C++ on every `.bin` dataset
- `python/benchmarks/compare_industry.py` — Gorilla, DoD (12-bit), RLE, GZIP, ZLIB vs DLC with error analysis
- `python/benchmarks/compare_industry_fair.py` — DoD forced to 16-bit (equal precision as DLC) for fair evaluation

**Key results at equal 16-bit precision:** DLC v2 wins on **80% (20/25) of datasets** against all industry techniques.

**Error Analysis (round-trip reconstruction error):**

| Technique | Typical Max Error | Precision |
|---|---|---|
| Gorilla | 0 (exact, lossless) | XOR-based |
| DoD (12-bit) | ~1.2x10^-4 | 12-bit quantization |
| DoD (16-bit) | ~7.6x10^-6 | 16-bit quantization |
| RLE | 0 (exact, lossless) | No quantization |
| GZIP | 0 (exact, lossless) | Byte-level |
| ZLIB | 0 (exact, lossless) | Byte-level |
| **DLC v2** | **<8x10^-6** | **16-bit adaptive** |

> DoD achieves higher ratios than DLC only when DoD is given a 15x looser error tolerance (12-bit vs 16-bit). At truly equal precision, DLC strongly dominates.

### Throughput Scaling

The throughput chart is generated dynamically by `visualize_final_dashboards.py`:
- **GZIP (Python):** ~21 MB/s, flat regardless of thread count (single-threaded)
- **DLC C++ (single-threaded):** ~13 MB/s on 8 MB datasets
- **DLC C++ (multi-threaded):** Scales near-linearly on large datasets (100MB+)

The apparent GZIP throughput advantage on small files is due to **subprocess launch overhead** for the C++ binary. On large datasets, DLC's parallel pipeline significantly outperforms.

---

## 5. Project Structure

```
capstonev2/
├── README.md                          # Project overview
├── DLC_SYSTEM_ARCHITECTURE.md         # THIS FILE — LLM knowledge base
│
├── python/                            # Python prototype & tools
│   ├── dlc/                           # Core Python package
│   │   ├── __init__.py                # Package init
│   │   ├── codec.py                   # High-level compress()/decompress() API
│   │   ├── pipeline.py                # Parallel block dispatch & ordered assembly
│   │   ├── windowing.py               # Adaptive variance-guided segmentation
│   │   ├── models.py                  # 6 models: Linear/Quad/XOR/Const/Sine/PredXOR
│   │   ├── encoders.py                # 5 encoding strategies + trial_encode()
│   │   ├── format.py                  # .dlc binary format (per-window precision)
│   │   └── cli.py                     # Command-line interface
│   │
│   ├── tests/                         # Pytest test suite
│   │   ├── test_windowing.py          # Window segmentation tests
│   │   ├── test_models.py             # Model fitting tests
│   │   ├── test_encoders.py           # Encoder round-trip tests
│   │   ├── test_format.py             # Binary format tests
│   │   ├── test_codec.py              # End-to-end compress/decompress
│   │   ├── test_cross_language.py     # Python↔C++ compatibility
│   │   └── test_fuzz.py               # Hypothesis-based property tests
│   │
│   ├── benchmarks/                    # Benchmarking & visualization
│   │   ├── compare_all.py            # Terminal: GZIP vs ZLIB vs DLC on ALL datasets
│   │   ├── bench_cpp.py               # C++ engine throughput benchmark
│   │   ├── bench_throughput.py        # Python engine throughput benchmark
│   │   ├── visualize_real_data.py     # AWS CPU overlay + residual histogram
│   │   └── visualize_final_dashboards.py  # Dynamic ratio & throughput charts
│   │
│   ├── app.py                         # Streamlit interactive demo dashboard
│   ├── setup.py                       # pip install -e .
│   └── requirements.txt               # numpy, matplotlib, streamlit
│
├── cpp/                               # C++17 production engine
│   ├── CMakeLists.txt                 # CMake build (MinGW/MSYS2)
│   ├── include/dlc/                   # Header files (mirror Python modules)
│   │   ├── codec.hpp                  # compress()/decompress() API
│   │   ├── pipeline.hpp               # Parallel block dispatch
│   │   ├── windowing.hpp              # Adaptive segmentation
│   │   ├── models.hpp                 # 6 models + SSR-based selection
│   │   ├── encoders.hpp               # 5 encoding strategies + XOR residuals
│   │   └── format.hpp                 # .dlc format (per-window precision)
│   ├── src/                           # Implementation files
│   │   ├── main.cpp                   # CLI entry point
│   │   ├── codec.cpp                  # High-level API
│   │   ├── pipeline.cpp               # Thread pool dispatch (all 6 models)
│   │   ├── windowing.cpp              # Adaptive windowing (stagnation guard)
│   │   ├── models.cpp                 # 6 models (incl. DFT + Gauss-Newton)
│   │   ├── encoders.cpp               # 5 strategies + RLE + XOR encoding
│   │   └── format.cpp                 # Binary format I/O
│   ├── tests/                         # CTest unit tests
│   └── build/                         # Build output (dlc.exe)
│
├── test_data/                         # Binary test datasets
│   ├── generate_test_data.py          # Generate 17 synthetic + NAB datasets
│   ├── download_massive_data.py       # Download Jena Climate (420K samples)
│   ├── download_large_datasets.py     # Generate 100MB-730MB+ large datasets
│   ├── sine_1m.bin                    # 1M continuous sine wave
│   ├── ramp_1m.bin                    # 1M linear ramp
│   ├── synthetic_random_walk_1m.bin   # 1M cumulative random walk
│   ├── synthetic_noisy_sine_1m.bin    # 1M multi-freq sine + drift + noise
│   ├── massive_sensor_3m.bin          # 3M industrial sensor simulation
│   ├── real_cpu_4k.bin                # AWS EC2 CPU utilization (4,032 pts)
│   ├── real_jena_temperature_420k.bin # Jena Climate temperature (420,551 pts)
│   ├── real_jena_pressure_420k.bin    # Jena Climate pressure (420,551 pts)
│   └── ... (20+ more .bin files)
│
└── output/                            # Generated charts
    ├── final_ratios.png               # GZIP vs ZLIB vs DLC ratio comparison
    ├── final_throughput.png           # Throughput scaling chart
    ├── aws_cpu_overlay.png            # Signal + prediction overlay
    └── aws_cpu_residuals.png          # Residual Laplacian histogram
```

---

## 6. Build & Run Commands

### Python Engine
```bash
cd capstonev2/python
pip install -e .                             # Install package
python -m dlc compress -i data.bin -o out.dlc  # Compress
python -m dlc decompress -i out.dlc -o recovered.bin  # Decompress
pytest tests/                                # Run all tests
```

### C++ Engine (MSYS2/MinGW on Windows)
```bash
cd capstonev2/cpp/build
cmake .. -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
./dlc compress -i ../../test_data/sine_1m.bin -o out.dlc --workers 4
./dlc decompress -i out.dlc -o recovered.bin
```

### Benchmarks & Visualization
```bash
# Generate all test datasets
python test_data/generate_test_data.py
python test_data/download_massive_data.py

# Terminal comparison across ALL datasets
python python/benchmarks/compare_all.py

# Generate presentation charts
python python/benchmarks/visualize_real_data.py
python python/benchmarks/visualize_final_dashboards.py

# Launch interactive Streamlit demo
streamlit run python/app.py
```

---

## 7. Key Algorithmic Constants

| Constant | Value | Location | Purpose |
|---|---|---|---|
| `MIN_WINDOW` | 16 | windowing.py/cpp | Minimum adaptive window size |
| `MAX_WINDOW_SMALL` | 512 | windowing.py/cpp | Max window for blocks < 5K samples |
| `MAX_WINDOW_MEDIUM` | 1,024 | windowing.py/cpp | Max window for blocks < 100K samples |
| `MAX_WINDOW_LARGE` | 4,096 | windowing.py/cpp | Max window for blocks ≥ 100K samples |
| `MEAN_VAR_FACTOR` | 10.0 | windowing.py/cpp | Threshold = initial_var × 10 (was 13) |
| `GLOBAL_VAR_FACTOR` | 0.10 | windowing.py/cpp | Threshold = global_var × 0.10 (was 0.12) |
| `CONST_VAR_EPS` | 1e-20 | windowing.py/cpp | Below this, block treated as constant |
| `DEFAULT_PRECISION` | 16 | format.py/cpp | File-level default quantization bits |
| `MIN_PRECISION` | 8 | encoders.py/cpp | Minimum per-window adaptive precision |
| `MAX_PRECISION` | 20 | encoders.py/cpp | Maximum per-window adaptive precision |
| `DEFAULT_CHUNK_SIZE` | 100,000 | format.py/cpp | Block size for parallel dispatch |
| `MAGIC_BYTES` | `b'DLC\x02'` | format.py/cpp | File header magic |
| `FOOTER_MAGIC` | `0xED 0xDC 0xBA 0x01` | format.py/cpp | File footer magic |
| `XOR_DELTA_PREFERENCE` | 1.02 | models.py/cpp | XOR selected if within 2% of best (was 5%) |
| `CONST_PREFERENCE` | 1.01 | models.py/cpp | Constant selected if within 1% of best |
| `MAD_OUTLIER_FACTOR` | 4.5 | encoders.py/cpp | Outlier threshold = 4.5 × MAD (was 6.0) |
| `SINUSOID_MIN_WINDOW` | 32 | models.py/cpp | Min window size for sinusoidal fitting |

---

## 8. Implementation Phases (Completed)

| Phase | Description | Status |
|---|---|---|
| 0 | Implementation plan & approval | ✅ |
| 1 | Environment & scaffold | ✅ |
| 2 | Binary file format (header, footer, CRC-32) | ✅ |
| 3 | Adaptive windowing (variance-guided) | ✅ |
| 4 | Predictive models (Linear, Quadratic, XOR-Delta) | ✅ |
| 5 | Residual encoding (4 strategies + trial_encode) | ✅ |
| 6 | Parallel pipeline (ThreadPoolExecutor / std::thread) | ✅ |
| 7 | Zlib post-compression & final assembly | ✅ |
| 8 | C++17 full port | ✅ |
| 9 | Cross-language & fuzz testing | ✅ |
| 10 | Benchmarks, CLI, documentation | ✅ |
| 11 | C++ benchmarking & real-world NAB data | ✅ |
| 12 | Real-world data visualization | ✅ |
| 13 | Final dashboards & expanded datasets | ✅ |
| 14 | Streamlit showcase app | ✅ |
| 15 | LLM Knowledge Base Export (this document) | ✅ |
| **16** | **Claude Improvements Integration** -- 3 new models (Constant, Sinusoidal, Pred-XOR), RLE encoding, adaptive precision, per-window precision bits, complete MAD decoder, delta-of-XOR, tuned constants. Full C++ port with explicit Sinusoidal/Pred-XOR decompress. | ✅ |
| **17** | **Industry Comparison & DoD Integration** -- Delta-of-Delta as 6th encoding strategy, block-level DoD fast path (subtracts mean, uses select_precision), industry benchmark suite (Gorilla/DoD/RLE/GZIP/ZLIB), fair 16-bit evaluation, round-trip error analysis, Streamlit dashboard upgrade (5 tabs with error tables and fair comparison), configurable `--precision` flag (8-20 bits), 730MB+ dataset support. | ✅ |

---

*End of DLC System Architecture. This document contains the complete mathematical, architectural, and operational knowledge needed to understand, extend, or reimplement the DLC v2 compression engine.*
