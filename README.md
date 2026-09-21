# DLC — Delta-Linear Compression Engine

A domain-specific, **near-lossless compression engine** for high-frequency time-series sensor data.  
Built as a **dual-language** implementation: Python (prototype) + C++17 (production), with strict **bit-for-bit cross-language compatibility**.  
**Beats industry standards** (Gorilla, Delta-of-Delta, RLE, GZIP, ZLIB) on **80% of datasets** at equal precision.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Compression Pipeline](#compression-pipeline)
- [Binary File Format (.dlc)](#binary-file-format-dlc)
- [Predictive Models](#predictive-models)
- [Encoding Strategies](#encoding-strategies)
- [Parallelism & Determinism](#parallelism--determinism)
- [Project Structure](#project-structure)
- [Build & Run](#build--run)
- [Testing](#testing)
- [Technical Addendums](#technical-addendums)
- [Performance Characteristics](#performance-characteristics)

---

## Overview

DLC is designed for **sensor telemetry, IoT data streams, and scientific instrumentation** where data arrives as sequences of `float64` values at high sample rates. Instead of using general-purpose compressors (gzip, zstd), DLC exploits the **mathematical structure** of time-series data:

- **Smooth signals** → fit a regression model, compress tiny residuals
- **Noisy signals** → adaptive windowing isolates stable regions
- **Random/chaotic data** → XOR-Delta encoding preserves exact bits

### Key Guarantees

| Property | Guarantee |
|----------|-----------|
| **Max reconstruction error** | < 8×10⁻⁶ (configurable via precision bits) |
| **Deterministic output** | Same input → identical `.dlc` file regardless of thread count |
| **Cross-language compatibility** | Python `.dlc` ↔ C++ `.dlc` are bit-for-bit identical |
| **Thread safety** | All shared state is read-only; no locks needed |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DLC Compression Pipeline                     │
├─────────────┬──────────────┬──────────────┬────────────┬───────────┤
│  1. Chunk   │  2. Window   │  3. Model    │ 4. Encode  │ 5. Pack   │
│  (100K pts) │  (16–4096)   │  Selection   │  Residuals │  .dlc     │
│             │  Variance-   │  6 Models:   │  6 Strats: │  Header + │
│  Split into │  guided      │  Linear /    │  Zigzag /  │  Zlib +   │
│  blocks for │  greedy      │  Quadratic / │  Bitpack / │  Footer   │
│  parallel   │  expansion   │  XOR-Delta / │  Delta /   │  (CRC-32) │
│  dispatch   │  + binary    │  Const/Sine/ │  Outlier / │           │
│             │  search      │  PredXOR     │  RLE/DoD   │           │
└─────────────┴──────────────┴──────────────┴────────────┴───────────┘
```

---

## Compression Pipeline

### Stage 1: Chunking
The input signal is split into fixed-size **blocks** (default 100,000 samples). Each block is dispatched to a thread pool for independent processing. Results are assembled in **strict original order** for deterministic output.

### Stage 2: Adaptive Windowing (`windowing.py` / `windowing.cpp`)
Each block is segmented into variable-length **windows** (16–4096 samples) using a variance-guided algorithm:

1. Start with a minimum window (16 samples)
2. Compute initial variance and auto-tune threshold:
   ```
   threshold = max(initial_variance × 13, global_variance × 0.12)
   ```
3. **Greedy doubling**: expand the window (×2) while variance stays below threshold
4. **Binary search**: find the exact split point within the last doubling interval
5. Emit the window and advance

This produces **large windows for smooth regions** and **small windows at discontinuities**.

### Stage 3: Model Selection (`models.py` / `models.cpp`)
For each window, **six predictive models** are fitted and the best is selected by **Sum of Squared Residuals (SSR)**:

| Model | Parameters | Formula | Best For |
|-------|-----------|---------|----------|
| **Linear** | m, c (2 params) | y = mx + c | Ramps, trends |
| **Quadratic** | a, b, c (3 params) | y = ax² + bx + c | Accelerating signals |
| **XOR-Delta** | anchor (1 param) | bitwise XOR chain | Random/noisy data |
| **Constant** | c (1 param) | y = c | Flat regions, step functions |
| **Sinusoidal** | A, w, phi, dc (4 params) | y = A*sin(wx+phi)+dc | Periodic sensor data |
| **Pred-XOR** | a0, a1 (2 params) | linearXOR prediction | Trending noisy data |

- If XOR-Delta SSR is within **2% of the best**, it's preferred (fewer parameters, lossless)
- Non-finite values (NaN, Inf) automatically use XOR-Delta
- Constant preferred if within **1%** of best (cheapest model)

### Stage 4: Residual Encoding (`encoders.py` / `encoders.cpp`)
Residuals from the selected model are **quantized** to integer multiples of 2⁻ᵖ (where p = precision bits, default 12). Then all six encoding strategies are trial-run and the **smallest output** is selected:

| ID | Strategy | How It Works | Best For |
|----|----------|-------------|----------|
| 0 | **Zigzag + Varint** | Map signed→unsigned, variable-length bytes | General purpose |
| 1 | **Fixed-Width Bitpack** | All values at same bit width | Uniform magnitude |
| 2 | **Delta-of-Residuals** | Encode differences between consecutive residuals | Smooth residuals |
| 3 | **MAD Outlier Separation** | Separate outliers (>4.5xMAD), bitpack the rest | Spiky signals |
| 4 | **RLE + Varint** | Run-length encode repeated values | Constant model windows |
| 5 | **Delta-of-Delta** | Second-order differencing with zigzag-varint | Trending data |

**Block-level DoD Fast Path:** Before windowed compression, the pipeline evaluates encoding the entire block as a single Constant(mean) window with DoD encoding. If this beats windowing, it is used directly.

### Stage 5: Binary Assembly (`format.py` / `format.cpp`, `codec.py` / `codec.cpp`)
Window blocks are serialized, concatenated, compressed with **zlib deflate**, and wrapped in the `.dlc` binary format with CRC-32 integrity checking.

---

## Binary File Format (.dlc)

### Overall Layout
```
┌──────────────────┐
│  Header (64 B)   │  Magic + version + config
├──────────────────┤
│  Uncomp Size (4) │  uint32_le: uncompressed payload size
├──────────────────┤
│  Zlib Data (N B) │  Deflate-compressed window blocks
├──────────────────┤
│  Footer (8 B)    │  Footer magic + CRC-32
└──────────────────┘
```

### Header (64 bytes, packed, little-endian)
```
Offset  Size  Field             Default
──────  ────  ────────────────  ───────
0       4     magic             "DLC\x02"
4       2     major_version     0
6       2     minor_version     1
8       2     precision_bits    12
10      4     chunk_size        100000
14      8     total_samples     (varies)
22      42    reserved          0x00...
                                ──────
                                64 bytes total
```

### Window Block (variable length, inside zlib payload)
```
┌────────────────┬──────────┬────────────┬──────────────────────┬──────────────────┬──────────────┐
│ window_length  │ model_id │ encoding_id│ model_params         │ residual_len (4) │ residuals    │
│ uint32_le      │ uint8    │ uint8      │ N × float64_le       │ uint32_le        │ (bytes)      │
└────────────────┴──────────┴────────────┴──────────────────────┴──────────────────┴──────────────┘
```
- Model param count: Linear=2, Quadratic=3, XOR-Delta=1, Constant=1, Sinusoidal=4, Pred-XOR=2
- **Per-window precision_bits** field (uint8, 8-20 bits) stored after encoding_id

### Footer (8 bytes, packed)
```
Offset  Size  Field
──────  ────  ──────────
0       4     magic       0xED 0xDC 0xBA 0x01
4       4     crc32       CRC-32 of uncompressed payload
```

---

## Predictive Models

### Linear Regression
Closed-form least squares via the **normal equation**:
```
m = (n·Σxy − Σx·Σy) / (n·Σx² − (Σx)²)
c = (Σy − m·Σx) / n
```

### Quadratic Regression
3×3 normal equations solved via **Cramer's rule** (no linear algebra library needed):
```
| Σx⁴ Σx³ Σx² | | a |   | Σx²y |
| Σx³ Σx² Σx  | | b | = | Σxy  |
| Σx² Σx  n   | | c |   | Σy   |
```

### XOR-Delta
Lossless bitwise encoding:
```
anchor = y[0]
residual[i] = float64_bits(y[i]) XOR float64_bits(y[i-1])
```
In Python, uses `struct.unpack('<Q', struct.pack('<d', val))[0]` for safe float64→uint64 conversion.  
In C++, uses `std::memcpy(&bits, &val, 8)`.

---

## Encoding Strategies

### Zigzag + Varint (Strategy 0)
Maps signed integers to unsigned via zigzag encoding: `(n << 1) ^ (n >> 63)`, then encodes each as a variable-length integer (7 bits per byte, MSB = continuation flag).

### Fixed-Width Bitpack (Strategy 1)
Finds the maximum zigzag value across all residuals, determines the minimum bit width, and packs all values at that fixed width. Header: 1 byte storing the bit width.

### Delta-of-Residuals (Strategy 2)
Computes first-order differences between consecutive quantized residuals, then applies zigzag+varint to the deltas. Exploits smoothness in the residual signal itself.

### MAD Outlier Separation (Strategy 3)
1. Compute **Median Absolute Deviation (MAD)** of the quantized residuals
2. Flag values > 3×MAD from the median as **outliers**
3. Store outliers in a separate table (index + zigzag-varint value)
4. Bitpack the remaining inliers at reduced bit width (outlier positions zeroed)

---

## Parallelism & Determinism

### Thread Pool
- **Python**: `concurrent.futures.ThreadPoolExecutor`
- **C++**: `std::async(std::launch::async, ...)`
- Worker count capped at `os.cpu_count()` / `std::thread::hardware_concurrency()`

### Deterministic Assembly
Blocks are dispatched to the pool but results are collected in **strict sequential order** via ordered futures:
```python
futures = [pool.submit(process_block, block) for block in blocks]
payload = b''.join(f.result() for f in futures)  # ordered!
```
This guarantees **identical output** regardless of which thread finishes first.

---

## Project Structure

```
capstonev2/
├── python/
│   ├── dlc/
│   │   ├── __init__.py           # Package init (v0.1.0)
│   │   ├── format.py             # Binary format serialization
│   │   ├── windowing.py          # Adaptive variance-guided segmentation
│   │   ├── models.py             # 6 Predictive Models
│   │   ├── encoders.py           # 6 encoding strategies + trial_encode() + encode_dod() / decode_dod()
│   │   ├── pipeline.py           # Parallel block processing
│   │   ├── codec.py              # File-level compress/decompress + zlib
│   │   └── cli.py                # Command-line interface
│   ├── tests/
│   │   ├── test_format.py        # Binary format round-trip
│   │   ├── test_windowing.py     # Window bounds & segmentation
│   │   ├── test_models.py        # Model fitting & selection
│   │   ├── test_encoders.py      # All 6 encoding strategies
│   │   ├── test_pipeline.py      # Determinism & error bound
│   │   ├── test_codec.py         # File I/O round-trip
│   │   ├── test_cross_lang.py    # Python <-> C++ compatibility
│   │   └── test_fuzz.py          # Hypothesis property-based fuzzing
│   ├── benchmarks/
│   │   ├── compare_all.py        # GZIP vs ZLIB vs DLC
│   │   ├── compare_industry.py   # Gorilla/DoD/RLE/GZIP/ZLIB vs DLC + error analysis
│   │   ├── compare_industry_fair.py # Fair 16-bit DoD evaluation
│   │   ├── bench_cpp.py          # C++ throughput benchmark
│   │   ├── bench_throughput.py   # Python throughput benchmark
│   │   └── visualize_*.py        # Chart generation scripts
│   ├── app.py                    # Streamlit dashboard (5 tabs)
│   ├── setup.py                  # pip install -e .
│   └── requirements.txt          # numpy, matplotlib, streamlit, plotly
├── cpp/
│   ├── include/dlc/
│   │   ├── format.hpp            # Binary format structs (pragma-packed)
│   │   ├── windowing.hpp         # Segmentation API
│   │   ├── models.hpp            # Model types & selection
│   │   ├── encoders.hpp          # Encoding strategies
│   │   ├── pipeline.hpp          # Parallel pipeline config
│   │   └── codec.hpp             # High-level compress/decompress
│   ├── src/
│   │   ├── format.cpp            # Binary serialization with LE helpers
│   │   ├── windowing.cpp         # Variance-guided segmentation
│   │   ├── models.cpp            # Cramer's rule regression + XOR-Delta
│   │   ├── encoders.cpp          # 4 encoding strategies with bit-level ops
│   │   ├── pipeline.cpp          # std::async thread pool
│   │   ├── codec.cpp             # File I/O + zlib integration
│   │   └── main.cpp              # CLI entry point
│   ├── tests/
│   │   ├── test_format.cpp       # GTest: format round-trip
│   │   ├── test_windowing.cpp    # GTest: segmentation
│   │   ├── test_models.cpp       # GTest: model fitting
│   │   ├── test_encoders.cpp     # GTest: encoding strategies
│   │   ├── test_pipeline.cpp     # GTest: pipeline determinism
│   │   └── test_codec.cpp        # GTest: file round-trip
│   └── CMakeLists.txt            # C++17 build (GTest FetchContent, zlib)
├── test_data/
│   ├── gen_test_data.py          # Deterministic test data generator
│   ├── small_200.bin             # 200-sample sine wave
│   ├── sine_10k.bin              # 10K-sample sine
│   ├── noisy_10k.bin             # 10K-sample noisy sine
│   ├── ramp_10k.bin              # 10K-sample linear ramp
│   ├── step_func_10k.bin         # 10K-sample step function
│   └── random_10k.bin            # 10K-sample random walk
└── README.md                     # This file
```

---

## Quick Start (Run This Project From Scratch)

> **Prerequisites:** Python 3.10+, MSYS2/MinGW with GCC 11+ (Windows) or GCC/Clang (Linux), CMake 3.16+, zlib

### Step 1: Clone & Setup Python

```powershell
git clone https://github.com/YOUR_USERNAME/capstonev2.git
cd capstonev2

# Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell
# source .venv/bin/activate          # Linux/Mac

# Install all Python dependencies
pip install -r python/requirements.txt
pip install -e python/
```

### Step 2: Generate Test Datasets

The `.bin` data files are not included in the repo (too large). Generate them:

```powershell
python test_data/generate_test_data.py           # Creates 17+ datasets (~60 MB)
python test_data/download_massive_data.py         # Downloads Jena Climate (~7 MB)
# python test_data/download_large_datasets.py     # Optional: 700MB+ stress-test data
```

### Step 3: Build the C++ Engine

```powershell
# Windows (MSYS2/MinGW)
$env:PATH = "C:\msys64\ucrt64\bin;$env:PATH"
cd cpp
mkdir build; cd build
cmake .. -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="C:/msys64/ucrt64" -DZLIB_ROOT="C:/msys64/ucrt64"
cmake --build . --config Release
cd ..\..

# Linux/Mac
# cd cpp && mkdir build && cd build
# cmake .. -DCMAKE_BUILD_TYPE=Release
# make -j$(nproc)
# cd ../..
```

### Step 4: Run!

```powershell
# Launch the Interactive Dashboard
streamlit run python/app.py

# Or compress a file from terminal
cpp\build\dlc.exe compress -i test_data\sine_1m.bin -o output.dlc --precision 16 --workers 4
cpp\build\dlc.exe decompress -i output.dlc -o recovered.bin

# Run benchmarks
python python/benchmarks/compare_all.py                # GZIP vs ZLIB vs DLC
python python/benchmarks/compare_industry.py            # All 6 industry techniques
python python/benchmarks/compare_industry_fair.py       # Fair 16-bit evaluation

# Run tests
cd python && pytest tests/ -v && cd ..
cd cpp\build && .\dlc_tests.exe && cd ..\..
```

---

## Detailed Build & Run

### Python

```bash
# Install
cd python
pip install -e ".[dev]"

# Run tests (70 core tests)
pytest tests/ -v --tb=short --ignore=tests/test_fuzz.py --ignore=tests/test_cross_lang.py

# Compress / Decompress
python -m dlc.cli compress -i ../test_data/sine_10k.bin -o output.dlc --precision 12 --workers 4
python -m dlc.cli decompress -i output.dlc -o recovered.bin

# Run benchmarks
python benchmarks/bench_throughput.py
```

### C++ (MSYS2 / MinGW on Windows)

```powershell
cd cpp
mkdir build; cd build

# Configure (set MSYS2 on PATH first)
$env:PATH = "C:\msys64\ucrt64\bin;$env:PATH"
cmake .. -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_PREFIX_PATH="C:/msys64/ucrt64" -DZLIB_ROOT="C:/msys64/ucrt64"

# Build
mingw32-make -j8

# Test (29 tests)
.\dlc_tests.exe

# Compress / Decompress
.\dlc.exe compress -i ..\..\test_data\sine_10k.bin -o output.dlc
.\dlc.exe decompress -i output.dlc -o recovered.bin
```

### C++ (Linux / macOS)

```bash
cd cpp && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
ctest --output-on-failure
./dlc compress -i ../../test_data/sine_10k.bin -o output.dlc
```

---

## Testing

### Test Summary

| Suite | Tests | What's Verified |
|-------|-------|-----------------|
| Python Core | 70 | Format round-trip, windowing bounds, model fitting, encoder round-trip, pipeline determinism, codec file I/O |
| Python Fuzz | 3 | NaN/Inf/subnormal handling, error bound on random data, small arrays |
| Python Cross-Lang | 3 | Python→C++ decompress, C++→Python decompress, SHA-256 file identity |
| C++ GTest | 29 | Same coverage as Python core (format, windowing, models, encoders, pipeline, codec) |
| **Total** | **105** | |

### Running All Tests

```bash
# Python (core + fuzz)
cd python && pytest tests/ -v

# C++ (GTest)
cd cpp/build && ./dlc_tests

# Cross-language (requires C++ binary built)
cd python && pytest tests/test_cross_lang.py -v
```

### Error Bound Verification
Every round-trip test verifies:
```python
max_error = np.max(np.abs(original - reconstructed))
assert max_error < 8e-6  # strict bound
```

---

## Technical Addendums

Three critical rules enforced throughout the codebase:

### 1. Python CRC-32 Unsigned Masking
```python
crc = zlib.crc32(payload) & 0xFFFFFFFF  # Force unsigned 32-bit
```
Without this mask, Python 2/3 differences in signed/unsigned CRC returns break C++ compatibility.

### 2. Float64 Bitwise Conversion (Python)
```python
bits = struct.unpack('<Q', struct.pack('<d', val))[0]  # float64 → uint64
val  = struct.unpack('<d', struct.pack('<Q', bits))[0]  # uint64 → float64
```
Python has no C-style pointer casting. The `struct` module provides safe, portable, little-endian conversion.

### 3. Thread Pool Worker Cap
```python
num_workers = min(requested_workers, os.cpu_count())       # Python
int workers = std::min(n, (int)std::thread::hardware_concurrency());  // C++
```
Prevents OS context-thrashing when processing massive files.

---

## Performance Characteristics

| Signal Type | Typical Compression Ratio | Notes |
|------------|--------------------------|-------|
| Pure sine wave | 10–50× | Linear model captures well, tiny residuals |
| Noisy sensor data | 3–8× | Windowing adapts to noise level |
| Linear ramp | 50–100× | Perfect linear fit, near-zero residuals |
| Random walk | 1.5–3× | XOR-Delta + zlib still finds redundancy |
| White noise | 0.9–1.2× | Close to incompressible (expected) |

Run the benchmark suite for exact numbers on your hardware:
```bash
python benchmarks/bench_throughput.py
```

---

## Dependencies

| Language | Dependency | Purpose |
|----------|-----------|---------|
| Python | NumPy ≥ 1.24 | Array operations, polyfit |
| Python | Hypothesis ≥ 6.0 | Property-based fuzz testing |
| Python | zlib (stdlib) | Deflate + CRC-32 |
| C++ | zlib | Deflate + CRC-32 |
| C++ | GoogleTest | Unit testing (fetched via CMake) |

No external linear algebra libraries (Eigen, LAPACK) are used — all regression is computed via closed-form normal equations.

---

*Built as a capstone project demonstrating systems-level engineering: binary format design, multi-model compression, parallel pipelines, industry benchmarking, and strict cross-language interoperability.*

> **Last Updated:** 2026-04-15 | 6 models, 6 encoders, DoD fast path, `--precision 8-20`, 80% industry win rate
