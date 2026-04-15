# DEEP PROJECT GUIDE — DLC v2 Compression Engine

> **Last updated:** 2026-04-15  
> **Audience:** Any developer, reviewer, or AI encountering this project for the first time.

---

## 1. What the Project Is

### The Problem

Scientific instruments, IoT sensors, and monitoring systems produce time-series data stored as IEEE 754 `float64` values (8 bytes each). A temperature sensor logging at 1 kHz for one day produces **691 MB of raw data**. Multiplied across thousands of sensors, storage costs explode.

### Why GZIP Fails — The "Entropy Trap"

Every `float64` value is 64 bits arranged as:

```
[Sign 1 bit][Exponent 11 bits][Mantissa 52 bits]
```

When a temperature sensor reads 23.456°C then 23.457°C, the sign and exponent bits stay the same — but the **lower ~40 bits of the mantissa contain thermal noise**. This noise is effectively random bytes.

GZIP uses LZ77 — a dictionary-based algorithm that slides a 32KB window across byte sequences looking for repeated patterns. **Random bytes have no repeating patterns.** GZIP sees noise, can't compress it, and falls back to storing raw data plus its own overhead. Result: ratio ≈ 1.0×.

This is the **"Entropy Trap"** — float64 sensor data looks high-entropy at the byte level even though the underlying signal is highly structured.

### How DLC Solves It

DLC operates in the **mathematical domain** instead of the byte domain:

1. **Fit a predictive model** to the signal (e.g., linear regression) — captures the mathematical structure
2. **Compute residuals** — the difference between prediction and actual values (small numbers centered around zero)
3. **Quantize** — round residuals to the nearest multiple of `2^-p` (p = precision bits, default 16), discarding the noise below precision threshold
4. **Encode** — the quantized integers follow a Laplacian distribution (peaked at zero) that encodes extremely efficiently

### "Near-Lossless" and the Error Bound

DLC is **not** bit-for-bit lossless. It quantizes residuals to discard noise below a precision threshold. The guarantee:

```
|original_value - reconstructed_value| < 8x10^-6   (at 16-bit precision)
```

This means every single sample is recovered within 0.000008 of its original value. For sensor data, this is far below the measurement noise floor. In practice, the C++ engine achieves errors as low as **7x10^-10** (4 orders of magnitude better than the bound).

**Precision is configurable** via the `--precision` CLI flag (both Python and C++ engines):
- `--precision 20` → near-lossless (tightest error bound, lower compression)
- `--precision 16` → standard (default, <8x10^-6 error)
- `--precision 12` → aggressive (higher compression, ~1.2x10^-4 error)
- `--precision 8` → maximum compression (lossy, visible error on sensitive data)

### Python Prototype vs C++ Engine

| Feature | Python (`python/dlc/`) | C++ (`cpp/`) |
|---------|----------------------|-------------|
| Purpose | Reference prototype, testing, visualization | Production engine |
| Parallelism | `ThreadPoolExecutor` (GIL-limited, ~4-21 MB/s) | `std::thread` + `std::future` (true parallel) |
| Output | Identical `.dlc` files | Identical `.dlc` files |
| Cross-compatible | ✅ Python compress → C++ decompress works | ✅ C++ compress → Python decompress works |

---

## 2. Every Core File — Deep Explanation

### Python Core: `python/dlc/`

---

#### `codec.py` — High-Level API

**Responsibility:** The user-facing entry point. Wraps the entire pipeline into two simple calls.

**Key Functions:**

| Function | In | Out | Description |
|---|---|---|---|
| `compress(data, output_path, precision_bits, chunk_size, num_workers)` | `np.ndarray` of float64, file path | `.dlc` file on disk | Creates a `DLCHeader`, calls `compress_parallel()`, writes via `write_dlc_file()` |
| `decompress(input_path)` | Path to `.dlc` file | `np.ndarray` of float64 | Reads via `read_dlc_file()`, calls `decompress_payload()` |

**Talks to:** `format.py` (header/file I/O), `pipeline.py` (compression/decompression)

---

#### `pipeline.py` — Parallel Block Processing

**Responsibility:** Splits data into blocks, dispatches to thread pool, assembles results in order. Contains the core per-block logic.

**Key Functions:**

| Function | In | Out | Description |
|---|---|---|---|
| `_process_block(block, precision_bits)` | `np.ndarray` of one block | `bytes` (packed window blocks) | Segments → models → encodes → packs each window |
| `compress_parallel(data, precision_bits, chunk_size, num_workers)` | Full signal array | `bytes` (raw payload) | Splits into 100K-sample blocks, dispatches to `ThreadPoolExecutor` |
| `decompress_payload(payload, precision_bits)` | Raw payload bytes | `np.ndarray` | Unpacks window blocks, reconstructs all 6 model types, dequantizes |

**Talks to:** `windowing.py`, `models.py`, `encoders.py`, `format.py`

**Compression path inside `_process_block`:**
1. `segment_block(block)` → list of windows
2. For each window: `select_model(window)` → `(model_id, params, residuals)`
3. If XOR model: `encode_xor_residuals(xor_residuals)` → encoded bytes
4. If analytical: `select_precision(residuals)` → `quantize(residuals, precision)` → `trial_encode(quantized)` → encoded bytes
5. `pack_window_block(meta, encoded_data)` → bytes

**Decompression path inside `decompress_payload`:**
1. Loop: `unpack_window_block(payload, offset)` → `(meta, encoded_data, consumed)`
2. Switch on `model_id`:
   - **0 (Linear):** predicted = `m*x + c`, decode residuals, add back
   - **1 (Quadratic):** predicted = `a*x² + b*x + c`, decode residuals, add back
   - **2 (XOR-Delta):** `decode_xor_residuals()`, `reconstruct_xor_delta(anchor, xor_res)`
   - **3 (Constant):** predicted = `c` for all samples, decode residuals, add back
   - **4 (Sinusoidal):** predicted = `A*sin(w*x + phi) + dc`, decode residuals, add back
   - **5 (Pred-XOR):** `decode_xor_residuals()`, `reconstruct_pred_xor(anchor0, anchor1, xor_res)`

---

#### `windowing.py` — Adaptive Segmentation

**Responsibility:** Split a block into variable-length windows where each window has homogeneous statistical properties.

**Key Functions:**

| Function | In | Out | Description |
|---|---|---|---|
| `segment_block(block, min_w, max_w)` | `np.ndarray` (one block) | `List[np.ndarray]` (list of windows) | Variance-guided greedy expansion + binary search |
| `_variance(data)` | array | float | Population variance (ddof=0) |
| `_max_window_for_block(block_len)` | int | int | Returns 512/1024/4096 based on block size |

**Algorithm:**
1. Compute global variance of entire block
2. If variance ≈ 0 → emit entire block as single constant window
3. Start with `MIN_WINDOW = 16`, expand by doubling while `variance ≤ threshold`
4. Threshold = `max(initial_var × 10.0, global_var × 0.10)`
5. Stagnation guard: stop if variance jumps > 1.5× between doublings
6. Binary search for exact split point

**Talks to:** Nobody — pure algorithm, called by `pipeline.py`

---

#### `models.py` — Predictive Models

**Responsibility:** Fit 6 different mathematical models to each window and select the best one.

**Key Functions:**

| Function | In | Out | Description |
|---|---|---|---|
| `fit_linear(y)` | window array | `((m,c), residuals, ssr)` | Closed-form least squares |
| `fit_quadratic(y)` | window array | `((a,b,c), residuals, ssr)` | `np.polyfit` degree 2 |
| `fit_xor_delta(y)` | window array | `(anchor, xor_residuals)` | Bitwise XOR of consecutive float64 |
| `fit_constant(y)` | window array | `((c,), residuals, ssr)` | Mean value |
| `fit_sinusoidal(y)` | window array | `((A,w,phi,dc), residuals, ssr)` | FFT seed + Gauss-Newton |
| `fit_pred_xor(y)` | window array | `((a0,a1), xor_residuals)` | Predicted XOR: `bits(yᵢ) ⊕ bits(2yᵢ₋₁ - yᵢ₋₂)` |
| `select_model(window)` | window array | `(model_id, params, residuals)` | Evaluates all 6, applies preference rules |
| `reconstruct_xor_delta(anchor, xor_res)` | anchor + XOR list | `np.ndarray` | Reconstructs signal from XOR-Delta |
| `reconstruct_pred_xor(a0, a1, xor_res)` | two anchors + XOR list | `np.ndarray` | Reconstructs signal from Pred-XOR |

**Selection rules:**
1. Non-finite values → force XOR-Delta
2. If Constant SSR is within 1% of best → prefer Constant
3. Pick better XOR variant (XOR-Delta vs Pred-XOR)
4. If best XOR SSR is within 2% of best analytical → prefer XOR
5. Otherwise, pick analytical with lowest SSR

**Talks to:** Nobody — pure algorithm, called by `pipeline.py`

---

#### `encoders.py` — Residual Encoding

**Responsibility:** Quantize float residuals to integers, then encode with 5 competing strategies. The smallest output wins.

**Key Functions:**

| Function | In | Out | Description |
|---|---|---|---|
| `quantize(residuals, precision_bits)` | float array, int | `List[int]` | `round(r × 2^p)` |
| `dequantize(quantized, precision_bits)` | int list, int | `np.ndarray` | `q / 2^p` |
| `select_precision(residuals, requested)` | float array, int | int | Adaptive 8-20 bits based on max |residual| |
| `trial_encode(quantized)` | `List[int]` | `(encoding_id, bytes)` | Tries all 5 strategies, returns smallest |
| `decode_residuals(encoding_id, data, count)` | id, bytes, int | `List[int]` | Dispatches to correct decoder |
| `encode_xor_residuals(xor_res)` | `List[int]` (uint64) | `bytes` | Raw or delta-of-XOR encoding |
| `decode_xor_residuals(data, count)` | bytes, int | `List[int]` | Prefix byte 0x00=raw, 0x01=delta |

**Encoding/Decoding pairs:**

| ID | Encoder | Decoder |
|---|---|---|
| 0 | `encode_zigzag_varint()` | `decode_zigzag_varint()` |
| 1 | `encode_bitpack()` | `decode_bitpack()` |
| 2 | `encode_delta_of_residuals()` | `decode_delta_of_residuals()` |
| 3 | `encode_outlier_sep()` | `decode_outlier_sep()` |
| 4 | `encode_rle()` | `decode_rle()` |
| 5 | `encode_dod()` | `decode_dod()` |

**Block-level DoD Fast Path:** Before windowed compression, the pipeline evaluates encoding the entire block as a single Constant(mean) window with DoD encoding. If this produces smaller output than windowed compression, it is used directly — eliminating per-window metadata overhead.

**Talks to:** Nobody — pure algorithm, called by `pipeline.py`

---

#### `format.py` — Binary File Format

**Responsibility:** Serialize/deserialize the `.dlc` binary format. Defines the header, footer, and window block structures.

**Key Functions:**

| Function | In | Out | Description |
|---|---|---|---|
| `pack_header(header)` | `DLCHeader` | `bytes` (64B) | Serializes header to LE bytes |
| `unpack_header(buf)` | `bytes` | `DLCHeader` | Deserializes, validates magic bytes |
| `pack_footer(crc32_val)` | int | `bytes` (8B) | Footer magic + CRC-32 |
| `unpack_footer(buf)` | `bytes` | `DLCFooter` | Validates footer magic |
| `pack_window_block(meta, encoded_residuals)` | `WindowBlockMeta`, bytes | `bytes` | Window header (7B) + params + residuals |
| `unpack_window_block(buf, offset)` | bytes, int | `(meta, encoded_residuals, consumed)` | Reads one window block |
| `write_dlc_file(filepath, header, payload)` | path, header, raw payload | `.dlc` file | Header + zlib(payload) + footer |
| `read_dlc_file(filepath)` | path | `(header, payload)` | Reads + validates + decompresses |

**Data classes:**
- `DLCHeader`: 64 bytes — magic, version, precision, chunk_size, total_samples, reserved
- `DLCFooter`: 8 bytes — magic, CRC-32
- `WindowBlockMeta`: window_length, model_id, encoding_id, precision_bits, model_params

**Talks to:** `zlib` (standard library), called by `codec.py` and `pipeline.py`

---

#### `cli.py` — Command-Line Interface

**Responsibility:** Provides `dlc compress` and `dlc decompress` commands.

**Flags:**

| Command | Flag | Type | Default | Description |
|---|---|---|---|---|
| `compress` | `-i` / `--input` | str | required | Input `.bin` file (raw float64 LE) |
| | `-o` / `--output` | str | required | Output `.dlc` file |
| | `--precision` | int | 12 | Quantization bits |
| | `--chunk-size` | int | 100000 | Block size for parallel dispatch |
| | `--workers` | int | auto | Thread count |
| `decompress` | `-i` / `--input` | str | required | Input `.dlc` file |
| | `-o` / `--output` | str | required | Output `.bin` file |

**Talks to:** `codec.py`

---

### C++ Engine: `cpp/`

Each module has a **header** (`.hpp` = the contract/interface) and a **source** (`.cpp` = the implementation).

---

#### `codec.hpp` / `codec.cpp` — High-Level C++ API

**Contract (`codec.hpp`):**
```cpp
struct CompressOptions {
  int precision_bits = 12;
  uint32_t chunk_size = 100000;
  int num_workers = 0; // 0 = auto
};
void compress(const double *data, size_t n, const std::string &path, const CompressOptions &opts);
std::vector<double> decompress(const std::string &path);
```

**Implementation (`codec.cpp`):**
- `compress()`: Builds `DLCHeader` → calls `compress_parallel()` → computes CRC-32 → `zlib_compress()` → writes Header + uncomp_size + zlib_data + Footer
- `decompress()`: Reads file → parses Header → reads uncomp_size → `zlib_decompress()` → verifies CRC-32 → calls `decompress_payload()`

**Talks to:** `format.hpp`, `pipeline.hpp`, `<zlib.h>`

---

#### `pipeline.hpp` / `pipeline.cpp` — Parallel Pipeline

**Contract (`pipeline.hpp`):**
```cpp
std::vector<uint8_t> process_block(const double *data, size_t n, const DLCConfig &config);
std::vector<uint8_t> compress_parallel(const double *data, size_t n, const DLCConfig &config);
std::vector<double> decompress_payload(const uint8_t *data, size_t len, const DLCConfig &config);
```

**Implementation:** Mirrors Python `pipeline.py` exactly. Uses `std::async(std::launch::async, ...)` for true parallel block dispatch. `decompress_payload()` explicitly handles all 6 model types including Sinusoidal (`A*sin(w*x+phi)+dc`) and Pred-XOR reconstruction.

---

#### `windowing.hpp` / `windowing.cpp` — Adaptive Segmentation

**Contract:** `std::vector<Span> segment_block(const double* data, size_t n, size_t min_w=16, size_t max_w=1024);`

**Implementation:** Same variance-guided algorithm. Adaptive MAX_WINDOW (512/1024/4096), stagnation guard, constant-block fast-path.

---

#### `models.hpp` / `models.cpp` — 6 Predictive Models

**Contract:**
```cpp
enum class ModelType : uint8_t { Linear=0, Quadratic=1, XorDelta=2, Constant=3, Sinusoidal=4, PredXor=5 };
struct ModelResult { ModelType model; vector<double> params; vector<double> residuals; vector<uint64_t> xor_residuals; double ssr; };
ModelResult select_model(const double *y, size_t n);
```

**Key C++ detail:** Sinusoidal fitting uses a custom **O(n) DFT** (no FFTW dependency) for frequency estimation, then a handwritten 4×4 Gauss-Newton solver using Gaussian elimination with partial pivoting.

---

#### `encoders.hpp` / `encoders.cpp` — 5 Encoding Strategies

**Contract:** Same as Python — `quantize()`, `dequantize()`, `trial_encode()`, `decode_residuals()`, plus `encode_rle()`/`decode_rle()`, `encode_xor_residuals()`/`decode_xor_residuals()`, `select_precision()`.

---

#### `format.hpp` / `format.cpp` — Binary Format I/O

**Contract:** Window block is now 7 bytes (was 6): `window_length(4) + model_id(1) + encoding_id(1) + precision_bits(1)`. Added model IDs 3, 4, 5 to `model_param_count()`.

---

#### `main.cpp` — CLI Entry Point

**Flags:** Identical to Python CLI (`-i`, `-o`, `--precision`, `--chunk-size`, `--workers`).

**Usage:**
```bash
./dlc compress   -i input.bin -o output.dlc --precision 12 --chunk-size 100000 --workers 4
./dlc decompress -i input.dlc -o output.bin
```

---

### Benchmark Scripts: `python/benchmarks/`

---

#### `compare_all.py` — Terminal Comparison

**What it does:** Finds every `.bin` file in `test_data/`, compresses each with GZIP, ZLIB, and DLC C++, prints a formatted table.

**Output columns:** Dataset name, Samples, Raw size (MB), GZIP ratio + time + MB/s, ZLIB ratio + time + MB/s, DLC ratio + time + MB/s, Winner.

**Key functions:** `compress_gzip()`, `compress_zlib()`, `compress_dlc()` (subprocess to C++ binary), `find_cpp_binary()`, `main()`.

---

#### `compare_industry.py` — Industry Standard Comparison

**What it does:** Runs Gorilla (Facebook), Delta-of-Delta (InfluxDB), RLE, GZIP, ZLIB, and DLC on all datasets. Includes **round-trip error measurement** for every technique.

**Output:** Compression ratio table + Error Analysis table showing max reconstruction error per technique.

---

#### `compare_industry_fair.py` — Fair 16-bit Comparison

**What it does:** Forces DoD to use 16-bit quantization (same as DLC). This is the definitive "apples-to-apples" benchmark. DLC wins on **80% of datasets** at equal precision.

**Output:** Fair ratio table + error analysis proving DLC is both more accurate AND more efficient.

---

#### `bench_cpp.py` — C++ Throughput

Uses `subprocess.run()` to call the compiled C++ binary. Measures compress/decompress times and throughput in MB/s across different datasets.

---

#### `bench_throughput.py` — Python Throughput

Runs the Python engine with 1, 2, 4, 8 worker threads on 4 datasets (sine, noisy sine, random walk, ramp). Demonstrates the GIL bottleneck: throughput doesn't scale with threads.

---

#### `visualize_final_dashboards.py` / `visualize_real_data.py`

Matplotlib scripts that generate charts for the capstone presentation:
- `final_ratios.png`: Grouped bar chart (log Y-axis) comparing GZIP/ZLIB/DLC ratios
- `final_throughput.png`: Python vs C++ throughput scaling chart
- `aws_cpu_overlay.png`: Original signal + DLC prediction overlay
- `aws_cpu_residuals.png`: Residual Laplacian histogram

---

### Data Generation

#### `test_data/generate_test_data.py`
Generates 17+ synthetic and real-world `.bin` datasets: sine, ramp, random walk, noisy sine, step functions, edge cases, and NAB datasets.

#### `test_data/download_massive_data.py`
Downloads the **Jena Climate Dataset** (420K+ continuous temperature/pressure readings) from Google's servers for large-scale benchmarking.

#### `test_data/download_large_datasets.py`
Fetches and generates large-scale datasets (100MB-730MB+) for stress testing: massive sensor simulations (3M+ samples), multi-frequency composite signals, random walk data, and industrial-grade test vectors.

---

### App: `python/app.py` — Streamlit Dashboard

**Sections (5 tabs):**
1. **Signal & Compress:** Dataset selector, signal preview (Plotly interactive), DLC vs GZIP quick comparison
2. **Industry Comparison:** Gorilla, Delta-of-Delta (16-bit fair), RLE, GZIP, ZLIB vs DLC on selected dataset with interactive bar charts, size comparison, and fairness note
3. **Round-Trip Verify:** Compress → Decompress → Overlay original vs recovered, error histogram
4. **Batch All Datasets:** Run all techniques on every `.bin` file. Three sections:
   - Standard compression ratios table (DoD at 12-bit)
   - **Fair comparison** table (DoD at 16-bit = equal precision to DLC)
   - **Round-trip error analysis** table showing max reconstruction error per technique per dataset
5. **How It Works:** Architecture diagram, 6 models, 6 encoders explained

**Launch:** `streamlit run python/app.py`

---

## 3. The Full Data Flow — Step by Step

### Compression: `float64[]` → `.dlc`

```
Input: [23.456, 23.457, 23.459, 23.460, ...]  (raw float64 array)
```

**Step 1: `codec.compress(data, "output.dlc")`**
- Creates `DLCHeader(precision_bits=12, chunk_size=100000, total_samples=len(data))`
- Calls `compress_parallel(data, 12, 100000, None)`

**Step 2: `pipeline.compress_parallel()`**
- Splits data into blocks of 100,000 samples
- Dispatches each to `ThreadPoolExecutor`

**Step 3: `pipeline._process_block(block)` — per block:**

**Step 3a: `windowing.segment_block(block)`**
```
Block: [23.456, 23.457, ..., 23.890, 24.100, 24.102, ...]
                          ↓
Windows: [[23.456..23.890], [24.100..24.200], ...]
          window 1 (256)    window 2 (128)
```

**Step 3b: `models.select_model(window)` — per window:**
```
Window: [23.456, 23.457, 23.459, 23.460]

Linear fit: y = 0.00133x + 23.456
  params = [0.00133, 23.456]
  residuals = [0.0, -0.0003, 0.0017, 0.0007]
  SSR = 0.0000037

Selected: model_id=0 (Linear)
```

**Step 3c: `encoders.select_precision(residuals, 12)` → 12**

**Step 3d: `encoders.quantize(residuals, 12)` → `[0, -1, 7, 3]`**
```
0.0 × 4096 = 0    →  0
-0.0003 × 4096 = -1.23  →  -1
0.0017 × 4096 = 6.96    →  7
0.0007 × 4096 = 2.87    →  3
```

**Step 3e: `encoders.trial_encode([0, -1, 7, 3])`**
```
Strategy 0 (Zigzag+Varint): [0, 1, 14, 6] → [0x00, 0x01, 0x0E, 0x06] = 4 bytes ← WINNER
Strategy 1 (Bitpack, 4-bit): 1 + 2 = 3 bytes
Strategy 4 (RLE): 4 runs × 2 = 8 bytes
→ Returns: (encoding_id=1, 3 bytes)
```

**Step 3f: `format.pack_window_block(meta, encoded_bytes)`**
```
[window_length=4][model_id=0][encoding_id=1][precision=12][m as f64][c as f64][res_len=3][encoded...]
     4 bytes       1 byte      1 byte          1 byte      8 bytes   8 bytes   4 bytes   3 bytes
                                                                                    = 30 bytes total
```

**Step 4: `format.write_dlc_file("output.dlc", header, payload)`**
```
         ┌───────────────────────┐
Offset 0 │ Header (64 bytes)     │
      64 │ Uncomp size (4 bytes) │
      68 │ Zlib(payload) (N B)   │
  68 + N │ Footer (8 bytes)      │
         └───────────────────────┘
```

### Decompression: `.dlc` → `float64[]`

**Step 1: `codec.decompress("output.dlc")`**
- Calls `read_dlc_file()` → validates magic, inflates zlib, verifies CRC-32

**Step 2: `pipeline.decompress_payload(payload, 12)`**
- Loops through packed window blocks

**Step 3: Per window block:**
- `unpack_window_block()` → `meta(model_id=0, precision=12, params=[0.00133, 23.456]), encoded_data`
- `decode_residuals(1, encoded_data, 4)` → `[0, -1, 7, 3]`
- `dequantize([0, -1, 7, 3], 12)` → `[0.0, -0.000244, 0.001709, 0.000732]`
- predicted = `[23.456, 23.4573, 23.4587, 23.4600]`
- result = predicted + residuals = `[23.456, 23.4571, 23.4604, 23.4607]`

---

## 4. All 6 Models — Plain English

### Model 0: Linear (`y = mx + c`)

**Targets:** Steady ramps, slow trends, monotonically increasing/decreasing data.

**How it fits:** Closed-form least squares — two summations over the data, solve a 2-variable system. O(n).

**Example:**
```
Input:  [10.0, 12.0, 14.0, 16.0]
Fit:    y = 2.0x + 10.0
Params: [m=2.0, c=10.0]  →  2 × float64 stored  →  model_id = 0
Pred:   [10.0, 12.0, 14.0, 16.0]
Resid:  [0.0, 0.0, 0.0, 0.0]  →  perfect fit!
```

### Model 1: Quadratic (`y = ax² + bx + c`)

**Targets:** Parabolic curves, acceleration/deceleration profiles.

**How it fits:** `np.polyfit(x, y, 2)` — 3×3 normal equations. C++ uses manual Cramer's rule. O(n).

**Example:**
```
Input:  [0.0, 1.0, 4.0, 9.0, 16.0]
Fit:    y = 1.0x² + 0.0x + 0.0
Params: [a=1.0, b=0.0, c=0.0]  →  3 × float64 stored  →  model_id = 1
Resid:  [0.0, 0.0, 0.0, 0.0, 0.0]
```

### Model 2: XOR-Delta (`bits(yᵢ) ⊕ bits(yᵢ₋₁)`)

**Targets:** Random/noisy data, non-finite values (NaN, Inf).

**How it works:** Store first value as anchor. For each subsequent pair, XOR their IEEE 754 bit patterns. Similar consecutive values produce XOR results with many leading zeros.

**Example:**
```
Input:  [1.0, 1.0000001]
Anchor: 1.0    (bits: 0x3FF0000000000000)
y[1] bits:     0x3FF00000000006A8
XOR residual:  0x00000000000006A8  →  small!  →  model_id = 2
Stored: [anchor] = 1 × float64
```

### Model 3: Constant (`y = c`)

**Targets:** Step functions, flat signal regions, piecewise-constant data.

**How it fits:** `c = mean(window)`. Simplest possible model — 1 parameter.

**Example:**
```
Input:  [5.0, 5.0, 5.0, 5.0, 5.0]
Fit:    y = 5.0
Params: [c=5.0]  →  1 × float64 stored  →  model_id = 3
Resid:  [0.0, 0.0, 0.0, 0.0, 0.0]
```

**Why it matters:** Constant windows + RLE encoding = extreme compression. The 5 residuals are all 0, RLE encodes as `(value=0, count=5)` = 3 bytes.

### Model 4: Sinusoidal (`y = A·sin(ωx + φ) + dc`)

**Targets:** Periodic sensor data — AC current, vibration, temperature cycling.

**How it fits:**
1. FFT (Python) or O(n) DFT (C++) to find dominant frequency
2. Seed: `A = (max-min)/2`, `dc = mean`, `ω = 2π·f₀`
3. Gauss-Newton refinement: solve 4×4 `JᵀJ · δ = Jᵀr` iteratively

**Example:**
```
Input:  [0.0, 0.707, 1.0, 0.707, 0.0, -0.707, ...]  (sine wave)
Fit:    y = 1.0·sin(π/2·x + 0.0) + 0.0
Params: [A=1.0, ω=1.5708, φ=0.0, dc=0.0]  →  4 × float64  →  model_id = 4
Resid:  [~0.0, ~0.0, ...]  →  near-zero for clean sine
```

**Minimum window:** 32 samples (need enough data for FFT peak detection).

### Model 5: Pred-XOR (Predictive XOR)

**Targets:** Trending noisy data — the prediction makes XOR residuals smaller.

**How it works:** Store two anchors (`y[0]`, `y[1]`). For `i ≥ 2`, predict `ŷᵢ = 2·yᵢ₋₁ - yᵢ₋₂` (linear extrapolation), then XOR `bits(yᵢ) ⊕ bits(ŷᵢ)`. If the signal is approximately linear, predictions are close → XOR values are small.

**Example:**
```
Input:     [1.0, 2.0, 3.0, 4.0]
Predicted: [—,   —,   3.0, 4.0]  (from 2*y[i-1] - y[i-2])
XOR:       [bits(3.0) ⊕ bits(3.0), bits(4.0) ⊕ bits(4.0)] = [0, 0]  →  perfect!
Params:    [anchor0=1.0, anchor1=2.0]  →  2 × float64  →  model_id = 5
```

---

## 5. All 5 Encoding Strategies — Plain English

### Strategy 0: Zigzag + Varint (encoding_id = 0)

**What it does:** Maps signed integers to unsigned via zigzag `(n<<1) ^ (n>>63)`, then encodes each unsigned integer as a variable-length byte sequence (7 data bits per byte, MSB = continuation flag).

**Example:**
```
Input:   [-1, 0, 3, -2]
Zigzag:  [1, 0, 6, 3]
Varint:  [0x01, 0x00, 0x06, 0x03]  →  4 bytes (each fits in 1 byte)
```

**Wins when:** Residuals are small and varied (no extreme outliers, no long runs of same value).

### Strategy 1: Fixed-Width Bitpack (encoding_id = 1)

**What it does:** Finds the maximum zigzag value, determines its bit width, then packs all values at that fixed width. Header: 1 byte (bit_width).

**Example:**
```
Input:   [0, 1, -1, 2]
Zigzag:  [0, 2, 1, 4]  →  max = 4  →  bit_width = 3
Pack:    [000, 010, 001, 100] = 12 bits → 2 bytes packed
Total:   1 (header) + 2 (body) = 3 bytes
```

**Wins when:** Values have a narrow, uniform range (no outliers pulling the bit width up).

### Strategy 2: Delta-of-Residuals (encoding_id = 2)

**What it does:** Computes first-order differences between consecutive quantized values, then zigzag+varint encodes the deltas. First value stored directly.

**Example:**
```
Input:   [10, 11, 13, 12]
Deltas:  [10, 1, 2, -1]
Zigzag+Varint encodes deltas (which are smaller than originals)
```

**Wins when:** Residuals change slowly — consecutive residuals are similar, so their differences are tiny.

### Strategy 3: MAD Outlier Separation (encoding_id = 3)

**What it does:**
1. Compute median and Median Absolute Deviation (MAD) of values
2. Values beyond `4.5 × MAD` → classified as outliers
3. Outliers stored as `(uint32 index, zigzag-varint value)` pairs
4. Inliers (with outlier positions zeroed) bitpacked at reduced bit width

**Example:**
```
Input:   [0, 1, -1, 0, 500, 1, 0, -1]  (500 is an outlier)
Median:  0, MAD: 1.0, Threshold: 4.5
Outliers: [(index=4, value=500)]
Inliers:  [0, 1, -1, 0, 0, 1, 0, -1]  →  bitpacked at 2-bit width
```

**Wins when:** Signal has occasional spikes in otherwise smooth data (e.g., sensor glitches).

### Strategy 4: RLE + Varint (encoding_id = 4)

**What it does:** Run-length encoding. Groups consecutive identical values into `(value, count)` pairs, encoded as zigzag-varint.

**Example:**
```
Input:   [0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
Runs:    [(0, 5), (1, 2), (0, 3)]
Encode:  varint(3 runs) + [(zz(0), 5), (zz(1), 2), (zz(0), 3)]
Total:   ~7 bytes (vs 10 bytes raw)
```

**Wins when:** Constant model produces all-zero residuals, or step functions produce long runs of same value. The Constant model + RLE combination can compress 1000 identical samples to ~5 bytes.

---

## 6. How to Run Everything — Exact Commands

### Setup

```bash
# Create virtual environment
cd capstonev2
python -m venv .venv
.venv\Scripts\Activate.ps1    # Windows
source .venv/bin/activate      # Linux/Mac

# Install Python package
cd python
pip install -e .
pip install numpy matplotlib streamlit pandas
```

### C++ Build (Windows MinGW/MSYS2)
```bash
cd capstonev2/cpp/build
cmake .. -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
# Output: cpp/build/dlc.exe
```

### C++ Build (Linux)
```bash
cd capstonev2/cpp/build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
# Output: cpp/build/dlc
```

### Generate Test Data
```bash
cd capstonev2
python test_data/generate_test_data.py       # 17+ synthetic + NAB datasets
python test_data/download_massive_data.py    # Jena Climate 420K samples
```

### Compress & Decompress

**Python CLI:**
```bash
cd capstonev2/python
python -m dlc compress -i ../test_data/sine_1m.bin -o output.dlc --precision 12 --chunk-size 100000 --workers 4
python -m dlc decompress -i output.dlc -o recovered.bin
```

**C++ CLI:**
```bash
cd capstonev2/cpp/build
.\dlc.exe compress -i ..\..\test_data\sine_1m.bin -o output.dlc --precision 12 --chunk-size 100000 --workers 4
.\dlc.exe decompress -i output.dlc -o recovered.bin
```

### Run Benchmarks
```bash
cd capstonev2

# Full comparison table (GZIP vs ZLIB vs DLC on ALL datasets)
python python/benchmarks/compare_all.py

# Python-only throughput (1/2/4/8 threads)
python python/benchmarks/bench_throughput.py

# C++ throughput
python python/benchmarks/bench_cpp.py
```

**`compare_all.py` output columns explained:**

| Column | Meaning |
|---|---|
| Dataset | Name of the `.bin` file (minus extension) |
| Samples | Number of float64 values in the file |
| Raw | Raw file size in MB |
| GZIP | Compression ratio (`raw_size / compressed_size`) |
| Time | Wall-clock time to compress |
| MB/s | Throughput (`raw_size / time`) |
| ZLIB | Same as GZIP columns |
| DLC | Same, using C++ engine via subprocess |
| Winner | Which method achieved the highest ratio. `>> DLC` means DLC won |

### Run Tests
```bash
# Python tests
cd capstonev2/python
pytest tests/ -v

# C++ tests
cd capstonev2/cpp/build
ctest --output-on-failure
```

### Visualizations
```bash
cd capstonev2
python python/benchmarks/visualize_real_data.py           # AWS CPU overlay + residuals
python python/benchmarks/visualize_final_dashboards.py     # Ratio + throughput charts

# Interactive Streamlit dashboard
streamlit run python/app.py
```

---

## 7. Benchmark Results — Row by Row

### Understanding the Output

- **Ratio `5.4×`:** The raw data is 5.4 times larger than the compressed file. A 1 MB file compresses to ~185 KB.
- **`>> DLC`:** DLC won (highest ratio). The `>>` prefix makes DLC wins visually distinct.
- **`GZIP` / `ZLIB` (no `>>`):** GZIP or ZLIB won.
- **3.94× overall:** If you concatenated ALL datasets into one giant file, the total would compress 3.94× vs raw.

### Why DLC Wins or Loses

| Dataset | Typical Winner | Why |
|---|---|---|
| `sine_1m` | DLC or ZLIB | Smooth periodic — Linear/Sinusoidal models fit well → tiny residuals → varint encodes efficiently |
| `ramp_1m` | ZLIB | Pure linear ramp — ZLIB's Huffman perfectly encodes the repetitive byte patterns in a linear sequence |
| `real_cpu_4k` | DLC | Real AWS CPU data — mathematical model captures trends GZIP can't see |
| `real_machine_temp_22k` | DLC | Temperature data — smooth, slow-varying → Linear/Constant windows compress well |
| `real_jena_temperature_420k` | DLC or GZIP | Large real dataset — adaptive windowing gives DLC the edge on structure |
| `real_jena_pressure_420k` | DLC or GZIP | Similar to temperature — smooth trends with occasional noise |
| `step_10k` | **ZLIB** (permanent) | Pure step function — ZLIB sees long runs of identical bytes. DLC's Constant model + RLE helps but can't beat byte-level dictionary compression on synthetic binary patterns |
| `edge_cases_100` | ZLIB | Only 100 samples — DLC's 64-byte header dominates the output. GZIP's header is 20 bytes |
| `noisy_ramp_100k` | DLC or Tie | Noisy ramp — mathematical model captures the trend, quantization discards noise |
| `synthetic_random_walk_1m` | DLC | Random walk — Pred-XOR or Linear on each window segment captures local trends |
| `synthetic_noisy_sine_1m` | DLC | Multi-frequency sine — Sinusoidal model fits dominant frequency per window |

### Why `step_10k` is a Permanent ZLIB Win

A step function produces values like `[5.0, 5.0, ..., 5.0, 10.0, 10.0, ..., 10.0]`. Each `5.0` is the exact same 8 bytes: `0x4014000000000000`. ZLIB's LZ77 sees this as a single 8-byte pattern repeated thousands of times — perfect dictionary match, negligible overhead.

DLC's Constant model + RLE also handles this well, but DLC adds a 64-byte file header, per-window metadata (7 bytes + params), and zlib post-compression. On a tiny 10K-sample file (80 KB raw), this overhead matters. On larger step datasets, DLC would close the gap.

**This is acceptable** because DLC is designed for real-world sensor data, not synthetic step functions.

---

## 8. The `.dlc` Binary Format — Byte by Byte

### Complete File Layout

```
Offset   Size     Field                           Notes
───────  ───────  ──────────────────────────────  ──────────────
0x00     4        Magic bytes                     b'DLC\x02' (identifies file type + version)
0x04     2        Major version (uint16 LE)       0
0x06     2        Minor version (uint16 LE)       1
0x08     2        Precision bits (uint16 LE)      12 (file-level default)
0x0A     4        Chunk size (uint32 LE)          100000
0x0E     8        Total samples (uint64 LE)       Number of float64 values
0x16     42       Reserved                        Zero-padded for future use
0x40     4        Uncompressed payload size        uint32 LE — needed for zlib inflate
0x44     N        Zlib-compressed payload          Window blocks packed sequentially
0x44+N   4        Footer magic                    0xED 0xDC 0xBA 0x01
0x44+N+4 4        CRC-32                          zlib.crc32 of UNCOMPRESSED payload
```

### Window Block Layout (inside the uncompressed payload)

```
Offset   Size     Field
───────  ───────  ──────────────────────
+0       4        window_length (uint32 LE)     Number of samples in this window
+4       1        model_id (uint8)               0-5 (see model table)
+5       1        encoding_id (uint8)            0-4 (see encoding table)
+6       1        precision_bits (uint8)         Per-window adaptive precision (8-20)
+7       N×8      model_params                   N = param count for this model_id:
                                                   model 0: 2 (m, c)
                                                   model 1: 3 (a, b, c)
                                                   model 2: 1 (anchor)
                                                   model 3: 1 (mean)
                                                   model 4: 4 (A, ω, φ, dc)
                                                   model 5: 2 (anchor0, anchor1)
+7+N×8   4        encoded_residuals_len (uint32 LE)
+11+N×8  M        encoded_residuals (raw bytes)  Encoding-specific format
```

### Why Little-Endian?

All modern x86/x64 and ARM processors use little-endian byte ordering natively. Using LE throughout means no byte-swapping is needed on the vast majority of hardware. Both Python's `struct.pack('<d', ...)` and C++'s native `memcpy` produce identical bytes.

### Why the Footer Exists

The CRC-32 is computed on the **uncompressed payload** (before zlib deflate). This catches:
- Corrupted zlib data (inflate would produce wrong bytes)
- Truncated files
- Bit-rot in storage

The footer magic (`0xED 0xDC 0xBA 0x01`) serves as a sentinel — if the file is truncated, the footer magic won't be present and parsing fails immediately.

---

## 9. Common Errors & Fixes

### C++ Build Errors on Windows MinGW

**Error:** `zlib.h not found`
```
Fix: Install zlib via MSYS2:
  pacman -S mingw-w64-ucrt-x86_64-zlib
```

**Error:** `dlc/models.hpp not found`
```
Fix: Ensure cmake is run from the build/ directory WITH the parent reference:
  cd cpp/build
  cmake .. -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
```

**Error:** `'std::thread' not found` or similar C++17 errors
```
Fix: Ensure CMakeLists.txt sets CMAKE_CXX_STANDARD to 17:
  set(CMAKE_CXX_STANDARD 17)
```

### Precision Bound Violations (error > 8×10⁻⁶)

**Cause:** Usually happens with very large float64 values where 12-bit precision is insufficient.

**Fix:** Increase precision bits:
```bash
dlc compress -i data.bin -o out.dlc --precision 16
```

Or the adaptive precision system may already handle this — check `select_precision()` in `encoders.py`.

### Cross-Language CRC Mismatch

**Cause:** Python uses `zlib.crc32(data) & 0xFFFFFFFF` (unsigned 32-bit). C++ uses `crc32()` from `<zlib.h>`. They produce identical results IF the input bytes are identical.

**Fix:** Verify that:
1. Both engines produce the same uncompressed payload (byte-for-byte)
2. The `& 0xFFFFFFFF` mask is applied in Python (without it, Python returns a signed 32-bit value on some platforms)

### Model ID Not Recognized

**Error:** `Unknown model_id: 3` (or 4 or 5)

**Cause:** Old C++ binary that doesn't know about the new models.

**Fix:** Rebuild the C++ engine:
```bash
cd cpp/build
cmake --build . --config Release
```

### `test_cross_lang.py` Failures

**Cause:** Python and C++ format versions are out of sync.

**Fix:** Ensure both engines have the same:
- Window block size (7 bytes for header, not 6)
- Model param counts in `_MODEL_PARAM_COUNTS` / `model_param_count()`
- Encoding IDs (0-4, not 0-3)

---

## 10. Glossary

| Term | Definition |
|---|---|
| **Entropy Trap** | The phenomenon where float64 sensor data appears high-entropy at the byte level (defeating LZ77) despite being mathematically structured |
| **IEEE 754** | The standard defining float64 as 1 sign bit + 11 exponent bits + 52 mantissa bits |
| **float64** | 8-byte double-precision floating-point number |
| **Laplacian distribution** | A probability distribution peaked at zero, shaped like `∧`. Quantized residuals from good model fits follow this distribution |
| **Welford's algorithm** | Numerically stable single-pass algorithm for computing variance |
| **Zigzag encoding** | Maps signed integers to unsigned: 0→0, -1→1, 1→2, -2→3, ... Concentrates small magnitudes near zero |
| **Varint** | Variable-length integer encoding: 7 data bits per byte, MSB = "more bytes follow" flag |
| **SSR** | Sum of Squared Residuals — `Σ(yᵢ - ŷᵢ)²`. Lower SSR = better model fit |
| **MAD** | Median Absolute Deviation — robust outlier detection metric: `median(|xᵢ - median(x)|)` |
| **Quantization** | Rounding continuous values to discrete multiples of a step size (`2⁻ᵖ`) |
| **Near-lossless** | Reconstruction error is bounded but not zero. Every sample recovers within `< 8×10⁻⁶` |
| **XOR-Delta** | Encoding where consecutive float64 bit patterns are XORed. Similar values produce small XOR results |
| **CRC-32** | 32-bit Cyclic Redundancy Check — polynomial-based error detection checksum |
| **Deflate** | The compression algorithm inside zlib/gzip. Combines LZ77 + Huffman coding |
| **Residual** | The difference between the actual value and the model's prediction: `r = y - ŷ` |
| **Precision bits** | Number of bits used for quantization. 12 bits → step size = `2⁻¹² ≈ 0.000244` |
| **GIL** | Global Interpreter Lock — Python's mutex that prevents true parallel execution of CPU-bound threads |
| **RLE** | Run-Length Encoding — represents sequences of repeated values as `(value, count)` pairs |
| **Pred-XOR** | Predictive XOR — uses linear extrapolation to predict values before XORing, reducing residual sizes |
| **Gauss-Newton** | Iterative nonlinear least-squares optimization method used for sinusoidal fitting |

---

*End of Deep Project Guide. This document contains everything needed to understand, run, debug, and extend the DLC v2 compression engine.*

> **Last Updated:** 2026-04-15  
> **Engine Status:** 6 models, 6 encoders, block-level DoD fast path, configurable precision (8-20 bits via `--precision`), industry comparison (Gorilla/DoD/RLE/GZIP/ZLIB), fair 16-bit evaluation, 80% win rate at equal precision.
