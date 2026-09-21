"""
Benchmark script: Collects comprehensive metrics for DLC paper revision.
Metrics: compression ratio, max error, MAE, RMSE, throughput (comp/decomp), memory.
"""

import os
import sys
import time
import gzip
import zlib
import struct
import tempfile
import subprocess
import tracemalloc
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
TEST_DATA_DIR = os.path.join(PROJECT_ROOT, 'test_data')

def _find_cpp_binary():
    candidates = [
        os.path.join(PROJECT_ROOT, 'cpp', 'build', 'dlc.exe'),
        os.path.join(PROJECT_ROOT, 'cpp', 'build', 'Release', 'dlc.exe'),
        os.path.join(PROJECT_ROOT, 'cpp', 'build', 'dlc'),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None

def _get_env():
    env = os.environ.copy()
    msys2_bin = r"C:\msys64\ucrt64\bin"
    if os.path.isdir(msys2_bin):
        env["PATH"] = msys2_bin + os.pathsep + env.get("PATH", "")
    return env

def _list_bin_files():
    if not os.path.isdir(TEST_DATA_DIR):
        return {}
    bins = {}
    for f in sorted(os.listdir(TEST_DATA_DIR)):
        if f.endswith('.bin'):
            bins[f[:-4]] = os.path.join(TEST_DATA_DIR, f)
    return bins

# ── Gorilla (from app.py) ─────────────────────────────────────────────────────

class BitWriter:
    def __init__(self):
        self.buf = bytearray()
        self.current = 0
        self.bit_pos = 0
    def write_bit(self, bit):
        self.current = (self.current << 1) | (bit & 1)
        self.bit_pos += 1
        if self.bit_pos == 8:
            self.buf.append(self.current)
            self.current = 0
            self.bit_pos = 0
    def write_bits(self, value, num_bits):
        for i in range(num_bits - 1, -1, -1):
            self.write_bit((value >> i) & 1)
    def flush(self):
        if self.bit_pos > 0:
            self.current <<= (8 - self.bit_pos)
            self.buf.append(self.current)
        return bytes(self.buf)

def _f64_to_u64(val):
    return struct.unpack('<Q', struct.pack('<d', val))[0]

def compress_gorilla(data):
    if len(data) == 0: return b''
    writer = BitWriter()
    prev = _f64_to_u64(data[0])
    writer.write_bits(prev, 64)
    prev_leading, prev_trailing, prev_meaningful = 64, 0, 64
    for i in range(1, len(data)):
        curr = _f64_to_u64(data[i])
        xor = prev ^ curr
        if xor == 0:
            writer.write_bit(0)
        else:
            writer.write_bit(1)
            leading = 0
            t = xor
            for _ in range(64):
                if t & (1 << 63): break
                leading += 1; t <<= 1
            trailing = 0
            t = xor
            for _ in range(64):
                if t & 1: break
                trailing += 1; t >>= 1
            meaningful = max(1, 64 - leading - trailing)
            if leading >= prev_leading and trailing >= prev_trailing:
                writer.write_bit(0)
                val = (xor >> prev_trailing) & ((1 << prev_meaningful) - 1)
                writer.write_bits(val, prev_meaningful)
            else:
                writer.write_bit(1)
                writer.write_bits(leading, 6)
                writer.write_bits(meaningful, 6)
                val = (xor >> trailing) & ((1 << meaningful) - 1)
                writer.write_bits(val, meaningful)
                prev_leading, prev_trailing, prev_meaningful = leading, trailing, meaningful
        prev = curr
    return writer.flush()

def compress_delta_of_delta_fair(data):
    if len(data) == 0: return b''
    clean = np.where(np.isfinite(data), data, 0.0)
    scale = 1 << 16
    q = [int(round(v * scale)) for v in clean]
    buf = bytearray()
    buf.extend(struct.pack('<q', q[0]))
    if len(q) < 2: return bytes(buf)
    delta = q[1] - q[0]
    zz = (delta << 1) ^ (delta >> 63)
    while zz > 0x7F: buf.append((zz & 0x7F) | 0x80); zz >>= 7
    buf.append(zz & 0x7F)
    prev_delta = delta
    for i in range(2, len(q)):
        d = q[i] - q[i-1]
        dod = d - prev_delta
        zz = (dod << 1) ^ (dod >> 63)
        while zz > 0x7F: buf.append((zz & 0x7F) | 0x80); zz >>= 7
        buf.append(zz & 0x7F)
        prev_delta = d
    return bytes(buf)

def compress_rle_float(data):
    if len(data) == 0: return b''
    buf = bytearray()
    cur, cnt = data[0], 1
    for i in range(1, len(data)):
        if data[i] == cur: cnt += 1
        else:
            buf.extend(struct.pack('<dI', cur, cnt))
            cur, cnt = data[i], 1
    buf.extend(struct.pack('<dI', cur, cnt))
    return bytes(buf)

def _zigzag_decode(n):
    return (n >> 1) ^ -(n & 1)

def _varint_decode(buf, offset):
    result = 0; shift = 0
    while offset < len(buf):
        b = buf[offset]; offset += 1
        result |= (b & 0x7F) << shift
        if (b & 0x80) == 0: break
        shift += 7
    return result, offset

def decompress_delta_of_delta(compressed, n_samples, precision_bits=16):
    if n_samples == 0 or len(compressed) == 0:
        return np.array([], dtype=np.float64)
    scale = 1 << precision_bits
    first_q = struct.unpack('<q', compressed[:8])[0]
    if n_samples == 1:
        return np.array([first_q / scale], dtype=np.float64)
    offset = 8
    zz_delta, offset = _varint_decode(compressed, offset)
    delta = _zigzag_decode(zz_delta)
    quantized = [first_q, first_q + delta]
    prev_delta = delta
    for _ in range(2, n_samples):
        zz_dod, offset = _varint_decode(compressed, offset)
        dod = _zigzag_decode(zz_dod)
        curr_delta = prev_delta + dod
        quantized.append(quantized[-1] + curr_delta)
        prev_delta = curr_delta
    return np.array([q / scale for q in quantized], dtype=np.float64)

# ── Main benchmark ────────────────────────────────────────────────────────────

def run_full_benchmark():
    binary = _find_cpp_binary()
    if not binary:
        print("ERROR: C++ binary not found. Build first.")
        return
    
    bins = _list_bin_files()
    if not bins:
        print("ERROR: No .bin datasets found in test_data/")
        return
    
    env = _get_env()
    
    print(f"Found {len(bins)} datasets, C++ binary at {binary}")
    print("=" * 120)
    
    # Collect all results
    all_results = []
    
    for name, path in bins.items():
        data = np.fromfile(path, dtype='<f8')
        raw_bytes = data.tobytes()
        raw_size = len(raw_bytes)
        n = len(data)
        
        result = {
            'dataset': name,
            'samples': n,
            'raw_size_kb': raw_size / 1024,
        }
        
        # ── DLC (C++) compress + decompress ──
        dlc_path = tempfile.mktemp(suffix='.dlc')
        rec_path = tempfile.mktemp(suffix='.bin')
        try:
            # Compression
            tracemalloc.start()
            t0 = time.perf_counter()
            r = subprocess.run([binary, "compress", "-i", path, "-o", dlc_path,
                                "--workers", "4"], capture_output=True, timeout=120, env=env)
            comp_time = time.perf_counter() - t0
            _, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            if r.returncode == 0 and os.path.isfile(dlc_path):
                dlc_size = os.path.getsize(dlc_path)
                result['dlc_ratio'] = raw_size / dlc_size if dlc_size > 0 else 0
                result['dlc_comp_size_kb'] = dlc_size / 1024
                result['dlc_comp_time'] = comp_time
                result['dlc_comp_throughput_mbs'] = (raw_size / 1e6) / comp_time if comp_time > 0 else 0
                
                # Decompression
                t0 = time.perf_counter()
                subprocess.run([binary, "decompress", "-i", dlc_path, "-o", rec_path],
                               capture_output=True, timeout=120, env=env)
                decomp_time = time.perf_counter() - t0
                result['dlc_decomp_time'] = decomp_time
                result['dlc_decomp_throughput_mbs'] = (raw_size / 1e6) / decomp_time if decomp_time > 0 else 0
                
                # Error analysis
                if os.path.isfile(rec_path):
                    recovered = np.fromfile(rec_path, dtype=np.float64)
                    if len(recovered) == len(data):
                        fm = np.isfinite(data) & np.isfinite(recovered)
                        if np.any(fm):
                            errors = np.abs(data[fm] - recovered[fm])
                            result['dlc_max_error'] = float(np.max(errors))
                            result['dlc_mae'] = float(np.mean(errors))
                            result['dlc_rmse'] = float(np.sqrt(np.mean(errors**2)))
                        else:
                            result['dlc_max_error'] = 0.0
                            result['dlc_mae'] = 0.0
                            result['dlc_rmse'] = 0.0
        except Exception as e:
            result['dlc_error'] = str(e)
        finally:
            for p in [dlc_path, rec_path]:
                if os.path.isfile(p):
                    try: os.unlink(p)
                    except: pass
        
        # ── Gorilla ──
        try:
            sample = data[:min(50000, n)]
            t0 = time.perf_counter()
            g = compress_gorilla(sample)
            t = time.perf_counter() - t0
            ratio_sample = len(sample) * 8 / len(g) if len(g) > 0 else 1
            result['gorilla_ratio'] = ratio_sample
            result['gorilla_time'] = t * (n / len(sample))  # extrapolate
        except:
            result['gorilla_ratio'] = 1.0
        
        # ── DoD 16-bit ──
        try:
            t0 = time.perf_counter()
            d = compress_delta_of_delta_fair(data)
            t = time.perf_counter() - t0
            result['dod16_ratio'] = raw_size / len(d) if len(d) > 0 else 0
            result['dod16_time'] = t
            # Error
            rec = decompress_delta_of_delta(d, n, 16)
            fm = np.isfinite(data) & np.isfinite(rec)
            if np.any(fm):
                errors = np.abs(data[fm] - rec[fm])
                result['dod16_max_error'] = float(np.max(errors))
                result['dod16_mae'] = float(np.mean(errors))
        except:
            result['dod16_ratio'] = 0
        
        # ── RLE ──
        try:
            t0 = time.perf_counter()
            r = compress_rle_float(data)
            t = time.perf_counter() - t0
            result['rle_ratio'] = raw_size / len(r) if len(r) > 0 else 0
            result['rle_time'] = t
        except:
            result['rle_ratio'] = 0
        
        # ── GZIP ──
        t0 = time.perf_counter()
        gz = gzip.compress(raw_bytes, compresslevel=6)
        t = time.perf_counter() - t0
        result['gzip_ratio'] = raw_size / len(gz)
        result['gzip_time'] = t
        
        # ── ZLIB ──
        t0 = time.perf_counter()
        zl = zlib.compress(raw_bytes, level=9)
        t = time.perf_counter() - t0
        result['zlib_ratio'] = raw_size / len(zl)
        result['zlib_time'] = t
        
        all_results.append(result)
        print(f"  [{len(all_results):2d}/{len(bins)}] {name:40s}  DLC={result.get('dlc_ratio',0):.1f}x  "
              f"Gorilla={result.get('gorilla_ratio',0):.1f}x  DoD16={result.get('dod16_ratio',0):.1f}x  "
              f"GZIP={result.get('gzip_ratio',0):.1f}x  "
              f"MaxErr={result.get('dlc_max_error','N/A')}")
    
    # ── Output as formatted tables ──
    print("\n" + "=" * 120)
    print("TABLE VII: COMPREHENSIVE PERFORMANCE METRICS (DLC v2)")
    print("=" * 120)
    
    # Header
    print(f"{'Dataset':<40s} {'Samples':>10s} {'Ratio':>8s} {'CompMB/s':>10s} {'DecompMB/s':>10s} "
          f"{'MaxErr':>12s} {'MAE':>12s} {'RMSE':>12s}")
    print("-" * 120)
    
    for r in all_results:
        print(f"{r['dataset']:<40s} {r['samples']:>10,d} "
              f"{r.get('dlc_ratio',0):>8.1f}x "
              f"{r.get('dlc_comp_throughput_mbs',0):>10.1f} "
              f"{r.get('dlc_decomp_throughput_mbs',0):>10.1f} "
              f"{r.get('dlc_max_error',0):>12.2e} "
              f"{r.get('dlc_mae',0):>12.2e} "
              f"{r.get('dlc_rmse',0):>12.2e}")
    
    # ── Output CSV for plotting ──
    csv_path = os.path.join(SCRIPT_DIR, 'benchmark_results.csv')
    with open(csv_path, 'w') as f:
        keys = ['dataset','samples','raw_size_kb',
                'dlc_ratio','dlc_comp_throughput_mbs','dlc_decomp_throughput_mbs',
                'dlc_max_error','dlc_mae','dlc_rmse',
                'gorilla_ratio','dod16_ratio','rle_ratio','gzip_ratio','zlib_ratio',
                'dlc_comp_time','dlc_decomp_time']
        f.write(','.join(keys) + '\n')
        for r in all_results:
            vals = [str(r.get(k, '')) for k in keys]
            f.write(','.join(vals) + '\n')
    
    print(f"\nCSV saved to: {csv_path}")
    print(f"Total datasets: {len(all_results)}")

if __name__ == '__main__':
    run_full_benchmark()
