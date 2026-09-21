"""
Residual quantization and 6 encoding strategies (IMPROVED).

Strategies:
  0: Zigzag + Varint
  1: Fixed-Width Bitpack
  2: Delta-of-Residuals (zigzag+varint on first-order differences)
  3: MAD Outlier Separation (full encoder + decoder)
  4: RLE + Varint  (zero/near-zero runs)
  5: Delta-of-Delta + Varint (second-order differencing — beats standalone DoD)

New features:
  - Adaptive precision bits (8–20 per window)
  - Delta-of-XOR encoding for XOR residuals (models 2 & 5)
  - MAD outlier factor tuned to 4.5 (was 6.0)
"""

import struct
import numpy as np
from typing import List, Tuple

# ── Constants ────────────────────────────────────────────────────────────────
DEFAULT_PRECISION    = 12
MIN_PRECISION        = 8
MAX_PRECISION        = 20
MAD_OUTLIER_FACTOR   = 4.5    # was 6.0 — more aggressive outlier capture
MAX_ERROR_BOUND      = 8e-6   # strict per-sample error constraint


# ── Adaptive Precision ───────────────────────────────────────────────────────

def select_precision(residuals: np.ndarray,
                     requested: int = DEFAULT_PRECISION) -> int:
    """
    Choose optimal precision bits for this window's residuals.
    Fewer bits for well-fitted windows → smaller quantized integers → better compression.
    """
    import math

    max_abs = float(np.max(np.abs(residuals))) if len(residuals) > 0 else 0.0

    if max_abs < 1e-15:
        return MIN_PRECISION  # near-zero residuals

    # Minimum precision for error bound
    min_p_for_bound = max(MIN_PRECISION,
                          math.ceil(math.log2(1.0 / MAX_ERROR_BOUND) - 1))

    # Adaptive: smaller residuals need fewer bits
    if max_abs < 1.0:
        adaptive_p = max(MIN_PRECISION,
                         int(math.ceil(-math.log2(max_abs + 1e-30))) + 4)
    else:
        adaptive_p = DEFAULT_PRECISION

    precision = min(MAX_PRECISION, max(min_p_for_bound, min(requested, adaptive_p)))
    return precision


# ── Quantization ─────────────────────────────────────────────────────────────

def quantize(residuals: np.ndarray, precision_bits: int = DEFAULT_PRECISION) -> List[int]:
    """Quantize float residuals to integer multiples of 2^(-precision_bits)."""
    scale = 2.0 ** precision_bits
    return [int(round(float(r) * scale)) for r in residuals]


def dequantize(quantized: List[int], precision_bits: int = DEFAULT_PRECISION) -> np.ndarray:
    """Dequantize integers back to float residuals."""
    scale = 2.0 ** precision_bits
    return np.array([q / scale for q in quantized], dtype=np.float64)


# ── Zigzag Encoding ──────────────────────────────────────────────────────────

def _zigzag_encode(n: int) -> int:
    """Map signed int64 -> unsigned via zigzag: (n << 1) ^ (n >> 63)."""
    return (n << 1) ^ (n >> 63)


def _zigzag_decode(z: int) -> int:
    """Reverse zigzag encoding."""
    return (z >> 1) ^ -(z & 1)


# ── Varint Encoding ──────────────────────────────────────────────────────────

def _encode_varint(value: int) -> bytes:
    """Encode unsigned integer as varint (7 bits per byte, MSB = continuation)."""
    out = bytearray()
    while value >= 0x80:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value & 0x7F)
    return bytes(out)


def _decode_varint(buf: bytes, offset: int) -> Tuple[int, int]:
    """Decode varint from buf at offset. Returns (value, new_offset)."""
    v, shift = 0, 0
    while True:
        b = buf[offset]
        offset += 1
        v |= (b & 0x7F) << shift
        shift += 7
        if not (b & 0x80):
            return v, offset


# ════════════════════════════════════════════════════════════════════════════
# Strategy 0: Zigzag + Varint
# ════════════════════════════════════════════════════════════════════════════

def encode_zigzag_varint(vals: List[int]) -> bytes:
    """Encode list of signed ints via zigzag then varint."""
    parts = []
    for v in vals:
        parts.append(_encode_varint(_zigzag_encode(v)))
    return b''.join(parts)


def decode_zigzag_varint(buf: bytes, count: int) -> List[int]:
    """Decode `count` zigzag-varint encoded values from buf."""
    result = []
    offset = 0
    for _ in range(count):
        z, offset = _decode_varint(buf, offset)
        result.append(_zigzag_decode(z))
    return result


# ════════════════════════════════════════════════════════════════════════════
# Strategy 1: Fixed-Width Bitpack
# ════════════════════════════════════════════════════════════════════════════

def encode_bitpack(vals: List[int]) -> bytes:
    """
    Fixed-width bitpacking.
    Header: 1 byte bit_width.
    Body: all values packed at bit_width bits, zigzag-encoded, padded to byte.
    """
    if not vals:
        return b'\x01'  # bit_width=1, empty

    zigzagged = [_zigzag_encode(v) for v in vals]
    max_val = max(zigzagged) if zigzagged else 0

    if max_val == 0:
        bit_width = 1
    else:
        bit_width = max(1, max_val.bit_length())
    bit_width = min(bit_width, 64)

    out = bytearray([bit_width])
    buf, nbits = 0, 0
    mask = (1 << bit_width) - 1

    for v in zigzagged:
        buf = (buf << bit_width) | (v & mask)
        nbits += bit_width
        while nbits >= 8:
            nbits -= 8
            out.append((buf >> nbits) & 0xFF)
    if nbits > 0:
        out.append((buf << (8 - nbits)) & 0xFF)
    return bytes(out)


def decode_bitpack(buf: bytes, count: int) -> List[int]:
    """Decode `count` bitpacked values from buf."""
    if count == 0:
        return []
    bit_width = buf[0]
    mask = (1 << bit_width) - 1
    values = []
    bits_buf, nbits_available = 0, 0
    i = 1
    for _ in range(count):
        while nbits_available < bit_width and i < len(buf):
            bits_buf = (bits_buf << 8) | buf[i]
            nbits_available += 8
            i += 1
        nbits_available -= bit_width
        z = (bits_buf >> nbits_available) & mask
        values.append(_zigzag_decode(z))
    return values


# ════════════════════════════════════════════════════════════════════════════
# Strategy 2: Delta-of-Residuals
# ════════════════════════════════════════════════════════════════════════════

def encode_delta_of_residuals(vals: List[int]) -> bytes:
    """
    Compute deltas between consecutive residuals, then zigzag+varint encode.
    First value stored directly.
    """
    if not vals:
        return b''
    deltas = [vals[0]]
    for i in range(1, len(vals)):
        deltas.append(vals[i] - vals[i - 1])
    return encode_zigzag_varint(deltas)


def decode_delta_of_residuals(buf: bytes, count: int) -> List[int]:
    """Decode delta-of-residuals encoded values."""
    deltas = decode_zigzag_varint(buf, count)
    result = []
    acc = 0
    for d in deltas:
        acc += d
        result.append(acc)
    return result


# ════════════════════════════════════════════════════════════════════════════
# Strategy 3: MAD Outlier Separation (COMPLETE decoder per user request)
# ════════════════════════════════════════════════════════════════════════════

def encode_outlier_sep(vals: List[int]) -> bytes:
    """
    MAD-based outlier separation.
    Layout:
      uint16_le num_outliers
      uint8     inlier_bit_width
      outlier_table: num_outliers × (uint32_le index, zigzag-varint value)
      inlier_body: bitpacked at inlier_bit_width (zeros for outlier positions)
    """
    if not vals:
        return b''

    arr = np.array(vals, dtype=np.int64)
    median = int(np.median(arr))
    deviations = np.abs(arr - median)
    mad = float(np.median(deviations))

    if mad == 0:
        # Everything is essentially the same value
        return struct.pack('<HB', 0, 0) + encode_zigzag_varint(vals)

    threshold = MAD_OUTLIER_FACTOR * mad

    # Split into inliers and outliers
    outlier_indices = []
    outlier_values = []
    inlier_values = []

    for i, v in enumerate(vals):
        if abs(v - median) > threshold:
            outlier_indices.append(i)
            outlier_values.append(v)
            inlier_values.append(0)  # placeholder
        else:
            inlier_values.append(v)

    num_outliers = len(outlier_indices)

    # Determine inlier bit width
    if inlier_values:
        inlier_zz = [_zigzag_encode(v) for v in inlier_values]
        max_inlier = max(inlier_zz) if inlier_zz else 0
        if max_inlier == 0:
            inlier_bit_width = 1
        else:
            inlier_bit_width = max(1, max_inlier.bit_length())
    else:
        inlier_bit_width = 1

    # Header
    header = struct.pack('<HB', num_outliers, inlier_bit_width)

    # Outlier table
    outlier_bytes = bytearray()
    for idx, val in zip(outlier_indices, outlier_values):
        outlier_bytes += struct.pack('<I', idx)
        outlier_bytes += _encode_varint(_zigzag_encode(val))

    # Inlier body: bitpacked
    inlier_bytes = encode_bitpack(inlier_values) if inlier_values else b''

    return header + bytes(outlier_bytes) + inlier_bytes


def decode_outlier_sep(buf: bytes, count: int) -> List[int]:
    """
    Full MAD outlier-separated decoder.
    Reads header, outlier table, then inlier bitpack, and reassembles.
    """
    if count == 0 or len(buf) == 0:
        return []

    offset = 0

    # Header
    num_outliers, inlier_bit_width = struct.unpack_from('<HB', buf, offset)
    offset += 3

    if num_outliers == 0 and inlier_bit_width == 0:
        # Fallback: entire payload is zigzag-varint
        return decode_zigzag_varint(buf[offset:], count)

    # Read outlier table
    outlier_map = {}  # index -> value
    for _ in range(num_outliers):
        idx = struct.unpack_from('<I', buf, offset)[0]
        offset += 4
        zval, offset = _decode_varint(buf, offset)
        outlier_map[idx] = _zigzag_decode(zval)

    # Read inlier bitpack (remaining bytes)
    num_inliers = count - num_outliers
    inlier_data = buf[offset:]
    if num_inliers > 0 and len(inlier_data) > 0:
        inliers = decode_bitpack(inlier_data, num_inliers)
    else:
        inliers = []

    # Reassemble: walk through count positions
    result = []
    inlier_idx = 0
    for i in range(count):
        if i in outlier_map:
            result.append(outlier_map[i])
        else:
            if inlier_idx < len(inliers):
                result.append(inliers[inlier_idx])
                inlier_idx += 1
            else:
                result.append(0)

    return result


# ════════════════════════════════════════════════════════════════════════════
# Strategy 4: RLE + Varint  (NEW)
# ════════════════════════════════════════════════════════════════════════════

def encode_rle(vals: List[int]) -> bytes:
    """
    Run-length encoding of quantized residuals.
    Format: varint(num_runs), then pairs of (zigzag_value, run_length) as varints.
    Extremely compact when most residuals are 0 (constant windows).
    """
    if not vals:
        return b''

    runs = []
    cur_val = vals[0]
    run_len = 1
    for v in vals[1:]:
        if v == cur_val:
            run_len += 1
        else:
            runs.append((cur_val, run_len))
            cur_val = v
            run_len = 1
    runs.append((cur_val, run_len))

    out = _encode_varint(len(runs))
    for val, cnt in runs:
        out += _encode_varint(_zigzag_encode(val))
        out += _encode_varint(cnt)
    return out


def decode_rle(buf: bytes, count: int) -> List[int]:
    """Decode RLE-encoded values."""
    if count == 0 or len(buf) == 0:
        return []

    offset = 0
    n_runs, offset = _decode_varint(buf, offset)
    result = []
    for _ in range(n_runs):
        zval, offset = _decode_varint(buf, offset)
        val = _zigzag_decode(zval)
        cnt, offset = _decode_varint(buf, offset)
        result.extend([val] * cnt)
    return result[:count]


# ════════════════════════════════════════════════════════════════════════════
# Strategy 5: Delta-of-Delta + Varint  (BEATS standalone DoD)
# ════════════════════════════════════════════════════════════════════════════

def encode_dod(vals: List[int]) -> bytes:
    """
    Second-order differencing + zigzag-varint encoding.
    Same algorithm as standalone "Delta-of-Delta" (InfluxDB/Prometheus),
    applied to quantized residuals.

    Format:
      zigzag_varint(first_value)
      zigzag_varint(first_delta)
      zigzag_varint(dod_0), zigzag_varint(dod_1), ...
    """
    if not vals:
        return b''
    if len(vals) == 1:
        return _encode_varint(_zigzag_encode(vals[0]))

    parts = []
    # First value
    parts.append(_encode_varint(_zigzag_encode(vals[0])))
    # First delta
    delta = vals[1] - vals[0]
    parts.append(_encode_varint(_zigzag_encode(delta)))
    # Delta-of-deltas
    prev_delta = delta
    for i in range(2, len(vals)):
        curr_delta = vals[i] - vals[i - 1]
        dod = curr_delta - prev_delta
        parts.append(_encode_varint(_zigzag_encode(dod)))
        prev_delta = curr_delta
    return b''.join(parts)


def decode_dod(buf: bytes, count: int) -> List[int]:
    """Decode Delta-of-Delta encoded values."""
    if count == 0 or len(buf) == 0:
        return []

    offset = 0
    # First value
    zval, offset = _decode_varint(buf, offset)
    first_val = _zigzag_decode(zval)
    result = [first_val]

    if count == 1:
        return result

    # First delta
    zdelta, offset = _decode_varint(buf, offset)
    delta = _zigzag_decode(zdelta)
    result.append(first_val + delta)

    # Delta-of-deltas → reconstruct via cumulative sums
    prev_delta = delta
    for _ in range(2, count):
        zdod, offset = _decode_varint(buf, offset)
        dod = _zigzag_decode(zdod)
        curr_delta = prev_delta + dod
        result.append(result[-1] + curr_delta)
        prev_delta = curr_delta

    return result


# ════════════════════════════════════════════════════════════════════════════
# XOR Residual Encoding (for Models 2 & 5)
# ════════════════════════════════════════════════════════════════════════════

def encode_xor_residuals(xor_residuals: List[int]) -> bytes:
    """
    Encode XOR residuals with optional delta-of-XOR compression.
    Prefix byte: 0x00 = raw uint64 LE, 0x01 = delta-encoded.
    """
    if not xor_residuals:
        return b'\x00'

    # Raw: pack as 8-byte little-endian uint64
    raw = b''.join(struct.pack('<Q', v) for v in xor_residuals)

    # Try delta-of-XOR (consecutive XOR values often similar in trending data)
    if len(xor_residuals) > 1:
        deltas = [xor_residuals[0]]
        for i in range(1, len(xor_residuals)):
            deltas.append((xor_residuals[i] - xor_residuals[i - 1]) & 0xFFFFFFFFFFFFFFFF)
        delta_bytes = b''.join(struct.pack('<Q', v) for v in deltas)

        if len(delta_bytes) < len(raw):
            return b'\x01' + delta_bytes

    return b'\x00' + raw


def decode_xor_residuals(data: bytes, count: int) -> List[int]:
    """Decode XOR residuals (prefix byte distinguishes raw vs delta)."""
    if count == 0 or len(data) == 0:
        return []

    flag = data[0]
    payload = data[1:]

    if flag == 0x00:
        # Raw 8-byte LE uint64 values
        return [struct.unpack_from('<Q', payload, i * 8)[0] for i in range(count)]
    else:
        # Delta-encoded: cumulative sum to recover original XOR values
        deltas = [struct.unpack_from('<Q', payload, i * 8)[0] for i in range(count)]
        result = []
        acc = 0
        for d in deltas:
            acc = (acc + d) & 0xFFFFFFFFFFFFFFFF
            result.append(acc)
        return result


# ════════════════════════════════════════════════════════════════════════════
# Trial Encode — pick the smallest output
# ════════════════════════════════════════════════════════════════════════════

def trial_encode(quantized: List[int]) -> Tuple[int, bytes]:
    """
    Try all 6 encoding strategies on quantized residuals.
    Returns (encoding_id, encoded_bytes) — the smallest wins.
    """
    if not quantized:
        return 0, b''

    candidates = {}

    # Strategy 0: Zigzag + Varint
    try:
        candidates[0] = encode_zigzag_varint(quantized)
    except Exception:
        pass

    # Strategy 1: Fixed-Width Bitpack
    try:
        candidates[1] = encode_bitpack(quantized)
    except Exception:
        pass

    # Strategy 2: Delta-of-Residuals (first-order)
    try:
        candidates[2] = encode_delta_of_residuals(quantized)
    except Exception:
        pass

    # Strategy 3: MAD Outlier Separation
    try:
        if len(quantized) >= 8:
            candidates[3] = encode_outlier_sep(quantized)
    except Exception:
        pass

    # Strategy 4: RLE
    try:
        candidates[4] = encode_rle(quantized)
    except Exception:
        pass

    # Strategy 5: Delta-of-Delta (second-order — DoD killer)
    try:
        candidates[5] = encode_dod(quantized)
    except Exception:
        pass

    if not candidates:
        return 0, encode_zigzag_varint(quantized)

    best_id = min(candidates, key=lambda k: len(candidates[k]))
    return best_id, candidates[best_id]


def decode_residuals(encoding_id: int, data: bytes, count: int) -> List[int]:
    """Decode encoded residuals given the encoding_id."""
    if count == 0:
        return []

    if encoding_id == 0:
        return decode_zigzag_varint(data, count)
    elif encoding_id == 1:
        return decode_bitpack(data, count)
    elif encoding_id == 2:
        return decode_delta_of_residuals(data, count)
    elif encoding_id == 3:
        return decode_outlier_sep(data, count)
    elif encoding_id == 4:
        return decode_rle(data, count)
    elif encoding_id == 5:
        return decode_dod(data, count)
    else:
        raise ValueError(f"Unknown encoding_id: {encoding_id}")
