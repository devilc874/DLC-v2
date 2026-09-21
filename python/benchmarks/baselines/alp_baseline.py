"""
ALP-style lossless floating-point baseline (pure-Python reference).

ALP (Afroozeh et al., VLDB) maps doubles to integers via carefully chosen
combinations of factor/exponent, then encodes the integers. This reference
implements that core idea for paper comparisons when the original ALP library
is unavailable:

  For each vector chunk:
    - Search a small grid of (e, f) such that v * 10^e / f is an exact int.
    - Encode successful integers with zigzag+varint.
    - Fall back to raw float64 when no exact map exists / raw is smaller.

Always available (backend='pure-python'). Marked lossless.
"""

from __future__ import annotations

import struct
import time
from typing import List, Optional, Tuple

import numpy as np

from .common import BaselineResult


def available() -> bool:
    return True


def _zigzag(n: int) -> int:
    return (n << 1) ^ (n >> 63)


def _unzigzag(z: int) -> int:
    return (z >> 1) ^ -(z & 1)


def _encode_varint(v: int) -> bytes:
    out = bytearray()
    v &= (1 << 64) - 1
    while v >= 0x80:
        out.append((v & 0x7F) | 0x80)
        v >>= 7
    out.append(v & 0x7F)
    return bytes(out)


def _decode_varint(buf: bytes, offset: int) -> Tuple[int, int]:
    v, shift = 0, 0
    while True:
        b = buf[offset]
        offset += 1
        v |= (b & 0x7F) << shift
        if not (b & 0x80):
            return v, offset
        shift += 7
        if shift > 70:
            raise ValueError("varint overflow")


def _try_map(values: np.ndarray, exp: int, factor: int) -> Optional[List[int]]:
    """Lossless map q_i = round(v_i * 10^exp / factor); require exact reconstruct."""
    scale = (10.0 ** exp) / float(factor)
    qs: List[int] = []
    for v in values:
        fv = float(v)
        if not np.isfinite(fv):
            return None
        q = int(round(fv * scale))
        if (q / scale) != fv:
            return None
        qs.append(q)
    return qs


def _best_encoding(chunk: np.ndarray):
    best = None
    best_cost = None
    for exp in range(0, 12):
        for factor in (1, 2, 4, 5, 8, 10, 16, 20, 25, 50, 100):
            mapped = _try_map(chunk, exp, factor)
            if mapped is None:
                continue
            zz = [_zigzag(q) for q in mapped]
            width = max((z.bit_length() for z in zz), default=1)
            cost = 8 + (width * len(zz) + 7) // 8
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best = (exp, factor, mapped, width)
    return best


def _encode_chunk_raw(chunk: np.ndarray) -> bytes:
    out = bytearray()
    out.append(0x00)
    out.extend(struct.pack("<I", len(chunk)))
    out.extend(chunk.astype("<f8").tobytes())
    return bytes(out)


def _encode_chunk_alp(exp: int, factor: int, ints: List[int], width: int) -> bytes:
    parts = [b"\x01", bytes([exp & 0xFF, factor & 0xFF, width & 0xFF])]
    parts.append(struct.pack("<I", len(ints)))
    for q in ints:
        parts.append(_encode_varint(_zigzag(q)))
    return b"".join(parts)


def _encode(data: np.ndarray, chunk_size: int = 1024) -> bytes:
    out = bytearray()
    out.extend(struct.pack("<I", len(data)))
    n = len(data)
    i = 0
    while i < n:
        chunk = data[i:i + chunk_size]
        raw = _encode_chunk_raw(chunk)
        best = _best_encoding(chunk)
        if best is None:
            out.extend(raw)
        else:
            exp, factor, ints, width = best
            alp = _encode_chunk_alp(exp, factor, ints, width)
            out.extend(alp if len(alp) < len(raw) else raw)
        i += chunk_size
    return bytes(out)


def _decode(blob: bytes) -> np.ndarray:
    if len(blob) < 4:
        return np.array([], dtype=np.float64)
    n = struct.unpack_from("<I", blob, 0)[0]
    offset = 4
    values: List[float] = []
    while len(values) < n and offset < len(blob):
        marker = blob[offset]
        offset += 1
        if marker == 0x00:
            count = struct.unpack_from("<I", blob, offset)[0]
            offset += 4
            raw = np.frombuffer(blob, dtype="<f8", count=count, offset=offset).copy()
            offset += count * 8
            values.extend(raw.tolist())
        elif marker == 0x01:
            exp = blob[offset]
            factor = blob[offset + 1]
            offset += 3  # skip exp, factor, width
            count = struct.unpack_from("<I", blob, offset)[0]
            offset += 4
            scale = (10.0 ** exp) / float(factor)
            for _ in range(count):
                z, offset = _decode_varint(blob, offset)
                q = _unzigzag(z)
                values.append(q / scale)
        else:
            raise ValueError(f"unknown ALP chunk marker {marker}")
    return np.asarray(values[:n], dtype=np.float64)


def compress_alp(data: np.ndarray, abs_error: float = 0.0) -> BaselineResult:
    """
    Lossless ALP-style factor/exponent mapping + varint.
    abs_error ignored (lossless); accepted for API uniformity.
    """
    arr = np.asarray(data, dtype=np.float64).reshape(-1)

    t0 = time.perf_counter()
    blob = _encode(arr)
    t1 = time.perf_counter()
    recovered = _decode(blob)
    t2 = time.perf_counter()

    return BaselineResult(
        name="ALP",
        compressed_size=len(blob),
        recovered=recovered,
        compress_seconds=t1 - t0,
        decompress_seconds=t2 - t1,
        lossless=True,
        backend="pure-python",
        notes="lossless factor/exponent reference (ALP-family)",
    )
