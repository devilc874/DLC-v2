"""
ZFP baseline — fixed-accuracy floating-point compressor (LLNL).

1) Prefer native `zfpy` when installed.
2) Otherwise use a bundled pure-Python block fixed-accuracy reference so
   batch / frontend comparisons always produce ZFP numbers.
"""

from __future__ import annotations

import struct
import time
from typing import List, Tuple

import numpy as np

from .common import BaselineResult

DEFAULT_ABS_ERROR = 8e-6
BLOCK = 64  # ZFP uses small blocks; 64 is a practical 1D chunk size


def available() -> bool:
    """Always available (native or pure-Python fallback)."""
    return True


def native_available() -> bool:
    try:
        import zfpy  # noqa: F401
        return True
    except Exception:
        return False


def _zigzag(n: int) -> int:
    return (n << 1) ^ (n >> 63)


def _unzigzag(z: int) -> int:
    return (z >> 1) ^ -(z & 1)


def _enc_varint(v: int) -> bytes:
    out = bytearray()
    v &= (1 << 64) - 1
    while v >= 0x80:
        out.append((v & 0x7F) | 0x80)
        v >>= 7
    out.append(v & 0x7F)
    return bytes(out)


def _dec_varint(buf: bytes, offset: int) -> Tuple[int, int]:
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


def _encode_ref(data: np.ndarray, abs_error: float) -> bytes:
    """
    Fixed-accuracy block compressor (ZFP-family idea for 1D):
      - Split into blocks
      - Store block mean as float64
      - Quantize (v - mean) with step = abs_error
      - Zigzag-varint the integers
    Guarantees |error| ≤ abs_error/2 from quantization (≤ abs_error practically).
    """
    arr = np.asarray(data, dtype=np.float64).reshape(-1)
    n = len(arr)
    step = float(abs_error) if abs_error > 0 else DEFAULT_ABS_ERROR

    out = bytearray()
    out.extend(struct.pack("<I", n))
    out.extend(struct.pack("<d", step))
    out.extend(struct.pack("<I", BLOCK))

    i = 0
    while i < n:
        chunk = arr[i:i + BLOCK]
        mean = float(np.mean(chunk)) if len(chunk) else 0.0
        out.extend(struct.pack("<d", mean))
        out.extend(struct.pack("<I", len(chunk)))
        for v in chunk:
            fv = float(v) if np.isfinite(v) else mean
            q = int(round((fv - mean) / step))
            out.extend(_enc_varint(_zigzag(q)))
        i += BLOCK
    return bytes(out)


def _decode_ref(blob: bytes) -> np.ndarray:
    if len(blob) < 16:
        return np.array([], dtype=np.float64)
    n = struct.unpack_from("<I", blob, 0)[0]
    step = struct.unpack_from("<d", blob, 4)[0]
    # block size at offset 12
    offset = 16
    values: List[float] = []
    while len(values) < n and offset < len(blob):
        mean = struct.unpack_from("<d", blob, offset)[0]
        offset += 8
        count = struct.unpack_from("<I", blob, offset)[0]
        offset += 4
        for _ in range(count):
            z, offset = _dec_varint(blob, offset)
            q = _unzigzag(z)
            values.append(mean + q * step)
    return np.asarray(values[:n], dtype=np.float64)


def _compress_ref(data: np.ndarray, abs_error: float) -> BaselineResult:
    arr = np.asarray(data, dtype=np.float64).reshape(-1)
    t0 = time.perf_counter()
    blob = _encode_ref(arr, abs_error)
    t1 = time.perf_counter()
    recovered = _decode_ref(blob)
    t2 = time.perf_counter()
    return BaselineResult(
        name="ZFP",
        compressed_size=len(blob),
        recovered=recovered,
        compress_seconds=t1 - t0,
        decompress_seconds=t2 - t1,
        lossless=False,
        backend="pure-python",
        notes=f"block fixed-accuracy reference, tol={abs_error:g}",
    )


def _compress_native(data: np.ndarray, abs_error: float) -> BaselineResult | None:
    try:
        import zfpy
    except Exception:
        return None

    arr = np.ascontiguousarray(data, dtype=np.float64)
    t0 = time.perf_counter()
    compressed = zfpy.compress_numpy(arr, tolerance=float(abs_error))
    t1 = time.perf_counter()
    blob = compressed if isinstance(compressed, (bytes, bytearray)) else bytes(compressed)
    t2 = time.perf_counter()
    recovered = zfpy.decompress_numpy(compressed)
    t3 = time.perf_counter()
    recovered = np.asarray(recovered, dtype=np.float64).reshape(-1)

    return BaselineResult(
        name="ZFP",
        compressed_size=len(blob),
        recovered=recovered,
        compress_seconds=t1 - t0,
        decompress_seconds=t3 - t2,
        lossless=False,
        backend="zfpy",
        notes=f"tolerance={abs_error:g}",
    )


def compress_zfp(data: np.ndarray,
                 abs_error: float = DEFAULT_ABS_ERROR) -> BaselineResult:
    """
    Always returns a result. Uses zfpy when possible, else pure-Python reference.
    """
    native = _compress_native(data, abs_error)
    if native is not None:
        return native
    return _compress_ref(data, abs_error)
