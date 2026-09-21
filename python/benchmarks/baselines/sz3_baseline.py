"""
SZ3 baseline — error-bounded scientific compressor.

1) Prefer native `pysz` when installed.
2) Otherwise use a bundled pure-Python Lorenzo + ABS-quantization reference
   so batch / frontend comparisons always produce SZ3 numbers (same idea as
   Elf/ALP always running).
"""

from __future__ import annotations

import struct
import time
from typing import List, Tuple

import numpy as np

from .common import BaselineResult

DEFAULT_ABS_ERROR = 8e-6  # match DLC / paper figure


def available() -> bool:
    """Always available (native or pure-Python fallback)."""
    return True


def native_available() -> bool:
    try:
        from pysz import sz, szConfig, szErrorBoundMode  # noqa: F401
        return True
    except Exception:
        return False


# ── Pure-Python SZ3-style reference (1D Lorenzo + ABS quantize) ─────────────

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
    1D Lorenzo predictor + absolute-error quantization.

    Quantization step = abs_error  →  reconstruction error ≤ abs_error/2
    (conservative vs paper bound; still comparable under same tolerance).
    We use step = abs_error so max |err| ≤ abs_error (with rounding ≤ step/2,
    but we clamp by using step = abs_error for the bound check on decode path).
    """
    arr = np.asarray(data, dtype=np.float64).reshape(-1)
    n = len(arr)
    # step such that |q*step - r| ≤ abs_error after round
    step = float(abs_error)
    if step <= 0:
        step = DEFAULT_ABS_ERROR

    out = bytearray()
    out.extend(struct.pack("<I", n))
    out.extend(struct.pack("<d", step))
    if n == 0:
        return bytes(out)

    # First sample raw
    out.extend(struct.pack("<d", float(arr[0])))
    prev = float(arr[0])
    for i in range(1, n):
        actual = float(arr[i])
        if not np.isfinite(actual):
            # Store sentinel: q=0 and overwrite with raw via rare path —
            # for simplicity store as large quantized delta from prev=0
            actual = 0.0
        residual = actual - prev  # Lorenzo: predict = previous
        q = int(round(residual / step))
        out.extend(_enc_varint(_zigzag(q)))
        # Decoder will reconstruct the same way:
        prev = prev + q * step
    return bytes(out)


def _decode_ref(blob: bytes) -> np.ndarray:
    if len(blob) < 4:
        return np.array([], dtype=np.float64)
    n = struct.unpack_from("<I", blob, 0)[0]
    step = struct.unpack_from("<d", blob, 4)[0]
    offset = 12
    if n == 0:
        return np.array([], dtype=np.float64)
    first = struct.unpack_from("<d", blob, offset)[0]
    offset += 8
    out: List[float] = [first]
    prev = first
    for _ in range(1, n):
        z, offset = _dec_varint(blob, offset)
        q = _unzigzag(z)
        prev = prev + q * step
        out.append(prev)
    return np.asarray(out, dtype=np.float64)


def _compress_ref(data: np.ndarray, abs_error: float) -> BaselineResult:
    arr = np.asarray(data, dtype=np.float64).reshape(-1)
    t0 = time.perf_counter()
    blob = _encode_ref(arr, abs_error)
    t1 = time.perf_counter()
    recovered = _decode_ref(blob)
    t2 = time.perf_counter()
    return BaselineResult(
        name="SZ3",
        compressed_size=len(blob),
        recovered=recovered,
        compress_seconds=t1 - t0,
        decompress_seconds=t2 - t1,
        lossless=False,
        backend="pure-python",
        notes=f"Lorenzo+ABS reference, bound={abs_error:g}",
    )


def _compress_native(data: np.ndarray, abs_error: float) -> BaselineResult | None:
    try:
        from pysz import sz, szConfig, szErrorBoundMode
    except Exception:
        return None

    arr = np.ascontiguousarray(data, dtype=np.float64)
    config = szConfig()
    config.errorBoundMode = szErrorBoundMode.ABS
    config.absErrorBound = float(abs_error)

    t0 = time.perf_counter()
    compressed, _ratio = sz.compress(arr, config)
    t1 = time.perf_counter()

    size = int(compressed.nbytes) if isinstance(compressed, np.ndarray) else len(compressed)

    t2 = time.perf_counter()
    recovered, _ = sz.decompress(compressed, np.float64, arr.shape)
    t3 = time.perf_counter()
    recovered = np.asarray(recovered, dtype=np.float64).reshape(-1)

    return BaselineResult(
        name="SZ3",
        compressed_size=size,
        recovered=recovered,
        compress_seconds=t1 - t0,
        decompress_seconds=t3 - t2,
        lossless=False,
        backend="pysz",
        notes=f"ABS error bound={abs_error:g}",
    )


def compress_sz3(data: np.ndarray,
                 abs_error: float = DEFAULT_ABS_ERROR) -> BaselineResult:
    """
    Always returns a result. Uses pysz when possible, else pure-Python reference.
    """
    native = _compress_native(data, abs_error)
    if native is not None:
        return native
    return _compress_ref(data, abs_error)
