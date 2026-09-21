"""
Elf-style lossless floating-point baseline (pure-Python reference).

Elf (Li et al.) erases insignificant trailing mantissa bits then XOR-encodes
the stream. This reference implements a practical lossless variant used for
paper comparisons when the original Elf binary is unavailable:

  1. Store the first double raw (64 bits).
  2. For each later value, XOR with the previous value's bits.
  3. Pack (leading_zeros, trailing_zeros, meaningful_bits) using a compact
     bit-writer — the same family of tricks as Gorilla/Elf.

Always available (backend='pure-python'). Marked lossless.
"""

from __future__ import annotations

import struct
import time
from typing import List

import numpy as np

from .common import BaselineResult


def available() -> bool:
    return True


class _BitWriter:
    def __init__(self):
        self.buf = bytearray()
        self.cur = 0
        self.nbits = 0

    def write_bits(self, value: int, width: int):
        if width <= 0:
            return
        value &= (1 << width) - 1 if width < 64 else (1 << 64) - 1
        for i in range(width - 1, -1, -1):
            self.cur = (self.cur << 1) | ((value >> i) & 1)
            self.nbits += 1
            if self.nbits == 8:
                self.buf.append(self.cur & 0xFF)
                self.cur = 0
                self.nbits = 0

    def finish(self) -> bytes:
        if self.nbits:
            self.cur <<= (8 - self.nbits)
            self.buf.append(self.cur & 0xFF)
            self.cur = 0
            self.nbits = 0
        return bytes(self.buf)


class _BitReader:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0
        self.cur = 0
        self.nbits = 0

    def read_bits(self, width: int) -> int:
        out = 0
        for _ in range(width):
            if self.nbits == 0:
                if self.pos >= len(self.data):
                    raise ValueError("Elf bitstream underrun")
                self.cur = self.data[self.pos]
                self.pos += 1
                self.nbits = 8
            self.nbits -= 1
            bit = (self.cur >> self.nbits) & 1
            out = (out << 1) | bit
        return out


def _f64_to_u64(v: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", float(v)))[0]


def _u64_to_f64(v: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", v & ((1 << 64) - 1)))[0]


def _leading_zeros64(x: int) -> int:
    if x == 0:
        return 64
    n = 0
    for i in range(63, -1, -1):
        if (x >> i) & 1:
            break
        n += 1
    return n


def _trailing_zeros64(x: int) -> int:
    if x == 0:
        return 64
    n = 0
    while (x & 1) == 0:
        n += 1
        x >>= 1
    return n


def _encode(data: np.ndarray) -> bytes:
    if len(data) == 0:
        return b""

    w = _BitWriter()
    prev = _f64_to_u64(float(data[0]))
    w.write_bits(prev, 64)

    prev_lead = 64
    prev_trail = 0
    prev_center = 64

    for i in range(1, len(data)):
        cur = _f64_to_u64(float(data[i]))
        xor = prev ^ cur
        if xor == 0:
            w.write_bits(0, 1)
        else:
            w.write_bits(1, 1)
            lead = _leading_zeros64(xor)
            trail = _trailing_zeros64(xor)
            # Elf-style: keep at least 1 meaningful bit
            center = max(1, 64 - lead - trail)
            # Prefer reusing prior window when possible (Gorilla/Elf family)
            if lead >= prev_lead and trail >= prev_trail and prev_center > 0:
                w.write_bits(0, 1)
                meaningful = (xor >> prev_trail) & ((1 << prev_center) - 1)
                w.write_bits(meaningful, prev_center)
            else:
                w.write_bits(1, 1)
                w.write_bits(min(lead, 63), 6)
                w.write_bits(min(center, 64), 6)
                meaningful = (xor >> trail) & ((1 << center) - 1)
                w.write_bits(meaningful, center)
                prev_lead, prev_trail, prev_center = lead, trail, center
        prev = cur

    return w.finish()


def _decode(blob: bytes, n: int) -> np.ndarray:
    if n == 0:
        return np.array([], dtype=np.float64)
    r = _BitReader(blob)
    out: List[float] = []
    prev = r.read_bits(64)
    out.append(_u64_to_f64(prev))

    prev_lead = 64
    prev_trail = 0
    prev_center = 64

    for _ in range(1, n):
        flag = r.read_bits(1)
        if flag == 0:
            out.append(_u64_to_f64(prev))
            continue
        reuse = r.read_bits(1)
        if reuse == 0:
            meaningful = r.read_bits(prev_center)
            xor = meaningful << prev_trail
        else:
            lead = r.read_bits(6)
            center = r.read_bits(6)
            if center == 0:
                center = 64
            meaningful = r.read_bits(center)
            trail = 64 - lead - center
            if trail < 0:
                trail = 0
            xor = meaningful << trail
            prev_lead, prev_trail, prev_center = lead, trail, center
        prev = prev ^ xor
        out.append(_u64_to_f64(prev))

    return np.asarray(out, dtype=np.float64)


def compress_elf(data: np.ndarray, abs_error: float = 0.0) -> BaselineResult:
    """
    Lossless Elf-style XOR/erasure bit-packer.
    abs_error is ignored (method is lossless); accepted for API uniformity.
    """
    arr = np.asarray(data, dtype=np.float64).reshape(-1)
    # Non-finite → still encoding raw bits (lossless)
    t0 = time.perf_counter()
    blob = _encode(arr)
    t1 = time.perf_counter()
    recovered = _decode(blob, len(arr))
    t2 = time.perf_counter()

    return BaselineResult(
        name="Elf",
        compressed_size=len(blob),
        recovered=recovered,
        compress_seconds=t1 - t0,
        decompress_seconds=t2 - t1,
        lossless=True,
        backend="pure-python",
        notes="lossless XOR+erasure reference (Elf-family)",
    )
