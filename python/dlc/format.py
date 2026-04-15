"""
DLC binary file format: header, footer, window block I/O.

All packing uses little-endian ('<') byte ordering via the struct module.
CRC-32 is always masked to unsigned 32-bit: zlib.crc32(data) & 0xFFFFFFFF.

Binary Layout:
  Header  (64 B) | Uncomp Size (4 B) | Zlib Data (N B) | Footer (8 B)
"""

import struct
import zlib
from dataclasses import dataclass, field
from typing import List, Tuple


# ── Constants ────────────────────────────────────────────────────────────────

MAGIC_BYTES = b'DLC\x02'
FOOTER_MAGIC = b'\xED\xDC\xBA\x01'
VERSION_MAJOR = 0
VERSION_MINOR = 1
DEFAULT_PRECISION = 12
DEFAULT_CHUNK_SIZE = 100_000

# struct format strings (all little-endian)
_HEADER_FMT = '<4s HH H I Q 42s'       # 4+2+2+2+4+8+42 = 64 bytes
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
assert _HEADER_SIZE == 64, f"Header must be 64 bytes, got {_HEADER_SIZE}"

_FOOTER_FMT = '<4s I'                   # 4+4 = 8 bytes
_FOOTER_SIZE = struct.calcsize(_FOOTER_FMT)
assert _FOOTER_SIZE == 8, f"Footer must be 8 bytes, got {_FOOTER_SIZE}"

_UNCOMP_SIZE_FMT = '<I'                 # 4 bytes
_UNCOMP_SIZE_LEN = 4


# ── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class DLCHeader:
    """64-byte DLC file header."""
    magic: bytes = MAGIC_BYTES
    major_version: int = VERSION_MAJOR
    minor_version: int = VERSION_MINOR
    precision_bits: int = DEFAULT_PRECISION
    chunk_size: int = DEFAULT_CHUNK_SIZE
    total_samples: int = 0
    reserved: bytes = b'\x00' * 42


@dataclass
class DLCFooter:
    """8-byte DLC file footer."""
    magic: bytes = FOOTER_MAGIC
    crc32: int = 0


@dataclass
class WindowBlockMeta:
    """Metadata for a single window block inside the payload."""
    window_length: int = 0
    model_id: int = 0          # 0=Linear, 1=Quad, 2=XOR, 3=Const, 4=Sine, 5=PredXOR
    encoding_id: int = 0       # 0=ZzVar, 1=Bitpack, 2=DeltaRes, 3=OutlierSep, 4=RLE
    precision_bits: int = 12   # per-window adaptive precision (uint8)
    model_params: List[float] = field(default_factory=list)


# ── Model param sizes (number of float64 values per model_id) ────────────────

_MODEL_PARAM_COUNTS = {
    0: 2,   # Linear:    m, c
    1: 3,   # Quadratic: a, b, c
    2: 1,   # XOR-Delta: anchor
    3: 1,   # Constant:  mean
    4: 4,   # Sinusoidal: A, w, phi, dc
    5: 2,   # Pred-XOR:  anchor0, anchor1
}


# ── Header Pack / Unpack ─────────────────────────────────────────────────────

def pack_header(header: DLCHeader) -> bytes:
    """Serialize a DLCHeader to exactly 64 bytes (little-endian)."""
    reserved = header.reserved
    if len(reserved) < 42:
        reserved = reserved + b'\x00' * (42 - len(reserved))
    return struct.pack(
        _HEADER_FMT,
        header.magic,
        header.major_version,
        header.minor_version,
        header.precision_bits,
        header.chunk_size,
        header.total_samples,
        reserved[:42],
    )


def unpack_header(buf: bytes) -> DLCHeader:
    """Deserialize 64 bytes into a DLCHeader."""
    if len(buf) < _HEADER_SIZE:
        raise ValueError(f"Header buffer too short: {len(buf)} < {_HEADER_SIZE}")
    fields = struct.unpack(_HEADER_FMT, buf[:_HEADER_SIZE])
    h = DLCHeader(
        magic=fields[0],
        major_version=fields[1],
        minor_version=fields[2],
        precision_bits=fields[3],
        chunk_size=fields[4],
        total_samples=fields[5],
        reserved=fields[6],
    )
    if h.magic != MAGIC_BYTES:
        raise ValueError(f"Invalid magic bytes: {h.magic!r}, expected {MAGIC_BYTES!r}")
    return h


# ── Footer Pack / Unpack ─────────────────────────────────────────────────────

def pack_footer(crc32_val: int) -> bytes:
    """Serialize footer: 4-byte magic + 4-byte CRC-32 (little-endian)."""
    return struct.pack(_FOOTER_FMT, FOOTER_MAGIC, crc32_val & 0xFFFFFFFF)


def unpack_footer(buf: bytes) -> DLCFooter:
    """Deserialize 8 bytes into a DLCFooter."""
    if len(buf) < _FOOTER_SIZE:
        raise ValueError(f"Footer buffer too short: {len(buf)} < {_FOOTER_SIZE}")
    magic, crc = struct.unpack(_FOOTER_FMT, buf[:_FOOTER_SIZE])
    if magic != FOOTER_MAGIC:
        raise ValueError(f"Invalid footer magic: {magic!r}")
    return DLCFooter(magic=magic, crc32=crc)


# ── Window Block Pack / Unpack ───────────────────────────────────────────────

def pack_window_block(meta: WindowBlockMeta, encoded_residuals: bytes) -> bytes:
    """
    Pack a single window block into bytes:
      window_length    (uint32_le)
      model_id         (uint8)
      encoding_id      (uint8)
      precision_bits   (uint8)   ← per-window adaptive precision
      model_params     (N × float64_le, N depends on model_id)
      encoded_residuals_len (uint32_le)
      encoded_residuals     (raw bytes)
    """
    parts = []

    # window_length + model_id + encoding_id + precision_bits
    parts.append(struct.pack('<I B B B',
                             meta.window_length,
                             meta.model_id,
                             meta.encoding_id,
                             meta.precision_bits))

    # model params
    param_count = _MODEL_PARAM_COUNTS.get(meta.model_id, len(meta.model_params))
    if len(meta.model_params) != param_count:
        raise ValueError(
            f"Model {meta.model_id} expects {param_count} params, got {len(meta.model_params)}"
        )
    for p in meta.model_params:
        parts.append(struct.pack('<d', p))

    # encoded residuals
    parts.append(struct.pack('<I', len(encoded_residuals)))
    parts.append(encoded_residuals)

    return b''.join(parts)


def unpack_window_block(buf: bytes, offset: int = 0) -> Tuple[WindowBlockMeta, bytes, int]:
    """
    Unpack a single window block from buf starting at offset.

    Returns:
        (meta, encoded_residuals, bytes_consumed)
    """
    pos = offset

    # window_length + model_id + encoding_id + precision_bits (4+1+1+1 = 7 bytes)
    window_length, model_id, encoding_id, precision_bits = struct.unpack_from(
        '<I B B B', buf, pos)
    pos += 7

    # model params
    param_count = _MODEL_PARAM_COUNTS.get(model_id)
    if param_count is None:
        raise ValueError(f"Unknown model_id: {model_id}")

    model_params = []
    for _ in range(param_count):
        (val,) = struct.unpack_from('<d', buf, pos)
        model_params.append(val)
        pos += 8

    # encoded residuals
    (res_len,) = struct.unpack_from('<I', buf, pos)
    pos += 4
    encoded_residuals = buf[pos:pos + res_len]
    pos += res_len

    meta = WindowBlockMeta(
        window_length=window_length,
        model_id=model_id,
        encoding_id=encoding_id,
        precision_bits=precision_bits,
        model_params=model_params,
    )
    return meta, encoded_residuals, pos - offset


# ── Full File Write / Read ───────────────────────────────────────────────────

def write_dlc_file(filepath: str, header: DLCHeader, uncompressed_payload: bytes):
    """
    Write a complete .dlc file:
      Header (64 B) | Uncomp Size (4 B) | Zlib Data (N B) | Footer (8 B)
    """
    crc = zlib.crc32(uncompressed_payload) & 0xFFFFFFFF
    compressed = zlib.compress(uncompressed_payload, 6)

    with open(filepath, 'wb') as f:
        f.write(pack_header(header))
        f.write(struct.pack(_UNCOMP_SIZE_FMT, len(uncompressed_payload)))
        f.write(compressed)
        f.write(pack_footer(crc))


def read_dlc_file(filepath: str) -> Tuple[DLCHeader, bytes]:
    """
    Read a .dlc file and return (header, uncompressed_payload).
    Validates magic bytes, footer magic, and CRC-32.
    """
    with open(filepath, 'rb') as f:
        data = f.read()

    # Header
    header = unpack_header(data[:_HEADER_SIZE])

    # Uncompressed size
    pos = _HEADER_SIZE
    (uncomp_size,) = struct.unpack_from(_UNCOMP_SIZE_FMT, data, pos)
    pos += _UNCOMP_SIZE_LEN

    # Footer is last 8 bytes
    footer_buf = data[-_FOOTER_SIZE:]
    footer = unpack_footer(footer_buf)

    # Zlib data is everything between uncomp_size field and footer
    compressed = data[pos:-_FOOTER_SIZE]
    uncompressed = zlib.decompress(compressed)

    if len(uncompressed) != uncomp_size:
        raise ValueError(
            f"Uncompressed size mismatch: header says {uncomp_size}, got {len(uncompressed)}"
        )

    # CRC-32 verify
    expected_crc = zlib.crc32(uncompressed) & 0xFFFFFFFF
    if expected_crc != footer.crc32:
        raise ValueError(
            f"CRC-32 mismatch: expected {expected_crc:#010x}, got {footer.crc32:#010x}"
        )

    return header, uncompressed
