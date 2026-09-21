"""
Tests for DLC binary format (format.py).

Tests header/footer round-trip, exact byte sizes, window block
serialization, CRC-32 integrity, and full file write/read.
"""

import os
import struct
import tempfile
import zlib

import pytest

from dlc.format import (
    DLCHeader, DLCFooter, WindowBlockMeta,
    MAGIC_BYTES, FOOTER_MAGIC,
    pack_header, unpack_header,
    pack_footer, unpack_footer,
    pack_window_block, unpack_window_block,
    write_dlc_file, read_dlc_file,
)


# ── Header Tests ─────────────────────────────────────────────────────────────

class TestHeader:
    def test_header_size_is_64_bytes(self):
        buf = pack_header(DLCHeader())
        assert len(buf) == 64

    def test_header_round_trip_default(self):
        h = DLCHeader()
        buf = pack_header(h)
        h2 = unpack_header(buf)
        assert h2.magic == MAGIC_BYTES
        assert h2.major_version == 0
        assert h2.minor_version == 1
        assert h2.precision_bits == 12
        assert h2.chunk_size == 100_000
        assert h2.total_samples == 0

    def test_header_round_trip_custom(self):
        h = DLCHeader(
            precision_bits=16,
            chunk_size=50_000,
            total_samples=1_000_000,
        )
        buf = pack_header(h)
        h2 = unpack_header(buf)
        assert h2.precision_bits == 16
        assert h2.chunk_size == 50_000
        assert h2.total_samples == 1_000_000

    def test_header_invalid_magic_raises(self):
        buf = bytearray(pack_header(DLCHeader()))
        buf[0:4] = b'XXXX'
        with pytest.raises(ValueError, match="Invalid magic"):
            unpack_header(bytes(buf))

    def test_header_too_short_raises(self):
        with pytest.raises(ValueError, match="too short"):
            unpack_header(b'\x00' * 10)

    def test_header_little_endian(self):
        """Verify precision_bits field is at offset 8, little-endian."""
        h = DLCHeader(precision_bits=0x1234)
        buf = pack_header(h)
        # offset 8, 2 bytes LE: 0x34, 0x12
        assert buf[8] == 0x34
        assert buf[9] == 0x12


# ── Footer Tests ─────────────────────────────────────────────────────────────

class TestFooter:
    def test_footer_size_is_8_bytes(self):
        buf = pack_footer(0)
        assert len(buf) == 8

    def test_footer_round_trip(self):
        crc_val = 0xDEADBEEF
        buf = pack_footer(crc_val)
        f = unpack_footer(buf)
        assert f.magic == FOOTER_MAGIC
        assert f.crc32 == crc_val

    def test_footer_invalid_magic_raises(self):
        buf = bytearray(pack_footer(0))
        buf[0:4] = b'\x00\x00\x00\x00'
        with pytest.raises(ValueError, match="Invalid footer"):
            unpack_footer(bytes(buf))

    def test_footer_crc_unsigned(self):
        """CRC should always be stored as unsigned 32-bit."""
        crc = zlib.crc32(b'test data') & 0xFFFFFFFF
        buf = pack_footer(crc)
        f = unpack_footer(buf)
        assert f.crc32 == crc
        assert f.crc32 >= 0


# ── Window Block Tests ───────────────────────────────────────────────────────

class TestWindowBlock:
    def test_linear_round_trip(self):
        meta = WindowBlockMeta(
            window_length=256,
            model_id=0,
            encoding_id=1,
            model_params=[1.5, -0.3],
        )
        residuals = b'\x01\x02\x03\x04'
        buf = pack_window_block(meta, residuals)
        m2, r2, consumed = unpack_window_block(buf)
        assert m2.window_length == 256
        assert m2.model_id == 0
        assert m2.encoding_id == 1
        assert len(m2.model_params) == 2
        assert abs(m2.model_params[0] - 1.5) < 1e-15
        assert abs(m2.model_params[1] - (-0.3)) < 1e-15
        assert r2 == residuals
        assert consumed == len(buf)

    def test_quadratic_round_trip(self):
        meta = WindowBlockMeta(
            window_length=512,
            model_id=1,
            encoding_id=0,
            model_params=[0.001, -2.0, 100.0],
        )
        residuals = bytes(range(20))
        buf = pack_window_block(meta, residuals)
        m2, r2, consumed = unpack_window_block(buf)
        assert m2.model_id == 1
        assert len(m2.model_params) == 3
        assert r2 == residuals

    def test_xor_delta_round_trip(self):
        meta = WindowBlockMeta(
            window_length=16,
            model_id=2,
            encoding_id=3,
            model_params=[3.14159],
        )
        residuals = b'\xFF' * 10
        buf = pack_window_block(meta, residuals)
        m2, r2, _ = unpack_window_block(buf)
        assert m2.model_id == 2
        assert len(m2.model_params) == 1
        assert abs(m2.model_params[0] - 3.14159) < 1e-10

    def test_multiple_blocks_sequential(self):
        """Pack multiple blocks and unpack them from a concatenated buffer."""
        blocks_data = []
        for i in range(5):
            meta = WindowBlockMeta(
                window_length=100 + i,
                model_id=0,
                encoding_id=0,
                model_params=[float(i), float(i * 2)],
            )
            res = bytes([i] * (10 + i))
            blocks_data.append((meta, res))

        # Concatenate
        full_buf = b''.join(pack_window_block(m, r) for m, r in blocks_data)

        # Unpack sequentially
        offset = 0
        for i, (orig_meta, orig_res) in enumerate(blocks_data):
            meta, res, consumed = unpack_window_block(full_buf, offset)
            assert meta.window_length == orig_meta.window_length
            assert res == orig_res
            offset += consumed

        assert offset == len(full_buf)

    def test_wrong_param_count_raises(self):
        meta = WindowBlockMeta(
            window_length=16,
            model_id=0,
            encoding_id=0,
            model_params=[1.0],  # Linear needs 2 params, not 1
        )
        with pytest.raises(ValueError, match="expects 2 params"):
            pack_window_block(meta, b'')


# ── Full File Round-Trip Tests ───────────────────────────────────────────────

class TestFileIO:
    def test_write_read_round_trip(self):
        header = DLCHeader(total_samples=42)
        payload = b'Hello DLC binary format test data!'

        with tempfile.NamedTemporaryFile(suffix='.dlc', delete=False) as f:
            path = f.name

        try:
            write_dlc_file(path, header, payload)
            h2, p2 = read_dlc_file(path)
            assert h2.total_samples == 42
            assert p2 == payload
        finally:
            os.unlink(path)

    def test_corrupted_payload_fails_crc(self):
        header = DLCHeader()
        payload = b'valid payload bytes'

        with tempfile.NamedTemporaryFile(suffix='.dlc', delete=False) as f:
            path = f.name

        try:
            write_dlc_file(path, header, payload)

            # Corrupt one byte in the compressed data region
            with open(path, 'r+b') as f:
                data = bytearray(f.read())
                # Flip a bit in the middle of compressed data
                mid = 64 + 4 + len(data) // 4
                if mid < len(data) - 8:
                    data[mid] ^= 0x01
                    f.seek(0)
                    f.write(data)
                    f.truncate()

            # Should raise on CRC or decompression error
            with pytest.raises((ValueError, zlib.error)):
                read_dlc_file(path)
        finally:
            os.unlink(path)

    def test_empty_payload(self):
        header = DLCHeader(total_samples=0)
        payload = b''

        with tempfile.NamedTemporaryFile(suffix='.dlc', delete=False) as f:
            path = f.name

        try:
            write_dlc_file(path, header, payload)
            h2, p2 = read_dlc_file(path)
            assert p2 == b''
        finally:
            os.unlink(path)
