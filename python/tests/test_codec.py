"""
End-to-end codec round-trip tests (codec.py).

Tests file-based compress/decompress, CRC integrity, and error bounds.
"""

import os
import tempfile
import numpy as np
import pytest

from dlc.codec import compress, decompress


class TestCodec:
    def test_round_trip_sine(self):
        data = np.sin(np.linspace(0, 10 * np.pi, 5000))
        with tempfile.NamedTemporaryFile(suffix='.dlc', delete=False) as f:
            path = f.name

        try:
            compress(data, path, precision_bits=12, chunk_size=1000, num_workers=2)
            result = decompress(path)
            assert len(result) == len(data)
            max_err = np.max(np.abs(data - result))
            assert max_err < 8e-6, f"Max error {max_err} >= 8e-6"
        finally:
            os.unlink(path)

    def test_round_trip_ramp(self):
        data = np.linspace(-100, 100, 10_000)
        with tempfile.NamedTemporaryFile(suffix='.dlc', delete=False) as f:
            path = f.name

        try:
            compress(data, path, num_workers=1)
            result = decompress(path)
            max_err = np.max(np.abs(data - result))
            assert max_err < 8e-6
        finally:
            os.unlink(path)

    def test_round_trip_random(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal(2000)
        with tempfile.NamedTemporaryFile(suffix='.dlc', delete=False) as f:
            path = f.name

        try:
            compress(data, path, num_workers=2)
            result = decompress(path)
            max_err = np.max(np.abs(data - result))
            assert max_err < 8e-6
        finally:
            os.unlink(path)

    def test_corrupted_file_raises(self):
        data = np.sin(np.linspace(0, np.pi, 200))
        with tempfile.NamedTemporaryFile(suffix='.dlc', delete=False) as f:
            path = f.name

        try:
            compress(data, path, num_workers=1)

            # Corrupt a byte in the middle
            with open(path, 'r+b') as f:
                content = bytearray(f.read())
                mid = len(content) // 2
                content[mid] ^= 0xFF
                f.seek(0)
                f.write(content)
                f.truncate()

            with pytest.raises(Exception):
                decompress(path)
        finally:
            os.unlink(path)

    def test_deterministic_output(self):
        """Same data compressed twice should produce identical files."""
        data = np.arange(1000, dtype=np.float64)

        with tempfile.NamedTemporaryFile(suffix='.dlc', delete=False) as f1:
            p1 = f1.name
        with tempfile.NamedTemporaryFile(suffix='.dlc', delete=False) as f2:
            p2 = f2.name

        try:
            compress(data, p1, num_workers=1)
            compress(data, p2, num_workers=4)

            with open(p1, 'rb') as f:
                b1 = f.read()
            with open(p2, 'rb') as f:
                b2 = f.read()

            assert b1 == b2, "Files differ between 1 and 4 workers"
        finally:
            os.unlink(p1)
            os.unlink(p2)
