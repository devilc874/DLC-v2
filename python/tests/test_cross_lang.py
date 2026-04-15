"""
Cross-language Python ↔ C++ tests.

Test 1: Python compress → C++ decompress
Test 2: C++ compress → Python decompress
Test 3: Bit-for-bit identity (Python .dlc == C++ .dlc)

Requirements: C++ binary must be built at cpp/build/dlc (or cpp/build/Release/dlc.exe on Windows).
"""

import os
import sys
import subprocess
import tempfile
import hashlib
import numpy as np
import pytest

from dlc.codec import compress, decompress


# ── Locate C++ binary ───────────────────────────────────────────────────────

def _find_cpp_binary():
    """Find the compiled C++ DLC binary."""
    candidates = [
        os.path.join(os.path.dirname(__file__), '..', '..', 'cpp', 'build', 'Release', 'dlc.exe'),
        os.path.join(os.path.dirname(__file__), '..', '..', 'cpp', 'build', 'dlc.exe'),
        os.path.join(os.path.dirname(__file__), '..', '..', 'cpp', 'build', 'dlc'),
    ]
    for c in candidates:
        p = os.path.abspath(c)
        if os.path.isfile(p):
            return p
    return None


CPP_BINARY = _find_cpp_binary()
SKIP_REASON = "C++ binary not found — build with cmake first"


# ── Test Data ────────────────────────────────────────────────────────────────

def _make_test_signal(n=5000):
    """Generate a deterministic test signal."""
    t = np.linspace(0, 10 * np.pi, n)
    return np.sin(t) + 0.3 * np.cos(3 * t)


# ── Tests ────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(CPP_BINARY is None, reason=SKIP_REASON)
class TestCrossLanguage:

    def test_python_compress_cpp_decompress(self, tmp_path):
        """Compress in Python, decompress in C++, verify error bound."""
        data = _make_test_signal()
        dlc_path = str(tmp_path / "test.dlc")
        out_path = str(tmp_path / "recovered.bin")

        # Compress with Python
        compress(data, dlc_path, num_workers=1)

        # Decompress with C++
        result = subprocess.run(
            [CPP_BINARY, "decompress", "-i", dlc_path, "-o", out_path],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"C++ decompress failed: {result.stderr}"

        # Verify
        recovered = np.fromfile(out_path, dtype='<f8')
        assert len(recovered) == len(data)
        max_err = np.max(np.abs(data - recovered))
        assert max_err < 8e-6, f"Max error {max_err} >= 8e-6"

    def test_cpp_compress_python_decompress(self, tmp_path):
        """Compress in C++, decompress in Python, verify error bound."""
        data = _make_test_signal()
        bin_path = str(tmp_path / "input.bin")
        dlc_path = str(tmp_path / "test.dlc")

        # Write raw float64
        data.astype('<f8').tofile(bin_path)

        # Compress with C++
        result = subprocess.run(
            [CPP_BINARY, "compress", "-i", bin_path, "-o", dlc_path, "--workers", "1"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"C++ compress failed: {result.stderr}"

        # Decompress with Python
        recovered = decompress(dlc_path)
        assert len(recovered) == len(data)
        max_err = np.max(np.abs(data - recovered))
        assert max_err < 8e-6, f"Max error {max_err} >= 8e-6"

    def test_bitidentical_dlc_files(self, tmp_path):
        """Python and C++ must produce bit-for-bit identical .dlc files."""
        data = _make_test_signal(1000)
        bin_path = str(tmp_path / "input.bin")
        py_dlc = str(tmp_path / "python.dlc")
        cpp_dlc = str(tmp_path / "cpp.dlc")

        data.astype('<f8').tofile(bin_path)

        # Python compress
        compress(data, py_dlc, num_workers=1)

        # C++ compress
        result = subprocess.run(
            [CPP_BINARY, "compress", "-i", bin_path, "-o", cpp_dlc, "--workers", "1"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"C++ compress failed: {result.stderr}"

        # Compare SHA-256
        with open(py_dlc, 'rb') as f:
            py_hash = hashlib.sha256(f.read()).hexdigest()
        with open(cpp_dlc, 'rb') as f:
            cpp_hash = hashlib.sha256(f.read()).hexdigest()

        assert py_hash == cpp_hash, (
            f"Bit-for-bit mismatch!\nPython SHA-256: {py_hash}\nC++ SHA-256: {cpp_hash}"
        )
