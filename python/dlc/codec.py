"""
High-level compress / decompress API for .dlc files.

Usage:
    from dlc.codec import compress, decompress
    compress(data, "output.dlc")
    result = decompress("output.dlc")
"""

import numpy as np
from typing import Optional

from dlc.format import DLCHeader, write_dlc_file, read_dlc_file
from dlc.pipeline import compress_parallel, decompress_payload


def compress(data: np.ndarray,
             output_path: str,
             precision_bits: int = 12,
             chunk_size: int = 100_000,
             num_workers: Optional[int] = None):
    """
    Compress float64 data and write a .dlc file.

    Args:
        data:           1D numpy array of float64.
        output_path:    Path to output .dlc file.
        precision_bits: Quantization precision (default 12).
        chunk_size:     Block size for parallel processing (default 100000).
        num_workers:    Thread count (default = auto).
    """
    header = DLCHeader(
        precision_bits=precision_bits,
        chunk_size=chunk_size,
        total_samples=len(data),
    )

    payload = compress_parallel(data, precision_bits, chunk_size, num_workers)
    write_dlc_file(output_path, header, payload)


def decompress(input_path: str) -> np.ndarray:
    """
    Decompress a .dlc file and return the float64 signal.

    Validates magic bytes, CRC-32, and footer integrity.

    Args:
        input_path: Path to .dlc file.

    Returns:
        1D numpy array of float64 values.
    """
    header, payload = read_dlc_file(input_path)
    return decompress_payload(payload, header.precision_bits)
