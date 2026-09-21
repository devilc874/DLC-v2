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
             precision_bits: int = 16,
             chunk_size: int = 100_000,
             num_workers: Optional[int] = None,
             enforce_error_bound: bool = True,
             max_error_bound: float = 8e-6):
    """
    Compress float64 data and write a .dlc file.

    Args:
        data:                 1D numpy array of float64.
        output_path:          Path to output .dlc file.
        precision_bits:       Requested precision cap (8–20, default 16).
        chunk_size:           Block size for parallel processing.
        num_workers:          Thread count (default = auto).
        enforce_error_bound:  True (default) → 16-bit floor for 8e-6 contract.
                              False → ablation; full adaptive 8–20 range.
        max_error_bound:      Absolute error used to derive the production floor.
    """
    header = DLCHeader(
        precision_bits=precision_bits,
        chunk_size=chunk_size,
        total_samples=len(data),
    )

    payload = compress_parallel(
        data, precision_bits, chunk_size, num_workers,
        enforce_error_bound=enforce_error_bound,
        max_error_bound=max_error_bound,
    )
    write_dlc_file(output_path, header, payload)


def decompress(input_path: str) -> np.ndarray:
    """
    Decompress a .dlc file and return the float64 signal.

    Validates magic bytes, CRC-32, and footer integrity.
    """
    header, payload = read_dlc_file(input_path)
    return decompress_payload(payload, header.precision_bits)
