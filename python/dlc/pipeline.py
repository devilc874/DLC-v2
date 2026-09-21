"""
Parallel block processing and deterministic assembly (IMPROVED).

Handles all 6 model types:
  0 = Linear, 1 = Quadratic, 2 = XOR-Delta,
  3 = Constant, 4 = Sinusoidal, 5 = Pred-XOR

Supports per-window adaptive precision bits and all 5 encoding strategies.
Thread pool capped at os.cpu_count() per addendum #3.
"""

import os
import struct
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from dlc.windowing import segment_block
from dlc.models import (
    select_model,
    MODEL_LINEAR, MODEL_QUADRATIC, MODEL_XOR_DELTA,
    MODEL_CONSTANT, MODEL_SINUSOIDAL, MODEL_PRED_XOR,
    fit_xor_delta, reconstruct_xor_delta,
    fit_pred_xor, reconstruct_pred_xor,
)
from dlc.encoders import (
    quantize, dequantize, trial_encode, decode_residuals,
    select_precision, encode_xor_residuals, decode_xor_residuals,
)
from dlc.format import (
    WindowBlockMeta, pack_window_block, unpack_window_block,
)


def _process_block(block: np.ndarray, precision_bits: int = 12) -> bytes:
    """
    Process a single block (runs entirely within one thread):
      1. Segment into windows
      2. For each window: select model -> encode residuals
      3. Pack window blocks
      4. Also try single-window DoD fast path (beats standalone DoD)
      5. Return the smaller of windowed vs DoD
    """
    windows = segment_block(block)
    parts = []

    for window in windows:
        model_id, params, residuals = select_model(window)

        if model_id == MODEL_XOR_DELTA:
            # XOR-Delta: encode XOR residuals
            anchor, xor_residuals = fit_xor_delta(window)
            encoded_data = encode_xor_residuals(xor_residuals)
            meta = WindowBlockMeta(
                window_length=len(window),
                model_id=model_id,
                encoding_id=0,  # not used for XOR models
                precision_bits=precision_bits,
                model_params=[anchor],
            )
            parts.append(pack_window_block(meta, encoded_data))

        elif model_id == MODEL_PRED_XOR:
            # Pred-XOR: encode XOR residuals with two anchors
            pxor_params, xor_residuals = fit_pred_xor(window)
            encoded_data = encode_xor_residuals(xor_residuals)
            meta = WindowBlockMeta(
                window_length=len(window),
                model_id=model_id,
                encoding_id=0,  # not used for XOR models
                precision_bits=precision_bits,
                model_params=list(pxor_params),
            )
            parts.append(pack_window_block(meta, encoded_data))

        else:
            # Analytical models (Linear, Quadratic, Constant, Sinusoidal)
            # Adaptive precision for this window
            effective_precision = select_precision(residuals, precision_bits)
            quantized = quantize(residuals, effective_precision)
            encoding_id, encoded_data = trial_encode(quantized)

            meta = WindowBlockMeta(
                window_length=len(window),
                model_id=model_id,
                encoding_id=encoding_id,
                precision_bits=effective_precision,
                model_params=params,
            )
            parts.append(pack_window_block(meta, encoded_data))

    windowed_result = b''.join(parts)

    # ── Block-level DoD fast path ─────────────────────────────────────
    # Try encoding the entire block as a single Constant(mean) window
    # with DoD (encoding 5). This eliminates per-window overhead and
    # directly competes with standalone DoD compression.
    # Key: subtract block mean first so residuals are small → better varint
    if len(block) >= 64:
        try:
            from dlc.encoders import encode_dod
            block_mean = float(np.mean(block))
            block_residuals = block - block_mean
            block_precision = select_precision(block_residuals, precision_bits)
            block_quantized = quantize(block_residuals, block_precision)
            dod_encoded = encode_dod(block_quantized)

            dod_meta = WindowBlockMeta(
                window_length=len(block),
                model_id=MODEL_CONSTANT,
                encoding_id=5,  # DoD encoding
                precision_bits=block_precision,
                model_params=[block_mean],
            )
            dod_result = pack_window_block(dod_meta, dod_encoded)

            # Pick the smaller representation
            if len(dod_result) < len(windowed_result):
                return dod_result
        except Exception:
            pass

    return windowed_result


def compress_parallel(data: np.ndarray,
                      precision_bits: int = 12,
                      chunk_size: int = 100_000,
                      num_workers: Optional[int] = None) -> bytes:
    """
    Compress an array of float64s into the uncompressed payload.

    Dispatches blocks to a thread pool, assembles in strict order
    for deterministic output regardless of thread count.
    """
    n = len(data)
    if n == 0:
        return b''

    max_workers = os.cpu_count() or 4
    if num_workers is None or num_workers <= 0:
        num_workers = max_workers
    else:
        num_workers = min(num_workers, max_workers)

    blocks = []
    for i in range(0, n, chunk_size):
        blocks.append(data[i:i + chunk_size])

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = [
            pool.submit(_process_block, block, precision_bits)
            for block in blocks
        ]
        ordered_payloads = [f.result() for f in futures]

    return b''.join(ordered_payloads)


def decompress_payload(payload: bytes, precision_bits: int = 12) -> np.ndarray:
    """
    Decompress an uncompressed payload back into float64 values.

    Handles all 6 model types and 5 encoding strategies.
    Uses per-window precision_bits from each window block header.
    """
    if len(payload) == 0:
        return np.array([], dtype=np.float64)

    samples = []
    offset = 0

    while offset < len(payload):
        meta, encoded_data, consumed = unpack_window_block(payload, offset)
        offset += consumed

        win_len = meta.window_length
        # Use per-window precision (overrides file-level default)
        wp = meta.precision_bits

        if meta.model_id == MODEL_XOR_DELTA:
            # XOR-Delta: decode XOR residuals and reconstruct
            anchor = meta.model_params[0]
            num_xor = win_len - 1
            xor_residuals = decode_xor_residuals(encoded_data, num_xor)
            reconstructed = reconstruct_xor_delta(anchor, xor_residuals)
            samples.append(reconstructed)

        elif meta.model_id == MODEL_PRED_XOR:
            # Pred-XOR: two anchors + predictive XOR residuals
            anchor0 = meta.model_params[0]
            anchor1 = meta.model_params[1]
            num_xor = win_len - 2
            xor_residuals = decode_xor_residuals(encoded_data, max(0, num_xor))
            reconstructed = reconstruct_pred_xor(anchor0, anchor1, xor_residuals)
            samples.append(reconstructed)

        elif meta.model_id == MODEL_LINEAR:
            # Linear: y = m*x + c
            m, c = meta.model_params[0], meta.model_params[1]
            x = np.arange(win_len, dtype=np.float64)
            predicted = m * x + c
            quantized = decode_residuals(meta.encoding_id, encoded_data, win_len)
            residuals = dequantize(quantized, wp)
            samples.append(predicted + residuals)

        elif meta.model_id == MODEL_QUADRATIC:
            # Quadratic: y = a*x^2 + b*x + c
            a, b, c = meta.model_params[0], meta.model_params[1], meta.model_params[2]
            x = np.arange(win_len, dtype=np.float64)
            predicted = a * x * x + b * x + c
            quantized = decode_residuals(meta.encoding_id, encoded_data, win_len)
            residuals = dequantize(quantized, wp)
            samples.append(predicted + residuals)

        elif meta.model_id == MODEL_CONSTANT:
            # Constant: y = c
            c = meta.model_params[0]
            predicted = np.full(win_len, c, dtype=np.float64)
            quantized = decode_residuals(meta.encoding_id, encoded_data, win_len)
            residuals = dequantize(quantized, wp)
            samples.append(predicted + residuals)

        elif meta.model_id == MODEL_SINUSOIDAL:
            # Sinusoidal: y = A*sin(w*x + phi) + dc
            A = meta.model_params[0]
            w = meta.model_params[1]
            phi = meta.model_params[2]
            dc = meta.model_params[3]
            x = np.arange(win_len, dtype=np.float64)
            predicted = A * np.sin(w * x + phi) + dc
            quantized = decode_residuals(meta.encoding_id, encoded_data, win_len)
            residuals = dequantize(quantized, wp)
            samples.append(predicted + residuals)

        else:
            raise ValueError(f"Unknown model_id: {meta.model_id}")

    return np.concatenate(samples) if samples else np.array([], dtype=np.float64)
