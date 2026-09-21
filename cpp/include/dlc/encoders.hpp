#pragma once
/**
 * @file encoders.hpp
 * @brief Residual quantization and 6 encoding strategies.
 */

#include <cstdint>
#include <vector>

namespace dlc {

// ── Quantization ────────────────────────────────────────────────────────────

std::vector<int64_t> quantize(const std::vector<double> &residuals,
                              int precision_bits = 12);

std::vector<double> dequantize(const std::vector<int64_t> &quantized,
                               int precision_bits = 12);

// ── Encoding Strategies ─────────────────────────────────────────────────────

std::vector<uint8_t> encode_zigzag_varint(const std::vector<int64_t> &vals);
std::vector<int64_t> decode_zigzag_varint(const uint8_t *data, size_t len,
                                          size_t count);

std::vector<uint8_t> encode_bitpack(const std::vector<int64_t> &vals);
std::vector<int64_t> decode_bitpack(const uint8_t *data, size_t len,
                                    size_t count);

std::vector<uint8_t>
encode_delta_of_residuals(const std::vector<int64_t> &vals);
std::vector<int64_t> decode_delta_of_residuals(const uint8_t *data, size_t len,
                                               size_t count);

std::vector<uint8_t> encode_outlier_sep(const std::vector<int64_t> &vals);
std::vector<int64_t> decode_outlier_sep(const uint8_t *data, size_t len,
                                        size_t count);

// ── NEW: RLE encoding ───────────────────────────────────────────────────────

std::vector<uint8_t> encode_rle(const std::vector<int64_t> &vals);
std::vector<int64_t> decode_rle(const uint8_t *data, size_t len, size_t count);

// ── NEW: DoD encoding (Delta-of-Delta) ──────────────────────────────────────

std::vector<uint8_t> encode_dod(const std::vector<int64_t> &vals);
std::vector<int64_t> decode_dod(const uint8_t *data, size_t len, size_t count);

// ── NEW: XOR residual encoding (for Models 2 & 5) ──────────────────────────

std::vector<uint8_t> encode_xor_residuals(const std::vector<uint64_t> &xor_res);
std::vector<uint64_t> decode_xor_residuals(const uint8_t *data, size_t len,
                                           size_t count);

// ── Trial Encoder ───────────────────────────────────────────────────────────

struct EncodeResult {
  uint8_t encoding_id;
  std::vector<uint8_t> data;
};

EncodeResult trial_encode(const std::vector<int64_t> &quantized);

std::vector<int64_t> decode_residuals(uint8_t encoding_id, const uint8_t *data,
                                      size_t len, size_t count);

// ── Adaptive precision ─────────────────────────────────────────────────────

/** Bits required so quantization error 2^(-(p+1)) <= max_error_bound. */
int precision_floor_for_error_bound(double max_error_bound = 8e-6);

/**
 * Choose per-window precision (adaptive).
 * enforce_error_bound=true  → production floor (16 for 8e-6); adapt typically 16–20.
 * enforce_error_bound=false → ablation; floor at 8 so full 8–20 range is available.
 */
int select_precision(const std::vector<double> &residuals, int requested = 16,
                     bool enforce_error_bound = true,
                     double max_error_bound = 8e-6);

} // namespace dlc
