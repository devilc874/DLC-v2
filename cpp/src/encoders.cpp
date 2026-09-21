/**
 * @file encoders.cpp
 * @brief Residual quantization and 5 encoding strategies (IMPROVED).
 *
 * New: Strategy 4 (RLE), XOR residual encoding, adaptive precision,
 *      MAD factor tuned to 4.5.
 */

#include "dlc/encoders.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>

namespace dlc {

// ── Quantization ────────────────────────────────────────────────────────────

std::vector<int64_t> quantize(const std::vector<double> &residuals,
                              int precision_bits) {
  int64_t scale = static_cast<int64_t>(1) << precision_bits;
  std::vector<int64_t> result;
  result.reserve(residuals.size());
  for (double r : residuals) {
    result.push_back(
        static_cast<int64_t>(std::round(r * static_cast<double>(scale))));
  }
  return result;
}

std::vector<double> dequantize(const std::vector<int64_t> &quantized,
                               int precision_bits) {
  double scale = static_cast<double>(static_cast<int64_t>(1) << precision_bits);
  std::vector<double> result;
  result.reserve(quantized.size());
  for (int64_t q : quantized) {
    result.push_back(static_cast<double>(q) / scale);
  }
  return result;
}

// ── Adaptive Precision ──────────────────────────────────────────────────────

int select_precision(const std::vector<double> &residuals, int requested) {
  if (residuals.empty())
    return requested;

  double max_abs = 0;
  for (double r : residuals) {
    double a = std::abs(r);
    if (a > max_abs)
      max_abs = a;
  }

  if (max_abs < 1e-15)
    return 8; // MIN_PRECISION

  int min_p_for_bound =
      std::max(8, static_cast<int>(std::ceil(std::log2(1.0 / 8e-6) - 1)));

  int adaptive_p = requested;
  if (max_abs < 1.0) {
    adaptive_p = std::max(
        8, static_cast<int>(std::ceil(-std::log2(max_abs + 1e-30))) + 4);
  }

  int precision =
      std::min(20, std::max(min_p_for_bound, std::min(requested, adaptive_p)));
  return precision;
}

// ── Zigzag ──────────────────────────────────────────────────────────────────

static uint64_t zigzag_encode(int64_t n) {
  return static_cast<uint64_t>((n << 1) ^ (n >> 63));
}

static int64_t zigzag_decode(uint64_t z) {
  return static_cast<int64_t>((z >> 1) ^ -(z & 1));
}

// ── Varint ──────────────────────────────────────────────────────────────────

static void encode_varint(uint64_t value, std::vector<uint8_t> &out) {
  while (value > 0x7F) {
    out.push_back(static_cast<uint8_t>((value & 0x7F) | 0x80));
    value >>= 7;
  }
  out.push_back(static_cast<uint8_t>(value & 0x7F));
}

static uint64_t decode_varint(const uint8_t *data, size_t len, size_t &offset) {
  uint64_t value = 0;
  int shift = 0;
  while (offset < len) {
    uint8_t b = data[offset++];
    value |= static_cast<uint64_t>(b & 0x7F) << shift;
    if ((b & 0x80) == 0)
      return value;
    shift += 7;
  }
  throw std::runtime_error("Truncated varint");
}

// ── Bit width ───────────────────────────────────────────────────────────────

static int bit_width(uint64_t v) {
  if (v == 0)
    return 1;
  int w = 0;
  while (v > 0) {
    ++w;
    v >>= 1;
  }
  return w;
}

// ════════════════════════════════════════════════════════════════════════════
// Strategy 0: Zigzag + Varint
// ════════════════════════════════════════════════════════════════════════════

std::vector<uint8_t> encode_zigzag_varint(const std::vector<int64_t> &vals) {
  std::vector<uint8_t> out;
  for (int64_t v : vals) {
    encode_varint(zigzag_encode(v), out);
  }
  return out;
}

std::vector<int64_t> decode_zigzag_varint(const uint8_t *data, size_t len,
                                          size_t count) {
  std::vector<int64_t> result;
  result.reserve(count);
  size_t offset = 0;
  for (size_t i = 0; i < count; ++i) {
    uint64_t z = decode_varint(data, len, offset);
    result.push_back(zigzag_decode(z));
  }
  return result;
}

// ════════════════════════════════════════════════════════════════════════════
// Strategy 1: Fixed-Width Bitpack
// ════════════════════════════════════════════════════════════════════════════

std::vector<uint8_t> encode_bitpack(const std::vector<int64_t> &vals) {
  if (vals.empty())
    return {0};

  uint64_t max_zz = 0;
  std::vector<uint64_t> zz_vals;
  zz_vals.reserve(vals.size());
  for (int64_t v : vals) {
    uint64_t z = zigzag_encode(v);
    zz_vals.push_back(z);
    if (z > max_zz)
      max_zz = z;
  }

  int bw = (max_zz > 0) ? bit_width(max_zz) : 1;

  size_t total_bits = static_cast<size_t>(bw) * zz_vals.size();
  size_t n_bytes = (total_bits + 7) / 8;

  std::vector<uint8_t> out;
  out.push_back(static_cast<uint8_t>(bw));
  out.resize(1 + n_bytes, 0);

  size_t bit_pos = 0;
  for (uint64_t zv : zz_vals) {
    for (int b = 0; b < bw; ++b) {
      if (zv & (1ULL << b)) {
        size_t byte_idx = 1 + (bit_pos / 8);
        size_t bit_idx = bit_pos % 8;
        out[byte_idx] |= (1 << bit_idx);
      }
      ++bit_pos;
    }
  }

  return out;
}

std::vector<int64_t> decode_bitpack(const uint8_t *data, size_t len,
                                    size_t count) {
  if (count == 0 || len == 0)
    return {};

  int bw = data[0];
  const uint8_t *body = data + 1;

  std::vector<int64_t> result;
  result.reserve(count);

  size_t bit_pos = 0;
  for (size_t i = 0; i < count; ++i) {
    uint64_t zv = 0;
    for (int b = 0; b < bw; ++b) {
      size_t byte_idx = bit_pos / 8;
      size_t bit_idx = bit_pos % 8;
      if (body[byte_idx] & (1 << bit_idx)) {
        zv |= (1ULL << b);
      }
      ++bit_pos;
    }
    result.push_back(zigzag_decode(zv));
  }

  return result;
}

// ════════════════════════════════════════════════════════════════════════════
// Strategy 2: Delta-of-Residuals
// ════════════════════════════════════════════════════════════════════════════

std::vector<uint8_t>
encode_delta_of_residuals(const std::vector<int64_t> &vals) {
  if (vals.empty())
    return {};
  std::vector<int64_t> deltas;
  deltas.reserve(vals.size());
  deltas.push_back(vals[0]);
  for (size_t i = 1; i < vals.size(); ++i) {
    deltas.push_back(vals[i] - vals[i - 1]);
  }
  return encode_zigzag_varint(deltas);
}

std::vector<int64_t> decode_delta_of_residuals(const uint8_t *data, size_t len,
                                               size_t count) {
  if (count == 0)
    return {};
  auto deltas = decode_zigzag_varint(data, len, count);
  std::vector<int64_t> result;
  result.reserve(count);
  result.push_back(deltas[0]);
  for (size_t i = 1; i < deltas.size(); ++i) {
    result.push_back(result.back() + deltas[i]);
  }
  return result;
}

// ════════════════════════════════════════════════════════════════════════════
// Strategy 3: MAD Outlier Separation  (tuned: factor 4.5)
// ════════════════════════════════════════════════════════════════════════════

static constexpr double MAD_FACTOR = 4.5;

static double compute_median(std::vector<double> &vals) {
  if (vals.empty())
    return 0.0;
  size_t n = vals.size();
  std::nth_element(vals.begin(), vals.begin() + n / 2, vals.end());
  if (n % 2 == 1)
    return vals[n / 2];
  double mid = vals[n / 2];
  std::nth_element(vals.begin(), vals.begin() + n / 2 - 1, vals.end());
  return (vals[n / 2 - 1] + mid) / 2.0;
}

static void put_u16le(std::vector<uint8_t> &buf, uint16_t v) {
  buf.push_back(static_cast<uint8_t>(v & 0xFF));
  buf.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
}
static uint16_t get_u16le(const uint8_t *p) {
  return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}
static void put_u32le(std::vector<uint8_t> &buf, uint32_t v) {
  buf.push_back(static_cast<uint8_t>(v & 0xFF));
  buf.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
  buf.push_back(static_cast<uint8_t>((v >> 16) & 0xFF));
  buf.push_back(static_cast<uint8_t>((v >> 24) & 0xFF));
}
static uint32_t get_u32le(const uint8_t *p) {
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) |
         (static_cast<uint32_t>(p[3]) << 24);
}

std::vector<uint8_t> encode_outlier_sep(const std::vector<int64_t> &vals) {
  std::vector<uint8_t> out;
  if (vals.empty()) {
    put_u16le(out, 0);
    out.push_back(0);
    return out;
  }

  std::vector<double> fvals(vals.begin(), vals.end());
  auto fvals_copy = fvals;
  double median = compute_median(fvals_copy);

  std::vector<double> abs_devs(fvals.size());
  for (size_t i = 0; i < fvals.size(); ++i)
    abs_devs[i] = std::abs(fvals[i] - median);
  double mad = compute_median(abs_devs);

  if (mad == 0.0) {
    auto bp = encode_bitpack(vals);
    put_u16le(out, 0);
    out.push_back(bp[0]);
    out.insert(out.end(), bp.begin() + 1, bp.end());
    return out;
  }

  double threshold = MAD_FACTOR * mad;
  std::vector<size_t> outlier_indices;
  std::vector<int64_t> outlier_values;
  std::vector<int64_t> inlier_vals = vals;

  for (size_t i = 0; i < vals.size(); ++i) {
    if (std::abs(static_cast<double>(vals[i]) - median) > threshold) {
      outlier_indices.push_back(i);
      outlier_values.push_back(vals[i]);
      inlier_vals[i] = 0;
    }
  }

  auto inlier_bp = encode_bitpack(inlier_vals);
  uint8_t inlier_bw = inlier_bp[0];

  put_u16le(out, static_cast<uint16_t>(outlier_indices.size()));
  out.push_back(inlier_bw);

  for (size_t j = 0; j < outlier_indices.size(); ++j) {
    put_u32le(out, static_cast<uint32_t>(outlier_indices[j]));
    encode_varint(zigzag_encode(outlier_values[j]), out);
  }

  out.insert(out.end(), inlier_bp.begin() + 1, inlier_bp.end());
  return out;
}

std::vector<int64_t> decode_outlier_sep(const uint8_t *data, size_t len,
                                        size_t count) {
  if (count == 0)
    return {};

  size_t offset = 0;
  uint16_t num_outliers = get_u16le(data + offset);
  offset += 2;
  uint8_t inlier_bw = data[offset];
  offset += 1;

  std::vector<std::pair<uint32_t, int64_t>> outliers;
  for (uint16_t j = 0; j < num_outliers; ++j) {
    uint32_t idx = get_u32le(data + offset);
    offset += 4;
    uint64_t z = decode_varint(data, len, offset);
    outliers.push_back({idx, zigzag_decode(z)});
  }

  std::vector<uint8_t> inlier_buf;
  inlier_buf.push_back(inlier_bw);
  inlier_buf.insert(inlier_buf.end(), data + offset, data + len);
  auto result = decode_bitpack(inlier_buf.data(), inlier_buf.size(), count);

  for (auto &[idx, val] : outliers) {
    result[idx] = val;
  }
  return result;
}

// ════════════════════════════════════════════════════════════════════════════
// Strategy 4: RLE + Varint  (NEW)
// ════════════════════════════════════════════════════════════════════════════

std::vector<uint8_t> encode_rle(const std::vector<int64_t> &vals) {
  if (vals.empty())
    return {};

  std::vector<std::pair<int64_t, size_t>> runs;
  int64_t cur = vals[0];
  size_t cnt = 1;
  for (size_t i = 1; i < vals.size(); ++i) {
    if (vals[i] == cur) {
      ++cnt;
    } else {
      runs.push_back({cur, cnt});
      cur = vals[i];
      cnt = 1;
    }
  }
  runs.push_back({cur, cnt});

  std::vector<uint8_t> out;
  encode_varint(runs.size(), out);
  for (auto &[val, len] : runs) {
    encode_varint(zigzag_encode(val), out);
    encode_varint(len, out);
  }
  return out;
}

std::vector<int64_t> decode_rle(const uint8_t *data, size_t len, size_t count) {
  if (count == 0 || len == 0)
    return {};

  size_t offset = 0;
  uint64_t n_runs = decode_varint(data, len, offset);
  std::vector<int64_t> result;
  result.reserve(count);

  for (uint64_t r = 0; r < n_runs && result.size() < count; ++r) {
    int64_t val = zigzag_decode(decode_varint(data, len, offset));
    uint64_t cnt = decode_varint(data, len, offset);
    for (uint64_t c = 0; c < cnt && result.size() < count; ++c) {
      result.push_back(val);
    }
  }
  return result;
}

// ════════════════════════════════════════════════════════════════════════════
// Strategy 5: Delta-of-Delta + Varint  (DoD killer)
// ════════════════════════════════════════════════════════════════════════════

std::vector<uint8_t> encode_dod(const std::vector<int64_t> &vals) {
  if (vals.empty())
    return {};
  std::vector<uint8_t> out;
  if (vals.size() == 1) {
    encode_varint(zigzag_encode(vals[0]), out);
    return out;
  }
  // First value
  encode_varint(zigzag_encode(vals[0]), out);
  // First delta
  int64_t delta = vals[1] - vals[0];
  encode_varint(zigzag_encode(delta), out);
  // Delta-of-deltas
  int64_t prev_delta = delta;
  for (size_t i = 2; i < vals.size(); ++i) {
    int64_t curr_delta = vals[i] - vals[i - 1];
    int64_t dod = curr_delta - prev_delta;
    encode_varint(zigzag_encode(dod), out);
    prev_delta = curr_delta;
  }
  return out;
}

std::vector<int64_t> decode_dod(const uint8_t *data, size_t len, size_t count) {
  if (count == 0 || len == 0)
    return {};

  size_t offset = 0;
  // First value
  int64_t first_val = zigzag_decode(decode_varint(data, len, offset));
  std::vector<int64_t> result;
  result.reserve(count);
  result.push_back(first_val);

  if (count == 1)
    return result;

  // First delta
  int64_t delta = zigzag_decode(decode_varint(data, len, offset));
  result.push_back(first_val + delta);

  // DoD -> cumulative sums
  int64_t prev_delta = delta;
  for (size_t i = 2; i < count; ++i) {
    int64_t dod = zigzag_decode(decode_varint(data, len, offset));
    int64_t curr_delta = prev_delta + dod;
    result.push_back(result.back() + curr_delta);
    prev_delta = curr_delta;
  }
  return result;
}

// ════════════════════════════════════════════════════════════════════════════
// XOR Residual Encoding (for Models 2 & 5)
// ════════════════════════════════════════════════════════════════════════════

std::vector<uint8_t>
encode_xor_residuals(const std::vector<uint64_t> &xor_res) {
  if (xor_res.empty()) {
    return {0x00};
  }

  // Raw encoding
  std::vector<uint8_t> raw;
  raw.reserve(xor_res.size() * 8);
  for (uint64_t v : xor_res) {
    for (int i = 0; i < 8; ++i) {
      raw.push_back(static_cast<uint8_t>((v >> (i * 8)) & 0xFF));
    }
  }

  // Try delta-of-XOR
  if (xor_res.size() > 1) {
    std::vector<uint8_t> delta_bytes;
    std::vector<uint64_t> deltas;
    deltas.push_back(xor_res[0]);
    for (size_t i = 1; i < xor_res.size(); ++i) {
      deltas.push_back(xor_res[i] - xor_res[i - 1]); // wrapping subtraction
    }
    delta_bytes.reserve(deltas.size() * 8);
    for (uint64_t d : deltas) {
      for (int i = 0; i < 8; ++i) {
        delta_bytes.push_back(static_cast<uint8_t>((d >> (i * 8)) & 0xFF));
      }
    }
    if (delta_bytes.size() < raw.size()) {
      std::vector<uint8_t> result = {0x01};
      result.insert(result.end(), delta_bytes.begin(), delta_bytes.end());
      return result;
    }
  }

  std::vector<uint8_t> result = {0x00};
  result.insert(result.end(), raw.begin(), raw.end());
  return result;
}

std::vector<uint64_t> decode_xor_residuals(const uint8_t *data, size_t len,
                                           size_t count) {
  if (count == 0 || len == 0)
    return {};

  uint8_t flag = data[0];
  const uint8_t *payload = data + 1;
  size_t payload_len = len - 1;

  if (flag == 0x00) {
    // Raw 8-byte LE uint64
    std::vector<uint64_t> result;
    result.reserve(count);
    for (size_t i = 0; i < count && (i * 8 + 7) < payload_len; ++i) {
      uint64_t v = 0;
      for (int j = 0; j < 8; ++j) {
        v |= static_cast<uint64_t>(payload[i * 8 + j]) << (j * 8);
      }
      result.push_back(v);
    }
    return result;
  } else {
    // Delta-encoded
    std::vector<uint64_t> deltas;
    deltas.reserve(count);
    for (size_t i = 0; i < count && (i * 8 + 7) < payload_len; ++i) {
      uint64_t v = 0;
      for (int j = 0; j < 8; ++j) {
        v |= static_cast<uint64_t>(payload[i * 8 + j]) << (j * 8);
      }
      deltas.push_back(v);
    }
    std::vector<uint64_t> result;
    result.reserve(count);
    uint64_t acc = 0;
    for (uint64_t d : deltas) {
      acc += d;
      result.push_back(acc);
    }
    return result;
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Trial Encode — now includes Strategy 5 (DoD)
// ════════════════════════════════════════════════════════════════════════════

EncodeResult trial_encode(const std::vector<int64_t> &quantized) {
  struct Candidate {
    uint8_t id;
    std::vector<uint8_t> data;
  };
  std::vector<Candidate> candidates;
  candidates.push_back({0, encode_zigzag_varint(quantized)});
  candidates.push_back({1, encode_bitpack(quantized)});
  candidates.push_back({2, encode_delta_of_residuals(quantized)});
  if (quantized.size() >= 8)
    candidates.push_back({3, encode_outlier_sep(quantized)});
  candidates.push_back({4, encode_rle(quantized)});
  candidates.push_back({5, encode_dod(quantized)});

  auto best = std::min_element(candidates.begin(), candidates.end(),
                               [](const Candidate &a, const Candidate &b) {
                                 return a.data.size() < b.data.size();
                               });

  return {best->id, std::move(best->data)};
}

std::vector<int64_t> decode_residuals(uint8_t encoding_id, const uint8_t *data,
                                      size_t len, size_t count) {
  switch (encoding_id) {
  case 0:
    return decode_zigzag_varint(data, len, count);
  case 1:
    return decode_bitpack(data, len, count);
  case 2:
    return decode_delta_of_residuals(data, len, count);
  case 3:
    return decode_outlier_sep(data, len, count);
  case 4:
    return decode_rle(data, len, count);
  case 5:
    return decode_dod(data, len, count);
  default:
    throw std::runtime_error("Unknown encoding_id: " +
                             std::to_string(encoding_id));
  }
}

} // namespace dlc

