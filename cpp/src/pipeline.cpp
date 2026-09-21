/**
 * @file pipeline.cpp
 * @brief Parallel block processing and deterministic assembly (IMPROVED).
 *
 * Handles all 6 model types in both compress and decompress paths.
 * Uses per-window adaptive precision.
 * User requirement #2: explicit Sinusoidal and Pred-XOR reconstruction.
 */

#include "dlc/pipeline.hpp"
#include "dlc/encoders.hpp"
#include "dlc/format.hpp"
#include "dlc/models.hpp"
#include "dlc/windowing.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <future>
#include <thread>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace dlc {

// ── Helper: float64 bits ────────────────────────────────────────────────────

static uint64_t f64_to_u64(double v) {
  uint64_t bits;
  std::memcpy(&bits, &v, 8);
  return bits;
}

static double u64_to_f64(uint64_t bits) {
  double v;
  std::memcpy(&v, &bits, 8);
  return v;
}

// ── Process a single block ──────────────────────────────────────────────────

std::vector<uint8_t> process_block(const double *data, size_t n,
                                   const DLCConfig &config) {
  std::vector<uint8_t> result;
  auto spans = segment_block(data, n);

  for (const auto &span : spans) {
    const double *window = data + span.offset;
    size_t win_len = span.length;

    auto model = select_model(window, win_len);

    WindowBlockMeta meta;
    meta.window_length = static_cast<uint32_t>(win_len);

    if (model.model == ModelType::XorDelta) {
      meta.model_id = ModelId::XorDelta;
      meta.encoding_id = EncodingId::ZigzagVarint;
      meta.precision_bits = static_cast<uint8_t>(config.precision_bits);

      auto encoded_data = encode_xor_residuals(model.xor_residuals);
      pack_window_block(meta, model.params, encoded_data, result);

    } else if (model.model == ModelType::PredXor) {
      meta.model_id = ModelId::PredXor;
      meta.encoding_id = EncodingId::ZigzagVarint;
      meta.precision_bits = static_cast<uint8_t>(config.precision_bits);

      auto encoded_data = encode_xor_residuals(model.xor_residuals);
      pack_window_block(meta, model.params, encoded_data, result);

    } else {
      // Analytical models: Linear, Quadratic, Constant, Sinusoidal
      switch (model.model) {
      case ModelType::Linear:
        meta.model_id = ModelId::Linear;
        break;
      case ModelType::Quadratic:
        meta.model_id = ModelId::Quadratic;
        break;
      case ModelType::Constant:
        meta.model_id = ModelId::Constant;
        break;
      case ModelType::Sinusoidal:
        meta.model_id = ModelId::Sinusoidal;
        break;
      default:
        meta.model_id = ModelId::Linear;
        break;
      }

      // Adaptive precision
      int effective_precision =
          select_precision(model.residuals, config.precision_bits);
      meta.precision_bits = static_cast<uint8_t>(effective_precision);

      auto quantized = quantize(model.residuals, effective_precision);
      auto enc = trial_encode(quantized);
      meta.encoding_id = static_cast<EncodingId>(enc.encoding_id);

      pack_window_block(meta, model.params, enc.data, result);
    }
  }

  // ── Block-level DoD fast path ───────────────────────────────────────
  // Try encoding entire block as single Constant(mean) window + DoD encoding.
  // Subtract mean first so residuals are small → better varint compression.
  if (n >= 64) {
    // Compute block mean
    double block_mean = 0.0;
    for (size_t i = 0; i < n; ++i) block_mean += data[i];
    block_mean /= static_cast<double>(n);

    // Residuals = data - mean
    std::vector<double> block_residuals(n);
    for (size_t i = 0; i < n; ++i) block_residuals[i] = data[i] - block_mean;

    int block_precision = select_precision(block_residuals, config.precision_bits);
    auto block_quantized = quantize(block_residuals, block_precision);
    auto dod_encoded = encode_dod(block_quantized);

    WindowBlockMeta dod_meta;
    dod_meta.window_length = static_cast<uint32_t>(n);
    dod_meta.model_id = ModelId::Constant;
    dod_meta.encoding_id = static_cast<EncodingId>(5); // DoD encoding
    dod_meta.precision_bits = static_cast<uint8_t>(block_precision);

    std::vector<double> dod_params = {block_mean};
    std::vector<uint8_t> dod_result;
    pack_window_block(dod_meta, dod_params, dod_encoded, dod_result);

    if (dod_result.size() < result.size()) {
      return dod_result;
    }
  }

  return result;
}

// ── Compress with parallel dispatch ─────────────────────────────────────────

std::vector<uint8_t> compress_parallel(const double *data, size_t n,
                                       const DLCConfig &config) {
  if (n == 0)
    return {};

  int max_workers = static_cast<int>(std::thread::hardware_concurrency());
  if (max_workers < 1)
    max_workers = 4;
  int num_workers = (config.num_workers <= 0)
                        ? max_workers
                        : std::min(config.num_workers, max_workers);

  size_t chunk = config.chunk_size;
  std::vector<std::pair<size_t, size_t>> blocks;
  for (size_t i = 0; i < n; i += chunk) {
    blocks.push_back({i, std::min(chunk, n - i)});
  }

  std::vector<std::future<std::vector<uint8_t>>> futures;
  futures.reserve(blocks.size());

  for (const auto &[off, len] : blocks) {
    futures.push_back(std::async(std::launch::async, [&, off, len]() {
      return process_block(data + off, len, config);
    }));
  }

  std::vector<uint8_t> payload;
  for (auto &f : futures) {
    auto block_data = f.get();
    payload.insert(payload.end(), block_data.begin(), block_data.end());
  }

  return payload;
}

// ── Decompress payload ──────────────────────────────────────────────────────

std::vector<double> decompress_payload(const uint8_t *data, size_t len,
                                       const DLCConfig &config) {
  if (len == 0)
    return {};

  std::vector<double> samples;
  size_t offset = 0;

  while (offset < len) {
    WindowBlockMeta meta;
    std::vector<double> params;
    std::vector<uint8_t> encoded_data;
    size_t consumed =
        unpack_window_block(data, len, offset, meta, params, encoded_data);
    offset += consumed;

    uint8_t model_id = static_cast<uint8_t>(meta.model_id);
    uint32_t win_len = meta.window_length;
    int wp = meta.precision_bits; // per-window precision

    if (model_id == 2) {
      // ── XOR-Delta ───────────────────────────────────────────────────
      double anchor = params[0];
      size_t num_xor = win_len - 1;
      auto xor_res = decode_xor_residuals(encoded_data.data(),
                                          encoded_data.size(), num_xor);

      samples.push_back(anchor);
      uint64_t prev = f64_to_u64(anchor);
      for (size_t i = 0; i < xor_res.size(); ++i) {
        uint64_t curr = prev ^ xor_res[i];
        samples.push_back(u64_to_f64(curr));
        prev = curr;
      }

    } else if (model_id == 5) {
      // ── Pred-XOR (user requirement #2: explicit reconstruction) ────
      double anchor0 = params[0];
      double anchor1 = params[1];
      samples.push_back(anchor0);
      if (win_len >= 2)
        samples.push_back(anchor1);

      size_t num_xor = (win_len > 2) ? win_len - 2 : 0;
      auto xor_res = decode_xor_residuals(encoded_data.data(),
                                          encoded_data.size(), num_xor);

      for (size_t i = 0; i < xor_res.size(); ++i) {
        size_t idx = samples.size();
        double predicted = 2.0 * samples[idx - 1] - samples[idx - 2];
        uint64_t pred_bits = f64_to_u64(predicted);
        uint64_t actual_bits = pred_bits ^ xor_res[i];
        samples.push_back(u64_to_f64(actual_bits));
      }

    } else {
      // ── Analytical models ─────────────────────────────────────────

      std::vector<double> predicted(win_len);

      if (model_id == 0) {
        // Linear: y = m*x + c
        double m = params[0], c = params[1];
        for (uint32_t i = 0; i < win_len; ++i) {
          predicted[i] = m * static_cast<double>(i) + c;
        }

      } else if (model_id == 1) {
        // Quadratic: y = a*x^2 + b*x + c
        double a = params[0], b = params[1], c = params[2];
        for (uint32_t i = 0; i < win_len; ++i) {
          double x = static_cast<double>(i);
          predicted[i] = a * x * x + b * x + c;
        }

      } else if (model_id == 3) {
        // Constant: y = c
        double c = params[0];
        for (uint32_t i = 0; i < win_len; ++i) {
          predicted[i] = c;
        }

      } else if (model_id == 4) {
        // Sinusoidal: y = A*sin(w*x + phi) + dc
        // (user requirement #2: explicit decompress)
        double A = params[0];
        double w = params[1];
        double phi = params[2];
        double dc = params[3];
        for (uint32_t i = 0; i < win_len; ++i) {
          double x = static_cast<double>(i);
          predicted[i] = A * std::sin(w * x + phi) + dc;
        }

      } else {
        // Unknown — treat as constant zero
        for (uint32_t i = 0; i < win_len; ++i) {
          predicted[i] = 0.0;
        }
      }

      // Decode and dequantize residuals
      uint8_t enc_id = static_cast<uint8_t>(meta.encoding_id);
      auto quantized = decode_residuals(enc_id, encoded_data.data(),
                                        encoded_data.size(), win_len);
      auto residuals = dequantize(quantized, wp);

      for (uint32_t i = 0; i < win_len; ++i) {
        samples.push_back(predicted[i] + residuals[i]);
      }
    }
  }

  return samples;
}

} // namespace dlc
