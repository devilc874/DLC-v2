/**
 * @file format.cpp
 * @brief DLC binary format serialization (IMPROVED).
 *
 * Window block now includes per-window precision_bits (uint8).
 * New model IDs: 3=Constant, 4=Sinusoidal, 5=PredXor.
 */

#include "dlc/format.hpp"
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <zlib.h>

namespace dlc {

// ── Helper: write/read raw bytes ────────────────────────────────────────────

static void write_bytes(std::ostream &out, const void *data, size_t n) {
  out.write(reinterpret_cast<const char *>(data),
            static_cast<std::streamsize>(n));
  if (!out)
    throw std::runtime_error("Failed to write to output stream");
}

static void read_bytes(std::istream &in, void *data, size_t n) {
  in.read(reinterpret_cast<char *>(data), static_cast<std::streamsize>(n));
  if (!in)
    throw std::runtime_error("Failed to read from input stream");
}

// ── LE helpers ──────────────────────────────────────────────────────────────

static void put_u8(std::vector<uint8_t> &buf, uint8_t v) { buf.push_back(v); }

static void put_le32(std::vector<uint8_t> &buf, uint32_t v) {
  buf.push_back(static_cast<uint8_t>(v & 0xFF));
  buf.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
  buf.push_back(static_cast<uint8_t>((v >> 16) & 0xFF));
  buf.push_back(static_cast<uint8_t>((v >> 24) & 0xFF));
}

static void put_le64(std::vector<uint8_t> &buf, uint64_t v) {
  for (int i = 0; i < 8; ++i) {
    buf.push_back(static_cast<uint8_t>((v >> (i * 8)) & 0xFF));
  }
}

static void put_le_double(std::vector<uint8_t> &buf, double v) {
  uint64_t bits;
  std::memcpy(&bits, &v, 8);
  put_le64(buf, bits);
}

static uint8_t get_u8(const uint8_t *p) { return p[0]; }

static uint32_t get_le32(const uint8_t *p) {
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) |
         (static_cast<uint32_t>(p[3]) << 24);
}

static uint64_t get_le64(const uint8_t *p) {
  uint64_t v = 0;
  for (int i = 0; i < 8; ++i) {
    v |= static_cast<uint64_t>(p[i]) << (i * 8);
  }
  return v;
}

static double get_le_double(const uint8_t *p) {
  uint64_t bits = get_le64(p);
  double v;
  std::memcpy(&v, &bits, 8);
  return v;
}

// ── Model param counts ──────────────────────────────────────────────────────

static size_t model_param_count(uint8_t model_id) {
  switch (model_id) {
  case 0:
    return 2; // Linear: m, c
  case 1:
    return 3; // Quadratic: a, b, c
  case 2:
    return 1; // XOR-Delta: anchor
  case 3:
    return 1; // Constant: mean
  case 4:
    return 4; // Sinusoidal: A, w, phi, dc
  case 5:
    return 2; // Pred-XOR: anchor0, anchor1
  default:
    throw std::runtime_error("Unknown model_id: " + std::to_string(model_id));
  }
}

// ── Header / Footer ─────────────────────────────────────────────────────────

void pack_header(const DLCHeader &h, std::ostream &out) {
  write_bytes(out, &h, sizeof(DLCHeader));
}

DLCHeader unpack_header(std::istream &in) {
  DLCHeader h{};
  read_bytes(in, &h, sizeof(DLCHeader));
  if (std::memcmp(h.magic, MAGIC_BYTES, 4) != 0) {
    throw std::runtime_error("Invalid DLC magic bytes");
  }
  return h;
}

void pack_footer(const DLCFooter &f, std::ostream &out) {
  write_bytes(out, &f, sizeof(DLCFooter));
}

DLCFooter unpack_footer(std::istream &in) {
  DLCFooter f{};
  read_bytes(in, &f, sizeof(DLCFooter));
  if (std::memcmp(f.magic, FOOTER_MAGIC, 4) != 0) {
    throw std::runtime_error("Invalid DLC footer magic");
  }
  return f;
}

// ── Window Block Pack ───────────────────────────────────────────────────────

void pack_window_block(const WindowBlockMeta &meta,
                       const std::vector<double> &model_params,
                       const std::vector<uint8_t> &encoded_residuals,
                       std::vector<uint8_t> &out) {
  size_t expected = model_param_count(static_cast<uint8_t>(meta.model_id));
  if (model_params.size() != expected) {
    throw std::runtime_error(
        "Model " + std::to_string(static_cast<int>(meta.model_id)) +
        " expects " + std::to_string(expected) + " params, got " +
        std::to_string(model_params.size()));
  }

  // window_length (4B) + model_id (1B) + encoding_id (1B) + precision_bits (1B)
  // = 7B
  put_le32(out, meta.window_length);
  put_u8(out, static_cast<uint8_t>(meta.model_id));
  put_u8(out, static_cast<uint8_t>(meta.encoding_id));
  put_u8(out, meta.precision_bits);

  // model params
  for (double p : model_params) {
    put_le_double(out, p);
  }

  // encoded residuals
  put_le32(out, static_cast<uint32_t>(encoded_residuals.size()));
  out.insert(out.end(), encoded_residuals.begin(), encoded_residuals.end());
}

// ── Window Block Unpack ─────────────────────────────────────────────────────

size_t unpack_window_block(const uint8_t *data, size_t len, size_t offset,
                           WindowBlockMeta &meta, std::vector<double> &params,
                           std::vector<uint8_t> &encoded_residuals) {
  size_t pos = offset;

  // 7 bytes: window_length(4) + model_id(1) + encoding_id(1) +
  // precision_bits(1)
  if (pos + 7 > len) {
    throw std::runtime_error("Window block: unexpected end of data (header)");
  }

  meta.window_length = get_le32(data + pos);
  meta.model_id = static_cast<ModelId>(get_u8(data + pos + 4));
  meta.encoding_id = static_cast<EncodingId>(get_u8(data + pos + 5));
  meta.precision_bits = get_u8(data + pos + 6);
  pos += 7;

  // Model params
  size_t n_params = model_param_count(static_cast<uint8_t>(meta.model_id));
  if (pos + n_params * 8 > len) {
    throw std::runtime_error("Window block: unexpected end of data (params)");
  }

  params.clear();
  params.reserve(n_params);
  for (size_t i = 0; i < n_params; ++i) {
    params.push_back(get_le_double(data + pos));
    pos += 8;
  }

  // Encoded residuals
  if (pos + 4 > len) {
    throw std::runtime_error("Window block: unexpected end of data (res len)");
  }
  uint32_t res_len = get_le32(data + pos);
  pos += 4;

  if (pos + res_len > len) {
    throw std::runtime_error(
        "Window block: unexpected end of data (residuals)");
  }
  encoded_residuals.assign(data + pos, data + pos + res_len);
  pos += res_len;

  return pos - offset;
}

} // namespace dlc
