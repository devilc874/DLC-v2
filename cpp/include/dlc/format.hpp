#pragma once
/**
 * @file format.hpp
 * @brief DLC binary file format: header, footer, window block I/O.
 *
 * All structs are packed (#pragma pack) and all multi-byte fields
 * are explicitly little-endian.
 *
 * Window block now includes per-window precision_bits (uint8).
 */

#include <array>
#include <cstdint>
#include <iosfwd>
#include <vector>


namespace dlc {

// ── Portable Little-Endian Helpers ──────────────────────────────────────────

inline uint16_t write_le16(uint16_t v) { return v; }
inline uint32_t write_le32(uint32_t v) { return v; }
inline uint64_t write_le64(uint64_t v) { return v; }
inline uint16_t read_le16(uint16_t v) { return v; }
inline uint32_t read_le32(uint32_t v) { return v; }
inline uint64_t read_le64(uint64_t v) { return v; }

// ── Constants ───────────────────────────────────────────────────────────────

constexpr uint8_t MAGIC_BYTES[4] = {'D', 'L', 'C', 0x02};
constexpr uint8_t FOOTER_MAGIC[4] = {0xED, 0xDC, 0xBA, 0x01};
constexpr uint16_t VERSION_MAJOR = 0;
constexpr uint16_t VERSION_MINOR = 1;
constexpr uint16_t DEFAULT_PRECISION = 12;
constexpr uint32_t DEFAULT_CHUNK_SIZE = 100000;

// ── Header (64 bytes, packed) ───────────────────────────────────────────────

#pragma pack(push, 1)
struct DLCHeader {
  uint8_t magic[4];
  uint16_t major_version;
  uint16_t minor_version;
  uint16_t precision_bits;
  uint32_t chunk_size;
  uint64_t total_samples;
  uint8_t reserved[42];
};
static_assert(sizeof(DLCHeader) == 64, "DLCHeader must be exactly 64 bytes");
#pragma pack(pop)

// ── Footer (8 bytes, packed) ────────────────────────────────────────────────

#pragma pack(push, 1)
struct DLCFooter {
  uint8_t magic[4];
  uint32_t crc32;
};
static_assert(sizeof(DLCFooter) == 8, "DLCFooter must be exactly 8 bytes");
#pragma pack(pop)

// ── Model / Encoding IDs ───────────────────────────────────────────────────

enum class ModelId : uint8_t {
  Linear = 0,
  Quadratic = 1,
  XorDelta = 2,
  Constant = 3,
  Sinusoidal = 4,
  PredXor = 5,
};

enum class EncodingId : uint8_t {
  ZigzagVarint = 0,
  Bitpack = 1,
  DeltaOfResiduals = 2,
  OutlierSep = 3,
  Rle = 4,
};

// ── Window Block Metadata ───────────────────────────────────────────────────

struct WindowBlockMeta {
  uint32_t window_length;
  ModelId model_id;
  EncodingId encoding_id;
  uint8_t precision_bits; // per-window adaptive precision
};

// ── DLC Config ──────────────────────────────────────────────────────────────

struct DLCConfig {
  int precision_bits = DEFAULT_PRECISION;
  int chunk_size = DEFAULT_CHUNK_SIZE;
  int num_workers = 0; // 0 = auto
};

// ── Serialization Functions ─────────────────────────────────────────────────

void pack_header(const DLCHeader &h, std::ostream &out);
DLCHeader unpack_header(std::istream &in);

void pack_footer(const DLCFooter &f, std::ostream &out);
DLCFooter unpack_footer(std::istream &in);

void pack_window_block(const WindowBlockMeta &meta,
                       const std::vector<double> &model_params,
                       const std::vector<uint8_t> &encoded_residuals,
                       std::vector<uint8_t> &out);

size_t unpack_window_block(const uint8_t *data, size_t len, size_t offset,
                           WindowBlockMeta &meta,
                           std::vector<double> &model_params,
                           std::vector<uint8_t> &encoded_residuals);

} // namespace dlc
