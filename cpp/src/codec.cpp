/**
 * @file codec.cpp
 * @brief High-level compress/decompress with file I/O + zlib.
 *
 * File format: Header(64B) | UncompSize(4B) | ZlibData(NB) | Footer(8B)
 */

#include "dlc/codec.hpp"
#include "dlc/format.hpp"
#include "dlc/pipeline.hpp"

#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <zlib.h>


namespace dlc {

// ── Helper: zlib compress ───────────────────────────────────────────────────

static std::vector<uint8_t> zlib_compress(const std::vector<uint8_t> &data,
                                          int level = 6) {
  uLongf dest_len = compressBound(static_cast<uLong>(data.size()));
  std::vector<uint8_t> compressed(dest_len);
  int ret = compress2(compressed.data(), &dest_len, data.data(),
                      static_cast<uLong>(data.size()), level);
  if (ret != Z_OK) {
    throw std::runtime_error("zlib compress failed: " + std::to_string(ret));
  }
  compressed.resize(dest_len);
  return compressed;
}

static std::vector<uint8_t> zlib_decompress(const uint8_t *data, size_t len,
                                            size_t uncomp_size) {
  std::vector<uint8_t> result(uncomp_size);
  uLongf dest_len = static_cast<uLongf>(uncomp_size);
  int ret = uncompress(result.data(), &dest_len, data, static_cast<uLong>(len));
  if (ret != Z_OK) {
    throw std::runtime_error("zlib decompress failed: " + std::to_string(ret));
  }
  result.resize(dest_len);
  return result;
}

// ── Helper: write little-endian uint32 ──────────────────────────────────────

static void write_le32(std::ostream &out, uint32_t v) {
  uint8_t buf[4];
  buf[0] = static_cast<uint8_t>(v & 0xFF);
  buf[1] = static_cast<uint8_t>((v >> 8) & 0xFF);
  buf[2] = static_cast<uint8_t>((v >> 16) & 0xFF);
  buf[3] = static_cast<uint8_t>((v >> 24) & 0xFF);
  out.write(reinterpret_cast<const char *>(buf), 4);
}

static uint32_t read_le32(const uint8_t *p) {
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) |
         (static_cast<uint32_t>(p[3]) << 24);
}

// ── Compress ────────────────────────────────────────────────────────────────

void compress(const double *data, size_t n, const std::string &output_path,
              const CompressOptions &opts) {
  // Build header
  DLCHeader header{};
  std::memcpy(header.magic, MAGIC_BYTES, 4);
  header.major_version = VERSION_MAJOR;
  header.minor_version = VERSION_MINOR;
  header.precision_bits = static_cast<uint16_t>(opts.precision_bits);
  header.chunk_size = opts.chunk_size;
  header.total_samples = static_cast<uint64_t>(n);
  std::memset(header.reserved, 0, 42);

  // Build pipeline config
  DLCConfig config;
  config.precision_bits = opts.precision_bits;
  config.chunk_size = opts.chunk_size;
  config.num_workers = opts.num_workers;

  // Compress
  auto payload = compress_parallel(data, n, config);

  // CRC-32
  uint32_t crc = static_cast<uint32_t>(
      crc32(0L, payload.data(), static_cast<uInt>(payload.size())));

  // Zlib compress
  auto compressed = zlib_compress(payload, 6);

  // Write file
  std::ofstream out(output_path, std::ios::binary);
  if (!out)
    throw std::runtime_error("Cannot open file for writing: " + output_path);

  pack_header(header, out);
  write_le32(out, static_cast<uint32_t>(payload.size()));
  out.write(reinterpret_cast<const char *>(compressed.data()),
            static_cast<std::streamsize>(compressed.size()));

  DLCFooter footer{};
  std::memcpy(footer.magic, FOOTER_MAGIC, 4);
  footer.crc32 = crc;
  pack_footer(footer, out);

  out.close();
}

// ── Decompress ──────────────────────────────────────────────────────────────

std::vector<double> decompress(const std::string &input_path) {
  // Read entire file
  std::ifstream in(input_path, std::ios::binary);
  if (!in)
    throw std::runtime_error("Cannot open file for reading: " + input_path);

  std::vector<uint8_t> file_data((std::istreambuf_iterator<char>(in)),
                                 std::istreambuf_iterator<char>());
  in.close();

  if (file_data.size() < 64 + 4 + 8) {
    throw std::runtime_error("File too short to be a valid .dlc file");
  }

  // Parse header
  std::istringstream hdr_stream(
      std::string(reinterpret_cast<const char *>(file_data.data()), 64));
  DLCHeader header = unpack_header(hdr_stream);

  // Uncompressed size
  uint32_t uncomp_size = read_le32(file_data.data() + 64);

  // Footer (last 8 bytes)
  size_t footer_offset = file_data.size() - 8;
  std::istringstream ftr_stream(std::string(
      reinterpret_cast<const char *>(file_data.data() + footer_offset), 8));
  DLCFooter footer = unpack_footer(ftr_stream);

  // Compressed data
  const uint8_t *comp_data = file_data.data() + 68;
  size_t comp_len = footer_offset - 68;

  auto payload = zlib_decompress(comp_data, comp_len, uncomp_size);

  if (payload.size() != uncomp_size) {
    throw std::runtime_error("Uncompressed size mismatch");
  }

  // Verify CRC-32
  uint32_t expected_crc = static_cast<uint32_t>(
      crc32(0L, payload.data(), static_cast<uInt>(payload.size())));
  if (expected_crc != footer.crc32) {
    throw std::runtime_error("CRC-32 mismatch");
  }

  // Decompress payload
  DLCConfig config;
  config.precision_bits = header.precision_bits;
  return decompress_payload(payload.data(), payload.size(), config);
}

} // namespace dlc
