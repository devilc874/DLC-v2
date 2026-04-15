#pragma once
/**
 * @file codec.hpp
 * @brief High-level compress/decompress API — file I/O + zlib.
 */

#include <cstdint>
#include <string>
#include <vector>


namespace dlc {

struct CompressOptions {
  int precision_bits = 12;
  uint32_t chunk_size = 100000;
  int num_workers = 0; // 0 = auto
};

/**
 * Compress float64 data and write a .dlc file.
 */
void compress(const double *data, size_t n, const std::string &output_path,
              const CompressOptions &opts = {});

/**
 * Decompress a .dlc file and return the float64 signal.
 */
std::vector<double> decompress(const std::string &input_path);

} // namespace dlc
