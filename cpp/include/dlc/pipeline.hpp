#pragma once
/**
 * @file pipeline.hpp
 * @brief Parallel block processing and deterministic assembly.
 */

#include "dlc/format.hpp" // DLCConfig defined here
#include <cstdint>
#include <vector>


namespace dlc {

/**
 * Process a single block: segment → model → encode → pack window blocks.
 */
std::vector<uint8_t> process_block(const double *data, size_t n,
                                   const DLCConfig &config);

/**
 * Compress an array of float64 into the uncompressed payload (pre-zlib).
 */
std::vector<uint8_t> compress_parallel(const double *data, size_t n,
                                       const DLCConfig &config);

/**
 * Decompress an uncompressed payload back into float64 values.
 */
std::vector<double> decompress_payload(const uint8_t *data, size_t len,
                                       const DLCConfig &config);

} // namespace dlc
