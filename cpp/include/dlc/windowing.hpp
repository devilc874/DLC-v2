#pragma once
/**
 * @file windowing.hpp
 * @brief Adaptive variance-guided window segmentation.
 */

#include <cstddef>
#include <vector>

namespace dlc {

struct Span {
    size_t offset;
    size_t length;
};

/**
 * Segment a block of float64 data into variable-length windows
 * using variance-guided greedy expansion + binary search.
 *
 * @param data   Pointer to the signal block.
 * @param n      Number of samples in the block.
 * @param min_w  Minimum window size (default 16).
 * @param max_w  Maximum window size (default 1024).
 * @return       Vector of Span {offset, length} covering the block with no gaps.
 */
std::vector<Span> segment_block(const double* data, size_t n,
                                size_t min_w = 16, size_t max_w = 1024);

}  // namespace dlc
