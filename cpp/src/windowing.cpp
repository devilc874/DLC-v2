/**
 * @file windowing.cpp
 * @brief Adaptive variance-guided window segmentation (IMPROVED).
 *
 * Changes: Adaptive MAX_WINDOW (512/1024/4096), stagnation guard,
 *          constant-block fast-path, tuned thresholds (10.0, 0.10).
 */

#include "dlc/windowing.hpp"
#include <algorithm>
#include <cmath>

namespace dlc {

static constexpr double MEAN_VAR_FACTOR = 10.0;
static constexpr double GLOBAL_VAR_FACTOR = 0.10;
static constexpr double CONST_VAR_EPS = 1e-20;

static double variance(const double *data, size_t n) {
  if (n < 2)
    return 0.0;
  double mean = 0.0;
  for (size_t i = 0; i < n; ++i)
    mean += data[i];
  mean /= static_cast<double>(n);
  double var = 0.0;
  for (size_t i = 0; i < n; ++i) {
    double d = data[i] - mean;
    var += d * d;
  }
  return var / static_cast<double>(n);
}

static size_t adaptive_max_window(size_t block_len) {
  if (block_len < 5000)
    return std::min<size_t>(512, block_len);
  if (block_len < 100000)
    return std::min<size_t>(1024, block_len);
  return std::min<size_t>(4096, block_len);
}

std::vector<Span> segment_block(const double *data, size_t n, size_t min_w,
                                size_t max_w) {
  std::vector<Span> windows;
  if (n == 0)
    return windows;

  if (n <= min_w) {
    windows.push_back({0, n});
    return windows;
  }

  // Fast-path: constant block
  double global_var = variance(data, n);
  if (global_var < CONST_VAR_EPS) {
    windows.push_back({0, n});
    return windows;
  }

  // Use adaptive max if default was passed
  if (max_w == 1024)
    max_w = adaptive_max_window(n);

  size_t i = 0;

  while (i < n) {
    size_t remaining = n - i;

    if (remaining <= min_w) {
      windows.push_back({i, remaining});
      break;
    }

    size_t win_size = std::min(min_w, remaining);
    double initial_var = variance(data + i, win_size);

    double threshold =
        std::max(initial_var * MEAN_VAR_FACTOR, global_var * GLOBAL_VAR_FACTOR);
    if (threshold == 0.0)
      threshold = 1e-30;

    // Greedy doubling with stagnation guard
    size_t candidate = win_size;
    double prev_var = initial_var;

    while (candidate < max_w && i + candidate < n) {
      size_t next_size = std::min({candidate * 2, max_w, remaining});
      if (next_size <= candidate) {
        candidate = next_size;
        break;
      }
      double seg_var = variance(data + i, next_size);

      if (seg_var > threshold)
        break;

      // Stagnation guard
      if (prev_var > CONST_VAR_EPS && seg_var / prev_var > 1.5)
        break;

      prev_var = seg_var;
      candidate = next_size;
    }

    // Binary search for exact boundary
    size_t lo = win_size;
    size_t hi = std::min(candidate, remaining);

    while (lo < hi) {
      size_t mid = (lo + hi + 1) / 2;
      if (variance(data + i, mid) <= threshold) {
        lo = mid;
      } else {
        hi = mid - 1;
      }
    }

    win_size = lo;
    windows.push_back({i, win_size});
    i += win_size;
  }

  return windows;
}

} // namespace dlc
