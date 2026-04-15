/**
 * @file test_windowing.cpp
 * @brief GTest tests for adaptive windowing.
 */

#include "dlc/windowing.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <numeric>


namespace dlc {
namespace test {

TEST(WindowingTest, ConstantSignalMinimalSplits) {
  std::vector<double> data(2000, 1.0);
  auto spans = segment_block(data.data(), data.size());
  size_t total = 0;
  for (const auto &s : spans)
    total += s.length;
  EXPECT_EQ(total, 2000u);
}

TEST(WindowingTest, AllWindowsWithinBounds) {
  std::vector<double> data(5000);
  for (size_t i = 0; i < data.size(); ++i)
    data[i] = std::sin(static_cast<double>(i) * 0.01);
  auto spans = segment_block(data.data(), data.size());

  for (size_t i = 0; i < spans.size(); ++i) {
    EXPECT_LE(spans[i].length, 1024u) << "Window " << i << " too large";
    if (i < spans.size() - 1) {
      EXPECT_GE(spans[i].length, 16u) << "Window " << i << " too small";
    }
  }
}

TEST(WindowingTest, NoGapsNoOverlaps) {
  std::vector<double> data(3000);
  for (size_t i = 0; i < data.size(); ++i)
    data[i] = static_cast<double>(i);
  auto spans = segment_block(data.data(), data.size());

  size_t expected = 0;
  for (const auto &s : spans) {
    EXPECT_EQ(s.offset, expected);
    expected += s.length;
  }
  EXPECT_EQ(expected, 3000u);
}

TEST(WindowingTest, EmptyInput) {
  auto spans = segment_block(nullptr, 0);
  EXPECT_TRUE(spans.empty());
}

TEST(WindowingTest, TinyInput) {
  std::vector<double> data = {1.0, 2.0, 3.0};
  auto spans = segment_block(data.data(), data.size());
  EXPECT_EQ(spans.size(), 1u);
  EXPECT_EQ(spans[0].length, 3u);
}

} // namespace test
} // namespace dlc
