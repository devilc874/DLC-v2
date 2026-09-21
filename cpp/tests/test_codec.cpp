/**
 * @file test_codec.cpp
 * @brief GTest tests for high-level codec.
 */

#include "dlc/codec.hpp"
#include <cmath>
#include <cstdio>
#include <gtest/gtest.h>


namespace dlc {
namespace test {

TEST(CodecTest, RoundTripSine) {
  std::vector<double> data(5000);
  for (size_t i = 0; i < data.size(); ++i)
    data[i] = std::sin(static_cast<double>(i) * 0.01);

  std::string path = "test_codec_sine.dlc";
  CompressOptions opts;
  opts.num_workers = 2;
  opts.chunk_size = 1000;

  compress(data.data(), data.size(), path, opts);
  auto result = decompress(path);

  ASSERT_EQ(result.size(), data.size());
  double max_err = 0;
  for (size_t i = 0; i < data.size(); ++i)
    max_err = std::max(max_err, std::abs(data[i] - result[i]));
  EXPECT_LT(max_err, 8e-6);

  std::remove(path.c_str());
}

TEST(CodecTest, DefaultConfig) {
  CompressOptions opts;
  EXPECT_EQ(opts.precision_bits, 12);
  EXPECT_EQ(opts.chunk_size, 100000u);
}

} // namespace test
} // namespace dlc
