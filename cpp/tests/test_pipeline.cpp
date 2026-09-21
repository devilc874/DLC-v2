/**
 * @file test_pipeline.cpp
 * @brief GTest tests for parallel pipeline.
 */

#include "dlc/pipeline.hpp"
#include <cmath>
#include <gtest/gtest.h>


namespace dlc {
namespace test {

TEST(PipelineTest, SmallBlockRoundTrip) {
  std::vector<double> data(200);
  for (size_t i = 0; i < 200; ++i)
    data[i] = std::sin(static_cast<double>(i) * 0.05);

  DLCConfig config;
  config.precision_bits = 12;
  auto payload = process_block(data.data(), data.size(), config);
  auto result = decompress_payload(payload.data(), payload.size(), config);

  ASSERT_EQ(result.size(), data.size());
  double max_err = 0;
  for (size_t i = 0; i < data.size(); ++i) {
    max_err = std::max(max_err, std::abs(data[i] - result[i]));
  }
  EXPECT_LT(max_err, 8e-6);
}

TEST(PipelineTest, DeterminismAcrossWorkers) {
  std::vector<double> data(10000);
  for (size_t i = 0; i < data.size(); ++i)
    data[i] = std::sin(static_cast<double>(i) * 0.01);

  DLCConfig config1, config2;
  config1.chunk_size = 2000;
  config1.num_workers = 1;
  config2.chunk_size = 2000;
  config2.num_workers = 4;

  auto p1 = compress_parallel(data.data(), data.size(), config1);
  auto p2 = compress_parallel(data.data(), data.size(), config2);

  EXPECT_EQ(p1, p2) << "Output differs between 1 and 4 workers!";
}

TEST(PipelineTest, EmptyData) {
  DLCConfig config;
  auto payload = compress_parallel(nullptr, 0, config);
  EXPECT_TRUE(payload.empty());
}

} // namespace test
} // namespace dlc
