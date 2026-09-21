/**
 * @file test_encoders.cpp
 * @brief GTest tests for residual encoders.
 */

#include "dlc/encoders.hpp"
#include <gtest/gtest.h>


namespace dlc {
namespace test {

TEST(EncodersTest, ZigzagVarintRoundTrip) {
  std::vector<int64_t> vals = {0, 1, -1, 127, -128, 10000, -10000};
  auto encoded = encode_zigzag_varint(vals);
  auto decoded =
      decode_zigzag_varint(encoded.data(), encoded.size(), vals.size());
  EXPECT_EQ(decoded, vals);
}

TEST(EncodersTest, BitpackRoundTrip) {
  std::vector<int64_t> vals = {0, 1, -1, 3, -3, 7, -7};
  auto encoded = encode_bitpack(vals);
  auto decoded = decode_bitpack(encoded.data(), encoded.size(), vals.size());
  EXPECT_EQ(decoded, vals);
}

TEST(EncodersTest, DeltaOfResidualsRoundTrip) {
  std::vector<int64_t> vals = {10, 12, 14, 16, 18};
  auto encoded = encode_delta_of_residuals(vals);
  auto decoded =
      decode_delta_of_residuals(encoded.data(), encoded.size(), vals.size());
  EXPECT_EQ(decoded, vals);
}

TEST(EncodersTest, OutlierSepRoundTrip) {
  std::vector<int64_t> vals = {1, 1, 1, 1, 1000, 1, 1, 1, -1000, 1};
  auto encoded = encode_outlier_sep(vals);
  auto decoded =
      decode_outlier_sep(encoded.data(), encoded.size(), vals.size());
  EXPECT_EQ(decoded, vals);
}

TEST(EncodersTest, QuantizeRoundTrip) {
  std::vector<double> vals = {0.5, -0.25, 0.001};
  auto q = quantize(vals, 12);
  auto d = dequantize(q, 12);
  for (size_t i = 0; i < vals.size(); ++i) {
    EXPECT_NEAR(d[i], vals[i], 0.5 / 4096.0 + 1e-15);
  }
}

TEST(EncodersTest, TrialEncodeDecodesCorrectly) {
  std::vector<int64_t> vals = {1, 2, 3, 4, 5};
  auto result = trial_encode(vals);
  auto decoded = decode_residuals(result.encoding_id, result.data.data(),
                                  result.data.size(), vals.size());
  EXPECT_EQ(decoded, vals);
}

} // namespace test
} // namespace dlc
