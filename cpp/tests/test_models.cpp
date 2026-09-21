/**
 * @file test_models.cpp
 * @brief GTest tests for predictive models.
 */

#include "dlc/models.hpp"
#include <cmath>
#include <gtest/gtest.h>


namespace dlc {
namespace test {

TEST(ModelsTest, LinearPerfectFit) {
  std::vector<double> y(100);
  for (size_t i = 0; i < 100; ++i)
    y[i] = 2.0 * static_cast<double>(i) + 5.0;
  auto result = fit_linear(y.data(), y.size());
  EXPECT_NEAR(result.params[0], 2.0, 1e-8);
  EXPECT_NEAR(result.params[1], 5.0, 1e-8);
}

TEST(ModelsTest, QuadraticPerfectFit) {
  std::vector<double> y(50);
  for (size_t i = 0; i < 50; ++i) {
    double x = static_cast<double>(i);
    y[i] = 0.5 * x * x - 3.0 * x + 1.0;
  }
  auto result = fit_quadratic(y.data(), y.size());
  EXPECT_NEAR(result.params[0], 0.5, 1e-6);
  EXPECT_NEAR(result.params[1], -3.0, 1e-6);
  EXPECT_NEAR(result.params[2], 1.0, 1e-6);
}

TEST(ModelsTest, XorDeltaLossless) {
  std::vector<double> y = {1.0, 2.5, -3.0, 0.0, 99.9};
  auto result = fit_xor_delta(y.data(), y.size());
  EXPECT_EQ(result.params[0], 1.0);
  EXPECT_EQ(result.xor_residuals.size(), 4u);
}

TEST(ModelsTest, SelectModelReturnsValid) {
  std::vector<double> y(100);
  for (size_t i = 0; i < 100; ++i)
    y[i] = static_cast<double>(i);
  auto result = select_model(y.data(), y.size());
  uint8_t mid = static_cast<uint8_t>(result.model);
  EXPECT_TRUE(mid == 0 || mid == 1 || mid == 2);
}

} // namespace test
} // namespace dlc
