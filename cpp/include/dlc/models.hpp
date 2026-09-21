#pragma once
/**
 * @file models.hpp
 * @brief Predictive models: Linear, Quadratic, XOR-Delta, Constant,
 *        Sinusoidal, Pred-XOR.
 */

#include <cstddef>
#include <cstdint>
#include <vector>

namespace dlc {

enum class ModelType : uint8_t {
  Linear = 0,
  Quadratic = 1,
  XorDelta = 2,
  Constant = 3,
  Sinusoidal = 4,
  PredXor = 5,
};

struct ModelResult {
  ModelType model;
  std::vector<double> params;
  std::vector<double> residuals; // For Linear/Quadratic/Constant/Sinusoidal
  std::vector<uint64_t> xor_residuals; // For XOR-Delta / Pred-XOR only
  double ssr = 0.0;
};

ModelResult fit_linear(const double *y, size_t n);
ModelResult fit_quadratic(const double *y, size_t n);
ModelResult fit_xor_delta(const double *y, size_t n);
ModelResult fit_constant(const double *y, size_t n);
ModelResult fit_sinusoidal(const double *y, size_t n);
ModelResult fit_pred_xor(const double *y, size_t n);

ModelResult select_model(const double *y, size_t n);

} // namespace dlc
