/**
 * @file models.cpp
 * @brief Predictive models: Linear, Quadratic, XOR-Delta, Constant,
 *        Sinusoidal, Pred-XOR.
 */

#include "dlc/models.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace dlc {

// ── Helper: float64 bits ────────────────────────────────────────────────────

static uint64_t float64_to_bits(double v) {
  uint64_t bits;
  std::memcpy(&bits, &v, 8);
  return bits;
}

static double bits_to_float64(uint64_t bits) {
  double v;
  std::memcpy(&v, &bits, 8);
  return v;
}

// ── Helper: SSR ─────────────────────────────────────────────────────────────

static double compute_ssr(const std::vector<double> &res) {
  double s = 0;
  for (double r : res)
    s += r * r;
  return s;
}

static double xor_popcount_ssr(const std::vector<uint64_t> &xor_res) {
  double s = 0;
  for (uint64_t x : xor_res) {
    // popcount as SSR proxy
    int cnt = 0;
    uint64_t v = x;
    while (v) {
      cnt += (v & 1);
      v >>= 1;
    }
    s += cnt;
  }
  return s;
}

// ── Model 0: Linear ────────────────────────────────────────────────────────

ModelResult fit_linear(const double *y, size_t n) {
  ModelResult r;
  r.model = ModelType::Linear;

  if (n < 2) {
    double c = (n == 1) ? y[0] : 0.0;
    r.params = {0.0, c};
    r.residuals.resize(n, 0.0);
    r.ssr = 0.0;
    return r;
  }

  double sx = 0, sy = 0, sxy = 0, sx2 = 0;
  for (size_t i = 0; i < n; ++i) {
    double x = static_cast<double>(i);
    sx += x;
    sy += y[i];
    sxy += x * y[i];
    sx2 += x * x;
  }
  double dn = static_cast<double>(n);
  double denom = dn * sx2 - sx * sx;
  double m = (std::abs(denom) > 1e-12) ? (dn * sxy - sx * sy) / denom : 0.0;
  double c = (sy - m * sx) / dn;

  r.params = {m, c};
  r.residuals.resize(n);
  for (size_t i = 0; i < n; ++i) {
    r.residuals[i] = y[i] - (m * static_cast<double>(i) + c);
  }
  r.ssr = compute_ssr(r.residuals);
  return r;
}

// ── Model 1: Quadratic ─────────────────────────────────────────────────────

ModelResult fit_quadratic(const double *y, size_t n) {
  ModelResult r;
  r.model = ModelType::Quadratic;

  if (n < 3) {
    auto lin = fit_linear(y, n);
    r.params = {0.0, lin.params[0], lin.params[1]};
    r.residuals = lin.residuals;
    r.ssr = lin.ssr;
    return r;
  }

  double s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0;
  double sy = 0, sxy = 0, sx2y = 0;

  for (size_t i = 0; i < n; ++i) {
    double x = static_cast<double>(i);
    double x2 = x * x, x3 = x2 * x, x4 = x3 * x;
    s0 += 1.0;
    s1 += x;
    s2 += x2;
    s3 += x3;
    s4 += x4;
    sy += y[i];
    sxy += x * y[i];
    sx2y += x2 * y[i];
  }

  double D = s4 * (s2 * s0 - s1 * s1) - s3 * (s3 * s0 - s1 * s2) +
             s2 * (s3 * s1 - s2 * s2);

  double a = 0, b = 0, c_val = 0;
  if (std::abs(D) > 1e-30) {
    a = (sx2y * (s2 * s0 - s1 * s1) - s3 * (sxy * s0 - s1 * sy) +
         s2 * (sxy * s1 - s2 * sy)) /
        D;
    b = (s4 * (sxy * s0 - s1 * sy) - sx2y * (s3 * s0 - s1 * s2) +
         s2 * (s3 * sy - sxy * s2)) /
        D;
    c_val = (s4 * (s2 * sy - sxy * s1) - s3 * (s3 * sy - sxy * s2) +
             sx2y * (s3 * s1 - s2 * s2)) /
            D;
  }

  r.params = {a, b, c_val};
  r.residuals.resize(n);
  for (size_t i = 0; i < n; ++i) {
    double x = static_cast<double>(i);
    r.residuals[i] = y[i] - (a * x * x + b * x + c_val);
  }
  r.ssr = compute_ssr(r.residuals);
  return r;
}

// ── Model 2: XOR-Delta ─────────────────────────────────────────────────────

ModelResult fit_xor_delta(const double *y, size_t n) {
  ModelResult r;
  r.model = ModelType::XorDelta;
  double anchor = (n > 0) ? y[0] : 0.0;
  r.params = {anchor};

  if (n > 1) {
    r.xor_residuals.resize(n - 1);
    uint64_t prev = float64_to_bits(y[0]);
    for (size_t i = 1; i < n; ++i) {
      uint64_t curr = float64_to_bits(y[i]);
      r.xor_residuals[i - 1] = prev ^ curr;
      prev = curr;
    }
  }
  r.ssr = xor_popcount_ssr(r.xor_residuals);
  return r;
}

// ── Model 3: Constant ──────────────────────────────────────────── NEW ────

ModelResult fit_constant(const double *y, size_t n) {
  ModelResult r;
  r.model = ModelType::Constant;

  if (n == 0) {
    r.params = {0.0};
    r.ssr = 0.0;
    return r;
  }

  double sum = 0;
  for (size_t i = 0; i < n; ++i)
    sum += y[i];
  double c = sum / static_cast<double>(n);

  r.params = {c};
  r.residuals.resize(n);
  for (size_t i = 0; i < n; ++i) {
    r.residuals[i] = y[i] - c;
  }
  r.ssr = compute_ssr(r.residuals);
  return r;
}

// ── Model 4: Sinusoidal ────────────────────────────────────────── NEW ────

// Direct DFT magnitude search: O(n^2). Seeds Sinusoidal only (n <= 4096).
// Python twin uses O(n log n) FFT via numpy; both only need the peak bin.
static double dominant_frequency(const double *y, size_t n) {
  if (n < 4)
    return 0.0;

  // Compute mean
  double mean = 0;
  for (size_t i = 0; i < n; ++i)
    mean += y[i];
  mean /= static_cast<double>(n);

  // Find peak in magnitude spectrum (skip DC)
  double best_mag = 0;
  size_t best_k = 0;
  size_t max_k = n / 2 + 1;

  for (size_t k = 1; k < max_k; ++k) {
    double re = 0, im = 0;
    for (size_t i = 0; i < n; ++i) {
      double angle = 2.0 * M_PI * k * i / static_cast<double>(n);
      double centered = y[i] - mean;
      re += centered * std::cos(angle);
      im -= centered * std::sin(angle);
    }
    double mag = re * re + im * im;
    if (mag > best_mag) {
      best_mag = mag;
      best_k = k;
    }
  }
  return static_cast<double>(best_k) / static_cast<double>(n);
}

ModelResult fit_sinusoidal(const double *y, size_t n) {
  ModelResult r;
  r.model = ModelType::Sinusoidal;

  constexpr size_t SINUSOID_MIN = 32;
  constexpr int MAX_ITER = 50;

  if (n < SINUSOID_MIN) {
    // Fallback to linear
    auto lin = fit_linear(y, n);
    r.params = {0.0, 0.0, 0.0, lin.params[1]};
    r.residuals = lin.residuals;
    r.ssr = lin.ssr;
    return r;
  }

  double f0 = dominant_frequency(y, n);
  if (f0 < 1e-6) {
    // DC — constant better
    auto con = fit_constant(y, n);
    r.params = {0.0, 0.0, 0.0, con.params[0]};
    r.residuals = con.residuals;
    r.ssr = con.ssr;
    return r;
  }

  // Mean as DC offset
  double mean = 0;
  for (size_t i = 0; i < n; ++i)
    mean += y[i];
  mean /= static_cast<double>(n);

  double ymin = y[0], ymax = y[0];
  for (size_t i = 1; i < n; ++i) {
    if (y[i] < ymin)
      ymin = y[i];
    if (y[i] > ymax)
      ymax = y[i];
  }

  double A = (ymax - ymin) / 2.0;
  double w = 2.0 * M_PI * f0;
  double phi = 0.0;
  double dc = mean;

  // Gauss-Newton / Levenberg-Marquardt
  for (int iter = 0; iter < MAX_ITER; ++iter) {
    // Compute residuals and Jacobian (4x4 J^T J system)
    double JtJ[4][4] = {};
    double JtR[4] = {};

    for (size_t i = 0; i < n; ++i) {
      double x = static_cast<double>(i);
      double s = std::sin(w * x + phi);
      double c = std::cos(w * x + phi);
      double pred = A * s + dc;
      double ri = y[i] - pred;

      double j[4] = {s, A * c * x, A * c, 1.0};

      for (int a = 0; a < 4; ++a) {
        JtR[a] += j[a] * ri;
        for (int b = 0; b < 4; ++b) {
          JtJ[a][b] += j[a] * j[b];
        }
      }
    }

    // Add damping
    for (int a = 0; a < 4; ++a)
      JtJ[a][a] += 1e-4;

    // Solve 4x4 system via Gaussian elimination
    double aug[4][5];
    for (int a = 0; a < 4; ++a) {
      for (int b = 0; b < 4; ++b)
        aug[a][b] = JtJ[a][b];
      aug[a][4] = JtR[a];
    }

    for (int col = 0; col < 4; ++col) {
      // Partial pivot
      int piv = col;
      for (int row = col + 1; row < 4; ++row) {
        if (std::abs(aug[row][col]) > std::abs(aug[piv][col]))
          piv = row;
      }
      if (piv != col)
        for (int j = 0; j < 5; ++j)
          std::swap(aug[col][j], aug[piv][j]);

      if (std::abs(aug[col][col]) < 1e-30)
        break;

      for (int row = col + 1; row < 4; ++row) {
        double fac = aug[row][col] / aug[col][col];
        for (int j = col; j < 5; ++j)
          aug[row][j] -= fac * aug[col][j];
      }
    }

    double delta[4] = {};
    for (int row = 3; row >= 0; --row) {
      double s = aug[row][4];
      for (int j = row + 1; j < 4; ++j)
        s -= aug[row][j] * delta[j];
      delta[row] = (std::abs(aug[row][row]) > 1e-30) ? s / aug[row][row] : 0.0;
    }

    A += delta[0];
    w += delta[1];
    phi += delta[2];
    dc += delta[3];

    double norm = 0;
    for (int a = 0; a < 4; ++a)
      norm += delta[a] * delta[a];
    if (norm < 1e-20)
      break;
  }

  r.params = {A, w, phi, dc};
  r.residuals.resize(n);
  for (size_t i = 0; i < n; ++i) {
    double x = static_cast<double>(i);
    r.residuals[i] = y[i] - (A * std::sin(w * x + phi) + dc);
  }
  r.ssr = compute_ssr(r.residuals);

  // Sanity: if worse than linear, keep sinusoidal params but use linear
  // residuals
  auto lin = fit_linear(y, n);
  if (r.ssr > lin.ssr) {
    r.residuals = lin.residuals;
    r.ssr = lin.ssr;
  }

  return r;
}

// ── Model 5: Pred-XOR ──────────────────────────────────────────── NEW ────

ModelResult fit_pred_xor(const double *y, size_t n) {
  ModelResult r;
  r.model = ModelType::PredXor;

  if (n < 2) {
    auto xd = fit_xor_delta(y, n);
    r.params = {(n > 0 ? y[0] : 0.0), 0.0};
    r.xor_residuals = xd.xor_residuals;
    r.ssr = xd.ssr;
    return r;
  }

  r.params = {y[0], y[1]};
  r.xor_residuals.reserve(n > 2 ? n - 2 : 0);

  for (size_t i = 2; i < n; ++i) {
    double predicted = 2.0 * y[i - 1] - y[i - 2];
    uint64_t pred_bits = float64_to_bits(predicted);
    uint64_t actual_bits = float64_to_bits(y[i]);
    r.xor_residuals.push_back(actual_bits ^ pred_bits);
  }
  r.ssr = xor_popcount_ssr(r.xor_residuals);
  return r;
}

// ── Model Selection ─────────────────────────────────────────────────────────

static constexpr double XOR_DELTA_PREFERENCE = 1.02;
static constexpr double CONST_PREFERENCE = 1.01;

ModelResult select_model(const double *y, size_t n) {
  if (n == 0) {
    return {ModelType::Linear, {0.0, 0.0}, {}, {}, 0.0};
  }

  // Non-finite guard → XOR-Delta
  for (size_t i = 0; i < n; ++i) {
    if (!std::isfinite(y[i])) {
      return fit_xor_delta(y, n);
    }
  }

  // Fit all analytical models
  auto con = fit_constant(y, n);
  auto lin = fit_linear(y, n);

  ModelResult *best_analytical = (con.ssr <= lin.ssr) ? &con : &lin;

  ModelResult quad;
  if (n >= 4) {
    quad = fit_quadratic(y, n);
    if (quad.ssr < best_analytical->ssr)
      best_analytical = &quad;
  }

  ModelResult sine;
  if (n >= 32) {
    sine = fit_sinusoidal(y, n);
    if (sine.ssr < best_analytical->ssr)
      best_analytical = &sine;
  }

  // Prefer constant if within 1% of best
  if (con.ssr <= best_analytical->ssr * CONST_PREFERENCE)
    best_analytical = &con;

  // Fit XOR-based models
  auto xor_d = fit_xor_delta(y, n);

  ModelResult *best_xor = &xor_d;

  ModelResult pxor;
  if (n >= 3) {
    pxor = fit_pred_xor(y, n);
    if (pxor.ssr < xor_d.ssr)
      best_xor = &pxor;
  }

  // Compare XOR vs analytical
  if (best_xor->ssr <= best_analytical->ssr * XOR_DELTA_PREFERENCE)
    return *best_xor;
  return *best_analytical;
}

} // namespace dlc
