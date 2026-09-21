"""
Predictive models: Linear, Quadratic, XOR-Delta, Constant, Sinusoidal, Pred-XOR.

Model ID table:
  0 = Linear         y = mx + c                        (2 params)
  1 = Quadratic      y = ax^2 + bx + c                 (3 params)
  2 = XOR-Delta      bits(yi) XOR bits(yi-1)            (1 param: anchor)
  3 = Constant       y = c                              (1 param: mean)
  4 = Sinusoidal     y = A*sin(w*x + phi) + dc          (4 params)
  5 = Pred-XOR       bits(yi) XOR bits(2*yi-1 - yi-2)   (2 params: anchor0, anchor1)

Selection: lowest SSR wins, with preference rules for simpler models.
"""

import struct
import numpy as np
from typing import Tuple, List

# ── Model ID constants ───────────────────────────────────────────────────────

MODEL_LINEAR     = 0
MODEL_QUADRATIC  = 1
MODEL_XOR_DELTA  = 2
MODEL_CONSTANT   = 3
MODEL_SINUSOIDAL = 4
MODEL_PRED_XOR   = 5

# ── Preference constants ────────────────────────────────────────────────────

XOR_DELTA_PREFERENCE = 1.02    # was 1.05 — tighter
CONST_PREFERENCE     = 1.01    # prefer constant if within 1% of best
SINUSOID_MIN_WINDOW  = 32      # need at least 32 samples for sinusoidal fit
MAX_SINUSOID_ITER    = 50      # max Gauss-Newton iterations


# ── Utility: float64 <-> uint64 ─────────────────────────────────────────────

def _float64_to_bits(val: float) -> int:
    """Convert float64 to its unsigned 64-bit integer representation."""
    return struct.unpack('<Q', struct.pack('<d', val))[0]


def _bits_to_float64(bits: int) -> float:
    """Convert unsigned 64-bit integer back to float64."""
    return struct.unpack('<d', struct.pack('<Q', bits))[0]


# ── Model 0: Linear   y = mx + c ────────────────────────────────────────────

def fit_linear(y: np.ndarray) -> Tuple[Tuple, np.ndarray, float]:
    """
    Fit y = m*x + c via closed-form least squares.
    Returns: ((m, c), residuals, ssr)
    """
    n = len(y)
    if n < 2:
        c = float(y[0]) if n == 1 else 0.0
        res = np.zeros(n)
        return (0.0, c), res, 0.0

    x = np.arange(n, dtype=np.float64)
    sx = x.sum()
    sx2 = (x * x).sum()
    sy = y.sum()
    sxy = (x * y).sum()
    denom = n * sx2 - sx * sx
    if abs(denom) < 1e-12:
        m, c = 0.0, float(y.mean())
    else:
        m = float((n * sxy - sx * sy) / denom)
        c = float((sy - m * sx) / n)
    pred = m * x + c
    residuals = y - pred
    ssr = float(np.dot(residuals, residuals))
    return (m, c), residuals, ssr


# ── Model 1: Quadratic   y = ax^2 + bx + c ──────────────────────────────────

def fit_quadratic(y: np.ndarray) -> Tuple[Tuple, np.ndarray, float]:
    """
    Fit y = a*x^2 + b*x + c via least squares.
    Returns: ((a, b, c), residuals, ssr)
    """
    n = len(y)
    if n < 3:
        params, res, ssr = fit_linear(y)
        return (0.0, params[0], params[1]), res, ssr

    x = np.arange(n, dtype=np.float64)
    coeffs = np.polyfit(x, y, 2)
    a, b, c = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
    pred = a * x * x + b * x + c
    residuals = y - pred
    ssr = float(np.dot(residuals, residuals))
    return (a, b, c), residuals, ssr


# ── Model 2: XOR-Delta   bits(yi) XOR bits(yi-1) ────────────────────────────

def fit_xor_delta(y: np.ndarray) -> Tuple[float, List[int]]:
    """
    XOR-Delta encoding: anchor = y[0],
    residuals[i] = float64_bits(y[i]) XOR float64_bits(y[i-1]).

    Returns:
        (anchor, xor_residuals) where xor_residuals is a list of uint64.
    """
    n = len(y)
    anchor = float(y[0]) if n > 0 else 0.0
    xor_residuals = []

    if n <= 1:
        return anchor, xor_residuals

    prev_bits = _float64_to_bits(float(y[0]))
    for i in range(1, n):
        curr_bits = _float64_to_bits(float(y[i]))
        xor_residuals.append(prev_bits ^ curr_bits)
        prev_bits = curr_bits

    return anchor, xor_residuals


def reconstruct_xor_delta(anchor: float, xor_residuals: List[int]) -> np.ndarray:
    """Reconstruct signal from XOR-Delta encoding."""
    result = [anchor]
    prev_bits = _float64_to_bits(anchor)
    for xor_val in xor_residuals:
        curr_bits = prev_bits ^ xor_val
        result.append(_bits_to_float64(curr_bits))
        prev_bits = curr_bits
    return np.array(result, dtype=np.float64)


def _xor_ssr(y: np.ndarray, anchor: float, xor_residuals: List[int]) -> float:
    """Compute SSR proxy for XOR-Delta: popcount of all XOR values."""
    return float(sum(bin(x).count('1') for x in xor_residuals))


# ── Model 3: Constant   y = c ─────────────────────────────────────── NEW ──

def fit_constant(y: np.ndarray) -> Tuple[Tuple, np.ndarray, float]:
    """
    Fit y = mean(window).  Single float64 param.
    Returns: ((c,), residuals, ssr)
    """
    c = float(y.mean())
    residuals = y - c
    ssr = float(np.dot(residuals, residuals))
    return (c,), residuals, ssr


# ── Model 4: Sinusoidal   y = A*sin(w*x + phi) + dc ─────────────── NEW ──

def _dominant_frequency(window: np.ndarray) -> float:
    """Estimate dominant frequency via FFT (cycles per sample)."""
    n = len(window)
    spectrum = np.abs(np.fft.rfft(window - window.mean()))
    spectrum[0] = 0  # ignore DC
    idx = int(np.argmax(spectrum))
    return idx / n


def fit_sinusoidal(y: np.ndarray) -> Tuple[Tuple, np.ndarray, float]:
    """
    Fit y = A*sin(2*pi*f*x + phi) + dc using FFT seed + Gauss-Newton.
    Falls back to linear if window too short or fit diverges.
    Returns: ((A, w, phi, dc), residuals, ssr)
    """
    n = len(y)
    if n < SINUSOID_MIN_WINDOW:
        params, res, ssr = fit_linear(y)
        return (0.0, 0.0, 0.0, params[1]), res, ssr

    x = np.arange(n, dtype=np.float64)

    # FFT seed
    f0 = _dominant_frequency(y)
    if f0 < 1e-6:
        # Essentially DC — constant is better
        return fit_constant(y)

    dc0 = float(y.mean())
    A0 = float((y.max() - y.min()) / 2.0)
    phi0 = 0.0
    params = np.array([A0, 2 * np.pi * f0, phi0, dc0])

    # Gauss-Newton / Levenberg-Marquardt refinement
    for _ in range(MAX_SINUSOID_ITER):
        A, w, phi, dc = params
        pred = A * np.sin(w * x + phi) + dc
        r = y - pred

        # Jacobian
        dA   = np.sin(w * x + phi)
        dw   = A * np.cos(w * x + phi) * x
        dphi = A * np.cos(w * x + phi)
        ddc  = np.ones(n)
        J = np.column_stack([dA, dw, dphi, ddc])

        # Levenberg-Marquardt step
        lm = 1e-4 * np.eye(4)
        try:
            delta = np.linalg.solve(J.T @ J + lm, J.T @ r)
        except np.linalg.LinAlgError:
            break

        params = params + delta
        if np.linalg.norm(delta) < 1e-10:
            break

    A, w, phi, dc = params
    pred = A * np.sin(w * x + phi) + dc
    residuals = y - pred
    ssr = float(np.dot(residuals, residuals))

    # Sanity: if sinusoid SSR is worse than linear, fall back
    _, lin_res, lin_ssr = fit_linear(y)
    if ssr > lin_ssr:
        return (float(A), float(w), float(phi), float(dc)), lin_res, lin_ssr

    return (float(A), float(w), float(phi), float(dc)), residuals, ssr


# ── Model 5: Predictive XOR (PXOR) ──────────────────────────────── NEW ──

def fit_pred_xor(y: np.ndarray) -> Tuple[Tuple, List[int]]:
    """
    PXOR: stores two anchors (y[0], y[1]).
    predicted[i] = 2*y[i-1] - y[i-2]   (linear extrapolation)
    xor_residual[i] = bits(y[i]) XOR bits(predicted[i])

    Returns: ((anchor0, anchor1), xor_residuals)
    """
    if len(y) < 2:
        anchor, xor_res = fit_xor_delta(y)
        return (anchor, 0.0), xor_res

    anchor0, anchor1 = float(y[0]), float(y[1])
    xor_residuals = []
    for i in range(2, len(y)):
        predicted = 2.0 * y[i - 1] - y[i - 2]
        predicted_bits = _float64_to_bits(predicted)
        actual_bits = _float64_to_bits(float(y[i]))
        xor_residuals.append(actual_bits ^ predicted_bits)
    return (anchor0, anchor1), xor_residuals


def reconstruct_pred_xor(anchor0: float, anchor1: float,
                          xor_residuals: List[int]) -> np.ndarray:
    """Reconstruct signal from Pred-XOR encoding."""
    result = [anchor0, anchor1]
    for i, xor_val in enumerate(xor_residuals):
        predicted = 2.0 * result[-1] - result[-2]
        predicted_bits = _float64_to_bits(predicted)
        actual_bits = predicted_bits ^ xor_val
        result.append(_bits_to_float64(actual_bits))
    return np.array(result, dtype=np.float64)


def _pred_xor_ssr(y: np.ndarray, xor_residuals: List[int]) -> float:
    """SSR proxy for Pred-XOR: popcount of all XOR values."""
    return float(sum(bin(x).count('1') for x in xor_residuals))


# ── Model Selection ──────────────────────────────────────────────────────────

def _ssr(residuals: np.ndarray) -> float:
    """Sum of squared residuals."""
    return float(np.sum(residuals ** 2))


def select_model(window: np.ndarray) -> Tuple[int, list, np.ndarray]:
    """
    Evaluate all models and select the best one.

    Returns:
        (model_id, params_list, residuals)

        For Linear:    params = [m, c],          residuals = float array
        For Quadratic: params = [a, b, c],       residuals = float array
        For XOR-Delta: params = [anchor],         residuals = empty array
        For Constant:  params = [c],              residuals = float array
        For Sinusoidal:params = [A, w, phi, dc],  residuals = float array
        For Pred-XOR:  params = [anchor0,anchor1],residuals = empty array
    """
    n = len(window)
    if n == 0:
        return MODEL_LINEAR, [0.0, 0.0], np.array([], dtype=np.float64)

    # Guard: non-finite values — force XOR-Delta
    if not np.all(np.isfinite(window)):
        anchor, xor_res = fit_xor_delta(window)
        return MODEL_XOR_DELTA, [anchor], np.array([], dtype=np.float64)

    # ── Fit all analytical models ──────────────────────────────────────────

    # Model 3: Constant (cheapest — 1 param)
    const_params, const_res, const_ssr = fit_constant(window)

    # Model 0: Linear (2 params)
    lin_params, lin_res, lin_ssr = fit_linear(window)

    # Model 1: Quadratic (3 params) — only for n >= 4
    if n >= 4:
        quad_params, quad_res, quad_ssr = fit_quadratic(window)
    else:
        quad_params, quad_res, quad_ssr = lin_params, lin_res, lin_ssr * 10  # effectively disabled

    # Model 4: Sinusoidal (4 params) — only for n >= SINUSOID_MIN_WINDOW
    if n >= SINUSOID_MIN_WINDOW:
        sin_params, sin_res, sin_ssr = fit_sinusoidal(window)
    else:
        sin_params, sin_res, sin_ssr = lin_params, lin_res, lin_ssr * 10

    # Find best analytical model
    analytical = {
        MODEL_CONSTANT:   (list(const_params), const_res, const_ssr),
        MODEL_LINEAR:     (list(lin_params), lin_res, lin_ssr),
        MODEL_QUADRATIC:  (list(quad_params), quad_res, quad_ssr),
        MODEL_SINUSOIDAL: (list(sin_params), sin_res, sin_ssr),
    }

    best_id = min(analytical, key=lambda k: analytical[k][2])
    best_ssr = analytical[best_id][2]

    # Prefer constant if nearly as good (fewer params → cheaper to store)
    if const_ssr <= best_ssr * CONST_PREFERENCE:
        best_id = MODEL_CONSTANT
        best_ssr = const_ssr

    # ── Fit XOR-based models ──────────────────────────────────────────────

    anchor, xor_res = fit_xor_delta(window)
    xor_ssr = _xor_ssr(window, anchor, xor_res)

    # Pred-XOR (only for n >= 3)
    if n >= 3:
        pxor_params, pxor_res = fit_pred_xor(window)
        pxor_ssr = _pred_xor_ssr(window, pxor_res)
    else:
        pxor_params, pxor_res = (float(anchor), 0.0), []
        pxor_ssr = xor_ssr * 10

    # Pick better XOR variant
    if pxor_ssr < xor_ssr:
        xor_best_id = MODEL_PRED_XOR
        xor_best_ssr = pxor_ssr
        xor_best_params = list(pxor_params)
    else:
        xor_best_id = MODEL_XOR_DELTA
        xor_best_ssr = xor_ssr
        xor_best_params = [anchor]

    # Compare XOR vs analytical
    if xor_best_ssr <= best_ssr * XOR_DELTA_PREFERENCE:
        return xor_best_id, xor_best_params, np.array([], dtype=np.float64)
    else:
        params, residuals, _ = analytical[best_id]
        return best_id, params, residuals
