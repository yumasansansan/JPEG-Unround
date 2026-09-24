# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Bounds of rounding error, from which the tests take their tolerances.

U is the unit roundoff of binary64, 2^-53. A sum of n terms, or a dot product of n
pairs, computed in floating point in any order, differs from the exact one by at
most gamma(n) times the sum of the magnitudes of its terms (N. J. Higham, Accuracy
and Stability of Numerical Algorithms, 2nd ed., sections 3.1 and 4.2).

Every entry of dct.BASIS is the double nearest to the exact one, within U of its
magnitude (test_dct checks it against 70 digits). A product B X of the basis and a
block is then within e1 |B| |X| of the exact one, e1 = gamma(8) (1 + U) + U, about
9U; and the block DCT, (B X) B^T, within
gamma(8) (1 + e1) (1 + U) + U (1 + e1) + e1, about 18U, times |B| |X| |B|^T. The
bounds below take 19U, which covers the terms of second order.
"""

from typing import Final

import numpy as np
import numpy.typing as npt

from unround import dct, model, operators
from unround.model import TV, Dual, Primal, Problem

type Array = npt.NDArray[np.float64]

U: Final = 2.0**-53


def gamma(n: int) -> float:
    """Higham's gamma_n = n U / (1 - n U)."""
    return n * U / (1.0 - n * U)


def sum_error(terms: npt.ArrayLike) -> float:
    """A bound of the rounding of the floating-point sum of these terms."""
    values = np.abs(np.asarray(terms, dtype=np.float64))
    return gamma(values.size) * float(np.sum(values))


def forward_error(canvas: npt.ArrayLike) -> Array:
    """A bound, coefficient by coefficient, of the rounding of dct.forward of a canvas."""
    magnitude = dct.blocks(np.abs(np.asarray(canvas, dtype=np.float64)))
    basis = np.abs(dct.BASIS)
    return np.asarray(19.0 * U * (basis @ magnitude @ basis.T), dtype=np.float64)


def inverse_error(coefficients: npt.ArrayLike) -> Array:
    """A bound, sample by sample, of the rounding of dct.inverse of coefficients."""
    magnitude = np.abs(np.asarray(coefficients, dtype=np.float64))
    basis = np.abs(dct.BASIS)
    return dct.from_blocks(np.asarray(19.0 * U * (basis.T @ magnitude @ basis), dtype=np.float64))


def roundtrip_error(coefficients: npt.ArrayLike) -> Array:
    """A bound, coefficient by coefficient, of |dct.forward(dct.inverse(c)) - c|.

    The inverse rounds within inverse_error; the canvas it gives is within |B^T| |c| |B| plus
    that; the forward rounds within forward_error of it, and carries the inverse's rounding
    through |B| e |B|^T.
    """
    magnitude = np.abs(np.asarray(coefficients, dtype=np.float64))
    basis = np.abs(dct.BASIS)
    first = inverse_error(magnitude)
    canvas = dct.from_blocks(np.asarray(basis.T @ magnitude @ basis, dtype=np.float64)) + first
    return np.asarray(forward_error(canvas) + basis @ dct.blocks(first) @ basis.T, dtype=np.float64)


def tv_values_error(problem: Problem, weights: TV, point: Primal, dual: Dual) -> float:
    """A bound of the rounding of model.tv_values against the exact values at the point.

    The primal value: the canvas is within inverse_error of D^T c, and a sample is in at most
    four differences, so the total variation moves by at most 4 alpha times their sum; its
    terms and the data term's are non-negative, each within a few roundings, and summed
    within gamma(n). The dual value: every sample of div p adds at most four entries, each
    within alpha, with three roundings; the DCT carries that and rounds itself; and G* moves
    by at most conjugate_error.
    """
    samples = problem.samples
    moved = 4.0 * weights.alpha * float(np.sum(inverse_error(point.coefficients)))
    primal_value = model.tv_objective(problem, weights, point)
    primal = moved + (gamma(2 * samples) + 6.0 * U) * primal_value
    divergence = operators.div(dual.p)
    rounded = np.full(problem.shape, 12.0 * U * weights.alpha)
    basis = np.abs(dct.BASIS)
    s_error = forward_error(np.abs(divergence) + rounded) + basis @ dct.blocks(rounded) @ basis.T
    return primal + conjugate_error(problem, dct.forward(divergence), s_error)


def conjugate_magnitude(problem: Problem, s: npt.ArrayLike) -> float:
    """A bound of the sum of the magnitudes of the terms of G*(s) and of their parts.

    A term is s c* - (m/2) (c* - centre)^2 with c* in the interval, or max(s a, s b).
    """
    ends = np.maximum(np.abs(problem.lower), np.abs(problem.upper))
    width = problem.upper - problem.lower
    weights = np.broadcast_to(problem.weights, problem.lower.shape)
    return float(np.sum(np.abs(np.asarray(s, dtype=np.float64)) * ends + 0.5 * weights * width * width))


def conjugate_error(problem: Problem, s: npt.ArrayLike, s_error: npt.ArrayLike = 0.0) -> float:
    """A bound of |model.conjugate(problem, t) - G*(s)| for coefficients t within s_error of s.

    G* is Lipschitz in each coefficient with the constant max(|a|, |b|), its derivative being
    the maximizer c*. Each term is computed with at most eight roundings, each within U of
    the term's parts, and the sum within gamma(n) of their magnitudes.
    """
    ends = np.maximum(np.abs(problem.lower), np.abs(problem.upper))
    reach = np.abs(np.asarray(s, dtype=np.float64)) + np.asarray(s_error, dtype=np.float64)
    carried = float(np.sum(ends * np.asarray(s_error, dtype=np.float64)))
    return carried + (gamma(problem.lower.size) + 8.0 * U) * conjugate_magnitude(problem, reach)
