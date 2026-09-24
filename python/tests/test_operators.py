# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Finite differences: the adjoints in exact arithmetic, the norms, and the projections.

The adjoints are identities of the discrete operators, and are checked in rational
numbers, where they hold exactly. The norms are the largest eigenvalues of the
operators' matrices on small grids, against the bounds of docs/math.md, 3.3.
"""

import math
from collections.abc import Callable
from fractions import Fraction

import numpy as np
import numpy.typing as npt
import pytest

import rounding
from unround import operators, pdhg

type Array = npt.NDArray[np.float64]
type Objects = npt.NDArray[np.object_]

SHAPES = [(1, 1), (1, 6), (5, 1), (3, 4), (8, 8), (7, 5)]


def rationals(shape: tuple[int, ...], seed: int) -> Objects:
    """An array of random rationals, as Python Fractions."""
    rng = np.random.default_rng(seed)
    numerators = rng.integers(-1000, 1001, size=shape)
    denominators = rng.integers(1, 97, size=shape)
    result = np.empty(shape, dtype=object)
    for index in np.ndindex(*shape):
        result[index] = Fraction(int(numerators[index]), int(denominators[index]))
    return result


def total(values: Objects) -> Fraction:
    return sum((Fraction(value) for value in values.ravel()), Fraction(0))


@pytest.mark.parametrize("shape", SHAPES)
def test_div_is_exactly_minus_the_adjoint_of_grad(shape: tuple[int, int]) -> None:
    x = rationals(shape, 1)
    p = rationals((2, *shape), 2)
    assert total(operators.grad(x) * p) == -total(x * operators.div(p))


@pytest.mark.parametrize("shape", SHAPES)
def test_backward_differences_are_exactly_minus_the_adjoints_of_forward_ones(shape: tuple[int, int]) -> None:
    f, g = rationals(shape, 3), rationals(shape, 4)
    assert total(operators.forward_x(f) * g) == -total(f * operators.backward_x(g))
    assert total(operators.forward_y(f) * g) == -total(f * operators.backward_y(g))


@pytest.mark.parametrize("shape", SHAPES)
def test_div2_is_exactly_minus_the_adjoint_of_sym_grad(shape: tuple[int, int]) -> None:
    # In the inner product of symmetric tensors, which counts the off-diagonal entry twice.
    w = rationals((2, *shape), 5)
    r = rationals((3, *shape), 6)
    e = operators.sym_grad(w)
    left = total(e[0] * r[0]) + total(e[1] * r[1]) + 2 * total(e[2] * r[2])
    assert left == -total(w * operators.div2(r))


def test_the_gradient_has_neumann_boundaries() -> None:
    x = np.arange(12, dtype=np.float64).reshape(3, 4) ** 2
    gradient = operators.grad(x)
    np.testing.assert_array_equal(gradient[0][:, -1], 0.0)
    np.testing.assert_array_equal(gradient[1][-1, :], 0.0)
    np.testing.assert_array_equal(gradient[0][:, :-1], np.diff(x, axis=1))
    np.testing.assert_array_equal(gradient[1][:-1, :], np.diff(x, axis=0))


def test_the_symmetrized_gradient_of_a_ramp_is_zero_inside() -> None:
    y, x = np.mgrid[0:10, 0:12].astype(np.float64)
    hessian = operators.sym_grad(operators.grad(3.0 * x - 2.0 * y))
    np.testing.assert_array_equal(hessian[:, 1:-2, 1:-2], 0.0)


def matrix(apply: Callable[[Array], Array], shape: tuple[int, ...]) -> Array:
    """The matrix of a linear operator, column by column from the unit vectors."""
    size = math.prod(shape)
    columns = []
    for index in range(size):
        unit = np.zeros(size)
        unit[index] = 1.0
        columns.append(np.asarray(apply(unit.reshape(shape)), dtype=np.float64).ravel())
    return np.stack(columns, axis=1)


def largest_eigenvalue(gram: Array) -> tuple[float, float]:
    """The largest eigenvalue of a symmetric matrix, and a bound of its error.

    LAPACK's symmetric eigensolvers are backward stable: the eigenvalues are those of a
    matrix within a small multiple of n U ||A|| of it (LAPACK Users' Guide, 4.7); 10 n U ||A||
    is taken, ||A|| bounded by the largest absolute row sum.
    """
    size = gram.shape[0]
    norm = float(np.abs(gram).sum(axis=1).max())
    return float(np.linalg.eigvalsh(gram)[-1]), 10.0 * size * rounding.U * norm


def tensor_weights(fields: int, samples: int) -> Array:
    """The diagonal of the inner product: 1 for vector entries and r11, r22; 2 for r12."""
    return np.concatenate((np.ones(fields * samples), np.ones(2 * samples), np.full(samples, 2.0)))


@pytest.mark.parametrize("shape", [(1, 2), (2, 3), (4, 4), (5, 7), (8, 6)])
def test_the_norm_of_the_gradient(shape: tuple[int, int]) -> None:
    # grad^T grad is the Neumann Laplacian, whose eigenvalues are
    # 4 sin^2(pi k / (2W)) + 4 sin^2(pi l / (2H)): the largest is below 8.
    height, width = shape
    gradient = matrix(operators.grad, shape)
    largest, error = largest_eigenvalue(gradient.T @ gradient)
    closed = 4.0 * math.sin(math.pi * (width - 1) / (2 * width)) ** 2
    closed += 4.0 * math.sin(math.pi * (height - 1) / (2 * height)) ** 2
    assert abs(largest - closed) <= error + 8 * rounding.U * 8.0
    assert largest + error < pdhg.TV_NORM_SQUARED


@pytest.mark.parametrize("shape", [(1, 2), (2, 3), (4, 4), (5, 7), (8, 6)])
def test_the_norms_of_tgv_are_within_their_bounds(shape: tuple[int, int]) -> None:
    samples = math.prod(shape)
    symmetrized = matrix(lambda w: operators.sym_grad(w.reshape(2, *shape)), (2, *shape))
    weights = tensor_weights(0, samples)
    largest, error = largest_eigenvalue(symmetrized.T @ (weights[:, np.newaxis] * symmetrized))
    assert largest + error < 8.0

    def k(z: Array) -> Array:
        x, w = z[0], z[1:]
        return np.concatenate(((operators.grad(x) - w).ravel(), operators.sym_grad(w).ravel()))

    whole = matrix(k, (3, *shape))
    weights = tensor_weights(2, samples)
    largest, error = largest_eigenvalue(whole.T @ (weights[:, np.newaxis] * whole))
    assert largest + error < pdhg.TGV_NORM_SQUARED


def test_the_bounds_of_the_steps() -> None:
    # (17 + sqrt 33) / 2 is the square of the largest singular value of [[sqrt 8, 1], [0, sqrt 8]].
    singular = np.linalg.svd(np.array([[math.sqrt(8.0), 1.0], [0.0, math.sqrt(8.0)]]), compute_uv=False)[0]
    assert singular**2 == pytest.approx(pdhg.TGV_NORM_SQUARED, rel=8 * rounding.U)


def test_projections_onto_balls() -> None:
    rng = np.random.default_rng(8)
    p = rng.normal(0.0, 2.0, size=(2, 9, 11))
    r = rng.normal(0.0, 2.0, size=(3, 9, 11))
    projected_p = operators.project_vectors(p, 1.5)
    projected_r = operators.project_tensors(r, 1.5)
    # The norm of a projected vector: its computed norm within 2U, the quotient by the radius
    # within 3U, the division within U, and the test's own norm within 2U: 8U. Tensors have
    # a third term: 10U.
    assert np.all(operators.vector_norms(projected_p) <= 1.5 * (1.0 + 8.0 * rounding.U))
    assert np.all(operators.tensor_norms(projected_r) <= 1.5 * (1.0 + 10.0 * rounding.U))
    # What is inside is divided by exactly 1, and stays as it is.
    inside = operators.vector_norms(p) <= 1.5
    np.testing.assert_array_equal(projected_p[:, inside], p[:, inside])
    # Each is the nearest point of its ball, in the norm of its own inner product: the
    # distances are each within a few units of 2^-53 of their magnitudes.
    for _ in range(20):
        candidate_p = operators.project_vectors(rng.normal(0.0, 2.0, size=p.shape), 1.5)
        candidate_r = operators.project_tensors(rng.normal(0.0, 2.0, size=r.shape), 1.5)
        for projected, candidate, original, norms in (
            (projected_p, candidate_p, p, operators.vector_norms),
            (projected_r, candidate_r, r, operators.tensor_norms),
        ):
            near, far = norms(projected - original), norms(candidate - original)
            slack = 16.0 * rounding.U * (norms(original) + far + 1.5)
            assert np.all(near <= far + slack)


def test_the_tensor_norm_counts_the_off_diagonal_twice() -> None:
    r = np.array([[[3.0]], [[4.0]], [[1.0]]])
    assert operators.tensor_norms(r)[0, 0] == math.sqrt(27.0)
    assert operators.tensor_inner(r, r) == 27.0
