# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Finite differences on a canvas, their adjoints, and pointwise norms (docs/math.md, 3).

A scalar field has the shape (H, W): rows from the top, columns from the left. A
vector field has the shape (2, H, W), its x (across) and y (down) components. A
symmetric tensor field has the shape (3, H, W): r11, r22 and r12. Its inner
product counts r12 twice, as the full 2x2 matrix does, and so do its norm and
the projection onto its ball.

grad and div are negative adjoints, and so are sym_grad and div2:
<grad x, p> = -<x, div p> and <sym_grad w, r> = -<w, div2 r>. They are linear maps
with integer coefficients, save the halving, and hold whatever numbers they are
given: floating point, or Python's rationals, in which the tests check the adjoints
exactly.
"""

import numpy as np
import numpy.typing as npt

__all__ = [
    "backward_x",
    "backward_y",
    "div",
    "div2",
    "forward_x",
    "forward_y",
    "grad",
    "project_tensors",
    "project_vectors",
    "sym_grad",
    "tensor_inner",
    "tensor_norms",
    "vector_norms",
]

type Array = npt.NDArray[np.float64]


def forward_x[T: (np.float64, np.complex128, np.object_)](f: npt.NDArray[T]) -> npt.NDArray[T]:
    """The forward difference across, 0 in the last column (Neumann)."""
    result = np.zeros_like(f)
    result[..., :, :-1] = f[..., :, 1:] - f[..., :, :-1]
    return result


def forward_y[T: (np.float64, np.complex128, np.object_)](f: npt.NDArray[T]) -> npt.NDArray[T]:
    """The forward difference down, 0 in the last row (Neumann)."""
    result = np.zeros_like(f)
    result[..., :-1, :] = f[..., 1:, :] - f[..., :-1, :]
    return result


def backward_x[T: (np.float64, np.complex128, np.object_)](f: npt.NDArray[T]) -> npt.NDArray[T]:
    """The negative adjoint of forward_x: f[j] for j < W - 1, less f[j - 1] for j > 0."""
    result = np.zeros_like(f)
    result[..., :, :-1] += f[..., :, :-1]
    result[..., :, 1:] -= f[..., :, :-1]
    return result


def backward_y[T: (np.float64, np.complex128, np.object_)](f: npt.NDArray[T]) -> npt.NDArray[T]:
    """The negative adjoint of forward_y."""
    result = np.zeros_like(f)
    result[..., :-1, :] += f[..., :-1, :]
    result[..., 1:, :] -= f[..., :-1, :]
    return result


def grad[T: (np.float64, np.complex128, np.object_)](x: npt.NDArray[T]) -> npt.NDArray[T]:
    """The gradient of a scalar field, by forward differences."""
    return np.stack((forward_x(x), forward_y(x)))


def div[T: (np.float64, np.complex128, np.object_)](p: npt.NDArray[T]) -> npt.NDArray[T]:
    """The divergence of a vector field, -grad transposed."""
    return np.asarray(backward_x(p[0]) + backward_y(p[1]), dtype=p.dtype)


def sym_grad[T: (np.float64, np.complex128, np.object_)](w: npt.NDArray[T]) -> npt.NDArray[T]:
    """The symmetrized gradient of a vector field, by backward differences: (r11, r22, r12)."""
    # Halved by dividing by the integer 2, which is exact in floating point and keeps exact
    # arithmetic exact, so that the tests can check the adjoint in rational numbers.
    halved = (backward_y(w[0]) + backward_x(w[1])) / 2
    return np.asarray(np.stack((backward_x(w[0]), backward_y(w[1]), halved)), dtype=w.dtype)


def div2[T: (np.float64, np.complex128, np.object_)](r: npt.NDArray[T]) -> npt.NDArray[T]:
    """The divergence of a symmetric tensor field, -sym_grad transposed, by forward differences."""
    return np.stack((forward_x(r[0]) + forward_y(r[2]), forward_x(r[2]) + forward_y(r[1])))


def vector_norms(p: Array) -> Array:
    """The Euclidean norm of each vector of a field, shape (H, W)."""
    return np.asarray(np.sqrt(p[0] * p[0] + p[1] * p[1]), dtype=np.float64)


def tensor_norms(r: Array) -> Array:
    """The Frobenius norm of each tensor of a field, the off-diagonal entry counted twice."""
    return np.asarray(np.sqrt(r[0] * r[0] + r[1] * r[1] + 2.0 * r[2] * r[2]), dtype=np.float64)


def tensor_inner(r: Array, s: Array) -> float:
    """The inner product of two symmetric tensor fields, the off-diagonal entries counted twice."""
    return float(np.sum(r[0] * s[0]) + np.sum(r[1] * s[1]) + 2.0 * np.sum(r[2] * s[2]))


def project_vectors(p: Array, radius: float) -> Array:
    """Each vector of the field projected onto the ball of the radius."""
    return np.asarray(p / np.maximum(1.0, vector_norms(p) / radius), dtype=np.float64)


def project_tensors(r: Array, radius: float) -> Array:
    """Each tensor of the field projected onto the Frobenius ball of the radius."""
    return np.asarray(r / np.maximum(1.0, tensor_norms(r) / radius), dtype=np.float64)
