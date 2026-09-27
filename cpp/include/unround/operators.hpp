// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Finite differences on a grid (docs/math.md, 3): the gradient by forward
// differences with Neumann boundaries, the divergence as its negative adjoint, the
// symmetrized gradient of a vector field by backward differences, and the
// divergence of a tensor field; and the projections onto the balls of their
// pixelwise norms. Every field is a Grid, rows x columns (arrays.hpp), and all the
// grids of a call have one shape.
#ifndef UNROUND_OPERATORS_HPP
#define UNROUND_OPERATORS_HPP

#include "unround/arrays.hpp"

#include <cstddef>
#include <span>

namespace unround::operators {

// A vector field: its entries across (x) and down (y).
template <typename T>
struct VectorView {
  Grid<T> x;
  Grid<T> y;
};

// A symmetric tensor field: r11 (xx), r22 (yy) and r12 (xy); the off-diagonal
// entry counts twice in the inner product and the norm (3.2).
template <typename T>
struct TensorView {
  Grid<T> xx;
  Grid<T> yy;
  Grid<T> xy;
};

// The fields that own their entries, and their views.
struct Vector {
  GridArray<double> x;
  GridArray<double> y;

  [[nodiscard]] VectorView<double> view(void) noexcept { return {.x = x.view(), .y = y.view()}; }
  [[nodiscard]] VectorView<const double> view(void) const noexcept { return {.x = x.view(), .y = y.view()}; }
};

struct Tensor {
  GridArray<double> xx;
  GridArray<double> yy;
  GridArray<double> xy;

  [[nodiscard]] TensorView<double> view(void) noexcept { return {.xx = xx.view(), .yy = yy.view(), .xy = xy.view()}; }
  [[nodiscard]] TensorView<const double> view(void) const noexcept {
    return {.xx = xx.view(), .yy = yy.view(), .xy = xy.view()};
  }
};

// Fields of rows x columns zeros.
[[nodiscard]] Vector vector_of(std::size_t rows, std::size_t columns);
[[nodiscard]] Tensor tensor_of(std::size_t rows, std::size_t columns);

// nabla x: the forward differences across and down, 0 out of the grid.
void gradient(Grid<const double> field, VectorView<double> out) noexcept;

// div p = delta_x p_1 + delta_y p_2, the backward differences: the negative adjoint
// of the gradient.
void divergence(VectorView<const double> p, Grid<double> out) noexcept;

// E w = (delta_x w_1, delta_y w_2, (delta_y w_1 + delta_x w_2) / 2).
void symmetrized_gradient(VectorView<const double> w, TensorView<double> out) noexcept;

// div_2 r = (d_x r11 + d_y r12, d_x r12 + d_y r22), the forward differences: the
// negative adjoint of E.
void tensor_divergence(TensorView<const double> r, VectorView<double> out) noexcept;

// The inner products, the tensors' off-diagonal entries twice; summed in the order
// of docs/math.md, Arithmetic, row by row.
[[nodiscard]] double inner(Grid<const double> a, Grid<const double> b);
[[nodiscard]] double inner(VectorView<const double> a, VectorView<const double> b);
[[nodiscard]] double inner(TensorView<const double> a, TensorView<const double> b);

// Projects every pixel of the channels' fields onto the ball of radius `radius`:
// coupled, the ball of the Euclidean norm of all the channels' entries at the pixel;
// apart, each channel's own. p <- p / max(1, |p| / radius).
void project(std::span<const VectorView<double>> fields, double radius, bool coupled) noexcept;
void project(std::span<const TensorView<double>> fields, double radius, bool coupled) noexcept;

}  // namespace unround::operators

#endif  // UNROUND_OPERATORS_HPP
