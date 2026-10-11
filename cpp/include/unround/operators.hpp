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

#include <cmath>
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

// The rows [first, end) of the output that a call writes: the part of the canvas that
// a thread takes. The rows of the input next to them are read as they are.
struct Rows {
  std::size_t first = 0;
  std::size_t end = 0;
};

// Each difference comes whole, of the rows of a part, or of one row i into spans of
// the grid's width, where a step takes that row at once; every entry is the same
// whichever is taken.

// nabla x: the forward differences across and down, 0 out of the grid.
void gradient(Grid<const double> field, VectorView<double> out) noexcept;
void gradient(Grid<const double> field, VectorView<double> out, Rows rows) noexcept;
void gradient(Grid<const double> field, std::size_t i, std::span<double> x, std::span<double> y) noexcept;
// That of a row from its entries and those of the row below it (none in the last row).
void gradient(std::span<const double> row, std::span<const double> below, std::span<double> x,
              std::span<double> y) noexcept;

// div p = delta_x p_1 + delta_y p_2, the backward differences: the negative adjoint
// of the gradient.
void divergence(VectorView<const double> p, Grid<double> out) noexcept;
void divergence(VectorView<const double> p, Grid<double> out, Rows rows) noexcept;
void divergence(VectorView<const double> p, std::size_t i, std::span<double> out) noexcept;

// E w = (delta_x w_1, delta_y w_2, (delta_y w_1 + delta_x w_2) / 2).
void symmetrized_gradient(VectorView<const double> w, TensorView<double> out) noexcept;
void symmetrized_gradient(VectorView<const double> w, TensorView<double> out, Rows rows) noexcept;
void symmetrized_gradient(VectorView<const double> w, std::size_t i, std::span<double> xx, std::span<double> yy,
                          std::span<double> xy) noexcept;
// That of a row from its entries and those of the row above it (none in the first row),
// `last` where it is the last.
void symmetrized_gradient(std::span<const double> wx, std::span<const double> wy, std::span<const double> above_x,
                          std::span<const double> above_y, bool last, std::span<double> xx, std::span<double> yy,
                          std::span<double> xy) noexcept;

// div_2 r = (d_x r11 + d_y r12, d_x r12 + d_y r22), the forward differences: the
// negative adjoint of E.
void tensor_divergence(TensorView<const double> r, VectorView<double> out) noexcept;
void tensor_divergence(TensorView<const double> r, VectorView<double> out, Rows rows) noexcept;
void tensor_divergence(TensorView<const double> r, std::size_t i, std::span<double> x, std::span<double> y) noexcept;

// The inner products, the tensors' off-diagonal entries twice; summed in the order
// of docs/math.md, Arithmetic, row by row.
[[nodiscard]] double inner(Grid<const double> a, Grid<const double> b);
[[nodiscard]] double inner(VectorView<const double> a, VectorView<const double> b);
[[nodiscard]] double inner(TensorView<const double> a, TensorView<const double> b);

// What a pixel of the squared norm `square` is multiplied by to be projected onto the
// ball of radius `radius`: radius / |p| beyond it, and 1 exactly within it; a root and
// a quotient.
[[nodiscard]] inline double ball_scale(double square, double radius) noexcept {
  const double norm = std::sqrt(square);
  return norm > radius ? radius / norm : 1.0;
}

// Projects every pixel of the channels' fields onto the ball of radius `radius`:
// coupled, the ball of the Euclidean norm of all the channels' entries at the pixel;
// apart, each channel's own. p <- p / max(1, |p| / radius), p times its ball_scale.
// The calls of a part take its rows, and rows of the fields' width for the squares and
// the scales of a row.
void project(std::span<const VectorView<double>> fields, double radius, bool coupled) noexcept;
void project(std::span<const TensorView<double>> fields, double radius, bool coupled) noexcept;
void project(std::span<const VectorView<double>> fields, double radius, bool coupled, Rows rows,
             std::span<double> squares, std::span<double> scales) noexcept;
void project(std::span<const TensorView<double>> fields, double radius, bool coupled, Rows rows,
             std::span<double> squares, std::span<double> scales) noexcept;

}  // namespace unround::operators

#endif  // UNROUND_OPERATORS_HPP
