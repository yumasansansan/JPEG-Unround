// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Finite differences on a grid of height x width samples, row by row
// (docs/math.md, 3): the gradient by forward differences with Neumann boundaries,
// the divergence as its negative adjoint, the symmetrized gradient of a vector
// field by backward differences, and the divergence of a tensor field; and the
// projections onto the balls of their pixelwise norms.
#ifndef UNROUND_OPERATORS_HPP
#define UNROUND_OPERATORS_HPP

#include <cstddef>
#include <span>
#include <vector>

namespace unround::operators {

// A vector field: its entries across (x) and down (y), each height x width.
struct Vector {
  std::vector<double> x;
  std::vector<double> y;
};

// A symmetric tensor field: r11 (xx), r22 (yy) and r12 (xy), each height x width;
// the off-diagonal entry counts twice in the inner product and the norm (3.2).
struct Tensor {
  std::vector<double> xx;
  std::vector<double> yy;
  std::vector<double> xy;
};

[[nodiscard]] Vector vector_of(std::size_t samples);
[[nodiscard]] Tensor tensor_of(std::size_t samples);

// nabla x: the forward difference across and down, 0 out of the grid.
void gradient(std::span<const double> field, std::size_t height, std::size_t width, Vector& out) noexcept;

// div p = delta_x p_1 + delta_y p_2, the backward differences: the negative adjoint
// of the gradient.
void divergence(const Vector& p, std::size_t height, std::size_t width, std::span<double> out) noexcept;

// E w = (delta_x w_1, delta_y w_2, (delta_y w_1 + delta_x w_2) / 2).
void symmetrized_gradient(const Vector& w, std::size_t height, std::size_t width, Tensor& out) noexcept;

// div_2 r = (d_x r11 + d_y r12, d_x r12 + d_y r22), the forward differences: the
// negative adjoint of E.
void tensor_divergence(const Tensor& r, std::size_t height, std::size_t width, Vector& out) noexcept;

// The inner products, the tensors' off-diagonal entries twice; summed in the order
// of docs/math.md, Arithmetic, rows of `width` terms.
[[nodiscard]] double inner(std::span<const double> a, std::span<const double> b, std::size_t width);
[[nodiscard]] double inner(const Vector& a, const Vector& b, std::size_t width);
[[nodiscard]] double inner(const Tensor& a, const Tensor& b, std::size_t width);

// Projects every pixel of the channels' fields onto the ball of radius `radius`:
// coupled, the ball of the Euclidean norm of all the channels' entries at the pixel;
// apart, each channel's own. p <- p / max(1, |p| / radius).
void project(std::span<Vector> fields, double radius, bool coupled) noexcept;
void project(std::span<Tensor> fields, double radius, bool coupled) noexcept;

}  // namespace unround::operators

#endif  // UNROUND_OPERATORS_HPP
