// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Tests of the finite differences (docs/math.md, 3): the divergence is the negative
// adjoint of the gradient, and that of tensors of the symmetrized gradient, exactly,
// on fields of integers, where every product and sum is exact in binary64; the norms
// within the bounds of 3.3; and the projections onto the balls: a field within its
// ball stays as it is, one beyond it lands on the sphere, within the rounding of the
// norm and the quotients, and projecting again moves nothing by more than that.

#include "unround/operators.hpp"

#include "unround/arrays.hpp"

#include "support/check.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <span>
#include <utility>

namespace {

using unround::Grid;
using unround::GridArray;
using unround::operators::Tensor;
using unround::operators::TensorView;
using unround::operators::Vector;
using unround::operators::VectorView;
using unround::test::gamma;
using unround::test::Numbers;

void fill(GridArray<double>& field, Numbers& numbers, std::int64_t size) {
  for (double& value : field.entries()) value = static_cast<double>(numbers.between(-size, size));
}

void adjoints(void) {
  Numbers numbers(21);
  for (const auto [height, width] : std::array<std::array<std::size_t, 2>, 4>{{{1, 1}, {1, 7}, {9, 1}, {13, 17}}}) {
    GridArray<double> x(unround::grid_extents(height, width));
    fill(x, numbers, 1000);
    Vector p = unround::operators::vector_of(height, width);
    fill(p.x, numbers, 1000);
    fill(p.y, numbers, 1000);
    Vector gradient = unround::operators::vector_of(height, width);
    unround::operators::gradient(std::as_const(x).view(), gradient.view());
    GridArray<double> divergence(unround::grid_extents(height, width));
    unround::operators::divergence(std::as_const(p).view(), divergence.view());
    // <nabla x, p> = -<x, div p>, every value an integer below 2^53.
    CHECK(unround::operators::inner(std::as_const(gradient).view(), std::as_const(p).view()) ==
          -unround::operators::inner(std::as_const(x).view(), std::as_const(divergence).view()));

    Vector w = unround::operators::vector_of(height, width);
    fill(w.x, numbers, 1000);
    fill(w.y, numbers, 1000);
    Tensor r = unround::operators::tensor_of(height, width);
    fill(r.xx, numbers, 1000);
    fill(r.yy, numbers, 1000);
    fill(r.xy, numbers, 1000);
    Tensor symmetrized = unround::operators::tensor_of(height, width);
    unround::operators::symmetrized_gradient(std::as_const(w).view(), symmetrized.view());
    Vector tensor_divergence = unround::operators::vector_of(height, width);
    unround::operators::tensor_divergence(std::as_const(r).view(), tensor_divergence.view());
    // <E w, r> = -<w, div_2 r>, the off-diagonal counted twice: halves of integers,
    // doubled.
    CHECK(unround::operators::inner(std::as_const(symmetrized).view(), std::as_const(r).view()) ==
          -unround::operators::inner(std::as_const(w).view(), std::as_const(tensor_divergence).view()));
  }
  // The differences out of the grid are 0.
  GridArray<double> ramp(unround::grid_extents(3, 4));
  for (std::size_t i = 0; i < 3u; ++i) {
    for (std::size_t j = 0; j < 4u; ++j) ramp.view()[i, j] = static_cast<double>(4u * i + j);
  }
  Vector gradient = unround::operators::vector_of(3, 4);
  unround::operators::gradient(std::as_const(ramp).view(), gradient.view());
  for (std::size_t i = 0; i < 3u; ++i) {
    for (std::size_t j = 0; j < 4u; ++j) {
      CHECK(gradient.x.view()[i, j] == (j < 3u ? 1.0 : 0.0));
      CHECK(gradient.y.view()[i, j] == (i < 2u ? 4.0 : 0.0));
    }
  }
}

void norms(void) {
  Numbers numbers(22);
  const std::size_t height = 16;
  const std::size_t width = 24;
  for (int trial = 0; trial < 50; ++trial) {
    GridArray<double> x(unround::grid_extents(height, width));
    for (double& value : x.entries()) value = numbers.uniform(-1.0, 1.0);
    Vector gradient = unround::operators::vector_of(height, width);
    unround::operators::gradient(std::as_const(x).view(), gradient.view());
    // ||nabla x||^2 <= 8 ||x||^2 (3.3), with the rounding of both sums.
    const double left = unround::operators::inner(std::as_const(gradient).view(), std::as_const(gradient).view());
    const double right = unround::operators::inner(std::as_const(x).view(), std::as_const(x).view());
    CHECK(left <= 8.0 * right * (1.0 + gamma(64)));
    Tensor symmetrized = unround::operators::tensor_of(height, width);
    Vector w = unround::operators::vector_of(height, width);
    std::ranges::copy(x.entries(), w.x.entries().begin());
    std::ranges::reverse_copy(x.entries(), w.y.entries().begin());
    unround::operators::symmetrized_gradient(std::as_const(w).view(), symmetrized.view());
    CHECK(unround::operators::inner(std::as_const(symmetrized).view(), std::as_const(symmetrized).view()) <=
          8.0 * unround::operators::inner(std::as_const(w).view(), std::as_const(w).view()) * (1.0 + gamma(64)));
  }
}

// The fields of three channels, their views, and copies.
template <typename Field, typename View>
std::array<View, 3> views_of(std::array<Field, 3>& fields) {
  return {fields[0].view(), fields[1].view(), fields[2].view()};
}

void projections(void) {
  Numbers numbers(23);
  // 500 pixels, in rows of 25.
  const std::size_t height = 20;
  const std::size_t width = 25;
  for (const bool coupled : {true, false}) {
    std::array<Vector, 3> fields{};
    std::array<Tensor, 3> tensors{};
    for (Vector& field : fields) {
      field = unround::operators::vector_of(height, width);
      for (std::size_t k = 0; k < field.x.size(); ++k) {
        // A third within the ball, the rest beyond it.
        const double size = k % 3u == 0u ? 0.1 : 5.0;
        field.x.entries()[k] = numbers.uniform(-size, size);
        field.y.entries()[k] = numbers.uniform(-size, size);
      }
    }
    for (Tensor& field : tensors) {
      field = unround::operators::tensor_of(height, width);
      for (std::size_t k = 0; k < field.xx.size(); ++k) {
        const double size = k % 3u == 0u ? 0.1 : 5.0;
        field.xx.entries()[k] = numbers.uniform(-size, size);
        field.yy.entries()[k] = numbers.uniform(-size, size);
        field.xy.entries()[k] = numbers.uniform(-size, size);
      }
    }
    const double radius = 1.5;
    const std::array<Vector, 3> before = fields;
    const std::array<Tensor, 3> tensors_before = tensors;
    unround::operators::project(views_of<Vector, VectorView<double>>(fields), radius, coupled);
    unround::operators::project(views_of<Tensor, TensorView<double>>(tensors), radius, coupled);
    // The norm of a projected pixel is within the rounding of the norm (a sum of k
    // squares and a root: gamma_{k+1}), of its quotient by the radius, and of the
    // entries' quotients: gamma_{k+4} (docs/math.md, 9.3), with k = 6 coupled.
    const auto norm_of = [&](const std::array<Vector, 3>& vectors, std::size_t channel, std::size_t i, std::size_t j) {
      double squared = 0.0;
      for (std::size_t c = 0; c < 3u; ++c) {
        if (!coupled && c != channel) continue;
        const double x = vectors[c].x.view()[i, j];
        const double y = vectors[c].y.view()[i, j];
        squared += x * x + y * y;
      }
      return std::sqrt(squared);
    };
    const auto tensor_norm_of = [&](const std::array<Tensor, 3>& fields_of, std::size_t channel, std::size_t i,
                                    std::size_t j) {
      double squared = 0.0;
      for (std::size_t c = 0; c < 3u; ++c) {
        if (!coupled && c != channel) continue;
        const double xx = fields_of[c].xx.view()[i, j];
        const double yy = fields_of[c].yy.view()[i, j];
        const double xy = fields_of[c].xy.view()[i, j];
        squared += xx * xx + yy * yy + 2.0 * (xy * xy);
      }
      return std::sqrt(squared);
    };
    const double slack = gamma(9 + 4) * 2.0;
    for (std::size_t i = 0; i < height; ++i) {
      for (std::size_t j = 0; j < width; ++j) {
        for (std::size_t c = 0; c < 3u; ++c) {
          CHECK(norm_of(fields, c, i, j) <= radius * (1.0 + slack));
          CHECK(tensor_norm_of(tensors, c, i, j) <= radius * (1.0 + slack));
          if (norm_of(before, c, i, j) <= radius) {
            CHECK(fields[c].x.view()[i, j] == before[c].x.view()[i, j] &&
                  fields[c].y.view()[i, j] == before[c].y.view()[i, j]);
          } else {
            CHECK(norm_of(fields, c, i, j) >= radius * (1.0 - slack));
          }
          if (tensor_norm_of(tensors_before, c, i, j) <= radius) {
            CHECK(tensors[c].xx.view()[i, j] == tensors_before[c].xx.view()[i, j] &&
                  tensors[c].xy.view()[i, j] == tensors_before[c].xy.view()[i, j]);
          }
        }
      }
    }
    // Projecting again moves a pixel by at most the rounding of its norm.
    std::array<Vector, 3> again = fields;
    unround::operators::project(views_of<Vector, VectorView<double>>(again), radius, coupled);
    for (std::size_t c = 0; c < 3u; ++c) {
      for (std::size_t k = 0; k < again[c].x.size(); ++k) {
        const double moved = again[c].x.entries()[k];
        const double was = fields[c].x.entries()[k];
        CHECK(std::abs(moved - was) <= slack * std::abs(was) + 1e-300);
      }
    }
  }
  // Fields of shapes other than the first's are the caller's mistake: none here, and
  // an empty set of fields changes nothing.
  unround::operators::project(std::span<const VectorView<double>>{}, 1.0, true);
}

}  // namespace

int main(void) {
  // A test that throws fails with what it says, rather than ending the program without
  // a word.
  try {
    adjoints();
    norms();
    projections();
    return unround::test::finish("operators");
  } catch (const std::exception& error) {
    (void)std::fputs(error.what(), stderr);
    (void)std::fputc('\n', stderr);
    return EXIT_FAILURE;
  }
}
