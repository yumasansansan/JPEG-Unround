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

#include "support/check.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <span>
#include <vector>

namespace {

using unround::operators::Tensor;
using unround::operators::Vector;
using unround::test::gamma;
using unround::test::Numbers;

void fill(std::vector<double>& field, Numbers& numbers, std::int64_t size) {
  for (double& value : field) value = static_cast<double>(numbers.between(-size, size));
}

void adjoints(void) {
  Numbers numbers(21);
  for (const auto [height, width] : std::array<std::array<std::size_t, 2>, 4>{{{1, 1}, {1, 7}, {9, 1}, {13, 17}}}) {
    const std::size_t samples = height * width;
    std::vector<double> x(samples);
    fill(x, numbers, 1000);
    Vector p = unround::operators::vector_of(samples);
    fill(p.x, numbers, 1000);
    fill(p.y, numbers, 1000);
    Vector gradient = unround::operators::vector_of(samples);
    unround::operators::gradient(x, height, width, gradient);
    std::vector<double> divergence(samples);
    unround::operators::divergence(p, height, width, divergence);
    // <nabla x, p> = -<x, div p>, every value an integer below 2^53.
    CHECK(unround::operators::inner(gradient, p, width) == -unround::operators::inner(x, divergence, width));

    Vector w = unround::operators::vector_of(samples);
    fill(w.x, numbers, 1000);
    fill(w.y, numbers, 1000);
    Tensor r = unround::operators::tensor_of(samples);
    fill(r.xx, numbers, 1000);
    fill(r.yy, numbers, 1000);
    fill(r.xy, numbers, 1000);
    Tensor symmetrized = unround::operators::tensor_of(samples);
    unround::operators::symmetrized_gradient(w, height, width, symmetrized);
    Vector tensor_divergence = unround::operators::vector_of(samples);
    unround::operators::tensor_divergence(r, height, width, tensor_divergence);
    // <E w, r> = -<w, div_2 r>, the off-diagonal counted twice: halves of integers,
    // doubled.
    CHECK(unround::operators::inner(symmetrized, r, width) == -unround::operators::inner(w, tensor_divergence, width));
  }
  // The differences out of the grid are 0.
  std::vector<double> ramp(12);
  for (std::size_t k = 0; k < ramp.size(); ++k) ramp[k] = static_cast<double>(k);
  Vector gradient = unround::operators::vector_of(12);
  unround::operators::gradient(ramp, 3, 4, gradient);
  for (std::size_t i = 0; i < 3u; ++i) {
    for (std::size_t j = 0; j < 4u; ++j) {
      CHECK(gradient.x[i * 4u + j] == (j < 3u ? 1.0 : 0.0));
      CHECK(gradient.y[i * 4u + j] == (i < 2u ? 4.0 : 0.0));
    }
  }
}

void norms(void) {
  Numbers numbers(22);
  const std::size_t height = 16;
  const std::size_t width = 24;
  for (int trial = 0; trial < 50; ++trial) {
    std::vector<double> x(height * width);
    for (double& value : x) value = numbers.uniform(-1.0, 1.0);
    Vector gradient = unround::operators::vector_of(x.size());
    unround::operators::gradient(x, height, width, gradient);
    // ||nabla x||^2 <= 8 ||x||^2 (3.3), with the rounding of both sums.
    const double left = unround::operators::inner(gradient, gradient, width);
    const double right = unround::operators::inner(x, x, width);
    CHECK(left <= 8.0 * right * (1.0 + gamma(64)));
    Tensor symmetrized = unround::operators::tensor_of(x.size());
    Vector w{.x = x, .y = std::vector<double>(x.rbegin(), x.rend())};
    unround::operators::symmetrized_gradient(w, height, width, symmetrized);
    CHECK(unround::operators::inner(symmetrized, symmetrized, width) <=
          8.0 * unround::operators::inner(w, w, width) * (1.0 + gamma(64)));
  }
}

void projections(void) {
  Numbers numbers(23);
  const std::size_t samples = 500;
  for (const bool coupled : {true, false}) {
    std::array<Vector, 3> fields{};
    std::array<Tensor, 3> tensors{};
    for (Vector& field : fields) {
      field = unround::operators::vector_of(samples);
      for (std::size_t k = 0; k < samples; ++k) {
        // A third within the ball, the rest beyond it.
        const double size = k % 3u == 0u ? 0.1 : 5.0;
        field.x[k] = numbers.uniform(-size, size);
        field.y[k] = numbers.uniform(-size, size);
      }
    }
    for (Tensor& field : tensors) {
      field = unround::operators::tensor_of(samples);
      for (std::size_t k = 0; k < samples; ++k) {
        const double size = k % 3u == 0u ? 0.1 : 5.0;
        field.xx[k] = numbers.uniform(-size, size);
        field.yy[k] = numbers.uniform(-size, size);
        field.xy[k] = numbers.uniform(-size, size);
      }
    }
    const double radius = 1.5;
    const std::array<Vector, 3> before = fields;
    const std::array<Tensor, 3> tensors_before = tensors;
    unround::operators::project(fields, radius, coupled);
    unround::operators::project(tensors, radius, coupled);
    // The norm of a projected pixel is within the rounding of the norm (a sum of k
    // squares and a root: gamma_{k+1}), of its quotient by the radius, and of the
    // entries' quotients: gamma_{k+4} (docs/math.md, 9.3), with k = 6 coupled.
    const auto norm_of = [&](const std::array<Vector, 3>& vectors, std::size_t channel, std::size_t k) {
      double squared = 0.0;
      for (std::size_t c = 0; c < 3u; ++c) {
        if (!coupled && c != channel) continue;
        squared += vectors[c].x[k] * vectors[c].x[k] + vectors[c].y[k] * vectors[c].y[k];
      }
      return std::sqrt(squared);
    };
    const auto tensor_norm_of = [&](const std::array<Tensor, 3>& fields_of, std::size_t channel, std::size_t k) {
      double squared = 0.0;
      for (std::size_t c = 0; c < 3u; ++c) {
        if (!coupled && c != channel) continue;
        squared += fields_of[c].xx[k] * fields_of[c].xx[k] + fields_of[c].yy[k] * fields_of[c].yy[k] +
                   2.0 * (fields_of[c].xy[k] * fields_of[c].xy[k]);
      }
      return std::sqrt(squared);
    };
    const double slack = gamma(9 + 4) * 2.0;
    for (std::size_t k = 0; k < samples; ++k) {
      for (std::size_t c = 0; c < 3u; ++c) {
        CHECK(norm_of(fields, c, k) <= radius * (1.0 + slack));
        CHECK(tensor_norm_of(tensors, c, k) <= radius * (1.0 + slack));
        if (norm_of(before, c, k) <= radius) {
          CHECK(fields[c].x[k] == before[c].x[k] && fields[c].y[k] == before[c].y[k]);
        } else {
          CHECK(norm_of(fields, c, k) >= radius * (1.0 - slack));
        }
        if (tensor_norm_of(tensors_before, c, k) <= radius) {
          CHECK(tensors[c].xx[k] == tensors_before[c].xx[k] && tensors[c].xy[k] == tensors_before[c].xy[k]);
        }
      }
    }
    // Projecting again moves a pixel by at most the rounding of its norm.
    std::array<Vector, 3> again = fields;
    unround::operators::project(again, radius, coupled);
    for (std::size_t c = 0; c < 3u; ++c) {
      for (std::size_t k = 0; k < samples; ++k) {
        CHECK(std::abs(again[c].x[k] - fields[c].x[k]) <= slack * std::abs(fields[c].x[k]) + 1e-300);
      }
    }
  }
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
