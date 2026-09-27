// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The finite differences of docs/math.md, 3.

#include "unround/operators.hpp"

#include "unround/exact.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

namespace unround::operators {

Vector vector_of(std::size_t samples) {
  return Vector{.x = std::vector<double>(samples), .y = std::vector<double>(samples)};
}

Tensor tensor_of(std::size_t samples) {
  return Tensor{
      .xx = std::vector<double>(samples), .yy = std::vector<double>(samples), .xy = std::vector<double>(samples)};
}

void gradient(std::span<const double> field, std::size_t height, std::size_t width, Vector& out) noexcept {
  for (std::size_t i = 0; i < height; ++i) {
    for (std::size_t j = 0; j < width; ++j) {
      const std::size_t at = i * width + j;
      out.x[at] = j + 1u < width ? field[at + 1u] - field[at] : 0.0;
      out.y[at] = i + 1u < height ? field[at + width] - field[at] : 0.0;
    }
  }
}

namespace {

// (delta_x f)_{i,j} = f_{i,j} [j < W - 1] - f_{i,j-1} [j > 0], and down alike.
double backward_across(std::span<const double> f, std::size_t i, std::size_t j, std::size_t width) noexcept {
  const std::size_t at = i * width + j;
  const double here = j + 1u < width ? f[at] : 0.0;
  const double before = j > 0u ? f[at - 1u] : 0.0;
  return here - before;
}

double backward_down(std::span<const double> f, std::size_t i, std::size_t j, std::size_t height,
                     std::size_t width) noexcept {
  const std::size_t at = i * width + j;
  const double here = i + 1u < height ? f[at] : 0.0;
  const double before = i > 0u ? f[at - width] : 0.0;
  return here - before;
}

double forward_across(std::span<const double> f, std::size_t i, std::size_t j, std::size_t width) noexcept {
  const std::size_t at = i * width + j;
  return j + 1u < width ? f[at + 1u] - f[at] : 0.0;
}

double forward_down(std::span<const double> f, std::size_t i, std::size_t j, std::size_t height,
                    std::size_t width) noexcept {
  const std::size_t at = i * width + j;
  return i + 1u < height ? f[at + width] - f[at] : 0.0;
}

}  // namespace

void divergence(const Vector& p, std::size_t height, std::size_t width, std::span<double> out) noexcept {
  for (std::size_t i = 0; i < height; ++i) {
    for (std::size_t j = 0; j < width; ++j) {
      out[i * width + j] = backward_across(p.x, i, j, width) + backward_down(p.y, i, j, height, width);
    }
  }
}

void symmetrized_gradient(const Vector& w, std::size_t height, std::size_t width, Tensor& out) noexcept {
  for (std::size_t i = 0; i < height; ++i) {
    for (std::size_t j = 0; j < width; ++j) {
      const std::size_t at = i * width + j;
      out.xx[at] = backward_across(w.x, i, j, width);
      out.yy[at] = backward_down(w.y, i, j, height, width);
      out.xy[at] = 0.5 * (backward_down(w.x, i, j, height, width) + backward_across(w.y, i, j, width));
    }
  }
}

void tensor_divergence(const Tensor& r, std::size_t height, std::size_t width, Vector& out) noexcept {
  for (std::size_t i = 0; i < height; ++i) {
    for (std::size_t j = 0; j < width; ++j) {
      const std::size_t at = i * width + j;
      out.x[at] = forward_across(r.xx, i, j, width) + forward_down(r.xy, i, j, height, width);
      out.y[at] = forward_across(r.xy, i, j, width) + forward_down(r.yy, i, j, height, width);
    }
  }
}

double inner(std::span<const double> a, std::span<const double> b, std::size_t width) {
  std::vector<double> terms(a.size());
  for (std::size_t k = 0; k < a.size(); ++k) terms[k] = a[k] * b[k];
  return exact::sum_rows(terms, width);
}

double inner(const Vector& a, const Vector& b, std::size_t width) {
  std::vector<double> terms(a.x.size());
  for (std::size_t k = 0; k < terms.size(); ++k) terms[k] = a.x[k] * b.x[k] + a.y[k] * b.y[k];
  return exact::sum_rows(terms, width);
}

double inner(const Tensor& a, const Tensor& b, std::size_t width) {
  std::vector<double> terms(a.xx.size());
  for (std::size_t k = 0; k < terms.size(); ++k) {
    terms[k] = a.xx[k] * b.xx[k] + a.yy[k] * b.yy[k] + 2.0 * (a.xy[k] * b.xy[k]);
  }
  return exact::sum_rows(terms, width);
}

void project(std::span<Vector> fields, double radius, bool coupled) noexcept {
  if (fields.empty()) return;
  const std::size_t samples = fields.front().x.size();
  const auto scale_of = [radius](double squared) noexcept { return std::max(1.0, std::sqrt(squared) / radius); };
  for (std::size_t k = 0; k < samples; ++k) {
    if (coupled) {
      double squared = 0.0;
      for (const Vector& field : fields) squared += field.x[k] * field.x[k] + field.y[k] * field.y[k];
      const double scale = scale_of(squared);
      for (Vector& field : fields) {
        field.x[k] /= scale;
        field.y[k] /= scale;
      }
    } else {
      for (Vector& field : fields) {
        const double scale = scale_of(field.x[k] * field.x[k] + field.y[k] * field.y[k]);
        field.x[k] /= scale;
        field.y[k] /= scale;
      }
    }
  }
}

void project(std::span<Tensor> fields, double radius, bool coupled) noexcept {
  if (fields.empty()) return;
  const std::size_t samples = fields.front().xx.size();
  const auto scale_of = [radius](double squared) noexcept { return std::max(1.0, std::sqrt(squared) / radius); };
  const auto squared_of = [](const Tensor& field, std::size_t k) noexcept {
    return field.xx[k] * field.xx[k] + field.yy[k] * field.yy[k] + 2.0 * (field.xy[k] * field.xy[k]);
  };
  for (std::size_t k = 0; k < samples; ++k) {
    if (coupled) {
      double squared = 0.0;
      for (const Tensor& field : fields) squared += squared_of(field, k);
      const double scale = scale_of(squared);
      for (Tensor& field : fields) {
        field.xx[k] /= scale;
        field.yy[k] /= scale;
        field.xy[k] /= scale;
      }
    } else {
      for (Tensor& field : fields) {
        const double scale = scale_of(squared_of(field, k));
        field.xx[k] /= scale;
        field.yy[k] /= scale;
        field.xy[k] /= scale;
      }
    }
  }
}

}  // namespace unround::operators
