// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The finite differences of docs/math.md, 3.
//
// The fields are taken row by row. A difference of a row is one loop over its
// samples, the same operation on neighbouring values: vector code. Its ends, where it
// reaches out of the grid, are taken outside the loop, and so are the first and the
// last row. An output that sums two differences takes the first, then adds the
// second to the same row while it is at hand.

#include "unround/operators.hpp"

#include "unround/arrays.hpp"
#include "unround/exact.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <span>
#include <utility>
#include <vector>

namespace unround::operators {

namespace {

// Grids of other shapes than the first's are a mistake of the caller, which ends
// the program.
template <typename... Grids>
void same_shape(const GridExtents& shape, const Grids&... grids) noexcept {
  if (((grids.extents() != shape) || ...)) std::abort();
}

// A value written into its place, or added to what is there.
template <bool add>
void put(double& place, double value) noexcept {
  if constexpr (add) {
    place += value;
  } else {
    place = value;
  }
}

// Row i of d_x f: f[i, j + 1] - f[i, j], and 0 in the last column.
template <bool add>
void forward_across(Grid<const double> f, Grid<double> out, std::size_t i) noexcept {
  const std::size_t width = f.extent(1);
  for (std::size_t j = 0; j + 1u < width; ++j) put<add>(out[i, j], f[i, j + 1u] - f[i, j]);
  put<add>(out[i, width - 1u], 0.0);
}

// Row i of d_y f: f[i + 1, j] - f[i, j], and 0 in the last row.
template <bool add>
void forward_down(Grid<const double> f, Grid<double> out, std::size_t i) noexcept {
  const std::size_t width = f.extent(1);
  if (i + 1u < f.extent(0)) {
    for (std::size_t j = 0; j < width; ++j) put<add>(out[i, j], f[i + 1u, j] - f[i, j]);
  } else {
    for (std::size_t j = 0; j < width; ++j) put<add>(out[i, j], 0.0);
  }
}

// Row i of delta_x f: f[i, j] (not in the last column) - f[i, j - 1] (not in the
// first).
template <bool add>
void backward_across(Grid<const double> f, Grid<double> out, std::size_t i) noexcept {
  const std::size_t width = f.extent(1);
  if (width == 1u) {
    put<add>(out[i, 0], 0.0);
    return;
  }
  put<add>(out[i, 0], f[i, 0]);
  for (std::size_t j = 1; j + 1u < width; ++j) put<add>(out[i, j], f[i, j] - f[i, j - 1u]);
  put<add>(out[i, width - 1u], -f[i, width - 2u]);
}

// Row i of delta_y f: f[i, j] (not in the last row) - f[i - 1, j] (not in the first).
template <bool add>
void backward_down(Grid<const double> f, Grid<double> out, std::size_t i) noexcept {
  const std::size_t width = f.extent(1);
  const bool first = i == 0u;
  const bool last = i + 1u == f.extent(0);
  if (!first && !last) {
    for (std::size_t j = 0; j < width; ++j) put<add>(out[i, j], f[i, j] - f[i - 1u, j]);
  } else if (!last) {
    for (std::size_t j = 0; j < width; ++j) put<add>(out[i, j], f[i, j]);
  } else if (!first) {
    for (std::size_t j = 0; j < width; ++j) put<add>(out[i, j], -f[i - 1u, j]);
  } else {
    for (std::size_t j = 0; j < width; ++j) put<add>(out[i, j], 0.0);
  }
}

}  // namespace

Vector vector_of(std::size_t rows, std::size_t columns) {
  return Vector{.x = GridArray<double>(grid_extents(rows, columns)),
                .y = GridArray<double>(grid_extents(rows, columns))};
}

Tensor tensor_of(std::size_t rows, std::size_t columns) {
  return Tensor{.xx = GridArray<double>(grid_extents(rows, columns)),
                .yy = GridArray<double>(grid_extents(rows, columns)),
                .xy = GridArray<double>(grid_extents(rows, columns))};
}

void gradient(Grid<const double> field, VectorView<double> out) noexcept {
  same_shape(field.extents(), out.x, out.y);
  if (field.extent(1) == 0u) return;
  for (std::size_t i = 0; i < field.extent(0); ++i) {
    forward_across<false>(field, out.x, i);
    forward_down<false>(field, out.y, i);
  }
}

void divergence(VectorView<const double> p, Grid<double> out) noexcept {
  same_shape(out.extents(), p.x, p.y);
  if (out.extent(1) == 0u) return;
  for (std::size_t i = 0; i < out.extent(0); ++i) {
    backward_down<false>(p.y, out, i);
    backward_across<true>(p.x, out, i);
  }
}

void symmetrized_gradient(VectorView<const double> w, TensorView<double> out) noexcept {
  same_shape(w.x.extents(), w.y, out.xx, out.yy, out.xy);
  const std::size_t width = w.x.extent(1);
  if (width == 0u) return;
  for (std::size_t i = 0; i < w.x.extent(0); ++i) {
    backward_across<false>(w.x, out.xx, i);
    backward_down<false>(w.y, out.yy, i);
    // (delta_y w_1 + delta_x w_2) / 2: the sum, then its half, which is exact.
    const Grid<double> mixed = out.xy;
    backward_down<false>(w.x, mixed, i);
    backward_across<true>(w.y, mixed, i);
    for (std::size_t j = 0; j < width; ++j) mixed[i, j] = 0.5 * mixed[i, j];
  }
}

void tensor_divergence(TensorView<const double> r, VectorView<double> out) noexcept {
  same_shape(r.xx.extents(), r.yy, r.xy, out.x, out.y);
  if (r.xx.extent(1) == 0u) return;
  for (std::size_t i = 0; i < r.xx.extent(0); ++i) {
    // d_x r11 + d_y r12, and d_x r12 + d_y r22.
    forward_across<false>(r.xx, out.x, i);
    forward_down<true>(r.xy, out.x, i);
    forward_across<false>(r.xy, out.y, i);
    forward_down<true>(r.yy, out.y, i);
  }
}

namespace {

// The sum of terms given sample by sample, row by row as exact::sum_rows adds them.
template <typename Term>
double sum_of(const GridExtents& shape, const Term& term) {
  const std::size_t height = shape.extent(0);
  const std::size_t width = shape.extent(1);
  GridArray<double> terms(shape);
  const Grid<double> into = terms.view();
  for (std::size_t i = 0; i < height; ++i) {
    for (std::size_t j = 0; j < width; ++j) into[i, j] = term(i, j);
  }
  return exact::sum_rows(std::as_const(terms).view());
}

}  // namespace

double inner(Grid<const double> a, Grid<const double> b) {
  same_shape(a.extents(), b);
  return sum_of(a.extents(), [a, b](std::size_t i, std::size_t j) { return a[i, j] * b[i, j]; });
}

double inner(VectorView<const double> a, VectorView<const double> b) {
  same_shape(a.x.extents(), a.y, b.x, b.y);
  return sum_of(a.x.extents(),
                [a, b](std::size_t i, std::size_t j) { return a.x[i, j] * b.x[i, j] + a.y[i, j] * b.y[i, j]; });
}

double inner(TensorView<const double> a, TensorView<const double> b) {
  same_shape(a.xx.extents(), a.yy, a.xy, b.xx, b.yy, b.xy);
  return sum_of(a.xx.extents(), [a, b](std::size_t i, std::size_t j) {
    return a.xx[i, j] * b.xx[i, j] + a.yy[i, j] * b.yy[i, j] + 2.0 * (a.xy[i, j] * b.xy[i, j]);
  });
}

namespace {

// max(1, |p| / radius) of every pixel of a row from its squared norm.
void scales_of(std::span<const double> squares, double radius, std::span<double> scales) noexcept {
  for (std::size_t j = 0; j < scales.size(); ++j) scales[j] = std::max(1.0, std::sqrt(squares[j]) / radius);
}

// Row i of a field's entries divided by the scales of its pixels.
void divide(Grid<double> entries, std::size_t i, std::span<const double> scales) noexcept {
  for (std::size_t j = 0; j < scales.size(); ++j) entries[i, j] = entries[i, j] / scales[j];
}

// The squares of the entries of the pixels of row i added to `squares`.
void add_squares(const VectorView<double>& field, std::size_t i, std::span<double> squares) noexcept {
  const Grid<double> x = field.x;
  const Grid<double> y = field.y;
  for (std::size_t j = 0; j < squares.size(); ++j) squares[j] += x[i, j] * x[i, j] + y[i, j] * y[i, j];
}

void add_squares(const TensorView<double>& field, std::size_t i, std::span<double> squares) noexcept {
  const Grid<double> xx = field.xx;
  const Grid<double> yy = field.yy;
  const Grid<double> xy = field.xy;
  for (std::size_t j = 0; j < squares.size(); ++j) {
    squares[j] += xx[i, j] * xx[i, j] + yy[i, j] * yy[i, j] + 2.0 * (xy[i, j] * xy[i, j]);
  }
}

std::array<Grid<double>, 2> entries_of(const VectorView<double>& field) noexcept { return {field.x, field.y}; }
std::array<Grid<double>, 3> entries_of(const TensorView<double>& field) noexcept {
  return {field.xx, field.yy, field.xy};
}

// The projection onto the balls, row by row: the squared norms of the pixels of a
// row, summed over the channels in their order (coupled: all of them; apart: each
// channel's own), and then every entry of a pixel divided by max(1, |p| / radius).
template <typename Field>
void project_rows(std::span<const Field> fields, double radius, bool coupled) noexcept {
  if (fields.empty()) return;
  const GridExtents shape = entries_of(fields.front())[0].extents();
  for (const Field& field : fields) {
    for (const Grid<double>& grid : entries_of(field)) same_shape(shape, grid);
  }
  const std::size_t height = shape.extent(0);
  const std::size_t width = shape.extent(1);
  std::vector<double> squares_storage(width);
  std::vector<double> scales_storage(width);
  const std::span<double> squares(squares_storage);
  const std::span<double> scales(scales_storage);
  for (std::size_t i = 0; i < height; ++i) {
    if (coupled) {
      std::ranges::fill(squares, 0.0);
      for (const Field& field : fields) add_squares(field, i, squares);
      scales_of(squares, radius, scales);
      for (const Field& field : fields) {
        for (const Grid<double>& grid : entries_of(field)) divide(grid, i, scales);
      }
    } else {
      for (const Field& field : fields) {
        std::ranges::fill(squares, 0.0);
        add_squares(field, i, squares);
        scales_of(squares, radius, scales);
        for (const Grid<double>& grid : entries_of(field)) divide(grid, i, scales);
      }
    }
  }
}

}  // namespace

void project(std::span<const VectorView<double>> fields, double radius, bool coupled) noexcept {
  project_rows(fields, radius, coupled);
}

void project(std::span<const TensorView<double>> fields, double radius, bool coupled) noexcept {
  project_rows(fields, radius, coupled);
}

}  // namespace unround::operators
