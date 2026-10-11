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

// The entries of row i of a grid of at least one column.
std::span<const double> row_of(Grid<const double> f, std::size_t i) noexcept { return {&f[i, 0], f.extent(1)}; }
std::span<double> row_of(Grid<double> f, std::size_t i) noexcept { return {&f[i, 0], f.extent(1)}; }

// Row i of d_x f into `out`, from f's row i: f[i, j + 1] - f[i, j], and 0 in the last
// column.
template <bool add>
void forward_across(std::span<const double> row, std::span<double> out) noexcept {
  const std::size_t width = out.size();
  const double* from = row.data();
  double* to = out.data();
  for (std::size_t j = 0; j + 1u < width; ++j) put<add>(to[j], from[j + 1u] - from[j]);
  put<add>(to[width - 1u], 0.0);
}

// Row i of d_y f into `out`, from f's rows i and i + 1 (`below`, none in the last row):
// f[i + 1, j] - f[i, j], and 0 in the last row.
template <bool add>
void forward_down(std::span<const double> row, std::span<const double> below, std::span<double> out) noexcept {
  const std::size_t width = out.size();
  const double* from = row.data();
  double* to = out.data();
  if (!below.empty()) {
    const double* next = below.data();
    for (std::size_t j = 0; j < width; ++j) put<add>(to[j], next[j] - from[j]);
  } else {
    for (std::size_t j = 0; j < width; ++j) put<add>(to[j], 0.0);
  }
}

// Row i of delta_x f into `out`, from f's row i: f[i, j] (not in the last column) -
// f[i, j - 1] (not in the first).
template <bool add>
void backward_across(std::span<const double> row, std::span<double> out) noexcept {
  const std::size_t width = out.size();
  const double* from = row.data();
  double* to = out.data();
  if (width == 1u) {
    put<add>(to[0], 0.0);
    return;
  }
  put<add>(to[0], from[0]);
  for (std::size_t j = 1; j + 1u < width; ++j) put<add>(to[j], from[j] - from[j - 1u]);
  put<add>(to[width - 1u], -from[width - 2u]);
}

// Row i of delta_y f into `out`, from f's rows i and i - 1 (`above`, none in the first
// row), `last` where i is the last: f[i, j] (not in the last row) - f[i - 1, j] (not in
// the first).
template <bool add>
void backward_down(std::span<const double> row, std::span<const double> above, bool last,
                   std::span<double> out) noexcept {
  const std::size_t width = out.size();
  const double* from = row.data();
  double* to = out.data();
  const bool first = above.empty();
  if (!first && !last) {
    const double* previous = above.data();
    for (std::size_t j = 0; j < width; ++j) put<add>(to[j], from[j] - previous[j]);
  } else if (!last) {
    for (std::size_t j = 0; j < width; ++j) put<add>(to[j], from[j]);
  } else if (!first) {
    const double* previous = above.data();
    for (std::size_t j = 0; j < width; ++j) put<add>(to[j], -previous[j]);
  } else {
    for (std::size_t j = 0; j < width; ++j) put<add>(to[j], 0.0);
  }
}

// The same of row i of a grid of at least one column.
template <bool add>
void forward_across(Grid<const double> f, std::size_t i, std::span<double> out) noexcept {
  forward_across<add>(row_of(f, i), out);
}

template <bool add>
void forward_down(Grid<const double> f, std::size_t i, std::span<double> out) noexcept {
  forward_down<add>(row_of(f, i), i + 1u < f.extent(0) ? row_of(f, i + 1u) : std::span<const double>{}, out);
}

template <bool add>
void backward_across(Grid<const double> f, std::size_t i, std::span<double> out) noexcept {
  backward_across<add>(row_of(f, i), out);
}

template <bool add>
void backward_down(Grid<const double> f, std::size_t i, std::span<double> out) noexcept {
  backward_down<add>(row_of(f, i), i > 0u ? row_of(f, i - 1u) : std::span<const double>{}, i + 1u == f.extent(0), out);
}

// out = out / 2, which is exact.
void halve(std::span<double> out) noexcept {
  double* to = out.data();
  const std::size_t width = out.size();
  for (std::size_t j = 0; j < width; ++j) to[j] = 0.5 * to[j];
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

namespace {

// Rows beyond the grid are a mistake of the caller, which ends the program.
void within(Rows rows, std::size_t height) noexcept {
  if (rows.first > rows.end || rows.end > height) std::abort();
}

// So is a row beyond it, or a row of the output of another width than the grid's.
template <typename... Spans>
void row_within(const GridExtents& shape, std::size_t i, const Spans&... out) noexcept {
  if (i >= shape.extent(0) || ((out.size() != shape.extent(1)) || ...)) std::abort();
}

// The rows of a difference, of a grid of at least one column.
void gradient_of(Grid<const double> field, std::size_t i, std::span<double> x, std::span<double> y) noexcept {
  forward_across<false>(field, i, x);
  forward_down<false>(field, i, y);
}

void divergence_of(VectorView<const double> p, std::size_t i, std::span<double> out) noexcept {
  backward_down<false>(p.y, i, out);
  backward_across<true>(p.x, i, out);
}

void symmetrized_gradient_of(VectorView<const double> w, std::size_t i, std::span<double> xx, std::span<double> yy,
                             std::span<double> xy) noexcept {
  backward_across<false>(w.x, i, xx);
  backward_down<false>(w.y, i, yy);
  // (delta_y w_1 + delta_x w_2) / 2: the sum, then its half, which is exact.
  backward_down<false>(w.x, i, xy);
  backward_across<true>(w.y, i, xy);
  halve(xy);
}

void tensor_divergence_of(TensorView<const double> r, std::size_t i, std::span<double> x,
                          std::span<double> y) noexcept {
  // d_x r11 + d_y r12, and d_x r12 + d_y r22.
  forward_across<false>(r.xx, i, x);
  forward_down<true>(r.xy, i, x);
  forward_across<false>(r.xy, i, y);
  forward_down<true>(r.yy, i, y);
}

}  // namespace

void gradient(Grid<const double> field, VectorView<double> out) noexcept {
  gradient(field, out, Rows{.first = 0, .end = field.extent(0)});
}

void gradient(Grid<const double> field, VectorView<double> out, Rows rows) noexcept {
  same_shape(field.extents(), out.x, out.y);
  within(rows, field.extent(0));
  if (field.extent(1) == 0u) return;
  const std::size_t end = rows.end;
  for (std::size_t i = rows.first; i < end; ++i) gradient_of(field, i, row_of(out.x, i), row_of(out.y, i));
}

void gradient(Grid<const double> field, std::size_t i, std::span<double> x, std::span<double> y) noexcept {
  row_within(field.extents(), i, x, y);
  if (field.extent(1) == 0u) return;
  gradient_of(field, i, x, y);
}

void gradient(std::span<const double> row, std::span<const double> below, std::span<double> x,
              std::span<double> y) noexcept {
  const std::size_t width = row.size();
  if (x.size() != width || y.size() != width || (!below.empty() && below.size() != width)) std::abort();
  if (width == 0u) return;
  forward_across<false>(row, x);
  forward_down<false>(row, below, y);
}

void divergence(VectorView<const double> p, Grid<double> out) noexcept {
  divergence(p, out, Rows{.first = 0, .end = out.extent(0)});
}

void divergence(VectorView<const double> p, Grid<double> out, Rows rows) noexcept {
  same_shape(out.extents(), p.x, p.y);
  within(rows, out.extent(0));
  if (out.extent(1) == 0u) return;
  const std::size_t end = rows.end;
  for (std::size_t i = rows.first; i < end; ++i) divergence_of(p, i, row_of(out, i));
}

void divergence(VectorView<const double> p, std::size_t i, std::span<double> out) noexcept {
  same_shape(p.x.extents(), p.y);
  row_within(p.x.extents(), i, out);
  if (p.x.extent(1) == 0u) return;
  divergence_of(p, i, out);
}

void symmetrized_gradient(VectorView<const double> w, TensorView<double> out) noexcept {
  symmetrized_gradient(w, out, Rows{.first = 0, .end = w.x.extent(0)});
}

void symmetrized_gradient(VectorView<const double> w, TensorView<double> out, Rows rows) noexcept {
  same_shape(w.x.extents(), w.y, out.xx, out.yy, out.xy);
  within(rows, w.x.extent(0));
  if (w.x.extent(1) == 0u) return;
  const std::size_t end = rows.end;
  for (std::size_t i = rows.first; i < end; ++i) {
    symmetrized_gradient_of(w, i, row_of(out.xx, i), row_of(out.yy, i), row_of(out.xy, i));
  }
}

void symmetrized_gradient(VectorView<const double> w, std::size_t i, std::span<double> xx, std::span<double> yy,
                          std::span<double> xy) noexcept {
  same_shape(w.x.extents(), w.y);
  row_within(w.x.extents(), i, xx, yy, xy);
  if (w.x.extent(1) == 0u) return;
  symmetrized_gradient_of(w, i, xx, yy, xy);
}

void symmetrized_gradient(std::span<const double> wx, std::span<const double> wy, std::span<const double> above_x,
                          std::span<const double> above_y, bool last, std::span<double> xx, std::span<double> yy,
                          std::span<double> xy) noexcept {
  const std::size_t width = wx.size();
  const bool first = above_x.empty();
  if (wy.size() != width || xx.size() != width || yy.size() != width || xy.size() != width ||
      above_y.empty() != first || (!first && (above_x.size() != width || above_y.size() != width))) {
    std::abort();
  }
  if (width == 0u) return;
  backward_across<false>(wx, xx);
  backward_down<false>(wy, above_y, last, yy);
  // (delta_y w_1 + delta_x w_2) / 2: the sum, then its half, which is exact.
  backward_down<false>(wx, above_x, last, xy);
  backward_across<true>(wy, xy);
  halve(xy);
}

void tensor_divergence(TensorView<const double> r, VectorView<double> out) noexcept {
  tensor_divergence(r, out, Rows{.first = 0, .end = r.xx.extent(0)});
}

void tensor_divergence(TensorView<const double> r, VectorView<double> out, Rows rows) noexcept {
  same_shape(r.xx.extents(), r.yy, r.xy, out.x, out.y);
  within(rows, r.xx.extent(0));
  if (r.xx.extent(1) == 0u) return;
  const std::size_t end = rows.end;
  for (std::size_t i = rows.first; i < end; ++i) tensor_divergence_of(r, i, row_of(out.x, i), row_of(out.y, i));
}

void tensor_divergence(TensorView<const double> r, std::size_t i, std::span<double> x, std::span<double> y) noexcept {
  same_shape(r.xx.extents(), r.yy, r.xy);
  row_within(r.xx.extents(), i, x, y);
  if (r.xx.extent(1) == 0u) return;
  tensor_divergence_of(r, i, x, y);
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

// What every pixel of a row is multiplied by, from its squared norm (ball_scale).
void scales_of(std::span<const double> squares, double radius, std::span<double> scales) noexcept {
  for (std::size_t j = 0; j < scales.size(); ++j) scales[j] = ball_scale(squares[j], radius);
}

// Row i of a field's entries multiplied by the scales of its pixels.
void scale(Grid<double> entries, std::size_t i, std::span<const double> scales) noexcept {
  for (std::size_t j = 0; j < scales.size(); ++j) entries[i, j] = entries[i, j] * scales[j];
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
// channel's own), and then every entry of a pixel multiplied by its scale.
// The squares and the scales of a row are kept in the first entries of
// `squares_room` and `scales_room`, as many as the fields' columns.
template <typename Field>
void project_rows(std::span<const Field> fields, double radius, bool coupled, Rows rows, std::span<double> squares_room,
                  std::span<double> scales_room) noexcept {
  if (fields.empty()) return;
  const GridExtents shape = entries_of(fields.front())[0].extents();
  for (const Field& field : fields) {
    for (const Grid<double>& grid : entries_of(field)) same_shape(shape, grid);
  }
  within(rows, shape.extent(0));
  const std::size_t width = shape.extent(1);
  if (squares_room.size() < width || scales_room.size() < width) std::abort();
  const std::span<double> squares = squares_room.first(width);
  const std::span<double> scales = scales_room.first(width);
  const std::size_t end = rows.end;
  for (std::size_t i = rows.first; i < end; ++i) {
    if (coupled) {
      std::ranges::fill(squares, 0.0);
      for (const Field& field : fields) add_squares(field, i, squares);
      scales_of(squares, radius, scales);
      for (const Field& field : fields) {
        for (const Grid<double>& grid : entries_of(field)) scale(grid, i, scales);
      }
    } else {
      for (const Field& field : fields) {
        std::ranges::fill(squares, 0.0);
        add_squares(field, i, squares);
        scales_of(squares, radius, scales);
        for (const Grid<double>& grid : entries_of(field)) scale(grid, i, scales);
      }
    }
  }
}

// The projection of every row, with room of its own.
template <typename Field>
void project_all(std::span<const Field> fields, double radius, bool coupled) noexcept {
  if (fields.empty()) return;
  const GridExtents shape = entries_of(fields.front())[0].extents();
  std::vector<double> squares(shape.extent(1));
  std::vector<double> scales(shape.extent(1));
  project_rows(fields, radius, coupled, Rows{.first = 0, .end = shape.extent(0)}, squares, scales);
}

}  // namespace

void project(std::span<const VectorView<double>> fields, double radius, bool coupled) noexcept {
  project_all(fields, radius, coupled);
}

void project(std::span<const TensorView<double>> fields, double radius, bool coupled) noexcept {
  project_all(fields, radius, coupled);
}

void project(std::span<const VectorView<double>> fields, double radius, bool coupled, Rows rows,
             std::span<double> squares, std::span<double> scales) noexcept {
  project_rows(fields, radius, coupled, rows, squares, scales);
}

void project(std::span<const TensorView<double>> fields, double radius, bool coupled, Rows rows,
             std::span<double> squares, std::span<double> scales) noexcept {
  project_rows(fields, radius, coupled, rows, squares, scales);
}

}  // namespace unround::operators
