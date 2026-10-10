// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The components of a picture on one canvas: docs/math.md, 1.3, 4.4 and 6.6.
//
// A cell is taken row by row of the canvas: the rows of a cell are added down the
// columns, a loop over the row, and then the samples of each cell across, a loop
// over the cells of the row whose number of samples across is a constant of its
// template (1 to 4). The change of a cell is spread back over its samples across the
// same way, and added to the rows of the cell one after another. Each is the same
// operation on neighbouring values: vector code.

#include "unround/frames.hpp"

#include "unround/arrays.hpp"
#include "unround/dct.hpp"
#include "unround/error.hpp"
#include "unround/exact.hpp"
#include "unround/model.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <expected>
#include <format>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace unround {

namespace {

constexpr std::size_t block = 8;

std::unexpected<Error> refuse(std::string message) {
  return std::unexpected(Error{.code = Errc::unsupported, .message = std::move(message)});
}

std::size_t ceiling(std::size_t numerator, std::size_t denominator) noexcept {
  return denominator == 0u ? 0u : (numerator + denominator - 1u) / denominator;
}

// Row i of a grid, as a span of its samples.
std::span<const double> row_of(Grid<const double> grid, std::size_t i) noexcept {
  return std::span<const double>(&grid[i, 0], grid.extent(1));
}
std::span<double> row_of(Grid<double> grid, std::size_t i) noexcept {
  return std::span<double>(&grid[i, 0], grid.extent(1));
}

// out[j] = the sum of from[across j + b] over b < across, j < count, divided by
// `cells` where `divide`.
template <std::size_t across, bool divide>
void gather(std::span<const double> from, std::span<double> out, std::size_t count, double cells) noexcept {
  const double* source = from.data();
  double* target = out.data();
  for (std::size_t j = 0; j < count; ++j) {
    double sum = source[across * j];
    for (std::size_t b = 1; b < across; ++b) sum += source[across * j + b];
    if constexpr (divide) {
      target[j] = sum / cells;
    } else {
      target[j] = sum;
    }
  }
}

template <bool divide>
void gather_by(std::size_t across, std::span<const double> from, std::span<double> out, std::size_t count,
               double cells) noexcept {
  switch (across) {
    case 1:
      gather<1, divide>(from, out, count, cells);
      return;
    case 2:
      gather<2, divide>(from, out, count, cells);
      return;
    case 3:
      gather<3, divide>(from, out, count, cells);
      return;
    case 4:
      gather<4, divide>(from, out, count, cells);
      return;
    default:
      std::abort();
  }
}

// out[across j + b] = from[j] for b < across, j < count.
template <std::size_t across>
void spread(std::span<const double> from, std::span<double> out, std::size_t count) noexcept {
  const double* source = from.data();
  double* target = out.data();
  for (std::size_t j = 0; j < count; ++j) {
    for (std::size_t b = 0; b < across; ++b) target[across * j + b] = source[j];
  }
}

void spread_by(std::size_t across, std::span<const double> from, std::span<double> out, std::size_t count) noexcept {
  switch (across) {
    case 1:
      spread<1>(from, out, count);
      return;
    case 2:
      spread<2>(from, out, count);
      return;
    case 3:
      spread<3>(from, out, count);
      return;
    case 4:
      spread<4>(from, out, count);
      return;
    default:
      std::abort();
  }
}

// out[k] = from[k] + change[k], k < count.
void add_rows(std::span<const double> from, std::span<const double> change, std::span<double> out,
              std::size_t count) noexcept {
  const double* a = from.data();
  const double* b = change.data();
  double* target = out.data();
  for (std::size_t k = 0; k < count; ++k) target[k] = a[k] + b[k];
}

// The cells of rows `down` x `across` of x, summed (or their means) into out, 8 b^v x
// 8 b^h: those of the first 8 b^v down rows and 8 b^h across columns. `sums` holds a
// row of the canvas.
template <bool divide>
void cells_of(Grid<const double> x, std::size_t down, std::size_t across, Grid<double> out, std::vector<double>& sums) {
  const std::size_t rows = out.extent(0);
  const std::size_t columns = out.extent(1);
  const std::size_t span = columns * across;
  const double cells = static_cast<double>(down * across);
  sums.resize(std::max(sums.size(), span));
  for (std::size_t i = 0; i < rows; ++i) {
    const std::span<const double> first = row_of(x, down * i);
    std::copy_n(first.begin(), span, sums.begin());
    for (std::size_t a = 1; a < down; ++a) {
      const std::span<const double> next = row_of(x, down * i + a);
      add_rows(sums, next, sums, span);
    }
    gather_by<divide>(across, sums, row_of(out, i), columns, cells);
  }
}

}  // namespace

std::expected<Frame, Error> Frame::make(std::size_t height, std::size_t width, ColourSpace space,
                                        std::span<const ComponentInput> components, std::span<const DataTerm> data) {
  const std::size_t count = components.size();
  const std::size_t expected_count = space == ColourSpace::grayscale ? 1u : 3u;
  if (count != expected_count) {
    return std::unexpected(Error{
        .code = Errc::options,
        .message = std::format("a picture of this colour space has {} components, not {}", expected_count, count)});
  }
  if (data.size() != 1u && data.size() != count) {
    return std::unexpected(
        Error{.code = Errc::options,
              .message = std::format("the data term is one for every component or one for each, not {}", data.size())});
  }
  if (height == 0u || width == 0u) return refuse("a picture without samples is refused");
  std::uint32_t h_max = 0;
  std::uint32_t v_max = 0;
  for (const ComponentInput& component : components) {
    if (component.h < 1u || component.h > 4u || component.v < 1u || component.v > 4u) {
      return refuse(std::format("sampling factors are 1 to 4, not {} by {}", component.h, component.v));
    }
    h_max = std::max(h_max, component.h);
    v_max = std::max(v_max, component.v);
  }
  Frame frame;
  frame.height_ = height;
  frame.width_ = width;
  frame.space_ = space;
  if (count == 1u) {
    // The canvas is the component's blocks.
    const Blocks<const std::int16_t> levels = components[0].levels;
    frame.rows_ = block * levels.extent(0);
    frame.columns_ = block * levels.extent(1);
    if (frame.rows_ < height || frame.columns_ < width) {
      return refuse(std::format("the blocks, {} by {}, do not cover the picture, {} by {}", levels.extent(0),
                                levels.extent(1), height, width));
    }
  } else {
    frame.rows_ = block * v_max * ceiling(height, block * v_max);
    frame.columns_ = block * h_max * ceiling(width, block * h_max);
  }
  for (std::size_t c = 0; c < count; ++c) {
    const ComponentInput& input = components[c];
    if (h_max % input.h != 0u || v_max % input.v != 0u) {
      return refuse(
          std::format("the sampling factors {}x{} of component {} are not whole fractions of the largest, {}x{}",
                      input.h, input.v, c, h_max, v_max));
    }
    const std::size_t down = count == 1u ? 1u : v_max / input.v;
    const std::size_t across = count == 1u ? 1u : h_max / input.h;
    const std::size_t block_rows = count == 1u ? input.levels.extent(0) : ceiling(height, block * down);
    const std::size_t block_columns = count == 1u ? input.levels.extent(1) : ceiling(width, block * across);
    if (input.levels.extent(0) < block_rows || input.levels.extent(1) < block_columns) {
      return refuse(std::format("component {} has {} by {} blocks, fewer than the picture covers, {} by {}", c,
                                input.levels.extent(0), input.levels.extent(1), block_rows, block_columns));
    }
    // The blocks kept, the top left ones of the file's.
    BlocksArray<std::int16_t> kept(block_extents(block_rows, block_columns));
    const Blocks<std::int16_t> kept_view = kept.view();
    const Blocks<const std::int16_t> given = input.levels;
    for (std::size_t by = 0; by < block_rows; ++by) {
      for (std::size_t bx = 0; bx < block_columns; ++bx) {
        const std::span<const std::int16_t, 64> from = block_entries(given, by, bx);
        std::ranges::copy(from, block_entries(kept_view, by, bx).begin());
      }
    }
    DataTerm term = data.size() == 1u ? data[0] : data[c];
    if (space == ColourSpace::ycbcr && c > 0u) term = term.of_chroma();
    std::expected<Problem, Error> problem = Problem::make(std::as_const(kept).view(), input.table, term);
    if (!problem) {
      return std::unexpected(
          Error{.code = problem.error().code, .message = std::format("component {}: {}", c, problem.error().message)});
    }
    frame.components_.push_back(Component{.problem = std::move(*problem), .cell_rows = down, .cell_columns = across});
  }
  return frame;
}

bool Frame::free(std::size_t c) const noexcept {
  const Component& component = components_[c];
  return component.cells() > 1u || block * component.block_rows() * component.cell_rows < rows_ ||
         block * component.block_columns() * component.cell_columns < columns_;
}

bool Frame::free(void) const noexcept {
  for (std::size_t c = 0; c < components_.size(); ++c) {
    if (free(c)) return true;
  }
  return false;
}

Room Frame::room(std::size_t c) const {
  const Component& component = components_[c];
  const GridExtents cells = component.cells() == 1u
                                ? grid_extents(0, 0)
                                : grid_extents(block * component.block_rows(), block * component.block_columns());
  return Room{.coefficients = BlocksArray<double>(block_extents(component.block_rows(), component.block_columns())),
              .means = GridArray<double>(cells),
              .change = GridArray<double>(cells),
              .row = std::vector<double>(columns_)};
}

void Frame::moved(std::size_t c, Grid<const double> v, Grid<const double> change, Grid<double> out,
                  std::vector<double>& row) const {
  const Component& component = components_[c];
  const std::size_t down = component.cell_rows;
  const std::size_t across = component.cell_columns;
  const std::size_t cell_rows = change.extent(0);
  const std::size_t cell_columns = change.extent(1);
  const std::size_t span = cell_columns * across;
  const std::size_t rows = v.extent(0);
  const std::size_t columns = v.extent(1);
  // Where out is v, what lies beyond the blocks is in its place already.
  const bool apart = out.data_handle() != v.data_handle();
  row.resize(std::max(row.size(), span));
  for (std::size_t i = 0; i < cell_rows; ++i) {
    spread_by(across, row_of(change, i), row, cell_columns);
    for (std::size_t a = 0; a < down; ++a) {
      const std::size_t canvas_row = down * i + a;
      const std::span<const double> from = row_of(v, canvas_row);
      const std::span<double> to = row_of(out, canvas_row);
      add_rows(from, row, to, span);
      if (apart) {
        std::copy(from.begin() + static_cast<std::ptrdiff_t>(span), from.end(),
                  to.begin() + static_cast<std::ptrdiff_t>(span));
      }
    }
  }
  if (!apart) return;
  for (std::size_t canvas_row = down * cell_rows; canvas_row < rows; ++canvas_row) {
    const std::span<const double> from = row_of(v, canvas_row);
    std::copy_n(from.begin(), columns, row_of(out, canvas_row).begin());
  }
}

void Frame::coefficients(std::size_t c, Grid<const double> x, Blocks<double> out) const {
  const Component& component = components_[c];
  if (component.cells() == 1u) {
    dct::forward(x, out);
    return;
  }
  GridArray<double> cell_means(grid_extents(block * component.block_rows(), block * component.block_columns()));
  std::vector<double> row;
  cells_of<true>(x, component.cell_rows, component.cell_columns, cell_means.view(), row);
  dct::forward(std::as_const(cell_means).view(), out);
}

template <typename Map>
void Frame::map(std::size_t c, Grid<const double> v, Grid<double> out, Blocks<double> kept, Room& room,
                const Map& zeta_of) const {
  const Component& component = components_[c];
  const Blocks<double> at = room.coefficients.view();
  if (component.cells() == 1u) {
    dct::forward(v, at);
    zeta_of(std::as_const(room.coefficients).view(), kept);
    if (out.data_handle() != v.data_handle()) {
      const std::span<const double> from(v.data_handle(), v.size());
      std::ranges::copy(from, out.data_handle());
    }
    dct::inverse(kept, out);
    return;
  }
  cells_of<true>(v, component.cell_rows, component.cell_columns, room.means.view(), room.row);
  dct::forward(std::as_const(room.means).view(), at);
  zeta_of(std::as_const(room.coefficients).view(), kept);
  dct::inverse(kept, room.change.view());
  // The change of each cell: the inverse DCT of zeta less the mean of v.
  const std::span<double> changes = room.change.entries();
  const std::span<const double> means = std::as_const(room.means).entries();
  const std::size_t count = changes.size();
  for (std::size_t k = 0; k < count; ++k) changes[k] -= means[k];
  moved(c, v, std::as_const(room.change).view(), out, room.row);
}

void Frame::prox(std::size_t c, Grid<const double> v, double tau, Grid<double> out, Blocks<double> kept,
                 Room& room) const {
  const Component& component = components_[c];
  const double step = tau / static_cast<double>(component.cells());
  const Problem& problem = component.problem;
  map(c, v, out, kept, room,
      [&problem, step](Blocks<const double> e, Blocks<double> zeta) { problem.prox(e, step, zeta); });
}

void Frame::project(std::size_t c, Grid<const double> v, Grid<double> out, Blocks<double> kept, Room& room) const {
  const Problem& problem = components_[c].problem;
  map(c, v, out, kept, room, [&problem](Blocks<const double> e, Blocks<double> zeta) { problem.clip(e, zeta); });
}

void Frame::canvas(std::size_t c, Blocks<const double> coefficients, double beyond, Grid<double> out) const {
  const Component& component = components_[c];
  const std::span<double> entries(out.data_handle(), out.size());
  std::ranges::fill(entries, beyond);
  if (component.cells() == 1u) {
    dct::inverse(coefficients, out);
    return;
  }
  const std::size_t down = component.cell_rows;
  const std::size_t across = component.cell_columns;
  GridArray<double> samples(grid_extents(block * component.block_rows(), block * component.block_columns()));
  dct::inverse(coefficients, samples.view());
  const Grid<const double> back = std::as_const(samples).view();
  const std::size_t cell_rows = back.extent(0);
  const std::size_t cell_columns = back.extent(1);
  for (std::size_t i = 0; i < cell_rows; ++i) {
    for (std::size_t a = 0; a < down; ++a) spread_by(across, row_of(back, i), row_of(out, down * i + a), cell_columns);
  }
}

void Frame::start(std::size_t c, Grid<double> out, Blocks<double> kept) const {
  const Problem& problem = components_[c].problem;
  problem.clip(problem.centres(), kept);
  canvas(c, kept, beyond_blocks, out);
}

double Frame::conjugate(std::size_t c, Grid<const double> xi, double radius) const {
  const Component& component = components_[c];
  const BlockExtents extents = block_extents(component.block_rows(), component.block_columns());
  BlocksArray<double> s(extents);
  std::vector<double> buffer;
  if (component.cells() == 1u) {
    dct::forward(xi, s.view());
  } else {
    GridArray<double> cell_sums(grid_extents(block * component.block_rows(), block * component.block_columns()));
    cells_of<false>(xi, component.cell_rows, component.cell_columns, cell_sums.view(), buffer);
    dct::forward(std::as_const(cell_sums).view(), s.view());
  }
  const double value = component.problem.conjugate(std::as_const(s).view());
  if (!free(c)) return value;
  // The terms of <zeta, m> + R |zeta|_1, sample by sample: within the blocks, R times
  // the deviation from the mean of the cell; beyond them, 128 xi + R |xi|.
  const std::size_t down = component.cell_rows;
  const std::size_t across = component.cell_columns;
  const std::size_t rows = xi.extent(0);
  const std::size_t columns = xi.extent(1);
  const std::size_t cell_rows = block * component.block_rows();
  const std::size_t cell_columns = block * component.block_columns();
  const std::size_t span = cell_columns * across;
  GridArray<double> terms(grid_extents(rows, columns));
  const Grid<double> term_view = terms.view();
  GridArray<double> cell_means(grid_extents(cell_rows, cell_columns));
  cells_of<true>(xi, down, across, cell_means.view(), buffer);
  std::vector<double> spread_row(span);
  for (std::size_t i = 0; i < cell_rows; ++i) {
    spread_by(across, row_of(std::as_const(cell_means).view(), i), spread_row, cell_columns);
    for (std::size_t a = 0; a < down; ++a) {
      const std::size_t row = down * i + a;
      const std::span<const double> from = row_of(xi, row);
      const std::span<double> to = row_of(term_view, row);
      const double* mean_row = spread_row.data();
      for (std::size_t k = 0; k < span; ++k) to[k] = radius * std::abs(from[k] - mean_row[k]);
      for (std::size_t k = span; k < columns; ++k) to[k] = beyond_blocks * from[k] + radius * std::abs(from[k]);
    }
  }
  for (std::size_t row = down * cell_rows; row < rows; ++row) {
    const std::span<const double> from = row_of(xi, row);
    const std::span<double> to = row_of(term_view, row);
    for (std::size_t k = 0; k < columns; ++k) to[k] = beyond_blocks * from[k] + radius * std::abs(from[k]);
  }
  return value + exact::sum_rows(std::as_const(terms).view());
}

}  // namespace unround
