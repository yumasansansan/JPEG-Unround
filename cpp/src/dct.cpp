// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The DCT of docs/math.md, 1.1.

#include "unround/dct.hpp"

#include "unround/arrays.hpp"

#include <array>
#include <cstddef>
#include <cstdlib>
#include <span>
#include <utility>

namespace unround::dct {

namespace {

// cos(pi j / 16), j = 0, ..., 8, to 25 digits: the double of each literal is the
// double nearest to the cosine.
constexpr std::array<double, 9> cosines = {
    1.0,
    9.807852804032304491261822e-1,
    9.238795325112867561281832e-1,
    8.314696123025452370787884e-1,
    7.071067811865475244008444e-1,
    5.555702330196022247428308e-1,
    3.826834323650897717284600e-1,
    1.950903220161282678482849e-1,
    0.0,
};

// sqrt(1/8), gamma_0 times cos 0, to 25 digits.
constexpr double root_eighth = 3.535533905932737622004222e-1;

// cos(pi m / 16) for m in [0, 32): over a period, [0, 8] as it is, (8, 16] and
// (16, 24] negated, (24, 32) mirrored.
constexpr double cosine(std::size_t m) noexcept {
  if (m <= 8u) return cosines[m];
  if (m <= 16u) return -cosines[16u - m];
  if (m <= 24u) return -cosines[m - 16u];
  return cosines[32u - m];
}

// C[k][n]: gamma_k is sqrt(1/8) for k = 0, whose cosine is 1, and 1/2 otherwise,
// by which halving is exact. (2 n + 1) k is reduced modulo 32 in integers.
constexpr Basis make_basis(void) noexcept {
  Basis entries{};
  for (std::size_t n = 0; n < block; ++n) entries[0][n] = root_eighth;
  for (std::size_t k = 1; k < block; ++k) {
    for (std::size_t n = 0; n < block; ++n) entries[k][n] = 0.5 * cosine(((2u * n + 1u) * k) % 32u);
  }
  return entries;
}

constexpr Basis table = make_basis();

}  // namespace

const Basis& basis(void) noexcept { return table; }

namespace {

// Eight rows of a band, `width` samples of each: rows `top` to top + 7 of a grid.
template <typename T>
using Rows8 = std::array<std::span<T>, block>;

template <typename T>
Rows8<T> rows_of(Grid<T> grid, std::size_t top, std::size_t width) noexcept {
  Rows8<T> rows;
  for (std::size_t m = 0; m < block; ++m) rows[m] = std::span<T>(&grid[top + m, 0], width);
  return rows;
}

// y = C x down every column of eight rows: for each column, the sums and differences
// of the ends, then each output as a sum of four products (docs/math.md, 1.1). One
// loop over the columns, the same operations on neighbouring values: vector code. The
// rows are held in local spans, which the compiler keeps in registers.
void forward_columns(const Rows8<const double>& x, const Rows8<double>& y) noexcept {
  const std::size_t width = x[0].size();
  const std::span<const double> x0 = x[0];
  const std::span<const double> x1 = x[1];
  const std::span<const double> x2 = x[2];
  const std::span<const double> x3 = x[3];
  const std::span<const double> x4 = x[4];
  const std::span<const double> x5 = x[5];
  const std::span<const double> x6 = x[6];
  const std::span<const double> x7 = x[7];
  const std::span<double> y0 = y[0];
  const std::span<double> y1 = y[1];
  const std::span<double> y2 = y[2];
  const std::span<double> y3 = y[3];
  const std::span<double> y4 = y[4];
  const std::span<double> y5 = y[5];
  const std::span<double> y6 = y[6];
  const std::span<double> y7 = y[7];
  const std::array<double, block> c0 = table[0];
  const std::array<double, block> c1 = table[1];
  const std::array<double, block> c2 = table[2];
  const std::array<double, block> c3 = table[3];
  const std::array<double, block> c4 = table[4];
  const std::array<double, block> c5 = table[5];
  const std::array<double, block> c6 = table[6];
  const std::array<double, block> c7 = table[7];
  for (std::size_t j = 0; j < width; ++j) {
    const double s0 = x0[j] + x7[j];
    const double s1 = x1[j] + x6[j];
    const double s2 = x2[j] + x5[j];
    const double s3 = x3[j] + x4[j];
    const double d0 = x0[j] - x7[j];
    const double d1 = x1[j] - x6[j];
    const double d2 = x2[j] - x5[j];
    const double d3 = x3[j] - x4[j];
    // Even outputs from the sums, odd ones from the differences.
    y0[j] = c0[0] * s0 + c0[1] * s1 + c0[2] * s2 + c0[3] * s3;
    y1[j] = c1[0] * d0 + c1[1] * d1 + c1[2] * d2 + c1[3] * d3;
    y2[j] = c2[0] * s0 + c2[1] * s1 + c2[2] * s2 + c2[3] * s3;
    y3[j] = c3[0] * d0 + c3[1] * d1 + c3[2] * d2 + c3[3] * d3;
    y4[j] = c4[0] * s0 + c4[1] * s1 + c4[2] * s2 + c4[3] * s3;
    y5[j] = c5[0] * d0 + c5[1] * d1 + c5[2] * d2 + c5[3] * d3;
    y6[j] = c6[0] * s0 + c6[1] * s1 + c6[2] * s2 + c6[3] * s3;
    y7[j] = c7[0] * d0 + c7[1] * d1 + c7[2] * d2 + c7[3] * d3;
  }
}

// x = C^T y down every column likewise: for each output n, the even part from the
// even frequencies and the odd part from the odd ones, each a sum of four products;
// then their sum and difference.
void inverse_columns(const Rows8<const double>& y, const Rows8<double>& x) noexcept {
  const std::size_t width = y[0].size();
  const std::span<const double> y0 = y[0];
  const std::span<const double> y1 = y[1];
  const std::span<const double> y2 = y[2];
  const std::span<const double> y3 = y[3];
  const std::span<const double> y4 = y[4];
  const std::span<const double> y5 = y[5];
  const std::span<const double> y6 = y[6];
  const std::span<const double> y7 = y[7];
  const std::span<double> x0 = x[0];
  const std::span<double> x1 = x[1];
  const std::span<double> x2 = x[2];
  const std::span<double> x3 = x[3];
  const std::span<double> x4 = x[4];
  const std::span<double> x5 = x[5];
  const std::span<double> x6 = x[6];
  const std::span<double> x7 = x[7];
  const std::array<double, block> c0 = table[0];
  const std::array<double, block> c1 = table[1];
  const std::array<double, block> c2 = table[2];
  const std::array<double, block> c3 = table[3];
  const std::array<double, block> c4 = table[4];
  const std::array<double, block> c5 = table[5];
  const std::array<double, block> c6 = table[6];
  const std::array<double, block> c7 = table[7];
  for (std::size_t j = 0; j < width; ++j) {
    const double a0 = y0[j];
    const double a1 = y1[j];
    const double a2 = y2[j];
    const double a3 = y3[j];
    const double a4 = y4[j];
    const double a5 = y5[j];
    const double a6 = y6[j];
    const double a7 = y7[j];
    const double even0 = c0[0] * a0 + c2[0] * a2 + c4[0] * a4 + c6[0] * a6;
    const double odd0 = c1[0] * a1 + c3[0] * a3 + c5[0] * a5 + c7[0] * a7;
    const double even1 = c0[1] * a0 + c2[1] * a2 + c4[1] * a4 + c6[1] * a6;
    const double odd1 = c1[1] * a1 + c3[1] * a3 + c5[1] * a5 + c7[1] * a7;
    const double even2 = c0[2] * a0 + c2[2] * a2 + c4[2] * a4 + c6[2] * a6;
    const double odd2 = c1[2] * a1 + c3[2] * a3 + c5[2] * a5 + c7[2] * a7;
    const double even3 = c0[3] * a0 + c2[3] * a2 + c4[3] * a4 + c6[3] * a6;
    const double odd3 = c1[3] * a1 + c3[3] * a3 + c5[3] * a5 + c7[3] * a7;
    x0[j] = even0 + odd0;
    x7[j] = even0 - odd0;
    x1[j] = even1 + odd1;
    x6[j] = even1 - odd1;
    x2[j] = even2 + odd2;
    x5[j] = even2 - odd2;
    x3[j] = even3 + odd3;
    x4[j] = even3 - odd3;
  }
}

// Every 8 x 8 square of the first `width` columns of eight rows turned about its
// diagonal: from[m, 8 k + n] into to[n, 8 k + m].
void transpose(Grid<const double> from, Grid<double> to, std::size_t width) noexcept {
  for (std::size_t first = 0; first < width; first += block) {
    for (std::size_t m = 0; m < block; ++m) {
      for (std::size_t n = 0; n < block; ++n) to[n, first + m] = from[m, first + n];
    }
  }
}

// The two buffers of a band of eight rows, `width` columns each.
struct Band {
  explicit Band(std::size_t width) : first(grid_extents(block, width)), second(grid_extents(block, width)) {}

  GridArray<double> first;
  GridArray<double> second;
};

// The DCT of the blocks of a band, rows `top` to top + 7 of a canvas: down the
// columns, then along the rows of each block as the columns of the block turned. The
// result is left in band.first, turned: [u, 8 bx + v] holds coefficient (v, u) of
// block bx.
void forward_band(Grid<const double> canvas, std::size_t top, std::size_t width, Band& band) noexcept {
  forward_columns(rows_of(canvas, top, width), rows_of(band.first.view(), 0, width));
  transpose(std::as_const(band.first).view(), band.second.view(), width);
  forward_columns(rows_of(std::as_const(band.second).view(), 0, width), rows_of(band.first.view(), 0, width));
}

// The inverse, from coefficients turned in band.first: along the rows of each block,
// turned back, then down the columns into rows `top` to top + 7 of a canvas.
void inverse_band(Band& band, std::size_t width, Grid<double> canvas, std::size_t top) noexcept {
  inverse_columns(rows_of(std::as_const(band.first).view(), 0, width), rows_of(band.second.view(), 0, width));
  transpose(std::as_const(band.second).view(), band.first.view(), width);
  inverse_columns(rows_of(std::as_const(band.first).view(), 0, width), rows_of(canvas, top, width));
}

}  // namespace

Block forward(const Block& samples) noexcept {
  Band band(block);
  forward_band(Grid<const double>(samples.data(), block, block), 0, block, band);
  Block coefficients{};
  const Block8x8<double> y(coefficients.data());
  const Grid<const double> turned = std::as_const(band.first).view();
  for (std::size_t v = 0; v < block; ++v) {
    for (std::size_t u = 0; u < block; ++u) y[v, u] = turned[u, v];
  }
  return coefficients;
}

Block inverse(const Block& coefficients) noexcept {
  Band band(block);
  const Block8x8<const double> y(coefficients.data());
  const Grid<double> turned = band.first.view();
  for (std::size_t v = 0; v < block; ++v) {
    for (std::size_t u = 0; u < block; ++u) turned[u, v] = y[v, u];
  }
  Block samples{};
  inverse_band(band, block, Grid<double>(samples.data(), block, block), 0);
  return samples;
}

void forward(Grid<const double> canvas, Blocks<double> coefficients) noexcept {
  const std::size_t rows = coefficients.extent(0);
  const std::size_t columns = coefficients.extent(1);
  if (canvas.extent(0) < block * rows || canvas.extent(1) < block * columns) std::abort();
  const std::size_t width = block * columns;
  Band band(width);
  const Grid<const double> turned = std::as_const(band.first).view();
  for (std::size_t by = 0; by < rows; ++by) {
    forward_band(canvas, block * by, width, band);
    for (std::size_t bx = 0; bx < columns; ++bx) {
      for (std::size_t v = 0; v < block; ++v) {
        for (std::size_t u = 0; u < block; ++u) coefficients[by, bx, v, u] = turned[u, block * bx + v];
      }
    }
  }
}

void inverse(Blocks<const double> coefficients, Grid<double> canvas) noexcept {
  const std::size_t rows = coefficients.extent(0);
  const std::size_t columns = coefficients.extent(1);
  if (canvas.extent(0) < block * rows || canvas.extent(1) < block * columns) std::abort();
  const std::size_t width = block * columns;
  Band band(width);
  const Grid<double> turned = band.first.view();
  for (std::size_t by = 0; by < rows; ++by) {
    for (std::size_t bx = 0; bx < columns; ++bx) {
      for (std::size_t v = 0; v < block; ++v) {
        for (std::size_t u = 0; u < block; ++u) turned[u, block * bx + v] = coefficients[by, bx, v, u];
      }
    }
    inverse_band(band, width, canvas, block * by);
  }
}

}  // namespace unround::dct
