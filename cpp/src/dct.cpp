// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The DCT of docs/math.md, 1.1.

#include "unround/dct.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <span>
#include <vector>

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

// The sums and differences of the ends, then each output as a sum of four
// products taken from its first term on; a product may be fused with the sum
// that takes it (docs/math.md, Arithmetic).
void forward8(const double* in, std::size_t in_stride, double* out, std::size_t out_stride) noexcept {
  std::array<double, 4> sums{};
  std::array<double, 4> differences{};
  for (std::size_t j = 0; j < 4u; ++j) {
    const double first = in[j * in_stride];
    const double last = in[(7u - j) * in_stride];
    sums[j] = first + last;
    differences[j] = first - last;
  }
  for (std::size_t i = 0; i < 4u; ++i) {
    const std::array<double, block>& even = table[2u * i];
    const std::array<double, block>& odd = table[2u * i + 1u];
    double y_even = even[0] * sums[0];
    double y_odd = odd[0] * differences[0];
    for (std::size_t j = 1; j < 4u; ++j) {
      y_even = even[j] * sums[j] + y_even;
      y_odd = odd[j] * differences[j] + y_odd;
    }
    out[2u * i * out_stride] = y_even;
    out[(2u * i + 1u) * out_stride] = y_odd;
  }
}

// The even part from the even frequencies and the odd part from the odd ones,
// each a sum of four products from its first term on; then their sum and
// difference.
void inverse8(const double* in, std::size_t in_stride, double* out, std::size_t out_stride) noexcept {
  for (std::size_t n = 0; n < 4u; ++n) {
    double even = table[0][n] * in[0];
    double odd = table[1][n] * in[in_stride];
    for (std::size_t i = 1; i < 4u; ++i) {
      even = table[2u * i][n] * in[2u * i * in_stride] + even;
      odd = table[2u * i + 1u][n] * in[(2u * i + 1u) * in_stride] + odd;
    }
    out[n * out_stride] = even + odd;
    out[(7u - n) * out_stride] = even - odd;
  }
}

Block forward(const Block& samples) noexcept {
  Block columns{};
  for (std::size_t u = 0; u < block; ++u) forward8(&samples[u], block, &columns[u], block);
  Block coefficients{};
  for (std::size_t v = 0; v < block; ++v) forward8(&columns[v * block], 1u, &coefficients[v * block], 1u);
  return coefficients;
}

Block inverse(const Block& coefficients) noexcept {
  Block rows{};
  for (std::size_t v = 0; v < block; ++v) inverse8(&coefficients[v * block], 1u, &rows[v * block], 1u);
  Block samples{};
  for (std::size_t u = 0; u < block; ++u) inverse8(&rows[u], block, &samples[u], block);
  return samples;
}

std::vector<double> forward(std::span<const double> canvas, std::size_t width, std::size_t rows, std::size_t columns) {
  assert(width >= columns * block && canvas.size() >= rows * block * width);
  std::vector<double> coefficients(rows * columns * block_size);
  for (std::size_t by = 0; by < rows; ++by) {
    for (std::size_t bx = 0; bx < columns; ++bx) {
      Block samples{};
      for (std::size_t m = 0; m < block; ++m) {
        for (std::size_t n = 0; n < block; ++n)
          samples[m * block + n] = canvas[(by * block + m) * width + bx * block + n];
      }
      const Block transformed = forward(samples);
      std::copy(transformed.begin(), transformed.end(),
                coefficients.begin() + static_cast<std::ptrdiff_t>((by * columns + bx) * block_size));
    }
  }
  return coefficients;
}

void inverse(std::span<const double> coefficients, std::size_t rows, std::size_t columns, std::span<double> canvas,
             std::size_t width) {
  assert(width >= columns * block && canvas.size() >= rows * block * width &&
         coefficients.size() >= rows * columns * block_size);
  for (std::size_t by = 0; by < rows; ++by) {
    for (std::size_t bx = 0; bx < columns; ++bx) {
      Block given{};
      const std::size_t start = (by * columns + bx) * block_size;
      std::copy(coefficients.begin() + static_cast<std::ptrdiff_t>(start),
                coefficients.begin() + static_cast<std::ptrdiff_t>(start + block_size), given.begin());
      const Block samples = inverse(given);
      for (std::size_t m = 0; m < block; ++m) {
        for (std::size_t n = 0; n < block; ++n)
          canvas[(by * block + m) * width + bx * block + n] = samples[m * block + n];
      }
    }
  }
}

}  // namespace unround::dct
