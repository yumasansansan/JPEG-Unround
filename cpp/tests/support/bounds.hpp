// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The bounds of the DCT's rounding that the tests share (docs/math.md, 1.1 and 9.3):
// a transform of a block rounds each output within epsilon times the entry of
// |C| |X| |C|^T (forward) or |C|^T |Y| |C| (inverse), epsilon = e1 (2 + e1), e1 within
// gamma_6.
#ifndef UNROUND_TESTS_BOUNDS_HPP
#define UNROUND_TESTS_BOUNDS_HPP

#include "unround/arrays.hpp"
#include "unround/dct.hpp"

#include "check.hpp"

#include <cmath>
#include <cstddef>

namespace unround::test {

inline const double epsilon = gamma(6) * (2.0 + gamma(6));

// The bounds are sums of products of at most 64 terms, each with at most 3 roundings:
// computed within gamma_200 of their values, by which they are raised.
inline const double bound_rounding = 1.0 + gamma(200);

// |C| |X| |C|^T, entry (v, w), of an 8 x 8 block.
inline dct::Block forward_magnitudes(const dct::Block& samples) {
  const dct::Basis& c = dct::basis();
  const Block8x8<const double> x(samples.data());
  dct::Block result{};
  const Block8x8<double> out(result.data());
  for (std::size_t v = 0; v < 8u; ++v) {
    for (std::size_t w = 0; w < 8u; ++w) {
      double sum = 0.0;
      for (std::size_t m = 0; m < 8u; ++m) {
        for (std::size_t n = 0; n < 8u; ++n) sum += std::abs(c[v][m]) * std::abs(c[w][n]) * std::abs(x[m, n]);
      }
      out[v, w] = sum;
    }
  }
  return result;
}

// |C|^T |Y| |C|, entry (m, n).
inline dct::Block inverse_magnitudes(const dct::Block& coefficients) {
  const dct::Basis& c = dct::basis();
  const Block8x8<const double> y(coefficients.data());
  dct::Block result{};
  const Block8x8<double> out(result.data());
  for (std::size_t m = 0; m < 8u; ++m) {
    for (std::size_t n = 0; n < 8u; ++n) {
      double sum = 0.0;
      for (std::size_t v = 0; v < 8u; ++v) {
        for (std::size_t w = 0; w < 8u; ++w) sum += std::abs(c[v][m]) * std::abs(c[w][n]) * std::abs(y[v, w]);
      }
      out[m, n] = sum;
    }
  }
  return result;
}

// Block (by, bx) of a canvas, and of coefficients, as a block of 64.
inline dct::Block block_of(Grid<const double> canvas, std::size_t by, std::size_t bx) {
  dct::Block values{};
  const Block8x8<double> into(values.data());
  for (std::size_t m = 0; m < 8u; ++m) {
    for (std::size_t n = 0; n < 8u; ++n) into[m, n] = canvas[8u * by + m, 8u * bx + n];
  }
  return values;
}

inline dct::Block block_of(Blocks<const double> coefficients, std::size_t by, std::size_t bx) {
  dct::Block values{};
  const Block8x8<double> into(values.data());
  for (std::size_t v = 0; v < 8u; ++v) {
    for (std::size_t w = 0; w < 8u; ++w) into[v, w] = coefficients[by, bx, v, w];
  }
  return values;
}

}  // namespace unround::test

#endif  // UNROUND_TESTS_BOUNDS_HPP
