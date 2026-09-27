// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The Laplace model of docs/math.md, 2.

#include "unround/laplace.hpp"

#include "unround/arrays.hpp"
#include "unround/exact.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>
#include <span>

namespace unround::laplace {

namespace {

using exact::i128;
using exact::u128;

// The factorials (2k)!, k = 1, ..., 11: 22! is below 2^71.
i128 factorial(int n) noexcept {
  i128 product = 1;
  for (int k = 2; k <= n; ++k) product *= static_cast<i128>(k);
  return product;
}

// Eleven rationals, in an array of this file's own.
struct Eleven {
  exact::Rational values[11];
};

}  // namespace

std::span<const exact::Rational, 11> bernoulli(void) noexcept {
  static const Eleven numbers = {{
      exact::Rational{.numerator = 1, .denominator = 6},
      exact::Rational{.numerator = -1, .denominator = 30},
      exact::Rational{.numerator = 1, .denominator = 42},
      exact::Rational{.numerator = -1, .denominator = 30},
      exact::Rational{.numerator = 5, .denominator = 66},
      exact::Rational{.numerator = -691, .denominator = 2730},
      exact::Rational{.numerator = 7, .denominator = 6},
      exact::Rational{.numerator = -3617, .denominator = 510},
      exact::Rational{.numerator = 43867, .denominator = 798},
      exact::Rational{.numerator = -174611, .denominator = 330},
      exact::Rational{.numerator = 854513, .denominator = 138},
  }};
  return std::span<const exact::Rational, 11>(numbers.values);
}

std::span<const exact::Rational, 11> series_coefficients(void) noexcept {
  static const Eleven coefficients = [](void) {
    Eleven values{};
    for (std::size_t k = 0; k < 11u; ++k) {
      const exact::MaybeRational inverse = exact::rational(1, factorial(2 * static_cast<int>(k) + 2));
      const exact::MaybeRational value = exact::multiply(bernoulli()[k], *inverse);
      // Every numerator and denominator fits in 128 bits: the largest, 138 x 22!, is
      // below 2^78.
      if (!inverse || !value) std::abort();
      values.values[k] = *value;
    }
    return values;
  }();
  return std::span<const exact::Rational, 11>(coefficients.values);
}

const std::array<double, 11>& series(void) noexcept {
  static const std::array<double, 11> doubles = [](void) {
    std::array<double, 11> values{};
    for (std::size_t k = 0; k < values.size(); ++k) values[k] = exact::nearest(series_coefficients()[k]);
    return values;
  }();
  return doubles;
}

double scale(std::uint64_t zeros, std::uint64_t others, std::uint64_t sum, double step) noexcept {
  if (sum == 0u) return 0.0;
  const u128 n0 = zeros;
  const u128 n1 = others;
  const u128 s = sum;
  // A t^2 + n0 t - S = 0, and its discriminant D = n0^2 + 4 A S, exactly: the
  // counts of a component fit in 40 bits, and D in 82.
  const u128 a = n0 + s + 2u * n1;
  const u128 discriminant = n0 * n0 + 4u * a * s;
  const double root = std::sqrt(exact::nearest(discriminant));
  const double n0_double = exact::nearest(n0);
  // t* = 2S / (n0 + sqrt D), written so that nothing cancels.
  const double t = exact::nearest(2u * s) / (n0_double + root);
  double log_inverse = 0.0;
  if (t >= 0.5) {
    // Near 1, 1 - t* = 8 S (n0 + n1) / ((sqrt D + 2S - n0)(n0 + sqrt D)), every term
    // positive (t* >= 1/2 gives 2S >= n0), and log(1/t*) = -log1p(-(1 - t*)).
    const double top = exact::nearest(8u * s * (n0 + n1));
    const double first = root + exact::nearest(2u * s - n0);
    const double second = n0_double + root;
    const double complement = top / (first * second);
    log_inverse = -std::log1p(-complement);
  } else {
    log_inverse = -std::log(t);
  }
  return step / (2.0 * log_inverse);
}

std::array<double, 64> scales(Blocks<const std::int16_t> levels, const std::array<std::uint16_t, 64>& table) noexcept {
  // The counts of every frequency in natural order, block by block (DC's too, which
  // no scale reads).
  std::array<std::uint64_t, 64> zeros{};
  std::array<std::uint64_t, 64> others{};
  std::array<std::uint64_t, 64> sums{};
  for (std::size_t by = 0; by < levels.extent(0); ++by) {
    for (std::size_t bx = 0; bx < levels.extent(1); ++bx) {
      const std::span<const std::int16_t, 64> block = block_entries(levels, by, bx);
      for (std::size_t k = 0; k < 64u; ++k) {
        const std::int64_t level = block[k];
        const std::uint64_t zero = level == 0 ? 1u : 0u;
        const std::int64_t magnitude = level < 0 ? -level : level;
        zeros[k] += zero;
        others[k] += 1u - zero;
        // 2 |q| - 1 for a level other than 0, and 0 for 0.
        sums[k] += zero != 0u ? 0u : static_cast<std::uint64_t>(2 * magnitude - 1);
      }
    }
  }
  std::array<double, 64> result{};
  for (std::size_t k = 1; k < 64u; ++k) result[k] = scale(zeros[k], others[k], sums[k], static_cast<double>(table[k]));
  return result;
}

double shrinkage(double rho) noexcept {
  if (!(rho > 0.0)) return 0.0;
  if (std::isinf(rho)) return 0.5;
  if (rho < 1.0) {
    // rho (c_1 + rho^2 (c_2 + rho^2 (... + rho^2 c_11))), which converges for rho below
    // 2 pi: eleven terms leave less than 2e-19 below 1.
    const std::array<double, 11>& c = series();
    const double square = rho * rho;
    double sum = c[10];
    for (std::size_t k = 10; k-- > 0;) sum = sum * square + c[k];
    return sum * rho;
  }
  // 1/2 - 1/rho + e^-rho / (1 - e^-rho), every term at most 1.
  const double decay = std::exp(-rho);
  return (0.5 - 1.0 / rho) + decay / -std::expm1(-rho);
}

std::array<double, 64> shrinkages(const std::array<std::uint16_t, 64>& table,
                                  const std::array<double, 64>& scale) noexcept {
  std::array<double, 64> result{};
  for (std::size_t k = 0; k < 64u; ++k) {
    const double step = static_cast<double>(table[k]);
    const double rho = scale[k] > 0.0 ? step / scale[k] : std::numeric_limits<double>::infinity();
    result[k] = shrinkage(rho);
  }
  return result;
}

}  // namespace unround::laplace
