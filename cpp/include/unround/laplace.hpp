// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The Laplace model of the AC coefficients (docs/math.md, 2): the scale of each
// frequency by maximum likelihood from the quantized coefficients, the shrinkage
// of a bin's mean towards 0, and the MMSE centres.
#ifndef UNROUND_LAPLACE_HPP
#define UNROUND_LAPLACE_HPP

#include "unround/arrays.hpp"
#include "unround/exact.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>

namespace unround::laplace {

// The maximum-likelihood scale beta of a frequency whose coefficients of step Q
// are `zeros` levels of 0 and `others` of other levels, S the sum of 2 |q| - 1
// over those (2.1). 0 where every coefficient is 0.
[[nodiscard]] double scale(std::uint64_t zeros, std::uint64_t others, std::uint64_t sum, double step) noexcept;

// The scale of every AC frequency of a component from the levels of its blocks,
// in natural order. DC follows no Laplace model, and its entry is 0.
[[nodiscard]] std::array<double, 64> scales(Blocks<const std::int16_t> levels,
                                            const std::array<std::uint16_t, 64>& table) noexcept;

// delta / Q of the mean of a bin, as a function of rho = Q / beta (2.2), in
// [0, 1/2): its Taylor series of eleven terms below 1, the closed form from 1 on;
// 1/2 for rho infinite (a scale of 0) and 0 for rho = 0.
[[nodiscard]] double shrinkage(double rho) noexcept;

// The coefficients of the series, B_2k / (2k)! for k = 1, ..., 11: exact rationals,
// and the doubles nearest to them. (The rationals are in an array of their own, not a
// std::array: exact.hpp says why.)
[[nodiscard]] std::span<const exact::Rational, 11> series_coefficients(void) noexcept;
[[nodiscard]] const std::array<double, 11>& series(void) noexcept;

// The Bernoulli numbers B_2k, k = 1, ..., 11, as exact rationals.
[[nodiscard]] std::span<const exact::Rational, 11> bernoulli(void) noexcept;

// delta / Q of every frequency of a component: rho = Q / beta of its step and its
// scale.
[[nodiscard]] std::array<double, 64> shrinkages(const std::array<std::uint16_t, 64>& table,
                                                const std::array<double, 64>& scale) noexcept;

// The MMSE centre of an AC coefficient of level q and step Q, given delta / Q:
// sign(q) (|q| - delta / Q) Q, and 0 for q = 0. |q| - delta / Q rounds to within
// [|q| - 1/2, |q|], whose ends are exact, and its product with Q to within the
// interval (2.2). Inline and without branches, so that the loops over a
// component's coefficients are vector code.
[[nodiscard]] inline double centre(std::int16_t level, double step, double shrink) noexcept {
  const double q = static_cast<double>(level);
  const double size = (std::abs(q) - shrink) * step;
  const double signed_size = q < 0.0 ? -size : size;
  return q == 0.0 ? 0.0 : signed_size;
}

}  // namespace unround::laplace

#endif  // UNROUND_LAPLACE_HPP
