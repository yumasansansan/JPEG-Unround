// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Exact arithmetic (docs/math.md, Arithmetic): what is rational is computed in
// integers and enters floating point once, rounded to the nearest double; and
// the sums of many doubles are taken in an order that depends only on the order
// of their terms.
//
// The integers of 128 bits are Clang's __int128. Their sums, products, shifts
// and comparisons compile to the processor's own instructions everywhere; their
// division and their conversions to and from double call the compiler's runtime,
// which programs on Windows do not have (tools/feature-probe), so this file does
// both itself. A value of 128 bits needs an alignment of 16, which the C++ library
// of Windows does not give the members of its own templates (it defines them under
// #pragma pack(8)): such values are never put into std::optional, std::array,
// std::pair and their kin, whose storage would then be misaligned, but into
// structures of this file's own.
#ifndef UNROUND_EXACT_HPP
#define UNROUND_EXACT_HPP

#include "unround/arrays.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace unround::exact {

__extension__ typedef __int128 i128;
__extension__ typedef unsigned __int128 u128;

// The double nearest to an integer, ties to even.
[[nodiscard]] double nearest(u128 value) noexcept;
[[nodiscard]] double nearest(i128 value) noexcept;

// A rational number in lowest terms, its denominator positive.
struct Rational {
  i128 numerator = 0;
  i128 denominator = 1;

  friend bool operator==(const Rational&, const Rational&) = default;
};

// A rational that is there or not: not where a result, or a step of reducing it,
// does not fit in 128 bits, or where a denominator is 0.
struct MaybeRational {
  Rational value{};
  bool present = false;

  [[nodiscard]] explicit operator bool(void) const noexcept { return present; }
  [[nodiscard]] const Rational& operator*(void) const noexcept { return value; }
  [[nodiscard]] const Rational* operator->(void) const noexcept { return &value; }
};

// The rational numerator / denominator in lowest terms.
[[nodiscard]] MaybeRational rational(i128 numerator, i128 denominator) noexcept;

// Sums, differences and products, in lowest terms.
[[nodiscard]] MaybeRational add(const Rational& a, const Rational& b) noexcept;
[[nodiscard]] MaybeRational subtract(const Rational& a, const Rational& b) noexcept;
[[nodiscard]] MaybeRational multiply(const Rational& a, const Rational& b) noexcept;

// The double nearest to a rational, ties to even, by long division.
[[nodiscard]] double nearest(const Rational& value) noexcept;

// The greatest common divisor, by halving and subtracting (no division).
[[nodiscard]] u128 gcd(u128 a, u128 b) noexcept;

// The quotient and the remainder of a / b, b > 0, by long division.
struct Division {
  u128 quotient;
  u128 remainder;
};
[[nodiscard]] Division divide(u128 a, u128 b) noexcept;

// The sum of the terms of a row in the order of docs/math.md, Arithmetic: term
// 8k + l goes to lane l, each lane adds its terms one after another, the last
// terms of the row as one more eight padded with 0; then the lanes are added one
// after another. A term goes through at most ceil(n / 8) + 7 additions.
[[nodiscard]] double row_sum(std::span<const double> terms) noexcept;

// The sum of rows' sums: added in blocks of 128, one after another within a
// block, and the blocks' sums in a tree, pairs of neighbours first. It depends
// only on the order in which the rows are added.
class Sum {
 public:
  void add(double row) noexcept;
  [[nodiscard]] double total(void) const;

 private:
  std::vector<double> blocks_;
  double block_ = 0.0;
  std::size_t count_ = 0;
};

// The sum of a grid of terms: the sum of each row as row_sum takes it, and those
// of the rows from the top as Sum adds them.
[[nodiscard]] double sum_rows(Grid<const double> terms);

}  // namespace unround::exact

#endif  // UNROUND_EXACT_HPP
