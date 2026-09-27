// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Tests of the exact arithmetic (docs/math.md, Arithmetic): every conversion of a
// 128-bit integer and of a rational to a double is the nearest double, ties to
// even, which the tests check in integers; long division and the greatest common
// divisor by their definitions; and the sums of rows in the order that the
// document gives, which the tests follow step by step.

#include "unround/exact.hpp"

#include "unround/arrays.hpp"

#include "support/check.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <vector>

namespace {

using unround::exact::i128;
using unround::exact::MaybeRational;
using unround::exact::nearest;
using unround::exact::Rational;
using unround::exact::u128;
using unround::test::Numbers;

constexpr u128 one = 1u;

// A double as m 2^k with an integer m in [2^52, 2^53).
struct Split {
  std::uint64_t mantissa;
  int exponent;
};

Split split(double value) {
  int exponent = 0;
  const double fraction = std::frexp(value, &exponent);
  return Split{.mantissa = static_cast<std::uint64_t>(std::ldexp(fraction, 53)), .exponent = exponent - 53};
}

u128 difference(u128 a, u128 b) noexcept { return a > b ? a - b : b - a; }

// Whether d is the double nearest to the integer v, ties to even: v lies within half
// the spacing of the doubles on its side of d, and at half of it, d's mantissa is
// even.
bool nearest_to(u128 v, double d) {
  if (v == 0u) return d == 0.0;
  const Split s = split(d);
  if (s.exponent < 0) {
    // Below 2^53 the double is the integer itself: its mantissa shifted down, with
    // nothing shifted out. (A conversion from double to a 128-bit integer calls the
    // compiler's runtime, which Windows does not have.)
    const auto down = static_cast<unsigned>(-s.exponent);
    return (s.mantissa & ((std::uint64_t{1} << down) - 1u)) == 0u && static_cast<u128>(s.mantissa >> down) == v;
  }
  const u128 value = static_cast<u128>(s.mantissa) << static_cast<unsigned>(s.exponent);
  // Below a power of two the doubles are twice as close.
  const bool power_of_two = s.mantissa == (std::uint64_t{1} << 52u);
  const int spacing = v < value && power_of_two ? s.exponent - 1 : s.exponent;
  const u128 twice = difference(v, value) << 1u;
  if (spacing < 0) return twice == 0u;
  const u128 whole = one << static_cast<unsigned>(spacing);
  if (twice < whole) return true;
  return twice == whole && (s.mantissa % 2u == 0u);
}

void integers(void) {
  CHECK(nearest(u128{0}) == 0.0);
  CHECK(nearest(u128{1}) == 1.0);
  const u128 big = one << 53u;
  CHECK(nearest(big + 1u) == 0x1p53);        // a tie, to the even 2^53
  CHECK(nearest(big + 3u) == 0x1p53 + 4.0);  // a tie, to the even 2^53 + 4
  CHECK(nearest((one << 54u) + 2u) == 0x1p54);
  CHECK(nearest((one << 54u) + 3u) == 0x1p54 + 4.0);
  CHECK(nearest(~u128{0}) == 0x1p128);
  CHECK(nearest(i128{-5}) == -5.0);
  const i128 least = static_cast<i128>(one << 127u);  // the least 128-bit value
  CHECK(nearest(least) == -0x1p127);
  Numbers numbers(1);
  for (int trial = 0; trial < 20000; ++trial) {
    // Values of every length, with runs of ones and zeros where rounding turns.
    const unsigned length = static_cast<unsigned>(numbers.between(1, 128));
    u128 v = (static_cast<u128>(numbers.next()) << 64u) | numbers.next();
    if (length < 128u) v &= (one << length) - one;
    if (trial % 3 == 0) v |= (one << (length > 60u ? length - 60u : 0u)) - one;
    const double d = nearest(v);
    CHECK(nearest_to(v, d));
    if (v <= (one << 126u)) CHECK(nearest(-static_cast<i128>(v)) == -d);
  }
}

// Whether d is the double nearest to p / q (p >= 0, 0 < q < 2^62, p < 2^62): with
// d = m 2^k, |p / q - m 2^k| is within half the spacing, and at half, m is even.
bool nearest_to(std::uint64_t p, std::uint64_t q, double d) {
  if (p == 0u) return d == 0.0;
  const Split s = split(d);
  const bool power_of_two = s.mantissa == (std::uint64_t{1} << 52u);
  // Scaled by q 2^-k (k < 0) or by 1 (k >= 0): 2 |p 2^-k - m q| against q, or
  // 2 |p - m q 2^k| against q 2^k.
  u128 left = 0u;
  u128 right = 0u;
  if (s.exponent < 0) {
    left = static_cast<u128>(p) << static_cast<unsigned>(-s.exponent);
    right = static_cast<u128>(s.mantissa) * q;
  } else {
    left = p;
    right = (static_cast<u128>(s.mantissa) * q) << static_cast<unsigned>(s.exponent);
  }
  const u128 twice = difference(left, right) << 1u;
  u128 whole = s.exponent < 0 ? static_cast<u128>(q) : static_cast<u128>(q) << static_cast<unsigned>(s.exponent);
  if (left < right && power_of_two) {
    // Half the spacing below a power of two: the comparison takes twice the error.
    if ((twice << 1u) < whole) return true;
    return (twice << 1u) == whole;
  }
  if (twice < whole) return true;
  return twice == whole && s.mantissa % 2u == 0u;
}

void rationals(void) {
  struct Known {
    i128 numerator;
    i128 denominator;
    double value;
  };
  // The doubles nearest to each, as Python's Fraction gives them.
  const std::array<Known, 10> known = {{
      {.numerator = 1, .denominator = 3, .value = 0x1.5555555555555p-2},
      {.numerator = 2, .denominator = 3, .value = 0x1.5555555555555p-1},
      {.numerator = 701, .denominator = 500, .value = 0x1.66e978d4fdf3bp+0},
      {.numerator = 443, .denominator = 250, .value = 0x1.c5a1cac083127p+0},
      {.numerator = 25251, .denominator = 73375, .value = 0x1.6065433a66b17p-2},
      {.numerator = 209599, .denominator = 293500, .value = 0x1.6da345743d962p-1},
      {.numerator = 1, .denominator = 10, .value = 0x1.999999999999ap-4},
      {.numerator = -7, .denominator = 9, .value = -0x1.8e38e38e38e39p-1},
      {.numerator = 1, .denominator = 6, .value = 0x1.5555555555555p-3},
      {.numerator = 854513, .denominator = 138, .value = 0x1.8301f89467e25p+12},
  }};
  for (const Known& case_ : known) {
    const MaybeRational value = unround::exact::rational(case_.numerator, case_.denominator);
    CHECK(value && nearest(*value) == case_.value);
  }
  Numbers numbers(2);
  for (int trial = 0; trial < 20000; ++trial) {
    const auto p = static_cast<std::uint64_t>(numbers.between(0, (std::int64_t{1} << 62) - 1) >>
                                              static_cast<unsigned>(numbers.between(0, 61)));
    auto q = static_cast<std::uint64_t>(numbers.between(1, (std::int64_t{1} << 62) - 1) >>
                                        static_cast<unsigned>(numbers.between(0, 61)));
    if (q == 0u) q = 1u;
    const MaybeRational value = unround::exact::rational(static_cast<i128>(p), static_cast<i128>(q));
    CHECK(value && nearest_to(p, q, nearest(*value)));
  }
  // Sums, differences and products in lowest terms.
  const auto of = [](i128 n, i128 d) { return *unround::exact::rational(n, d); };
  const auto is = [](const MaybeRational& value, const Rational& expected) { return value && *value == expected; };
  CHECK(is(unround::exact::rational(6, -4), of(-3, 2)));
  CHECK(!unround::exact::rational(1, 0));
  CHECK(is(unround::exact::add(of(1, 6), of(1, 3)), of(1, 2)));
  CHECK(is(unround::exact::subtract(of(1, 6), of(1, 3)), of(-1, 6)));
  CHECK(is(unround::exact::multiply(of(-691, 2730), of(1, 479001600)), of(-691, 1307674368000)));
  const i128 huge = static_cast<i128>(one << 100u);
  CHECK(!unround::exact::multiply(of(huge, 1), of(huge, 1)));
}

void division(void) {
  Numbers numbers(3);
  for (int trial = 0; trial < 5000; ++trial) {
    const u128 a = (static_cast<u128>(numbers.next()) << 64u) | numbers.next();
    u128 b = (static_cast<u128>(numbers.next()) << 64u) | numbers.next();
    b >>= static_cast<unsigned>(numbers.between(0, 127));
    if (b == 0u) b = 1u;
    const unround::exact::Division d = unround::exact::divide(a, b);
    CHECK(d.remainder < b);
    // a = q b + r, where q b cannot overflow: q b <= a.
    CHECK(d.quotient * b + d.remainder == a);
    const u128 g = unround::exact::gcd(a, b);
    CHECK(g != 0u && unround::exact::divide(a, g).remainder == 0u && unround::exact::divide(b, g).remainder == 0u);
    const u128 reduced_a = unround::exact::divide(a, g).quotient;
    const u128 reduced_b = unround::exact::divide(b, g).quotient;
    CHECK(unround::exact::gcd(reduced_a, reduced_b) == 1u);
  }
  CHECK(unround::exact::gcd(0u, 12u) == 12u);
  CHECK(unround::exact::gcd(18u, 12u) == 6u);
}

// The sum of a row as docs/math.md, Arithmetic, gives it, written out step by step.
double row_by_the_document(const std::vector<double>& terms) {
  std::array<double, 8> lanes{};
  for (std::size_t k = 0; k < terms.size(); ++k) lanes[k % 8u] = lanes[k % 8u] + terms[k];
  double total = lanes[0];
  for (std::size_t lane = 1; lane < 8u; ++lane) total = total + lanes[lane];
  return total;
}

void sums(void) {
  Numbers numbers(4);
  for (int trial = 0; trial < 200; ++trial) {
    const auto count = static_cast<std::size_t>(numbers.between(0, 300));
    std::vector<double> terms(count);
    // Terms of very different sizes, so that the order shows in the rounding.
    for (double& term : terms)
      term = std::ldexp(numbers.uniform(-1.0, 1.0), static_cast<int>(numbers.between(-40, 60)));
    const double sum = unround::exact::row_sum(terms);
    CHECK(std::bit_cast<std::uint64_t>(sum) == std::bit_cast<std::uint64_t>(row_by_the_document(terms)));
  }
  // Integers sum exactly in any order.
  std::vector<double> integers(1000);
  for (std::size_t k = 0; k < integers.size(); ++k) integers[k] = static_cast<double>(k) - 400.0;
  CHECK(unround::exact::row_sum(integers) == 99500.0);
  // Rows in blocks of 128, one after another, and the blocks in a tree: 300 rows make
  // blocks of 128, 128 and 44, added as (b0 + b1) + b2.
  std::vector<double> rows(300);
  for (double& row : rows) row = std::ldexp(numbers.uniform(-1.0, 1.0), static_cast<int>(numbers.between(-40, 60)));
  unround::exact::Sum sum;
  for (const double row : rows) sum.add(row);
  std::array<double, 3> blocks{};
  for (std::size_t k = 0; k < rows.size(); ++k) blocks[k / 128u] = blocks[k / 128u] + rows[k];
  const double expected = (blocks[0] + blocks[1]) + blocks[2];
  CHECK(std::bit_cast<std::uint64_t>(sum.total()) == std::bit_cast<std::uint64_t>(expected));
  CHECK(unround::exact::Sum{}.total() == 0.0);

  // A grid's sum: each row's sum as the document takes it, and those of the rows, from
  // the top, in blocks of 128 and a tree.
  const std::size_t height = 150;
  const std::size_t width = 21;
  std::vector<double> grid(height * width);
  for (double& term : grid) term = std::ldexp(numbers.uniform(-1.0, 1.0), static_cast<int>(numbers.between(-40, 60)));
  std::array<double, 2> halves{};
  for (std::size_t i = 0; i < height; ++i) {
    const std::vector<double> row(grid.begin() + static_cast<std::ptrdiff_t>(i * width),
                                  grid.begin() + static_cast<std::ptrdiff_t>((i + 1u) * width));
    halves[i / 128u] = halves[i / 128u] + row_by_the_document(row);
  }
  const double of_grid = unround::exact::sum_rows(unround::Grid<const double>(grid.data(), height, width));
  CHECK(std::bit_cast<std::uint64_t>(of_grid) == std::bit_cast<std::uint64_t>(halves[0] + halves[1]));
  CHECK(unround::exact::sum_rows(unround::Grid<const double>(grid.data(), height, 0)) == 0.0);
  CHECK(unround::exact::sum_rows(unround::Grid<const double>(grid.data(), 0, width)) == 0.0);
}

}  // namespace

int main(void) {
  // A test that throws fails with what it says, rather than ending the program without
  // a word.
  try {
    integers();
    rationals();
    division();
    sums();
    return unround::test::finish("exact");
  } catch (const std::exception& error) {
    (void)std::fputs(error.what(), stderr);
    (void)std::fputc('\n', stderr);
    return EXIT_FAILURE;
  }
}
