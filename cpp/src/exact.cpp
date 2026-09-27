// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Exact arithmetic in 128-bit integers, and sums in a fixed order (docs/math.md,
// Arithmetic). No value of 128 bits goes into a template of the C++ library
// (exact.hpp says why): the steps that may not fit return whether they did.

#include "unround/exact.hpp"

#include "unround/arrays.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <mdspan>
#include <span>
#include <utility>
#include <vector>

namespace unround::exact {

namespace {

constexpr u128 one = 1u;
constexpr u128 signed_max = (one << 127u) - one;
// The least value of 128 bits, -2^127. (std::numeric_limits has no specialization for
// __int128 in every C++ library, and where it has none, min() is 0.)
constexpr i128 signed_min = static_cast<i128>(one << 127u);

// How many bits a nonzero value needs.
int bit_length(u128 value) noexcept {
  const auto high = static_cast<std::uint64_t>(value >> 64u);
  if (high != 0u) return 128 - std::countl_zero(high);
  return 64 - std::countl_zero(static_cast<std::uint64_t>(value));
}

int trailing_zeros(u128 value) noexcept {
  const auto low = static_cast<std::uint64_t>(value);
  if (low != 0u) return std::countr_zero(low);
  return 64 + std::countr_zero(static_cast<std::uint64_t>(value >> 64u));
}

// The magnitude of a signed value, as an unsigned one (the least value too).
u128 magnitude(i128 value) noexcept {
  const auto bits = static_cast<u128>(value);
  return value < 0 ? u128{0} - bits : bits;
}

// The value of the given sign and magnitude, where it fits.
bool signed_of(bool negative, u128 value, i128& out) noexcept {
  if (value <= signed_max) {
    const auto positive = static_cast<i128>(value);
    out = negative ? -positive : positive;
    return true;
  }
  if (negative && value == signed_max + one) {
    out = static_cast<i128>(u128{0} - value);
    return true;
  }
  return false;
}

// The product of two magnitudes, where it fits in 128 bits.
bool multiply_magnitudes(u128 a, u128 b, u128& out) noexcept {
  const auto a_high = static_cast<std::uint64_t>(a >> 64u);
  const auto a_low = static_cast<std::uint64_t>(a);
  const auto b_high = static_cast<std::uint64_t>(b >> 64u);
  const auto b_low = static_cast<std::uint64_t>(b);
  if (a_high != 0u && b_high != 0u) return false;
  const u128 low = static_cast<u128>(a_low) * static_cast<u128>(b_low);
  // One of the two terms is 0, and the other a product of two 64-bit values.
  const u128 cross =
      static_cast<u128>(a_high) * static_cast<u128>(b_low) + static_cast<u128>(a_low) * static_cast<u128>(b_high);
  if ((cross >> 64u) != 0u) return false;
  const u128 sum = low + (cross << 64u);
  if (sum < low) return false;
  out = sum;
  return true;
}

bool multiply_signed(i128 a, i128 b, i128& out) noexcept {
  u128 product = 0u;
  return multiply_magnitudes(magnitude(a), magnitude(b), product) && signed_of((a < 0) != (b < 0), product, out);
}

bool add_signed(i128 a, i128 b, i128& out) noexcept {
  const u128 sum = static_cast<u128>(a) + static_cast<u128>(b);
  const auto result = static_cast<i128>(sum);
  // The sum overflows only where both terms have one sign and the result the other.
  if ((a < 0) == (b < 0) && (result < 0) != (a < 0)) return false;
  out = result;
  return true;
}

// The double nearest to a value, rounding to odd first where it needs more than 64
// bits: the conversion of 64 bits rounds to nearest, ties to even, as the processor
// does, and rounding to odd to 64 bits and then to nearest to 53 rounds as rounding to
// nearest to 53 at once would, 64 being at least 53 + 2.
double nearest_rounded_to_odd(u128 value) noexcept {
  const int length = bit_length(value);
  if (length <= 64) return static_cast<double>(static_cast<std::uint64_t>(value));
  const int shift = length - 64;
  const auto top = static_cast<std::uint64_t>(value >> static_cast<unsigned>(shift));
  const bool sticky = (value & ((one << static_cast<unsigned>(shift)) - one)) != 0u;
  return std::ldexp(static_cast<double>(top | (sticky ? std::uint64_t{1} : std::uint64_t{0})), shift);
}

}  // namespace

double nearest(u128 value) noexcept {
  if (value == 0u) return 0.0;
  return nearest_rounded_to_odd(value);
}

double nearest(i128 value) noexcept {
  const double size = nearest(magnitude(value));
  return value < 0 ? -size : size;
}

u128 gcd(u128 a, u128 b) noexcept {
  if (a == 0u) return b;
  if (b == 0u) return a;
  const int common = std::min(trailing_zeros(a), trailing_zeros(b));
  a >>= static_cast<unsigned>(trailing_zeros(a));
  while (b != 0u) {
    b >>= static_cast<unsigned>(trailing_zeros(b));
    if (a > b) std::swap(a, b);
    b -= a;
  }
  return a << static_cast<unsigned>(common);
}

Division divide(u128 a, u128 b) noexcept {
  if (b > a) return {.quotient = 0u, .remainder = a};
  const int shift = bit_length(a) - bit_length(b);
  u128 divisor = b << static_cast<unsigned>(shift);
  u128 quotient = 0u;
  u128 remainder = a;
  for (int bit = shift; bit >= 0; --bit) {
    quotient <<= 1u;
    if (remainder >= divisor) {
      remainder -= divisor;
      quotient |= one;
    }
    divisor >>= 1u;
  }
  return {.quotient = quotient, .remainder = remainder};
}

MaybeRational rational(i128 numerator, i128 denominator) noexcept {
  MaybeRational result;
  if (denominator == 0) return result;
  const bool negative = (numerator < 0) != (denominator < 0);
  u128 top = magnitude(numerator);
  u128 bottom = magnitude(denominator);
  const u128 common = gcd(top, bottom);
  top = divide(top, common).quotient;
  bottom = divide(bottom, common).quotient;
  if (bottom > signed_max || !signed_of(negative, top, result.value.numerator)) return result;
  result.value.denominator = static_cast<i128>(bottom);
  result.present = true;
  return result;
}

MaybeRational add(const Rational& a, const Rational& b) noexcept {
  // Over the least common multiple of the denominators, so that the terms stay as
  // small as the result allows.
  const u128 common = gcd(magnitude(a.denominator), magnitude(b.denominator));
  i128 a_scale = 0;
  i128 b_scale = 0;
  if (!signed_of(false, divide(magnitude(b.denominator), common).quotient, a_scale) ||
      !signed_of(false, divide(magnitude(a.denominator), common).quotient, b_scale)) {
    return {};
  }
  i128 left = 0;
  i128 right = 0;
  i128 bottom = 0;
  i128 top = 0;
  if (!multiply_signed(a.numerator, a_scale, left) || !multiply_signed(b.numerator, b_scale, right) ||
      !multiply_signed(a.denominator, a_scale, bottom) || !add_signed(left, right, top)) {
    return {};
  }
  return rational(top, bottom);
}

MaybeRational subtract(const Rational& a, const Rational& b) noexcept {
  if (b.numerator == signed_min) return {};
  return add(a, Rational{.numerator = -b.numerator, .denominator = b.denominator});
}

MaybeRational multiply(const Rational& a, const Rational& b) noexcept {
  // Reduce across first, so that the products stay as small as the result.
  const auto shrink = [](i128 value, u128 by, i128& out) noexcept {
    return signed_of(value < 0, divide(magnitude(value), by == 0u ? one : by).quotient, out);
  };
  const u128 across_one = gcd(magnitude(a.numerator), magnitude(b.denominator));
  const u128 across_two = gcd(magnitude(b.numerator), magnitude(a.denominator));
  i128 a_top = 0;
  i128 b_bottom = 0;
  i128 b_top = 0;
  i128 a_bottom = 0;
  i128 top = 0;
  i128 bottom = 0;
  if (!shrink(a.numerator, across_one, a_top) || !shrink(b.denominator, across_one, b_bottom) ||
      !shrink(b.numerator, across_two, b_top) || !shrink(a.denominator, across_two, a_bottom) ||
      !multiply_signed(a_top, b_top, top) || !multiply_signed(a_bottom, b_bottom, bottom)) {
    return {};
  }
  return rational(top, bottom);
}

double nearest(const Rational& value) noexcept {
  if (value.numerator == 0) return 0.0;
  const bool negative = value.numerator < 0;
  const u128 top = magnitude(value.numerator);
  const u128 bottom = magnitude(value.denominator);
  const Division whole = divide(top, bottom);
  double size = 0.0;
  if (whole.quotient >= (one << 54u)) {
    // 55 or more bits before the point: the rest only rounds, to odd.
    size = nearest(whole.quotient | (whole.remainder != 0u ? one : u128{0}));
  } else {
    // Take bits after the point until there are 56 of them in all, then the rest
    // rounds to odd. The remainder stays below the denominator, which is below
    // 2^127, so that doubling it cannot overflow.
    u128 quotient = whole.quotient;
    u128 remainder = whole.remainder;
    int fraction_bits = 0;
    while (quotient < (one << 55u)) {
      remainder <<= 1u;
      quotient <<= 1u;
      if (remainder >= bottom) {
        remainder -= bottom;
        quotient |= one;
      }
      ++fraction_bits;
    }
    size = std::ldexp(nearest(quotient | (remainder != 0u ? one : u128{0})), -fraction_bits);
  }
  return negative ? -size : size;
}

double row_sum(std::span<const double> terms) noexcept {
  // The terms eight to a row: lane l adds column l, row after row, eight lanes side
  // by side (a vector of the processor's width, or two).
  std::array<double, 8> lanes{};
  const std::size_t whole = terms.size() / 8u;
  const std::mdspan<const double, std::extents<std::size_t, std::dynamic_extent, 8>> eights(terms.data(), whole);
  for (std::size_t row = 0; row < whole; ++row) {
    for (std::size_t lane = 0; lane < 8u; ++lane) lanes[lane] += eights[row, lane];
  }
  const std::span<const double> rest = terms.subspan(8u * whole);
  for (std::size_t lane = 0; lane < rest.size(); ++lane) lanes[lane] += rest[lane];
  double total = lanes[0];
  for (std::size_t lane = 1; lane < 8u; ++lane) total += lanes[lane];
  return total;
}

void Sum::add(double row) noexcept {
  block_ += row;
  ++count_;
  if (count_ % 128u == 0u) {
    blocks_.push_back(block_);
    block_ = 0.0;
  }
}

double Sum::total(void) const {
  std::vector<double> level = blocks_;
  if (count_ % 128u != 0u) level.push_back(block_);
  if (level.empty()) return 0.0;
  while (level.size() > 1u) {
    std::vector<double> next;
    next.reserve((level.size() + 1u) / 2u);
    for (std::size_t index = 0; index + 1u < level.size(); index += 2u)
      next.push_back(level[index] + level[index + 1u]);
    if (level.size() % 2u != 0u) next.push_back(level.back());
    level = std::move(next);
  }
  return level.front();
}

double sum_rows(Grid<const double> terms) {
  const std::size_t width = terms.extent(1);
  if (width == 0u) return 0.0;
  Sum sum;
  for (std::size_t i = 0; i < terms.extent(0); ++i) sum.add(row_sum(std::span<const double>(&terms[i, 0], width)));
  return sum.total();
}

}  // namespace unround::exact
