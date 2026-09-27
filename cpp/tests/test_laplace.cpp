// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Tests of the Laplace model (docs/math.md, 2 and 9.2): the scale within 16 units
// of the maximum of the likelihood, relatively, and the shrinkage within 16 units
// of 2^-53 of its exact value, against the references of 60 digits
// (conformance/references); a centre within (2 |q| + 17) u Q of the mean of its bin
// at the scale given; the Bernoulli numbers by their recurrence, in exact
// rationals, and the series' coefficients the doubles nearest to them.

#include "unround/exact.hpp"
#include "unround/laplace.hpp"

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
#include <filesystem>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

using unround::exact::i128;
using unround::exact::MaybeRational;
using unround::exact::Rational;
using unround::test::Exact;
using unround::test::u;

bool within(double value, const Exact& exact, double bound) {
  const double error = std::abs((value - exact.hi) - exact.lo);
  return error <= bound * (1.0 - 4.0 * u);
}

std::uint64_t whole(const std::string& text) { return static_cast<std::uint64_t>(std::stoull(text)); }

void scales(const std::filesystem::path& path) {
  int count = 0;
  for (const std::vector<std::string>& fields : unround::test::lines_of(path)) {
    const Exact exact = unround::test::exact_of(fields.at(4));
    const double scale = unround::laplace::scale(whole(fields.at(0)), whole(fields.at(1)), whole(fields.at(2)),
                                                 unround::test::number(fields.at(3)));
    CHECK(within(scale, exact, 16.0 * u * std::abs(exact.hi)));
    ++count;
  }
  CHECK(count == 10);
  // The counts of a component: every AC frequency its own. Frequency 1 is (v, u) =
  // (0, 1), and 2 is (0, 2).
  unround::BlocksArray<std::int16_t> levels(unround::block_extents(1, 3));
  levels.view()[0, 0, 0, 1] = 3;
  levels.view()[0, 1, 0, 1] = -2;
  levels.view()[0, 0, 0, 2] = 1;
  std::array<std::uint16_t, 64> table{};
  table.fill(10);
  const std::array<double, 64> found = unround::laplace::scales(std::as_const(levels).view(), table);
  CHECK(found[0] == 0.0);
  // Frequency 1: one zero, two others, S = 5 + 3.
  CHECK(std::bit_cast<std::uint64_t>(found[1]) ==
        std::bit_cast<std::uint64_t>(unround::laplace::scale(1u, 2u, 8u, 10.0)));
  CHECK(std::bit_cast<std::uint64_t>(found[2]) ==
        std::bit_cast<std::uint64_t>(unround::laplace::scale(2u, 1u, 1u, 10.0)));
  CHECK(found[3] == 0.0);
}

void shrinkages(const std::filesystem::path& path) {
  int count = 0;
  for (const std::vector<std::string>& fields : unround::test::lines_of(path)) {
    const double rho = unround::test::number(fields.at(0));
    CHECK(within(unround::laplace::shrinkage(rho), unround::test::exact_of(fields.at(1)), 16.0 * u));
    ++count;
  }
  CHECK(count > 3000);
  CHECK(unround::laplace::shrinkage(0.0) == 0.0);
  CHECK(unround::laplace::shrinkage(HUGE_VAL) == 0.5);
}

void centres(const std::filesystem::path& path) {
  int count = 0;
  for (const std::vector<std::string>& fields : unround::test::lines_of(path)) {
    const auto level = static_cast<std::int16_t>(std::stoi(fields.at(0)));
    const double step = unround::test::number(fields.at(1));
    const double scale = unround::test::number(fields.at(2));
    const double shrink = unround::laplace::shrinkage(step / scale);
    const double centre = unround::laplace::centre(level, step, shrink);
    const double bound = (2.0 * std::abs(static_cast<double>(level)) + 17.0) * u * step;
    CHECK(within(centre, unround::test::exact_of(fields.at(3)), bound));
    ++count;
  }
  CHECK(count > 20);
  CHECK(unround::laplace::centre(0, 12.0, 0.25) == 0.0);
}

// sum_{k=0}^{m} C(m + 1, k) B_k = 0 for m >= 1, with B_0 = 1, B_1 = -1/2 and the odd
// numbers beyond 0, in exact rationals.
void bernoulli(void) {
  const std::span<const Rational, 11> even = unround::laplace::bernoulli();
  std::vector<Rational> numbers(23);
  numbers[0] = Rational{.numerator = 1, .denominator = 1};
  numbers[1] = Rational{.numerator = -1, .denominator = 2};
  for (std::size_t k = 1; k <= 11u; ++k) numbers[2u * k] = even[k - 1u];
  for (std::size_t k = 1; 2u * k + 1u < numbers.size(); ++k)
    numbers[2u * k + 1u] = Rational{.numerator = 0, .denominator = 1};
  for (std::size_t m = 1; m <= 22u; ++m) {
    MaybeRational sum{.value = Rational{.numerator = 0, .denominator = 1}, .present = true};
    // C(m + 1, k), from k = 0, in 64 bits (a 128-bit division would call the
    // compiler's runtime, which Windows does not have).
    std::uint64_t binomial = 1;
    for (std::size_t k = 0; k <= m && sum; ++k) {
      const MaybeRational term =
          unround::exact::multiply(numbers[k], Rational{.numerator = static_cast<i128>(binomial), .denominator = 1});
      sum = term ? unround::exact::add(*sum, *term) : MaybeRational{};
      binomial = binomial * (m + 1u - k) / (k + 1u);
    }
    CHECK(sum && sum->numerator == 0);
  }
  // The series' coefficients B_2k / (2k)!, as Python's Fraction rounds them.
  const std::array<double, 11> expected = {
      0x1.5555555555555p-4,  -0x1.6c16c16c16c17p-10, 0x1.1566abc011567p-15, -0x1.bbd779334ef0bp-21,
      0x1.66a8f2bf70ebep-26, -0x1.22805d644267fp-31, 0x1.d6db2c4e09162p-37, -0x1.7da4e1f79955cp-42,
      0x1.355871d652e9ep-47, -0x1.f57d968caacf1p-53, 0x1.967e1f09c376fp-58,
  };
  for (std::size_t k = 0; k < expected.size(); ++k) {
    CHECK(std::bit_cast<std::uint64_t>(unround::laplace::series()[k]) == std::bit_cast<std::uint64_t>(expected[k]));
  }
  CHECK(unround::laplace::series_coefficients()[5] == (Rational{.numerator = -691, .denominator = 1307674368000}));
}

int run(std::span<char* const> arguments) {
  if (arguments.size() < 2u) return 2;
  const std::filesystem::path references = arguments[1];
  scales(references / "scale.txt");
  shrinkages(references / "shrinkage.txt");
  centres(references / "centres.txt");
  bernoulli();
  return unround::test::finish("laplace");
}

}  // namespace

int main(int argc, char** argv) {
  // A test that throws fails with what it says, rather than ending the program without
  // a word.
  try {
    return run(std::span<char* const>{argv, static_cast<std::size_t>(argc)});
  } catch (const std::exception& error) {
    (void)std::fputs(error.what(), stderr);
    (void)std::fputc('\n', stderr);
    return EXIT_FAILURE;
  }
}
