// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Tests of the model of one component (docs/math.md, 1.2 and 4.1): the intervals
// and the middles exact, as integers give them; the weights and mu to the last bit
// of their formulas; options out of their ranges refused; the proximal map within
// gamma_8 phi of the exact map (9.3), which exact rationals compute on inputs whose
// values are dyadic; the conjugate the supremum of s c - g(c) over the interval,
// within the rounding of both; and the clipping idempotent.

#include "unround/exact.hpp"
#include "unround/model.hpp"

#include "support/check.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <expected>
#include <optional>
#include <string>
#include <vector>

namespace {

using unround::Centres;
using unround::DataTerm;
using unround::Problem;
using unround::exact::i128;
using unround::exact::MaybeRational;
using unround::exact::Rational;
using unround::test::gamma;
using unround::test::Numbers;

std::array<std::uint16_t, 64> table_of(Numbers& numbers, std::int64_t low, std::int64_t high) {
  std::array<std::uint16_t, 64> table{};
  for (std::uint16_t& step : table) step = static_cast<std::uint16_t>(numbers.between(low, high));
  return table;
}

std::vector<std::int16_t> levels_of(Numbers& numbers, std::size_t blocks, std::int64_t low, std::int64_t high) {
  std::vector<std::int16_t> levels(blocks * 64u);
  for (std::int16_t& level : levels) level = static_cast<std::int16_t>(numbers.between(low, high));
  return levels;
}

void intervals(void) {
  Numbers numbers(11);
  const std::array<std::uint16_t, 64> table = table_of(numbers, 1, 255);
  const std::vector<std::int16_t> levels = levels_of(numbers, 6, -2048, 2047);
  DataTerm data;
  data.mu = 0.5;
  const std::expected<Problem, unround::Error> problem = Problem::make(levels, 2, 3, table, data);
  CHECK(problem.has_value());
  if (!problem) return;
  for (std::size_t index = 0; index < levels.size(); ++index) {
    const std::size_t k = index % 64u;
    const std::int64_t q = levels[index];
    const std::int64_t step = table[k];
    const double shift = k == 0u ? 1024.0 : 0.0;
    // (q - 1/2) Q = (2q - 1) Q / 2, an integer over 2, exact in a double.
    const double lower = static_cast<double>((2 * q - 1) * step) / 2.0 + shift;
    const double upper = static_cast<double>((2 * q + 1) * step) / 2.0 + shift;
    CHECK(problem->lower()[index] == lower && problem->upper()[index] == upper);
    CHECK(problem->centres()[index] == static_cast<double>(q * step) + shift);
  }
  CHECK(problem->rows() == 2u && problem->columns() == 3u && !problem->cost());
  // A slack widens every interval by that many steps on each side.
  data.slack = 0.5;
  const std::expected<Problem, unround::Error> slack = Problem::make(levels, 2, 3, table, data);
  CHECK(slack.has_value());
  if (!slack) return;
  for (std::size_t index = 0; index < levels.size(); ++index) {
    const std::int64_t q = levels[index];
    const std::int64_t step = table[index % 64u];
    const double shift = index % 64u == 0u ? 1024.0 : 0.0;
    CHECK(slack->lower()[index] == static_cast<double>((q - 1) * step) + shift);
    CHECK(slack->upper()[index] == static_cast<double>((q + 1) * step) + shift);
  }
}

void weights(void) {
  Numbers numbers(12);
  const std::array<std::uint16_t, 64> table = table_of(numbers, 1, 255);
  const std::vector<std::int16_t> levels = levels_of(numbers, 1, -10, 10);
  for (const double power : {2.0, 1.0, 0.7}) {
    DataTerm data;
    data.mu = 0.3;
    data.power = power;
    data.dc_weight = 3.0;
    const std::expected<Problem, unround::Error> problem = Problem::make(levels, 1, 1, table, data);
    CHECK(problem.has_value());
    if (!problem) continue;
    for (std::size_t k = 0; k < 64u; ++k) {
      const double step = static_cast<double>(table[k]);
      const double q_power = power == 2.0 ? step * step : (power == 1.0 ? step : std::pow(step, power));
      double expected = 0.3 / q_power;
      if (k == 0u) expected *= 3.0;
      CHECK(std::bit_cast<std::uint64_t>(problem->weights()[k]) == std::bit_cast<std::uint64_t>(expected));
    }
  }
  // mu by the rule of the mean step: exact for the power 1, the platform's pow else;
  // in the chroma, the scale times its factor first.
  std::uint32_t sum = 0;
  for (const std::uint16_t step : table) sum += step;
  const double mean = static_cast<double>(sum) / 64.0;
  DataTerm rule;
  CHECK(!rule.mu && rule.mu_scale == 9.0 && rule.mu_power == 0.9 && rule.mu_chroma == 0.3);
  CHECK(rule.dc_weight == 1.0 && rule.centres == Centres::midpoint && rule.power == 2.0);
  CHECK(unround::rule_mu(rule, table) == 9.0 * std::pow(mean, 0.9));
  CHECK(unround::rule_mu(rule.of_chroma(), table) == (9.0 * 0.3) * std::pow(mean, 0.9));
  rule.mu_power = 1.0;
  rule.mu_scale = 1e-3;
  CHECK(unround::rule_mu(rule, table) == 1e-3 * mean);
  DataTerm given;
  given.mu = 2.5;
  CHECK(unround::rule_mu(given.of_chroma(), table) == 2.5);
}

void refusals(void) {
  const std::vector<std::int16_t> levels(64u, 0);
  std::array<std::uint16_t, 64> table{};
  table.fill(1);
  const auto refused = [&](const DataTerm& data, const std::string& fragment) {
    const std::expected<Problem, unround::Error> problem = Problem::make(levels, 1, 1, table, data);
    return !problem && problem.error().code == unround::Errc::options && problem.error().message.contains(fragment);
  };
  DataTerm data;
  data.mu = -1.0;
  CHECK(refused(data, "at least 0"));
  data = DataTerm{};
  data.slack = std::nan("");
  CHECK(refused(data, "at least 0"));
  data = DataTerm{};
  data.dc_weight = -1.0;
  CHECK(refused(data, "at least 0"));
  data = DataTerm{};
  data.power = -1.0;
  CHECK(refused(data, "power"));
  data = DataTerm{};
  data.slack_cost = HUGE_VAL;
  CHECK(refused(data, "cost of the slack"));
  data = DataTerm{};
  data.mu_scale = -1.0;
  CHECK(refused(data, "scale of mu"));
  data = DataTerm{};
  data.mu_chroma = std::nan("");
  CHECK(refused(data, "chroma"));
  data = DataTerm{};
  data.mu_power = HUGE_VAL;
  CHECK(refused(data, "power"));
  // A Laplace scale is for the MMSE centres; levels are whole blocks; steps at least 1.
  std::array<double, 64> scale{};
  scale.fill(1.0);
  const std::expected<Problem, unround::Error> midpoint = Problem::make(levels, 1, 1, table, DataTerm{}, &scale);
  CHECK(!midpoint && midpoint.error().message.contains("Laplace scale"));
  CHECK(!Problem::make(levels, 1, 2, table, DataTerm{}));
  table[5] = 0;
  const std::expected<Problem, unround::Error> zero = Problem::make(levels, 1, 1, table, DataTerm{});
  CHECK(!zero && zero.error().message.contains("at least 1"));
}

void mmse_centres(void) {
  Numbers numbers(13);
  const std::array<std::uint16_t, 64> table = table_of(numbers, 1, 60);
  const std::vector<std::int16_t> levels = levels_of(numbers, 8, -6, 6);
  DataTerm data;
  data.centres = Centres::mmse;
  const std::expected<Problem, unround::Error> problem = Problem::make(levels, 2, 4, table, data);
  CHECK(problem.has_value());
  if (!problem) return;
  for (std::size_t index = 0; index < levels.size(); ++index) {
    const std::size_t k = index % 64u;
    const double q = static_cast<double>(levels[index]);
    const double step = static_cast<double>(table[k]);
    const double centre = problem->centres()[index];
    if (k == 0u) {
      CHECK(centre == q * step + 1024.0);
    } else if (q == 0.0) {
      CHECK(centre == 0.0);
    } else {
      // In the half of the interval towards 0 (2.2), in floating point too.
      CHECK((std::abs(q) - 0.5) * step <= std::abs(centre) && std::abs(centre) <= std::abs(q) * step);
      CHECK((centre < 0.0) == (q < 0.0));
    }
  }
}

// The exact value of a double whose exponent is not too small, as a rational.
MaybeRational rational_of(double value) {
  if (value == 0.0) return MaybeRational{.value = Rational{.numerator = 0, .denominator = 1}, .present = true};
  int exponent = 0;
  const double fraction = std::frexp(value, &exponent);
  const auto mantissa = static_cast<std::int64_t>(std::ldexp(fraction, 53));
  const int power = exponent - 53;
  if (power >= 0) {
    if (power > 60) return {};
    return unround::exact::rational(static_cast<i128>(mantissa) * (i128{1} << static_cast<unsigned>(power)), 1);
  }
  if (power < -120) return {};
  return unround::exact::rational(mantissa, i128{1} << static_cast<unsigned>(-power));
}

bool less(const Rational& a, const Rational& b) {
  const MaybeRational d = unround::exact::subtract(a, b);
  return d && d->numerator < 0;
}

MaybeRational divide(const Rational& a, const Rational& b) {
  const MaybeRational inverse = unround::exact::rational(b.denominator, b.numerator);
  if (!inverse) return {};
  return unround::exact::multiply(a, *inverse);
}

// A rational that has to be there: where an exact step does not fit in 128 bits, the
// check fails and 0 stands in.
Rational need(const MaybeRational& value) {
  CHECK(static_cast<bool>(value));
  return value ? *value : Rational{.numerator = 0, .denominator = 1};
}

// The proximal map of one coefficient, in exact rationals, from the same doubles:
// z = (e + t c) / (1 + t) with t = step m; with a cost, moved back by step lambda /
// (1 + t) where it lies beyond the file's own interval, but not past its end; then
// clipped to the interval.
Rational exact_prox(const Problem& problem, std::size_t k, double e, double step) {
  using unround::exact::add;
  using unround::exact::multiply;
  using unround::exact::subtract;
  const std::size_t frequency = k % 64u;
  const Rational t = need(multiply(need(rational_of(step)), need(rational_of(problem.weights()[frequency]))));
  const Rational one_plus = need(add(Rational{.numerator = 1, .denominator = 1}, t));
  const Rational top = need(add(need(rational_of(e)), need(multiply(t, need(rational_of(problem.centres()[k]))))));
  Rational z = need(divide(top, one_plus));
  if (problem.cost()) {
    const Rational price = need(rational_of(problem.cost()->costs[frequency]));
    const Rational shrink = need(divide(need(multiply(need(rational_of(step)), price)), one_plus));
    const Rational own_low = need(rational_of(problem.cost()->inner_lower[k]));
    const Rational own_high = need(rational_of(problem.cost()->inner_upper[k]));
    if (less(own_high, z)) {
      const Rational back = need(subtract(z, shrink));
      z = less(back, own_high) ? own_high : back;
    } else if (less(z, own_low)) {
      const Rational back = need(add(z, shrink));
      z = less(own_low, back) ? own_low : back;
    }
  }
  const Rational low = need(rational_of(problem.lower()[k]));
  const Rational high = need(rational_of(problem.upper()[k]));
  if (less(z, low)) return low;
  if (less(high, z)) return high;
  return z;
}

void proximal_map(void) {
  Numbers numbers(14);
  // Steps that are powers of two keep the weights dyadic, and every value of the exact
  // map within 128-bit rationals.
  std::array<std::uint16_t, 64> table{};
  for (std::uint16_t& step : table) step = static_cast<std::uint16_t>(1 << numbers.between(0, 4));
  const std::vector<std::int16_t> levels = levels_of(numbers, 4, -20, 20);
  for (const bool costed : {false, true}) {
    DataTerm data;
    data.mu = static_cast<double>(numbers.between(1, 64)) / 16.0;
    data.dc_weight = 0.5;
    if (costed) {
      data.slack = 0.5;
      data.slack_cost = 3.0;
    }
    const std::expected<Problem, unround::Error> problem = Problem::make(levels, 2, 2, table, data);
    CHECK(problem.has_value() && problem->cost().has_value() == costed);
    if (!problem) continue;
    for (int trial = 0; trial < 4000; ++trial) {
      const auto k = static_cast<std::size_t>(numbers.between(0, static_cast<std::int64_t>(levels.size()) - 1));
      const double step = static_cast<double>(numbers.between(1, 64)) / 64.0;
      // e near the interval and away from it.
      const double centre = problem->centres()[k];
      const double e = centre + static_cast<double>(numbers.between(-1 << 14, 1 << 14)) / 256.0;
      const double zeta = problem->prox_one(k, e, step);
      const Rational exact = exact_prox(*problem, k, e, step);
      const Rational error = need(unround::exact::subtract(need(rational_of(zeta)), exact));
      // phi = (|e| + t |c|) / (1 + t) + h, computed within gamma_6 of its value.
      const double t = step * problem->weights()[k % 64u];
      const double h = problem->cost() ? step * problem->cost()->costs[k % 64u] / (1.0 + t) : 0.0;
      const double phi = ((std::abs(e) + t * std::abs(centre)) / (1.0 + t) + h) * (1.0 + gamma(6));
      CHECK(std::abs(unround::exact::nearest(error)) <= gamma(8) * phi);
      CHECK(problem->lower()[k] <= zeta && zeta <= problem->upper()[k]);
    }
  }
}

// s c - g(c), the objective of the conjugate, and what bounds its rounding.
struct Objective {
  double value;
  double magnitude;
};

Objective objective(const Problem& problem, std::size_t k, double s, double c) {
  const double m = problem.weights()[k % 64u];
  const double away = c - problem.centres()[k];
  double beyond = 0.0;
  double lambda = 0.0;
  if (problem.cost()) {
    lambda = problem.cost()->costs[k % 64u];
    const double own_low = problem.cost()->inner_lower[k];
    const double own_high = problem.cost()->inner_upper[k];
    beyond = c > own_high ? c - own_high : (c < own_low ? own_low - c : 0.0);
  }
  const double square = 0.5 * m * (away * away);
  return Objective{.value = s * c - square - lambda * beyond, .magnitude = std::abs(s * c) + square + lambda * beyond};
}

void conjugate(void) {
  Numbers numbers(15);
  const std::array<std::uint16_t, 64> table = table_of(numbers, 1, 40);
  const std::vector<std::int16_t> levels = levels_of(numbers, 2, -8, 8);
  for (const int variant : {0, 1, 2}) {
    DataTerm data;
    data.mu = variant == 2 ? 0.0 : 0.7;
    data.dc_weight = 0.0;  // DC without weight: its conjugate is the support function
    if (variant == 1) {
      data.slack = 1.0;
      data.slack_cost = 2.0;
    }
    const std::expected<Problem, unround::Error> problem = Problem::make(levels, 1, 2, table, data);
    CHECK(problem.has_value());
    if (!problem) continue;
    for (int trial = 0; trial < 600; ++trial) {
      const auto k = static_cast<std::size_t>(numbers.between(0, static_cast<std::int64_t>(levels.size()) - 1));
      const double s = numbers.uniform(-3.0, 3.0);
      const double value = problem->conjugate_one(k, s);
      // g*(s) is the supremum of s c - g(c) over the interval: no point of a grid over
      // it, with the ends of the file's own interval, lies above it by more than the
      // rounding of both.
      const double low = problem->lower()[k];
      const double high = problem->upper()[k];
      std::vector<double> points;
      for (int i = 0; i <= 200; ++i) points.push_back(low + (high - low) * static_cast<double>(i) / 200.0);
      if (problem->cost()) {
        points.push_back(problem->cost()->inner_lower[k]);
        points.push_back(problem->cost()->inner_upper[k]);
      }
      double best = -HUGE_VAL;
      for (const double c : points) {
        const Objective at = objective(*problem, k, s, std::clamp(c, low, high));
        CHECK(at.value <= value + gamma(8) * (at.magnitude + std::abs(value)));
        best = std::max(best, at.value);
      }
      // And it is reached: a fine grid comes within the grid's spacing of it.
      const double spacing = (high - low) / 200.0;
      const double slope = std::abs(s) + problem->weights()[k % 64u] * (high - low) +
                           (problem->cost() ? problem->cost()->costs[k % 64u] : 0.0);
      CHECK(value <= best + slope * spacing + gamma(8) * std::abs(value) + 1e-12);
      if (problem->weights()[k % 64u] == 0.0 && !problem->cost()) CHECK(value == std::max(s * low, s * high));
    }
  }
}

void clipping(void) {
  Numbers numbers(16);
  const std::array<std::uint16_t, 64> table = table_of(numbers, 1, 50);
  const std::vector<std::int16_t> levels = levels_of(numbers, 3, -30, 30);
  const std::expected<Problem, unround::Error> problem = Problem::make(levels, 1, 3, table, DataTerm{});
  CHECK(problem.has_value());
  if (!problem) return;
  std::vector<double> c(levels.size());
  for (std::size_t k = 0; k < c.size(); ++k) c[k] = problem->centres()[k] + numbers.uniform(-100.0, 100.0);
  std::vector<double> once(c.size());
  std::vector<double> twice(c.size());
  problem->clip(c, once);
  problem->clip(once, twice);
  CHECK(std::ranges::equal(once, twice));
  const std::vector<double> excess = problem->excess(once);
  CHECK(std::ranges::all_of(excess, [](double value) { return value == 0.0; }));
  const std::vector<double> before = problem->excess(c);
  for (std::size_t k = 0; k < c.size(); ++k) {
    CHECK(before[k] == std::max({problem->lower()[k] - c[k], c[k] - problem->upper()[k], 0.0}));
  }
  // The value of G at the middles is 0, and the proximal map of a step of 0 is the
  // clipping.
  CHECK(problem->value(problem->centres()) == 0.0);
  for (std::size_t k = 0; k < c.size(); ++k) CHECK(problem->prox_one(k, c[k], 0.0) == once[k]);
}

}  // namespace

int main(void) {
  // A test that throws fails with what it says, rather than ending the program without
  // a word.
  try {
    intervals();
    weights();
    refusals();
    mmse_centres();
    proximal_map();
    conjugate();
    clipping();
    return unround::test::finish("model");
  } catch (const std::exception& error) {
    (void)std::fputs(error.what(), stderr);
    (void)std::fputc('\n', stderr);
    return EXIT_FAILURE;
  }
}
