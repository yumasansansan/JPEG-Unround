// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Tests of the model of one component (docs/math.md, 1.2 and 4.1): the intervals
// and the middles exact, as integers give them; the weights and mu to the last bit
// of their formulas; options out of their ranges refused; the proximal map within
// gamma_8 phi of the exact map (9.3), which exact rationals compute on inputs whose
// values are dyadic; the conjugate the supremum of s c - g(c) over the interval,
// within the rounding of both; and the clipping idempotent.

#include "unround/arrays.hpp"
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
#include <utility>
#include <vector>

namespace {

using unround::Block8x8;
using unround::BlockExtents;
using unround::Blocks;
using unround::BlocksArray;
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

BlocksArray<std::int16_t> levels_of(Numbers& numbers, std::size_t rows, std::size_t columns, std::int64_t low,
                                    std::int64_t high) {
  BlocksArray<std::int16_t> levels(unround::block_extents(rows, columns));
  for (std::int16_t& level : levels.entries()) level = static_cast<std::int16_t>(numbers.between(low, high));
  return levels;
}

// An entry of the blocks: coefficient (v, u) of block (by, bx).
struct At {
  std::size_t by;
  std::size_t bx;
  std::size_t v;
  std::size_t u;
};

// Every entry of blocks of a shape, block by block.
std::vector<At> every(const BlockExtents& shape) {
  std::vector<At> all;
  for (std::size_t by = 0; by < shape.extent(0); ++by) {
    for (std::size_t bx = 0; bx < shape.extent(1); ++bx) {
      for (std::size_t v = 0; v < 8u; ++v) {
        for (std::size_t u = 0; u < 8u; ++u) all.push_back(At{.by = by, .bx = bx, .v = v, .u = u});
      }
    }
  }
  return all;
}

At random_at(Numbers& numbers, const BlockExtents& shape) {
  const auto pick = [&numbers](std::size_t count) {
    return static_cast<std::size_t>(numbers.between(0, static_cast<std::int64_t>(count) - 1));
  };
  return At{.by = pick(shape.extent(0)), .bx = pick(shape.extent(1)), .v = pick(8), .u = pick(8)};
}

template <typename T>
T entry(Blocks<const T> blocks, const At& at) {
  return blocks[at.by, at.bx, at.v, at.u];
}

// The table's step of an entry's frequency.
double step_of(const std::array<std::uint16_t, 64>& table, const At& at) {
  return static_cast<double>(Block8x8<const std::uint16_t>(table.data())[at.v, at.u]);
}

bool dc(const At& at) { return at.v == 0u && at.u == 0u; }

void intervals(void) {
  Numbers numbers(11);
  const std::array<std::uint16_t, 64> table = table_of(numbers, 1, 255);
  const BlocksArray<std::int16_t> levels = levels_of(numbers, 2, 3, -2048, 2047);
  DataTerm data;
  data.mu = 0.5;
  const std::expected<Problem, unround::Error> problem = Problem::make(levels.view(), table, data);
  CHECK(problem.has_value());
  if (!problem) return;
  for (const At& at : every(levels.extents())) {
    const std::int64_t q = entry(levels.view(), at);
    const auto step = static_cast<std::int64_t>(step_of(table, at));
    const double shift = dc(at) ? 1024.0 : 0.0;
    // (q - 1/2) Q = (2q - 1) Q / 2, an integer over 2, exact in a double.
    const double lower = static_cast<double>((2 * q - 1) * step) / 2.0 + shift;
    const double upper = static_cast<double>((2 * q + 1) * step) / 2.0 + shift;
    CHECK(entry(problem->lower(), at) == lower && entry(problem->upper(), at) == upper);
    CHECK(entry(problem->centres(), at) == static_cast<double>(q * step) + shift);
    CHECK(entry(problem->levels(), at) == q);
  }
  CHECK(problem->rows() == 2u && problem->columns() == 3u && !problem->cost());
  // A slack widens every interval by that many steps on each side.
  data.slack = 0.5;
  const std::expected<Problem, unround::Error> slack = Problem::make(levels.view(), table, data);
  CHECK(slack.has_value());
  if (!slack) return;
  for (const At& at : every(levels.extents())) {
    const std::int64_t q = entry(levels.view(), at);
    const auto step = static_cast<std::int64_t>(step_of(table, at));
    const double shift = dc(at) ? 1024.0 : 0.0;
    CHECK(entry(slack->lower(), at) == static_cast<double>((q - 1) * step) + shift);
    CHECK(entry(slack->upper(), at) == static_cast<double>((q + 1) * step) + shift);
  }
}

void weights(void) {
  Numbers numbers(12);
  const std::array<std::uint16_t, 64> table = table_of(numbers, 1, 255);
  const BlocksArray<std::int16_t> levels = levels_of(numbers, 1, 1, -10, 10);
  const Block8x8<const std::uint16_t> steps(table.data());
  for (const double power : {2.0, 1.0, 0.7}) {
    DataTerm data;
    data.mu = 0.3;
    data.power = power;
    data.dc_weight = 3.0;
    const std::expected<Problem, unround::Error> problem = Problem::make(levels.view(), table, data);
    CHECK(problem.has_value());
    if (!problem) continue;
    for (std::size_t v = 0; v < 8u; ++v) {
      for (std::size_t u = 0; u < 8u; ++u) {
        const double step = static_cast<double>(steps[v, u]);
        const double q_power = power == 2.0 ? step * step : (power == 1.0 ? step : std::pow(step, power));
        double expected = 0.3 / q_power;
        if (v == 0u && u == 0u) expected *= 3.0;
        CHECK(std::bit_cast<std::uint64_t>(problem->weights()[v, u]) == std::bit_cast<std::uint64_t>(expected));
        CHECK(problem->steps()[v, u] == step);
      }
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
  const BlocksArray<std::int16_t> levels(unround::block_extents(1, 1));
  std::array<std::uint16_t, 64> table{};
  table.fill(1);
  const auto refused = [&](const DataTerm& data, const std::string& fragment) {
    const std::expected<Problem, unround::Error> problem = Problem::make(levels.view(), table, data);
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
  // A Laplace scale is for the MMSE centres; steps are at least 1. (The levels are
  // blocks by their type.)
  std::array<double, 64> scale{};
  scale.fill(1.0);
  const std::expected<Problem, unround::Error> midpoint = Problem::make(levels.view(), table, DataTerm{}, &scale);
  CHECK(!midpoint && midpoint.error().message.contains("Laplace scale"));
  table[5] = 0;
  const std::expected<Problem, unround::Error> zero = Problem::make(levels.view(), table, DataTerm{});
  CHECK(!zero && zero.error().message.contains("at least 1"));
}

void mmse_centres(void) {
  Numbers numbers(13);
  const std::array<std::uint16_t, 64> table = table_of(numbers, 1, 60);
  const BlocksArray<std::int16_t> levels = levels_of(numbers, 2, 4, -6, 6);
  DataTerm data;
  data.centres = Centres::mmse;
  const std::expected<Problem, unround::Error> problem = Problem::make(levels.view(), table, data);
  CHECK(problem.has_value());
  if (!problem) return;
  for (const At& at : every(levels.extents())) {
    const double q = static_cast<double>(entry(levels.view(), at));
    const double step = step_of(table, at);
    const double centre = entry(problem->centres(), at);
    if (dc(at)) {
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

// The price of leaving the file's own interval at an entry's frequency, and the ends
// of that interval.
double price_of(const unround::Cost& cost, const At& at) {
  return Block8x8<const double>(cost.costs.data())[at.v, at.u];
}
double own_low_of(const unround::Cost& cost, const At& at) { return entry(cost.inner_lower.view(), at); }
double own_high_of(const unround::Cost& cost, const At& at) { return entry(cost.inner_upper.view(), at); }

// The proximal map of one coefficient, in exact rationals, from the same doubles:
// z = (e + t c) / (1 + t) with t = step m; with a cost, moved back by step lambda /
// (1 + t) where it lies beyond the file's own interval, but not past its end; then
// clipped to the interval.
Rational exact_prox(const Problem& problem, const At& at, double e, double step) {
  using unround::exact::add;
  using unround::exact::multiply;
  using unround::exact::subtract;
  const Rational t = need(multiply(need(rational_of(step)), need(rational_of(problem.weights()[at.v, at.u]))));
  const Rational one_plus = need(add(Rational{.numerator = 1, .denominator = 1}, t));
  const Rational top =
      need(add(need(rational_of(e)), need(multiply(t, need(rational_of(entry(problem.centres(), at)))))));
  Rational z = need(divide(top, one_plus));
  if (const std::optional<unround::Cost>& cost = problem.cost()) {
    const Rational price = need(rational_of(price_of(*cost, at)));
    const Rational shrink = need(divide(need(multiply(need(rational_of(step)), price)), one_plus));
    const Rational own_low = need(rational_of(own_low_of(*cost, at)));
    const Rational own_high = need(rational_of(own_high_of(*cost, at)));
    if (less(own_high, z)) {
      const Rational back = need(subtract(z, shrink));
      z = less(back, own_high) ? own_high : back;
    } else if (less(z, own_low)) {
      const Rational back = need(add(z, shrink));
      z = less(own_low, back) ? own_low : back;
    }
  }
  const Rational low = need(rational_of(entry(problem.lower(), at)));
  const Rational high = need(rational_of(entry(problem.upper(), at)));
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
  const BlocksArray<std::int16_t> levels = levels_of(numbers, 2, 2, -20, 20);
  for (const bool costed : {false, true}) {
    DataTerm data;
    data.mu = static_cast<double>(numbers.between(1, 64)) / 16.0;
    data.dc_weight = 0.5;
    if (costed) {
      data.slack = 0.5;
      data.slack_cost = 3.0;
    }
    const std::expected<Problem, unround::Error> problem = Problem::make(levels.view(), table, data);
    CHECK(problem.has_value() && problem->cost().has_value() == costed);
    if (!problem) continue;
    for (int trial = 0; trial < 4000; ++trial) {
      const At at = random_at(numbers, levels.extents());
      const double step = static_cast<double>(numbers.between(1, 64)) / 64.0;
      // e near the interval and away from it.
      const double centre = entry(problem->centres(), at);
      const double e = centre + static_cast<double>(numbers.between(-1 << 14, 1 << 14)) / 256.0;
      const double zeta = problem->prox_at(at.by, at.bx, at.v, at.u, e, step);
      const Rational exact = exact_prox(*problem, at, e, step);
      const Rational error = need(unround::exact::subtract(need(rational_of(zeta)), exact));
      // phi = (|e| + t |c|) / (1 + t) + h, computed within gamma_6 of its value.
      const double t = step * problem->weights()[at.v, at.u];
      const std::optional<unround::Cost>& cost = problem->cost();
      const double h = cost ? step * price_of(*cost, at) / (1.0 + t) : 0.0;
      const double phi = ((std::abs(e) + t * std::abs(centre)) / (1.0 + t) + h) * (1.0 + gamma(6));
      CHECK(std::abs(unround::exact::nearest(error)) <= gamma(8) * phi);
      CHECK(entry(problem->lower(), at) <= zeta && zeta <= entry(problem->upper(), at));
    }
  }
}

// s c - g(c), the objective of the conjugate, and what bounds its rounding.
struct Objective {
  double value;
  double magnitude;
};

Objective objective(const Problem& problem, const At& at, double s, double c) {
  const double m = problem.weights()[at.v, at.u];
  const double away = c - entry(problem.centres(), at);
  double beyond = 0.0;
  double lambda = 0.0;
  if (const std::optional<unround::Cost>& cost = problem.cost()) {
    lambda = price_of(*cost, at);
    const double own_low = own_low_of(*cost, at);
    const double own_high = own_high_of(*cost, at);
    beyond = c > own_high ? c - own_high : (c < own_low ? own_low - c : 0.0);
  }
  const double square = 0.5 * m * (away * away);
  return Objective{.value = s * c - square - lambda * beyond, .magnitude = std::abs(s * c) + square + lambda * beyond};
}

void conjugate(void) {
  Numbers numbers(15);
  const std::array<std::uint16_t, 64> table = table_of(numbers, 1, 40);
  const BlocksArray<std::int16_t> levels = levels_of(numbers, 1, 2, -8, 8);
  for (const int variant : {0, 1, 2}) {
    DataTerm data;
    data.mu = variant == 2 ? 0.0 : 0.7;
    data.dc_weight = 0.0;  // DC without weight: its conjugate is the support function
    if (variant == 1) {
      data.slack = 1.0;
      data.slack_cost = 2.0;
    }
    const std::expected<Problem, unround::Error> problem = Problem::make(levels.view(), table, data);
    CHECK(problem.has_value());
    if (!problem) continue;
    for (int trial = 0; trial < 600; ++trial) {
      const At at = random_at(numbers, levels.extents());
      const double s = numbers.uniform(-3.0, 3.0);
      const double value = problem->conjugate_at(at.by, at.bx, at.v, at.u, s);
      // g*(s) is the supremum of s c - g(c) over the interval: no point of a grid over
      // it, with the ends of the file's own interval, lies above it by more than the
      // rounding of both.
      const double low = entry(problem->lower(), at);
      const double high = entry(problem->upper(), at);
      std::vector<double> points;
      for (int i = 0; i <= 200; ++i) points.push_back(low + (high - low) * static_cast<double>(i) / 200.0);
      const std::optional<unround::Cost>& cost = problem->cost();
      const double price = cost ? price_of(*cost, at) : 0.0;
      if (cost) {
        points.push_back(own_low_of(*cost, at));
        points.push_back(own_high_of(*cost, at));
      }
      double best = -HUGE_VAL;
      for (const double c : points) {
        const Objective found = objective(*problem, at, s, std::clamp(c, low, high));
        CHECK(found.value <= value + gamma(8) * (found.magnitude + std::abs(value)));
        best = std::max(best, found.value);
      }
      // And it is reached: a fine grid comes within the grid's spacing of it.
      const double spacing = (high - low) / 200.0;
      const double m = problem->weights()[at.v, at.u];
      const double slope = std::abs(s) + m * (high - low) + price;
      CHECK(value <= best + slope * spacing + gamma(8) * std::abs(value) + 1e-12);
      if (m == 0.0 && !problem->cost()) CHECK(value == std::max(s * low, s * high));
    }
  }
}

void clipping(void) {
  Numbers numbers(16);
  const std::array<std::uint16_t, 64> table = table_of(numbers, 1, 50);
  const BlocksArray<std::int16_t> levels = levels_of(numbers, 1, 3, -30, 30);
  const std::expected<Problem, unround::Error> problem = Problem::make(levels.view(), table, DataTerm{});
  CHECK(problem.has_value());
  if (!problem) return;
  const BlockExtents shape = levels.extents();
  BlocksArray<double> c(shape);
  for (const At& at : every(shape)) {
    c.view()[at.by, at.bx, at.v, at.u] = entry(problem->centres(), at) + numbers.uniform(-100.0, 100.0);
  }
  BlocksArray<double> once(shape);
  BlocksArray<double> twice(shape);
  problem->clip(std::as_const(c).view(), once.view());
  problem->clip(std::as_const(once).view(), twice.view());
  CHECK(once == twice);
  BlocksArray<double> excess(shape, -1.0);
  problem->excess(std::as_const(once).view(), excess.view());
  CHECK(std::ranges::all_of(excess.entries(), [](double value) { return value == 0.0; }));
  BlocksArray<double> before(shape, -1.0);
  problem->excess(std::as_const(c).view(), before.view());
  for (const At& at : every(shape)) {
    const double value = entry(std::as_const(c).view(), at);
    CHECK(entry(std::as_const(before).view(), at) ==
          std::max({entry(problem->lower(), at) - value, value - entry(problem->upper(), at), 0.0}));
  }
  // The value of G at the middles is 0, and the proximal map of a step of 0 is the
  // clipping, entry by entry and over all the blocks.
  CHECK(problem->value(problem->centres()) == 0.0);
  for (const At& at : every(shape)) {
    CHECK(problem->prox_at(at.by, at.bx, at.v, at.u, entry(std::as_const(c).view(), at), 0.0) ==
          entry(std::as_const(once).view(), at));
  }
  BlocksArray<double> mapped(shape);
  problem->prox(std::as_const(c).view(), 0.0, mapped.view());
  CHECK(mapped == once);
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
