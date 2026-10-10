// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Tests of pictures of rounded samples (docs/math.md, 11): the ends of the intervals,
// each the double nearest to its rational (against the same rational in 128-bit
// integers where it fits), those of 8 bits without slack exact, a slack so small that
// it moves no end (it lies orders of magnitude below the distance of any end from a
// point where the rounding turns), and one so large that every interval is 0 to 255;
// the proximal map, at mu = 0 the clip to the last bit and above 0 within the bound of
// its roundings of the exact minimizer; the conjugate by Fenchel and Young; the
// solvers' records, gaps and stops, and their results within the intervals; and PNG
// files reconstructed, their alpha alone, and the settings they refuse.

#include "unround/samples.hpp"

#include "unround/arrays.hpp"
#include "unround/exact.hpp"
#include "unround/jpegio.hpp"
#include "unround/pdhg.hpp"
#include "unround/settings.hpp"

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
#include <initializer_list>
#include <print>
#include <span>
#include <string_view>
#include <vector>

namespace {

using unround::GridArray;
using unround::exact::i128;
using unround::test::gamma;
using unround::test::u;

bool same(double a, double b) noexcept { return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b); }

// The end 255 (2 v + side (1 + 2 slack)) / (2 (2^b - 1)) in rationals, clamped to 0 to
// 255, for a slack m 2^-K that keeps every integer within 128 bits.
double exact_end(std::uint32_t level, std::int32_t bits, double slack, int side) {
  const auto most = static_cast<i128>((std::uint64_t{1} << static_cast<std::uint32_t>(bits)) - 1u);
  int exponent = 0;
  const double fraction = std::frexp(slack, &exponent);
  // Through 64 bits: a conversion of a double to 128 bits calls the compiler's runtime.
  const auto m = static_cast<i128>(static_cast<std::int64_t>(std::ldexp(fraction, 53)));
  const int k = slack == 0.0 ? 0 : 53 - exponent;
  const i128 scale = i128{1} << k;
  const i128 numerator = 255 * ((2 * static_cast<i128>(level) + side) * scale + 2 * (slack == 0.0 ? 0 : m) * side);
  const i128 denominator = 2 * most * scale;
  if (numerator <= 0) return 0.0;
  if (numerator >= 255 * denominator) return 255.0;
  return unround::exact::nearest(unround::exact::Rational{.numerator = numerator, .denominator = denominator});
}

void intervals(void) {
  bool exact = true;
  bool eight = true;
  for (const std::int32_t bits : {1, 2, 4, 8, 16}) {
    const std::uint32_t most = (std::uint32_t{1} << static_cast<std::uint32_t>(bits)) - 1u;
    for (const std::uint32_t level : {0u, 1u, most / 3u, most / 2u, most - 1u, most}) {
      for (const double slack : {0.0, 0.25, 0.5, 1.5, 0.1, 0.01, 3.0}) {
        const unround::samples::Interval ends = unround::samples::interval(level, bits, slack);
        exact = exact && same(ends.lower, exact_end(level, bits, slack, -1)) &&
                same(ends.upper, exact_end(level, bits, slack, 1)) &&
                same(ends.centre, unround::exact::nearest(unround::exact::Rational{
                                      .numerator = 255 * static_cast<i128>(level), .denominator = most}));
      }
      if (bits == 8) {
        const unround::samples::Interval ends = unround::samples::interval(level, 8, 0.0);
        const double v = static_cast<double>(level);
        eight =
            eight && ends.lower == std::max(0.0, v - 0.5) && ends.upper == std::min(255.0, v + 0.5) && ends.centre == v;
      }
    }
  }
  CHECK(exact);
  CHECK(eight);
  // A slack below 2^-80 moves 255 (2 slack) / (2 (2^b - 1)) by less than 2^-72, where an
  // end that is not a double lies at least 2^-81 from the nearest point where the
  // rounding turns (its denominator divides 2^18 times an odd number, the points'
  // spacing is at least 2^-62): every end is that of no slack.
  bool tiny = true;
  for (const std::int32_t bits : {1, 8, 16}) {
    const std::uint32_t most = (std::uint32_t{1} << static_cast<std::uint32_t>(bits)) - 1u;
    for (std::uint32_t level = 0; level <= most; level += std::max<std::uint32_t>(1u, most / 97u)) {
      for (const double slack : {1e-25, 1e-300, 5e-324}) {
        const unround::samples::Interval a = unround::samples::interval(level, bits, slack);
        const unround::samples::Interval b = unround::samples::interval(level, bits, 0.0);
        tiny = tiny && same(a.lower, b.lower) && same(a.upper, b.upper);
      }
    }
  }
  CHECK(tiny);
  // A slack of 2^20 levels or more reaches past 0 and 255 whatever the level.
  const unround::samples::Interval wide = unround::samples::interval(100, 8, 2e6);
  CHECK(wide.lower == 0.0 && wide.upper == 255.0);
}

// A picture of levels of the bits, of the channels, random.
unround::MdArray<std::uint16_t, unround::PictureExtents> levels_of(unround::test::Numbers& numbers, std::size_t rows,
                                                                   std::size_t columns, std::size_t channels,
                                                                   std::int32_t bits) {
  unround::MdArray<std::uint16_t, unround::PictureExtents> levels(unround::PictureExtents{rows, columns, channels});
  const auto most = static_cast<std::int64_t>((std::uint64_t{1} << static_cast<std::uint32_t>(bits)) - 1u);
  // A smooth ramp with a little noise, rounded.
  const unround::Picture<std::uint16_t> view = levels.view();
  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t j = 0; j < columns; ++j) {
      for (std::size_t c = 0; c < channels; ++c) {
        const double ramp =
            (static_cast<double>(i + j + 3 * c) / static_cast<double>(rows + columns + 9)) * static_cast<double>(most);
        const std::int64_t near = static_cast<std::int64_t>(std::llround(ramp)) + numbers.between(-1, 1);
        const std::int64_t level = std::clamp(near, std::int64_t{0}, most);
        view[i, j, c] = static_cast<std::uint16_t>(level);
      }
    }
  }
  return levels;
}

unround::samples::Model model_of(const unround::MdArray<std::uint16_t, unround::PictureExtents>& levels,
                                 std::int32_t bits, std::size_t first, std::size_t count, double slack, double mu) {
  auto model = unround::samples::Model::make(levels.view(), bits, first, count, slack, mu);
  if (!model) {
    std::println(stderr, "{}", model.error().message);
    std::abort();
  }
  return std::move(*model);
}

// The proximal map: at mu = 0 the clip, to the last bit; above 0, within the bound of
// its five roundings (t, the product t c and its sum with v, 1 + t and its reciprocal,
// the last product; the clip is exact) of the exact minimizer, clipped, of dyadic
// inputs, computed in rationals.
void prox(void) {
  unround::test::Numbers numbers(61);
  const auto levels = levels_of(numbers, 9, 11, 1, 8);
  for (const double mu : {0.0, 0.5}) {
    const unround::samples::Model model = model_of(levels, 8, 0, 1, 0.25, mu);
    GridArray<double> v(model.extents());
    for (double& value : v.entries())
      value = static_cast<double>(numbers.between(-512, std::int64_t{512} * 255)) / 512.0;
    GridArray<double> out(model.extents());
    const double tau = 0.75;
    model.prox(0, v.view(), tau, out.view());
    const std::span<const double> vs = std::as_const(v).entries();
    const std::span<const double> outs = std::as_const(out).entries();
    const std::span<const double> lows(model.lower(0).data_handle(), model.lower(0).size());
    const std::span<const double> highs(model.upper(0).data_handle(), model.upper(0).size());
    const std::span<const double> centres(model.centres(0).data_handle(), model.centres(0).size());
    bool within = true;
    for (std::size_t k = 0; k < vs.size(); ++k) {
      if (mu == 0.0) {
        within = within && same(outs[k], std::clamp(vs[k], lows[k], highs[k]));
        continue;
      }
      // z* = (v + t c) / (1 + t), t = tau mu = 3/8: rationals with denominators 512, 4 and 8.
      const i128 v_512 = static_cast<i128>(static_cast<std::int64_t>(vs[k] * 512.0));
      const i128 c_4 = static_cast<i128>(static_cast<std::int64_t>(centres[k] * 4.0));
      // (v + 3/8 c) / (11/8) = (8 v + 3 c) / 11 = (8 v_512 / 512 + 3 c_4 / 4) / 11.
      const i128 numerator = 8 * v_512 + 3 * (128 * c_4);
      const double exact_value =
          unround::exact::nearest(unround::exact::Rational{.numerator = numerator, .denominator = i128{512} * 11});
      const double clipped = std::clamp(exact_value, lows[k], highs[k]);
      const double bound =
          gamma(5) * (std::abs(vs[k]) + 0.375 * std::abs(centres[k])) / 1.375 + u * std::abs(exact_value);
      within = within && std::abs(outs[k] - clipped) <= bound;
    }
    CHECK(within);
  }
}

// Fenchel and Young: G(x) + G*(xi) >= <xi, x> at x within the intervals, with equality
// at x the maximizer of <xi, x> - G(x); each side's sum rounds within gamma_N of the
// sum of its terms' magnitudes.
void conjugate(void) {
  unround::test::Numbers numbers(62);
  const auto levels = levels_of(numbers, 7, 9, 1, 8);
  for (const double mu : {0.0, 0.5}) {
    const unround::samples::Model model = model_of(levels, 8, 0, 1, 0.5, mu);
    GridArray<double> xi(model.extents());
    for (double& value : xi.entries()) value = numbers.uniform(-3.0, 3.0);
    const std::span<const double> lows(model.lower(0).data_handle(), model.lower(0).size());
    const std::span<const double> highs(model.upper(0).data_handle(), model.upper(0).size());
    const std::span<const double> centres(model.centres(0).data_handle(), model.centres(0).size());
    const std::span<const double> s = std::as_const(xi).entries();
    const double conjugate_value = model.conjugate(0, xi.view());
    // A point within the intervals, and the maximizer.
    GridArray<double> point(model.extents());
    GridArray<double> best(model.extents());
    for (std::size_t k = 0; k < s.size(); ++k) {
      point.entries()[k] = numbers.uniform(lows[k], highs[k]);
      best.entries()[k] =
          mu > 0.0 ? std::clamp(centres[k] + s[k] / mu, lows[k], highs[k]) : (s[k] > 0.0 ? highs[k] : lows[k]);
    }
    for (const bool at_best : {false, true}) {
      const GridArray<double>& x = at_best ? best : point;
      const double g = model.value(0, x.view());
      double inner = 0.0;
      double magnitude = 0.0;
      for (std::size_t k = 0; k < s.size(); ++k) {
        inner += s[k] * x.entries()[k];
        magnitude += std::abs(s[k] * x.entries()[k]) + std::abs(s[k]) * std::max(std::abs(lows[k]), highs[k]) +
                     mu * (highs[k] - lows[k]) * (highs[k] - lows[k]);
      }
      const double slack = gamma(static_cast<int>(4u * s.size())) * magnitude;
      CHECK(g + conjugate_value >= inner - slack);
      if (at_best) CHECK(std::abs(g + conjugate_value - inner) <= slack);
    }
  }
}

// TV and TGV on pictures of one and three channels: the gap of every record at least 0
// (orders of magnitude above the values' rounding at the stop), the method stopping at
// its tolerance, and every sample of the result within its interval, exactly.
void solves(void) {
  unround::test::Numbers numbers(63);
  for (const std::size_t channels : {std::size_t{1}, std::size_t{3}}) {
    const auto levels = levels_of(numbers, 12, 16, channels, 8);
    for (const double mu : {0.0, 0.05}) {
      const unround::samples::Model model = model_of(levels, 8, 0, channels, 0.0, mu);
      for (const unround::pdhg::Model method : {unround::pdhg::Model::tv, unround::pdhg::Model::tgv}) {
        unround::pdhg::Options options;
        options.tolerance = 1e-3;
        const auto result = unround::samples::solve(model, method, {}, options);
        if (!CHECK(result.has_value())) continue;
        CHECK(result->stop == unround::Stop::converged);
        CHECK(std::ranges::all_of(result->records,
                                  [](const unround::Record& record) { return record.primal - record.dual >= 0.0; }));
        bool within = true;
        for (std::size_t c = 0; c < channels; ++c) {
          const std::span<const double> x = result->canvas[c].entries();
          const std::span<const double> lows(model.lower(c).data_handle(), model.lower(c).size());
          const std::span<const double> highs(model.upper(c).data_handle(), model.upper(c).size());
          for (std::size_t k = 0; k < x.size(); ++k) within = within && x[k] >= lows[k] && x[k] <= highs[k];
        }
        CHECK(within);
      }
    }
  }
  // The defaults of PNG files: TGV's ratio 0.3, a tolerance of 1e-4, 10000 iterations.
  const auto plan = unround::pdhg::plan(unround::pdhg::Model::tgv, {}, {}, 1, unround::samples::defaults);
  CHECK(plan.has_value() && plan->tolerance == 1e-4 && plan->iterations == 10000u &&
        same(plan->steps.tau, std::sqrt((0.3 * 0.99) / ((17.0 + std::sqrt(33.0)) / 2.0))));
}

unround::Settings settings_of(std::initializer_list<std::string_view> arguments) {
  const std::vector<std::string_view> list(arguments);
  auto settings = unround::parse_settings(list);
  if (!settings) {
    std::println(stderr, "{}", settings.error().message);
    std::abort();
  }
  return *settings;
}

// PNG files of the C layer: grey, RGB and RGBA of 8 and 16 bits, their pictures
// reconstructed, the alpha alone, and every sample within its interval.
void files(void) {
  unround::test::Numbers numbers(64);
  for (const std::size_t channels : {std::size_t{1}, std::size_t{3}, std::size_t{4}}) {
    for (const std::int32_t bits : {8, 16}) {
      const auto levels = levels_of(numbers, 10, 13, channels, bits);
      std::expected<unround::jpegio::PngFile, unround::jpegio::Error> file;
      if (bits == 8) {
        unround::MdArray<std::uint8_t, unround::PictureExtents> bytes(levels.extents());
        std::ranges::transform(levels.entries(), bytes.entries().begin(),
                               [](std::uint16_t level) { return static_cast<std::uint8_t>(level); });
        file = unround::jpegio::write_png(std::as_const(bytes).view());
      } else {
        file = unround::jpegio::write_png(levels.view());
      }
      if (!CHECK(file.has_value())) continue;
      CHECK(unround::samples::is_png(file->bytes));
      const auto result = unround::samples::reconstruct(
          file->bytes, settings_of({"--iterations", "200", "--tolerance", "0", "--method", "tgv"}));
      if (!CHECK(result.has_value())) {
        std::println(stderr, "{}", result.error().message);
        continue;
      }
      CHECK(result->bits == bits && result->alpha == (channels == 4u));
      CHECK(result->picture.extent(0) == 10u && result->picture.extent(1) == 13u &&
            result->picture.extent(2) == channels);
      CHECK(result->colours.iterations == 200u && (channels != 4u || result->alphas.iterations == 200u));
      bool within = true;
      const unround::Picture<const double> picture = result->picture.view();
      for (std::size_t i = 0; i < 10u; ++i) {
        for (std::size_t j = 0; j < 13u; ++j) {
          for (std::size_t c = 0; c < channels; ++c) {
            const unround::samples::Interval ends = unround::samples::interval(levels.view()[i, j, c], bits, 0.0);
            within = within && picture[i, j, c] >= ends.lower && picture[i, j, c] <= ends.upper;
          }
        }
      }
      CHECK(within);
    }
  }
  // What a PNG file refuses: the data term's other options, a method other than TV or
  // TGV, and mu by the rule.
  unround::MdArray<std::uint8_t, unround::PictureExtents> grey(unround::PictureExtents{4, 5, 1});
  const auto file = unround::jpegio::write_png(std::as_const(grey).view());
  if (!CHECK(file.has_value())) return;
  for (const auto& arguments : {std::initializer_list<std::string_view>{"--centres", "midpoint"},
                                std::initializer_list<std::string_view>{"--method", "mmse"},
                                std::initializer_list<std::string_view>{"--mu", "rule"},
                                std::initializer_list<std::string_view>{"--flat-weight", "1"}}) {
    const auto refused = unround::samples::reconstruct(file->bytes, settings_of(arguments));
    CHECK(!refused && refused.error().code == unround::Errc::options);
  }
  const auto weighted = unround::samples::reconstruct(file->bytes, settings_of({"--mu", "0.5", "--slack", "0.25"}));
  CHECK(weighted.has_value());
}

int run(void) {
  intervals();
  prox();
  conjugate();
  solves();
  files();
  return unround::test::finish("samples");
}

}  // namespace

int main(void) {
  try {
    return run();
  } catch (const std::exception& error) {
    (void)std::fputs(error.what(), stderr);
    (void)std::fputc('\n', stderr);
    return EXIT_FAILURE;
  }
}
