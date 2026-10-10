// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Tests of the conversions of JFIF (docs/math.md, 8): the constants are the doubles
// nearest to their rationals; a grey converts exactly; each output of integer samples
// lies within the bound of its rounding of its exact rational value, in both
// directions; and YCbCr converted to RGB and back lies within the bound of both
// conversions of where it started, the exact conversions being each other's inverses.
//
// The bounds are those of the standard model: a value computed with at most k
// roundings (a constant rounded to its double counting as one) lies within gamma_k of
// the sum of the magnitudes of its terms. An exact value D x = N, with N an integer and
// D the denominator of the rationals, is compared as D x^ - N, whose product the fused
// multiply-add gives with its error exactly.

#include "unround/colour.hpp"

#include "unround/exact.hpp"

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

namespace colour = unround::colour;
using unround::test::gamma;
using unround::test::u;

bool same(double a, double b) noexcept { return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b); }

double nearest(std::int64_t numerator, std::int64_t denominator) {
  return unround::exact::nearest(unround::exact::Rational{.numerator = numerator, .denominator = denominator});
}

void constants(void) {
  CHECK(same(colour::red_from_cr, nearest(701, 500)));
  CHECK(same(colour::blue_from_cb, nearest(443, 250)));
  CHECK(same(colour::green_from_cb, nearest(25251, 73375)));
  CHECK(same(colour::green_from_cr, nearest(209599, 293500)));
  CHECK(same(colour::luma_red, nearest(299, 1000)));
  CHECK(same(colour::luma_green, nearest(587, 1000)));
  CHECK(same(colour::luma_blue, nearest(114, 1000)));
  CHECK(same(colour::cb_from_blue, nearest(250, 443)));
  CHECK(same(colour::cr_from_red, nearest(500, 701)));
}

// Whether a computed value lies within `bound` of the exact numerator / denominator: the
// product denominator x^ and its error, exact by the fused multiply-add, less the
// numerator, within u of themselves; the comparison asks for a hair less than the
// bound for that.
bool within(double value, double denominator, double numerator, double bound) {
  const double product = denominator * value;
  const double error = std::fma(denominator, value, -product);
  const double difference = (product - numerator) + error;
  return std::abs(difference) <= denominator * bound * (1.0 - 4.0 * u);
}

// Samples to convert: integers of 0 to 255, every one of a few and random others.
std::vector<std::array<double, 3>> samples_of(void) {
  constexpr std::array<double, 10> levels = {0.0, 1.0, 17.0, 64.0, 127.0, 128.0, 129.0, 200.0, 254.0, 255.0};
  std::vector<std::array<double, 3>> samples;
  for (const double a : levels) {
    for (const double b : levels) {
      for (const double c : levels) samples.push_back({a, b, c});
    }
  }
  unround::test::Numbers numbers(41);
  for (int k = 0; k < 2000; ++k) {
    samples.push_back({static_cast<double>(numbers.between(0, 255)), static_cast<double>(numbers.between(0, 255)),
                       static_cast<double>(numbers.between(0, 255))});
  }
  return samples;
}

// The planes of a list of triples.
struct Planes {
  std::vector<double> first;
  std::vector<double> second;
  std::vector<double> third;
};

Planes planes_of(const std::vector<std::array<double, 3>>& samples) {
  Planes planes;
  for (const std::array<double, 3>& sample : samples) {
    planes.first.push_back(sample[0]);
    planes.second.push_back(sample[1]);
    planes.third.push_back(sample[2]);
  }
  return planes;
}

// The rounding bounds of the RGB of Y, Cb and Cr: R and B with the subtraction of 128,
// the constant, the product and the sum; G with two of each but the subtraction's.
struct Bounds {
  double first = 0.0;
  double second = 0.0;
  double third = 0.0;
};

Bounds rgb_bounds(double y, double cb, double cr) noexcept {
  const double from_cb = std::abs(cb - 128.0);
  const double from_cr = std::abs(cr - 128.0);
  return Bounds{.first = gamma(4) * (std::abs(y) + colour::red_from_cr * from_cr),
                .second = gamma(6) * (std::abs(y) + colour::green_from_cb * from_cb + colour::green_from_cr * from_cr),
                .third = gamma(4) * (std::abs(y) + colour::blue_from_cb * from_cb)};
}

// Those of the YCbCr of R, G and B: Y's three products of a constant and their two sums;
// Cb's and Cr's terms through Y, the difference, the constant, the product and the sum.
Bounds ycbcr_bounds(double r, double g, double b) noexcept {
  const double luma =
      colour::luma_red * std::abs(r) + colour::luma_green * std::abs(g) + colour::luma_blue * std::abs(b);
  return Bounds{.first = gamma(5) * luma,
                .second = gamma(9) * (128.0 + colour::cb_from_blue * (std::abs(b) + luma)),
                .third = gamma(9) * (128.0 + colour::cr_from_red * (std::abs(r) + luma))};
}

void to_rgb(void) {
  const std::vector<std::array<double, 3>> samples = samples_of();
  const Planes ycbcr = planes_of(samples);
  const std::size_t count = samples.size();
  std::vector<double> r(count);
  std::vector<double> g(count);
  std::vector<double> b(count);
  colour::to_rgb(ycbcr.first, ycbcr.second, ycbcr.third, r, g, b);
  bool grey = true;
  bool red = true;
  bool green = true;
  bool blue = true;
  for (std::size_t k = 0; k < count; ++k) {
    const auto [y, cb, cr] = samples[k];
    if (cb == 128.0 && cr == 128.0) grey = grey && same(r[k], y) && same(g[k], y) && same(b[k], y);
    // R = (500 Y + 701 (Cr - 128)) / 500, B = (250 Y + 443 (Cb - 128)) / 250, and
    // G = (293500 Y - 101004 (Cb - 128) - 209599 (Cr - 128)) / 293500, exact integers.
    const Bounds bounds = rgb_bounds(y, cb, cr);
    red = red && within(r[k], 500.0, 500.0 * y + 701.0 * (cr - 128.0), bounds.first);
    green = green &&
            within(g[k], 293500.0, 293500.0 * y - 101004.0 * (cb - 128.0) - 209599.0 * (cr - 128.0), bounds.second);
    blue = blue && within(b[k], 250.0, 250.0 * y + 443.0 * (cb - 128.0), bounds.third);
  }
  CHECK(grey);
  CHECK(red);
  CHECK(green);
  CHECK(blue);
}

void to_ycbcr(void) {
  const std::vector<std::array<double, 3>> samples = samples_of();
  const Planes rgb = planes_of(samples);
  const std::size_t count = samples.size();
  std::vector<double> y(count);
  std::vector<double> cb(count);
  std::vector<double> cr(count);
  colour::to_ycbcr(rgb.first, rgb.second, rgb.third, y, cb, cr);
  bool luma = true;
  bool blue = true;
  bool red = true;
  for (std::size_t k = 0; k < count; ++k) {
    const auto [r, g, b] = samples[k];
    // Y = (299 R + 587 G + 114 B) / 1000, Cb = 128 + (886 B - 299 R - 587 G) / 1772, and
    // Cr = 128 + (701 R - 587 G - 114 B) / 1402.
    const Bounds bounds = ycbcr_bounds(r, g, b);
    luma = luma && within(y[k], 1000.0, 299.0 * r + 587.0 * g + 114.0 * b, bounds.first);
    blue = blue && within(cb[k], 1772.0, 1772.0 * 128.0 + 886.0 * b - 299.0 * r - 587.0 * g, bounds.second);
    red = red && within(cr[k], 1402.0, 1402.0 * 128.0 + 701.0 * r - 587.0 * g - 114.0 * b, bounds.third);
  }
  CHECK(luma);
  CHECK(blue);
  CHECK(red);
}

// YCbCr to RGB and back: the exact conversions are each other's inverses, so the
// result is the start moved by the back conversion's linear part applied to the errors
// of the RGB, and by the back conversion's own rounding at the computed RGB.
void round_trip(void) {
  const std::vector<std::array<double, 3>> samples = samples_of();
  const Planes ycbcr = planes_of(samples);
  const std::size_t count = samples.size();
  std::vector<double> r(count);
  std::vector<double> g(count);
  std::vector<double> b(count);
  colour::to_rgb(ycbcr.first, ycbcr.second, ycbcr.third, r, g, b);
  std::vector<double> y(count);
  std::vector<double> cb(count);
  std::vector<double> cr(count);
  colour::to_ycbcr(r, g, b, y, cb, cr);
  bool back = true;
  for (std::size_t k = 0; k < count; ++k) {
    const Bounds forward = rgb_bounds(samples[k][0], samples[k][1], samples[k][2]);
    const Bounds own = ycbcr_bounds(r[k], g[k], b[k]);
    const double through_luma =
        colour::luma_red * forward.first + colour::luma_green * forward.second + colour::luma_blue * forward.third;
    const double bound_y = through_luma + own.first;
    const double bound_cb = colour::cb_from_blue * (forward.third + through_luma) + own.second;
    const double bound_cr = colour::cr_from_red * (forward.first + through_luma) + own.third;
    // The bounds round within gamma_8 of themselves, and so does the comparison.
    const double raise = 1.0 + gamma(8);
    back = back && std::abs(y[k] - samples[k][0]) <= bound_y * raise &&
           std::abs(cb[k] - samples[k][1]) <= bound_cb * raise && std::abs(cr[k] - samples[k][2]) <= bound_cr * raise;
  }
  CHECK(back);
}

int run(void) {
  constants();
  to_rgb();
  to_ycbcr();
  round_trip();
  return unround::test::finish("colour");
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
