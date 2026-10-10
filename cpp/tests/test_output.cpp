// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Tests of what the files are made of (docs/math.md, 10, and docs/cli.md): every level
// of the ordered dither against its definition in integers, the levels of a constant
// over the matrix keeping its mean, the channels' shifts, the levels of 16 bits of the
// exact product 257 x (samples whose rounded product lands on a place where the level
// turns, from below, among them), and rounding with its clamps; the eight orientations,
// each turned back by its inverse; and TIFF and PNM files read back, their samples to
// the last bit and their fields.

#include "unround/output.hpp"

#include "unround/arrays.hpp"
#include "unround/exact.hpp"

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
#include <limits>
#include <map>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

using unround::output::Dither;
using unround::output::LevelsArray;
using unround::output::PictureArray;

bool same(double a, double b) noexcept { return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b); }

using i128 = unround::exact::i128;

// The level of scale x, the product exact, where its fraction reaches bound / 8192 (8191
// - 2 R for the dither by rank R, 4096 for rounding), clamped to 0 to top: x = m 2^-k,
// and the whole part and the fraction of scale m / 2^k in 128-bit integers.
std::int64_t exact_level(double x, std::int64_t scale, std::int64_t bound, std::int64_t top) {
  if (!(x > 0.0)) return 0;
  int exponent = 0;
  const double fraction = std::frexp(x, &exponent);
  // Through 64 bits: a conversion of a double to 128 bits calls the compiler's runtime.
  const auto m = static_cast<i128>(static_cast<std::int64_t>(std::ldexp(fraction, 53)));
  const int k = 53 - exponent;
  const i128 product = m * scale;
  const i128 whole = product >> k;
  const i128 rest = product - (whole << k);
  const bool up = rest * 8192 >= static_cast<i128>(bound) << k;
  return std::clamp(static_cast<std::int64_t>(whole) + (up ? 1 : 0), std::int64_t{0}, top);
}

// Whether the product 257 x rounds to the double place, the exact one lying below it.
bool onto_from_below(double x, double place) noexcept {
  const double y = 257.0 * x;
  return y == place && std::fma(257.0, x, -y) < 0.0;
}

// The samples near place / 257 whose products round to place from below, where there is
// one; else the nearest.
double near_257th(double place) noexcept {
  const double near = place / 257.0;
  for (const double x : {std::nextafter(near, 0.0), near, std::nextafter(near, 256.0)}) {
    if (onto_from_below(x, place)) return x;
  }
  return near;
}

// The level of y against rank r, in integers: y = n / 2^20 for an integer n, floor(y)
// = n >> 20 (rounding toward minus infinity), and the fraction reaches 1 - t, t = (2 r +
// 1) / 8192, where 8192 (n - 2^20 floor(y)) + (2 r + 1) 2^20 >= 8192 2^20.
std::int64_t level_of(std::int64_t n, std::uint16_t rank, std::int64_t top) {
  const std::int64_t whole = n >> 20;
  const std::int64_t fraction = n - (whole << 20);
  const std::int64_t up =
      8192 * fraction + (2 * std::int64_t{rank} + 1) * (std::int64_t{1} << 20) >= std::int64_t{8192} << 20 ? 1 : 0;
  return std::clamp(whole + up, std::int64_t{0}, top);
}

void dither(void) {
  unround::test::Numbers numbers(71);
  // Samples n / 2^20 over -2 to 258, the levels of 8 bits.
  const std::size_t rows = 70;
  const std::size_t columns = 90;
  for (const std::size_t channels : {std::size_t{1}, std::size_t{3}, std::size_t{4}}) {
    for (const bool apart : {true, false}) {
      PictureArray picture(unround::PictureExtents{rows, columns, channels});
      std::vector<std::int64_t> ns(picture.size());
      for (std::size_t k = 0; k < ns.size(); ++k) {
        ns[k] = numbers.between(-2 * (std::int64_t{1} << 20), 258 * (std::int64_t{1} << 20));
        picture.entries()[k] = std::ldexp(static_cast<double>(ns[k]), -20);
      }
      const LevelsArray levels = unround::output::levels_of(picture.view(), 8, Dither::blue_noise, apart);
      constexpr std::array<std::array<std::size_t, 2>, 4> shifts = {{{0, 0}, {0, 32}, {32, 0}, {32, 32}}};
      bool equal = true;
      for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < columns; ++j) {
          for (std::size_t c = 0; c < channels; ++c) {
            const std::array<std::size_t, 2> shift = apart ? shifts[c] : std::array<std::size_t, 2>{0, 0};
            const std::uint16_t rank =
                unround::output::blue_noise()[((i + shift[0]) % 64u) * 64u + (j + shift[1]) % 64u];
            const std::size_t k = (i * columns + j) * channels + c;
            equal = equal && std::int64_t{levels.view()[i, j, c]} == level_of(ns[k], rank, 255);
          }
        }
      }
      CHECK(equal);
    }
  }
  // A constant of fraction f goes up at floor(4096 f + 1/2) of the matrix's 4096 places.
  for (const double y : {17.0, 17.25, 17.5, 17.0001, 17.9999, 200.125}) {
    PictureArray picture(unround::PictureExtents{64, 64, 1}, y);
    const LevelsArray levels = unround::output::levels_of(picture.view(), 8, Dither::blue_noise, true);
    std::size_t up = 0;
    for (const std::uint16_t level : levels.entries()) up += level > static_cast<std::uint16_t>(y) ? 1u : 0u;
    const double fraction = y - std::floor(y);
    CHECK(up == static_cast<std::size_t>(std::floor(4096.0 * fraction + 0.5)));
  }
  // 16 bits: y = 257 x, the product exact. Half the samples at random, half where the
  // rounded product lands on the place where the dither of the sample goes up.
  PictureArray picture(unround::PictureExtents{8, 64, 1});
  std::size_t onto = 0;
  for (std::size_t i = 0; i < 8u; ++i) {
    for (std::size_t j = 0; j < 64u; ++j) {
      const double limit = 1.0 - (2.0 * unround::output::blue_noise()[i * 64u + j] + 1.0) / 8192.0;
      const double place = static_cast<double>(997u * (i * 64u + j) % 65000u) + limit;
      const double x = j % 2u == 0u ? numbers.uniform(-1.0, 256.0) : near_257th(place);
      onto += onto_from_below(x, place) ? 1u : 0u;
      picture.view()[i, j, 0] = x;
    }
  }
  CHECK(onto > 20u);
  const LevelsArray sixteen = unround::output::levels_of(picture.view(), 16, Dither::blue_noise, true);
  bool scaled = true;
  for (std::size_t i = 0; i < 8u; ++i) {
    for (std::size_t j = 0; j < 64u; ++j) {
      const std::int64_t bound = 8191 - 2 * std::int64_t{unround::output::blue_noise()[i * 64u + j]};
      scaled =
          scaled && std::int64_t{sixteen.view()[i, j, 0]} == exact_level(picture.view()[i, j, 0], 257, bound, 65535);
    }
  }
  CHECK(scaled);
}

void rounding(void) {
  PictureArray picture(unround::PictureExtents{1, 9, 1});
  const std::array<double, 9> samples = {-0.5, -0.4, 0.49999999999999994, 0.5, 1.5, 2.5, 254.5, 255.4, 1e300};
  const std::array<std::uint16_t, 9> expected = {0, 0, 0, 1, 2, 3, 255, 255, 255};
  std::ranges::copy(samples, picture.entries().begin());
  const LevelsArray levels = unround::output::levels_of(picture.view(), 8, Dither::none, true);
  CHECK(std::ranges::equal(levels.entries(), expected));
  // 16 bits: the nearest level to the exact 257 x, at halves k + 1/2 where the rounded
  // product lands on the half from below.
  PictureArray halves(unround::PictureExtents{1, 3000, 1});
  std::size_t onto = 0;
  for (std::size_t j = 0; j < 3000u; ++j) {
    const double half = static_cast<double>(21u * j) + 0.5;
    const double x = near_257th(half);
    onto += onto_from_below(x, half) ? 1u : 0u;
    halves.view()[0, j, 0] = x;
  }
  CHECK(onto > 500u);
  const LevelsArray rounded = unround::output::levels_of(halves.view(), 16, Dither::none, true);
  bool nearest = true;
  for (std::size_t j = 0; j < 3000u; ++j) {
    nearest = nearest && std::int64_t{rounded.view()[0, j, 0]} == exact_level(halves.view()[0, j, 0], 257, 4096, 65535);
  }
  CHECK(nearest);
  // NaN and the infinities clamp.
  PictureArray odd(unround::PictureExtents{1, 3, 1});
  odd.entries()[0] = std::numeric_limits<double>::quiet_NaN();
  odd.entries()[1] = -std::numeric_limits<double>::infinity();
  odd.entries()[2] = std::numeric_limits<double>::infinity();
  for (const Dither dither : {Dither::none, Dither::blue_noise}) {
    const LevelsArray clamped = unround::output::levels_of(odd.view(), 16, dither, true);
    CHECK(clamped.entries()[0] == 0u && clamped.entries()[1] == 0u && clamped.entries()[2] == 65535u);
  }
}

void orientations(void) {
  const std::size_t h = 3;
  const std::size_t w = 5;
  PictureArray stored(unround::PictureExtents{h, w, 2});
  for (std::size_t i = 0; i < h; ++i) {
    for (std::size_t j = 0; j < w; ++j) {
      stored.view()[i, j, 0] = static_cast<double>(10 * i + j);
      stored.view()[i, j, 1] = -static_cast<double>(10 * i + j);
    }
  }
  const unround::Picture<const double> s = stored.view();
  for (std::int32_t o = 0; o <= 9; ++o) {
    const PictureArray turned = unround::output::oriented(s, o);
    const bool across = o >= 5 && o <= 8;
    CHECK(turned.extent(0) == (across ? w : h) && turned.extent(1) == (across ? h : w));
    CHECK(unround::output::turns(o) == (o >= 2 && o <= 8));
    bool equal = true;
    for (std::size_t r = 0; r < turned.extent(0); ++r) {
      for (std::size_t c = 0; c < turned.extent(1); ++c) {
        double expected = 0.0;
        switch (o) {
          case 2:
            expected = s[r, w - 1 - c, 0];
            break;
          case 3:
            expected = s[h - 1 - r, w - 1 - c, 0];
            break;
          case 4:
            expected = s[h - 1 - r, c, 0];
            break;
          case 5:
            expected = s[c, r, 0];
            break;
          case 6:
            expected = s[h - 1 - c, r, 0];
            break;
          case 7:
            expected = s[h - 1 - c, w - 1 - r, 0];
            break;
          case 8:
            expected = s[c, w - 1 - r, 0];
            break;
          default:
            expected = s[r, c, 0];
            break;
        }
        equal = equal && same(turned.view()[r, c, 0], expected) && same(turned.view()[r, c, 1], -expected);
      }
    }
    CHECK(equal);
    // Turned back by the inverse, it is the picture stored.
    const PictureArray back = unround::output::oriented(std::as_const(turned).view(), unround::output::inverse(o));
    CHECK(back.extent(0) == h && back.extent(1) == w);
    CHECK(std::ranges::equal(std::as_const(back).entries(), std::as_const(stored).entries(), same));
  }
}

std::uint32_t read32(std::span<const std::uint8_t> bytes, std::size_t at) {
  return static_cast<std::uint32_t>(bytes[at]) | static_cast<std::uint32_t>(bytes[at + 1u]) << 8u |
         static_cast<std::uint32_t>(bytes[at + 2u]) << 16u | static_cast<std::uint32_t>(bytes[at + 3u]) << 24u;
}

std::uint16_t read16(std::span<const std::uint8_t> bytes, std::size_t at) {
  return static_cast<std::uint16_t>(bytes[at] | bytes[at + 1u] << 8u);
}

// A TIFF file's fields, by tag: type, count, and the offset of the value (or of the
// entry's value bytes).
struct Entry {
  std::uint16_t type;
  std::uint32_t count;
  std::size_t at;
};

std::map<std::uint16_t, Entry> fields_of(std::span<const std::uint8_t> bytes) {
  std::map<std::uint16_t, Entry> fields;
  const std::size_t ifd = read32(bytes, 4);
  const std::uint16_t count = read16(bytes, ifd);
  for (std::size_t k = 0; k < count; ++k) {
    const std::size_t entry = ifd + 2u + 12u * k;
    const std::uint16_t type = read16(bytes, entry + 2u);
    const std::uint32_t values = read32(bytes, entry + 4u);
    const std::size_t size = std::size_t{type == 3 ? 2u : type == 4 ? 4u : type == 5 ? 8u : 1u} * values;
    fields[read16(bytes, entry)] =
        Entry{.type = type, .count = values, .at = size > 4u ? read32(bytes, entry + 8u) : entry + 8u};
  }
  return fields;
}

void tiff(void) {
  unround::test::Numbers numbers(72);
  const std::array<std::uint8_t, 7> profile = {1, 2, 3, 4, 5, 6, 7};
  for (const std::size_t channels : {std::size_t{1}, std::size_t{2}, std::size_t{3}, std::size_t{4}}) {
    PictureArray picture(unround::PictureExtents{6, 7, channels});
    for (double& sample : picture.entries()) sample = numbers.uniform(-10.0, 300.0);
    picture.entries()[0] = -0.0;
    const bool alpha = channels == 2u || channels == 4u;
    const auto photometric =
        channels == 1u || channels == 2u ? unround::output::Photometric::grey : unround::output::Photometric::rgb;
    const std::vector<std::uint8_t> bytes = unround::output::tiff_bytes(picture.view(), photometric, alpha, profile);
    CHECK(bytes.size() > 8u && bytes[0] == 'I' && bytes[1] == 'I' && read16(bytes, 2) == 42u);
    const std::map<std::uint16_t, Entry> fields = fields_of(bytes);
    CHECK(read32(bytes, fields.at(256).at) == 7u && read32(bytes, fields.at(257).at) == 6u);
    CHECK(fields.at(258).count == channels && read16(bytes, fields.at(258).at) == 64u);
    CHECK(fields.at(339).count == channels && read16(bytes, fields.at(339).at) == 3u);
    CHECK(read16(bytes, fields.at(277).at) == channels);
    CHECK(read16(bytes, fields.at(262).at) == (channels <= 2u ? 1u : 2u));
    CHECK(fields.contains(338) == alpha && (!alpha || read16(bytes, fields.at(338).at) == 2u));
    CHECK(
        fields.at(34675).count == profile.size() &&
        std::ranges::equal(std::span<const std::uint8_t>(bytes).subspan(fields.at(34675).at, profile.size()), profile));
    const std::size_t data = read32(bytes, fields.at(273).at);
    CHECK(read32(bytes, fields.at(279).at) == picture.size() * 8u && data + picture.size() * 8u == bytes.size());
    bool samples = true;
    for (std::size_t k = 0; k < picture.size(); ++k) {
      std::uint64_t value = 0;
      for (std::size_t b = 0; b < 8u; ++b) value |= static_cast<std::uint64_t>(bytes[data + 8u * k + b]) << (8u * b);
      samples = samples && value == std::bit_cast<std::uint64_t>(picture.entries()[k]);
    }
    CHECK(samples);
  }
  // YCbCr: its photometric interpretation, JFIF's coefficients, no subsampling, the full
  // range.
  PictureArray ycbcr(unround::PictureExtents{2, 3, 3}, 100.0);
  const std::vector<std::uint8_t> bytes =
      unround::output::tiff_bytes(ycbcr.view(), unround::output::Photometric::ycbcr, false, {});
  const std::map<std::uint16_t, Entry> fields = fields_of(bytes);
  CHECK(read16(bytes, fields.at(262).at) == 6u && !fields.contains(34675));
  CHECK(read32(bytes, fields.at(529).at) == 299u && read32(bytes, fields.at(529).at + 4u) == 1000u);
  CHECK(read16(bytes, fields.at(530).at) == 1u && read16(bytes, fields.at(530).at + 2u) == 1u);
  CHECK(read32(bytes, fields.at(532).at + 8u) == 255u && read32(bytes, fields.at(532).at + 16u) == 128u);
}

void pnm(void) {
  LevelsArray grey(unround::PictureExtents{2, 3, 1});
  for (std::size_t k = 0; k < grey.size(); ++k) grey.entries()[k] = static_cast<std::uint16_t>(k * 40u);
  const std::vector<std::uint8_t> eight = unround::output::pnm_bytes(grey.view(), 8);
  const std::string header(eight.begin(), eight.begin() + 11);
  CHECK(header == "P5\n3 2\n255\n" && eight.size() == 11u + 6u && eight[11u + 5u] == 200u);
  LevelsArray colour(unround::PictureExtents{1, 2, 3});
  for (std::size_t k = 0; k < colour.size(); ++k) colour.entries()[k] = static_cast<std::uint16_t>(1000u * k + 7u);
  const std::vector<std::uint8_t> sixteen = unround::output::pnm_bytes(colour.view(), 16);
  const std::string start(sixteen.begin(), sixteen.begin() + 13);
  CHECK(start == "P6\n2 1\n65535\n" && sixteen.size() == 13u + 12u);
  // Big-endian: 5007 = 0x138F.
  CHECK(sixteen[13u + 10u] == 0x13u && sixteen[13u + 11u] == 0x8Fu);
}

int run(void) {
  dither();
  rounding();
  orientations();
  tiff();
  pnm();
  return unround::test::finish("output");
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
