// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Tests of the reconstruction of a JPEG file: files written by the test encoder, in grey,
// in YCbCr 4:2:0 and in RGB, decoded by the MMSE decoder, whose canvas is the frame's
// start to the last bit, and by TV, TGV and the subgradient method, whose results lie
// within their intervals to 1e-4 Q (docs/math.md, 1.2); the picture, the canvas cut to
// the picture and, for YCbCr, its RGB to the last bit; the file's EXIF orientation and
// ICC profile; and the files and settings refused.

#include "unround/decode.hpp"

#include "unround/arrays.hpp"
#include "unround/colour.hpp"
#include "unround/frames.hpp"
#include "unround/jpegio.hpp"
#include "unround/settings.hpp"

#include "support/check.hpp"
#include "test_jpeg.h"

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

using unround::BlocksArray;
using unround::GridArray;

bool same(double a, double b) noexcept { return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b); }

// An EXIF APP1 payload with the orientation 6, and bytes that stand for a profile.
constexpr std::array<std::uint8_t, 32> exif = {'E',  'x',  'i', 'f', 0, 0, 'I', 'I', 42, 0, 8, 0, 0, 0, 1, 0,
                                               0x12, 0x01, 3,   0,   1, 0, 0,   0,   6,  0, 0, 0, 0, 0, 0, 0};
constexpr std::array<std::uint8_t, 12> profile = {'n', 'o', 't', ' ', 'a', ' ', 'p', 'r', 'o', 'f', 'i', 'l'};

// A file of random levels, of the colour space and the luma's sampling factor.
std::vector<std::uint8_t> file_of(std::uint32_t width, std::uint32_t height, std::int32_t color_space,
                                  std::int32_t luma_factor, std::uint32_t seed) {
  test_jpeg spec{};
  spec.width = width;
  spec.height = height;
  spec.color_space = color_space;
  spec.h_samp_factor[0] = luma_factor;
  spec.v_samp_factor[0] = luma_factor;
  spec.quant_table_slot[1] = 1;
  spec.quant_table_slot[2] = 1;
  spec.app1 = exif.data();
  spec.app1_size = static_cast<std::uint32_t>(exif.size());
  spec.icc_profile = profile.data();
  spec.icc_profile_size = static_cast<std::uint32_t>(profile.size());
  for (std::size_t slot = 0; slot < 2; ++slot) {
    for (std::size_t k = 0; k < 64; ++k) {
      spec.quant_tables[slot][k] = static_cast<std::uint16_t>(2 + (seed + 7 * k + 13 * slot) % 40);
    }
  }
  const std::int32_t components = test_jpeg_components(&spec);
  std::vector<std::vector<std::int16_t>> coefficients(static_cast<std::size_t>(components));
  unround::test::Numbers numbers(seed);
  for (std::int32_t ci = 0; ci < components; ++ci) {
    const std::size_t count = static_cast<std::size_t>(test_jpeg_blocks_wide(&spec, ci)) *
                              static_cast<std::size_t>(test_jpeg_blocks_high(&spec, ci)) * 64;
    std::vector<std::int16_t>& levels = coefficients[static_cast<std::size_t>(ci)];
    levels.resize(count);
    for (std::size_t i = 0; i < count; ++i) {
      const std::size_t k = i % 64;
      if (k == 0) {
        levels[i] = static_cast<std::int16_t>(numbers.between(-30, 30));
      } else if (k < 10 && numbers.between(0, 2) == 0) {
        levels[i] = static_cast<std::int16_t>(numbers.between(-6, 6));
      }
    }
    spec.coefficients[ci] = levels.data();
  }
  std::uint8_t* data = nullptr;
  std::size_t size = 0;
  char message[256];
  if (test_jpeg_write(&spec, &data, &size, message, sizeof message) != 0) {
    std::println(stderr, "the test encoder failed: {}", message);
    std::abort();
  }
  std::vector<std::uint8_t> file(data, data + size);
  std::free(data);
  return file;
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

// The frame of a file as decode makes it.
unround::Frame frame_of(std::span<const std::uint8_t> file, const unround::Settings& settings) {
  auto image = unround::jpegio::Image::read(file);
  if (!image) std::abort();
  std::vector<unround::ComponentInput> inputs;
  for (const unround_jpegio_component& component : image->components()) {
    unround::ComponentInput input{.levels = unround::jpegio::coefficients(component),
                                  .table = {},
                                  .h = static_cast<std::uint32_t>(component.h_samp_factor),
                                  .v = static_cast<std::uint32_t>(component.v_samp_factor)};
    std::ranges::copy(unround::jpegio::quant_table(component), input.table.begin());
    inputs.push_back(input);
  }
  const unround::ColourSpace space =
      image->color_space() == unround::jpegio::ColorSpace::grayscale ? unround::ColourSpace::grayscale
      : image->color_space() == unround::jpegio::ColorSpace::ycbcr   ? unround::ColourSpace::ycbcr
                                                                     : unround::ColourSpace::rgb;
  auto frame = unround::Frame::make(image->height(), image->width(), space, inputs, unround::data_of(settings));
  if (!frame) std::abort();
  return std::move(*frame);
}

// The picture: the canvas cut to the picture, and for YCbCr its RGB, to the last bit.
void check_picture(const unround::Decoded& decoded) {
  const std::size_t channels = decoded.canvas.size();
  CHECK(decoded.picture.extent(0) == decoded.height && decoded.picture.extent(1) == decoded.width &&
        decoded.picture.extent(2) == channels);
  bool equal = true;
  const unround::Picture<const double> picture = decoded.picture.view();
  for (std::size_t i = 0; i < decoded.height; ++i) {
    for (std::size_t j = 0; j < decoded.width; ++j) {
      std::array<double, 3> samples{};
      for (std::size_t c = 0; c < channels; ++c) samples[c] = decoded.canvas[c].view()[i, j];
      if (decoded.space == unround::ColourSpace::ycbcr) {
        std::array<double, 3> rgb{};
        unround::colour::to_rgb(std::span<const double>(&samples[0], 1), std::span<const double>(&samples[1], 1),
                                std::span<const double>(&samples[2], 1), std::span<double>(&rgb[0], 1),
                                std::span<double>(&rgb[1], 1), std::span<double>(&rgb[2], 1));
        samples = rgb;
      }
      for (std::size_t c = 0; c < channels; ++c) equal = equal && same(picture[i, j, c], samples[c]);
    }
  }
  CHECK(equal);
}

// The coefficients of the result lie within their intervals to 1e-4 Q.
void check_within(const unround::Decoded& decoded, const unround::Frame& frame) {
  bool within = true;
  for (std::size_t c = 0; c < frame.channels(); ++c) {
    const unround::Problem& problem = frame.component(c).problem;
    BlocksArray<double> coefficients(problem.extents());
    frame.coefficients(c, decoded.canvas[c].view(), coefficients.view());
    BlocksArray<double> excess(problem.extents());
    problem.excess(std::as_const(coefficients).view(), excess.view());
    const unround::Blocks<const double> view = std::as_const(excess).view();
    for (std::size_t by = 0; by < problem.rows(); ++by) {
      for (std::size_t bx = 0; bx < problem.columns(); ++bx) {
        for (std::size_t v = 0; v < 8u; ++v) {
          for (std::size_t w = 0; w < 8u; ++w) within = within && view[by, bx, v, w] <= 1e-4 * problem.steps()[v, w];
        }
      }
    }
  }
  CHECK(within);
}

void methods(void) {
  struct Kind {
    std::uint32_t width;
    std::uint32_t height;
    std::int32_t space;
    std::int32_t factor;
  };
  constexpr std::array<Kind, 3> kinds = {
      Kind{.width = 30, .height = 21, .space = TEST_JPEG_GRAYSCALE, .factor = 1},
      Kind{.width = 21, .height = 13, .space = TEST_JPEG_YCBCR, .factor = 2},
      Kind{.width = 24, .height = 16, .space = TEST_JPEG_RGB, .factor = 1},
  };
  for (const Kind kind : kinds) {
    const std::vector<std::uint8_t> file = file_of(kind.width, kind.height, kind.space, kind.factor, 51);
    {
      const unround::Settings settings = settings_of({"--method", "mmse"});
      const auto decoded = unround::decode(file, settings);
      if (!CHECK(decoded.has_value())) continue;
      CHECK(decoded->height == kind.height && decoded->width == kind.width);
      CHECK(decoded->records.empty() && decoded->iterations == 0u && !decoded->steps);
      CHECK(decoded->exif_orientation == 6);
      CHECK(std::ranges::equal(decoded->icc_profile, profile));
      const unround::Frame frame = frame_of(file, settings);
      bool start = true;
      for (std::size_t c = 0; c < frame.channels(); ++c) {
        GridArray<double> x(frame.canvas_extents());
        BlocksArray<double> kept(frame.component(c).problem.extents());
        frame.start(c, x.view(), kept.view());
        start = start && std::ranges::equal(std::as_const(x).entries(), decoded->canvas[c].entries(), same);
      }
      CHECK(start);
      check_picture(*decoded);
    }
    for (const std::string_view method : {std::string_view("tv"), std::string_view("tgv")}) {
      const unround::Settings settings = settings_of({"--method", method, "--iterations", "40", "--tolerance", "0"});
      const auto decoded = unround::decode(file, settings);
      if (!CHECK(decoded.has_value())) continue;
      CHECK(decoded->iterations == 40u && decoded->records.size() == 5u && decoded->steps.has_value());
      check_picture(*decoded);
      check_within(*decoded, frame_of(file, settings));
    }
    if (kind.space == TEST_JPEG_GRAYSCALE) {
      const unround::Settings settings = settings_of({"--method", "subgradient", "--subgradient-iterations", "10"});
      const auto decoded = unround::decode(file, settings);
      if (!CHECK(decoded.has_value())) continue;
      CHECK(decoded->iterations == 10u && decoded->records.size() == 11u);
      check_picture(*decoded);
      check_within(*decoded, frame_of(file, settings));
    }
  }
}

void refusals(void) {
  const std::vector<std::uint8_t> colour = file_of(21, 13, TEST_JPEG_YCBCR, 2, 52);
  const auto subgradient = unround::decode(colour, settings_of({"--method", "subgradient"}));
  CHECK(!subgradient && subgradient.error().code == unround::Errc::unsupported);
  const auto weights = unround::decode(colour, settings_of({"--method", "tv", "--channel-weights", "1,2"}));
  CHECK(!weights && weights.error().code == unround::Errc::options);
  const auto terms = unround::decode(colour, settings_of({"--mu", "0.1,0.2"}));
  CHECK(!terms && terms.error().code == unround::Errc::options);
  std::vector<std::uint8_t> garbage(colour.begin(), colour.begin() + 40);
  const auto cut = unround::decode(garbage, settings_of({}));
  CHECK(!cut && cut.error().code == unround::Errc::read);
  const std::array<std::uint8_t, 4> none = {1, 2, 3, 4};
  const auto nothing = unround::decode(none, settings_of({}));
  CHECK(!nothing && nothing.error().code == unround::Errc::read);
  // A file larger than the pixels allowed.
  const auto large = unround::decode(colour, settings_of({"--max-pixels", "100"}));
  CHECK(!large && large.error().code == unround::Errc::unsupported);
}

int run(void) {
  methods();
  refusals();
  return unround::test::finish("decode");
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
