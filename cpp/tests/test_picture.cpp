// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Tests of a picture made elsewhere, measured against a file's intervals and projected
// onto them (docs/math.md, 12): its canvas made as an encoder makes a file's, for grey
// and for YCbCr 4:2:0, to the last bit; its excess and its projection those of the
// frame's maps on that canvas, to the last bit; a picture of the middles of the
// intervals within them, a spike outside them in its own block alone, and the spike
// projected back within them to the rounding of the DCTs; the summary of an excess;
// and the pictures refused.

#include "unround/picture.hpp"

#include "unround/arrays.hpp"
#include "unround/colour.hpp"
#include "unround/dct.hpp"
#include "unround/decode.hpp"
#include "unround/frames.hpp"
#include "unround/settings.hpp"

#include "support/bounds.hpp"
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
#include <limits>
#include <print>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using unround::BlocksArray;
using unround::GridArray;
using unround::PictureArray;
using unround::PictureExtents;

bool same(double a, double b) noexcept { return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b); }

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

unround::Source source_of(std::span<const std::uint8_t> file, const unround::Settings& settings) {
  auto source = unround::read_source(file, settings);
  if (!source) {
    std::println(stderr, "{}", source.error().message);
    std::abort();
  }
  return std::move(*source);
}

// A picture of height x width x channels samples from numbers.
PictureArray picture_of(std::size_t height, std::size_t width, std::size_t channels, std::uint64_t seed) {
  PictureArray picture(PictureExtents{height, width, channels});
  unround::test::Numbers numbers(seed);
  for (double& sample : picture.entries()) sample = static_cast<double>(numbers.between(0, 255)) + 0.25;
  return picture;
}

// The canvas repeats the picture's last column and row; of YCbCr, JFIF's Y, Cb and Cr.
void canvases(void) {
  // Greyscale: 13 x 10 samples, on a canvas of 16 x 16.
  {
    const std::vector<std::uint8_t> file = file_of(10, 13, TEST_JPEG_GRAYSCALE, 1, 3);
    const unround::Source source = source_of(file, settings_of({}));
    CHECK(source.frame.rows() == 16u && source.frame.columns() == 16u);
    CHECK(unround::picture::channels_of(source.frame) == 1u);
    const PictureArray picture = picture_of(13, 10, 1, 4);
    const auto canvas = unround::picture::canvas_of(source.frame, std::as_const(picture).view());
    if (CHECK(canvas.has_value() && canvas->size() == 1u)) {
      bool repeated = true;
      for (std::size_t i = 0; i < 16u; ++i) {
        for (std::size_t j = 0; j < 16u; ++j) {
          repeated = repeated && same((*canvas)[0].view()[i, j],
                                      picture.view()[std::min<std::size_t>(i, 12u), std::min<std::size_t>(j, 9u), 0]);
        }
      }
      CHECK(repeated);
    }
  }
  // YCbCr 4:2:0: 20 x 30 pixels, on a canvas of 32 x 32.
  {
    const std::vector<std::uint8_t> file = file_of(30, 20, TEST_JPEG_YCBCR, 2, 5);
    const unround::Source source = source_of(file, settings_of({}));
    CHECK(source.frame.rows() == 32u && source.frame.columns() == 32u);
    const PictureArray rgb = picture_of(20, 30, 3, 6);
    const auto canvas = unround::picture::canvas_of(source.frame, std::as_const(rgb).view());
    if (CHECK(canvas.has_value() && canvas->size() == 3u)) {
      bool converted = true;
      for (std::size_t i = 0; i < 32u; ++i) {
        for (std::size_t j = 0; j < 32u; ++j) {
          const std::size_t r = std::min<std::size_t>(i, 19u);
          const std::size_t s = std::min<std::size_t>(j, 29u);
          std::array<double, 3> ycbcr{};
          unround::colour::to_ycbcr(std::span<const double>(&rgb.view()[r, s, 0], 1),
                                    std::span<const double>(&rgb.view()[r, s, 1], 1),
                                    std::span<const double>(&rgb.view()[r, s, 2], 1), std::span<double>(&ycbcr[0], 1),
                                    std::span<double>(&ycbcr[1], 1), std::span<double>(&ycbcr[2], 1));
          for (std::size_t c = 0; c < 3u; ++c) converted = converted && same((*canvas)[c].view()[i, j], ycbcr[c]);
        }
      }
      CHECK(converted);
    }
  }
}

// The excess and the projection of a picture are the frame's maps on its canvas.
void maps(void) {
  const std::vector<std::uint8_t> file = file_of(30, 20, TEST_JPEG_YCBCR, 2, 7);
  const unround::Settings settings = settings_of({});
  const unround::Source source = source_of(file, settings);
  const auto decoded = unround::decode(file, settings_of({"--method", "mmse"}));
  if (!CHECK(decoded.has_value())) return;
  // The MMSE decoder's picture, moved: a picture made elsewhere.
  PictureArray picture = decoded->picture;
  std::size_t index = 0;
  for (double& sample : picture.entries()) sample += index++ % 7 == 0 ? 9.0 : -2.5;
  const unround::Frame& frame = source.frame;
  const auto canvas = unround::picture::canvas_of(frame, std::as_const(picture).view());
  if (!CHECK(canvas.has_value())) return;
  // The excess, in steps, of the coefficients A x.
  const auto excess = unround::picture::excess(source, std::as_const(picture).view());
  if (!CHECK(excess.has_value() && excess->size() == 3u)) return;
  bool measured = true;
  bool outside = false;
  for (std::size_t c = 0; c < 3u; ++c) {
    const unround::Problem& problem = frame.component(c).problem;
    BlocksArray<double> coefficients(problem.extents());
    frame.coefficients(c, (*canvas)[c].view(), coefficients.view());
    BlocksArray<double> beyond(problem.extents());
    problem.excess(std::as_const(coefficients).view(), beyond.view());
    const unround::Blocks<const double> expected = std::as_const(beyond).view();
    const unround::Blocks<const double> got = (*excess)[c].view();
    for (std::size_t by = 0; by < problem.rows(); ++by) {
      for (std::size_t bx = 0; bx < problem.columns(); ++bx) {
        for (std::size_t v = 0; v < 8u; ++v) {
          for (std::size_t w = 0; w < 8u; ++w) {
            measured = measured && same(got[by, bx, v, w], expected[by, bx, v, w] / problem.steps()[v, w]);
            outside = outside || got[by, bx, v, w] > 0.0;
          }
        }
      }
    }
  }
  CHECK(measured);
  CHECK(outside);
  // The projection: each channel by the frame's projection, and the picture of the
  // projected canvas.
  const auto projection = unround::picture::project(source, std::as_const(picture).view());
  if (!CHECK(projection.has_value())) return;
  CHECK(*excess == projection->excess);
  bool projected = true;
  for (std::size_t c = 0; c < 3u; ++c) {
    GridArray<double> out(frame.canvas_extents());
    BlocksArray<double> kept(frame.component(c).problem.extents());
    unround::Room room = frame.room(c);
    frame.project(c, (*canvas)[c].view(), out.view(), kept.view(), room);
    projected = projected && std::ranges::equal(std::as_const(out).entries(),
                                                std::as_const(projection->decoded.canvas[c]).entries(), same);
  }
  CHECK(projected);
  const PictureArray expected = unround::picture_of(projection->decoded.canvas, 20, 30, unround::ColourSpace::ycbcr);
  CHECK(std::ranges::equal(std::as_const(expected).entries(), std::as_const(projection->decoded.picture).entries(),
                           same));
  CHECK(projection->decoded.records.empty() && projection->decoded.iterations == 0u);
  CHECK(projection->decoded.height == 20u && projection->decoded.width == 30u);
}

// The bound of a coefficient of a block measured again after its inverse DCT, from
// the coefficients zeta of the block: the inverse rounds each sample within e1 = epsilon
// |C|^T |zeta| |C| of it, and the forward DCT of the rounded samples rounds within
// epsilon |C| (|C|^T |zeta| |C| + e1) |C|^T, besides carrying e1 as |C| e1 |C|^T.
unround::dct::Block roundtrip_bound(const unround::dct::Block& zeta) {
  using unround::test::bound_rounding;
  using unround::test::epsilon;
  const unround::dct::Block magnitude = unround::test::inverse_magnitudes(zeta);
  unround::dct::Block first{};
  unround::dct::Block samples{};
  for (std::size_t k = 0; k < 64u; ++k) {
    first[k] = epsilon * magnitude[k] * bound_rounding;
    samples[k] = (magnitude[k] + first[k]) * bound_rounding;
  }
  const unround::dct::Block rounded = unround::test::forward_magnitudes(samples);
  const unround::dct::Block carried = unround::test::forward_magnitudes(first);
  unround::dct::Block bound{};
  for (std::size_t k = 0; k < 64u; ++k) bound[k] = (epsilon * rounded[k] + carried[k]) * bound_rounding;
  return bound;
}

// A picture of the middles lies within the intervals; a spike leaves them in its block
// alone, and its projection lies within them to the rounding of the DCTs.
void within(void) {
  // Of whole blocks, so that the canvas is the picture itself.
  const std::vector<std::uint8_t> file = file_of(32, 24, TEST_JPEG_GRAYSCALE, 1, 11);
  const unround::Source source = source_of(file, settings_of({}));
  const unround::Frame& frame = source.frame;
  const unround::Problem& problem = frame.component(0).problem;
  BlocksArray<double> middles(problem.extents());
  const std::span<double> middle = middles.entries();
  const std::span<const double> lower = unround::entries(problem.lower());
  const std::span<const double> upper = unround::entries(problem.upper());
  for (std::size_t k = 0; k < middle.size(); ++k) middle[k] = (lower[k] + upper[k]) / 2.0;
  GridArray<double> canvas(frame.canvas_extents());
  frame.canvas(0, std::as_const(middles).view(), 128.0, canvas.view());
  PictureArray picture(PictureExtents{24, 32, 1});
  for (std::size_t i = 0; i < 24u; ++i) {
    for (std::size_t j = 0; j < 32u; ++j) picture.view()[i, j, 0] = canvas.view()[i, j];
  }
  // The middles lie half a step from the ends, far beyond the rounding of a DCT.
  const auto excess = unround::picture::excess(source, std::as_const(picture).view());
  if (!CHECK(excess.has_value())) return;
  CHECK(std::ranges::all_of((*excess)[0].entries(), [](double value) { return value == 0.0; }));
  const unround::picture::Summary none = unround::picture::summary((*excess)[0].view());
  constexpr std::size_t blocks = 12;
  CHECK(none.coefficients == blocks * 64u && none.outside == 0u && none.largest == 0.0);
  // A spike leaves the intervals in its block alone.
  picture.view()[10, 13, 0] += 1500.0;
  const auto spiked = unround::picture::excess(source, std::as_const(picture).view());
  if (!CHECK(spiked.has_value())) return;
  const unround::Blocks<const double> view = (*spiked)[0].view();
  bool alone = true;
  for (std::size_t by = 0; by < 3u; ++by) {
    for (std::size_t bx = 0; bx < 4u; ++bx) {
      bool outside = false;
      for (std::size_t v = 0; v < 8u; ++v) {
        for (std::size_t w = 0; w < 8u; ++w) outside = outside || view[by, bx, v, w] > 0.0;
      }
      alone = alone && outside == (by == 1u && bx == 1u);
    }
  }
  CHECK(alone);
  const unround::picture::Summary summary = unround::picture::summary(view);
  double largest = 0.0;
  std::size_t outside = 0;
  for (const double value : (*spiked)[0].entries()) {
    largest = std::max(largest, value);
    outside += value > 0.0 ? 1u : 0u;
  }
  CHECK(summary.coefficients == blocks * 64u && summary.outside == outside && same(summary.largest, largest));
  CHECK(outside > 0u && largest > 1.0);
  // Projected, it lies within them to the rounding of the inverse DCT and of the
  // forward DCT that measures it again.
  const auto projection = unround::picture::project(source, std::as_const(picture).view());
  if (!CHECK(projection.has_value())) return;
  const auto again = unround::picture::excess(source, std::as_const(projection->decoded.picture).view());
  if (!CHECK(again.has_value())) return;
  // The coefficients that the projection kept: those of the spiked picture, clipped.
  GridArray<double> spiked_canvas(frame.canvas_extents());
  for (std::size_t i = 0; i < 24u; ++i) {
    for (std::size_t j = 0; j < 32u; ++j) spiked_canvas.view()[i, j] = picture.view()[i, j, 0];
  }
  BlocksArray<double> coefficients(problem.extents());
  frame.coefficients(0, std::as_const(spiked_canvas).view(), coefficients.view());
  BlocksArray<double> zeta(problem.extents());
  problem.clip(std::as_const(coefficients).view(), zeta.view());
  bool near = true;
  for (std::size_t by = 0; by < 3u; ++by) {
    for (std::size_t bx = 0; bx < 4u; ++bx) {
      const unround::dct::Block bound = roundtrip_bound(unround::test::block_of(std::as_const(zeta).view(), by, bx));
      for (std::size_t v = 0; v < 8u; ++v) {
        for (std::size_t w = 0; w < 8u; ++w) {
          const double value = (*again)[0].view()[by, bx, v, w];
          near = near && value * problem.steps()[v, w] <= bound[8u * v + w];
        }
      }
    }
  }
  CHECK(near);
}

// Pictures of other channels or sizes, or of samples not finite, are refused.
void refusals(void) {
  const std::vector<std::uint8_t> file = file_of(30, 20, TEST_JPEG_YCBCR, 2, 13);
  const unround::Source source = source_of(file, settings_of({}));
  CHECK(unround::picture::channels_of(source.frame) == 3u);
  const PictureArray grey = picture_of(20, 30, 1, 14);
  const auto channels = unround::picture::project(source, std::as_const(grey).view());
  CHECK(!channels && channels.error().code == unround::Errc::options);
  const PictureArray small = picture_of(19, 30, 3, 15);
  const auto size = unround::picture::excess(source, std::as_const(small).view());
  CHECK(!size && size.error().code == unround::Errc::options);
  PictureArray odd = picture_of(20, 30, 3, 16);
  for (const double bad : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
    odd.view()[3, 4, 1] = bad;
    const auto refused = unround::picture::project(source, std::as_const(odd).view());
    CHECK(!refused && refused.error().code == unround::Errc::options);
  }
  odd.view()[3, 4, 1] = 7.0;
  CHECK(unround::picture::project(source, std::as_const(odd).view()).has_value());
}

int run(void) {
  canvases();
  maps();
  within();
  refusals();
  return unround::test::finish("picture");
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
