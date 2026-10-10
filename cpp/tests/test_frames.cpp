// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Tests of the components on one canvas (docs/math.md, 1.3, 4.4 and 6.6): the
// canvas and the blocks each component keeps, and the layouts refused; the canvas of
// coefficients and back, within the bounds of the DCT's rounding; the proximal map,
// whose coefficients are each coefficient's proximal map at the step tau / n to the
// last bit and whose canvas moves only within the range of A's adjoint; the
// projection, likewise with the coefficients clipped, and its idempotence; the start;
// and the conjugate, which is the component's own at the sums of a field constant on
// the cells, and bounds the inner product with every point of the box by Fenchel and
// Young.

#include "unround/frames.hpp"

#include "unround/arrays.hpp"
#include "unround/dct.hpp"
#include "unround/model.hpp"

#include "support/bounds.hpp"
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
#include <print>
#include <span>
#include <utility>
#include <vector>

namespace {

using unround::BlocksArray;
using unround::ColourSpace;
using unround::ComponentInput;
using unround::DataTerm;
using unround::Frame;
using unround::GridArray;
using unround::test::Numbers;
using unround::test::u;

bool same(double a, double b) noexcept { return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b); }

// Levels of rows x columns blocks: DC within +-30, and a few AC levels within +-6.
BlocksArray<std::int16_t> levels_of(Numbers& numbers, std::size_t rows, std::size_t columns) {
  BlocksArray<std::int16_t> levels(unround::block_extents(rows, columns));
  const unround::Blocks<std::int16_t> view = levels.view();
  for (std::size_t by = 0; by < rows; ++by) {
    for (std::size_t bx = 0; bx < columns; ++bx) {
      view[by, bx, 0, 0] = static_cast<std::int16_t>(numbers.between(-30, 30));
      for (int k = 0; k < 6; ++k) {
        const auto v = static_cast<std::size_t>(numbers.between(0, 3));
        const auto w = static_cast<std::size_t>(numbers.between(v == 0u ? 1 : 0, 3));
        view[by, bx, v, w] = static_cast<std::int16_t>(numbers.between(-6, 6));
      }
    }
  }
  return levels;
}

std::array<std::uint16_t, 64> table_of(Numbers& numbers) {
  std::array<std::uint16_t, 64> table{};
  for (std::uint16_t& step : table) step = static_cast<std::uint16_t>(numbers.between(2, 40));
  return table;
}

// The data term of the tests: mu given, MMSE centres, a slack, and flat blocks of their
// own weight.
DataTerm data_of(void) {
  DataTerm data;
  data.mu = 0.01;
  data.centres = unround::Centres::mmse;
  data.slack = 0.25;
  data.flat_weight = 0.5;
  data.flat_dc = std::nullopt;
  return data;
}

// A picture of the sampling of each component (h, v), height x width, with the file's
// blocks of each component: those of its MCUs, ceil(height / (8 v_max)) MCUs down.
struct Picture {
  std::vector<BlocksArray<std::int16_t>> levels;
  std::vector<ComponentInput> inputs;
};

Picture picture_of(Numbers& numbers, std::span<const std::array<std::uint32_t, 2>> sampling, std::size_t height,
                   std::size_t width) {
  std::uint32_t h_max = 1;
  std::uint32_t v_max = 1;
  for (const std::array<std::uint32_t, 2>& factors : sampling) {
    h_max = std::max(h_max, factors[0]);
    v_max = std::max(v_max, factors[1]);
  }
  const std::size_t mcu_rows = 8u * static_cast<std::size_t>(v_max);
  const std::size_t mcu_columns = 8u * static_cast<std::size_t>(h_max);
  const std::size_t mcus_down = (height + mcu_rows - 1u) / mcu_rows;
  const std::size_t mcus_across = (width + mcu_columns - 1u) / mcu_columns;
  Picture picture;
  for (const std::array<std::uint32_t, 2>& factors : sampling) {
    const std::size_t rows = sampling.size() == 1u ? (height + 7u) / 8u : mcus_down * factors[1];
    const std::size_t columns = sampling.size() == 1u ? (width + 7u) / 8u : mcus_across * factors[0];
    picture.levels.push_back(levels_of(numbers, rows, columns));
  }
  for (std::size_t c = 0; c < sampling.size(); ++c) {
    picture.inputs.push_back(ComponentInput{.levels = std::as_const(picture.levels[c]).view(),
                                            .table = table_of(numbers),
                                            .h = sampling[c][0],
                                            .v = sampling[c][1]});
  }
  return picture;
}

Frame frame_of(const Picture& picture, std::size_t height, std::size_t width, ColourSpace space) {
  const std::array<DataTerm, 1> data{data_of()};
  std::expected<Frame, unround::Error> frame = Frame::make(height, width, space, picture.inputs, data);
  if (!frame) {
    std::println(stderr, "{}", frame.error().message);
    std::abort();
  }
  return std::move(*frame);
}

constexpr std::array<std::array<std::uint32_t, 2>, 1> grey = {{{1, 1}}};
constexpr std::array<std::array<std::uint32_t, 2>, 3> sampling_420 = {{{2, 2}, {1, 1}, {1, 1}}};
constexpr std::array<std::array<std::uint32_t, 2>, 3> sampling_422 = {{{2, 1}, {1, 1}, {1, 1}}};

void shapes(void) {
  Numbers numbers(11);
  {
    const Picture picture = picture_of(numbers, grey, 13, 21);
    const Frame frame = frame_of(picture, 13, 21, ColourSpace::grayscale);
    CHECK(frame.rows() == 16u && frame.columns() == 24u && frame.channels() == 1u);
    CHECK(frame.component(0).cells() == 1u && !frame.free(0) && !frame.free());
    CHECK(frame.component(0).block_rows() == 2u && frame.component(0).block_columns() == 3u);
  }
  {
    // 4:2:0: the canvas is 16 x 32 (whole MCUs of 16 x 16); Y keeps 2 x 3 of its 2 x 4
    // blocks, and the chroma 1 x 2, each sample a cell of 2 x 2.
    const Picture picture = picture_of(numbers, sampling_420, 13, 21);
    CHECK(picture.levels[0].extent(1) == 4u);
    const Frame frame = frame_of(picture, 13, 21, ColourSpace::ycbcr);
    CHECK(frame.rows() == 16u && frame.columns() == 32u && frame.channels() == 3u);
    CHECK(frame.component(0).block_rows() == 2u && frame.component(0).block_columns() == 3u);
    CHECK(frame.component(0).cells() == 1u && frame.free(0));
    CHECK(frame.component(1).block_rows() == 1u && frame.component(1).block_columns() == 2u);
    CHECK(frame.component(1).cell_rows == 2u && frame.component(1).cell_columns == 2u && frame.free(1));
    // The kept blocks are the file's at the top left.
    CHECK(frame.component(0).problem.levels()[1, 2, 0, 0] == picture.levels[0].view()[1, 2, 0, 0]);
  }
  {
    // 4:2:2: cells of 1 x 2.
    const Picture picture = picture_of(numbers, sampling_422, 13, 21);
    const Frame frame = frame_of(picture, 13, 21, ColourSpace::ycbcr);
    CHECK(frame.rows() == 16u && frame.columns() == 32u);
    CHECK(frame.component(2).cell_rows == 1u && frame.component(2).cell_columns == 2u);
    CHECK(frame.component(2).block_rows() == 2u && frame.component(2).block_columns() == 2u);
  }
  const std::array<DataTerm, 1> data{data_of()};
  {
    // Factors whose ratios are not integers, a factor out of 1 to 4, the count of the
    // components of the colour space, and a data term for neither every component nor
    // each.
    constexpr std::array<std::array<std::uint32_t, 2>, 3> thirds = {{{3, 1}, {2, 1}, {1, 1}}};
    const Picture picture = picture_of(numbers, thirds, 13, 21);
    CHECK(!Frame::make(13, 21, ColourSpace::ycbcr, picture.inputs, data));
    std::vector<ComponentInput> five = picture.inputs;
    five[0].h = 5;
    CHECK(!Frame::make(13, 21, ColourSpace::ycbcr, five, data));
    CHECK(!Frame::make(13, 21, ColourSpace::grayscale, picture.inputs, data));
    const std::array<DataTerm, 2> two{data_of(), data_of()};
    const Picture fine = picture_of(numbers, sampling_420, 13, 21);
    CHECK(!Frame::make(13, 21, ColourSpace::ycbcr, fine.inputs, two));
    // Blocks that do not cover the picture.
    const Picture small = picture_of(numbers, grey, 13, 21);
    CHECK(!Frame::make(17, 21, ColourSpace::grayscale, small.inputs, data));
    CHECK(!Frame::make(0, 21, ColourSpace::grayscale, small.inputs, data));
  }
}

// Random coefficients of a component's blocks, within +-200.
BlocksArray<double> coefficients_of(Numbers& numbers, const unround::Component& component) {
  BlocksArray<double> values(unround::block_extents(component.block_rows(), component.block_columns()));
  for (double& value : values.entries()) value = numbers.uniform(-200.0, 200.0);
  return values;
}

// The canvas of coefficients is constant on every cell and `beyond` past the blocks,
// and its coefficients are those given, within the rounding of the inverse DCT and the
// forward one: the means of a cell of equal samples are exact (cells of 1, 2 or 4).
void canvas_and_back(void) {
  Numbers numbers(12);
  const Picture picture = picture_of(numbers, sampling_420, 13, 21);
  const Frame frame = frame_of(picture, 13, 21, ColourSpace::ycbcr);
  for (std::size_t c = 0; c < frame.channels(); ++c) {
    const unround::Component& component = frame.component(c);
    const BlocksArray<double> z = coefficients_of(numbers, component);
    GridArray<double> x(frame.canvas_extents());
    frame.canvas(c, z.view(), 128.0, x.view());
    const unround::Grid<const double> canvas = std::as_const(x).view();
    const std::size_t down = component.cell_rows;
    const std::size_t across = component.cell_columns;
    const std::size_t block_rows = component.block_rows();
    const std::size_t block_columns = component.block_columns();
    bool constant = true;
    bool beyond = true;
    for (std::size_t i = 0; i < frame.rows(); ++i) {
      for (std::size_t j = 0; j < frame.columns(); ++j) {
        const bool within = i < 8u * block_rows * down && j < 8u * block_columns * across;
        if (!within) {
          beyond = beyond && canvas[i, j] == 128.0;
          continue;
        }
        const std::size_t top = (i / down) * down;
        const std::size_t left = (j / across) * across;
        constant = constant && same(canvas[i, j], canvas[top, left]);
      }
    }
    CHECK(constant);
    CHECK(beyond);
    BlocksArray<double> back(z.extents());
    frame.coefficients(c, canvas, back.view());
    // The samples of the cells, one to a cell, as the blocks of the component.
    GridArray<double> samples(unround::grid_extents(8u * block_rows, 8u * block_columns));
    for (std::size_t i = 0; i < samples.extent(0); ++i) {
      for (std::size_t j = 0; j < samples.extent(1); ++j) samples.view()[i, j] = canvas[down * i, across * j];
    }
    bool within = true;
    for (std::size_t by = 0; by < block_rows; ++by) {
      for (std::size_t bx = 0; bx < block_columns; ++bx) {
        const unround::dct::Block given = unround::test::block_of(z.view(), by, bx);
        const unround::dct::Block made = unround::test::block_of(std::as_const(samples).view(), by, bx);
        const unround::dct::Block own = unround::test::forward_magnitudes(made);
        const unround::dct::Block carried = unround::test::forward_magnitudes(unround::test::inverse_magnitudes(given));
        const unround::dct::Block result = unround::test::block_of(std::as_const(back).view(), by, bx);
        for (std::size_t k = 0; k < 64u; ++k) {
          const double bound =
              (unround::test::epsilon * own[k] + unround::test::epsilon * (1.0 + unround::test::epsilon) * carried[k]) *
              unround::test::bound_rounding;
          within = within && std::abs(result[k] - given[k]) <= bound * (1.0 - 2.0 * u);
        }
      }
    }
    CHECK(within);
  }
}

// A canvas of random samples within -20 to 280.
GridArray<double> samples_of(Numbers& numbers, const Frame& frame) {
  GridArray<double> v(frame.canvas_extents());
  for (double& sample : v.entries()) sample = numbers.uniform(-20.0, 280.0);
  return v;
}

// The coefficients of the map at A v are each coefficient's own map, to the last bit;
// the samples beyond the blocks are v's; where the cells have one sample, the blocks
// are the inverse DCT of the coefficients, to the last bit; where they have several,
// each sample moves by the change of its cell, and the coefficients of the result are
// the map's within the rounding of the change, of the means and of the DCT.
template <bool projection>
void map_of(const Frame& frame, std::size_t c, Numbers& numbers) {
  const unround::Component& component = frame.component(c);
  const unround::Problem& problem = component.problem;
  const GridArray<double> v = samples_of(numbers, frame);
  const double tau = 0.7;
  const auto cells = static_cast<double>(component.cells());
  BlocksArray<double> at(unround::block_extents(component.block_rows(), component.block_columns()));
  frame.coefficients(c, v.view(), at.view());
  GridArray<double> out(frame.canvas_extents(), -1.0);
  BlocksArray<double> kept(at.extents());
  unround::Room room = frame.room(c);
  if constexpr (projection) {
    frame.project(c, v.view(), out.view(), kept.view(), room);
  } else {
    frame.prox(c, v.view(), tau, out.view(), kept.view(), room);
  }
  bool each = true;
  const unround::Blocks<const double> e = std::as_const(at).view();
  const unround::Blocks<const double> zeta = std::as_const(kept).view();
  for (std::size_t by = 0; by < at.extent(0); ++by) {
    for (std::size_t bx = 0; bx < at.extent(1); ++bx) {
      for (std::size_t w = 0; w < 8u; ++w) {
        for (std::size_t x = 0; x < 8u; ++x) {
          const double value = e[by, bx, w, x];
          const double expected = projection
                                      ? std::clamp(value, problem.lower()[by, bx, w, x], problem.upper()[by, bx, w, x])
                                      : problem.prox_at(by, bx, w, x, value, tau / cells);
          each = each && same(zeta[by, bx, w, x], expected);
        }
      }
    }
  }
  CHECK(each);
  const std::size_t down = component.cell_rows;
  const std::size_t across = component.cell_columns;
  const std::size_t block_rows = component.block_rows();
  const std::size_t block_columns = component.block_columns();
  const unround::Grid<const double> before = v.view();
  const unround::Grid<const double> after = std::as_const(out).view();
  bool beyond = true;
  for (std::size_t i = 0; i < frame.rows(); ++i) {
    for (std::size_t j = 0; j < frame.columns(); ++j) {
      if (i >= 8u * block_rows * down || j >= 8u * block_columns * across) {
        beyond = beyond && same(after[i, j], before[i, j]);
      }
    }
  }
  CHECK(beyond);
  if (component.cells() == 1u) {
    GridArray<double> back(unround::grid_extents(8u * block_rows, 8u * block_columns));
    unround::dct::inverse(zeta, back.view());
    bool blocks = true;
    for (std::size_t i = 0; i < back.extent(0); ++i) {
      for (std::size_t j = 0; j < back.extent(1); ++j) blocks = blocks && same(after[i, j], back.view()[i, j]);
    }
    CHECK(blocks);
    return;
  }
  // Each sample of a cell moves alike: out = (v + c)(1 + d) with |d| <= u, so that
  // out - v, computed here with one more rounding, lies within u |out| + u |out - v| of
  // the cell's change c, to the second order (1 + gamma_2); two samples' within the sum
  // of their bounds.
  bool alike = true;
  const auto n = static_cast<int>(component.cells());
  GridArray<double> means(unround::grid_extents(8u * block_rows, 8u * block_columns));
  GridArray<double> errors(means.extents());
  for (std::size_t i = 0; i < means.extent(0); ++i) {
    for (std::size_t j = 0; j < means.extent(1); ++j) {
      double first_change = 0.0;
      double first_bound = 0.0;
      double sum_out = 0.0;
      double sum_v = 0.0;
      double magnitude_v = 0.0;
      double magnitude_out = 0.0;
      for (std::size_t a = 0; a < down; ++a) {
        for (std::size_t b = 0; b < across; ++b) {
          const double sample = after[down * i + a, across * j + b];
          const double original = before[down * i + a, across * j + b];
          const double moved = sample - original;
          const double bound = u * (std::abs(sample) + std::abs(moved)) * (1.0 + unround::test::gamma(2));
          if (a == 0u && b == 0u) {
            first_change = moved;
            first_bound = bound;
          }
          alike = alike && std::abs(moved - first_change) <= (bound + first_bound) * (1.0 + 2.0 * u);
          sum_out += sample;
          sum_v += original;
          magnitude_v += std::abs(original);
          magnitude_out += std::abs(sample);
        }
      }
      means.view()[i, j] = sum_out / cells;
      // The mean of out against the inverse DCT of zeta, B: the mean M of v's cell, within
      // gamma_{n-1} of the mean of |v|; the change c = B - M, within u |c|; each sum
      // v + c, within u |out|; and the mean of out taken here, within gamma_{n-1} of the
      // mean of |out|. |c| is |sum out - sum v| / n to within those, which the factor of
      // the bounds' rounding covers.
      const double change = std::abs(sum_out - sum_v) / cells;
      errors.view()[i, j] = unround::test::gamma(n - 1) * (magnitude_v / cells) + u * change +
                            (u + unround::test::gamma(n - 1)) * (magnitude_out / cells);
    }
  }
  CHECK(alike);
  bool within = true;
  BlocksArray<double> coefficients(at.extents());
  unround::dct::forward(std::as_const(means).view(), coefficients.view());
  for (std::size_t by = 0; by < block_rows; ++by) {
    for (std::size_t bx = 0; bx < block_columns; ++bx) {
      const unround::dct::Block made = unround::test::block_of(std::as_const(means).view(), by, bx);
      const unround::dct::Block given = unround::test::block_of(zeta, by, bx);
      // The inverse DCT of zeta (epsilon |C|^T |zeta| |C|) and the cells' errors, carried
      // through the forward DCT, and the forward DCT's own rounding.
      const unround::dct::Block inverse_errors = unround::test::inverse_magnitudes(given);
      unround::dct::Block cell_errors = unround::test::block_of(std::as_const(errors).view(), by, bx);
      for (std::size_t k = 0; k < 64u; ++k) cell_errors[k] += unround::test::epsilon * inverse_errors[k];
      const unround::dct::Block carried = unround::test::forward_magnitudes(cell_errors);
      const unround::dct::Block own = unround::test::forward_magnitudes(made);
      const unround::dct::Block result = unround::test::block_of(std::as_const(coefficients).view(), by, bx);
      for (std::size_t k = 0; k < 64u; ++k) {
        const double bound = ((1.0 + unround::test::epsilon) * carried[k] + unround::test::epsilon * own[k]) *
                             unround::test::bound_rounding;
        within = within && std::abs(result[k] - given[k]) <= bound * (1.0 - 2.0 * u);
      }
    }
  }
  CHECK(within);
}

void maps(void) {
  Numbers numbers(13);
  const Picture grey_picture = picture_of(numbers, grey, 13, 21);
  const Frame grey_frame = frame_of(grey_picture, 13, 21, ColourSpace::grayscale);
  map_of<false>(grey_frame, 0, numbers);
  map_of<true>(grey_frame, 0, numbers);
  for (const auto& sampling : {std::span<const std::array<std::uint32_t, 2>>(sampling_420),
                               std::span<const std::array<std::uint32_t, 2>>(sampling_422)}) {
    const Picture picture = picture_of(numbers, sampling, 13, 21);
    const Frame frame = frame_of(picture, 13, 21, ColourSpace::ycbcr);
    for (std::size_t c = 0; c < frame.channels(); ++c) {
      map_of<false>(frame, c, numbers);
      map_of<true>(frame, c, numbers);
    }
  }
}

// The projection is idempotent: projecting its result again clips its coefficients,
// which are the first projection's to the rounding that map_of bounds, and the clip,
// 1-Lipschitz and leaving the first's as they are, moves them from the first's by no
// more than that rounding: exactly so, the clip being exact.
void idempotence(void) {
  Numbers numbers(14);
  const Picture picture = picture_of(numbers, sampling_420, 13, 21);
  const Frame frame = frame_of(picture, 13, 21, ColourSpace::ycbcr);
  for (std::size_t c = 0; c < frame.channels(); ++c) {
    const unround::Component& component = frame.component(c);
    const GridArray<double> v = samples_of(numbers, frame);
    GridArray<double> once(frame.canvas_extents());
    BlocksArray<double> first(unround::block_extents(component.block_rows(), component.block_columns()));
    unround::Room room = frame.room(c);
    frame.project(c, v.view(), once.view(), first.view(), room);
    BlocksArray<double> at(first.extents());
    frame.coefficients(c, std::as_const(once).view(), at.view());
    GridArray<double> twice(frame.canvas_extents());
    BlocksArray<double> second(first.extents());
    frame.project(c, std::as_const(once).view(), twice.view(), second.view(), room);
    const std::span<const double> a = std::as_const(first).entries();
    const std::span<const double> b = std::as_const(second).entries();
    const std::span<const double> e = std::as_const(at).entries();
    bool within = true;
    for (std::size_t k = 0; k < a.size(); ++k) within = within && std::abs(b[k] - a[k]) <= std::abs(e[k] - a[k]);
    CHECK(within);
  }
}

// The start is the canvas of the centres clipped to their intervals, 128 beyond the
// blocks, and its coefficients are those clipped centres.
void start(void) {
  Numbers numbers(15);
  const Picture picture = picture_of(numbers, sampling_420, 13, 21);
  const Frame frame = frame_of(picture, 13, 21, ColourSpace::ycbcr);
  for (std::size_t c = 0; c < frame.channels(); ++c) {
    const unround::Problem& problem = frame.component(c).problem;
    GridArray<double> x(frame.canvas_extents());
    BlocksArray<double> kept(unround::block_extents(problem.rows(), problem.columns()));
    frame.start(c, x.view(), kept.view());
    BlocksArray<double> clipped(kept.extents());
    problem.clip(problem.centres(), clipped.view());
    CHECK(std::ranges::equal(std::as_const(kept).entries(), std::as_const(clipped).entries(), same));
    GridArray<double> expected(frame.canvas_extents());
    frame.canvas(c, std::as_const(clipped).view(), 128.0, expected.view());
    CHECK(std::ranges::equal(std::as_const(x).entries(), std::as_const(expected).entries(), same));
  }
}

// The conjugate. Where nothing is free, it is the component's at the DCT of the field,
// to the last bit. At a field constant on the cells and 0 beyond the blocks, the box
// adds nothing, and it is the component's at the DCT of the sums of the cells, to the
// last bit (the sums of 1, 2 or 4 equal samples are exact). And by Fenchel and Young
// it bounds the inner product with every point of the box: G(x) + G*(xi) >= <xi, x>
// for x whose coefficients lie within their intervals, whose deviations within the
// cells are at most R, and whose samples beyond the blocks lie within R of 128. The
// margin is orders of magnitude above the rounding of the three, and is taken as it
// is.
void conjugate(void) {
  Numbers numbers(16);
  const double radius = 255.0;
  {
    const Picture picture = picture_of(numbers, grey, 16, 24);
    const Frame frame = frame_of(picture, 16, 24, ColourSpace::grayscale);
    const unround::Problem& problem = frame.component(0).problem;
    const GridArray<double> xi = samples_of(numbers, frame);
    BlocksArray<double> s(unround::block_extents(problem.rows(), problem.columns()));
    unround::dct::forward(xi.view(), s.view());
    CHECK(same(frame.conjugate(0, xi.view(), radius), problem.conjugate(std::as_const(s).view())));
  }
  const Picture picture = picture_of(numbers, sampling_420, 13, 21);
  const Frame frame = frame_of(picture, 13, 21, ColourSpace::ycbcr);
  for (std::size_t c = 0; c < frame.channels(); ++c) {
    const unround::Component& component = frame.component(c);
    const unround::Problem& problem = component.problem;
    const std::size_t down = component.cell_rows;
    const std::size_t across = component.cell_columns;
    const auto cells = static_cast<double>(component.cells());
    const std::size_t block_rows = component.block_rows();
    const std::size_t block_columns = component.block_columns();
    // A field constant on the cells, 0 beyond the blocks.
    GridArray<double> values(unround::grid_extents(8u * block_rows, 8u * block_columns));
    for (double& value : values.entries()) value = numbers.uniform(-3.0, 3.0);
    GridArray<double> xi(frame.canvas_extents(), 0.0);
    GridArray<double> sums(values.extents());
    for (std::size_t i = 0; i < values.extent(0); ++i) {
      for (std::size_t j = 0; j < values.extent(1); ++j) {
        for (std::size_t a = 0; a < down; ++a) {
          for (std::size_t b = 0; b < across; ++b) xi.view()[down * i + a, across * j + b] = values.view()[i, j];
        }
        sums.view()[i, j] = cells * values.view()[i, j];
      }
    }
    BlocksArray<double> s(unround::block_extents(block_rows, block_columns));
    unround::dct::forward(std::as_const(sums).view(), s.view());
    CHECK(same(frame.conjugate(c, std::as_const(xi).view(), radius), problem.conjugate(std::as_const(s).view())));
    // Fenchel and Young, at points of the box and random fields.
    for (int trial = 0; trial < 20; ++trial) {
      // Coefficients within their intervals, their canvas, deviations within the cells
      // of at most R / 2 that sum to 0 (one sample up, another down), and samples beyond
      // the blocks within R of 128.
      BlocksArray<double> z(s.extents());
      const std::span<double> zs = z.entries();
      const std::span<const double> lows = std::span<const double>(problem.lower().data_handle(), zs.size());
      const std::span<const double> highs = std::span<const double>(problem.upper().data_handle(), zs.size());
      for (std::size_t k = 0; k < zs.size(); ++k) zs[k] = numbers.uniform(lows[k], highs[k]);
      GridArray<double> x(frame.canvas_extents());
      frame.canvas(c, std::as_const(z).view(), 128.0, x.view());
      const unround::Grid<double> point = x.view();
      for (std::size_t i = 0; i < frame.rows(); ++i) {
        for (std::size_t j = 0; j < frame.columns(); ++j) {
          if (i >= 8u * block_rows * down || j >= 8u * block_columns * across) {
            point[i, j] = 128.0 + numbers.uniform(-radius, radius);
          }
        }
      }
      if (component.cells() > 1u) {
        for (std::size_t i = 0; i < values.extent(0); ++i) {
          for (std::size_t j = 0; j < values.extent(1); ++j) {
            const double deviation = numbers.uniform(-radius / 2.0, radius / 2.0);
            point[down * i, across * j] += deviation;
            point[down * i + down - 1u, across * j + across - 1u] -= deviation;
          }
        }
      }
      BlocksArray<double> at(s.extents());
      frame.coefficients(c, std::as_const(x).view(), at.view());
      const double g = problem.value(std::as_const(at).view());
      GridArray<double> field(frame.canvas_extents());
      for (double& value : field.entries()) value = numbers.uniform(-2.0, 2.0);
      double inner = 0.0;
      const std::span<const double> fs = std::as_const(field).entries();
      const std::span<const double> xs = std::as_const(x).entries();
      for (std::size_t k = 0; k < fs.size(); ++k) inner += fs[k] * xs[k];
      CHECK(g + frame.conjugate(c, std::as_const(field).view(), radius) >= inner);
    }
  }
}

// Where the field's sums over the cells are 0, the conjugate is the box's alone, and
// Fenchel and Young hold with equality at the corner of the box that the field points
// to: x the start (its coefficients the centres, where G is 0) moved by R times the
// signs of the field within the cells (whose sums the moves keep) and beyond the
// blocks, so that G(x) + G*(xi) = <xi, x> exactly. Computed, the sum over the samples
// of the inner product and that of the box's terms each round within gamma_N of the
// sum of their terms' magnitudes; G(x), 0 exactly, is what the means of the moved
// cells round to.
void corner(void) {
  Numbers numbers(17);
  const double radius = 255.0;
  for (const auto& sampling : {std::span<const std::array<std::uint32_t, 2>>(sampling_420),
                               std::span<const std::array<std::uint32_t, 2>>(sampling_422)}) {
    const Picture picture = picture_of(numbers, sampling, 13, 21);
    const Frame frame = frame_of(picture, 13, 21, ColourSpace::ycbcr);
    for (std::size_t c = 0; c < frame.channels(); ++c) {
      const unround::Component& component = frame.component(c);
      const std::size_t down = component.cell_rows;
      const std::size_t across = component.cell_columns;
      const std::size_t block_rows = 8u * component.block_rows();
      const std::size_t block_columns = 8u * component.block_columns();
      GridArray<double> x(frame.canvas_extents());
      BlocksArray<double> kept(component.problem.extents());
      frame.start(c, x.view(), kept.view());
      GridArray<double> xi(frame.canvas_extents(), 0.0);
      const unround::Grid<double> field = xi.view();
      const unround::Grid<double> point = x.view();
      for (std::size_t i = 0; i < frame.rows(); ++i) {
        for (std::size_t j = 0; j < frame.columns(); ++j) {
          if (i < block_rows * down && j < block_columns * across) continue;
          field[i, j] = numbers.uniform(-2.0, 2.0);
          point[i, j] = 128.0 + std::copysign(radius, field[i, j]);
        }
      }
      if (component.cells() > 1u) {
        // In each cell, +a at its first sample and -a at its last: sums of 0, exactly.
        for (std::size_t i = 0; i < block_rows; ++i) {
          for (std::size_t j = 0; j < block_columns; ++j) {
            const double a = numbers.uniform(-2.0, 2.0);
            const std::size_t top = down * i;
            const std::size_t left = across * j;
            const std::size_t bottom = top + down - 1u;
            const std::size_t right = left + across - 1u;
            field[top, left] = a;
            field[bottom, right] = -a;
            point[top, left] += std::copysign(radius, a);
            point[bottom, right] -= std::copysign(radius, a);
          }
        }
      }
      BlocksArray<double> at(component.problem.extents());
      frame.coefficients(c, std::as_const(x).view(), at.view());
      const double g = component.problem.value(std::as_const(at).view());
      const double conjugate_value = frame.conjugate(c, std::as_const(xi).view(), radius);
      double inner = 0.0;
      double magnitude = 0.0;
      double box = 0.0;
      const std::span<const double> fs = std::as_const(xi).entries();
      const std::span<const double> xs = std::as_const(x).entries();
      for (std::size_t k = 0; k < fs.size(); ++k) {
        inner += fs[k] * xs[k];
        magnitude += std::abs(fs[k] * xs[k]);
        box += (radius + 128.0) * std::abs(fs[k]);
      }
      const auto count = static_cast<int>(fs.size());
      const double bound = unround::test::gamma(count) * (magnitude + box) + g;
      CHECK(std::abs((g + conjugate_value) - inner) <= bound * (1.0 + unround::test::gamma(4)));
    }
  }
}

// The data term of the chroma of a file in YCbCr is its of_chroma (the rule's scale
// times mu_chroma); of a file in RGB it is the data term given.
void chroma(void) {
  Numbers numbers(18);
  const DataTerm rule;  // mu by the rule
  const std::array<DataTerm, 1> data{rule};
  for (const ColourSpace space : {ColourSpace::ycbcr, ColourSpace::rgb}) {
    const Picture picture = picture_of(numbers, sampling_420, 13, 21);
    const auto frame = Frame::make(13, 21, space, picture.inputs, data);
    if (!CHECK(frame.has_value())) continue;
    for (std::size_t c = 0; c < 3u; ++c) {
      const unround::Problem& problem = frame->component(c).problem;
      const DataTerm term = space == ColourSpace::ycbcr && c > 0u ? rule.of_chroma() : rule;
      const auto expected = unround::Problem::make(problem.levels(), picture.inputs[c].table, term);
      CHECK(expected.has_value() &&
            std::ranges::equal(std::span<const double>(problem.weights().data_handle(), 64),
                               std::span<const double>(expected->weights().data_handle(), 64), same));
    }
  }
}

int run(void) {
  shapes();
  canvas_and_back();
  maps();
  idempotence();
  start();
  conjugate();
  corner();
  chroma();
  return unround::test::finish("frames");
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
