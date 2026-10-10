// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// A picture made elsewhere, measured against a file's intervals and projected onto
// them (docs/math.md, 12).

#include "unround/picture.hpp"

#include "unround/arrays.hpp"
#include "unround/colour.hpp"
#include "unround/decode.hpp"
#include "unround/error.hpp"
#include "unround/frames.hpp"
#include "unround/model.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <expected>
#include <format>
#include <span>
#include <utility>
#include <vector>

namespace unround::picture {

namespace {

// Each a count, or a largest value, of a loop and of its remainder alike, on one line:
// the check of vector code pairs a remainder with its vector loop by their lines.

// 1 for a sample that is not finite, and 0 for one that is.
std::size_t not_finite(double value) noexcept { return std::isfinite(value) ? 0u : 1u; }

// 1 for a value above 0, and 0 for one that is not.
std::size_t above_zero(double value) noexcept { return value > 0.0 ? 1u : 0u; }

// The larger of a value and a bound: the bound where the value is below it.
double at_least(double value, double bound) noexcept { return value < bound ? bound : value; }

// Channel c of a picture, height x width samples, row by row.
void plane_of(Picture<const double> picture, std::size_t c, std::span<double> plane) noexcept {
  const std::size_t height = picture.extent(0);
  const std::size_t width = picture.extent(1);
  for (std::size_t i = 0; i < height; ++i) {
    for (std::size_t j = 0; j < width; ++j) plane[i * width + j] = picture[i, j, c];
  }
}

// A plane of height x width samples over a grid, extended by repeating its last
// column, and then its last row.
void extend(std::span<const double> plane, std::size_t height, std::size_t width, Grid<double> grid) noexcept {
  const std::size_t rows = grid.extent(0);
  const std::size_t columns = grid.extent(1);
  for (std::size_t i = 0; i < rows; ++i) {
    const std::span<const double> from = plane.subspan(std::min(i, height - 1u) * width, width);
    const std::span<double> to(&grid[i, 0], columns);
    std::ranges::copy(from, to.begin());
    std::ranges::fill(to.subspan(width), from.back());
  }
}

}  // namespace

std::size_t channels_of(const Frame& frame) noexcept {
  return frame.colour_space() == ColourSpace::grayscale ? 1u : 3u;
}

std::expected<std::vector<GridArray<double>>, Error> canvas_of(const Frame& frame, Picture<const double> picture) {
  const std::size_t height = frame.height();
  const std::size_t width = frame.width();
  const std::size_t channels = channels_of(frame);
  if (picture.extent(0) != height || picture.extent(1) != width || picture.extent(2) != channels) {
    return std::unexpected(
        Error{.code = Errc::options,
              .message = std::format("a picture of the file is {} x {} x {} samples, not {} x {} x {}", height, width,
                                     channels, picture.extent(0), picture.extent(1), picture.extent(2))});
  }
  const std::span<const double> samples(picture.data_handle(), picture.size());
  std::size_t odd = 0;
  for (const double sample : samples) odd += not_finite(sample);
  if (odd != 0u) return std::unexpected(Error{.code = Errc::options, .message = "a picture's samples are finite"});
  const std::size_t count = height * width;
  std::vector<std::vector<double>> planes(channels, std::vector<double>(count));
  for (std::size_t c = 0; c < channels; ++c) plane_of(picture, c, planes[c]);
  if (frame.colour_space() == ColourSpace::ycbcr) {
    std::vector<std::vector<double>> ycbcr(3, std::vector<double>(count));
    colour::to_ycbcr(planes[0], planes[1], planes[2], ycbcr[0], ycbcr[1], ycbcr[2]);
    planes = std::move(ycbcr);
  }
  std::vector<GridArray<double>> canvas;
  for (const std::vector<double>& plane : planes) {
    GridArray<double> grid(frame.canvas_extents());
    extend(plane, height, width, grid.view());
    canvas.push_back(std::move(grid));
  }
  return canvas;
}

std::vector<BlocksArray<double>> excess(const Frame& frame, std::span<const GridArray<double>> canvas) {
  std::vector<BlocksArray<double>> result;
  for (std::size_t c = 0; c < frame.channels(); ++c) {
    const Problem& problem = frame.component(c).problem;
    BlocksArray<double> coefficients(problem.extents());
    frame.coefficients(c, canvas[c].view(), coefficients.view());
    BlocksArray<double> beyond(problem.extents());
    problem.excess(std::as_const(coefficients).view(), beyond.view());
    // In steps.
    const Blocks<double> blocks = beyond.view();
    const std::span<const double, 64> steps(problem.steps().data_handle(), 64);
    const std::size_t rows = blocks.extent(0);
    const std::size_t columns = blocks.extent(1);
    for (std::size_t by = 0; by < rows; ++by) {
      for (std::size_t bx = 0; bx < columns; ++bx) {
        const std::span<double, 64> block = block_entries(blocks, by, bx);
        for (std::size_t k = 0; k < 64; ++k) block[k] /= steps[k];
      }
    }
    result.push_back(std::move(beyond));
  }
  return result;
}

std::expected<std::vector<BlocksArray<double>>, Error> excess(const Source& source, Picture<const double> picture) {
  const std::expected<std::vector<GridArray<double>>, Error> canvas = canvas_of(source.frame, picture);
  if (!canvas) return std::unexpected(canvas.error());
  return excess(source.frame, *canvas);
}

std::expected<Projection, Error> project(const Source& source, Picture<const double> picture) {
  const Frame& frame = source.frame;
  std::expected<std::vector<GridArray<double>>, Error> canvas = canvas_of(frame, picture);
  if (!canvas) return std::unexpected(canvas.error());
  Projection projection{.decoded = source.decoded, .excess = excess(frame, *canvas)};
  // Each channel onto its component's set, in place: its coefficients clipped, its cells
  // moved alike, and the samples beyond its blocks as they are.
  for (std::size_t c = 0; c < frame.channels(); ++c) {
    GridArray<double>& grid = (*canvas)[c];
    BlocksArray<double> kept(frame.component(c).problem.extents());
    Room room = frame.room(c);
    frame.project(c, std::as_const(grid).view(), grid.view(), kept.view(), room);
  }
  Decoded& decoded = projection.decoded;
  decoded.canvas = std::move(*canvas);
  decoded.picture = picture_of(decoded.canvas, decoded.height, decoded.width, decoded.space);
  return projection;
}

Summary summary(Blocks<const double> values) noexcept {
  const std::span<const double> all = entries(values);
  const std::size_t count = all.size();
  std::size_t outside = 0;
  for (const double value : all) outside += above_zero(value);
  // The largest in eight lanes: vector code, where a largest taken one value after
  // another would not be, its order kept for NaNs.
  std::array<double, 8> lanes{};
  const std::size_t whole = count - count % lanes.size();
  for (std::size_t k = 0; k < whole; k += lanes.size()) {
    for (std::size_t lane = 0; lane < lanes.size(); ++lane) lanes[lane] = at_least(lanes[lane], all[k + lane]);
  }
  double largest = 0.0;
  for (const double lane : lanes) largest = at_least(largest, lane);
  for (std::size_t k = whole; k < count; ++k) largest = at_least(largest, all[k]);
  return Summary{.coefficients = count, .outside = outside, .largest = largest};
}

}  // namespace unround::picture
