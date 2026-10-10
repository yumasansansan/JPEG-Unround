// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The reconstruction of a JPEG file, from the C layer's reading to the result.

#include "unround/decode.hpp"

#include "unround/arrays.hpp"
#include "unround/colour.hpp"
#include "unround/error.hpp"
#include "unround/frames.hpp"
#include "unround/jpegio.hpp"
#include "unround/pdhg.hpp"
#include "unround/results.hpp"
#include "unround/settings.hpp"
#include "unround/subgradient.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace unround {

namespace {

Errc errc_of(jpegio::Errc code) noexcept {
  switch (code) {
    case jpegio::Errc::unsupported:
    case jpegio::Errc::limit:
      return Errc::unsupported;
    case jpegio::Errc::argument:
    case jpegio::Errc::decode:
      return Errc::read;
    case jpegio::Errc::memory:
    case jpegio::Errc::encode:
      return Errc::internal;
  }
  return Errc::internal;
}

ColourSpace space_of(jpegio::ColorSpace space) noexcept {
  switch (space) {
    case jpegio::ColorSpace::grayscale:
      return ColourSpace::grayscale;
    case jpegio::ColorSpace::ycbcr:
      return ColourSpace::ycbcr;
    case jpegio::ColorSpace::rgb:
      return ColourSpace::rgb;
  }
  return ColourSpace::grayscale;
}

// The samples of rows x columns of a grid, row by row, into a plane.
void cut(Grid<const double> grid, std::size_t rows, std::size_t columns, std::span<double> plane) noexcept {
  for (std::size_t i = 0; i < rows; ++i) {
    const std::span<const double> row(&grid[i, 0], columns);
    std::ranges::copy(row, plane.begin() + static_cast<std::ptrdiff_t>(i * columns));
  }
}

}  // namespace

PictureArray picture_of(std::span<const GridArray<double>> canvas, std::size_t height, std::size_t width,
                        ColourSpace space) {
  const std::size_t channels = canvas.size();
  const std::size_t count = height * width;
  std::vector<std::vector<double>> planes(channels, std::vector<double>(count));
  for (std::size_t c = 0; c < channels; ++c) cut(canvas[c].view(), height, width, planes[c]);
  if (space == ColourSpace::ycbcr && channels == 3u) {
    std::vector<std::vector<double>> rgb(3, std::vector<double>(count));
    colour::to_rgb(planes[0], planes[1], planes[2], rgb[0], rgb[1], rgb[2]);
    planes = std::move(rgb);
  }
  PictureArray picture(PictureExtents{height, width, channels});
  const Picture<double> view = picture.view();
  for (std::size_t c = 0; c < channels; ++c) {
    const std::span<const double> plane = planes[c];
    for (std::size_t i = 0; i < height; ++i) {
      for (std::size_t j = 0; j < width; ++j) view[i, j, c] = plane[i * width + j];
    }
  }
  return picture;
}

std::expected<Source, Error> read_source(std::span<const std::uint8_t> file, const Settings& settings) {
  const jpegio::Options reading{.max_pixels = settings.reading.max_pixels,
                                .max_scans = settings.reading.max_scans,
                                .warnings_are_errors = settings.reading.warnings_are_errors};
  std::expected<jpegio::Image, jpegio::Error> image = jpegio::Image::read(file, reading);
  if (!image) return std::unexpected(Error{.code = errc_of(image.error().code), .message = image.error().message});
  Decoded decoded;
  decoded.height = image->height();
  decoded.width = image->width();
  decoded.space = space_of(image->color_space());
  decoded.exif_orientation = image->exif_orientation();
  const std::span<const std::uint8_t> profile = image->icc_profile();
  decoded.icc_profile.assign(profile.begin(), profile.end());
  decoded.warning = image->warning();
  std::vector<ComponentInput> inputs;
  for (const unround_jpegio_component& component : image->components()) {
    ComponentInput input{.levels = jpegio::coefficients(component),
                         .h = static_cast<std::uint32_t>(component.h_samp_factor),
                         .v = static_cast<std::uint32_t>(component.v_samp_factor)};
    std::ranges::copy(jpegio::quant_table(component), input.table.begin());
    inputs.push_back(input);
  }
  const std::vector<DataTerm> data = data_of(settings);
  std::expected<Frame, Error> frame = Frame::make(decoded.height, decoded.width, decoded.space, inputs, data);
  if (!frame) return std::unexpected(frame.error());
  return Source{.decoded = std::move(decoded), .frame = std::move(*frame)};
}

std::expected<Decoded, Error> decode(std::span<const std::uint8_t> file, const Settings& settings,
                                     const Observer& observer) {
  std::expected<Source, Error> source = read_source(file, settings);
  if (!source) return std::unexpected(source.error());
  Decoded& decoded = source->decoded;
  const Frame& frame = source->frame;
  switch (settings.method) {
    case Method::mmse:
      for (std::size_t c = 0; c < frame.channels(); ++c) {
        GridArray<double> x(frame.canvas_extents());
        BlocksArray<double> kept(frame.component(c).problem.extents());
        frame.start(c, x.view(), kept.view());
        decoded.canvas.push_back(std::move(x));
      }
      break;
    case Method::tv:
    case Method::tgv: {
      const pdhg::Model model = settings.method == Method::tv ? pdhg::Model::tv : pdhg::Model::tgv;
      std::expected<pdhg::Result, Error> result =
          pdhg::solve(frame, model, settings.regularizer, settings.pdhg, nullptr, observer);
      if (!result) return std::unexpected(result.error());
      decoded.canvas = std::move(result->canvas);
      decoded.records = std::move(result->records);
      decoded.iterations = result->iterations;
      decoded.stop = result->stop;
      decoded.steps = result->steps;
      break;
    }
    case Method::subgradient: {
      std::expected<subgradient::Result, Error> result =
          subgradient::solve(frame, settings.regularizer.alpha, settings.subgradient, observer);
      if (!result) return std::unexpected(result.error());
      decoded.canvas = std::move(result->canvas);
      decoded.records = std::move(result->records);
      decoded.iterations = result->iterations;
      decoded.stop = result->stop;
      break;
    }
  }
  decoded.picture = picture_of(decoded.canvas, decoded.height, decoded.width, decoded.space);
  return std::move(decoded);
}

}  // namespace unround
