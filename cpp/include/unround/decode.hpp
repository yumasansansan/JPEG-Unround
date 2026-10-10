// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The reconstruction of a JPEG file: its components as the C layer reads them, the
// frame of its picture (docs/math.md, 1.3), the method of the settings, and the result
// in binary64, the canvas itself cut to the picture, or for a file in YCbCr the RGB of
// JFIF's conversion (docs/math.md, 8).
#ifndef UNROUND_DECODE_HPP
#define UNROUND_DECODE_HPP

#include "unround/arrays.hpp"
#include "unround/error.hpp"
#include "unround/frames.hpp"
#include "unround/pdhg.hpp"
#include "unround/results.hpp"
#include "unround/settings.hpp"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace unround {

// A picture of rows x columns x channels samples, those of a pixel together.
using PictureArray = MdArray<double, PictureExtents>;

struct Decoded {
  std::size_t height = 0;
  std::size_t width = 0;
  ColourSpace space = ColourSpace::grayscale;
  // The solution on its canvas, channel by channel: Y, Cb and Cr for a file in YCbCr.
  std::vector<GridArray<double>> canvas;
  // The result: the canvas cut to the picture, and for a file in YCbCr its RGB.
  PictureArray picture;
  // The solver's records, iterations and stop (none for the MMSE decoder), and the
  // steps of the primal-dual method.
  std::vector<Record> records;
  std::uint64_t iterations = 0;
  Stop stop = Stop::iterations;
  std::optional<pdhg::Steps> steps;
  // The file's EXIF orientation (0 where it has none), its ICC profile, and the first
  // corrupt-data warning of the reading, if any.
  std::int32_t exif_orientation = 0;
  std::vector<std::uint8_t> icc_profile;
  std::string warning;
};

// Called at each record with the canvas of its point; returning true stops the solver.
using Observer = pdhg::Observer;

// The reconstruction of a JPEG file with the settings. A file that is not read is
// refused (Errc::read), one of a layout that is not taken (Errc::unsupported), and
// settings that the file's components do not take (Errc::options).
[[nodiscard]] std::expected<Decoded, Error> decode(std::span<const std::uint8_t> file, const Settings& settings,
                                                   const Observer& observer = {});

// The picture of a canvas: each channel cut to height x width, and for YCbCr the RGB
// of JFIF's conversion.
[[nodiscard]] PictureArray picture_of(std::span<const GridArray<double>> canvas, std::size_t height, std::size_t width,
                                      ColourSpace space);

}  // namespace unround

#endif  // UNROUND_DECODE_HPP
