// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// A picture of a file's size made elsewhere -- by another decoder, say -- measured
// against the file's intervals, and projected onto them (docs/math.md, 12).
//
// Its canvas is made as an encoder makes the file's: of a file in YCbCr, the Y, Cb and
// Cr of JFIF's conversion of its RGB; of a greyscale or RGB file, its own channels;
// each channel extended over the canvas's whole MCUs by repeating its last column and
// then its last row. The picture is taken as the file stores it, before its EXIF
// orientation.
#ifndef UNROUND_PICTURE_HPP
#define UNROUND_PICTURE_HPP

#include "unround/arrays.hpp"
#include "unround/decode.hpp"
#include "unround/error.hpp"
#include "unround/frames.hpp"

#include <cstddef>
#include <expected>
#include <span>
#include <vector>

namespace unround::picture {

// A picture projected onto a file's intervals: the projection, as a reconstruction
// without a solver (its canvas and picture; no records), and how far each coefficient
// of the picture given lay outside its interval, in steps (0 within it), component by
// component.
struct Projection {
  Decoded decoded;
  std::vector<BlocksArray<double>> excess;
};

// How much of a component lies outside its intervals: its coefficients, how many of
// them lie outside, and the largest excess, in steps (0 where none does).
struct Summary {
  std::size_t coefficients = 0;
  std::size_t outside = 0;
  double largest = 0.0;
};

// The samples to a pixel of a picture of a frame: 1 of a greyscale file, 3 (RGB) of a
// file in colour.
[[nodiscard]] std::size_t channels_of(const Frame& frame) noexcept;

// The canvas of a picture of the frame's size and channels, as the file stores it,
// channel by channel. A picture of another size or of other channels, or of samples
// that are not all finite, is refused (Errc::options).
[[nodiscard]] std::expected<std::vector<GridArray<double>>, Error> canvas_of(const Frame& frame,
                                                                             Picture<const double> picture);

// How far each coefficient of a canvas lies outside its interval, in steps:
// max(0, a - A x, A x - b) / Q, component by component.
[[nodiscard]] std::vector<BlocksArray<double>> excess(const Frame& frame, std::span<const GridArray<double>> canvas);

// The excess of a picture of a file read, with the slack of the settings that it was
// read with; refused as canvas_of refuses it.
[[nodiscard]] std::expected<std::vector<BlocksArray<double>>, Error> excess(const Source& source,
                                                                            Picture<const double> picture);

// The projection of a picture of a file read onto its intervals, with the slack of the
// settings that it was read with, and the picture's excess; refused as canvas_of
// refuses it.
[[nodiscard]] std::expected<Projection, Error> project(const Source& source, Picture<const double> picture);

// The summary of a component's excess, whose values are 0 or more.
[[nodiscard]] Summary summary(Blocks<const double> values) noexcept;

}  // namespace unround::picture

#endif  // UNROUND_PICTURE_HPP
