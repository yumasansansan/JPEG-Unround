// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// What the files are made of (docs/math.md, 10, and docs/cli.md): the levels of samples
// of 8 or 16 bits, dithered or rounded; the picture turned upright as the file's EXIF
// orientation says; and the bytes of TIFF files of binary64 samples and of PNM files.
// A picture is rows x columns x channels, the samples of a pixel together.
#ifndef UNROUND_OUTPUT_HPP
#define UNROUND_OUTPUT_HPP

#include "unround/arrays.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace unround::output {

using PictureArray = MdArray<double, PictureExtents>;
using LevelsArray = MdArray<std::uint16_t, PictureExtents>;

// The ordered dither of a blue-noise matrix, or rounding to the nearest level.
enum class Dither : std::int32_t { none, blue_noise };

// The ranks of the blue-noise matrix, 64 x 64, row by row: each of 0 to 4095 once.
[[nodiscard]] std::span<const std::uint16_t, 4096> blue_noise(void) noexcept;

// The levels of a picture's samples in a file of 8 or 16 bits (docs/math.md, 10): of
// y = x or y = 257 x, the product exact, clamp(floor(y) + [y - floor(y) >= 1 - t], 0, T)
// with the threshold t of the pixel and the channel, or clamp(round(y), 0, T), halves
// away from 0. The channels take the matrix shifted apart, (0, 0), (0, 32), (32, 0) and
// (32, 32), or all at (0, 0). Every implementation makes the same levels of the same
// samples: each is computed exactly from x.
[[nodiscard]] LevelsArray levels_of(Picture<const double> picture, std::int32_t bits, Dither dither, bool apart);

// Whether an EXIF orientation turns a picture: 2 to 8; 1, and every other value, is the
// picture as it is stored.
[[nodiscard]] bool turns(std::int32_t orientation) noexcept;

// The orientation that turns back what an orientation turns, from the picture shown to
// the picture stored: each is its own but 6 and 8, the quarter turns, each the other's.
[[nodiscard]] std::int32_t inverse(std::int32_t orientation) noexcept;

// The picture as orientation o says to show it (docs/cli.md, Orientation): pixel
// (r, c) of the result is that of the stored picture S of h rows and w columns at
// S[r][c] (1), S[r][w-1-c] (2), S[h-1-r][w-1-c] (3), S[h-1-r][c] (4), and of w rows
// and h columns at S[c][r] (5), S[h-1-c][r] (6), S[h-1-c][w-1-r] (7), S[c][w-1-r] (8).
template <typename T>
[[nodiscard]] MdArray<T, PictureExtents> oriented(Picture<const T> picture, std::int32_t orientation) {
  const std::size_t h = picture.extent(0);
  const std::size_t w = picture.extent(1);
  const std::size_t channels = picture.extent(2);
  const bool across = orientation >= 5 && orientation <= 8;
  MdArray<T, PictureExtents> result(PictureExtents{across ? w : h, across ? h : w, channels});
  const Picture<T> out = result.view();
  const std::size_t rows = out.extent(0);
  const std::size_t columns = out.extent(1);
  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t c = 0; c < columns; ++c) {
      std::size_t i = r;
      std::size_t j = c;
      switch (orientation) {
        case 2:
          j = w - 1u - c;
          break;
        case 3:
          i = h - 1u - r;
          j = w - 1u - c;
          break;
        case 4:
          i = h - 1u - r;
          break;
        case 5:
          i = c;
          j = r;
          break;
        case 6:
          i = h - 1u - c;
          j = r;
          break;
        case 7:
          i = h - 1u - c;
          j = w - 1u - r;
          break;
        case 8:
          i = c;
          j = w - 1u - r;
          break;
        default:
          break;
      }
      for (std::size_t k = 0; k < channels; ++k) out[r, c, k] = picture[i, j, k];
    }
  }
  return result;
}

// The colour space a TIFF file declares: grey, RGB, or JFIF's YCbCr (full range, no
// subsampling); with alpha, the last channel is unassociated alpha (ExtraSamples 2).
enum class Photometric : std::int32_t { grey, rgb, ycbcr };

// A TIFF 6.0 file of the picture's binary64 samples, little-endian, in one strip,
// neither rounded nor clamped, with an ICC profile where one is given
// (InterColorProfile, 34675). Pictures of 4 GiB or more do not fit, and give no bytes.
[[nodiscard]] std::vector<std::uint8_t> tiff_bytes(Picture<const double> picture, Photometric photometric, bool alpha,
                                                   std::span<const std::uint8_t> icc_profile);

// A PNM file of the levels: P5 (one channel) or P6 (three), of 8 or 16 bits, the
// latter's samples big-endian.
[[nodiscard]] std::vector<std::uint8_t> pnm_bytes(Picture<const std::uint16_t> levels, std::int32_t bits);

}  // namespace unround::output

#endif  // UNROUND_OUTPUT_HPP
