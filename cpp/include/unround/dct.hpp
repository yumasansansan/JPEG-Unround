// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The orthonormal 8x8 DCT-II of JPEG and its inverse (docs/math.md, 1.1): the
// basis from nine cosines written to 25 digits, and the 8-point transforms by
// their even and odd halves, down the columns and along the rows of a block.
#ifndef UNROUND_DCT_HPP
#define UNROUND_DCT_HPP

#include <array>
#include <cstddef>
#include <span>
#include <vector>

namespace unround::dct {

inline constexpr std::size_t block = 8;
inline constexpr std::size_t block_size = 64;

// A block: 8 rows of 8 samples, or its coefficients in natural order, entry
// 8 v + u of the vertical frequency v and the horizontal u.
using Block = std::array<double, block_size>;

// The basis, C[k][n] = gamma_k cos(pi (2 n + 1) k / 16): every entry the double
// nearest to its value.
using Basis = std::array<std::array<double, block>, block>;
[[nodiscard]] const Basis& basis(void) noexcept;

// The 8-point transform y = C x of eight values `stride` apart, and its inverse
// x = C^T y.
void forward8(const double* in, std::size_t in_stride, double* out, std::size_t out_stride) noexcept;
void inverse8(const double* in, std::size_t in_stride, double* out, std::size_t out_stride) noexcept;

// A block's DCT, down its columns and then along its rows, and the inverse,
// along the rows and then down the columns.
[[nodiscard]] Block forward(const Block& samples) noexcept;
[[nodiscard]] Block inverse(const Block& coefficients) noexcept;

// The blocks of a canvas: `rows` x `columns` blocks at its top left, of a canvas
// `width` samples wide (at least 8 columns), row by row. The coefficients are
// block by block, block rows from the top, each block's in natural order.
[[nodiscard]] std::vector<double> forward(std::span<const double> canvas, std::size_t width, std::size_t rows,
                                          std::size_t columns);
// The samples of the blocks of `coefficients`, written into the top left of a
// canvas `width` samples wide; the rest of it is left as it is.
void inverse(std::span<const double> coefficients, std::size_t rows, std::size_t columns, std::span<double> canvas,
             std::size_t width);

}  // namespace unround::dct

#endif  // UNROUND_DCT_HPP
