// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The orthonormal 8x8 DCT-II of JPEG and its inverse (docs/math.md, 1.1): the
// basis from nine cosines written to 25 digits, and the 8-point transforms by
// their even and odd halves, down the columns and along the rows of a block.
#ifndef UNROUND_DCT_HPP
#define UNROUND_DCT_HPP

#include "unround/arrays.hpp"

#include <array>
#include <cstddef>

namespace unround::dct {

inline constexpr std::size_t block = 8;
inline constexpr std::size_t block_size = 64;

// A block: 8 rows of 8 samples, or its coefficients in natural order, entry
// 8 v + u of the vertical frequency v and the horizontal u. Block8x8 views it as
// [v, u] (arrays.hpp).
using Block = std::array<double, block_size>;

// The basis, C[k][n] = gamma_k cos(pi (2 n + 1) k / 16): every entry the double
// nearest to its value.
using Basis = std::array<std::array<double, block>, block>;
[[nodiscard]] const Basis& basis(void) noexcept;

// A block's DCT, down its columns and then along its rows, and the inverse,
// along the rows and then down the columns.
[[nodiscard]] Block forward(const Block& samples) noexcept;
[[nodiscard]] Block inverse(const Block& coefficients) noexcept;

// The DCT of the blocks at the top left of a canvas, as many as `coefficients`
// holds: block (by, bx) is the canvas's samples 8 by to 8 by + 7 down and 8 bx to
// 8 bx + 7 across, and the canvas has at least those. Each block as the block's
// DCT above, operation for operation.
void forward(Grid<const double> canvas, Blocks<double> coefficients) noexcept;
// The samples of the blocks of `coefficients`, written into the top left of a
// canvas; the rest of it is left as it is.
void inverse(Blocks<const double> coefficients, Grid<double> canvas) noexcept;

}  // namespace unround::dct

#endif  // UNROUND_DCT_HPP
