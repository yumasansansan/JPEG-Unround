// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The conversions of JFIF between YCbCr and RGB (docs/math.md, 8), in binary64, with
// the doubles nearest to their rationals, sample by sample.
#ifndef UNROUND_COLOUR_HPP
#define UNROUND_COLOUR_HPP

#include <span>

namespace unround::colour {

// The inverse of JFIF's YCbCr: R = Y + c_R (Cr - 128), B = Y + c_B (Cb - 128),
// G = (Y - c_GB (Cb - 128)) - c_GR (Cr - 128), with the doubles nearest to 701/500,
// 443/250, 25251/73375 and 209599/293500 (each literal to 25 digits).
inline constexpr double red_from_cr = 1.402;
inline constexpr double blue_from_cb = 1.772;
inline constexpr double green_from_cb = 0.3441362862010221465076661;
inline constexpr double green_from_cr = 0.7141362862010221465076661;

// JFIF's YCbCr: Y = (K_R R + K_G G) + K_B B, Cb = 128 + k_Cb (B - Y), Cr = 128 + k_Cr
// (R - Y), with the doubles nearest to 0.299, 0.587, 0.114, 250/443 and 500/701.
inline constexpr double luma_red = 0.299;
inline constexpr double luma_green = 0.587;
inline constexpr double luma_blue = 0.114;
inline constexpr double cb_from_blue = 0.5643340857787810383747178;
inline constexpr double cr_from_red = 0.7132667617689015691868759;

// R, G and B of Y, Cb and Cr, each a plane of the same samples. Each product may be
// fused with the sum or difference that takes it.
void to_rgb(std::span<const double> y, std::span<const double> cb, std::span<const double> cr, std::span<double> r,
            std::span<double> g, std::span<double> b) noexcept;

// Y, Cb and Cr of R, G and B.
void to_ycbcr(std::span<const double> r, std::span<const double> g, std::span<const double> b, std::span<double> y,
              std::span<double> cb, std::span<double> cr) noexcept;

}  // namespace unround::colour

#endif  // UNROUND_COLOUR_HPP
