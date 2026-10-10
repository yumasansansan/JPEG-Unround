// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The conversions of docs/math.md, 8: one loop over the samples each, vector code.

#include "unround/colour.hpp"

#include <cstddef>
#include <span>

namespace unround::colour {

void to_rgb(std::span<const double> y, std::span<const double> cb, std::span<const double> cr, std::span<double> r,
            std::span<double> g, std::span<double> b) noexcept {
  const double* luma = y.data();
  const double* blue_difference = cb.data();
  const double* red_difference = cr.data();
  double* red = r.data();
  double* green = g.data();
  double* blue = b.data();
  const std::size_t count = y.size();
  for (std::size_t k = 0; k < count; ++k) {
    const double from_cb = blue_difference[k] - 128.0;
    const double from_cr = red_difference[k] - 128.0;
    red[k] = luma[k] + red_from_cr * from_cr;
    blue[k] = luma[k] + blue_from_cb * from_cb;
    const double partial = luma[k] - green_from_cb * from_cb;
    green[k] = partial - green_from_cr * from_cr;
  }
}

void to_ycbcr(std::span<const double> r, std::span<const double> g, std::span<const double> b, std::span<double> y,
              std::span<double> cb, std::span<double> cr) noexcept {
  const double* red = r.data();
  const double* green = g.data();
  const double* blue = b.data();
  double* luma = y.data();
  double* blue_difference = cb.data();
  double* red_difference = cr.data();
  const std::size_t count = r.size();
  for (std::size_t k = 0; k < count; ++k) {
    const double partial = luma_red * red[k] + luma_green * green[k];
    const double value = partial + luma_blue * blue[k];
    luma[k] = value;
    blue_difference[k] = 128.0 + cb_from_blue * (blue[k] - value);
    red_difference[k] = 128.0 + cr_from_red * (red[k] - value);
  }
}

}  // namespace unround::colour
