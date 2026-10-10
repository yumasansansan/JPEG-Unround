// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The settings of a reconstruction, under the names of the command line's options
// (docs/cli.md): the method, the regularizer, the data term of every component or of
// each, the primal-dual method, the subgradient method, and the reading of the file.
#ifndef UNROUND_SETTINGS_HPP
#define UNROUND_SETTINGS_HPP

#include "unround/error.hpp"
#include "unround/model.hpp"
#include "unround/pdhg.hpp"
#include "unround/subgradient.hpp"

#include <cstdint>
#include <expected>
#include <span>
#include <string_view>
#include <vector>

namespace unround {

// The MMSE decoder (docs/math.md, 2.3), TV or TGV by the primal-dual method (5), or TV
// by the subgradient method (7).
enum class Method : std::int32_t { mmse, tv, tgv, subgradient };

// The limits of the reading of a file (docs/cli.md, Reading).
struct Reading {
  std::uint64_t max_pixels = 268435456;
  std::int32_t max_scans = 500;
  bool warnings_are_errors = false;
};

struct Settings {
  Method method = Method::tgv;
  pdhg::Regularizer regularizer;
  // One data term for every component, or one for each.
  std::vector<DataTerm> data = {DataTerm{}};
  pdhg::Options pdhg;
  subgradient::Options subgradient;
  Reading reading;
};

// The settings of the options given as the command line takes them: each option at
// most once, its value the next argument or after '=' (--mu 0.01, --mu=0.01), numbers
// in decimal; each option of the data term one value for every component or a
// comma-separated list of one for each. An option unknown, given twice or without its
// value, a value out of its range, and an option that the method does not take (a
// partial gap for TV) are refused, every one of them named at once.
[[nodiscard]] std::expected<Settings, Error> parse_settings(std::span<const std::string_view> arguments);

// The data terms of the settings for a picture: those given, with the MMSE centres
// where the method is the MMSE decoder, which takes them whatever --centres says.
[[nodiscard]] std::vector<DataTerm> data_of(const Settings& settings);

}  // namespace unround

#endif  // UNROUND_SETTINGS_HPP
