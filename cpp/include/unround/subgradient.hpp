// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// A subgradient method of jpeg2png's kind (docs/math.md, 7), to compare with: TV within
// the set by normalized subgradient steps of a length that falls as 1 / sqrt(n),
// projected onto the set, with the extrapolation of FISTA. It takes the frame of a
// greyscale picture, whose canvas is its blocks.
#ifndef UNROUND_SUBGRADIENT_HPP
#define UNROUND_SUBGRADIENT_HPP

#include "unround/arrays.hpp"
#include "unround/error.hpp"
#include "unround/frames.hpp"
#include "unround/results.hpp"

#include <cstdint>
#include <expected>
#include <functional>
#include <span>
#include <vector>

namespace unround::subgradient {

// The defaults (docs/cli.md).
inline constexpr std::uint64_t iterations = 50;
inline constexpr double step = 0.5;
inline constexpr double decay = 0.5;
inline constexpr std::uint64_t record_every = 1;

// The iterations, the step eta (above 0) and its decay beta (at least 0), whether
// FISTA's extrapolation is on, and the iterations between records (0: only after the
// last).
struct Options {
  std::uint64_t iterations = subgradient::iterations;
  double step = subgradient::step;
  double decay = subgradient::decay;
  bool momentum = true;
  std::uint64_t record_every = subgradient::record_every;
};

// The last iterate x, its coefficients (those of its projection), the records, the
// iterations taken, and why the method stopped: after the most iterations, or where
// its subgradient was 0, or where its observer asked it to.
struct Result {
  std::vector<GridArray<double>> canvas;
  std::vector<BlocksArray<double>> coefficients;
  std::vector<Record> records;
  std::uint64_t iterations = 0;
  Stop stop = Stop::iterations;
};

// Called at each record with the iterate's canvas; returning true stops the method.
using Observer = std::function<bool(const Record&, std::span<const GridArray<double>>)>;

// TV of weight alpha by the subgradient method, from the start of the solvers.
// Records are taken of the start, every record_every iterations and after the last,
// as the primal-dual method's (docs/math.md, 6.4). A frame of more than one
// component, or whose canvas is not its blocks, and options out of their ranges are
// refused.
[[nodiscard]] std::expected<Result, Error> solve(const Frame& frame, double alpha, const Options& options,
                                                 const Observer& observer = {});

}  // namespace unround::subgradient

#endif  // UNROUND_SUBGRADIENT_HPP
