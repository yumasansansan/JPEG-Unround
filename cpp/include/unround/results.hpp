// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// What the solvers record and why they stop (docs/math.md, 6.4 and 7).
#ifndef UNROUND_RESULTS_HPP
#define UNROUND_RESULTS_HPP

#include <cstdint>
#include <optional>

namespace unround {

// A record: the iteration, the primal and the dual value at the point of that
// iteration (the start at 0; the dual value of the subgradient method, which has
// none, -infinity), TGV's scaling theta of the feasible dual (6.2) and partial gap
// (6.3, where a radius is given), and the seconds since the solver began.
struct Record {
  std::uint64_t iteration = 0;
  double primal = 0.0;
  double dual = 0.0;
  std::optional<double> scaling;
  std::optional<double> partial_gap;
  double seconds = 0.0;
};

// Why a solver stopped: a tolerance was met at a record, it took the most
// iterations, its observer asked it to, or the subgradient method found a
// subgradient of 0.
enum class Stop : std::int32_t { converged, iterations, observer, stationary };

}  // namespace unround

#endif  // UNROUND_RESULTS_HPP
