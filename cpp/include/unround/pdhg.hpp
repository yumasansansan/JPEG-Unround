// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The primal-dual hybrid gradient method of docs/math.md, 5, for TV and TGV on the
// canvas of a Frame, channel by channel, with the gaps of 6 that tell when it stops.
// The fields are in the natural layout: each channel a Grid of the canvas.
#ifndef UNROUND_PDHG_HPP
#define UNROUND_PDHG_HPP

#include "unround/arrays.hpp"
#include "unround/error.hpp"
#include "unround/frames.hpp"
#include "unround/operators.hpp"
#include "unround/results.hpp"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <optional>
#include <span>
#include <vector>

namespace unround::pdhg {

enum class Model : std::int32_t { tv, tgv };

// The regularizer (4.2 to 4.4): TV's weight alpha, TGV's alpha1 and alpha0, the weight
// gamma of each channel's differences (none given: 1 for each), and whether the
// channels are taken together, pixel by pixel.
struct Regularizer {
  double alpha = 1.0;
  double alpha1 = 1.0;
  double alpha0 = 2.0;
  std::vector<double> channel_weights;
  bool coupled = true;
};

// The defaults of 6.5, for a weight of 1.
inline constexpr double tv_ratio = 10.0;
inline constexpr double tv_tolerance = 2e-4;
inline constexpr std::uint64_t tv_iterations = 5000;
inline constexpr double tgv_ratio = 3.0;
inline constexpr double tgv_tolerance = 2e-3;
inline constexpr std::uint64_t tgv_iterations = 10000;
inline constexpr double relaxation = 1.9;
inline constexpr double step_product = 0.99;
inline constexpr std::uint64_t record_every = 10;
inline constexpr double free_radius = 255.0;

// The options of the method and of its stop (5, 6.4). A value not given takes its
// default: the ratio of the steps divided by the first-order weight squared, and the
// tolerance multiplied by it, unless weight_scaling is off; L^2 the bound of 3.3 times
// the largest channel weight squared. A tolerance of 0 turns its test off; the partial
// gap needs a radius, and is TGV's alone.
struct Options {
  std::optional<std::uint64_t> iterations;
  std::optional<double> tolerance;
  double relative_tolerance = 0.0;
  double partial_tolerance = 0.0;
  std::optional<double> partial_radius;
  std::optional<double> step_ratio;
  double relaxation = pdhg::relaxation;
  double step_product = pdhg::step_product;
  std::optional<double> norm_squared;
  bool weight_scaling = true;
  // Iterations between records; 0: only the last.
  std::uint64_t record_every = pdhg::record_every;
  // R of the box of the free samples (6.6).
  double free_radius = pdhg::free_radius;
};

// The steps, tau and sigma, of the options' ratio, product and L^2: tau = sqrt(kappa
// theta / L^2) and sigma = sqrt(theta / (kappa L^2)), each in that order, correctly
// rounded (docs/math.md, 9.2).
struct Steps {
  double tau = 0.0;
  double sigma = 0.0;
};

// What the options come to for a frame: the steps, the most iterations, the
// tolerances, and the records.
struct Plan {
  Steps steps;
  std::uint64_t iterations = 0;
  double tolerance = 0.0;
  double relative_tolerance = 0.0;
  double partial_tolerance = 0.0;
  std::optional<double> partial_radius;
  double relaxation = 0.0;
  std::uint64_t record_every = 0;
  double free_radius = 0.0;
  std::vector<double> channel_weights;
};

// The plan of the options, or why they are refused: values out of their ranges, a
// partial gap for TV, channel weights not one for each channel.
[[nodiscard]] std::expected<Plan, Error> plan(Model model, const Regularizer& regularizer, const Options& options,
                                              std::size_t channels);

// A record (6) at the tilde point of its iteration (the start at 0), and why the
// solver stopped: a tolerance met at a record, the most iterations taken, or its
// observer (results.hpp).
using unround::Record;
using unround::Stop;

// A state to start from instead of the default one (5): the coefficients of each
// component, which are clipped to their intervals and give x; and the fields of each
// channel, which are projected onto their balls (none given: 0, and TGV's w the
// gradient of x).
struct Start {
  std::vector<BlocksArray<double>> coefficients;
  std::vector<operators::Vector> p;
  std::vector<operators::Vector> w;
  std::vector<operators::Tensor> r;
};

// The result: the last tilde point's canvas, channel by channel, and its coefficients
// zeta; the records; the iterations taken; why the solver stopped; and the steps.
struct Result {
  std::vector<GridArray<double>> canvas;
  std::vector<BlocksArray<double>> coefficients;
  std::vector<Record> records;
  std::uint64_t iterations = 0;
  Stop stop = Stop::iterations;
  Steps steps;
};

// Called at each record with the canvas of its tilde point; returning true stops the
// solver there.
using Observer = std::function<bool(const Record&, std::span<const GridArray<double>>)>;

// Solves TV or TGV on the frame. Records are taken at the start, every record_every
// iterations, and at the last; the solver stops at the first record where a tolerance
// is met, or after the most iterations.
[[nodiscard]] std::expected<Result, Error> solve(const Frame& frame, Model model, const Regularizer& regularizer,
                                                 const Options& options, const Start* start = nullptr,
                                                 const Observer& observer = {});

}  // namespace unround::pdhg

#endif  // UNROUND_PDHG_HPP
