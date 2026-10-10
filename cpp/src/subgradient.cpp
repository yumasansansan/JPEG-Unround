// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The subgradient method of docs/math.md, 7.
//
// Every step is one loop over the samples of the canvas, or over the 64 coefficients
// of a block, the same operation on neighbouring values: vector code.

#include "unround/subgradient.hpp"

#include "unround/arrays.hpp"
#include "unround/dct.hpp"
#include "unround/error.hpp"
#include "unround/exact.hpp"
#include "unround/frames.hpp"
#include "unround/model.hpp"
#include "unround/operators.hpp"
#include "unround/results.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace unround::subgradient {

namespace {

bool positive(double value) noexcept { return value > 0.0 && std::isfinite(value); }
bool at_least_zero(double value) noexcept { return value >= 0.0 && std::isfinite(value); }

std::unexpected<Error> refuse(std::string message) {
  return std::unexpected(Error{.code = Errc::options, .message = std::move(message)});
}

// The field of unit vectors along the gradient, (x, y) / |(x, y)|, and 0 where the
// gradient is 0.
void normalized(std::span<double> x, std::span<double> y) noexcept {
  double* across = x.data();
  double* down = y.data();
  const std::size_t count = x.size();
  for (std::size_t k = 0; k < count; ++k) {
    const double norm = std::sqrt(across[k] * across[k] + down[k] * down[k]);
    const bool moving = norm > 0.0;
    across[k] = moving ? across[k] / norm : 0.0;
    down[k] = moving ? down[k] / norm : 0.0;
  }
}

// out = b - alpha d.
void less_scaled(std::span<const double> b, double alpha, std::span<const double> d, std::span<double> out) noexcept {
  const double* from = b.data();
  const double* along = d.data();
  double* to = out.data();
  const std::size_t count = out.size();
  for (std::size_t k = 0; k < count; ++k) to[k] = from[k] - alpha * along[k];
}

// out = y - a g.
void stepped(std::span<const double> y, double a, std::span<const double> g, std::span<double> out) noexcept {
  const double* from = y.data();
  const double* along = g.data();
  double* to = out.data();
  const std::size_t count = out.size();
  for (std::size_t k = 0; k < count; ++k) to[k] = from[k] - a * along[k];
}

// y = x + a (x - previous).
void extrapolated(std::span<const double> x, std::span<const double> previous, double a, std::span<double> y) noexcept {
  const double* now = x.data();
  const double* before = previous.data();
  double* to = y.data();
  const std::size_t count = y.size();
  for (std::size_t k = 0; k < count; ++k) to[k] = now[k] + a * (now[k] - before[k]);
}

// The squares of the entries, into terms.
void squares_of(std::span<const double> values, std::span<double> terms) noexcept {
  const double* from = values.data();
  double* to = terms.data();
  const std::size_t count = terms.size();
  for (std::size_t k = 0; k < count; ++k) to[k] = from[k] * from[k];
}

// The norms of the pixels of a gradient, into terms.
void norms_of(std::span<const double> x, std::span<const double> y, std::span<double> terms) noexcept {
  const double* across = x.data();
  const double* down = y.data();
  double* to = terms.data();
  const std::size_t count = terms.size();
  for (std::size_t k = 0; k < count; ++k) to[k] = std::sqrt(across[k] * across[k] + down[k] * down[k]);
}

// The data term's part of the subgradient of a block, m (e - centre) + lambda s, s the
// side of the file's own interval that e lies beyond (0 within it).
void data_part(std::span<const double, 64> e, std::span<const double, 64> centres, std::span<const double, 64> weights,
               std::span<double, 64> out) noexcept {
  for (std::size_t k = 0; k < 64u; ++k) out[k] = weights[k] * (e[k] - centres[k]);
}

void cost_part(std::span<const double, 64> e, std::span<const double, 64> lows, std::span<const double, 64> highs,
               const std::array<double, 64>& costs, std::span<double, 64> out) noexcept {
  for (std::size_t k = 0; k < 64u; ++k) {
    const double side = static_cast<double>(e[k] > highs[k]) - static_cast<double>(e[k] < lows[k]);
    out[k] += costs[k] * side;
  }
}

class Machine {
 public:
  Machine(const Frame& frame, double alpha)
      : frame_(frame),
        problem_(frame.component(0).problem),
        alpha_(alpha),
        shape_(frame.canvas_extents()),
        x_(shape_),
        previous_(shape_),
        y_(shape_),
        v_(shape_),
        g_(shape_),
        back_(shape_),
        divergence_(shape_),
        terms_(shape_),
        gradient_(operators::vector_of(frame.rows(), frame.columns())),
        coefficients_(problem_.extents()),
        at_(problem_.extents()),
        parts_(problem_.extents()),
        room_(frame.room(0)) {
    frame.start(0, x_.view(), coefficients_.view());
    std::ranges::copy(std::as_const(x_).entries(), y_.entries().begin());
  }

  // The subgradient at y, into g; its norm.
  double subgradient(void) {
    operators::gradient(std::as_const(y_).view(), gradient_.view());
    normalized(gradient_.x.entries(), gradient_.y.entries());
    operators::divergence(std::as_const(gradient_).view(), divergence_.view());
    frame_.coefficients(0, std::as_const(y_).view(), at_.view());
    const Blocks<const double> e = std::as_const(at_).view();
    const Blocks<const double> centres = problem_.centres();
    const Blocks<double> parts = parts_.view();
    const std::optional<Cost>& cost = problem_.cost();
    const std::size_t rows = problem_.rows();
    const std::size_t columns = problem_.columns();
    for (std::size_t by = 0; by < rows; ++by) {
      for (std::size_t bx = 0; bx < columns; ++bx) {
        const Block8x8<const double> weights = problem_.block_weights(by, bx);
        const std::span<const double, 64> table(weights.data_handle(), 64);
        data_part(block_entries(e, by, bx), block_entries(centres, by, bx), table, block_entries(parts, by, bx));
        if (cost) {
          cost_part(block_entries(e, by, bx), block_entries(cost->inner_lower.view(), by, bx),
                    block_entries(cost->inner_upper.view(), by, bx), cost->costs, block_entries(parts, by, bx));
        }
      }
    }
    dct::inverse(std::as_const(parts_).view(), back_.view());
    less_scaled(std::as_const(back_).entries(), alpha_, std::as_const(divergence_).entries(), g_.entries());
    squares_of(std::as_const(g_).entries(), terms_.entries());
    return std::sqrt(exact::sum_rows(std::as_const(terms_).view()));
  }

  // x+ = P_C(y - (h / |g|) g), and y moved on from x+ (with FISTA's extrapolation, by
  // (t - 1) / t+).
  void step(double length, double norm, bool momentum) {
    stepped(std::as_const(y_).entries(), length / norm, std::as_const(g_).entries(), v_.entries());
    std::swap(previous_, x_);
    frame_.project(0, std::as_const(v_).view(), x_.view(), coefficients_.view(), room_);
    if (momentum) {
      const double next = (1.0 + std::sqrt(1.0 + 4.0 * (t_ * t_))) / 2.0;
      extrapolated(std::as_const(x_).entries(), std::as_const(previous_).entries(), (t_ - 1.0) / next, y_.entries());
      t_ = next;
    } else {
      std::ranges::copy(std::as_const(x_).entries(), y_.entries().begin());
    }
  }

  // The primal value of x: alpha TV(x) + G at its coefficients.
  Record record(std::uint64_t iteration) {
    operators::gradient(std::as_const(x_).view(), gradient_.view());
    norms_of(gradient_.x.entries(), gradient_.y.entries(), terms_.entries());
    const double variation = alpha_ * exact::sum_rows(std::as_const(terms_).view());
    Record result;
    result.iteration = iteration;
    result.primal = variation + problem_.value(std::as_const(coefficients_).view());
    result.dual = -std::numeric_limits<double>::infinity();
    return result;
  }

  [[nodiscard]] std::span<const GridArray<double>> canvas(void) const noexcept {
    return std::span<const GridArray<double>>(&x_, 1);
  }
  GridArray<double> take_canvas(void) { return std::move(x_); }
  BlocksArray<double> take_coefficients(void) { return std::move(coefficients_); }

 private:
  const Frame& frame_;
  const Problem& problem_;
  double alpha_;
  GridExtents shape_;
  GridArray<double> x_;
  GridArray<double> previous_;
  GridArray<double> y_;
  GridArray<double> v_;
  GridArray<double> g_;
  GridArray<double> back_;
  GridArray<double> divergence_;
  GridArray<double> terms_;
  operators::Vector gradient_;
  BlocksArray<double> coefficients_;
  BlocksArray<double> at_;
  BlocksArray<double> parts_;
  Room room_;
  double t_ = 1.0;
};

}  // namespace

std::expected<Result, Error> solve(const Frame& frame, double alpha, const Options& options, const Observer& observer) {
  const auto began = std::chrono::steady_clock::now();
  if (frame.channels() != 1u || frame.free()) {
    return std::unexpected(Error{.code = Errc::unsupported,
                                 .message = "the subgradient method takes greyscale pictures, whose canvas is their "
                                            "blocks"});
  }
  if (!positive(alpha) || !positive(options.step) || !at_least_zero(options.decay)) {
    return refuse(
        std::format("alpha and the step are above 0 and finite, and the decay at least 0 and finite, not "
                    "{}, {} and {}",
                    alpha, options.step, options.decay));
  }
  Machine machine(frame, alpha);
  Result result;
  const double root_samples = std::sqrt(static_cast<double>(frame.rows()) * static_cast<double>(frame.columns()));
  const std::uint64_t most = options.iterations;
  const std::uint64_t every = options.record_every;
  const auto take = [&](std::uint64_t iteration) -> bool {
    Record record = machine.record(iteration);
    record.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count();
    result.records.push_back(record);
    return observer && observer(result.records.back(), machine.canvas());
  };
  bool stopped = false;
  if (every > 0u || most == 0u) {
    if (take(0)) {
      result.stop = Stop::observer;
      stopped = true;
    }
  }
  std::uint64_t iteration = 0;
  while (!stopped && iteration < most) {
    const double norm = machine.subgradient();
    if (norm == 0.0) {
      result.stop = Stop::stationary;
      // The last record is of the iterate where the method stopped.
      if (result.records.empty() || result.records.back().iteration != iteration) (void)take(iteration);
      break;
    }
    const double length = options.step * root_samples / std::pow(1.0 + static_cast<double>(iteration), options.decay);
    machine.step(length, norm, options.momentum);
    ++iteration;
    if ((every > 0u && iteration % every == 0u) || iteration == most) {
      if (take(iteration)) {
        result.stop = Stop::observer;
        stopped = true;
      }
    }
  }
  result.iterations = iteration;
  result.canvas.push_back(machine.take_canvas());
  result.coefficients.push_back(machine.take_coefficients());
  return result;
}

}  // namespace unround::subgradient
