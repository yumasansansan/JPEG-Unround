// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The model of one component: docs/math.md, 1.2 and 4.1.

#include "unround/model.hpp"

#include "unround/exact.hpp"
#include "unround/laplace.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace unround {

namespace {

bool at_least_zero(double value) noexcept { return value >= 0.0 && std::isfinite(value); }

std::unexpected<Error> refuse(std::string message) {
  return std::unexpected(Error{.code = Errc::options, .message = std::move(message)});
}

// Q^p, exact for the powers 1 and 2 of steps of 16 bits.
double power_of(double step, double power) noexcept {
  if (power == 2.0) return step * step;
  if (power == 1.0) return step;
  return std::pow(step, power);
}

// An end of an interval: ((q + half) + widen) Q, with the shift of DC added last.
// Without slack every step of it is exact; with slack it rounds, and the product and
// the sum are separate statements so that they are not fused.
double interval_end(double level, double half, double widen, double step, double shift) noexcept {
  const double widened = (level + half) + widen;
  const double scaled = widened * step;
  return scaled + shift;
}

double clamp_to(double value, double low, double high) noexcept { return std::min(std::max(value, low), high); }

}  // namespace

DataTerm DataTerm::of_chroma(void) const noexcept {
  if (mu) return *this;
  DataTerm chroma = *this;
  chroma.mu_scale = mu_scale * mu_chroma;
  return chroma;
}

double rule_mu(const DataTerm& data, const std::array<std::uint16_t, 64>& table) noexcept {
  if (data.mu) return *data.mu;
  std::uint32_t sum = 0;
  for (const std::uint16_t step : table) sum += step;
  // The sum is an integer below 2^22, exact in a double, and dividing it by 64 is exact.
  const double mean = static_cast<double>(sum) / 64.0;
  if (data.mu_power == 1.0) return data.mu_scale * mean;
  return data.mu_scale * std::pow(mean, data.mu_power);
}

std::expected<Problem, Error> Problem::make(std::span<const std::int16_t> levels, std::size_t rows, std::size_t columns,
                                            const std::array<std::uint16_t, 64>& table, const DataTerm& data,
                                            const std::array<double, 64>* scale) {
  if (!(at_least_zero(data.mu_scale) && at_least_zero(data.mu_chroma) && std::isfinite(data.mu_power))) {
    return refuse(std::format(
        "the scale of mu and its factor in the chroma are at least 0 and finite, and its power finite, not {}, {} "
        "and {}",
        data.mu_scale, data.mu_chroma, data.mu_power));
  }
  const double mu = rule_mu(data, table);
  if (!(at_least_zero(mu) && at_least_zero(data.slack) && at_least_zero(data.dc_weight))) {
    return refuse(std::format("mu, slack and the weight of DC are at least 0 and finite, not {}, {} and {}", mu,
                              data.slack, data.dc_weight));
  }
  if (!at_least_zero(data.power)) {
    return refuse(std::format("the power of the steps in the weights is at least 0 and finite, not {}", data.power));
  }
  if (!at_least_zero(data.slack_cost)) {
    return refuse(std::format("the cost of the slack is at least 0 and finite, not {}", data.slack_cost));
  }
  if (data.centres == Centres::midpoint && scale != nullptr) {
    return refuse("a Laplace scale is for the MMSE centres, not the middles");
  }
  if (levels.size() != rows * columns * 64u) {
    return refuse(std::format("levels of {} x {} blocks of 64, not {} levels", rows, columns, levels.size()));
  }
  if (std::ranges::contains(table, std::uint16_t{0})) return refuse("a quantization step is at least 1");

  Problem problem;
  problem.rows_ = rows;
  problem.columns_ = columns;
  problem.levels_.assign(levels.begin(), levels.end());
  problem.slack_ = data.slack;
  for (std::size_t k = 0; k < 64u; ++k) {
    problem.steps_[k] = static_cast<double>(table[k]);
    problem.weights_[k] = mu / power_of(problem.steps_[k], data.power);
  }
  problem.weights_[0] *= data.dc_weight;

  // The shrinkage of each frequency for the MMSE centres; the middles need none.
  std::array<double, 64> shrink{};
  if (data.centres == Centres::mmse) {
    const std::array<double, 64> estimated = scale != nullptr ? *scale : laplace::scales(levels, table);
    shrink = laplace::shrinkages(table, estimated);
  }

  const std::size_t count = levels.size();
  problem.lower_.resize(count);
  problem.upper_.resize(count);
  problem.centres_.resize(count);
  const bool costed = data.slack > 0.0 && data.slack_cost > 0.0;
  Cost cost;
  if (costed) {
    cost.inner_lower.resize(count);
    cost.inner_upper.resize(count);
    for (std::size_t k = 0; k < 64u; ++k) cost.costs[k] = data.slack_cost / problem.steps_[k];
  }
  for (std::size_t index = 0; index < count; ++index) {
    const std::size_t k = index % 64u;
    const double level = static_cast<double>(levels[index]);
    const double step = problem.steps_[k];
    const double shift = k == 0u ? level_shift_dc : 0.0;
    problem.lower_[index] = interval_end(level, -0.5, -data.slack, step, shift);
    problem.upper_[index] = interval_end(level, 0.5, data.slack, step, shift);
    if (k == 0u || data.centres == Centres::midpoint) {
      // q Q is exact: |q| is below 2^15 and Q below 2^16.
      problem.centres_[index] = level * step + shift;
    } else {
      problem.centres_[index] = laplace::centre(levels[index], step, shrink[k]);
    }
    if (costed) {
      cost.inner_lower[index] = interval_end(level, -0.5, 0.0, step, shift);
      cost.inner_upper[index] = interval_end(level, 0.5, 0.0, step, shift);
    }
  }
  if (costed) problem.cost_ = std::move(cost);
  return problem;
}

double Problem::prox_one(std::size_t k, double e, double step) const noexcept {
  const std::size_t frequency = k % 64u;
  const double scaled = step * weights_[frequency];
  const double denominator = 1.0 + scaled;
  double z = (e + scaled * centres_[k]) / denominator;
  if (cost_) {
    // Beyond the file's own interval the objective adds step lambda per unit: the
    // minimizer moves back by step lambda / (1 + step m), but not past the end.
    const double shrink = step * cost_->costs[frequency] / denominator;
    if (z > cost_->inner_upper[k]) {
      z = std::max(cost_->inner_upper[k], z - shrink);
    } else if (z < cost_->inner_lower[k]) {
      z = std::min(cost_->inner_lower[k], z + shrink);
    }
  }
  return clamp_to(z, lower_[k], upper_[k]);
}

void Problem::prox(std::span<const double> e, double step, std::span<double> out) const noexcept {
  for (std::size_t k = 0; k < e.size(); ++k) out[k] = prox_one(k, e[k], step);
}

double Problem::conjugate_one(std::size_t k, double s) const noexcept {
  const std::size_t frequency = k % 64u;
  const double m = weights_[frequency];
  const double low = lower_[k];
  const double high = upper_[k];
  const double centre = centres_[k];
  if (!cost_) {
    if (m > 0.0) {
      const double best = clamp_to(centre + s / m, low, high);
      const double away = best - centre;
      return s * best - 0.5 * m * (away * away);
    }
    return std::max(s * low, s * high);
  }
  const double lambda = cost_->costs[frequency];
  const double own_low = cost_->inner_lower[k];
  const double own_high = cost_->inner_upper[k];
  double best = 0.0;
  if (m > 0.0) {
    const double free = centre + s / m;
    if (free > own_high) {
      best = clamp_to(centre + (s - lambda) / m, own_high, high);
    } else if (free < own_low) {
      best = clamp_to(centre + (s + lambda) / m, low, own_low);
    } else {
      best = free;
    }
  } else if (s > lambda) {
    best = high;
  } else if (s > 0.0) {
    best = own_high;
  } else if (s < -lambda) {
    best = low;
  } else {
    best = own_low;
  }
  const double away = best - centre;
  const double beyond = best > own_high ? best - own_high : (best < own_low ? own_low - best : 0.0);
  return s * best - 0.5 * m * (away * away) - lambda * beyond;
}

double Problem::value_one(std::size_t k, double c) const noexcept {
  const std::size_t frequency = k % 64u;
  const double away = c - centres_[k];
  double result = 0.5 * weights_[frequency] * (away * away);
  if (cost_) {
    const double own_low = cost_->inner_lower[k];
    const double own_high = cost_->inner_upper[k];
    const double beyond = c > own_high ? c - own_high : (c < own_low ? own_low - c : 0.0);
    result += cost_->costs[frequency] * beyond;
  }
  return result;
}

double Problem::conjugate(std::span<const double> s) const {
  std::vector<double> terms(s.size());
  for (std::size_t k = 0; k < s.size(); ++k) terms[k] = conjugate_one(k, s[k]);
  return exact::sum_rows(terms, columns_ * 64u);
}

double Problem::value(std::span<const double> c) const {
  std::vector<double> terms(c.size());
  for (std::size_t k = 0; k < c.size(); ++k) terms[k] = value_one(k, c[k]);
  return exact::sum_rows(terms, columns_ * 64u);
}

void Problem::clip(std::span<const double> c, std::span<double> out) const noexcept {
  for (std::size_t k = 0; k < c.size(); ++k) out[k] = clamp_to(c[k], lower_[k], upper_[k]);
}

std::vector<double> Problem::excess(std::span<const double> c) const {
  std::vector<double> result(c.size());
  for (std::size_t k = 0; k < c.size(); ++k) result[k] = std::max({lower_[k] - c[k], c[k] - upper_[k], 0.0});
  return result;
}

}  // namespace unround
