// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The model of one component: docs/math.md, 1.2 and 4.1.

#include "unround/model.hpp"

#include "unround/arrays.hpp"
#include "unround/exact.hpp"
#include "unround/laplace.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <expected>
#include <format>
#include <optional>
#include <span>
#include <string>
#include <utility>

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

// The shift of DC's intervals and centres, entry 0 of a block's 64.
constexpr std::array<double, 64> shifts = [](void) {
  std::array<double, 64> values{};
  values[0] = level_shift_dc;
  return values;
}();

// The proximal map of a coefficient: z = (e + t c) r with t = step m and r = 1 / (1 + t),
// clipped to the interval.
double prox_plain(double e, double t, double r, double centre, double low, double high) noexcept {
  return clamp_to((e + t * centre) * r, low, high);
}

// With a slack of a price: beyond the file's own interval the objective adds step
// lambda per unit, and z moves back by h = step lambda r, but not past the end.
double prox_costed(double e, double t, double r, double h, double centre, double low, double high, double own_low,
                   double own_high) noexcept {
  const double z = (e + t * centre) * r;
  const double above = std::max(own_high, z - h);
  const double below = std::min(own_low, z + h);
  const double inside = z < own_low ? below : z;
  return clamp_to(z > own_high ? above : inside, low, high);
}

// g*(s) of a coefficient without a price: s c* - m/2 (c* - centre)^2 at c* = clip(centre
// + s / m) where m > 0, max(s low, s high) where m = 0. Both are computed and one taken
// (s / m of m = 0 is discarded).
double conjugate_plain(double s, double m, double centre, double low, double high) noexcept {
  const double best = clamp_to(centre + s / m, low, high);
  const double away = best - centre;
  const double weighted = s * best - 0.5 * m * (away * away);
  const double support = std::max(s * low, s * high);
  return m > 0.0 ? weighted : support;
}

// The distance of c beyond the file's own interval.
double beyond_of(double c, double own_low, double own_high) noexcept {
  return std::max(std::max(c - own_high, own_low - c), 0.0);
}

// g*(s) of a coefficient whose slack has a price lambda: its maximizer where m > 0,
// centre + s / m within the file's own interval, centre + (s - lambda) / m above it (not
// below its end nor above the widened one), alike below; where m = 0, an end chosen by s.
double conjugate_costed(double s, double m, double centre, double low, double high, double lambda, double own_low,
                        double own_high) noexcept {
  const double free = centre + s / m;
  const double upward = clamp_to(centre + (s - lambda) / m, own_high, high);
  const double downward = clamp_to(centre + (s + lambda) / m, low, own_low);
  const double within = free < own_low ? downward : free;
  const double weighted = free > own_high ? upward : within;
  const double lower_end = s < -lambda ? low : own_low;
  const double upper_end = s > lambda ? high : own_high;
  const double flat = s > 0.0 ? upper_end : lower_end;
  const double best = m > 0.0 ? weighted : flat;
  const double away = best - centre;
  return s * best - 0.5 * m * (away * away) - lambda * beyond_of(best, own_low, own_high);
}

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

std::expected<Problem, Error> Problem::make(Blocks<const std::int16_t> levels,
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
  if (std::ranges::contains(table, std::uint16_t{0})) return refuse("a quantization step is at least 1");

  Problem problem;
  const BlockExtents extents = levels.extents();
  problem.levels_ = BlocksArray<std::int16_t>(extents);
  std::ranges::copy(entries(levels), problem.levels_.entries().begin());
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
  problem.lower_ = BlocksArray<double>(extents);
  problem.upper_ = BlocksArray<double>(extents);
  problem.centres_ = BlocksArray<double>(extents);
  const bool costed = data.slack > 0.0 && data.slack_cost > 0.0;
  Cost cost;
  if (costed) {
    cost.inner_lower = BlocksArray<double>(extents);
    cost.inner_upper = BlocksArray<double>(extents);
    for (std::size_t k = 0; k < 64u; ++k) cost.costs[k] = data.slack_cost / problem.steps_[k];
  }
  const bool mmse = data.centres == Centres::mmse;
  const std::array<double, 64>& steps = problem.steps_;
  for (std::size_t by = 0; by < extents.extent(0); ++by) {
    for (std::size_t bx = 0; bx < extents.extent(1); ++bx) {
      const std::span<const std::int16_t, 64> q = block_entries(levels, by, bx);
      const std::span<double, 64> lower = block_entries(problem.lower_.view(), by, bx);
      const std::span<double, 64> upper = block_entries(problem.upper_.view(), by, bx);
      const std::span<double, 64> centres = block_entries(problem.centres_.view(), by, bx);
      for (std::size_t k = 0; k < 64u; ++k) {
        const double level = static_cast<double>(q[k]);
        lower[k] = interval_end(level, -0.5, -data.slack, steps[k], shifts[k]);
        upper[k] = interval_end(level, 0.5, data.slack, steps[k], shifts[k]);
        // q Q is exact: |q| is below 2^15 and Q below 2^16.
        centres[k] = level * steps[k] + shifts[k];
      }
      // The MMSE centres of AC; DC's is the middle.
      if (mmse) {
        for (std::size_t k = 1; k < 64u; ++k) centres[k] = laplace::centre(q[k], steps[k], shrink[k]);
      }
      if (costed) {
        const std::span<double, 64> inner_lower = block_entries(cost.inner_lower.view(), by, bx);
        const std::span<double, 64> inner_upper = block_entries(cost.inner_upper.view(), by, bx);
        for (std::size_t k = 0; k < 64u; ++k) {
          const double level = static_cast<double>(q[k]);
          inner_lower[k] = interval_end(level, -0.5, 0.0, steps[k], shifts[k]);
          inner_upper[k] = interval_end(level, 0.5, 0.0, steps[k], shifts[k]);
        }
      }
    }
  }
  if (costed) problem.cost_ = std::move(cost);
  return problem;
}

double Problem::prox_at(std::size_t by, std::size_t bx, std::size_t v, std::size_t u, double e,
                        double step) const noexcept {
  const double t = step * weights()[v, u];
  const double r = 1.0 / (1.0 + t);
  const double centre = centres()[by, bx, v, u];
  const double low = lower()[by, bx, v, u];
  const double high = upper()[by, bx, v, u];
  if (!cost_) return prox_plain(e, t, r, centre, low, high);
  const double h = step * Block8x8<const double>(cost_->costs.data())[v, u] * r;
  return prox_costed(e, t, r, h, centre, low, high, cost_->inner_lower.view()[by, bx, v, u],
                     cost_->inner_upper.view()[by, bx, v, u]);
}

void Problem::prox(Blocks<const double> e, double step, Blocks<double> out) const noexcept {
  if (e.extents() != extents() || out.extents() != extents()) std::abort();
  // t, 1 / (1 + t) and the shrink of each frequency, once.
  std::array<double, 64> t{};
  std::array<double, 64> r{};
  std::array<double, 64> h{};
  for (std::size_t k = 0; k < 64u; ++k) {
    t[k] = step * weights_[k];
    r[k] = 1.0 / (1.0 + t[k]);
  }
  if (cost_) {
    const std::array<double, 64>& costs = cost_->costs;
    for (std::size_t k = 0; k < 64u; ++k) h[k] = step * costs[k] * r[k];
  }
  for (std::size_t by = 0; by < rows(); ++by) {
    for (std::size_t bx = 0; bx < columns(); ++bx) {
      const std::span<const double, 64> in = block_entries(e, by, bx);
      const std::span<double, 64> to = block_entries(out, by, bx);
      const std::span<const double, 64> centre = block_entries(centres(), by, bx);
      const std::span<const double, 64> low = block_entries(lower(), by, bx);
      const std::span<const double, 64> high = block_entries(upper(), by, bx);
      if (!cost_) {
        for (std::size_t k = 0; k < 64u; ++k) to[k] = prox_plain(in[k], t[k], r[k], centre[k], low[k], high[k]);
      } else {
        const Cost& cost = *cost_;
        const std::span<const double, 64> own_low = block_entries(cost.inner_lower.view(), by, bx);
        const std::span<const double, 64> own_high = block_entries(cost.inner_upper.view(), by, bx);
        for (std::size_t k = 0; k < 64u; ++k) {
          to[k] = prox_costed(in[k], t[k], r[k], h[k], centre[k], low[k], high[k], own_low[k], own_high[k]);
        }
      }
    }
  }
}

double Problem::conjugate_at(std::size_t by, std::size_t bx, std::size_t v, std::size_t u, double s) const noexcept {
  const double m = weights()[v, u];
  const double centre = centres()[by, bx, v, u];
  const double low = lower()[by, bx, v, u];
  const double high = upper()[by, bx, v, u];
  if (!cost_) return conjugate_plain(s, m, centre, low, high);
  return conjugate_costed(s, m, centre, low, high, Block8x8<const double>(cost_->costs.data())[v, u],
                          cost_->inner_lower.view()[by, bx, v, u], cost_->inner_upper.view()[by, bx, v, u]);
}

double Problem::value_at(std::size_t by, std::size_t bx, std::size_t v, std::size_t u, double c) const noexcept {
  const double away = c - centres()[by, bx, v, u];
  const double squares = 0.5 * weights()[v, u] * (away * away);
  if (!cost_) return squares;
  return squares + Block8x8<const double>(cost_->costs.data())[v, u] *
                       beyond_of(c, cost_->inner_lower.view()[by, bx, v, u], cost_->inner_upper.view()[by, bx, v, u]);
}

namespace {

// A sum over the coefficients of a term of each, in the order of docs/math.md,
// Arithmetic: rows of `columns` blocks of 64 terms, a block row a row. `term` fills
// the 64 terms of a block.
template <typename Terms>
double sum_of(const BlockExtents& extents, const Terms& terms_of) {
  BlocksArray<double> terms(extents);
  for (std::size_t by = 0; by < extents.extent(0); ++by) {
    for (std::size_t bx = 0; bx < extents.extent(1); ++bx) terms_of(by, bx, block_entries(terms.view(), by, bx));
  }
  return exact::sum_rows(Grid<const double>(terms.entries().data(), extents.extent(0), extents.extent(1) * 64u));
}

}  // namespace

double Problem::conjugate(Blocks<const double> s) const {
  if (s.extents() != extents()) std::abort();
  return sum_of(extents(), [this, s](std::size_t by, std::size_t bx, std::span<double, 64> terms) {
    const std::span<const double, 64> dual = block_entries(s, by, bx);
    const std::span<const double, 64> centre = block_entries(centres(), by, bx);
    const std::span<const double, 64> low = block_entries(lower(), by, bx);
    const std::span<const double, 64> high = block_entries(upper(), by, bx);
    if (!cost_) {
      for (std::size_t k = 0; k < 64u; ++k) {
        terms[k] = conjugate_plain(dual[k], weights_[k], centre[k], low[k], high[k]);
      }
      return;
    }
    const Cost& cost = *cost_;
    const std::span<const double, 64> own_low = block_entries(cost.inner_lower.view(), by, bx);
    const std::span<const double, 64> own_high = block_entries(cost.inner_upper.view(), by, bx);
    for (std::size_t k = 0; k < 64u; ++k) {
      terms[k] =
          conjugate_costed(dual[k], weights_[k], centre[k], low[k], high[k], cost.costs[k], own_low[k], own_high[k]);
    }
  });
}

double Problem::value(Blocks<const double> c) const {
  if (c.extents() != extents()) std::abort();
  return sum_of(extents(), [this, c](std::size_t by, std::size_t bx, std::span<double, 64> terms) {
    const std::span<const double, 64> at = block_entries(c, by, bx);
    const std::span<const double, 64> centre = block_entries(centres(), by, bx);
    for (std::size_t k = 0; k < 64u; ++k) {
      const double away = at[k] - centre[k];
      terms[k] = 0.5 * weights_[k] * (away * away);
    }
    if (!cost_) return;
    const Cost& cost = *cost_;
    const std::span<const double, 64> own_low = block_entries(cost.inner_lower.view(), by, bx);
    const std::span<const double, 64> own_high = block_entries(cost.inner_upper.view(), by, bx);
    for (std::size_t k = 0; k < 64u; ++k) {
      terms[k] = terms[k] + cost.costs[k] * beyond_of(at[k], own_low[k], own_high[k]);
    }
  });
}

void Problem::clip(Blocks<const double> c, Blocks<double> out) const noexcept {
  if (c.extents() != extents() || out.extents() != extents()) std::abort();
  const std::span<const double> values = entries(c);
  const std::span<double> clipped = entries(out);
  const std::span<const double> low = entries(lower());
  const std::span<const double> high = entries(upper());
  for (std::size_t k = 0; k < values.size(); ++k) clipped[k] = clamp_to(values[k], low[k], high[k]);
}

void Problem::excess(Blocks<const double> c, Blocks<double> out) const noexcept {
  if (c.extents() != extents() || out.extents() != extents()) std::abort();
  const std::span<const double> values = entries(c);
  const std::span<double> beyond = entries(out);
  const std::span<const double> low = entries(lower());
  const std::span<const double> high = entries(upper());
  for (std::size_t k = 0; k < values.size(); ++k) {
    beyond[k] = std::max(std::max(low[k] - values[k], values[k] - high[k]), 0.0);
  }
}

}  // namespace unround
