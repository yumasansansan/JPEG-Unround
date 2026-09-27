// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The model of one component (docs/math.md, 1.2 and 4.1): the interval of every
// coefficient, the quantization constraint set, and the data term G with its
// proximal map and its conjugate, which are separable in the coefficients.
//
// Coefficients are those of the canvas of samples that are not level-shifted:
// the level shift of 128 moves only the DC coefficient of a block, by 8 x 128 =
// 1024 exactly, which is added to the DC intervals and centres (1.1).
#ifndef UNROUND_MODEL_HPP
#define UNROUND_MODEL_HPP

#include "unround/error.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <vector>

namespace unround {

// What the level shift of 128 adds to the DC coefficient of a block.
inline constexpr double level_shift_dc = 1024.0;

// The centres of the data term: the MMSE ones of the Laplace model (2.2), or the
// middles of the intervals, q Q, which are exact.
enum class Centres : std::int32_t { mmse, midpoint };

// The options of G (4.1). mu weights the data term; where it is none, it is
// mu_scale times the mean of the component's 64 steps to the power mu_power,
// and in the chroma of a file in YCbCr the scale is mu_chroma times mu_scale
// (of_chroma). slack widens every interval by that many steps on each side;
// slack_cost is what leaving the file's own interval costs, within the slack, per
// step. The weight of an AC coefficient is mu / Q^power, that of DC dc_weight
// times it. The defaults are those of docs/math.md, 4.1.
struct DataTerm {
  std::optional<double> mu;
  double mu_scale = 9.0;
  double mu_power = 0.9;
  double mu_chroma = 0.3;
  double slack = 0.0;
  double dc_weight = 1.0;
  Centres centres = Centres::midpoint;
  double power = 2.0;
  double slack_cost = 0.0;

  // The data term of a chroma component, Cb or Cr of a file in YCbCr: where mu
  // follows the rule, its scale is mu_scale times mu_chroma, rounded once; a mu that
  // is given is taken as it is.
  [[nodiscard]] DataTerm of_chroma(void) const noexcept;

  friend bool operator==(const DataTerm&, const DataTerm&) = default;
};

// mu of the data term: the one given, or the rule's of the component's table. The
// mean of the 64 steps is exact (their integer sum over 64); the power 1 rounds
// nothing more, and another rounds as the platform's pow does.
[[nodiscard]] double rule_mu(const DataTerm& data, const std::array<std::uint16_t, 64>& table) noexcept;

// What leaving the file's own interval costs where the slack has a price: the
// file's own ends, and the price per unit of a coefficient of each frequency,
// beta / Q.
struct Cost {
  std::vector<double> inner_lower;
  std::vector<double> inner_upper;
  std::array<double, 64> costs{};
};

// A component to reconstruct: rows x columns blocks of 64 coefficients in natural
// order, each with its interval, its centre, and the weight and the step of its
// frequency.
class Problem {
 public:
  // The problem of a component's levels and quantization table (natural order), with
  // the options of G. `scale` is the Laplace scale of each frequency for the MMSE
  // centres, which laplace::scales estimates where it is not given. The ends of the
  // intervals are ((q - 1/2) - slack) Q and ((q + 1/2) + slack) Q, with 1024 added
  // on DC last; without slack every one of them is exact, and so is every middle.
  // Options out of their ranges, levels that are not rows x columns blocks, and
  // steps below 1 are refused.
  [[nodiscard]] static std::expected<Problem, Error> make(std::span<const std::int16_t> levels, std::size_t rows,
                                                          std::size_t columns,
                                                          const std::array<std::uint16_t, 64>& table,
                                                          const DataTerm& data,
                                                          const std::array<double, 64>* scale = nullptr);

  [[nodiscard]] std::size_t rows(void) const noexcept { return rows_; }
  [[nodiscard]] std::size_t columns(void) const noexcept { return columns_; }
  [[nodiscard]] std::size_t size(void) const noexcept { return levels_.size(); }
  [[nodiscard]] std::span<const std::int16_t> levels(void) const noexcept { return levels_; }
  [[nodiscard]] std::span<const double> lower(void) const noexcept { return lower_; }
  [[nodiscard]] std::span<const double> upper(void) const noexcept { return upper_; }
  [[nodiscard]] std::span<const double> centres(void) const noexcept { return centres_; }
  // The steps and the weights mu omega of the 64 frequencies.
  [[nodiscard]] const std::array<double, 64>& steps(void) const noexcept { return steps_; }
  [[nodiscard]] const std::array<double, 64>& weights(void) const noexcept { return weights_; }
  [[nodiscard]] double slack(void) const noexcept { return slack_; }
  [[nodiscard]] const std::optional<Cost>& cost(void) const noexcept { return cost_; }

  // The proximal map of `step` G in the coefficients (4.1): each output the minimizer
  // of (c - e)^2 / (2 step) + g_k(c).
  void prox(std::span<const double> e, double step, std::span<double> out) const noexcept;
  // The proximal map of one coefficient k (of frequency k % 64).
  [[nodiscard]] double prox_one(std::size_t k, double e, double step) const noexcept;

  // The conjugate of G at the coefficients s, the sum over them of g_k*(s_k), in the
  // order of docs/math.md, Arithmetic: block rows of `columns` blocks of 64 terms.
  [[nodiscard]] double conjugate(std::span<const double> s) const;
  [[nodiscard]] double conjugate_one(std::size_t k, double s) const noexcept;

  // G at coefficients within the intervals, without its indicator: the weighted
  // squares and the cost of the slack, summed as the conjugate is.
  [[nodiscard]] double value(std::span<const double> c) const;
  [[nodiscard]] double value_one(std::size_t k, double c) const noexcept;

  // The coefficients clipped to their intervals: the projection onto the set, in
  // the coefficients.
  void clip(std::span<const double> c, std::span<double> out) const noexcept;

  // How far each coefficient lies beyond its interval, 0 within it.
  [[nodiscard]] std::vector<double> excess(std::span<const double> c) const;

 private:
  Problem(void) = default;

  std::size_t rows_ = 0;
  std::size_t columns_ = 0;
  std::vector<std::int16_t> levels_;
  std::vector<double> lower_;
  std::vector<double> upper_;
  std::vector<double> centres_;
  std::array<double, 64> steps_{};
  std::array<double, 64> weights_{};
  double slack_ = 0.0;
  std::optional<Cost> cost_;
};

}  // namespace unround

#endif  // UNROUND_MODEL_HPP
