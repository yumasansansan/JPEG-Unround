// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The primal-dual method of docs/math.md, 5, and its records and stops, 6.
//
// Every step of an iteration is one loop over the entries of a channel's field, the
// same operation on neighbouring values (vector code), between the finite
// differences of operators.cpp and the proximal map of frames.cpp. The norms of the
// pixels of a record are summed over the channels into one grid, which is added up in
// the order of docs/math.md, Arithmetic.

#include "unround/pdhg.hpp"

#include "unround/arrays.hpp"
#include "unround/error.hpp"
#include "unround/exact.hpp"
#include "unround/frames.hpp"
#include "unround/model.hpp"
#include "unround/operators.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace unround::pdhg {

namespace {

using operators::Tensor;
using operators::TensorView;
using operators::Vector;
using operators::VectorView;

bool positive(double value) noexcept { return value > 0.0 && std::isfinite(value); }
bool at_least_zero(double value) noexcept { return value >= 0.0 && std::isfinite(value); }

// The bound of ||K||^2 of 3.3: 8 for TV, (17 + sqrt 33) / 2 for TGV, which is the
// double nearest to it.
double norm_bound(Model model) noexcept { return model == Model::tv ? 8.0 : (17.0 + std::sqrt(33.0)) / 2.0; }

// out = x + a d.
void step_along(std::span<const double> x, double a, std::span<const double> d, std::span<double> out) noexcept {
  const double* from = x.data();
  const double* along = d.data();
  double* to = out.data();
  const std::size_t count = out.size();
  for (std::size_t k = 0; k < count; ++k) to[k] = from[k] + a * along[k];
}

// out = 2 t - x.
void extrapolate(std::span<const double> tilde, std::span<const double> x, std::span<double> out) noexcept {
  const double* t = tilde.data();
  const double* from = x.data();
  double* to = out.data();
  const std::size_t count = out.size();
  for (std::size_t k = 0; k < count; ++k) to[k] = 2.0 * t[k] - from[k];
}

// out = p + a (g - w).
void ascend_less(std::span<const double> p, double a, std::span<const double> g, std::span<const double> w,
                 std::span<double> out) noexcept {
  const double* from = p.data();
  const double* plus = g.data();
  const double* minus = w.data();
  double* to = out.data();
  const std::size_t count = out.size();
  for (std::size_t k = 0; k < count; ++k) to[k] = from[k] + a * (plus[k] - minus[k]);
}

// out = w + a (p + q).
void descend_sum(std::span<const double> w, double a, std::span<const double> p, std::span<const double> q,
                 std::span<double> out) noexcept {
  const double* from = w.data();
  const double* first = p.data();
  const double* second = q.data();
  double* to = out.data();
  const std::size_t count = out.size();
  for (std::size_t k = 0; k < count; ++k) to[k] = from[k] + a * (first[k] + second[k]);
}

// x = rho t + rest x, rest = 1 - rho.
void relax(std::span<const double> tilde, double rho, double rest, std::span<double> x) noexcept {
  const double* t = tilde.data();
  double* to = x.data();
  const std::size_t count = x.size();
  for (std::size_t k = 0; k < count; ++k) to[k] = rho * t[k] + rest * to[k];
}

// out = a d.
void scaled(double a, std::span<const double> d, std::span<double> out) noexcept {
  const double* from = d.data();
  double* to = out.data();
  const std::size_t count = out.size();
  for (std::size_t k = 0; k < count; ++k) to[k] = a * from[k];
}

// squares += weight (x^2 + y^2), the entries of a vector field.
void add_squares(double weight, std::span<const double> x, std::span<const double> y,
                 std::span<double> squares) noexcept {
  const double* first = x.data();
  const double* second = y.data();
  double* to = squares.data();
  const std::size_t count = squares.size();
  for (std::size_t k = 0; k < count; ++k) to[k] += weight * (first[k] * first[k] + second[k] * second[k]);
}

// squares += weight |r|_F^2, the off-diagonal entry twice.
void add_tensor_squares(double weight, std::span<const double> xx, std::span<const double> yy,
                        std::span<const double> xy, std::span<double> squares) noexcept {
  const double* a = xx.data();
  const double* b = yy.data();
  const double* c = xy.data();
  double* to = squares.data();
  const std::size_t count = squares.size();
  for (std::size_t k = 0; k < count; ++k) to[k] += weight * (a[k] * a[k] + b[k] * b[k] + 2.0 * (c[k] * c[k]));
}

// norms += weight sqrt(squares), and squares cleared: a channel apart.
void add_roots(double weight, std::span<double> squares, std::span<double> norms) noexcept {
  double* from = squares.data();
  double* to = norms.data();
  const std::size_t count = norms.size();
  for (std::size_t k = 0; k < count; ++k) {
    to[k] += weight * std::sqrt(from[k]);
    from[k] = 0.0;
  }
}

// norms = sqrt(squares): the channels coupled.
void roots(std::span<const double> squares, std::span<double> norms) noexcept {
  const double* from = squares.data();
  double* to = norms.data();
  const std::size_t count = norms.size();
  for (std::size_t k = 0; k < count; ++k) to[k] = std::sqrt(from[k]);
}

// The largest of the values, which are at least 0. fmax, which no NaN stops, is
// vector code where std::max is not.
double largest(std::span<const double> values) noexcept {
  const double* from = values.data();
  const std::size_t count = values.size();
  double most = 0.0;
  for (std::size_t k = 0; k < count; ++k) most = std::fmax(most, from[k]);
  return most;
}

std::span<const double> entries_of(const GridArray<double>& grid) noexcept { return grid.entries(); }

// The fields of the method, channel by channel.
struct State {
  std::vector<GridArray<double>> x;
  std::vector<Vector> p;
  std::vector<Vector> w;
  std::vector<Tensor> r;
};

class Machine {
 public:
  Machine(const Frame& frame, Model model, const Regularizer& regularizer, const Plan& plan)
      : frame_(frame), model_(model), regularizer_(regularizer), plan_(plan), channels_(frame.channels()) {
    const std::size_t rows = frame.rows();
    const std::size_t columns = frame.columns();
    const GridExtents shape = frame.canvas_extents();
    for (std::size_t c = 0; c < channels_; ++c) {
      state_.x.emplace_back(shape);
      state_.p.push_back(operators::vector_of(rows, columns));
      tilde_.x.emplace_back(shape);
      tilde_.p.push_back(operators::vector_of(rows, columns));
      const Component& component = frame.component(c);
      coefficients_.emplace_back(block_extents(component.block_rows(), component.block_columns()));
      rooms_.push_back(frame.room(c));
      if (model == Model::tgv) {
        state_.w.push_back(operators::vector_of(rows, columns));
        state_.r.push_back(operators::tensor_of(rows, columns));
        tilde_.w.push_back(operators::vector_of(rows, columns));
        tilde_.r.push_back(operators::tensor_of(rows, columns));
      }
    }
    v_ = GridArray<double>(shape);
    bar_ = GridArray<double>(shape);
    gradient_ = operators::vector_of(rows, columns);
    if (model == Model::tgv) {
      bar_w_ = operators::vector_of(rows, columns);
      divergence_r_ = operators::vector_of(rows, columns);
      symmetrized_ = operators::tensor_of(rows, columns);
    }
    squares_.assign(rows * columns, 0.0);
    second_squares_.assign(rows * columns, 0.0);
  }

  // The state to start from, which is also the first tilde point.
  std::expected<void, Error> start(const Start* given) {
    for (std::size_t c = 0; c < channels_; ++c) {
      const Component& component = frame_.component(c);
      const Grid<double> x = state_.x[c].view();
      if (given != nullptr && !given->coefficients.empty()) {
        const BlocksArray<double>& from = given->coefficients[c];
        if (from.extents() != coefficients_[c].extents()) {
          return std::unexpected(Error{.code = Errc::options,
                                       .message = std::format("the start's coefficients of component {} are "
                                                              "not of its blocks",
                                                              c)});
        }
        component.problem.clip(from.view(), coefficients_[c].view());
        frame_.canvas(c, std::as_const(coefficients_[c]).view(), beyond_blocks, x);
      } else {
        frame_.start(c, x, coefficients_[c].view());
      }
      if (given != nullptr && !given->p.empty()) {
        if (!copy_vector(given->p[c], state_.p[c])) return mismatch("p", c);
      }
      if (model_ == Model::tgv) {
        if (given != nullptr && !given->w.empty()) {
          if (!copy_vector(given->w[c], state_.w[c])) return mismatch("w", c);
        } else {
          operators::gradient(std::as_const(state_.x[c]).view(), state_.w[c].view());
        }
        if (given != nullptr && !given->r.empty()) {
          if (!copy_tensor(given->r[c], state_.r[c])) return mismatch("r", c);
        }
      }
    }
    project_duals(state_);
    // The start is the first tilde point.
    for (std::size_t c = 0; c < channels_; ++c) {
      std::ranges::copy(state_.x[c].entries(), tilde_.x[c].entries().begin());
      copy_vector(state_.p[c], tilde_.p[c]);
      if (model_ == Model::tgv) {
        copy_vector(state_.w[c], tilde_.w[c]);
        copy_tensor(state_.r[c], tilde_.r[c]);
      }
    }
    return {};
  }

  // One iteration: the tilde point from the state, and the state relaxed toward it.
  void iterate(void) {
    const double tau = plan_.steps.tau;
    const double sigma = plan_.steps.sigma;
    const std::span<const double> weights = plan_.channel_weights;
    for (std::size_t c = 0; c < channels_; ++c) {
      const double descent = tau * weights[c];
      // v = x + tau gamma div p, and x~ its proximal map.
      operators::divergence(std::as_const(state_.p[c]).view(), bar_.view());
      step_along(entries_of(std::as_const(state_.x[c])), descent, entries_of(std::as_const(bar_)), v_.entries());
      frame_.prox(c, std::as_const(v_).view(), tau, tilde_.x[c].view(), coefficients_[c].view(), rooms_[c]);
      if (model_ == Model::tgv) {
        // w~ = w + tau gamma (p + div_2 r).
        operators::tensor_divergence(std::as_const(state_.r[c]).view(), divergence_r_.view());
        descend_sum(state_.w[c].x.entries(), descent, state_.p[c].x.entries(), divergence_r_.x.entries(),
                    tilde_.w[c].x.entries());
        descend_sum(state_.w[c].y.entries(), descent, state_.p[c].y.entries(), divergence_r_.y.entries(),
                    tilde_.w[c].y.entries());
      }
    }
    for (std::size_t c = 0; c < channels_; ++c) {
      const double ascent = sigma * weights[c];
      // x- = 2 x~ - x, and p~ = p + sigma gamma (grad x- - w-) before its projection.
      extrapolate(entries_of(std::as_const(tilde_.x[c])), entries_of(std::as_const(state_.x[c])), bar_.entries());
      operators::gradient(std::as_const(bar_).view(), gradient_.view());
      if (model_ == Model::tgv) {
        extrapolate(tilde_.w[c].x.entries(), state_.w[c].x.entries(), bar_w_.x.entries());
        extrapolate(tilde_.w[c].y.entries(), state_.w[c].y.entries(), bar_w_.y.entries());
        ascend_less(state_.p[c].x.entries(), ascent, gradient_.x.entries(), bar_w_.x.entries(),
                    tilde_.p[c].x.entries());
        ascend_less(state_.p[c].y.entries(), ascent, gradient_.y.entries(), bar_w_.y.entries(),
                    tilde_.p[c].y.entries());
        // r~ = r + sigma gamma E w- before its projection.
        operators::symmetrized_gradient(std::as_const(bar_w_).view(), symmetrized_.view());
        step_along(state_.r[c].xx.entries(), ascent, symmetrized_.xx.entries(), tilde_.r[c].xx.entries());
        step_along(state_.r[c].yy.entries(), ascent, symmetrized_.yy.entries(), tilde_.r[c].yy.entries());
        step_along(state_.r[c].xy.entries(), ascent, symmetrized_.xy.entries(), tilde_.r[c].xy.entries());
      } else {
        step_along(state_.p[c].x.entries(), ascent, gradient_.x.entries(), tilde_.p[c].x.entries());
        step_along(state_.p[c].y.entries(), ascent, gradient_.y.entries(), tilde_.p[c].y.entries());
      }
    }
    project_duals(tilde_);
    const double rho = plan_.relaxation;
    const double rest = 1.0 - rho;
    for (std::size_t c = 0; c < channels_; ++c) {
      relax(entries_of(std::as_const(tilde_.x[c])), rho, rest, state_.x[c].entries());
      relax(tilde_.p[c].x.entries(), rho, rest, state_.p[c].x.entries());
      relax(tilde_.p[c].y.entries(), rho, rest, state_.p[c].y.entries());
      if (model_ == Model::tgv) {
        relax(tilde_.w[c].x.entries(), rho, rest, state_.w[c].x.entries());
        relax(tilde_.w[c].y.entries(), rho, rest, state_.w[c].y.entries());
        relax(tilde_.r[c].xx.entries(), rho, rest, state_.r[c].xx.entries());
        relax(tilde_.r[c].yy.entries(), rho, rest, state_.r[c].yy.entries());
        relax(tilde_.r[c].xy.entries(), rho, rest, state_.r[c].xy.entries());
      }
    }
  }

  // The record of the tilde point.
  Record record(std::uint64_t iteration) {
    Record result;
    result.iteration = iteration;
    const std::span<const double> weights = plan_.channel_weights;
    double data = 0.0;
    for (std::size_t c = 0; c < channels_; ++c) {
      data += frame_.component(c).problem.value(std::as_const(coefficients_[c]).view());
    }
    if (model_ == Model::tv) {
      const double variation = regularizer_.alpha * norms(true);
      result.primal = variation + data;
      double conjugates = 0.0;
      for (std::size_t c = 0; c < channels_; ++c) {
        operators::divergence(std::as_const(tilde_.p[c]).view(), bar_.view());
        scaled(weights[c], entries_of(std::as_const(bar_)), v_.entries());
        conjugates += frame_.conjugate(c, std::as_const(v_).view(), plan_.free_radius);
      }
      result.dual = -conjugates;
      return result;
    }
    // TGV: alpha1 |grad x~ - w~| + alpha0 |E w~|_F, summed over the pixels.
    const double first = regularizer_.alpha1 * norms(true);
    const double second = regularizer_.alpha0 * norms(false);
    result.primal = (first + second) + data;
    // The feasible dual of 6.2: r scaled by theta, so that div_2 r lies within the ball
    // of alpha1, and p = -div_2 r.
    // The largest norm of div_2 r over the pixels: coupled, of all the channels' entries
    // at a pixel; apart, of each channel's own.
    std::ranges::fill(squares_, 0.0);
    std::vector<Vector> divergences;
    double most = 0.0;
    for (std::size_t c = 0; c < channels_; ++c) {
      divergences.push_back(operators::vector_of(frame_.rows(), frame_.columns()));
      operators::tensor_divergence(std::as_const(tilde_.r[c]).view(), divergences[c].view());
      add_squares(1.0, divergences[c].x.entries(), divergences[c].y.entries(), squares_);
      if (!regularizer_.coupled) {
        roots(squares_, second_squares_);
        most = std::max(most, largest(second_squares_));
        std::ranges::fill(squares_, 0.0);
      }
    }
    if (regularizer_.coupled) {
      roots(squares_, second_squares_);
      most = largest(second_squares_);
    }
    const double theta = most > regularizer_.alpha1 ? regularizer_.alpha1 / most : 1.0;
    result.scaling = theta;
    double conjugates = 0.0;
    for (std::size_t c = 0; c < channels_; ++c) {
      // xi = gamma div p, p = -theta div_2 r.
      scaled(-theta, divergences[c].x.entries(), gradient_.x.entries());
      scaled(-theta, divergences[c].y.entries(), gradient_.y.entries());
      operators::divergence(std::as_const(gradient_).view(), bar_.view());
      scaled(weights[c], entries_of(std::as_const(bar_)), v_.entries());
      conjugates += frame_.conjugate(c, std::as_const(v_).view(), plan_.free_radius);
    }
    result.dual = -conjugates;
    if (plan_.partial_radius) {
      // The partial gap of 6.3: G* at gamma div p~ of the iterate, and radius times the
      // norms of p~ + div_2 r~.
      double partial = 0.0;
      double residual = 0.0;
      for (std::size_t c = 0; c < channels_; ++c) {
        operators::divergence(std::as_const(tilde_.p[c]).view(), bar_.view());
        scaled(weights[c], entries_of(std::as_const(bar_)), v_.entries());
        partial += frame_.conjugate(c, std::as_const(v_).view(), plan_.free_radius);
        step_along(tilde_.p[c].x.entries(), 1.0, divergences[c].x.entries(), gradient_.x.entries());
        step_along(tilde_.p[c].y.entries(), 1.0, divergences[c].y.entries(), gradient_.y.entries());
        std::ranges::fill(squares_, 0.0);
        add_squares(1.0, gradient_.x.entries(), gradient_.y.entries(), squares_);
        roots(squares_, second_squares_);
        const Grid<const double> terms(second_squares_.data(), frame_.rows(), frame_.columns());
        residual += weights[c] * exact::sum_rows(terms);
      }
      result.partial_gap = result.primal + partial + *plan_.partial_radius * residual;
    }
    return result;
  }

  std::vector<GridArray<double>> take_canvas(void) { return std::move(tilde_.x); }
  std::vector<BlocksArray<double>> take_coefficients(void) { return std::move(coefficients_); }
  [[nodiscard]] std::span<const GridArray<double>> canvas(void) const noexcept { return tilde_.x; }

 private:
  static std::unexpected<Error> mismatch(const char* field, std::size_t c) {
    return std::unexpected(
        Error{.code = Errc::options,
              .message = std::format("the start's field {} of channel {} is not of the canvas's shape", field, c)});
  }

  static bool copy_vector(const Vector& from, Vector& to) {
    if (from.x.extents() != to.x.extents() || from.y.extents() != to.y.extents()) return false;
    std::ranges::copy(from.x.entries(), to.x.entries().begin());
    std::ranges::copy(from.y.entries(), to.y.entries().begin());
    return true;
  }

  static bool copy_tensor(const Tensor& from, Tensor& to) {
    if (from.xx.extents() != to.xx.extents() || from.yy.extents() != to.yy.extents() ||
        from.xy.extents() != to.xy.extents()) {
      return false;
    }
    std::ranges::copy(from.xx.entries(), to.xx.entries().begin());
    std::ranges::copy(from.yy.entries(), to.yy.entries().begin());
    std::ranges::copy(from.xy.entries(), to.xy.entries().begin());
    return true;
  }

  // The dual fields projected onto their balls (4.4, 5).
  void project_duals(State& state) const {
    std::vector<VectorView<double>> ps;
    ps.reserve(state.p.size());
    for (Vector& p : state.p) ps.push_back(p.view());
    const double radius = model_ == Model::tv ? regularizer_.alpha : regularizer_.alpha1;
    operators::project(ps, radius, regularizer_.coupled);
    if (model_ == Model::tgv) {
      std::vector<TensorView<double>> rs;
      rs.reserve(state.r.size());
      for (Tensor& r : state.r) rs.push_back(r.view());
      operators::project(rs, regularizer_.alpha0, regularizer_.coupled);
    }
  }

  // The sum over the pixels of the norms of the tilde point's first-order term (TV's
  // gradient, TGV's grad x~ - w~), or of TGV's second-order term, E w~, each channel
  // weighted by gamma: coupled, the Euclidean norm of all the channels' entries at a
  // pixel; apart, the sum of the channels' own.
  double norms(bool first_order) {
    const std::span<const double> weights = plan_.channel_weights;
    std::ranges::fill(squares_, 0.0);
    std::ranges::fill(second_squares_, 0.0);
    std::vector<double> sums(squares_.size(), 0.0);
    for (std::size_t c = 0; c < channels_; ++c) {
      const double gamma = weights[c];
      if (first_order) {
        operators::gradient(std::as_const(tilde_.x[c]).view(), gradient_.view());
        if (model_ == Model::tgv) {
          step_along(gradient_.x.entries(), -1.0, tilde_.w[c].x.entries(), gradient_.x.entries());
          step_along(gradient_.y.entries(), -1.0, tilde_.w[c].y.entries(), gradient_.y.entries());
        }
        add_squares(regularizer_.coupled ? gamma * gamma : 1.0, gradient_.x.entries(), gradient_.y.entries(), squares_);
      } else {
        operators::symmetrized_gradient(std::as_const(tilde_.w[c]).view(), symmetrized_.view());
        add_tensor_squares(regularizer_.coupled ? gamma * gamma : 1.0, symmetrized_.xx.entries(),
                           symmetrized_.yy.entries(), symmetrized_.xy.entries(), squares_);
      }
      if (!regularizer_.coupled) add_roots(gamma, squares_, sums);
    }
    if (regularizer_.coupled) roots(squares_, sums);
    const Grid<const double> terms(sums.data(), frame_.rows(), frame_.columns());
    return exact::sum_rows(terms);
  }

  const Frame& frame_;
  Model model_;
  const Regularizer& regularizer_;
  const Plan& plan_;
  std::size_t channels_;
  State state_;
  State tilde_;
  std::vector<BlocksArray<double>> coefficients_;
  std::vector<Room> rooms_;
  GridArray<double> v_;
  GridArray<double> bar_;
  Vector gradient_;
  Vector bar_w_;
  Vector divergence_r_;
  Tensor symmetrized_;
  std::vector<double> squares_;
  std::vector<double> second_squares_;
};

}  // namespace

std::expected<Plan, Error> plan(Model model, const Regularizer& regularizer, const Options& options,
                                std::size_t channels) {
  std::vector<std::string> errors;
  if (!positive(regularizer.alpha) || !positive(regularizer.alpha1) || !positive(regularizer.alpha0)) {
    errors.push_back(std::format("the weights alpha, alpha1 and alpha0 are above 0 and finite, not {}, {} and {}",
                                 regularizer.alpha, regularizer.alpha1, regularizer.alpha0));
  }
  if (!regularizer.channel_weights.empty() && regularizer.channel_weights.size() != channels) {
    errors.push_back(std::format("the channel weights are one for each of the {} channels, not {}", channels,
                                 regularizer.channel_weights.size()));
  }
  for (const double weight : regularizer.channel_weights) {
    if (!positive(weight)) errors.push_back(std::format("a channel weight is above 0 and finite, not {}", weight));
  }
  if (!(options.relaxation > 0.0 && options.relaxation < 2.0)) {
    errors.push_back(std::format("the relaxation lies between 0 and 2, not {}", options.relaxation));
  }
  if (!(options.step_product > 0.0 && options.step_product < 1.0)) {
    errors.push_back(std::format("the product of the steps lies between 0 and 1, not {}", options.step_product));
  }
  if (options.norm_squared && !positive(*options.norm_squared)) {
    errors.push_back(std::format("the bound of the norm squared is above 0 and finite, not {}", *options.norm_squared));
  }
  if (options.step_ratio && !positive(*options.step_ratio)) {
    errors.push_back(std::format("the ratio of the steps is above 0 and finite, not {}", *options.step_ratio));
  }
  if (options.tolerance && !at_least_zero(*options.tolerance)) {
    errors.push_back(std::format("the tolerance is at least 0 and finite, not {}", *options.tolerance));
  }
  if (!at_least_zero(options.relative_tolerance) || !at_least_zero(options.partial_tolerance)) {
    errors.push_back(std::format("the relative and the partial tolerance are at least 0 and finite, not {} and {}",
                                 options.relative_tolerance, options.partial_tolerance));
  }
  if (options.partial_radius && !at_least_zero(*options.partial_radius)) {
    errors.push_back(
        std::format("the radius of the partial gap is at least 0 and finite, not {}", *options.partial_radius));
  }
  if (model == Model::tv && (options.partial_radius || options.partial_tolerance > 0.0)) {
    errors.emplace_back("the partial gap is TGV's alone");
  }
  if (model == Model::tgv && options.partial_tolerance > 0.0 && !options.partial_radius) {
    errors.emplace_back("the partial tolerance needs the radius of the partial gap");
  }
  if (!at_least_zero(options.free_radius)) {
    errors.push_back(
        std::format("the radius of the free samples is at least 0 and finite, not {}", options.free_radius));
  }
  if (!errors.empty()) {
    std::string message;
    for (const std::string& error : errors) message += message.empty() ? error : "; " + error;
    return std::unexpected(Error{.code = Errc::options, .message = std::move(message)});
  }
  Plan result;
  result.channel_weights =
      regularizer.channel_weights.empty() ? std::vector<double>(channels, 1.0) : regularizer.channel_weights;
  double largest_weight = 0.0;
  for (const double weight : result.channel_weights) largest_weight = std::fmax(largest_weight, weight);
  const double norm_squared =
      options.norm_squared ? *options.norm_squared : norm_bound(model) * (largest_weight * largest_weight);
  const double weight = model == Model::tv ? regularizer.alpha : regularizer.alpha1;
  const double default_ratio = model == Model::tv ? tv_ratio : tgv_ratio;
  const double default_tolerance = model == Model::tv ? tv_tolerance : tgv_tolerance;
  const double ratio = options.step_ratio       ? *options.step_ratio
                       : options.weight_scaling ? default_ratio / (weight * weight)
                                                : default_ratio;
  result.tolerance = options.tolerance        ? *options.tolerance
                     : options.weight_scaling ? default_tolerance * weight
                                              : default_tolerance;
  result.iterations = options.iterations ? *options.iterations : model == Model::tv ? tv_iterations : tgv_iterations;
  const double product = options.step_product;
  result.steps.tau = std::sqrt((ratio * product) / norm_squared);
  result.steps.sigma = std::sqrt(product / (ratio * norm_squared));
  result.relative_tolerance = options.relative_tolerance;
  result.partial_tolerance = options.partial_tolerance;
  result.partial_radius = options.partial_radius;
  result.relaxation = options.relaxation;
  result.record_every = options.record_every;
  result.free_radius = options.free_radius;
  return result;
}

std::expected<Result, Error> solve(const Frame& frame, Model model, const Regularizer& regularizer,
                                   const Options& options, const Start* start, const Observer& observer) {
  const auto began = std::chrono::steady_clock::now();
  const std::size_t channels = frame.channels();
  std::expected<Plan, Error> made = plan(model, regularizer, options, channels);
  if (!made) return std::unexpected(made.error());
  const Plan& chosen = *made;
  if (start != nullptr) {
    const bool counts = (start->coefficients.empty() || start->coefficients.size() == channels) &&
                        (start->p.empty() || start->p.size() == channels) &&
                        (start->w.empty() || start->w.size() == channels) &&
                        (start->r.empty() || start->r.size() == channels);
    if (!counts) {
      return std::unexpected(
          Error{.code = Errc::options, .message = "the start's fields are one for each channel, or none"});
    }
    if (model == Model::tv && (!start->w.empty() || !start->r.empty())) {
      return std::unexpected(Error{.code = Errc::options, .message = "a start of TV has no fields w and r"});
    }
  }
  Machine machine(frame, model, regularizer, chosen);
  if (std::expected<void, Error> started = machine.start(start); !started) {
    return std::unexpected(started.error());
  }
  Result result;
  result.steps = chosen.steps;
  const double samples =
      static_cast<double>(channels) * static_cast<double>(frame.rows()) * static_cast<double>(frame.columns());
  const std::uint64_t most = chosen.iterations;
  const std::uint64_t every = chosen.record_every;
  // Records at the start (unless only the last is), every `every` iterations, and at
  // the last; the solver stops at the first record that meets a tolerance.
  const auto take = [&](std::uint64_t iteration) -> std::optional<Stop> {
    Record record = machine.record(iteration);
    record.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count();
    const double gap = record.primal - record.dual;
    bool met = false;
    if (chosen.tolerance > 0.0 && gap / samples <= chosen.tolerance) met = true;
    if (chosen.relative_tolerance > 0.0 && gap <= chosen.relative_tolerance * record.primal) met = true;
    if (chosen.partial_tolerance > 0.0 && record.partial_gap &&
        *record.partial_gap / samples <= chosen.partial_tolerance) {
      met = true;
    }
    result.records.push_back(record);
    if (met) return Stop::converged;
    if (observer && observer(result.records.back(), machine.canvas())) return Stop::observer;
    return std::nullopt;
  };
  std::optional<Stop> stopped;
  if (every > 0u || most == 0u) stopped = take(0);
  std::uint64_t iteration = 0;
  while (!stopped && iteration < most) {
    machine.iterate();
    ++iteration;
    if ((every > 0u && iteration % every == 0u) || iteration == most) stopped = take(iteration);
  }
  result.iterations = iteration;
  result.stop = stopped ? *stopped : Stop::iterations;
  result.canvas = machine.take_canvas();
  result.coefficients = machine.take_coefficients();
  return result;
}

}  // namespace unround::pdhg
