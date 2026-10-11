// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The iterations of the primal-dual method of docs/math.md, 5, and its records, 6, on
// the data term of a type that the caller gives: the coefficients of a JPEG file's
// components (pdhg.cpp) or the samples of a PNG file (samples.cpp).
//
// A step of an iteration takes a channel's fields a row at a time: the finite
// differences of the row that it reads (operators.cpp) into a room of a few rows, and
// then one loop over the row's entries, the same operation on neighbouring values
// (vector code). Where a channel's balls are its own, its duals are projected and
// relaxed in the loops of their steps, and x and w are relaxed where they are
// extrapolated, where the iteration reads each last; each value is computed from the
// same operands by the same operations as by passes over whole fields. The norms of
// the pixels of a record are summed over the channels into one grid, which is added
// up in the order of docs/math.md, Arithmetic.
//
// The threads of the plan take the canvas in parts of whole bands (the data's band, so
// that a proximal map's blocks lie in one part), each part's rows of every step at once;
// a difference that reads a row of another part reads it once every part has written
// the field (workers.hpp, meet). Every value is computed as by one thread, so that the
// result is the same, to the last bit, whatever the threads.
#ifndef UNROUND_PRIMAL_DUAL_HPP
#define UNROUND_PRIMAL_DUAL_HPP

#include "unround/arrays.hpp"
#include "unround/error.hpp"
#include "unround/exact.hpp"
#include "unround/operators.hpp"
#include "unround/pdhg.hpp"
#include "unround/results.hpp"

#include "workers.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <expected>
#include <format>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace unround::pdhg::detail {

using operators::Tensor;
using operators::TensorView;
using operators::Vector;
using operators::VectorView;

// out = x + a d.
inline void step_along(std::span<const double> x, double a, std::span<const double> d, std::span<double> out) noexcept {
  const double* from = x.data();
  const double* along = d.data();
  double* to = out.data();
  const std::size_t count = out.size();
  for (std::size_t k = 0; k < count; ++k) to[k] = from[k] + a * along[k];
}

// out = 2 t - x.
inline void extrapolate(std::span<const double> tilde, std::span<const double> x, std::span<double> out) noexcept {
  const double* t = tilde.data();
  const double* from = x.data();
  double* to = out.data();
  const std::size_t count = out.size();
  for (std::size_t k = 0; k < count; ++k) to[k] = 2.0 * t[k] - from[k];
}

// out = p + a (g - w).
inline void ascend_less(std::span<const double> p, double a, std::span<const double> g, std::span<const double> w,
                        std::span<double> out) noexcept {
  const double* from = p.data();
  const double* plus = g.data();
  const double* minus = w.data();
  double* to = out.data();
  const std::size_t count = out.size();
  for (std::size_t k = 0; k < count; ++k) to[k] = from[k] + a * (plus[k] - minus[k]);
}

// out = w + a (p + q).
inline void descend_sum(std::span<const double> w, double a, std::span<const double> p, std::span<const double> q,
                        std::span<double> out) noexcept {
  const double* from = w.data();
  const double* first = p.data();
  const double* second = q.data();
  double* to = out.data();
  const std::size_t count = out.size();
  for (std::size_t k = 0; k < count; ++k) to[k] = from[k] + a * (first[k] + second[k]);
}

// x = rho t + rest x, rest = 1 - rho.
inline void relax(std::span<const double> tilde, double rho, double rest, std::span<double> x) noexcept {
  const double* t = tilde.data();
  double* to = x.data();
  const std::size_t count = x.size();
  for (std::size_t k = 0; k < count; ++k) to[k] = rho * t[k] + rest * to[k];
}

// g - w, the step of TGV's p, or g, TV's.
template <bool less>
double less_of(double g, const double* w, std::size_t k) noexcept {
  if constexpr (less) {
    return g - w[k];
  } else {
    return g;
  }
}

// A row of a channel's p whose ball is its own: the step p~ = p + a (g - w) (TGV) or
// p + a g (TV, w unread), projected onto the ball of `radius` as operators::project
// projects a channel apart, kept in `kept` where `keep`, and p relaxed toward it. The
// operations of each value are those of the steps, the projection and the relaxation
// taken apart.
template <bool less, bool keep>
void ascend_own(std::span<double> px, std::span<double> py, double a, std::span<const double> gx,
                std::span<const double> gy, std::span<const double> wx, std::span<const double> wy, double radius,
                double rho, double rest, std::span<double> kept_x, std::span<double> kept_y) noexcept {
  double* x = px.data();
  double* y = py.data();
  const double* across = gx.data();
  const double* down = gy.data();
  const double* minus_x = wx.data();
  const double* minus_y = wy.data();
  const std::size_t count = px.size();
  for (std::size_t k = 0; k < count; ++k) {
    const double step_x = x[k] + a * less_of<less>(across[k], minus_x, k);
    const double step_y = y[k] + a * less_of<less>(down[k], minus_y, k);
    double square = 0.0;
    square += step_x * step_x + step_y * step_y;
    const double scale = operators::ball_scale(square, radius);
    const double projected_x = step_x * scale;
    const double projected_y = step_y * scale;
    if constexpr (keep) {
      kept_x[k] = projected_x;
      kept_y[k] = projected_y;
    }
    x[k] = rho * projected_x + rest * x[k];
    y[k] = rho * projected_y + rest * y[k];
  }
}

// A row of a channel's r whose ball is its own: the step r~ = r + a e, projected as
// operators::project projects a channel apart (the off-diagonal entry twice), kept
// where `keep`, and r relaxed toward it, each value as by the steps taken apart.
template <bool keep>
void step_own(std::span<double> rxx, std::span<double> ryy, std::span<double> rxy, double a,
              std::span<const double> exx, std::span<const double> eyy, std::span<const double> exy, double radius,
              double rho, double rest, std::span<double> kept_xx, std::span<double> kept_yy,
              std::span<double> kept_xy) noexcept {
  double* xx = rxx.data();
  double* yy = ryy.data();
  double* xy = rxy.data();
  const double* along_xx = exx.data();
  const double* along_yy = eyy.data();
  const double* along_xy = exy.data();
  const std::size_t count = rxx.size();
  for (std::size_t k = 0; k < count; ++k) {
    const double step_xx = xx[k] + a * along_xx[k];
    const double step_yy = yy[k] + a * along_yy[k];
    const double step_xy = xy[k] + a * along_xy[k];
    double square = 0.0;
    square += step_xx * step_xx + step_yy * step_yy + 2.0 * (step_xy * step_xy);
    const double scale = operators::ball_scale(square, radius);
    const double projected_xx = step_xx * scale;
    const double projected_yy = step_yy * scale;
    const double projected_xy = step_xy * scale;
    if constexpr (keep) {
      kept_xx[k] = projected_xx;
      kept_yy[k] = projected_yy;
      kept_xy[k] = projected_xy;
    }
    xx[k] = rho * projected_xx + rest * xx[k];
    yy[k] = rho * projected_yy + rest * yy[k];
    xy[k] = rho * projected_xy + rest * xy[k];
  }
}

// out = a d.
inline void scaled(double a, std::span<const double> d, std::span<double> out) noexcept {
  const double* from = d.data();
  double* to = out.data();
  const std::size_t count = out.size();
  for (std::size_t k = 0; k < count; ++k) to[k] = a * from[k];
}

// squares += weight (x^2 + y^2), the entries of a vector field.
inline void add_squares(double weight, std::span<const double> x, std::span<const double> y,
                        std::span<double> squares) noexcept {
  const double* first = x.data();
  const double* second = y.data();
  double* to = squares.data();
  const std::size_t count = squares.size();
  for (std::size_t k = 0; k < count; ++k) to[k] += weight * (first[k] * first[k] + second[k] * second[k]);
}

// squares += weight |r|_F^2, the off-diagonal entry twice.
inline void add_tensor_squares(double weight, std::span<const double> xx, std::span<const double> yy,
                               std::span<const double> xy, std::span<double> squares) noexcept {
  const double* a = xx.data();
  const double* b = yy.data();
  const double* c = xy.data();
  double* to = squares.data();
  const std::size_t count = squares.size();
  for (std::size_t k = 0; k < count; ++k) to[k] += weight * (a[k] * a[k] + b[k] * b[k] + 2.0 * (c[k] * c[k]));
}

// norms += weight sqrt(squares), and squares cleared: a channel apart.
inline void add_roots(double weight, std::span<double> squares, std::span<double> norms) noexcept {
  double* from = squares.data();
  double* to = norms.data();
  const std::size_t count = norms.size();
  for (std::size_t k = 0; k < count; ++k) {
    to[k] += weight * std::sqrt(from[k]);
    from[k] = 0.0;
  }
}

// norms = sqrt(squares): the channels coupled.
inline void roots(std::span<const double> squares, std::span<double> norms) noexcept {
  const double* from = squares.data();
  double* to = norms.data();
  const std::size_t count = norms.size();
  for (std::size_t k = 0; k < count; ++k) to[k] = std::sqrt(from[k]);
}

// The largest of the values, which are at least 0. fmax, which no NaN stops, is
// vector code where std::max is not.
inline double largest(std::span<const double> values) noexcept {
  const double* from = values.data();
  const std::size_t count = values.size();
  double most = 0.0;
  for (std::size_t k = 0; k < count; ++k) most = std::fmax(most, from[k]);
  return most;
}

// squares += x^2 + y^2, a row of a vector field's pixels, as operators::project adds a
// channel's squares to those of the channels before it.
inline void add_pixel_squares(std::span<const double> x, std::span<const double> y,
                              std::span<double> squares) noexcept {
  const double* first = x.data();
  const double* second = y.data();
  double* to = squares.data();
  const std::size_t count = squares.size();
  for (std::size_t k = 0; k < count; ++k) to[k] += first[k] * first[k] + second[k] * second[k];
}

// squares += |r|_F^2, the off-diagonal entry twice, likewise.
inline void add_pixel_tensor_squares(std::span<const double> xx, std::span<const double> yy, std::span<const double> xy,
                                     std::span<double> squares) noexcept {
  const double* a = xx.data();
  const double* b = yy.data();
  const double* c = xy.data();
  double* to = squares.data();
  const std::size_t count = squares.size();
  for (std::size_t k = 0; k < count; ++k) to[k] += a[k] * a[k] + b[k] * b[k] + 2.0 * (c[k] * c[k]);
}

// The squared norms of a row's pixels made the scales of their projections onto the
// ball of `radius` (operators::ball_scale), in place.
inline void scales_from(std::span<double> squares, double radius) noexcept {
  double* at = squares.data();
  const std::size_t count = squares.size();
  for (std::size_t k = 0; k < count; ++k) at[k] = operators::ball_scale(at[k], radius);
}

// A row of a channel's p from its step, coupled: the step projected by the row's scales,
// kept where `keep`, and p relaxed toward it, each value as by the projection and the
// relaxation of whole fields.
template <bool keep>
void settle(std::span<const double> step_x, std::span<const double> step_y, std::span<const double> scales,
            std::span<double> px, std::span<double> py, double rho, double rest, std::span<double> kept_x,
            std::span<double> kept_y) noexcept {
  const double* from_x = step_x.data();
  const double* from_y = step_y.data();
  const double* by = scales.data();
  double* x = px.data();
  double* y = py.data();
  const std::size_t count = px.size();
  for (std::size_t k = 0; k < count; ++k) {
    const double projected_x = from_x[k] * by[k];
    const double projected_y = from_y[k] * by[k];
    if constexpr (keep) {
      kept_x[k] = projected_x;
      kept_y[k] = projected_y;
    }
    x[k] = rho * projected_x + rest * x[k];
    y[k] = rho * projected_y + rest * y[k];
  }
}

// A row of a channel's r likewise.
template <bool keep>
void settle_tensor(std::span<const double> step_xx, std::span<const double> step_yy, std::span<const double> step_xy,
                   std::span<const double> scales, std::span<double> rxx, std::span<double> ryy, std::span<double> rxy,
                   double rho, double rest, std::span<double> kept_xx, std::span<double> kept_yy,
                   std::span<double> kept_xy) noexcept {
  const double* from_xx = step_xx.data();
  const double* from_yy = step_yy.data();
  const double* from_xy = step_xy.data();
  const double* by = scales.data();
  double* xx = rxx.data();
  double* yy = ryy.data();
  double* xy = rxy.data();
  const std::size_t count = rxx.size();
  for (std::size_t k = 0; k < count; ++k) {
    const double projected_xx = from_xx[k] * by[k];
    const double projected_yy = from_yy[k] * by[k];
    const double projected_xy = from_xy[k] * by[k];
    if constexpr (keep) {
      kept_xx[k] = projected_xx;
      kept_yy[k] = projected_yy;
      kept_xy[k] = projected_xy;
    }
    xx[k] = rho * projected_xx + rest * xx[k];
    yy[k] = rho * projected_yy + rest * yy[k];
    xy[k] = rho * projected_xy + rest * xy[k];
  }
}

inline std::span<const double> entries_of(const GridArray<double>& grid) noexcept { return grid.entries(); }

// The fields of the method, channel by channel.
struct State {
  std::vector<GridArray<double>> x;
  std::vector<Vector> p;
  std::vector<Vector> w;
  std::vector<Tensor> r;
};

// The method on the data term of `Data`, which gives the shape of the canvas
// (channels, rows, columns, extents), each channel's start (start(c, given, x)), its
// proximal map on the rows of a part of the canvas (prox_rows(c, v, tau, out, rows,
// row)) whose parts are of whole bands of rows (band()), and the terms of a part's rows
// of G at the point of the last of them or at the start (value_terms(c, rows)) and of
// G* at a field (conjugate_terms(c, xi, radius, rows, row)), and their values once
// every part has taken its terms (value_sum(c), conjugate_sum(c)).
template <typename Data>
class Machine {
 public:
  Machine(Data& data, Model model, const Regularizer& regularizer, const Plan& plan)
      : data_(data),
        model_(model),
        regularizer_(regularizer),
        plan_(plan),
        channels_(data.channels()),
        coupled_(regularizer.coupled && channels_ > 1u),
        parts_(parts_of(data.rows(), data.band(), plan.threads)),
        workers_(parts_.size()) {
    const std::size_t rows = data.rows();
    const std::size_t columns = data.columns();
    const GridExtents shape = data.extents();
    for (std::size_t c = 0; c < channels_; ++c) {
      state_.x.emplace_back(shape);
      state_.p.push_back(operators::vector_of(rows, columns));
      tilde_.x.emplace_back(shape);
      tilde_.p.push_back(operators::vector_of(rows, columns));
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
    squares_.assign(rows * columns, 0.0);
    second_squares_.assign(rows * columns, 0.0);
    first_terms_.assign(rows * columns, 0.0);
    if (model == Model::tgv) {
      second_terms_.assign(rows * columns, 0.0);
      for (std::size_t c = 0; c < channels_; ++c) divergences_.push_back(operators::vector_of(rows, columns));
    }
    most_of_parts_.assign(parts_.size(), 0.0);
    for (std::size_t part = 0; part < parts_.size(); ++part) {
      rows_room_.emplace_back(columns, 0.0);
      differences_room_.emplace_back(3u * columns, 0.0);
      squares_room_.emplace_back(columns, 0.0);
      scales_room_.emplace_back(columns, 0.0);
      rings_room_.emplace_back(6u * channels_ * columns, 0.0);
      edges_.emplace_back(3u * channels_ * columns, 0.0);
      if (coupled_) coupled_room_.emplace_back(((5u * channels_) + 2u) * columns, 0.0);
    }
    task_ = [this](std::size_t part) -> void { iterate_part(part); };
    record_task_ = [this](std::size_t part) -> void { record_part(part); };
  }

  Machine(const Machine&) = delete;
  Machine& operator=(const Machine&) = delete;
  Machine(Machine&&) = delete;
  Machine& operator=(Machine&&) = delete;
  ~Machine(void) = default;

  // The state to start from, which is also the first tilde point.
  std::expected<void, Error> start(const Start* given) {
    for (std::size_t c = 0; c < channels_; ++c) {
      const Grid<double> x = state_.x[c].view();
      if (std::expected<void, Error> started = data_.start(c, given, x); !started) {
        return std::unexpected(started.error());
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

  // One iteration: the tilde point from the state, and the state relaxed toward it,
  // the parts of the canvas at once. The tilde point's duals are kept for a record
  // where `keep`; its canvas and w are kept always.
  void iterate(bool keep) {
    keep_ = keep;
    workers_.run(task_);
  }

  // The record of the tilde point. The terms of its sums over the pixels, and the
  // fields between, are taken by the parts at once, each its own rows (record_part);
  // the sums are added in the order of the rows (exact::sum_rows), and G and G* are
  // taken, by the caller, so that every value is the same whatever the threads.
  Record record(std::uint64_t iteration) {
    Record result;
    result.iteration = iteration;
    const std::span<const double> weights = plan_.channel_weights;
    // G, the norms of the pixels, and for TGV div_2 r~ and its largest norms.
    take(Stage::norms, 0);
    double data = 0.0;
    for (std::size_t c = 0; c < channels_; ++c) data += data_.value_sum(c);
    const Grid<const double> first_terms(first_terms_.data(), data_.rows(), data_.columns());
    if (model_ == Model::tv) {
      const double variation = regularizer_.alpha * exact::sum_rows(first_terms);
      result.primal = variation + data;
      double conjugates = 0.0;
      for (std::size_t c = 0; c < channels_; ++c) {
        take(Stage::divergence_of_p, c);
        conjugates += data_.conjugate_sum(c);
      }
      result.dual = -conjugates;
      return result;
    }
    // TGV: alpha1 |grad x~ - w~| + alpha0 |E w~|_F, summed over the pixels.
    const Grid<const double> second_terms(second_terms_.data(), data_.rows(), data_.columns());
    const double first = regularizer_.alpha1 * exact::sum_rows(first_terms);
    const double second = regularizer_.alpha0 * exact::sum_rows(second_terms);
    result.primal = (first + second) + data;
    // The feasible dual of 6.2: r scaled by theta, so that div_2 r lies within the ball
    // of alpha1, and p = -div_2 r. The largest norm of div_2 r over the pixels (coupled,
    // of all the channels' entries at a pixel; apart, of each channel's own) is the
    // largest of the parts': of norms, at least 0, the same in any order.
    const double most = largest(most_of_parts_);
    const double theta = most > regularizer_.alpha1 ? regularizer_.alpha1 / most : 1.0;
    result.scaling = theta;
    theta_ = theta;
    double conjugates = 0.0;
    for (std::size_t c = 0; c < channels_; ++c) {
      // xi = gamma div p, p = -theta div_2 r.
      take(Stage::divergence_of_r, c);
      conjugates += data_.conjugate_sum(c);
    }
    result.dual = -conjugates;
    if (plan_.partial_radius) {
      // The partial gap of 6.3: G* at gamma div p~ of the iterate, and radius times the
      // norms of p~ + div_2 r~.
      double partial = 0.0;
      double residual = 0.0;
      for (std::size_t c = 0; c < channels_; ++c) {
        take(Stage::divergence_of_p, c);
        partial += data_.conjugate_sum(c);
        take(Stage::residual, c);
        const Grid<const double> terms(second_squares_.data(), data_.rows(), data_.columns());
        residual += weights[c] * exact::sum_rows(terms);
      }
      result.partial_gap = result.primal + partial + *plan_.partial_radius * residual;
    }
    return result;
  }

  std::vector<GridArray<double>> take_canvas(void) { return std::move(tilde_.x); }
  [[nodiscard]] std::span<const GridArray<double>> canvas(void) const noexcept { return tilde_.x; }

 private:
  // The parts of the canvas of `rows` rows: whole bands, as many parts as the threads but
  // no more than the bands, and as even as the bands allow.
  static std::vector<operators::Rows> parts_of(std::size_t rows, std::size_t band, std::size_t threads) {
    // A band of no rows is a mistake of the data, which ends the program.
    if (band == 0u) std::abort();
    const std::size_t bands = (rows + band - 1u) / band;
    const std::size_t count = std::max<std::size_t>(1u, std::min(threads, bands));
    std::vector<operators::Rows> parts;
    parts.reserve(count);
    for (std::size_t k = 0; k < count; ++k) {
      parts.push_back(operators::Rows{.first = std::min(rows, band * (bands * k / count)),
                                      .end = std::min(rows, band * (bands * (k + 1u) / count))});
    }
    return parts;
  }

  // The iteration of part `part` of the canvas: its rows of every step. A step takes a
  // row at a time, the differences that it reads made into the part's room first.
  void iterate_part(std::size_t part) noexcept {
    const operators::Rows rows = parts_[part];
    const std::size_t columns = data_.columns();
    // The entries of row i of a field.
    const auto row = [columns](auto entries, std::size_t i) noexcept { return entries.subspan(i * columns, columns); };
    // The rows of a row's differences.
    const std::span<double> room = differences_room_[part];
    const std::span<double> one = room.subspan(0, columns);
    const std::span<double> two = room.subspan(columns, columns);
    const double tau = plan_.steps.tau;
    const std::span<const double> weights = plan_.channel_weights;
    for (std::size_t c = 0; c < channels_; ++c) {
      const double descent = tau * weights[c];
      // v = x + tau gamma div p, and x~ its proximal map.
      const Vector& p = state_.p[c];
      const std::span<const double> x = entries_of(std::as_const(state_.x[c]));
      const std::span<double> v = v_.entries();
      for (std::size_t i = rows.first; i < rows.end; ++i) {
        operators::divergence(p.view(), i, one);
        step_along(row(x, i), descent, one, row(v, i));
      }
      data_.prox_rows(c, std::as_const(v_).view(), tau, tilde_.x[c].view(), rows, rows_room_[part]);
      if (model_ == Model::tgv) {
        // w~ = w + tau gamma (p + div_2 r).
        const TensorView<const double> r = std::as_const(state_.r[c]).view();
        const Vector& w = state_.w[c];
        Vector& tilde_w = tilde_.w[c];
        for (std::size_t i = rows.first; i < rows.end; ++i) {
          operators::tensor_divergence(r, i, one, two);
          descend_sum(row(w.x.entries(), i), descent, row(p.x.entries(), i), one, row(tilde_w.x.entries(), i));
          descend_sum(row(w.y.entries(), i), descent, row(p.y.entries(), i), two, row(tilde_w.y.entries(), i));
        }
      }
    }
    duals(part);
  }

  // The dual steps of every channel, a row at a time (5). The extrapolations of a row
  // are made where they are read, from x~ and x and from w~ and w, into rings of two
  // rows of each channel's own, and x and w are relaxed once their row's extrapolations
  // are made, which the iteration reads no more; a part's first row's x- and last row's
  // w-, which the parts beside read, are made first, and the parts meet once they all
  // have. Where a channel's balls are its own, its step is projected, kept only for a
  // record, and its dual relaxed in the step's loop; coupled, every channel's steps of
  // the row are made first, their squares added channel after channel in their order,
  // and then each channel's duals projected, kept only for a record, and relaxed. Each
  // value is that of passes over whole fields.
  void duals(std::size_t part) noexcept {
    const operators::Rows rows = parts_[part];
    const std::size_t columns = data_.columns();
    const std::size_t height = data_.rows();
    const auto row = [columns](auto entries, std::size_t i) noexcept { return entries.subspan(i * columns, columns); };
    const std::span<double> room = differences_room_[part];
    const std::span<double> one = room.subspan(0, columns);
    const std::span<double> two = room.subspan(columns, columns);
    const std::span<double> three = room.subspan(2u * columns, columns);
    // The rings of x- and of w-'s two entries, two rows each, of each channel: row i's in
    // the place of its parity within the part.
    const std::span<double> rings = rings_room_[part];
    const auto ring_of = [rings, columns](std::size_t c, std::size_t k, std::size_t parity) noexcept {
      return rings.subspan(((6u * c) + (2u * k) + parity) * columns, columns);
    };
    // The rows that the parts beside read: x- of a part's first row, w- of its last.
    const auto edge_of = [this, columns](std::size_t at, std::size_t c, std::size_t k) noexcept {
      return std::span<double>(edges_[at]).subspan(((3u * c) + k) * columns, columns);
    };
    // Coupled: the row's steps of each channel, p's two entries and r's three, and the
    // squares of p's and r's balls, which become their scales.
    const std::span<double> steps = coupled_ ? std::span<double>(coupled_room_[part]) : std::span<double>{};
    const auto step_of = [steps, columns](std::size_t c, std::size_t k) noexcept {
      return steps.subspan(((5u * c) + k) * columns, columns);
    };
    const std::size_t squares_at = 5u * channels_ * columns;
    const std::span<double> squares_p = coupled_ ? steps.subspan(squares_at, columns) : std::span<double>{};
    const std::span<double> squares_r = coupled_ ? steps.subspan(squares_at + columns, columns) : std::span<double>{};
    const std::span<const double> weights = plan_.channel_weights;
    const double sigma = plan_.steps.sigma;
    const bool keep = keep_;
    const bool tgv = model_ == Model::tgv;
    const double radius = model_ == Model::tv ? regularizer_.alpha : regularizer_.alpha1;
    const double radius0 = regularizer_.alpha0;
    const double rho = plan_.relaxation;
    const double rest = 1.0 - rho;
    if (rows.first < rows.end) {
      for (std::size_t c = 0; c < channels_; ++c) {
        extrapolate(row(entries_of(std::as_const(tilde_.x[c])), rows.first),
                    row(entries_of(std::as_const(state_.x[c])), rows.first), edge_of(part, c, 0));
        if (!tgv) continue;
        const std::size_t last_row = rows.end - 1u;
        extrapolate(row(std::as_const(tilde_.w[c]).x.entries(), last_row),
                    row(std::as_const(state_.w[c]).x.entries(), last_row), edge_of(part, c, 1));
        extrapolate(row(std::as_const(tilde_.w[c]).y.entries(), last_row),
                    row(std::as_const(state_.w[c]).y.entries(), last_row), edge_of(part, c, 2));
      }
    }
    workers_.meet();
    for (std::size_t i = rows.first; i < rows.end; ++i) {
      const std::size_t parity = (i - rows.first) % 2u;
      const std::size_t other = 1u - parity;
      const bool last = i + 1u == height;
      if (coupled_) {
        std::ranges::fill(squares_p, 0.0);
        std::ranges::fill(squares_r, 0.0);
      }
      for (std::size_t c = 0; c < channels_; ++c) {
        const double ascent = sigma * weights[c];
        const std::span<const double> tilde_x = entries_of(std::as_const(tilde_.x[c]));
        const std::span<double> x = state_.x[c].entries();
        // x- of row i, made at the row before (or now, at the part's first), and of the
        // row below: the part's, the first row of the part below, or none.
        const std::span<double> bar = ring_of(c, 0, parity);
        if (i == rows.first) extrapolate(row(tilde_x, i), row(std::as_const(x), i), bar);
        std::span<const double> below{};
        if (i + 1u < rows.end) {
          extrapolate(row(tilde_x, i + 1u), row(std::as_const(x), i + 1u), ring_of(c, 0, other));
          below = ring_of(c, 0, other);
        } else if (!last) {
          below = edge_of(part + 1u, c, 0);
        }
        operators::gradient(bar, below, one, two);
        relax(row(tilde_x, i), rho, rest, row(x, i));
        const std::span<double> px = row(state_.p[c].x.entries(), i);
        const std::span<double> py = row(state_.p[c].y.entries(), i);
        if (!tgv) {
          // p~ = p + sigma gamma grad x-.
          if (coupled_) {
            step_along(px, ascent, one, step_of(c, 0));
            step_along(py, ascent, two, step_of(c, 1));
            add_pixel_squares(step_of(c, 0), step_of(c, 1), squares_p);
          } else if (keep) {
            ascend_own<false, true>(px, py, ascent, one, two, {}, {}, radius, rho, rest,
                                    row(tilde_.p[c].x.entries(), i), row(tilde_.p[c].y.entries(), i));
          } else {
            ascend_own<false, false>(px, py, ascent, one, two, {}, {}, radius, rho, rest, {}, {});
          }
          continue;
        }
        // w- of row i, and of the row above: the part's, made at the row before, the last
        // row of the part above, or none.
        const Vector& tilde_w = tilde_.w[c];
        Vector& w = state_.w[c];
        const std::span<double> bar_wx = ring_of(c, 1, parity);
        const std::span<double> bar_wy = ring_of(c, 2, parity);
        extrapolate(row(tilde_w.x.entries(), i), row(std::as_const(w).x.entries(), i), bar_wx);
        extrapolate(row(tilde_w.y.entries(), i), row(std::as_const(w).y.entries(), i), bar_wy);
        std::span<const double> above_x{};
        std::span<const double> above_y{};
        if (i > rows.first) {
          above_x = ring_of(c, 1, other);
          above_y = ring_of(c, 2, other);
        } else if (i > 0u) {
          above_x = edge_of(part - 1u, c, 1);
          above_y = edge_of(part - 1u, c, 2);
        }
        relax(row(tilde_w.x.entries(), i), rho, rest, row(w.x.entries(), i));
        relax(row(tilde_w.y.entries(), i), rho, rest, row(w.y.entries(), i));
        // p~ = p + sigma gamma (grad x- - w-), and r~ = r + sigma gamma E w-.
        if (coupled_) {
          ascend_less(px, ascent, one, bar_wx, step_of(c, 0));
          ascend_less(py, ascent, two, bar_wy, step_of(c, 1));
          add_pixel_squares(step_of(c, 0), step_of(c, 1), squares_p);
        } else if (keep) {
          ascend_own<true, true>(px, py, ascent, one, two, bar_wx, bar_wy, radius, rho, rest,
                                 row(tilde_.p[c].x.entries(), i), row(tilde_.p[c].y.entries(), i));
        } else {
          ascend_own<true, false>(px, py, ascent, one, two, bar_wx, bar_wy, radius, rho, rest, {}, {});
        }
        operators::symmetrized_gradient(bar_wx, bar_wy, above_x, above_y, last, one, two, three);
        const std::span<double> rxx = row(state_.r[c].xx.entries(), i);
        const std::span<double> ryy = row(state_.r[c].yy.entries(), i);
        const std::span<double> rxy = row(state_.r[c].xy.entries(), i);
        if (coupled_) {
          step_along(rxx, ascent, one, step_of(c, 2));
          step_along(ryy, ascent, two, step_of(c, 3));
          step_along(rxy, ascent, three, step_of(c, 4));
          add_pixel_tensor_squares(step_of(c, 2), step_of(c, 3), step_of(c, 4), squares_r);
        } else if (keep) {
          step_own<true>(rxx, ryy, rxy, ascent, one, two, three, radius0, rho, rest, row(tilde_.r[c].xx.entries(), i),
                         row(tilde_.r[c].yy.entries(), i), row(tilde_.r[c].xy.entries(), i));
        } else {
          step_own<false>(rxx, ryy, rxy, ascent, one, two, three, radius0, rho, rest, {}, {}, {});
        }
      }
      if (!coupled_) continue;
      // The scales of the balls (4.4, 5), and every channel's duals settled.
      scales_from(squares_p, radius);
      if (tgv) scales_from(squares_r, radius0);
      for (std::size_t c = 0; c < channels_; ++c) {
        if (keep) {
          settle<true>(step_of(c, 0), step_of(c, 1), squares_p, row(state_.p[c].x.entries(), i),
                       row(state_.p[c].y.entries(), i), rho, rest, row(tilde_.p[c].x.entries(), i),
                       row(tilde_.p[c].y.entries(), i));
        } else {
          settle<false>(step_of(c, 0), step_of(c, 1), squares_p, row(state_.p[c].x.entries(), i),
                        row(state_.p[c].y.entries(), i), rho, rest, {}, {});
        }
        if (!tgv) continue;
        const std::span<double> rxx = row(state_.r[c].xx.entries(), i);
        const std::span<double> ryy = row(state_.r[c].yy.entries(), i);
        const std::span<double> rxy = row(state_.r[c].xy.entries(), i);
        if (keep) {
          settle_tensor<true>(step_of(c, 2), step_of(c, 3), step_of(c, 4), squares_r, rxx, ryy, rxy, rho, rest,
                              row(tilde_.r[c].xx.entries(), i), row(tilde_.r[c].yy.entries(), i),
                              row(tilde_.r[c].xy.entries(), i));
        } else {
          settle_tensor<false>(step_of(c, 2), step_of(c, 3), step_of(c, 4), squares_r, rxx, ryy, rxy, rho, rest, {}, {},
                               {});
        }
      }
    }
  }

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

  // What the parts of a record take: the terms of G and the norms of the pixels, of
  // TV's gradient or of TGV's grad x~ - w~ and E w~, each channel weighted by gamma
  // (coupled, the Euclidean norm of all the channels' entries at a pixel; apart, the
  // sum of the channels' own), and TGV's div_2 r~ and the largest of its norms;
  // xi = gamma div p of channel c, of p = -theta div_2 r~ or of p~, and the terms of G*
  // there; or the norms of p~ + div_2 r~.
  enum class Stage : std::uint8_t { norms, divergence_of_r, divergence_of_p, residual };

  void take(Stage stage, std::size_t c) {
    stage_ = stage;
    stage_channel_ = c;
    workers_.run(record_task_);
  }

  // A part's rows of a stage of the record, each value as by passes over whole fields.
  void record_part(std::size_t part) noexcept {
    const operators::Rows rows = parts_[part];
    const std::size_t columns = data_.columns();
    const std::size_t first = rows.first * columns;
    const std::size_t count = (rows.end - rows.first) * columns;
    const auto own = [first, count](auto entries) noexcept { return entries.subspan(first, count); };
    const std::span<const double> weights = plan_.channel_weights;
    const std::size_t c = stage_channel_;
    switch (stage_) {
      case Stage::norms:
        for (std::size_t channel = 0; channel < channels_; ++channel) data_.value_terms(channel, rows);
        norms_of(part);
        return;
      case Stage::divergence_of_r: {
        const Vector& divergence = divergences_[c];
        scaled(-theta_, own(divergence.x.entries()), own(gradient_.x.entries()));
        scaled(-theta_, own(divergence.y.entries()), own(gradient_.y.entries()));
        // The divergence reads the rows of the parts beside.
        workers_.meet();
        operators::divergence(std::as_const(gradient_).view(), bar_.view(), rows);
        scaled(weights[c], own(entries_of(std::as_const(bar_))), own(v_.entries()));
        // G*'s terms of the part's blocks and samples read only its own rows of xi.
        data_.conjugate_terms(c, std::as_const(v_).view(), plan_.free_radius, rows, rows_room_[part]);
        return;
      }
      case Stage::divergence_of_p:
        operators::divergence(std::as_const(tilde_.p[c]).view(), bar_.view(), rows);
        scaled(weights[c], own(entries_of(std::as_const(bar_))), own(v_.entries()));
        data_.conjugate_terms(c, std::as_const(v_).view(), plan_.free_radius, rows, rows_room_[part]);
        return;
      case Stage::residual: {
        const Vector& divergence = divergences_[c];
        step_along(own(tilde_.p[c].x.entries()), 1.0, own(divergence.x.entries()), own(gradient_.x.entries()));
        step_along(own(tilde_.p[c].y.entries()), 1.0, own(divergence.y.entries()), own(gradient_.y.entries()));
        const std::span<double> squares = own(std::span<double>(squares_));
        std::ranges::fill(squares, 0.0);
        add_squares(1.0, own(std::as_const(gradient_).x.entries()), own(std::as_const(gradient_).y.entries()), squares);
        roots(squares, own(std::span<double>(second_squares_)));
        return;
      }
    }
  }

  // A part's rows of the norms of the pixels, and of TGV's div_2 r~ and the largest of
  // its norms, a row at a time.
  void norms_of(std::size_t part) noexcept {
    const operators::Rows rows = parts_[part];
    const std::size_t columns = data_.columns();
    const auto row = [columns](auto entries, std::size_t i) noexcept { return entries.subspan(i * columns, columns); };
    const std::span<double> room = differences_room_[part];
    const std::span<double> one = room.subspan(0, columns);
    const std::span<double> two = room.subspan(columns, columns);
    const std::span<double> three = room.subspan(2u * columns, columns);
    const std::span<double> squares = std::span<double>(squares_room_[part]).first(columns);
    const std::span<double> norms = std::span<double>(scales_room_[part]).first(columns);
    const std::span<const double> weights = plan_.channel_weights;
    const bool coupled = regularizer_.coupled;
    const bool tgv = model_ == Model::tgv;
    double most = 0.0;
    for (std::size_t i = rows.first; i < rows.end; ++i) {
      std::ranges::fill(squares, 0.0);
      const std::span<double> first_sums = row(std::span<double>(first_terms_), i);
      std::ranges::fill(first_sums, 0.0);
      for (std::size_t c = 0; c < channels_; ++c) {
        const double gamma = weights[c];
        operators::gradient(std::as_const(tilde_.x[c]).view(), i, one, two);
        if (tgv) {
          step_along(one, -1.0, row(std::as_const(tilde_.w[c]).x.entries(), i), one);
          step_along(two, -1.0, row(std::as_const(tilde_.w[c]).y.entries(), i), two);
        }
        add_squares(coupled ? gamma * gamma : 1.0, one, two, squares);
        if (!coupled) add_roots(gamma, squares, first_sums);
      }
      if (coupled) roots(squares, first_sums);
      if (!tgv) continue;
      std::ranges::fill(squares, 0.0);
      const std::span<double> second_sums = row(std::span<double>(second_terms_), i);
      std::ranges::fill(second_sums, 0.0);
      for (std::size_t c = 0; c < channels_; ++c) {
        const double gamma = weights[c];
        operators::symmetrized_gradient(std::as_const(tilde_.w[c]).view(), i, one, two, three);
        add_tensor_squares(coupled ? gamma * gamma : 1.0, one, two, three, squares);
        if (!coupled) add_roots(gamma, squares, second_sums);
      }
      if (coupled) roots(squares, second_sums);
      std::ranges::fill(squares, 0.0);
      for (std::size_t c = 0; c < channels_; ++c) {
        Vector& divergence = divergences_[c];
        const std::span<double> x = row(divergence.x.entries(), i);
        const std::span<double> y = row(divergence.y.entries(), i);
        operators::tensor_divergence(std::as_const(tilde_.r[c]).view(), i, x, y);
        add_squares(1.0, x, y, squares);
        if (!coupled) {
          roots(squares, norms);
          most = std::max(most, largest(norms));
          std::ranges::fill(squares, 0.0);
        }
      }
      if (coupled) {
        roots(squares, norms);
        most = std::max(most, largest(norms));
      }
    }
    most_of_parts_[part] = most;
  }

  Data& data_;
  Model model_;
  const Regularizer& regularizer_;
  const Plan& plan_;
  std::size_t channels_;
  // Whether the channels' balls are coupled, of several: every channel's steps of a row
  // are made before the row's duals are projected.
  bool coupled_;
  State state_;
  State tilde_;
  GridArray<double> v_;
  GridArray<double> bar_;
  Vector gradient_;
  std::vector<double> squares_;
  std::vector<double> second_squares_;
  // A record's terms of the norms of the pixels, of the first order and of TGV's
  // second, TGV's div_2 r~ of each channel, and each part's largest norm of it.
  std::vector<double> first_terms_;
  std::vector<double> second_terms_;
  std::vector<Vector> divergences_;
  std::vector<double> most_of_parts_;
  // The parts of the canvas, and their own rooms: a row of the canvas for the proximal
  // map, three of a row's differences, and rows of the squares and the scales of the
  // coupled projections.
  std::vector<operators::Rows> parts_;
  std::vector<std::vector<double>> rows_room_;
  std::vector<std::vector<double>> differences_room_;
  std::vector<std::vector<double>> squares_room_;
  std::vector<std::vector<double>> scales_room_;
  // Each part's rings of the extrapolations, its rows that the parts beside read (x- of
  // its first row, w- of its last), and, coupled, its row's steps of every channel and
  // the squares and scales of the balls.
  std::vector<std::vector<double>> rings_room_;
  std::vector<std::vector<double>> edges_;
  std::vector<std::vector<double>> coupled_room_;
  std::function<void(std::size_t)> task_;
  std::function<void(std::size_t)> record_task_;
  // Whether the iteration keeps the tilde point's duals, and the stage of a record, its
  // channel and theta, which the threads read.
  bool keep_ = true;
  Stage stage_ = Stage::norms;
  std::size_t stage_channel_ = 0;
  double theta_ = 1.0;
  // Last, so that its threads are joined before the fields they take go.
  Workers workers_;
};

// What a run of the method made: the last tilde point's canvas, the records, the
// iterations taken and why it stopped.
struct Run {
  std::vector<GridArray<double>> canvas;
  std::vector<Record> records;
  std::uint64_t iterations = 0;
  Stop stop = Stop::iterations;
};

// The method on the data, from the start given or the data's own. Records are taken
// at the start (unless only the last is), every record_every iterations, and at the
// last; the method stops at the first record that meets a tolerance, or after the
// most iterations, or where the observer asks it to.
template <typename Data>
[[nodiscard]] std::expected<Run, Error> run(Data& data, Model model, const Regularizer& regularizer, const Plan& chosen,
                                            const Start* start, const Observer& observer) {
  const auto began = std::chrono::steady_clock::now();
  const std::size_t channels = data.channels();
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
  Machine<Data> machine(data, model, regularizer, chosen);
  if (std::expected<void, Error> started = machine.start(start); !started) {
    return std::unexpected(started.error());
  }
  Run result;
  const double samples =
      static_cast<double>(channels) * static_cast<double>(data.rows()) * static_cast<double>(data.columns());
  const std::uint64_t most = chosen.iterations;
  const std::uint64_t every = chosen.record_every;
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
    const std::uint64_t next = iteration + 1u;
    machine.iterate((every > 0u && next % every == 0u) || next == most);
    iteration = next;
    if ((every > 0u && iteration % every == 0u) || iteration == most) stopped = take(iteration);
  }
  result.iterations = iteration;
  result.stop = stopped ? *stopped : Stop::iterations;
  result.canvas = machine.take_canvas();
  return result;
}

}  // namespace unround::pdhg::detail

#endif  // UNROUND_PRIMAL_DUAL_HPP
