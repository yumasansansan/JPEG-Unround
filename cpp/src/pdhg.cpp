// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The primal-dual method of docs/math.md, 5, and its records and stops, 6, on the
// components of a JPEG file: the plan of its options, and the data term of a Frame,
// whose iterations primal_dual.hpp has.

#include "unround/pdhg.hpp"

#include "unround/arrays.hpp"
#include "unround/dct.hpp"
#include "unround/error.hpp"
#include "unround/exact.hpp"
#include "unround/frames.hpp"
#include "unround/model.hpp"
#include "unround/operators.hpp"

#include "primal_dual.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <expected>
#include <format>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace unround::pdhg {

namespace {

bool positive(double value) noexcept { return value > 0.0 && std::isfinite(value); }
bool at_least_zero(double value) noexcept { return value >= 0.0 && std::isfinite(value); }

// The bound of ||K||^2 of 3.3: 8 for TV, (17 + sqrt 33) / 2 for TGV, which is the
// double nearest to it.
double norm_bound(Model model) noexcept { return model == Model::tv ? 8.0 : (17.0 + std::sqrt(33.0)) / 2.0; }

// The data term of a frame's components: the coefficients of each that its last
// proximal map put within their intervals (or its start), at which G is taken.
class FrameData {
 public:
  explicit FrameData(const Frame& frame) : frame_(frame) {
    for (std::size_t c = 0; c < frame.channels(); ++c) {
      coefficients_.emplace_back(frame.component(c).problem.extents());
      rooms_.push_back(frame.room(c));
      value_terms_.emplace_back(frame.component(c).problem.extents());
      conjugate_terms_.push_back(frame.conjugate_room(c));
    }
  }

  [[nodiscard]] std::size_t channels(void) const noexcept { return frame_.channels(); }
  [[nodiscard]] std::size_t rows(void) const noexcept { return frame_.rows(); }
  [[nodiscard]] std::size_t columns(void) const noexcept { return frame_.columns(); }
  [[nodiscard]] GridExtents extents(void) const noexcept { return frame_.canvas_extents(); }

  // The start of channel c: the canvas of the coefficients given, clipped to their
  // intervals, or the frame's own start.
  std::expected<void, Error> start(std::size_t c, const Start* given, Grid<double> x) {
    if (given != nullptr && !given->coefficients.empty()) {
      const BlocksArray<double>& from = given->coefficients[c];
      if (from.extents() != coefficients_[c].extents()) {
        return std::unexpected(Error{.code = Errc::options,
                                     .message = std::format("the start's coefficients of component {} are not of its "
                                                            "blocks",
                                                            c)});
      }
      frame_.component(c).problem.clip(from.view(), coefficients_[c].view());
      frame_.canvas(c, std::as_const(coefficients_[c]).view(), beyond_blocks, x);
    } else {
      frame_.start(c, x, coefficients_[c].view());
    }
    return {};
  }

  // The rows of the canvas of a part that a thread takes whole.
  [[nodiscard]] std::size_t band(void) const noexcept { return frame_.band(); }

  // The proximal map of channel c on the rows of a part of the canvas, whose ends are
  // ends of bands (or of the canvas): the blocks of the component that they cover, with
  // `row` a row of the canvas of the part's own.
  void prox_rows(std::size_t c, Grid<const double> v, double tau, Grid<double> out, operators::Rows rows,
                 std::vector<double>& row) {
    const Component& component = frame_.component(c);
    const auto [first, end] = block_rows_of(c, rows);
    const std::size_t columns = component.block_columns();
    const std::size_t width = v.extent(1);
    const Blocks<double> kept(coefficients_[c].view().data_handle() + (first * columns * dct::block_size), end - first,
                              columns);
    const Grid<const double> from(v.data_handle() + (rows.first * width), rows.end - rows.first, width);
    const Grid<double> to(out.data_handle() + (rows.first * width), rows.end - rows.first, width);
    frame_.prox_rows(c, from, tau, to, kept, rooms_[c], first, row);
  }

  // G at the coefficients of the last proximal map, and G* at xi, of channel c: their
  // terms of the rows of a part (value_terms, conjugate_terms, with `row` a row of the
  // canvas of the part's own), and their sums once every part has taken its own.
  void value_terms(std::size_t c, operators::Rows rows) {
    const Component& component = frame_.component(c);
    const auto [first, end] = block_rows_of(c, rows);
    const std::size_t columns = component.block_columns();
    const std::size_t offset = first * columns * dct::block_size;
    component.problem.value_terms(
        Blocks<const double>(std::as_const(coefficients_[c]).view().data_handle() + offset, end - first, columns),
        first, Blocks<double>(value_terms_[c].view().data_handle() + offset, end - first, columns));
  }

  [[nodiscard]] double value_sum(std::size_t c) const { return Problem::sum(value_terms_[c].view()); }

  void conjugate_terms(std::size_t c, Grid<const double> xi, double radius, operators::Rows rows,
                       std::vector<double>& row) {
    frame_.conjugate_terms(c, xi, radius, rows.first, rows.end, conjugate_terms_[c], row);
  }

  [[nodiscard]] double conjugate_sum(std::size_t c) const { return frame_.conjugate_sum(c, conjugate_terms_[c]); }

  std::vector<BlocksArray<double>> take_coefficients(void) { return std::move(coefficients_); }

 private:
  // The block rows [first, end) of channel c that the rows of a part cover, whose ends
  // are ends of bands or of the canvas. An end within a block row would leave the block
  // out: a mistake of the caller, which ends the program.
  [[nodiscard]] std::pair<std::size_t, std::size_t> block_rows_of(std::size_t c, operators::Rows rows) const noexcept {
    const Component& component = frame_.component(c);
    const std::size_t height = dct::block * component.cell_rows;
    const std::size_t blocks = component.block_rows();
    if (rows.first % height != 0u || (rows.end % height != 0u && rows.end != frame_.rows())) std::abort();
    return {std::min(rows.first / height, blocks), std::min(rows.end / height, blocks)};
  }

  const Frame& frame_;
  std::vector<BlocksArray<double>> coefficients_;
  std::vector<Room> rooms_;
  std::vector<BlocksArray<double>> value_terms_;
  std::vector<Frame::ConjugateTerms> conjugate_terms_;
};

}  // namespace

std::expected<Plan, Error> plan(Model model, const Regularizer& regularizer, const Options& options,
                                std::size_t channels, const Defaults& defaults) {
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
  const double default_ratio = model == Model::tv ? defaults.tv_ratio : defaults.tgv_ratio;
  const double default_tolerance = model == Model::tv ? defaults.tv_tolerance : defaults.tgv_tolerance;
  const double ratio = options.step_ratio       ? *options.step_ratio
                       : options.weight_scaling ? default_ratio / (weight * weight)
                                                : default_ratio;
  result.tolerance = options.tolerance        ? *options.tolerance
                     : options.weight_scaling ? default_tolerance * weight
                                              : default_tolerance;
  result.iterations = options.iterations   ? *options.iterations
                      : model == Model::tv ? defaults.tv_iterations
                                           : defaults.tgv_iterations;
  const double product = options.step_product;
  result.steps.tau = std::sqrt((ratio * product) / norm_squared);
  result.steps.sigma = std::sqrt(product / (ratio * norm_squared));
  result.relative_tolerance = options.relative_tolerance;
  result.partial_tolerance = options.partial_tolerance;
  result.partial_radius = options.partial_radius;
  result.relaxation = options.relaxation;
  result.record_every = options.record_every;
  result.free_radius = options.free_radius;
  if (options.threads == 0u) {
    result.threads = std::max<std::size_t>(1u, std::thread::hardware_concurrency());
  } else {
    result.threads = static_cast<std::size_t>(std::min<std::uint64_t>(options.threads, SIZE_MAX));
  }
  return result;
}

std::expected<Result, Error> solve(const Frame& frame, Model model, const Regularizer& regularizer,
                                   const Options& options, const Start* start, const Observer& observer) {
  std::expected<Plan, Error> made = plan(model, regularizer, options, frame.channels());
  if (!made) return std::unexpected(made.error());
  FrameData data(frame);
  std::expected<detail::Run, Error> ran = detail::run(data, model, regularizer, *made, start, observer);
  if (!ran) return std::unexpected(ran.error());
  Result result;
  result.canvas = std::move(ran->canvas);
  result.coefficients = data.take_coefficients();
  result.records = std::move(ran->records);
  result.iterations = ran->iterations;
  result.stop = ran->stop;
  result.steps = made->steps;
  return result;
}

}  // namespace unround::pdhg
