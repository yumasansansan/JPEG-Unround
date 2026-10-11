// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Tests of the primal-dual method (docs/math.md, 5 and 6): the steps of the options to
// the last bit, the defaults of 6.5 and their scaling by the weight, and the options
// refused; TV and TGV on small frames, in grey and in 4:2:0, coupled and apart, the
// gap of every record at least 0 (the primal value of a tilde point is at least the
// least value, which is at least the dual value of a feasible dual), and the method
// stopping where it meets its tolerance, with a result whose coefficients lie within
// their intervals; the method on several threads, whose results are those of one to
// the last bit; a start given, which the method takes as the default one where it is
// that one, and whose dual fields it projects; an observer that stops the method; and
// no iterations.

#include "unround/pdhg.hpp"

#include "unround/arrays.hpp"
#include "unround/dct.hpp"
#include "unround/frames.hpp"
#include "unround/model.hpp"
#include "unround/operators.hpp"

#include "support/check.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <expected>
#include <optional>
#include <print>
#include <span>
#include <utility>
#include <vector>

namespace {

using unround::BlocksArray;
using unround::ColourSpace;
using unround::ComponentInput;
using unround::DataTerm;
using unround::Frame;
using unround::GridArray;
using unround::pdhg::Model;
using unround::pdhg::Options;
using unround::pdhg::Regularizer;
using unround::test::Numbers;

bool same(double a, double b) noexcept { return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b); }

void plans(void) {
  Regularizer regularizer;
  Options options;
  {
    const auto plan = unround::pdhg::plan(Model::tv, regularizer, options, 1);
    CHECK(plan.has_value());
    CHECK(same(plan->steps.tau, std::sqrt((10.0 * 0.99) / 8.0)));
    CHECK(same(plan->steps.sigma, std::sqrt(0.99 / (10.0 * 8.0))));
    CHECK(plan->tolerance == 2e-4 && plan->iterations == 5000u && plan->relaxation == 1.9);
    CHECK(plan->record_every == 10u && plan->free_radius == 255.0);
  }
  {
    const auto plan = unround::pdhg::plan(Model::tgv, regularizer, options, 3);
    CHECK(plan.has_value());
    const double bound = (17.0 + std::sqrt(33.0)) / 2.0;
    CHECK(same(plan->steps.tau, std::sqrt((3.0 * 0.99) / bound)));
    CHECK(same(plan->steps.sigma, std::sqrt(0.99 / (3.0 * bound))));
    CHECK(plan->tolerance == 2e-3 && plan->iterations == 10000u);
    CHECK(plan->channel_weights == std::vector<double>(3, 1.0));
  }
  {
    // The defaults scale with the weight; the values given do not.
    Regularizer weighted;
    weighted.alpha = 2.0;
    weighted.alpha1 = 4.0;
    const auto tv = unround::pdhg::plan(Model::tv, weighted, options, 1);
    CHECK(same(tv->steps.tau, std::sqrt((2.5 * 0.99) / 8.0)) && tv->tolerance == 4e-4);
    const auto tgv = unround::pdhg::plan(Model::tgv, weighted, options, 1);
    CHECK(tgv->tolerance == 8e-3);
    Options unscaled;
    unscaled.weight_scaling = false;
    const auto plain = unround::pdhg::plan(Model::tv, weighted, unscaled, 1);
    CHECK(same(plain->steps.tau, std::sqrt((10.0 * 0.99) / 8.0)) && plain->tolerance == 2e-4);
    Options given;
    given.step_ratio = 7.0;
    given.tolerance = 1e-6;
    given.iterations = 12;
    given.step_product = 0.5;
    given.norm_squared = 9.0;
    const auto chosen = unround::pdhg::plan(Model::tv, weighted, given, 1);
    CHECK(same(chosen->steps.tau, std::sqrt((7.0 * 0.5) / 9.0)) && chosen->tolerance == 1e-6);
    CHECK(chosen->iterations == 12u);
    // The channel weights raise L^2 by the largest squared.
    Regularizer channels;
    channels.channel_weights = {1.0, 2.0, 0.5};
    const auto raised = unround::pdhg::plan(Model::tv, channels, options, 3);
    CHECK(same(raised->steps.tau, std::sqrt((10.0 * 0.99) / 32.0)));
  }
  {
    Options relaxed;
    relaxed.relaxation = 2.0;
    CHECK(!unround::pdhg::plan(Model::tv, regularizer, relaxed, 1));
    Options product;
    product.step_product = 1.0;
    CHECK(!unround::pdhg::plan(Model::tv, regularizer, product, 1));
    Regularizer zero;
    zero.alpha0 = 0.0;
    CHECK(!unround::pdhg::plan(Model::tgv, zero, options, 1));
    Options partial;
    partial.partial_radius = 10.0;
    CHECK(!unround::pdhg::plan(Model::tv, regularizer, partial, 1));
    CHECK(unround::pdhg::plan(Model::tgv, regularizer, partial, 1).has_value());
    Options alone;
    alone.partial_tolerance = 1e-3;
    CHECK(!unround::pdhg::plan(Model::tgv, regularizer, alone, 1));
    Regularizer counted;
    counted.channel_weights = {1.0, 1.0};
    CHECK(!unround::pdhg::plan(Model::tv, counted, options, 3));
    Options negative;
    negative.tolerance = -1.0;
    CHECK(!unround::pdhg::plan(Model::tv, regularizer, negative, 1));
    // Every error is named at once.
    Options both;
    both.relaxation = 3.0;
    both.step_product = 2.0;
    const auto refused = unround::pdhg::plan(Model::tv, regularizer, both, 1);
    CHECK(!refused && refused.error().message.find("relaxation") != std::string::npos &&
          refused.error().message.find("product") != std::string::npos);
  }
}

BlocksArray<std::int16_t> levels_of(Numbers& numbers, std::size_t rows, std::size_t columns) {
  BlocksArray<std::int16_t> levels(unround::block_extents(rows, columns));
  const unround::Blocks<std::int16_t> view = levels.view();
  for (std::size_t by = 0; by < rows; ++by) {
    for (std::size_t bx = 0; bx < columns; ++bx) {
      view[by, bx, 0, 0] = static_cast<std::int16_t>(numbers.between(-40, 40));
      for (int k = 0; k < 5; ++k) {
        const auto v = static_cast<std::size_t>(numbers.between(0, 2));
        const auto w = static_cast<std::size_t>(numbers.between(v == 0u ? 1 : 0, 2));
        view[by, bx, v, w] = static_cast<std::int16_t>(numbers.between(-4, 4));
      }
    }
  }
  return levels;
}

struct Picture {
  std::vector<BlocksArray<std::int16_t>> levels;
  std::vector<ComponentInput> inputs;
};

// A grey picture of 2 x 3 blocks, or one in 4:2:0 of 13 x 21 samples.
Picture picture_of(Numbers& numbers, bool colour) {
  Picture picture;
  if (!colour) {
    picture.levels.push_back(levels_of(numbers, 2, 3));
  } else {
    picture.levels.push_back(levels_of(numbers, 2, 4));
    picture.levels.push_back(levels_of(numbers, 1, 2));
    picture.levels.push_back(levels_of(numbers, 1, 2));
  }
  for (std::size_t c = 0; c < picture.levels.size(); ++c) {
    std::array<std::uint16_t, 64> table{};
    for (std::size_t k = 0; k < 64u; ++k) table[k] = static_cast<std::uint16_t>(4u + 2u * (k / 8u + k % 8u));
    const std::uint32_t factor = colour && c == 0u ? 2u : 1u;
    picture.inputs.push_back(
        ComponentInput{.levels = std::as_const(picture.levels[c]).view(), .table = table, .h = factor, .v = factor});
  }
  return picture;
}

Frame frame_of(const Picture& picture, bool colour) {
  DataTerm data;
  data.mu = 0.05;
  data.flat_weight = 0.25;
  const std::array<DataTerm, 1> terms{data};
  auto frame = colour ? Frame::make(13, 21, ColourSpace::ycbcr, picture.inputs, terms)
                      : Frame::make(16, 24, ColourSpace::grayscale, picture.inputs, terms);
  if (!frame) {
    std::println(stderr, "{}", frame.error().message);
    std::abort();
  }
  return std::move(*frame);
}

// The result of a solve, which the test cannot go on without.
unround::pdhg::Result solved(const Frame& frame, Model model, const Regularizer& regularizer, const Options& options,
                             const unround::pdhg::Start* start = nullptr,
                             const unround::pdhg::Observer& observer = {}) {
  auto result = unround::pdhg::solve(frame, model, regularizer, options, start, observer);
  if (!result) {
    std::println(stderr, "{}", result.error().message);
    std::abort();
  }
  return std::move(*result);
}

// The gap of every record at least 0, the last within the tolerance per sample, the
// stop a convergence at the record of the last iteration, and the coefficients of the
// result within their intervals, the canvas their inverse DCT where the cells have one
// sample. The gap at the stop, the tolerance times hundreds to thousands of samples,
// is at least 0.03, and the rounding of the values, hundreds to thousands summed over
// thousands of terms, below 1e-9: the gaps are taken as they are.
void converges(const Frame& frame, Model model, const Regularizer& regularizer, double tolerance) {
  Options options;
  options.tolerance = tolerance;
  options.iterations = 50000;
  const unround::pdhg::Result result = solved(frame, model, regularizer, options);
  CHECK(result.stop == unround::pdhg::Stop::converged);
  CHECK(!result.records.empty() && result.records.front().iteration == 0u);
  CHECK(result.records.back().iteration == result.iterations);
  const double samples = static_cast<double>(frame.channels() * frame.rows() * frame.columns());
  bool gaps = true;
  bool scheduled = true;
  for (std::size_t k = 0; k < result.records.size(); ++k) {
    const unround::pdhg::Record& record = result.records[k];
    gaps = gaps && record.primal - record.dual >= 0.0;
    scheduled = scheduled && record.iteration == 10u * k;
    if (model == Model::tgv) {
      CHECK(record.scaling.has_value() && *record.scaling > 0.0 && *record.scaling <= 1.0);
    } else {
      CHECK(!record.scaling.has_value());
    }
  }
  CHECK(gaps);
  CHECK(scheduled);
  const unround::pdhg::Record& last = result.records.back();
  CHECK((last.primal - last.dual) / samples <= tolerance);
  CHECK(result.records.size() > 2u && result.records[1].primal - result.records[1].dual > last.primal - last.dual);
  for (std::size_t c = 0; c < frame.channels(); ++c) {
    const unround::Problem& problem = frame.component(c).problem;
    BlocksArray<double> excess(result.coefficients[c].extents());
    problem.excess(result.coefficients[c].view(), excess.view());
    CHECK(std::ranges::all_of(std::as_const(excess).entries(), [](double value) { return value == 0.0; }));
    if (frame.component(c).cells() == 1u) {
      GridArray<double> back(frame.canvas_extents(), 0.0);
      unround::dct::inverse(result.coefficients[c].view(), back.view());
      bool blocks = true;
      const std::size_t rows = 8u * problem.rows();
      const std::size_t columns = 8u * problem.columns();
      for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < columns; ++j) {
          blocks = blocks && same(result.canvas[c].view()[i, j], back.view()[i, j]);
        }
      }
      CHECK(blocks);
    }
  }
}

void solves(void) {
  Numbers numbers(21);
  const Picture grey = picture_of(numbers, false);
  const Frame grey_frame = frame_of(grey, false);
  const Picture colour = picture_of(numbers, true);
  const Frame colour_frame = frame_of(colour, true);
  Regularizer coupled;
  Regularizer apart;
  apart.coupled = false;
  converges(grey_frame, Model::tv, coupled, 1e-4);
  converges(grey_frame, Model::tgv, coupled, 1e-3);
  converges(colour_frame, Model::tv, coupled, 1e-3);
  converges(colour_frame, Model::tv, apart, 1e-3);
  converges(colour_frame, Model::tgv, coupled, 1e-2);
  converges(colour_frame, Model::tgv, apart, 1e-2);
  Regularizer weighted;
  weighted.channel_weights = {1.0, 0.5, 0.5};
  weighted.alpha = 2.0;
  converges(colour_frame, Model::tv, weighted, 1e-3);
  // One channel: coupled and apart are the same method, to the last bit.
  Options options;
  options.iterations = 40;
  for (const Model model : {Model::tv, Model::tgv}) {
    const unround::pdhg::Result one = solved(grey_frame, model, coupled, options);
    const unround::pdhg::Result other = solved(grey_frame, model, apart, options);
    CHECK(std::ranges::equal(one.canvas[0].entries(), other.canvas[0].entries(), same));
    CHECK(one.records.size() == other.records.size());
    for (std::size_t k = 0; k < std::min(one.records.size(), other.records.size()); ++k) {
      CHECK(same(one.records[k].primal, other.records[k].primal) && same(one.records[k].dual, other.records[k].dual));
    }
  }
  // TGV's partial gap, recorded where a radius is given, and at least the gap of the
  // feasible dual's primal value less the dual value of the iterate's p: here only
  // that it is taken, and finite.
  Options partial;
  partial.iterations = 30;
  partial.partial_radius = 50.0;
  const unround::pdhg::Result with_radius = solved(grey_frame, Model::tgv, coupled, partial);
  CHECK(std::ranges::all_of(with_radius.records, [](const unround::pdhg::Record& record) {
    return record.partial_gap.has_value() && std::isfinite(*record.partial_gap);
  }));
}

// Whether two results are the same to the last bit: their canvases, coefficients and
// records (but the seconds), iterations and stops.
bool same_results(const unround::pdhg::Result& one, const unround::pdhg::Result& other) {
  bool equal = one.iterations == other.iterations && one.stop == other.stop &&
               one.canvas.size() == other.canvas.size() && one.coefficients.size() == other.coefficients.size() &&
               one.records.size() == other.records.size();
  for (std::size_t c = 0; equal && c < one.canvas.size(); ++c) {
    equal = std::ranges::equal(one.canvas[c].entries(), other.canvas[c].entries(), same) &&
            std::ranges::equal(one.coefficients[c].entries(), other.coefficients[c].entries(), same);
  }
  const auto same_optional = [](const std::optional<double>& a, const std::optional<double>& b) {
    return a.has_value() == b.has_value() && (!a || same(*a, *b));
  };
  for (std::size_t k = 0; equal && k < one.records.size(); ++k) {
    const unround::pdhg::Record& a = one.records[k];
    const unround::pdhg::Record& b = other.records[k];
    equal = a.iteration == b.iteration && same(a.primal, b.primal) && same(a.dual, b.dual) &&
            same_optional(a.scaling, b.scaling) && same_optional(a.partial_gap, b.partial_gap);
  }
  return equal;
}

// The method on several threads, which take the canvas in parts of whole bands, ends
// where it ends on one, to the last bit: TV and TGV, a grey picture of 7 bands, one in
// 4:2:0 of 50 x 45 samples (4 bands of 16 rows, the luma's blocks ending a block row
// before the canvas does), and one of 8 x 20 samples whose components are all sampled
// 4 x 4 (4 bands of 8 rows in an MCU of 32, three of which hold no block), their
// channels coupled and apart; more threads than bands too.
void threads(void) {
  Numbers numbers(29);
  Picture grey;
  grey.levels.push_back(levels_of(numbers, 7, 5));
  Picture colour;
  colour.levels.push_back(levels_of(numbers, 7, 6));
  colour.levels.push_back(levels_of(numbers, 4, 3));
  colour.levels.push_back(levels_of(numbers, 4, 3));
  Picture sparse;
  for (std::size_t c = 0; c < 3u; ++c) sparse.levels.push_back(levels_of(numbers, 1, 3));
  for (Picture* picture : {&grey, &colour, &sparse}) {
    for (std::size_t c = 0; c < picture->levels.size(); ++c) {
      std::array<std::uint16_t, 64> table{};
      for (std::size_t k = 0; k < 64u; ++k) table[k] = static_cast<std::uint16_t>(4u + 2u * (k / 8u + k % 8u));
      const std::uint32_t factor = picture == &sparse ? 4u : (picture == &colour && c == 0u ? 2u : 1u);
      picture->inputs.push_back(
          ComponentInput{.levels = std::as_const(picture->levels[c]).view(), .table = table, .h = factor, .v = factor});
    }
  }
  DataTerm data;
  data.mu = 0.05;
  data.flat_weight = 0.25;
  const std::array<DataTerm, 1> terms{data};
  auto grey_frame = Frame::make(56, 40, ColourSpace::grayscale, grey.inputs, terms);
  auto colour_frame = Frame::make(50, 45, ColourSpace::ycbcr, colour.inputs, terms);
  auto sparse_frame = Frame::make(8, 20, ColourSpace::ycbcr, sparse.inputs, terms);
  CHECK(grey_frame.has_value() && colour_frame.has_value() && sparse_frame.has_value());
  if (!grey_frame || !colour_frame || !sparse_frame) return;
  CHECK(colour_frame->rows() == 64u && colour_frame->band() == 16u && grey_frame->band() == 8u);
  CHECK(sparse_frame->rows() == 32u && sparse_frame->band() == 8u && sparse_frame->component(0).block_rows() == 1u);
  Regularizer coupled;
  Regularizer apart;
  apart.coupled = false;
  Options options;
  options.iterations = 23;
  options.tolerance = 0.0;
  options.record_every = 5;
  options.partial_radius = 50.0;
  const std::array<std::pair<const Frame*, const Regularizer*>, 5> cases{{{&*grey_frame, &coupled},
                                                                          {&*colour_frame, &coupled},
                                                                          {&*colour_frame, &apart},
                                                                          {&*sparse_frame, &coupled},
                                                                          {&*sparse_frame, &apart}}};
  for (const auto& [frame, regularizer] : cases) {
    for (const Model model : {Model::tv, Model::tgv}) {
      Options chosen = options;
      if (model == Model::tv) chosen.partial_radius.reset();
      chosen.threads = 1;
      const unround::pdhg::Result one = solved(*frame, model, *regularizer, chosen);
      for (const std::uint64_t count : {2u, 3u, 4u, 9u, 0u}) {
        chosen.threads = count;
        CHECK(same_results(one, solved(*frame, model, *regularizer, chosen)));
      }
    }
  }
}

// A start given as the default one gives the default run, to the last bit; dual fields
// given are projected onto their balls first; an observer stops the method at the
// record where it asks to; and without iterations the result is the start.
void starts(void) {
  Numbers numbers(22);
  const Picture colour = picture_of(numbers, true);
  const Frame frame = frame_of(colour, true);
  const Regularizer regularizer;
  Options options;
  options.iterations = 20;
  for (const Model model : {Model::tv, Model::tgv}) {
    const unround::pdhg::Result plain = solved(frame, model, regularizer, options);
    unround::pdhg::Start start;
    for (std::size_t c = 0; c < frame.channels(); ++c) {
      const unround::Problem& problem = frame.component(c).problem;
      start.coefficients.emplace_back(problem.extents());
      std::ranges::copy(std::span<const double>(problem.centres().data_handle(), problem.centres().size()),
                        start.coefficients.back().entries().begin());
      start.p.push_back(unround::operators::vector_of(frame.rows(), frame.columns()));
    }
    const unround::pdhg::Result given = solved(frame, model, regularizer, options, &start);
    bool equal = true;
    for (std::size_t c = 0; c < frame.channels(); ++c) {
      equal = equal && std::ranges::equal(plain.canvas[c].entries(), given.canvas[c].entries(), same);
    }
    CHECK(equal);
    // Large fields p, which the method projects onto the ball of alpha before it starts:
    // the dual value of TV's start is that of the fields projected, to the last bit
    // (TGV's does not take p).
    if (model == Model::tv) {
      unround::pdhg::Start large = start;
      for (unround::operators::Vector& p : large.p) {
        for (double& entry : p.x.entries()) entry = numbers.uniform(-5.0, 5.0);
        for (double& entry : p.y.entries()) entry = numbers.uniform(-5.0, 5.0);
      }
      std::vector<unround::operators::Vector> projected = large.p;
      std::vector<unround::operators::VectorView<double>> views;
      views.reserve(projected.size());
      for (unround::operators::Vector& p : projected) views.push_back(p.view());
      unround::operators::project(views, regularizer.alpha, true);
      Options none;
      none.iterations = 0;
      none.tolerance = 0.0;
      const unround::pdhg::Result idle = solved(frame, model, regularizer, none, &large);
      double conjugates = 0.0;
      for (std::size_t c = 0; c < frame.channels(); ++c) {
        GridArray<double> xi(frame.canvas_extents());
        unround::operators::divergence(std::as_const(projected[c]).view(), xi.view());
        conjugates += frame.conjugate(c, std::as_const(xi).view(), 255.0);
      }
      CHECK(same(idle.records.front().dual, -conjugates));
    }
  }
  // TGV's feasible dual (6.2) at a start whose r lies within its ball (|r|_F <= 1 <
  // alpha0 = 2, which the projection leaves as it is) and whose div_2 r does not lie
  // within that of alpha1: the scaling is alpha1 over the largest norm of div_2 r, the
  // channels coupled. Both computations of that norm round each square, sum and root
  // within gamma_5 / 2 + u of its value, and the quotient within u more; they lie within
  // the sum of their bounds of each other.
  {
    unround::pdhg::Start start;
    for (std::size_t c = 0; c < frame.channels(); ++c) {
      unround::operators::Tensor r = unround::operators::tensor_of(frame.rows(), frame.columns());
      for (double& entry : r.xx.entries()) entry = numbers.uniform(-0.5, 0.5);
      for (double& entry : r.yy.entries()) entry = numbers.uniform(-0.5, 0.5);
      for (double& entry : r.xy.entries()) entry = numbers.uniform(-0.5, 0.5);
      start.r.push_back(std::move(r));
    }
    Options none;
    none.iterations = 0;
    none.tolerance = 0.0;
    const unround::pdhg::Result idle = solved(frame, Model::tgv, regularizer, none, &start);
    std::vector<double> squares(frame.rows() * frame.columns(), 0.0);
    for (const unround::operators::Tensor& r : start.r) {
      unround::operators::Vector divergence = unround::operators::vector_of(frame.rows(), frame.columns());
      unround::operators::tensor_divergence(r.view(), divergence.view());
      const std::span<const double> x = std::as_const(divergence.x).entries();
      const std::span<const double> y = std::as_const(divergence.y).entries();
      for (std::size_t k = 0; k < squares.size(); ++k) squares[k] += x[k] * x[k] + y[k] * y[k];
    }
    double most = 0.0;
    for (const double square : squares) most = std::max(most, std::sqrt(square));
    CHECK(most > regularizer.alpha1);
    const double theta = regularizer.alpha1 / most;
    const double bound = (unround::test::gamma(5) + 4.0 * unround::test::u) * theta;
    CHECK(idle.records.front().scaling.has_value() &&
          std::abs(*idle.records.front().scaling - theta) <= bound * (1.0 + unround::test::gamma(2)));
  }
  // The observer.
  int calls = 0;
  const unround::pdhg::Observer stop_at_twenty = [&calls](const unround::pdhg::Record& record,
                                                          std::span<const GridArray<double>> canvas) {
    ++calls;
    return record.iteration == 20u && canvas.size() == 3u;
  };
  Options long_run;
  long_run.iterations = 100;
  long_run.tolerance = 0.0;
  const unround::pdhg::Result observed = solved(frame, Model::tv, regularizer, long_run, nullptr, stop_at_twenty);
  CHECK(observed.stop == unround::pdhg::Stop::observer && observed.iterations == 20u && calls == 3);
  // No iterations: one record, of the start, and the start as the result.
  Options none;
  none.iterations = 0;
  none.tolerance = 0.0;
  const unround::pdhg::Result idle = solved(frame, Model::tgv, regularizer, none);
  CHECK(idle.iterations == 0u && idle.records.size() == 1u && idle.stop == unround::pdhg::Stop::iterations);
  for (std::size_t c = 0; c < frame.channels(); ++c) {
    GridArray<double> x(frame.canvas_extents());
    BlocksArray<double> kept(frame.component(c).problem.extents());
    frame.start(c, x.view(), kept.view());
    CHECK(std::ranges::equal(idle.canvas[c].entries(), std::as_const(x).entries(), same));
  }
  // Records only at the last iteration.
  Options last;
  last.iterations = 25;
  last.record_every = 0;
  last.tolerance = 0.0;
  const unround::pdhg::Result sparse = solved(frame, Model::tv, regularizer, last);
  CHECK(sparse.records.size() == 1u && sparse.records.front().iteration == 25u);
  // Fields of another shape are refused.
  unround::pdhg::Start wrong;
  wrong.p.push_back(unround::operators::vector_of(3, 3));
  wrong.p.push_back(unround::operators::vector_of(3, 3));
  wrong.p.push_back(unround::operators::vector_of(3, 3));
  CHECK(!unround::pdhg::solve(frame, Model::tv, regularizer, options, &wrong));
}

int run(void) {
  plans();
  solves();
  threads();
  starts();
  return unround::test::finish("pdhg");
}

}  // namespace

int main(void) {
  try {
    return run();
  } catch (const std::exception& error) {
    (void)std::fputs(error.what(), stderr);
    (void)std::fputc('\n', stderr);
    return EXIT_FAILURE;
  }
}
