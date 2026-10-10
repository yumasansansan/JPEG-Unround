// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Tests of the subgradient method (docs/math.md, 7): its records, of the start, every
// few iterations and of the last, with no dual value; its iterates within the set; its
// stop where the subgradient is 0 (a flat picture without a data term); the
// extrapolation and the price of a slack, each of which changes the iterates; its
// observer; and the frames and options it refuses. Its agreement with the reference is
// the conformance case's.

#include "unround/subgradient.hpp"

#include "unround/arrays.hpp"
#include "unround/frames.hpp"
#include "unround/model.hpp"
#include "unround/results.hpp"

#include "support/check.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <expected>
#include <limits>
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
using unround::test::Numbers;

struct Picture {
  std::vector<BlocksArray<std::int16_t>> levels;
  std::vector<ComponentInput> inputs;
};

// A greyscale picture of 3 x 4 blocks of random levels, or of the same DC level and
// no AC ones (flat).
Picture grey_of(Numbers& numbers, bool flat) {
  Picture picture;
  BlocksArray<std::int16_t> levels(unround::block_extents(3, 4));
  const unround::Blocks<std::int16_t> view = levels.view();
  for (std::size_t by = 0; by < 3u; ++by) {
    for (std::size_t bx = 0; bx < 4u; ++bx) {
      view[by, bx, 0, 0] = flat ? std::int16_t{5} : static_cast<std::int16_t>(numbers.between(-40, 40));
      if (flat) continue;
      for (int k = 0; k < 6; ++k) {
        const auto v = static_cast<std::size_t>(numbers.between(0, 3));
        const auto w = static_cast<std::size_t>(numbers.between(v == 0u ? 1 : 0, 3));
        view[by, bx, v, w] = static_cast<std::int16_t>(numbers.between(-5, 5));
      }
    }
  }
  picture.levels.push_back(std::move(levels));
  std::array<std::uint16_t, 64> table{};
  for (std::size_t k = 0; k < 64u; ++k) table[k] = static_cast<std::uint16_t>(6u + k / 4u);
  picture.inputs.push_back(
      ComponentInput{.levels = std::as_const(picture.levels[0]).view(), .table = table, .h = 1, .v = 1});
  return picture;
}

Frame frame_of(const Picture& picture, const DataTerm& data) {
  const std::array<DataTerm, 1> terms{data};
  auto frame = Frame::make(24, 32, ColourSpace::grayscale, picture.inputs, terms);
  if (!frame) {
    std::println(stderr, "{}", frame.error().message);
    std::abort();
  }
  return std::move(*frame);
}

unround::subgradient::Result solved(const Frame& frame, const unround::subgradient::Options& options,
                                    const unround::subgradient::Observer& observer = {}) {
  auto result = unround::subgradient::solve(frame, 1.0, options, observer);
  if (!result) {
    std::println(stderr, "{}", result.error().message);
    std::abort();
  }
  return std::move(*result);
}

bool same_canvas(const unround::subgradient::Result& a, const unround::subgradient::Result& b) {
  return std::ranges::equal(a.canvas.at(0).entries(), b.canvas.at(0).entries());
}

void records(void) {
  Numbers numbers(31);
  const Picture picture = grey_of(numbers, false);
  DataTerm data;
  data.mu = 0.01;
  const Frame frame = frame_of(picture, data);
  unround::subgradient::Options options;
  options.iterations = 12;
  const unround::subgradient::Result result = solved(frame, options);
  CHECK(result.stop == unround::Stop::iterations && result.iterations == 12u && result.records.size() == 13u);
  bool scheduled = true;
  for (std::size_t k = 0; k < result.records.size(); ++k) {
    const unround::Record& record = result.records[k];
    scheduled = scheduled && record.iteration == k && std::isfinite(record.primal) &&
                record.dual == -std::numeric_limits<double>::infinity() && !record.scaling && !record.partial_gap;
  }
  CHECK(scheduled);
  // The last iterate is a projection: its coefficients lie within their intervals.
  const unround::Problem& problem = frame.component(0).problem;
  BlocksArray<double> excess(problem.extents());
  problem.excess(result.coefficients.at(0).view(), excess.view());
  CHECK(std::ranges::all_of(std::as_const(excess).entries(), [](double value) { return value == 0.0; }));
  // Records every 5 iterations and after the last; only after the last.
  options.record_every = 5;
  const unround::subgradient::Result sparse = solved(frame, options);
  CHECK(sparse.records.size() == 4u && sparse.records[1].iteration == 5u && sparse.records[3].iteration == 12u);
  CHECK(same_canvas(result, sparse));
  options.record_every = 0;
  const unround::subgradient::Result last = solved(frame, options);
  CHECK(last.records.size() == 1u && last.records[0].iteration == 12u);
  // The extrapolation changes the iterates.
  options.momentum = false;
  CHECK(!same_canvas(result, solved(frame, options)));
  // The observer stops the method at the record where it asks to.
  options.momentum = true;
  options.record_every = 1;
  const unround::subgradient::Result observed =
      solved(frame, options, [](const unround::Record& record, std::span<const GridArray<double>> canvas) {
        return record.iteration == 4u && canvas.size() == 1u;
      });
  CHECK(observed.stop == unround::Stop::observer && observed.iterations == 4u && observed.records.size() == 5u);
}

// A flat picture without a data term: the start is flat, its gradient 0 and the data
// term's part 0, and the method stops there.
void stationary(void) {
  Numbers numbers(32);
  const Picture picture = grey_of(numbers, true);
  DataTerm data;
  data.mu = 0.0;
  const Frame frame = frame_of(picture, data);
  const unround::subgradient::Result result = solved(frame, {});
  CHECK(result.stop == unround::Stop::stationary && result.iterations == 0u && result.records.size() == 1u);
}

// The price of a slack: with a slack the projection lets the coefficients past the
// file's own intervals, which the price's part of the subgradient pushes back.
void price(void) {
  Numbers numbers(33);
  const Picture picture = grey_of(numbers, false);
  DataTerm free_slack;
  free_slack.mu = 0.01;
  free_slack.slack = 0.5;
  DataTerm priced = free_slack;
  priced.slack_cost = 50.0;
  unround::subgradient::Options options;
  options.iterations = 20;
  const unround::subgradient::Result without = solved(frame_of(picture, free_slack), options);
  const unround::subgradient::Result with = solved(frame_of(picture, priced), options);
  CHECK(!same_canvas(without, with));
}

void refusals(void) {
  Numbers numbers(34);
  const Picture picture = grey_of(numbers, false);
  const Frame frame = frame_of(picture, DataTerm{});
  unround::subgradient::Options options;
  options.step = 0.0;
  CHECK(!unround::subgradient::solve(frame, 1.0, options));
  options.step = 0.5;
  options.decay = -1.0;
  CHECK(!unround::subgradient::solve(frame, 1.0, options));
  CHECK(!unround::subgradient::solve(frame, 0.0, {}));
  // A picture in colour.
  std::vector<BlocksArray<std::int16_t>> levels;
  levels.emplace_back(unround::block_extents(2, 4));
  levels.emplace_back(unround::block_extents(1, 2));
  levels.emplace_back(unround::block_extents(1, 2));
  std::vector<ComponentInput> inputs;
  for (std::size_t c = 0; c < 3u; ++c) {
    std::array<std::uint16_t, 64> table{};
    table.fill(8);
    const std::uint32_t factor = c == 0u ? 2u : 1u;
    inputs.push_back(
        ComponentInput{.levels = std::as_const(levels[c]).view(), .table = table, .h = factor, .v = factor});
  }
  const std::array<DataTerm, 1> data{DataTerm{}};
  const auto colour = Frame::make(13, 21, ColourSpace::ycbcr, inputs, data);
  CHECK(colour.has_value() && !unround::subgradient::solve(*colour, 1.0, {}));
}

int run(void) {
  records();
  stationary();
  price();
  refusals();
  return unround::test::finish("subgradient");
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
