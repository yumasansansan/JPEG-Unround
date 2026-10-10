// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The conformance cases (conformance/cases, docs/math.md, 9, and conformance/README.md):
// each case's picture reconstructed with its options, from its start where it gives
// one, and compared with what the reference made of it. The intervals, the weights of
// the data term and the steps agree to the last bit; every centre lies within
// (2|q| + 20) u Q of its exact value where it is an MMSE one of an AC coefficient, and
// is that value elsewhere; the records are taken at the same iterations, each primal
// value within the case's tolerance of the reference's, each dual value at most its
// tolerance above the other's primal value, and the gaps within theirs; the method
// stops after the same iterations, for the same reason; and the canvas after the last
// iteration lies within the case's tolerance, in the Euclidean norm over all its
// samples.

#include "unround/arrays.hpp"
#include "unround/frames.hpp"
#include "unround/model.hpp"
#include "unround/operators.hpp"
#include "unround/pdhg.hpp"
#include "unround/results.hpp"
#include "unround/settings.hpp"
#include "unround/subgradient.hpp"

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
#include <filesystem>
#include <limits>
#include <map>
#include <optional>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using unround::BlocksArray;
using unround::GridArray;
using unround::test::Exact;
using unround::test::number;
using unround::test::u;

struct ComponentLine {
  std::uint32_t h = 1;
  std::uint32_t v = 1;
  std::size_t rows = 0;
  std::size_t columns = 0;
};

struct RecordLine {
  std::uint64_t iteration = 0;
  double primal = 0.0;
  double dual = 0.0;
};

struct Tolerance {
  double primal = 0.0;
  double dual = -1.0;
  double gap = -1.0;
};

struct Case {
  std::string name;
  std::size_t height = 0;
  std::size_t width = 0;
  unround::ColourSpace space = unround::ColourSpace::grayscale;
  std::vector<ComponentLine> components;
  std::vector<std::array<std::uint16_t, 64>> tables;
  std::vector<BlocksArray<std::int16_t>> levels;
  std::vector<std::string> options;
  std::optional<std::array<double, 2>> steps;
  std::vector<std::array<double, 64>> weights;
  std::vector<BlocksArray<double>> lower;
  std::vector<BlocksArray<double>> upper;
  std::vector<BlocksArray<double>> centres;
  std::vector<std::vector<Exact>> exact_centres;
  std::vector<BlocksArray<double>> first_coefficients;
  std::vector<std::vector<std::vector<double>>> first_p;  // [entry][channel] rows, flattened
  std::vector<std::vector<std::vector<double>>> first_w;
  std::vector<std::vector<std::vector<double>>> first_r;
  std::vector<RecordLine> records;
  std::vector<std::vector<double>> canvas;  // [channel] rows, flattened
  std::size_t canvas_columns = 0;
  std::uint64_t stop_iterations = 0;
  std::string stop_reason;
  double canvas_tolerance = 0.0;
  std::map<std::uint64_t, Tolerance> record_tolerances;
};

std::size_t index_of(const std::string& text) { return static_cast<std::size_t>(std::stoull(text)); }

// The values of a line from field `first` on.
std::vector<double> values_of(const std::vector<std::string>& fields, std::size_t first) {
  std::vector<double> values;
  for (std::size_t k = first; k < fields.size(); ++k) values.push_back(number(fields[k]));
  return values;
}

template <typename T>
T& at_least(std::vector<T>& items, std::size_t index) {
  if (items.size() <= index) items.resize(index + 1u);
  return items[index];
}

// A block row of a component's arrays of blocks from the values of a line.
template <typename T, typename Value>
void put_block_row(BlocksArray<T>& blocks, std::size_t by, std::span<const Value> values) {
  const std::size_t columns = blocks.extent(1);
  if (values.size() != 64u * columns || by >= blocks.extent(0)) {
    std::println(stderr, "a block row of {} values, not {}", 64u * columns, values.size());
    std::abort();
  }
  const unround::Blocks<T> view = blocks.view();
  for (std::size_t bx = 0; bx < columns; ++bx) {
    for (std::size_t k = 0; k < 64u; ++k) view[by, bx, k / 8u, k % 8u] = static_cast<T>(values[64u * bx + k]);
  }
}

// A field of fields given row by row: [entry][channel], rows flattened.
void put_field_row(std::vector<std::vector<std::vector<double>>>& field, const std::vector<std::string>& fields) {
  const std::size_t entry = index_of(fields.at(1));
  const std::size_t channel = index_of(fields.at(2));
  std::vector<double>& rows = at_least(at_least(field, entry), channel);
  const std::vector<double> values = values_of(fields, 4);
  rows.insert(rows.end(), values.begin(), values.end());
}

Case read_case(const std::filesystem::path& path) {
  Case result;
  // The arrays of blocks are made at their components' lines, which come first.
  const auto blocks_of = [&result](std::size_t component) {
    const ComponentLine& line = result.components.at(component);
    return unround::block_extents(line.rows, line.columns);
  };
  for (const std::vector<std::string>& fields : unround::test::lines_of(path)) {
    const std::string& key = fields.at(0);
    if (key == "name") {
      result.name = fields.at(1);
    } else if (key == "picture") {
      result.height = index_of(fields.at(1));
      result.width = index_of(fields.at(2));
      const std::string& space = fields.at(3);
      result.space = space == "grayscale" ? unround::ColourSpace::grayscale
                     : space == "ycbcr"   ? unround::ColourSpace::ycbcr
                                          : unround::ColourSpace::rgb;
    } else if (key == "component") {
      const std::size_t c = index_of(fields.at(1));
      at_least(result.components, c) = ComponentLine{.h = static_cast<std::uint32_t>(index_of(fields.at(2))),
                                                     .v = static_cast<std::uint32_t>(index_of(fields.at(3))),
                                                     .rows = index_of(fields.at(4)),
                                                     .columns = index_of(fields.at(5))};
      at_least(result.levels, c) = BlocksArray<std::int16_t>(blocks_of(c));
      at_least(result.lower, c) = BlocksArray<double>(blocks_of(c));
      at_least(result.upper, c) = BlocksArray<double>(blocks_of(c));
      at_least(result.centres, c) = BlocksArray<double>(blocks_of(c));
      at_least(result.exact_centres, c).assign(64u * result.components[c].rows * result.components[c].columns, {});
    } else if (key == "table") {
      std::array<std::uint16_t, 64>& table = at_least(result.tables, index_of(fields.at(1)));
      for (std::size_t k = 0; k < 64u; ++k) table[k] = static_cast<std::uint16_t>(index_of(fields.at(k + 2u)));
    } else if (key == "levels") {
      const std::vector<double> values = values_of(fields, 3);
      put_block_row(result.levels.at(index_of(fields.at(1))), index_of(fields.at(2)), std::span<const double>(values));
    } else if (key == "options") {
      result.options.assign(fields.begin() + 1, fields.end());
    } else if (key == "steps") {
      result.steps = std::array<double, 2>{number(fields.at(1)), number(fields.at(2))};
    } else if (key == "weights") {
      std::array<double, 64>& weights = at_least(result.weights, index_of(fields.at(1)));
      for (std::size_t k = 0; k < 64u; ++k) weights[k] = number(fields.at(k + 2u));
    } else if (key == "lower" || key == "upper" || key == "centres") {
      std::vector<BlocksArray<double>>& arrays = key == "lower"   ? result.lower
                                                 : key == "upper" ? result.upper
                                                                  : result.centres;
      const std::vector<double> values = values_of(fields, 3);
      put_block_row(arrays.at(index_of(fields.at(1))), index_of(fields.at(2)), std::span<const double>(values));
    } else if (key == "exact-centres") {
      const std::size_t c = index_of(fields.at(1));
      const std::size_t by = index_of(fields.at(2));
      const std::size_t row = 64u * result.components.at(c).columns;
      std::vector<Exact>& exact = result.exact_centres.at(c);
      for (std::size_t k = 0; k < row; ++k) exact.at(by * row + k) = unround::test::exact_of(fields.at(k + 3u));
    } else if (key == "first-coefficients") {
      const std::size_t c = index_of(fields.at(1));
      if (result.first_coefficients.size() <= c) {
        for (std::size_t i = result.first_coefficients.size(); i <= c; ++i) {
          result.first_coefficients.emplace_back(blocks_of(i));
        }
      }
      const std::vector<double> values = values_of(fields, 3);
      put_block_row(result.first_coefficients[c], index_of(fields.at(2)), std::span<const double>(values));
    } else if (key == "first-p") {
      put_field_row(result.first_p, fields);
    } else if (key == "first-w") {
      put_field_row(result.first_w, fields);
    } else if (key == "first-r") {
      put_field_row(result.first_r, fields);
    } else if (key == "record") {
      result.records.push_back(RecordLine{
          .iteration = index_of(fields.at(1)), .primal = number(fields.at(2)), .dual = number(fields.at(3))});
    } else if (key == "canvas") {
      std::vector<double>& rows = at_least(result.canvas, index_of(fields.at(1)));
      const std::vector<double> values = values_of(fields, 3);
      result.canvas_columns = values.size();
      rows.insert(rows.end(), values.begin(), values.end());
    } else if (key == "stop") {
      result.stop_iterations = index_of(fields.at(1));
      result.stop_reason = fields.at(2);
    } else if (key == "tolerance") {
      if (fields.at(1) == "canvas") {
        result.canvas_tolerance = number(fields.at(2));
      } else {
        result.record_tolerances[index_of(fields.at(2))] =
            Tolerance{.primal = number(fields.at(3)), .dual = number(fields.at(4)), .gap = number(fields.at(5))};
      }
    }
  }
  return result;
}

bool same(double a, double b) noexcept { return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b); }

std::string_view reason_of(unround::Stop stop) noexcept {
  switch (stop) {
    case unround::Stop::converged:
      return "converged";
    case unround::Stop::iterations:
      return "iterations";
    case unround::Stop::observer:
      return "observer";
    case unround::Stop::stationary:
      return "stationary";
  }
  return "?";
}

// The vector fields of a start, one for each channel, from the case's rows.
std::vector<unround::operators::Vector> vectors_of(const std::vector<std::vector<std::vector<double>>>& field,
                                                   std::size_t rows, std::size_t columns) {
  std::vector<unround::operators::Vector> vectors;
  if (field.empty()) return vectors;
  for (std::size_t c = 0; c < field.at(0).size(); ++c) {
    unround::operators::Vector vector = unround::operators::vector_of(rows, columns);
    std::ranges::copy(field.at(0).at(c), vector.x.entries().begin());
    std::ranges::copy(field.at(1).at(c), vector.y.entries().begin());
    vectors.push_back(std::move(vector));
  }
  return vectors;
}

std::vector<unround::operators::Tensor> tensors_of(const std::vector<std::vector<std::vector<double>>>& field,
                                                   std::size_t rows, std::size_t columns) {
  std::vector<unround::operators::Tensor> tensors;
  if (field.empty()) return tensors;
  for (std::size_t c = 0; c < field.at(0).size(); ++c) {
    unround::operators::Tensor tensor = unround::operators::tensor_of(rows, columns);
    std::ranges::copy(field.at(0).at(c), tensor.xx.entries().begin());
    std::ranges::copy(field.at(1).at(c), tensor.yy.entries().begin());
    std::ranges::copy(field.at(2).at(c), tensor.xy.entries().begin());
    tensors.push_back(std::move(tensor));
  }
  return tensors;
}

// What a method made: the canvas of each channel, the records, and the stop.
struct Made {
  std::vector<GridArray<double>> canvas;
  std::vector<unround::Record> records;
  std::uint64_t iterations = 0;
  unround::Stop stop = unround::Stop::iterations;
};

void check_case(const std::filesystem::path& path) {
  const Case given = read_case(path);
  const int failures_before = unround::test::failures();
  std::vector<std::string_view> arguments(given.options.begin(), given.options.end());
  const std::expected<unround::Settings, unround::Error> settings = unround::parse_settings(arguments);
  if (!CHECK(settings.has_value())) {
    std::println(stderr, "{}: {}", given.name, settings.error().message);
    return;
  }
  std::vector<unround::ComponentInput> inputs;
  inputs.reserve(given.components.size());
  for (std::size_t c = 0; c < given.components.size(); ++c) {
    inputs.push_back(unround::ComponentInput{.levels = std::as_const(given.levels[c]).view(),
                                             .table = given.tables.at(c),
                                             .h = given.components[c].h,
                                             .v = given.components[c].v});
  }
  const std::vector<unround::DataTerm> data = unround::data_of(*settings);
  const std::expected<unround::Frame, unround::Error> made_frame =
      unround::Frame::make(given.height, given.width, given.space, inputs, data);
  if (!CHECK(made_frame.has_value())) {
    std::println(stderr, "{}: {}", given.name, made_frame.error().message);
    return;
  }
  const unround::Frame& frame = *made_frame;
  // What is rational, to the last bit; the centres within their bounds.
  for (std::size_t c = 0; c < frame.channels(); ++c) {
    const unround::Problem& problem = frame.component(c).problem;
    CHECK(std::ranges::equal(std::span<const double>(problem.lower().data_handle(), problem.lower().size()),
                             std::as_const(given.lower[c]).entries(), same));
    CHECK(std::ranges::equal(std::span<const double>(problem.upper().data_handle(), problem.upper().size()),
                             std::as_const(given.upper[c]).entries(), same));
    CHECK(std::ranges::equal(std::span<const double>(problem.weights().data_handle(), 64),
                             std::span<const double>(given.weights.at(c)), same));
    const bool mmse = data[data.size() == 1u ? 0u : c].centres == unround::Centres::mmse;
    const std::span<const double> centres(problem.centres().data_handle(), problem.centres().size());
    const std::span<const std::int16_t> levels(problem.levels().data_handle(), problem.levels().size());
    bool within = true;
    for (std::size_t k = 0; k < centres.size(); ++k) {
      const Exact& exact = given.exact_centres[c].at(k);
      const double error = std::abs((centres[k] - exact.hi) - exact.lo);
      const std::size_t frequency = k % 64u;
      if (mmse && frequency != 0u && levels[k] != 0) {
        const double step = static_cast<double>(given.tables.at(c)[frequency]);
        const double bound = (2.0 * std::abs(static_cast<double>(levels[k])) + 20.0) * u * step;
        within = within && error <= bound * (1.0 - 4.0 * u);
      } else {
        within = within && error == 0.0;
      }
    }
    CHECK(within);
  }
  const unround::Settings& chosen = *settings;
  Made made;
  if (chosen.method == unround::Method::mmse) {
    for (std::size_t c = 0; c < frame.channels(); ++c) {
      GridArray<double> x(frame.canvas_extents());
      BlocksArray<double> kept(frame.component(c).problem.extents());
      frame.start(c, x.view(), kept.view());
      made.canvas.push_back(std::move(x));
    }
  } else if (chosen.method == unround::Method::subgradient) {
    auto result = unround::subgradient::solve(frame, chosen.regularizer.alpha, chosen.subgradient);
    if (!CHECK(result.has_value())) {
      std::println(stderr, "{}: {}", given.name, result.error().message);
      return;
    }
    made = Made{.canvas = std::move(result->canvas),
                .records = std::move(result->records),
                .iterations = result->iterations,
                .stop = result->stop};
  } else {
    const unround::pdhg::Model model =
        chosen.method == unround::Method::tv ? unround::pdhg::Model::tv : unround::pdhg::Model::tgv;
    const auto plan = unround::pdhg::plan(model, chosen.regularizer, chosen.pdhg, frame.channels());
    if (!CHECK(plan.has_value())) return;
    if (given.steps) {
      CHECK(same(plan->steps.tau, (*given.steps)[0]) && same(plan->steps.sigma, (*given.steps)[1]));
    }
    unround::pdhg::Start start;
    start.coefficients = given.first_coefficients;
    start.p = vectors_of(given.first_p, frame.rows(), frame.columns());
    start.w = vectors_of(given.first_w, frame.rows(), frame.columns());
    start.r = tensors_of(given.first_r, frame.rows(), frame.columns());
    auto result = unround::pdhg::solve(frame, model, chosen.regularizer, chosen.pdhg, &start);
    if (!CHECK(result.has_value())) {
      std::println(stderr, "{}: {}", given.name, result.error().message);
      return;
    }
    made = Made{.canvas = std::move(result->canvas),
                .records = std::move(result->records),
                .iterations = result->iterations,
                .stop = result->stop};
  }
  // The stop and the records.
  CHECK(made.iterations == given.stop_iterations && reason_of(made.stop) == given.stop_reason);
  CHECK(made.records.size() == given.records.size());
  double worst_primal = 0.0;
  for (std::size_t k = 0; k < std::min(made.records.size(), given.records.size()); ++k) {
    const unround::Record& ours = made.records[k];
    const RecordLine& theirs = given.records[k];
    CHECK(ours.iteration == theirs.iteration);
    const auto found = given.record_tolerances.find(theirs.iteration);
    if (!CHECK(found != given.record_tolerances.end())) continue;
    const Tolerance& tolerance = found->second;
    const double primal = std::abs(ours.primal - theirs.primal);
    worst_primal = std::max(worst_primal, primal / tolerance.primal);
    CHECK(primal <= tolerance.primal);
    if (tolerance.dual != -1.0) {
      CHECK(ours.dual <= theirs.primal + tolerance.dual && theirs.dual <= ours.primal + tolerance.dual);
    }
    if (tolerance.gap != -1.0) {
      CHECK(std::abs((ours.primal - ours.dual) - (theirs.primal - theirs.dual)) <= tolerance.gap);
    }
  }
  // The canvas.
  double squares = 0.0;
  bool shaped = made.canvas.size() == given.canvas.size();
  for (std::size_t c = 0; shaped && c < made.canvas.size(); ++c) {
    const std::span<const double> ours = std::as_const(made.canvas[c]).entries();
    const std::vector<double>& theirs = given.canvas[c];
    shaped = ours.size() == theirs.size() && made.canvas[c].extent(1) == given.canvas_columns;
    for (std::size_t k = 0; shaped && k < ours.size(); ++k) squares += (ours[k] - theirs[k]) * (ours[k] - theirs[k]);
  }
  CHECK(shaped);
  const double distance = std::sqrt(squares);
  CHECK(distance <= given.canvas_tolerance);
  std::println("{}: {} records, canvas {:.3e} of {:.3e}, primal at most {:.3f} of its tolerance{}", given.name,
               made.records.size(), distance, given.canvas_tolerance, worst_primal,
               unround::test::failures() == failures_before ? "" : "  FAILED");
}

int run(std::span<char* const> arguments) {
  if (arguments.size() < 2u) return 2;
  std::vector<std::filesystem::path> cases;
  for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(arguments[1])) {
    if (entry.path().extension() == ".txt") cases.push_back(entry.path());
  }
  std::ranges::sort(cases);
  CHECK(cases.size() == 17u);
  for (const std::filesystem::path& path : cases) check_case(path);
  return unround::test::finish("conformance");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(std::span<char* const>{argv, static_cast<std::size_t>(argc)});
  } catch (const std::exception& error) {
    (void)std::fputs(error.what(), stderr);
    (void)std::fputc('\n', stderr);
    return EXIT_FAILURE;
  }
}
