// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Tests of the settings of the command line's options (docs/cli.md): the defaults;
// every option read, with its value after it or after '='; the data term's lists of one
// value for each component; and every option refused named at once: unknown ones,
// options given twice or without their values, values out of their ranges, lists of
// lengths that differ, a flag with a value, and a partial gap for TV.

#include "unround/settings.hpp"

#include "unround/model.hpp"
#include "unround/pdhg.hpp"
#include "unround/subgradient.hpp"

#include "support/check.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <expected>
#include <initializer_list>
#include <optional>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

using unround::Centres;
using unround::Method;
using unround::Settings;

std::expected<Settings, unround::Error> parse(std::initializer_list<std::string_view> arguments) {
  const std::vector<std::string_view> list(arguments);
  return unround::parse_settings(list);
}

void defaults(void) {
  const auto settings = parse({});
  CHECK(settings.has_value());
  CHECK(settings->method == Method::tgv);
  CHECK(settings->regularizer.alpha == 1.0 && settings->regularizer.alpha1 == 1.0 &&
        settings->regularizer.alpha0 == 2.0 && settings->regularizer.coupled &&
        settings->regularizer.channel_weights.empty());
  CHECK(settings->data.size() == 1u && settings->data[0] == unround::DataTerm{});
  CHECK(!settings->pdhg.iterations && !settings->pdhg.tolerance && !settings->pdhg.step_ratio &&
        !settings->pdhg.norm_squared && !settings->pdhg.partial_radius);
  CHECK(settings->pdhg.relaxation == 1.9 && settings->pdhg.step_product == 0.99 && settings->pdhg.weight_scaling &&
        settings->pdhg.record_every == 10u && settings->pdhg.free_radius == 255.0);
  CHECK(settings->subgradient.iterations == 50u && settings->subgradient.step == 0.5 &&
        settings->subgradient.decay == 0.5 && settings->subgradient.momentum &&
        settings->subgradient.record_every == 1u);
  CHECK(settings->reading.max_pixels == 268435456u && settings->reading.max_scans == 500 &&
        !settings->reading.warnings_are_errors);
}

void every_option(void) {
  const auto settings = parse({"--method",
                               "tv",
                               "--alpha=2.5",
                               "--alpha1",
                               "3",
                               "--alpha0",
                               "7e-1",
                               "--channel-weights",
                               "1,0.5,0.25",
                               "--channels=apart",
                               "--mu",
                               "0.01",
                               "--mu-scale",
                               "8",
                               "--mu-power=1",
                               "--mu-chroma",
                               "0.5",
                               "--weight-power",
                               "1.5",
                               "--dc-weight",
                               "0",
                               "--centres",
                               "mmse",
                               "--slack",
                               "0.25",
                               "--slack-cost",
                               "3",
                               "--flat-weight",
                               "0.2",
                               "--flat-levels",
                               "2",
                               "--flat-dc",
                               "any",
                               "--iterations",
                               "300",
                               "--tolerance=1e-5",
                               "--relative-tolerance",
                               "1e-3",
                               "--step-ratio",
                               "4",
                               "--relaxation",
                               "1.5",
                               "--step-product",
                               "0.5",
                               "--norm-squared",
                               "9",
                               "--no-weight-scaling",
                               "--record-every",
                               "7",
                               "--free-radius",
                               "100",
                               "--threads",
                               "5",
                               "--subgradient-iterations",
                               "12",
                               "--subgradient-step",
                               "0.25",
                               "--subgradient-decay",
                               "0",
                               "--no-momentum",
                               "--subgradient-record-every",
                               "3",
                               "--max-pixels",
                               "1000",
                               "--max-scans",
                               "9",
                               "--warnings-are-errors"});
  if (!CHECK(settings.has_value())) {
    std::println(stderr, "{}", settings.error().message);
    return;
  }
  CHECK(settings->method == Method::tv);
  CHECK(settings->regularizer.alpha == 2.5 && settings->regularizer.alpha1 == 3.0 &&
        settings->regularizer.alpha0 == 0.7 && !settings->regularizer.coupled);
  CHECK(settings->regularizer.channel_weights == std::vector<double>({1.0, 0.5, 0.25}));
  CHECK(settings->data.size() == 1u);
  const unround::DataTerm& term = settings->data[0];
  CHECK(term.mu == 0.01 && term.mu_scale == 8.0 && term.mu_power == 1.0 && term.mu_chroma == 0.5);
  CHECK(term.power == 1.5 && term.dc_weight == 0.0 && term.centres == Centres::mmse && term.slack == 0.25);
  CHECK(term.slack_cost == 3.0 && term.flat_weight == 0.2 && term.flat_levels == 2u && !term.flat_dc);
  const unround::pdhg::Options& pdhg = settings->pdhg;
  CHECK(pdhg.iterations == 300u && pdhg.tolerance == 1e-5 && pdhg.relative_tolerance == 1e-3);
  CHECK(pdhg.step_ratio == 4.0 && pdhg.relaxation == 1.5 && pdhg.step_product == 0.5 && pdhg.norm_squared == 9.0);
  CHECK(!pdhg.weight_scaling && pdhg.record_every == 7u && pdhg.free_radius == 100.0 && pdhg.threads == 5u);
  const unround::subgradient::Options& sub = settings->subgradient;
  CHECK(sub.iterations == 12u && sub.step == 0.25 && sub.decay == 0.0 && !sub.momentum && sub.record_every == 3u);
  CHECK(settings->reading.max_pixels == 1000u && settings->reading.max_scans == 9 &&
        settings->reading.warnings_are_errors);
  // TGV takes the partial gap.
  const auto partial = parse({"--method", "tgv", "--partial-radius", "50", "--partial-tolerance", "1e-3"});
  CHECK(partial.has_value() && partial->pdhg.partial_radius == 50.0 && partial->pdhg.partial_tolerance == 1e-3);
}

void lists(void) {
  const auto settings =
      parse({"--mu", "0.01,rule,0.03", "--flat-dc", "1,any,2", "--flat-levels", "0,1,63", "--centres", "midpoint"});
  if (!CHECK(settings.has_value())) {
    std::println(stderr, "{}", settings.error().message);
    return;
  }
  CHECK(settings->data.size() == 3u);
  CHECK(settings->data[0].mu == 0.01 && !settings->data[1].mu && settings->data[2].mu == 0.03);
  CHECK(settings->data[0].flat_dc == 1u && !settings->data[1].flat_dc && settings->data[2].flat_dc == 2u);
  CHECK(settings->data[0].flat_levels == 0u && settings->data[1].flat_levels == 1u &&
        settings->data[2].flat_levels == 63u);
  CHECK(settings->data[1].centres == Centres::midpoint && settings->data[2].mu_scale == 9.0);
  // The MMSE decoder takes the MMSE centres whatever --centres says.
  const auto mmse = parse({"--method", "mmse", "--centres", "midpoint"});
  CHECK(mmse.has_value() && unround::data_of(*mmse).at(0).centres == Centres::mmse &&
        mmse->data[0].centres == Centres::midpoint);
  CHECK(unround::data_of(*settings).at(1).centres == Centres::midpoint);
}

// Each refusal names its option, and every refusal of a command line is named at once.
void refusals(void) {
  const auto refused = [](std::initializer_list<std::string_view> arguments,
                          std::initializer_list<std::string_view> named) {
    const auto settings = parse(arguments);
    if (!CHECK(!settings.has_value())) return;
    for (const std::string_view name : named) {
      if (!CHECK(settings.error().message.find(name) != std::string::npos)) {
        std::println(stderr, "'{}' not named in: {}", name, settings.error().message);
      }
    }
    CHECK(settings.error().code == unround::Errc::options);
  };
  refused({"--unknown", "1"}, {"--unknown is not an option"});
  refused({"--alpha", "1", "--alpha", "2"}, {"--alpha is given twice"});
  refused({"--alpha"}, {"--alpha needs a value"});
  refused({"input.jpg"}, {"input.jpg is not an option"});
  refused({"--no-momentum=1"}, {"--no-momentum takes no value"});
  refused({"--method", "fast"}, {"--method is one of mmse, tv, tgv, subgradient, not fast"});
  refused({"--alpha", "0", "--alpha0", "-1", "--relaxation", "2", "--step-product", "1"},
          {"--alpha is above 0", "--alpha0 is above 0", "--relaxation is between 0 and 2",
           "--step-product is between 0 and 1"});
  refused({"--mu", "-1", "--flat-levels", "64", "--flat-dc", "-1", "--mu-power", "inf"},
          {"--mu is at least 0 and finite, or rule, not -1", "--flat-levels is an integer from 0 to 63, not 64",
           "--flat-dc is a whole number or any, not -1", "--mu-power is finite, not inf"});
  refused({"--iterations", "2.5", "--max-pixels", "0", "--tolerance", "nan"},
          {"--iterations is a whole number, not 2.5", "--max-pixels is a whole number of at least 1, not 0",
           "--tolerance is at least 0 and finite, not nan"});
  refused({"--mu", "0.1,0.2", "--slack", "0,1,2"}, {"not 3 for --slack"});
  refused({"--method", "tv", "--partial-radius", "5"}, {"the partial gap is TGV's alone"});
  refused({"--method", "tgv", "--partial-tolerance", "1e-3"}, {"--partial-tolerance needs --partial-radius"});
  refused({"--alpha", "1x"}, {"--alpha is above 0 and finite, not 1x"});
  refused({"--channel-weights", "1,0,1"}, {"--channel-weights is a list of numbers above 0 and finite, not 0"});
}

int run(void) {
  defaults();
  every_option();
  lists();
  refusals();
  return unround::test::finish("settings");
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
