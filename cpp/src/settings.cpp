// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The options of a reconstruction as the command line takes them (docs/cli.md).

#include "unround/settings.hpp"

#include "unround/error.hpp"
#include "unround/model.hpp"
#include "unround/pdhg.hpp"
#include "unround/subgradient.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <limits>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace unround {

namespace {

// The options with a value, and those without (flags).
constexpr std::array<std::string_view, 36> valued = {"method",
                                                     "alpha",
                                                     "alpha1",
                                                     "alpha0",
                                                     "channel-weights",
                                                     "channels",
                                                     "mu",
                                                     "mu-scale",
                                                     "mu-power",
                                                     "mu-chroma",
                                                     "weight-power",
                                                     "dc-weight",
                                                     "centres",
                                                     "slack",
                                                     "slack-cost",
                                                     "flat-weight",
                                                     "flat-levels",
                                                     "flat-dc",
                                                     "iterations",
                                                     "tolerance",
                                                     "relative-tolerance",
                                                     "partial-tolerance",
                                                     "partial-radius",
                                                     "step-ratio",
                                                     "relaxation",
                                                     "step-product",
                                                     "norm-squared",
                                                     "record-every",
                                                     "free-radius",
                                                     "threads",
                                                     "subgradient-iterations",
                                                     "subgradient-step",
                                                     "subgradient-decay",
                                                     "subgradient-record-every",
                                                     "max-pixels",
                                                     "max-scans"};
constexpr std::array<std::string_view, 3> flags = {"no-weight-scaling", "no-momentum", "warnings-are-errors"};

bool is_valued(std::string_view name) noexcept { return std::ranges::find(valued, name) != valued.end(); }
bool is_flag(std::string_view name) noexcept { return std::ranges::find(flags, name) != flags.end(); }

// A decimal number, the whole of the text.
std::optional<double> real_of(std::string_view text) noexcept {
  double value = 0.0;
  const char* const first = text.data();
  const char* const last = first + text.size();
  const auto [stop, error] = std::from_chars(first, last, value);
  if (error != std::errc{} || stop != last || text.empty()) return std::nullopt;
  return value;
}

// A whole number in decimal digits, the whole of the text.
std::optional<std::uint64_t> whole_of(std::string_view text) noexcept {
  std::uint64_t value = 0;
  const char* const first = text.data();
  const char* const last = first + text.size();
  const auto [stop, error] = std::from_chars(first, last, value);
  if (error != std::errc{} || stop != last || text.empty()) return std::nullopt;
  return value;
}

// The items of a comma-separated list.
std::vector<std::string_view> items_of(std::string_view text) {
  std::vector<std::string_view> items;
  std::size_t start = 0;
  while (true) {
    const std::size_t comma = text.find(',', start);
    items.push_back(text.substr(start, comma == std::string_view::npos ? std::string_view::npos : comma - start));
    if (comma == std::string_view::npos) break;
    start = comma + 1u;
  }
  return items;
}

class Reader {
 public:
  explicit Reader(std::span<const std::string_view> arguments) {
    for (std::size_t i = 0; i < arguments.size(); ++i) {
      const std::string_view argument = arguments[i];
      if (!argument.starts_with("--")) {
        errors.push_back(std::format("{} is not an option", argument));
        continue;
      }
      const std::size_t equals = argument.find('=');
      const std::string_view name =
          argument.substr(2u, equals == std::string_view::npos ? std::string_view::npos : equals - 2u);
      const std::string key(name);
      if (is_flag(name)) {
        if (equals != std::string_view::npos) errors.push_back(std::format("--{} takes no value", name));
        if (given_.emplace(key, "").second) {
          order_.push_back(key);
        } else {
          errors.push_back(std::format("--{} is given twice", name));
        }
        continue;
      }
      if (!is_valued(name)) {
        errors.push_back(std::format("--{} is not an option", name));
        if (equals == std::string_view::npos && i + 1u < arguments.size() && !arguments[i + 1u].starts_with("--")) {
          ++i;
        }
        continue;
      }
      std::string_view value;
      if (equals != std::string_view::npos) {
        value = argument.substr(equals + 1u);
      } else if (i + 1u < arguments.size()) {
        value = arguments[++i];
      } else {
        errors.push_back(std::format("--{} needs a value", name));
        continue;
      }
      if (given_.emplace(key, std::string(value)).second) {
        order_.push_back(key);
      } else {
        errors.push_back(std::format("--{} is given twice", name));
      }
    }
  }

  [[nodiscard]] bool flag(std::string_view name) const { return given_.contains(std::string(name)); }

  // The names of the options given, in the order given.
  [[nodiscard]] const std::vector<std::string>& names(void) const noexcept { return order_; }

  [[nodiscard]] std::optional<std::string_view> text(std::string_view name) const {
    const auto found = given_.find(std::string(name));
    if (found == given_.end()) return std::nullopt;
    return std::string_view(found->second);
  }

  // A real number of the option, checked; none where it is not given or refused.
  template <typename Check>
  std::optional<double> real(std::string_view name, const Check& check, std::string_view range) {
    const std::optional<std::string_view> given = text(name);
    if (!given) return std::nullopt;
    const std::optional<double> value = real_of(*given);
    if (!value || !check(*value)) {
      errors.push_back(std::format("--{} is {}, not {}", name, range, *given));
      return std::nullopt;
    }
    return value;
  }

  // A whole number of the option, at least `least` and at most `most`.
  std::optional<std::uint64_t> whole(std::string_view name, std::uint64_t least, std::uint64_t most,
                                     std::string_view range) {
    const std::optional<std::string_view> given = text(name);
    if (!given) return std::nullopt;
    const std::optional<std::uint64_t> value = whole_of(*given);
    if (!value || *value < least || *value > most) {
      errors.push_back(std::format("--{} is {}, not {}", name, range, *given));
      return std::nullopt;
    }
    return value;
  }

  // One of the words of an option.
  std::optional<std::string_view> word(std::string_view name, std::span<const std::string_view> words) {
    const std::optional<std::string_view> given = text(name);
    if (!given) return std::nullopt;
    if (std::ranges::find(words, *given) == words.end()) {
      std::string choices;
      for (const std::string_view choice : words)
        choices += choices.empty() ? std::string(choice) : ", " + std::string(choice);
      errors.push_back(std::format("--{} is one of {}, not {}", name, choices, *given));
      return std::nullopt;
    }
    return given;
  }

  // The items of a list option, each read by `read`, which says how it failed by
  // returning none; none where it is not given or an item is refused.
  template <typename T, typename Read>
  std::optional<std::vector<T>> list(std::string_view name, const Read& read, std::string_view range) {
    const std::optional<std::string_view> given = text(name);
    if (!given) return std::nullopt;
    std::vector<T> values;
    for (const std::string_view item : items_of(*given)) {
      const std::optional<T> value = read(item);
      if (!value) {
        errors.push_back(std::format("--{} is {}, not {}", name, range, item));
        return std::nullopt;
      }
      values.push_back(*value);
    }
    return values;
  }

  std::vector<std::string> errors;

 private:
  std::map<std::string, std::string, std::less<>> given_;
  std::vector<std::string> order_;
};

bool positive(double value) noexcept { return value > 0.0 && std::isfinite(value); }
bool at_least_zero(double value) noexcept { return value >= 0.0 && std::isfinite(value); }
bool finite(double value) noexcept { return std::isfinite(value); }

std::optional<double> at_least_zero_of(std::string_view text) noexcept {
  const std::optional<double> value = real_of(text);
  return value && at_least_zero(*value) ? value : std::nullopt;
}

}  // namespace

std::expected<Settings, Error> parse_settings(std::span<const std::string_view> arguments) {
  Reader reader(arguments);
  Settings settings;
  constexpr std::array<std::string_view, 4> methods = {"mmse", "tv", "tgv", "subgradient"};
  if (const auto method = reader.word("method", methods)) {
    settings.method = *method == "mmse"  ? Method::mmse
                      : *method == "tv"  ? Method::tv
                      : *method == "tgv" ? Method::tgv
                                         : Method::subgradient;
  }
  constexpr std::string_view above_zero = "above 0 and finite";
  constexpr std::string_view from_zero = "at least 0 and finite";
  pdhg::Regularizer& regularizer = settings.regularizer;
  if (const auto value = reader.real("alpha", positive, above_zero)) regularizer.alpha = *value;
  if (const auto value = reader.real("alpha1", positive, above_zero)) regularizer.alpha1 = *value;
  if (const auto value = reader.real("alpha0", positive, above_zero)) regularizer.alpha0 = *value;
  if (const auto values = reader.list<double>(
          "channel-weights",
          [](std::string_view item) -> std::optional<double> {
            const std::optional<double> value = real_of(item);
            return value && positive(*value) ? value : std::nullopt;
          },
          "a list of numbers above 0 and finite")) {
    regularizer.channel_weights = *values;
  }
  constexpr std::array<std::string_view, 2> couplings = {"coupled", "apart"};
  if (const auto coupling = reader.word("channels", couplings)) regularizer.coupled = *coupling == "coupled";

  // The data term: lists of one value for every component or one for each.
  const auto mu = reader.list<std::optional<double>>(
      "mu",
      [](std::string_view item) -> std::optional<std::optional<double>> {
        if (item == "rule") return std::optional<double>{};
        const std::optional<double> value = at_least_zero_of(item);
        if (!value) return std::nullopt;
        return value;
      },
      "at least 0 and finite, or rule");
  const auto mu_scale = reader.list<double>("mu-scale", at_least_zero_of, from_zero);
  const auto mu_power = reader.list<double>(
      "mu-power",
      [](std::string_view item) -> std::optional<double> {
        const std::optional<double> value = real_of(item);
        return value && finite(*value) ? value : std::nullopt;
      },
      "finite");
  const auto mu_chroma = reader.list<double>("mu-chroma", at_least_zero_of, from_zero);
  const auto power = reader.list<double>("weight-power", at_least_zero_of, from_zero);
  const auto dc_weight = reader.list<double>("dc-weight", at_least_zero_of, from_zero);
  const auto centres = reader.list<Centres>(
      "centres",
      [](std::string_view item) -> std::optional<Centres> {
        if (item == "mmse") return Centres::mmse;
        if (item == "midpoint") return Centres::midpoint;
        return std::nullopt;
      },
      "mmse or midpoint");
  const auto slack = reader.list<double>("slack", at_least_zero_of, from_zero);
  const auto slack_cost = reader.list<double>("slack-cost", at_least_zero_of, from_zero);
  const auto flat_weight = reader.list<double>("flat-weight", at_least_zero_of, from_zero);
  const auto flat_levels = reader.list<std::uint32_t>(
      "flat-levels",
      [](std::string_view item) -> std::optional<std::uint32_t> {
        const std::optional<std::uint64_t> value = whole_of(item);
        if (!value || *value > 63u) return std::nullopt;
        return static_cast<std::uint32_t>(*value);
      },
      "an integer from 0 to 63");
  const auto flat_dc = reader.list<std::optional<std::uint32_t>>(
      "flat-dc",
      [](std::string_view item) -> std::optional<std::optional<std::uint32_t>> {
        if (item == "any") return std::optional<std::uint32_t>{};
        const std::optional<std::uint64_t> value = whole_of(item);
        if (!value || *value > std::numeric_limits<std::uint32_t>::max()) return std::nullopt;
        return std::optional<std::uint32_t>{static_cast<std::uint32_t>(*value)};
      },
      "a whole number or any");
  // The lists' lengths: 1, or one common length.
  std::size_t count = 1;
  const std::array<std::pair<std::string_view, std::size_t>, 12> lengths = {{
      {"mu", mu ? mu->size() : 1u},
      {"mu-scale", mu_scale ? mu_scale->size() : 1u},
      {"mu-power", mu_power ? mu_power->size() : 1u},
      {"mu-chroma", mu_chroma ? mu_chroma->size() : 1u},
      {"weight-power", power ? power->size() : 1u},
      {"dc-weight", dc_weight ? dc_weight->size() : 1u},
      {"centres", centres ? centres->size() : 1u},
      {"slack", slack ? slack->size() : 1u},
      {"slack-cost", slack_cost ? slack_cost->size() : 1u},
      {"flat-weight", flat_weight ? flat_weight->size() : 1u},
      {"flat-levels", flat_levels ? flat_levels->size() : 1u},
      {"flat-dc", flat_dc ? flat_dc->size() : 1u},
  }};
  for (const auto& [name, length] : lengths) {
    if (length == 1u) continue;
    if (count == 1u) {
      count = length;
    } else if (length != count) {
      reader.errors.push_back(std::format(
          "the options of the data term give one value for every component or one for each, not {} for --{}", length,
          name));
    }
  }
  settings.data.assign(count, DataTerm{});
  const auto pick = [](const auto& values, std::size_t index, auto& into) {
    if (values) into = (*values)[values->size() == 1u ? 0u : index];
  };
  for (std::size_t c = 0; c < count; ++c) {
    DataTerm& term = settings.data[c];
    pick(mu, c, term.mu);
    pick(mu_scale, c, term.mu_scale);
    pick(mu_power, c, term.mu_power);
    pick(mu_chroma, c, term.mu_chroma);
    pick(power, c, term.power);
    pick(dc_weight, c, term.dc_weight);
    pick(centres, c, term.centres);
    pick(slack, c, term.slack);
    pick(slack_cost, c, term.slack_cost);
    pick(flat_weight, c, term.flat_weight);
    pick(flat_levels, c, term.flat_levels);
    pick(flat_dc, c, term.flat_dc);
  }

  // The primal-dual method.
  constexpr std::uint64_t unbounded = std::numeric_limits<std::uint64_t>::max();
  pdhg::Options& pdhg = settings.pdhg;
  if (const auto value = reader.whole("iterations", 0, unbounded, "a whole number")) pdhg.iterations = value;
  if (const auto value = reader.real("tolerance", at_least_zero, from_zero)) pdhg.tolerance = value;
  if (const auto value = reader.real("relative-tolerance", at_least_zero, from_zero)) {
    pdhg.relative_tolerance = *value;
  }
  if (const auto value = reader.real("partial-tolerance", at_least_zero, from_zero)) pdhg.partial_tolerance = *value;
  if (const auto value = reader.real("partial-radius", at_least_zero, from_zero)) pdhg.partial_radius = value;
  if (const auto value = reader.real("step-ratio", positive, above_zero)) pdhg.step_ratio = value;
  if (const auto value = reader.real("relaxation", [](double x) { return x > 0.0 && x < 2.0; }, "between 0 and 2")) {
    pdhg.relaxation = *value;
  }
  if (const auto value = reader.real("step-product", [](double x) { return x > 0.0 && x < 1.0; }, "between 0 and 1")) {
    pdhg.step_product = *value;
  }
  if (const auto value = reader.real("norm-squared", positive, above_zero)) pdhg.norm_squared = value;
  pdhg.weight_scaling = !reader.flag("no-weight-scaling");
  if (const auto value = reader.whole("record-every", 0, unbounded, "a whole number")) pdhg.record_every = *value;
  if (const auto value = reader.real("free-radius", at_least_zero, from_zero)) pdhg.free_radius = *value;
  if (const auto value = reader.whole("threads", 0, unbounded, "a whole number")) pdhg.threads = *value;

  // The subgradient method.
  subgradient::Options& sub = settings.subgradient;
  if (const auto value = reader.whole("subgradient-iterations", 0, unbounded, "a whole number")) {
    sub.iterations = *value;
  }
  if (const auto value = reader.real("subgradient-step", positive, above_zero)) sub.step = *value;
  if (const auto value = reader.real("subgradient-decay", at_least_zero, from_zero)) sub.decay = *value;
  sub.momentum = !reader.flag("no-momentum");
  if (const auto value = reader.whole("subgradient-record-every", 0, unbounded, "a whole number")) {
    sub.record_every = *value;
  }

  // The reading.
  if (const auto value = reader.whole("max-pixels", 1, unbounded, "a whole number of at least 1")) {
    settings.reading.max_pixels = *value;
  }
  constexpr auto most_scans = static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max());
  if (const auto value = reader.whole("max-scans", 1, most_scans, "a whole number of at least 1")) {
    settings.reading.max_scans = static_cast<std::int32_t>(*value);
  }
  settings.reading.warnings_are_errors = reader.flag("warnings-are-errors");

  // What the method does not take.
  if (settings.method == Method::tv && (pdhg.partial_radius || pdhg.partial_tolerance > 0.0)) {
    reader.errors.emplace_back("the partial gap is TGV's alone, not TV's");
  }
  if (pdhg.partial_tolerance > 0.0 && !pdhg.partial_radius) {
    reader.errors.emplace_back("--partial-tolerance needs --partial-radius");
  }
  if (!reader.errors.empty()) {
    std::string message;
    for (const std::string& error : reader.errors) message += message.empty() ? error : "; " + error;
    return std::unexpected(Error{.code = Errc::options, .message = std::move(message)});
  }
  settings.given = reader.names();
  return settings;
}

bool is_setting(std::string_view name) noexcept { return is_valued(name) || is_flag(name); }

bool setting_takes_value(std::string_view name) noexcept { return is_valued(name); }

namespace {

std::string number(double value) { return std::format("{}", value); }

// The values of the data terms' field, joined by commas where they differ.
template <typename Text>
std::string joined(const std::vector<DataTerm>& data, const Text& text) {
  std::string first = text(data.front());
  bool alike = true;
  for (const DataTerm& term : data) alike = alike && text(term) == first;
  if (alike) return first;
  std::string all;
  for (const DataTerm& term : data) all += (all.empty() ? "" : ",") + text(term);
  return all;
}

}  // namespace

std::vector<std::string> options_of(const Settings& settings) {
  std::vector<std::string> options;
  const auto add = [&options](std::string_view name, const std::string& value) {
    options.push_back(std::format("--{}={}", name, value));
  };
  constexpr std::array<std::string_view, 4> methods = {"mmse", "tv", "tgv", "subgradient"};
  add("method", std::string(methods[static_cast<std::size_t>(settings.method)]));
  const pdhg::Regularizer& regularizer = settings.regularizer;
  add("alpha", number(regularizer.alpha));
  add("alpha1", number(regularizer.alpha1));
  add("alpha0", number(regularizer.alpha0));
  if (!regularizer.channel_weights.empty()) {
    std::string weights;
    for (const double weight : regularizer.channel_weights) weights += (weights.empty() ? "" : ",") + number(weight);
    add("channel-weights", weights);
  }
  add("channels", regularizer.coupled ? "coupled" : "apart");
  const std::vector<DataTerm>& data = settings.data;
  add("mu", joined(data, [](const DataTerm& term) { return term.mu ? number(*term.mu) : std::string("rule"); }));
  add("mu-scale", joined(data, [](const DataTerm& term) { return number(term.mu_scale); }));
  add("mu-power", joined(data, [](const DataTerm& term) { return number(term.mu_power); }));
  add("mu-chroma", joined(data, [](const DataTerm& term) { return number(term.mu_chroma); }));
  add("weight-power", joined(data, [](const DataTerm& term) { return number(term.power); }));
  add("dc-weight", joined(data, [](const DataTerm& term) { return number(term.dc_weight); }));
  add("centres", joined(data, [](const DataTerm& term) {
        return std::string(term.centres == Centres::mmse ? "mmse" : "midpoint");
      }));
  add("slack", joined(data, [](const DataTerm& term) { return number(term.slack); }));
  add("slack-cost", joined(data, [](const DataTerm& term) { return number(term.slack_cost); }));
  add("flat-weight", joined(data, [](const DataTerm& term) { return number(term.flat_weight); }));
  add("flat-levels", joined(data, [](const DataTerm& term) { return std::to_string(term.flat_levels); }));
  add("flat-dc", joined(data, [](const DataTerm& term) {
        return term.flat_dc ? std::to_string(*term.flat_dc) : std::string("any");
      }));
  const pdhg::Options& pdhg = settings.pdhg;
  if (pdhg.iterations) add("iterations", std::to_string(*pdhg.iterations));
  if (pdhg.tolerance) add("tolerance", number(*pdhg.tolerance));
  add("relative-tolerance", number(pdhg.relative_tolerance));
  add("partial-tolerance", number(pdhg.partial_tolerance));
  if (pdhg.partial_radius) add("partial-radius", number(*pdhg.partial_radius));
  if (pdhg.step_ratio) add("step-ratio", number(*pdhg.step_ratio));
  add("relaxation", number(pdhg.relaxation));
  add("step-product", number(pdhg.step_product));
  if (pdhg.norm_squared) add("norm-squared", number(*pdhg.norm_squared));
  if (!pdhg.weight_scaling) options.emplace_back("--no-weight-scaling");
  add("record-every", std::to_string(pdhg.record_every));
  add("free-radius", number(pdhg.free_radius));
  add("threads", std::to_string(pdhg.threads));
  const subgradient::Options& sub = settings.subgradient;
  add("subgradient-iterations", std::to_string(sub.iterations));
  add("subgradient-step", number(sub.step));
  add("subgradient-decay", number(sub.decay));
  if (!sub.momentum) options.emplace_back("--no-momentum");
  add("subgradient-record-every", std::to_string(sub.record_every));
  return options;
}

std::vector<DataTerm> data_of(const Settings& settings) {
  std::vector<DataTerm> data = settings.data;
  if (settings.method == Method::mmse) {
    for (DataTerm& term : data) term.centres = Centres::mmse;
  }
  return data;
}

}  // namespace unround
