// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Pictures of rounded samples: docs/math.md, 11.
//
// The ends of the intervals are rationals, 255 (2 v -+ 1 -+ 2 slack) / (2 (2^b - 1)),
// and each is rounded once to the nearest double: the slack, a double, is m 2^e, so that
// the numerator times 2^K, K = -e, is an integer; it is divided by the small
// denominator in a few words of 64 bits, and its top 64 bits, with a sticky bit for
// what lies below them, are converted once.

#include "unround/samples.hpp"

#include "unround/arrays.hpp"
#include "unround/error.hpp"
#include "unround/exact.hpp"
#include "unround/jpegio.hpp"
#include "unround/pdhg.hpp"
#include "unround/results.hpp"
#include "unround/settings.hpp"

#include "primal_dual.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace unround::samples {

namespace {

using exact::u128;

// An unsigned integer of 64-bit words, the least first.
using Words = std::vector<std::uint64_t>;

void trim(Words& words) {
  while (!words.empty() && words.back() == 0u) words.pop_back();
}

// value 2^shift.
Words shifted(std::uint64_t value, std::size_t shift) {
  Words words(shift / 64u + 2u, 0);
  const std::size_t word = shift / 64u;
  const std::size_t bit = shift % 64u;
  words[word] = value << bit;
  if (bit != 0u) words[word + 1u] = value >> (64u - bit);
  trim(words);
  return words;
}

void add_small(Words& words, std::uint64_t value) {
  for (std::size_t k = 0; value != 0u; ++k) {
    if (k == words.size()) words.push_back(0);
    const std::uint64_t before = words[k];
    words[k] = before + value;
    value = words[k] < before ? 1u : 0u;
  }
}

// words - value, which is at least value.
void subtract_small(Words& words, std::uint64_t value) {
  for (std::size_t k = 0; value != 0u; ++k) {
    const std::uint64_t before = words[k];
    words[k] = before - value;
    value = before < value ? 1u : 0u;
  }
  trim(words);
}

void multiply_small(Words& words, std::uint64_t factor) {
  std::uint64_t carry = 0;
  for (std::uint64_t& word : words) {
    const u128 product = static_cast<u128>(word) * factor + carry;
    word = static_cast<std::uint64_t>(product);
    carry = static_cast<std::uint64_t>(product >> 64u);
  }
  if (carry != 0u) words.push_back(carry);
}

// The quotient by a small divisor, in place, and the remainder: each step by exact::divide,
// whose long division needs none of the compiler's runtime.
std::uint64_t divide_small(Words& words, std::uint64_t divisor) {
  u128 remainder = 0;
  for (std::size_t k = words.size(); k-- > 0;) {
    const exact::Division step = exact::divide((remainder << 64u) | words[k], divisor);
    words[k] = static_cast<std::uint64_t>(step.quotient);
    remainder = step.remainder;
  }
  trim(words);
  return static_cast<std::uint64_t>(remainder);
}

std::size_t bit_length(const Words& words) noexcept {
  if (words.empty()) return 0;
  return 64u * (words.size() - 1u) + static_cast<std::size_t>(std::bit_width(words.back()));
}

int compare(const Words& a, const Words& b) noexcept {
  if (a.size() != b.size()) return a.size() < b.size() ? -1 : 1;
  for (std::size_t k = a.size(); k-- > 0;) {
    if (a[k] != b[k]) return a[k] < b[k] ? -1 : 1;
  }
  return 0;
}

// words 2^shift.
Words shifted_left(const Words& words, std::size_t shift) {
  const std::size_t word = shift / 64u;
  const std::size_t bit = shift % 64u;
  Words result(words.size() + word + 1u, 0);
  for (std::size_t k = 0; k < words.size(); ++k) {
    result[k + word] |= words[k] << bit;
    if (bit != 0u) result[k + word + 1u] |= words[k] >> (64u - bit);
  }
  trim(result);
  return result;
}

bool bit_of(const Words& words, std::size_t at) noexcept { return ((words[at / 64u] >> (at % 64u)) & 1u) != 0u; }

// The double nearest to numerator / (divisor 2^power): the quotient, of 64 bits at
// least, its top 64 bits with a sticky bit where any bit below them, or the remainder,
// is not 0, converted once.
double nearest_quotient(Words numerator, std::uint64_t divisor, std::size_t power) {
  if (numerator.empty()) return 0.0;
  const std::size_t wanted = 64u + static_cast<std::size_t>(std::bit_width(divisor));
  const std::size_t length = bit_length(numerator);
  if (length < wanted) {
    numerator = shifted_left(numerator, wanted - length);
    power += wanted - length;
  }
  const std::uint64_t remainder = divide_small(numerator, divisor);
  const std::size_t below = bit_length(numerator) - 64u;
  std::uint64_t top = 0;
  for (std::size_t bit = 0; bit < 64u; ++bit) {
    if (bit_of(numerator, below + bit)) top |= std::uint64_t{1} << bit;
  }
  bool sticky = remainder != 0u;
  for (std::size_t at = 0; at < below && !sticky; ++at) sticky = bit_of(numerator, at);
  const double rounded = exact::nearest(static_cast<u128>(top | (sticky ? 1u : 0u)));
  return std::ldexp(rounded, static_cast<int>(below) - static_cast<int>(power));
}

// The slack as m 2^-K: m an integer of at most 53 bits, K at least 0; none where the
// slack is so large (2^20 levels or more) that it reaches past 0 and 255 whatever the
// level.
struct Dyadic {
  std::uint64_t m = 0;
  std::size_t k = 0;
};

std::optional<Dyadic> dyadic_of(double slack) noexcept {
  if (slack == 0.0) return Dyadic{};
  if (slack >= 1048576.0) return std::nullopt;
  int exponent = 0;
  const double fraction = std::frexp(slack, &exponent);
  // slack = fraction 2^exponent, fraction in [1/2, 1): m = fraction 2^53, K = 53 - exponent.
  const auto m = static_cast<std::uint64_t>(std::ldexp(fraction, 53));
  return Dyadic{.m = m, .k = static_cast<std::size_t>(53 - exponent)};
}

}  // namespace

Interval interval(std::uint32_t level, std::int32_t bits, double slack) noexcept {
  const std::uint64_t most = (std::uint64_t{1} << static_cast<std::uint32_t>(bits)) - 1u;
  const std::uint64_t denominator = 2u * most;
  const std::uint64_t v = level;
  Interval result;
  // The centre: 255 v / (2^b - 1) = 255 (2 v) / (2 (2^b - 1)).
  result.centre = nearest_quotient(shifted(510u * v, 0), denominator, 0);
  const std::optional<Dyadic> widened = dyadic_of(slack);
  if (!widened) {
    result.lower = 0.0;
    result.upper = 255.0;
    return result;
  }
  const std::size_t k = widened->k;
  const std::uint64_t twice = 2u * widened->m;
  // Lower: 255 ((2 v - 1) 2^K - 2 m) / (2 (2^b - 1) 2^K), 0 where it is not above 0.
  if (v > 0u) {
    Words numerator = shifted(2u * v - 1u, k);
    const Words floor = shifted(twice, 0);
    if (compare(numerator, floor) > 0) {
      subtract_small(numerator, twice);
      multiply_small(numerator, 255);
      result.lower = nearest_quotient(std::move(numerator), denominator, k);
    }
  }
  // Upper: 255 ((2 v + 1) 2^K + 2 m) / (2 (2^b - 1) 2^K), 255 where it is not below.
  Words numerator = shifted(2u * v + 1u, k);
  add_small(numerator, twice);
  const Words ceiling = shifted(denominator, k);
  if (compare(numerator, ceiling) >= 0) {
    result.upper = 255.0;
  } else {
    multiply_small(numerator, 255);
    result.upper = nearest_quotient(std::move(numerator), denominator, k);
  }
  return result;
}

std::expected<Model, Error> Model::make(Picture<const std::uint16_t> levels, std::int32_t bits, std::size_t first,
                                        std::size_t count, double slack, double mu) {
  if (bits < 1 || bits > 16) {
    return std::unexpected(
        Error{.code = Errc::unsupported, .message = std::format("samples of {} bits are not taken", bits)});
  }
  if (!(slack >= 0.0 && std::isfinite(slack)) || !(mu >= 0.0 && std::isfinite(mu))) {
    return std::unexpected(
        Error{.code = Errc::options,
              .message = std::format("the slack and mu are at least 0 and finite, not {} and {}", slack, mu)});
  }
  if (first + count > levels.extent(2) || levels.extent(0) == 0u || levels.extent(1) == 0u) {
    return std::unexpected(Error{.code = Errc::options, .message = "the channels are not the picture's"});
  }
  Model model;
  model.rows_ = levels.extent(0);
  model.columns_ = levels.extent(1);
  model.mu_ = mu;
  // The intervals of every level, once.
  const std::size_t level_count = std::size_t{1} << static_cast<std::uint32_t>(bits);
  std::vector<Interval> table(level_count);
  for (std::size_t v = 0; v < level_count; ++v) table[v] = interval(static_cast<std::uint32_t>(v), bits, slack);
  for (std::size_t c = first; c < first + count; ++c) {
    GridArray<double> lower(model.extents());
    GridArray<double> upper(model.extents());
    GridArray<double> centres(model.extents());
    for (std::size_t i = 0; i < model.rows_; ++i) {
      for (std::size_t j = 0; j < model.columns_; ++j) {
        const std::uint16_t level = levels[i, j, c];
        if (level >= level_count) {
          return std::unexpected(
              Error{.code = Errc::options, .message = std::format("a level of {} bits is below 2^{}", bits, bits)});
        }
        const Interval& ends = table[level];
        lower.view()[i, j] = ends.lower;
        upper.view()[i, j] = ends.upper;
        centres.view()[i, j] = ends.centre;
      }
    }
    model.lower_.push_back(std::move(lower));
    model.upper_.push_back(std::move(upper));
    model.centres_.push_back(std::move(centres));
  }
  return model;
}

void Model::prox(std::size_t c, Grid<const double> v, double tau, Grid<double> out) const noexcept {
  const double t = tau * mu_;
  const double r = 1.0 / (1.0 + t);
  const double* from = v.data_handle();
  const double* lows = lower_[c].entries().data();
  const double* highs = upper_[c].entries().data();
  const double* middles = centres_[c].entries().data();
  double* to = out.data_handle();
  const std::size_t count = v.size();
  for (std::size_t k = 0; k < count; ++k) {
    to[k] = std::min(std::max((from[k] + t * middles[k]) * r, lows[k]), highs[k]);
  }
}

double Model::value(std::size_t c, Grid<const double> x) const {
  GridArray<double> terms(extents());
  const double half = 0.5 * mu_;
  const double* from = x.data_handle();
  const double* middles = centres_[c].entries().data();
  double* to = terms.entries().data();
  const std::size_t count = x.size();
  for (std::size_t k = 0; k < count; ++k) {
    const double away = from[k] - middles[k];
    to[k] = half * (away * away);
  }
  return exact::sum_rows(std::as_const(terms).view());
}

double Model::conjugate(std::size_t c, Grid<const double> xi) const {
  GridArray<double> terms(extents());
  const double m = mu_;
  const bool weighted = m > 0.0;
  const double* from = xi.data_handle();
  const double* lows = lower_[c].entries().data();
  const double* highs = upper_[c].entries().data();
  const double* middles = centres_[c].entries().data();
  double* to = terms.entries().data();
  const std::size_t count = xi.size();
  for (std::size_t k = 0; k < count; ++k) {
    const double s = from[k];
    const double best = std::min(std::max(middles[k] + s / m, lows[k]), highs[k]);
    const double away = best - middles[k];
    const double quadratic = s * best - 0.5 * m * (away * away);
    const double support = std::max(s * lows[k], s * highs[k]);
    to[k] = weighted ? quadratic : support;
  }
  return exact::sum_rows(std::as_const(terms).view());
}

namespace {

// The data term of a model for the primal-dual method: the samples of the last
// proximal map of each channel (or its start), at which G is taken.
class SamplesData {
 public:
  explicit SamplesData(const Model& model) : model_(model) {
    for (std::size_t c = 0; c < model.channels(); ++c) points_.emplace_back(model.extents());
  }

  [[nodiscard]] std::size_t channels(void) const noexcept { return model_.channels(); }
  [[nodiscard]] std::size_t rows(void) const noexcept { return model_.rows(); }
  [[nodiscard]] std::size_t columns(void) const noexcept { return model_.columns(); }
  [[nodiscard]] GridExtents extents(void) const noexcept { return model_.extents(); }

  // The start: the centres. A start of coefficients is a JPEG file's, and refused.
  std::expected<void, Error> start(std::size_t c, const pdhg::Start* given, Grid<double> x) {
    if (given != nullptr && !given->coefficients.empty()) {
      return std::unexpected(Error{.code = Errc::options, .message = "a picture of samples starts from its levels"});
    }
    const Grid<const double> centres = model_.centres(c);
    std::ranges::copy(std::span<const double>(centres.data_handle(), centres.size()), x.data_handle());
    std::ranges::copy(std::span<const double>(centres.data_handle(), centres.size()), points_[c].entries().begin());
    return {};
  }

  void prox(std::size_t c, Grid<const double> v, double tau, Grid<double> out) {
    model_.prox(c, v, tau, out);
    std::ranges::copy(std::span<const double>(out.data_handle(), out.size()), points_[c].entries().begin());
  }

  [[nodiscard]] double value(std::size_t c) const { return model_.value(c, points_[c].view()); }

  [[nodiscard]] double conjugate(std::size_t c, Grid<const double> xi, double /*radius*/) const {
    return model_.conjugate(c, xi);
  }

 private:
  const Model& model_;
  std::vector<GridArray<double>> points_;
};

std::unexpected<Error> refuse(Errc code, std::string message) {
  return std::unexpected(Error{.code = code, .message = std::move(message)});
}

Errc errc_of(jpegio::Errc code) noexcept {
  switch (code) {
    case jpegio::Errc::unsupported:
    case jpegio::Errc::limit:
      return Errc::unsupported;
    case jpegio::Errc::argument:
    case jpegio::Errc::decode:
      return Errc::read;
    case jpegio::Errc::memory:
    case jpegio::Errc::encode:
      return Errc::internal;
  }
  return Errc::internal;
}

}  // namespace

std::expected<Result, Error> solve(const Model& model, pdhg::Model method, const pdhg::Regularizer& regularizer,
                                   const pdhg::Options& options, const pdhg::Observer& observer) {
  std::expected<pdhg::Plan, Error> made = pdhg::plan(method, regularizer, options, model.channels(), defaults);
  if (!made) return std::unexpected(made.error());
  SamplesData data(model);
  std::expected<pdhg::detail::Run, Error> ran = pdhg::detail::run(data, method, regularizer, *made, nullptr, observer);
  if (!ran) return std::unexpected(ran.error());
  return Result{.canvas = std::move(ran->canvas),
                .records = std::move(ran->records),
                .iterations = ran->iterations,
                .stop = ran->stop,
                .steps = made->steps};
}

bool is_png(std::span<const std::uint8_t> file) noexcept {
  constexpr std::array<std::uint8_t, 8> signature = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'};
  return file.size() >= signature.size() && std::ranges::equal(file.first(signature.size()), signature);
}

std::expected<Reconstructed, Error> reconstruct(std::span<const std::uint8_t> file, const Settings& settings,
                                                const pdhg::Observer& observer) {
  // What a PNG file takes of the settings: of the data term mu, a number, and the slack.
  constexpr std::array<std::string_view, 10> data_options = {"mu-scale",    "mu-power", "mu-chroma",  "weight-power",
                                                             "dc-weight",   "centres",  "slack-cost", "flat-weight",
                                                             "flat-levels", "flat-dc"};
  std::vector<std::string> errors;
  for (const std::string& name : settings.given) {
    if (std::ranges::find(data_options, name) != data_options.end()) {
      errors.push_back(std::format("--{} is not for a PNG file", name));
    }
  }
  if (settings.method != Method::tv && settings.method != Method::tgv) {
    errors.emplace_back("a PNG file is reconstructed by tv or tgv");
  }
  if (settings.data.size() != 1u) errors.emplace_back("the data term of a PNG file is one for every channel");
  const DataTerm& term = settings.data.front();
  const bool given_mu = std::ranges::find(settings.given, "mu") != settings.given.end();
  if (given_mu && !term.mu) errors.emplace_back("--mu of a PNG file is a number, not rule");
  if (!errors.empty()) {
    std::string message;
    for (const std::string& error : errors) message += message.empty() ? error : "; " + error;
    return refuse(Errc::options, std::move(message));
  }
  std::expected<jpegio::PngImage, jpegio::Error> image = jpegio::PngImage::read(file, settings.reading.max_pixels);
  if (!image) return refuse(errc_of(image.error().code), image.error().message);
  const Picture<const std::uint16_t> levels = image->levels();
  const auto channels = static_cast<std::size_t>(image->channels());
  const bool alpha = channels == 2u || channels == 4u;
  const std::size_t colours = alpha ? channels - 1u : channels;
  const double mu = term.mu ? *term.mu : 0.0;
  const pdhg::Model method = settings.method == Method::tv ? pdhg::Model::tv : pdhg::Model::tgv;
  Reconstructed result;
  result.bits = image->bits();
  result.alpha = alpha;
  const std::span<const std::uint8_t> profile = image->icc_profile();
  result.icc_profile.assign(profile.begin(), profile.end());
  result.warning = image->warning();
  std::expected<Model, Error> colour_model = Model::make(levels, image->bits(), 0, colours, term.slack, mu);
  if (!colour_model) return std::unexpected(colour_model.error());
  std::expected<Result, Error> solved = solve(*colour_model, method, settings.regularizer, settings.pdhg, observer);
  if (!solved) return std::unexpected(solved.error());
  result.colours = std::move(*solved);
  if (alpha) {
    // The alpha alone: one channel, the same method and weights, no channel weights.
    pdhg::Regularizer alone = settings.regularizer;
    alone.channel_weights.clear();
    std::expected<Model, Error> alpha_model = Model::make(levels, image->bits(), colours, 1, term.slack, mu);
    if (!alpha_model) return std::unexpected(alpha_model.error());
    std::expected<Result, Error> alpha_solved = solve(*alpha_model, method, alone, settings.pdhg, observer);
    if (!alpha_solved) return std::unexpected(alpha_solved.error());
    result.alphas = std::move(*alpha_solved);
  }
  const std::size_t rows = levels.extent(0);
  const std::size_t columns = levels.extent(1);
  result.picture = MdArray<double, PictureExtents>(PictureExtents{rows, columns, channels});
  const Picture<double> picture = result.picture.view();
  for (std::size_t c = 0; c < channels; ++c) {
    const Grid<const double> plane = c < colours ? result.colours.canvas[c].view() : result.alphas.canvas[0].view();
    for (std::size_t i = 0; i < rows; ++i) {
      for (std::size_t j = 0; j < columns; ++j) picture[i, j, c] = plane[i, j];
    }
  }
  return result;
}

}  // namespace unround::samples
