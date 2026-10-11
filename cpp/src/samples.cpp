// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Pictures of rounded samples: docs/math.md, 11.
//
// Without slack, the ends of the intervals are rationals, 255 (2 v -+ 1) / (2 (2^b - 1)),
// rounded inward: each is divided into a quotient of 53 bits, the rational rounded down,
// and a remainder, which says whether the rational is that double itself. With a slack,
// the ends are computed in floating point.

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
#include <cstdlib>
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

// A positive rational n / d as q 2^-p, q of 53 bits (2^52 <= q < 2^53): the rational
// rounded down to a double, and whether anything is left below it.
struct Truncated {
  std::uint64_t q = 0;
  int p = 0;
  bool inexact = false;
};

Truncated truncated(std::uint64_t numerator, std::uint64_t denominator) noexcept {
  if (numerator == 0u) return Truncated{};
  // n 2^p / d near 2^52 from the lengths, then a step or two to put it within [2^52,
  // 2^53); for the ends, n < 2^26 and d < 2^18, so that n 2^p stays below 2^97.
  int p = 52 + std::bit_width(denominator) - std::bit_width(numerator);
  exact::Division division{};
  while (true) {
    division = exact::divide(static_cast<u128>(numerator) << static_cast<unsigned>(p), denominator);
    if (division.quotient >= (u128{1} << 53u)) {
      --p;
    } else if (division.quotient < (u128{1} << 52u)) {
      ++p;
    } else {
      break;
    }
  }
  return Truncated{.q = static_cast<std::uint64_t>(division.quotient), .p = p, .inexact = division.remainder != 0u};
}

// The least double at or above n / d.
double at_or_above(std::uint64_t numerator, std::uint64_t denominator) noexcept {
  const Truncated r = truncated(numerator, denominator);
  return std::ldexp(static_cast<double>(r.q + (r.inexact ? 1u : 0u)), -r.p);
}

// The greatest double below n / d.
double under(std::uint64_t numerator, std::uint64_t denominator) noexcept {
  const Truncated r = truncated(numerator, denominator);
  const double down = std::ldexp(static_cast<double>(r.q), -r.p);
  return r.inexact ? down : std::nextafter(down, 0.0);
}

}  // namespace

Interval interval(std::uint32_t level, std::int32_t bits, double slack) noexcept {
  const std::uint64_t top = (std::uint64_t{1} << static_cast<std::uint32_t>(bits)) - 1u;
  const std::uint64_t v = level;
  Interval result;
  const exact::MaybeRational centre = exact::rational(255 * static_cast<exact::i128>(v), static_cast<exact::i128>(top));
  result.centre = exact::nearest(*centre);
  if (slack == 0.0) {
    // max(0, (v - 1/2) s) <= x < (v + 1/2) s, s = 255 / top, or x <= 255 at the last level.
    result.lower = v == 0u ? 0.0 : at_or_above(255u * (2u * v - 1u), 2u * top);
    result.upper = v == top ? 255.0 : under(255u * (2u * v + 1u), 2u * top);
    return result;
  }
  const double step = exact::nearest(*exact::rational(255, static_cast<exact::i128>(top)));
  const double own = static_cast<double>(v);
  const double lower = ((own - 0.5) - slack) * step;
  const double upper = ((own + 0.5) + slack) * step;
  result.lower = lower > 0.0 ? lower : 0.0;
  result.upper = upper < 255.0 ? upper : 255.0;
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
  if (v.extents() != extents()) std::abort();
  prox_rows(c, v, tau, out, 0);
}

void Model::prox_rows(std::size_t c, Grid<const double> v, double tau, Grid<double> out,
                      std::size_t first) const noexcept {
  if (out.extents() != v.extents() || v.extent(1) != columns() || first > rows() || v.extent(0) > rows() - first) {
    std::abort();
  }
  const double t = tau * mu_;
  const double r = 1.0 / (1.0 + t);
  // The part's samples of the intervals and the centres.
  const std::size_t offset = first * columns();
  const double* from = v.data_handle();
  const double* lows = lower_[c].entries().data() + offset;
  const double* highs = upper_[c].entries().data() + offset;
  const double* middles = centres_[c].entries().data() + offset;
  double* to = out.data_handle();
  const std::size_t count = v.size();
  for (std::size_t k = 0; k < count; ++k) {
    to[k] = std::min(std::max((from[k] + t * middles[k]) * r, lows[k]), highs[k]);
  }
}

double Model::value(std::size_t c, Grid<const double> x) const {
  GridArray<double> terms(extents());
  value_terms(c, x, 0, terms.view());
  return exact::sum_rows(std::as_const(terms).view());
}

void Model::value_terms(std::size_t c, Grid<const double> x, std::size_t first, Grid<double> terms) const noexcept {
  if (terms.extents() != x.extents() || x.extent(1) != columns_ || first > rows_ || x.extent(0) > rows_ - first) {
    std::abort();
  }
  const double half = 0.5 * mu_;
  const double* from = x.data_handle();
  const double* middles = centres_[c].entries().data() + (first * columns_);
  double* to = terms.data_handle();
  const std::size_t count = x.size();
  for (std::size_t k = 0; k < count; ++k) {
    const double away = from[k] - middles[k];
    to[k] = half * (away * away);
  }
}

double Model::conjugate(std::size_t c, Grid<const double> xi) const {
  GridArray<double> terms(extents());
  conjugate_terms(c, xi, 0, terms.view());
  return exact::sum_rows(std::as_const(terms).view());
}

void Model::conjugate_terms(std::size_t c, Grid<const double> xi, std::size_t first,
                            Grid<double> terms) const noexcept {
  if (terms.extents() != xi.extents() || xi.extent(1) != columns_ || first > rows_ || xi.extent(0) > rows_ - first) {
    std::abort();
  }
  const double m = mu_;
  const bool weighted = m > 0.0;
  const std::size_t offset = first * columns_;
  const double* from = xi.data_handle();
  const double* lows = lower_[c].entries().data() + offset;
  const double* highs = upper_[c].entries().data() + offset;
  const double* middles = centres_[c].entries().data() + offset;
  double* to = terms.data_handle();
  const std::size_t count = xi.size();
  for (std::size_t k = 0; k < count; ++k) {
    const double s = from[k];
    const double best = std::min(std::max(middles[k] + s / m, lows[k]), highs[k]);
    const double away = best - middles[k];
    const double quadratic = s * best - 0.5 * m * (away * away);
    const double support = std::max(s * lows[k], s * highs[k]);
    to[k] = weighted ? quadratic : support;
  }
}

namespace {

// The data term of a model for the primal-dual method: the samples of the last
// proximal map of each channel (or its start), at which G is taken.
class SamplesData {
 public:
  explicit SamplesData(const Model& model) : model_(model) {
    for (std::size_t c = 0; c < model.channels(); ++c) {
      points_.emplace_back(model.extents());
      value_terms_.emplace_back(model.extents());
      conjugate_terms_.emplace_back(model.extents());
    }
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

  // A part of the canvas that a thread takes may begin at any row.
  [[nodiscard]] static std::size_t band(void) noexcept { return 1; }

  // The proximal map of channel c on the rows of a part of the canvas.
  void prox_rows(std::size_t c, Grid<const double> v, double tau, Grid<double> out, operators::Rows rows,
                 std::vector<double>& /*row*/) {
    const std::size_t width = v.extent(1);
    const std::size_t offset = rows.first * width;
    const std::size_t count = rows.end - rows.first;
    const Grid<const double> from(v.data_handle() + offset, count, width);
    const Grid<double> to(out.data_handle() + offset, count, width);
    model_.prox_rows(c, from, tau, to, rows.first);
    std::ranges::copy(std::span<const double>(to.data_handle(), to.size()),
                      points_[c].entries().begin() + static_cast<std::ptrdiff_t>(offset));
  }

  // G at the samples of the last proximal map, and G* at xi, of channel c: their terms
  // of the rows of a part, and their sums once every part has taken its own.
  void value_terms(std::size_t c, operators::Rows rows) {
    model_.value_terms(c, rows_of(std::as_const(points_[c]).view(), rows), rows.first,
                       rows_of(value_terms_[c].view(), rows));
  }

  [[nodiscard]] double value_sum(std::size_t c) const { return exact::sum_rows(value_terms_[c].view()); }

  void conjugate_terms(std::size_t c, Grid<const double> xi, double /*radius*/, operators::Rows rows,
                       std::vector<double>& /*row*/) {
    model_.conjugate_terms(c, rows_of(xi, rows), rows.first, rows_of(conjugate_terms_[c].view(), rows));
  }

  [[nodiscard]] double conjugate_sum(std::size_t c) const { return exact::sum_rows(conjugate_terms_[c].view()); }

 private:
  // The rows of a part of a grid of the canvas's shape.
  template <typename T>
  [[nodiscard]] static Grid<T> rows_of(Grid<T> grid, operators::Rows rows) noexcept {
    return Grid<T>(grid.data_handle() + (rows.first * grid.extent(1)), rows.end - rows.first, grid.extent(1));
  }

  const Model& model_;
  std::vector<GridArray<double>> points_;
  std::vector<GridArray<double>> value_terms_;
  std::vector<GridArray<double>> conjugate_terms_;
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
