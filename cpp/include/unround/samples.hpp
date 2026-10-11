// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Pictures of rounded samples (docs/math.md, 11): each sample of a file of b bits
// stands for every value that rounds to its own level, on the scale of 0 to 255, and
// the reconstruction is the picture within those intervals that TV or TGV finds most
// natural, with a data term of weight mu toward the levels (0 by default). The channels
// are the file's as they are, coupled or apart; a file's alpha is a picture of its own.
#ifndef UNROUND_SAMPLES_HPP
#define UNROUND_SAMPLES_HPP

#include "unround/arrays.hpp"
#include "unround/error.hpp"
#include "unround/pdhg.hpp"
#include "unround/results.hpp"
#include "unround/settings.hpp"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <vector>

namespace unround::samples {

// The defaults of PNG files (docs/math.md, 11): a gap per sample of 1e-4 within 10000
// iterations, and TGV's ratio of the steps 0.3; TV's ratio is that of JPEG files.
inline constexpr pdhg::Defaults defaults{.tv_ratio = pdhg::tv_ratio,
                                         .tgv_ratio = 0.3,
                                         .tv_tolerance = 1e-4,
                                         .tgv_tolerance = 1e-4,
                                         .tv_iterations = 10000,
                                         .tgv_iterations = 10000};

// The interval of level v of a file of b bits (1 to 16), s = 255 / (2^b - 1). Without
// slack, the doubles of max(0, (v - 1/2) s) <= x < (v + 1/2) s, or x <= 255 at the last
// level: its ends, rationals, rounded inward, the lower one up to the least double at or
// above it and the upper one down to the greatest double below it. With a slack in
// levels, max(0, ((v - 1/2) - slack) s) and min(255, ((v + 1/2) + slack) s), computed in
// floating point with s the nearest double. And its centre v s, the nearest double.
struct Interval {
  double lower = 0.0;
  double upper = 0.0;
  double centre = 0.0;
};
[[nodiscard]] Interval interval(std::uint32_t level, std::int32_t bits, double slack) noexcept;

// The data term of the channels of a picture of levels: rows x columns x channels.
class Model {
 public:
  // The model of the levels of a file of `bits` bits, of channels first to first +
  // count - 1 of the picture, with a slack (at least 0) and a weight mu (at least 0).
  [[nodiscard]] static std::expected<Model, Error> make(Picture<const std::uint16_t> levels, std::int32_t bits,
                                                        std::size_t first, std::size_t count, double slack, double mu);

  [[nodiscard]] std::size_t channels(void) const noexcept { return lower_.size(); }
  [[nodiscard]] std::size_t rows(void) const noexcept { return rows_; }
  [[nodiscard]] std::size_t columns(void) const noexcept { return columns_; }
  [[nodiscard]] GridExtents extents(void) const noexcept { return grid_extents(rows_, columns_); }
  [[nodiscard]] double mu(void) const noexcept { return mu_; }
  [[nodiscard]] Grid<const double> lower(std::size_t c) const noexcept { return lower_[c].view(); }
  [[nodiscard]] Grid<const double> upper(std::size_t c) const noexcept { return upper_[c].view(); }
  [[nodiscard]] Grid<const double> centres(std::size_t c) const noexcept { return centres_[c].view(); }

  // The proximal map of tau G, sample by sample: clip((v + t c) r, a, b), t = tau mu,
  // r = 1 / (1 + t).
  void prox(std::size_t c, Grid<const double> v, double tau, Grid<double> out) const noexcept;
  // The same of the rows of the canvas from `first` on that v and out hold: the part of
  // the canvas that a thread takes.
  void prox_rows(std::size_t c, Grid<const double> v, double tau, Grid<double> out, std::size_t first) const noexcept;

  // G at samples within their intervals, mu / 2 sum (x - c)^2, summed row by row.
  [[nodiscard]] double value(std::size_t c, Grid<const double> x) const;

  // G* at a field: the sum of g*(s), sample by sample.
  [[nodiscard]] double conjugate(std::size_t c, Grid<const double> xi) const;

  // The terms of those sums of the rows of the canvas from `first` on that x (or xi)
  // and `terms` hold: the part of the canvas that a thread takes. The sum of the terms
  // of every row, in their order (exact::sum_rows), is the value.
  void value_terms(std::size_t c, Grid<const double> x, std::size_t first, Grid<double> terms) const noexcept;
  void conjugate_terms(std::size_t c, Grid<const double> xi, std::size_t first, Grid<double> terms) const noexcept;

 private:
  Model(void) = default;

  std::size_t rows_ = 0;
  std::size_t columns_ = 0;
  double mu_ = 0.0;
  std::vector<GridArray<double>> lower_;
  std::vector<GridArray<double>> upper_;
  std::vector<GridArray<double>> centres_;
};

// The reconstruction of the channels of a model: the last tilde point's picture,
// channel by channel, the records, the iterations, the stop and the steps.
struct Result {
  std::vector<GridArray<double>> canvas;
  std::vector<Record> records;
  std::uint64_t iterations = 0;
  Stop stop = Stop::iterations;
  pdhg::Steps steps;
};

// TV or TGV on the model by the primal-dual method, from the centres, with the defaults
// of PNG files unless the options say otherwise.
[[nodiscard]] std::expected<Result, Error> solve(const Model& model, pdhg::Model method,
                                                 const pdhg::Regularizer& regularizer, const pdhg::Options& options,
                                                 const pdhg::Observer& observer = {});

// A PNG file reconstructed (docs/cli.md, PNG files as input): its picture, rows x
// columns x channels, the colours and the alpha where it has one; the file's bits and
// ICC profile; the colours' solver and the alpha's.
struct Reconstructed {
  MdArray<double, PictureExtents> picture;
  std::int32_t bits = 8;
  bool alpha = false;
  std::vector<std::uint8_t> icc_profile;
  Result colours;
  Result alphas;
  std::string warning;
};

// Whether bytes begin with PNG's signature.
[[nodiscard]] bool is_png(std::span<const std::uint8_t> file) noexcept;

// A PNG file reconstructed with the settings: of the data term only mu (a number) and
// the slack, and TV or TGV; the data term's other options, given (settings.given), are
// refused (Errc::options), as are a palette file and one the C layer does not read
// (Errc::read or Errc::unsupported).
[[nodiscard]] std::expected<Reconstructed, Error> reconstruct(std::span<const std::uint8_t> file,
                                                              const Settings& settings,
                                                              const pdhg::Observer& observer = {});

}  // namespace unround::samples

#endif  // UNROUND_SAMPLES_HPP
