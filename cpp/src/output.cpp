// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The levels of docs/math.md, 10, and the bytes of TIFF and PNM files.
//
// The levels are taken a channel and a row at a time: the row's samples copied out of
// the picture, then each step a loop over the row, the same operation on neighbouring
// values (vector code); the thresholds of a row are those of a row of the matrix,
// tiled across.

#include "unround/output.hpp"

#include "unround/arrays.hpp"

#include "blue_noise.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace unround::output {

namespace {

constexpr std::size_t side = 64;

// 1 - t of each rank R of the matrix, t = (2 R + 1) / 8192: exact, of 13 bits.
const std::array<double, side * side>& limits(void) noexcept {
  static const std::array<double, side * side> values = [](void) {
    std::array<double, side * side> table{};
    for (std::size_t k = 0; k < table.size(); ++k) {
      table[k] = 1.0 - (2.0 * static_cast<double>(blue_noise_ranks[k]) + 1.0) / 8192.0;
    }
    return table;
  }();
  return values;
}

// The shifts of the matrix of channels 0 to 3, apart.
constexpr std::array<std::array<std::size_t, 2>, 4> shifts = {{{0, 0}, {0, 32}, {32, 0}, {32, 32}}};

// y[j] = scale x[j] rounded, and e[j] the error of that rounding, exact: the product is
// y + e (docs/math.md, 10).
void scaled(double scale, std::span<const double> x, std::span<double> y, std::span<double> e) noexcept {
  const double* samples = x.data();
  double* products = y.data();
  double* errors = e.data();
  const std::size_t count = y.size();
  for (std::size_t j = 0; j < count; ++j) {
    const double product = scale * samples[j];
    products[j] = product;
    errors[j] = std::fma(scale, samples[j], -product);
  }
}

// levels[j] = the level of y[j] + e[j] by bounds[j], 1 - t of the dither or 1/2 of
// rounding (halves away from 0): floor(y), or the level above where the fraction of y
// passes the bound, or is the bound and e >= 0; clamped to 0 to top. y lies on the same
// side of floor(y) + bound, a double, as y + e does, unless it is that double itself,
// where e tells; floor(y) and the fraction are exact, and so are the comparisons.
void leveled(std::span<const double> y, std::span<const double> e, std::span<const double> bounds, double top,
             std::span<double> levels) noexcept {
  const double* values = y.data();
  const double* errors = e.data();
  const double* limits_row = bounds.data();
  double* out = levels.data();
  const std::size_t count = levels.size();
  for (std::size_t j = 0; j < count; ++j) {
    // Every value read before any is compared, and each choice a selection: vector code
    // with no masked loads, which NEON has none of.
    const double value = values[j];
    const double error = errors[j];
    const double bound = limits_row[j];
    const double whole = std::floor(value);
    const double fraction = value - whole;
    const double on = error >= 0.0 ? 1.0 : 0.0;
    const double at = fraction == bound ? on : 0.0;
    const double up = fraction > bound ? 1.0 : at;
    const double within = value < top ? whole + up : top;
    out[j] = value >= 0.0 ? within : 0.0;
  }
}

void put16(std::vector<std::uint8_t>& bytes, std::uint16_t value) {
  bytes.push_back(static_cast<std::uint8_t>(value & 0xFFu));
  bytes.push_back(static_cast<std::uint8_t>(value >> 8u));
}

void put32(std::vector<std::uint8_t>& bytes, std::uint32_t value) {
  for (std::uint32_t k = 0; k < 4u; ++k) bytes.push_back(static_cast<std::uint8_t>((value >> (8u * k)) & 0xFFu));
}

void put64(std::vector<std::uint8_t>& bytes, std::uint64_t value) {
  for (std::uint32_t k = 0; k < 8u; ++k) bytes.push_back(static_cast<std::uint8_t>((value >> (8u * k)) & 0xFFu));
}

// A TIFF field: its tag, type (3 SHORT, 4 LONG, 5 RATIONAL, 7 UNDEFINED), count, and its
// value's bytes, which go into the entry where they fit in 4 and after the entries
// otherwise.
struct Field {
  std::uint16_t tag;
  std::uint16_t type;
  std::uint32_t count;
  std::vector<std::uint8_t> value;
};

Field shorts(std::uint16_t tag, std::span<const std::uint16_t> values) {
  Field field{.tag = tag, .type = 3, .count = static_cast<std::uint32_t>(values.size()), .value = {}};
  for (const std::uint16_t value : values) put16(field.value, value);
  return field;
}

Field longs(std::uint16_t tag, std::uint32_t value) {
  Field field{.tag = tag, .type = 4, .count = 1, .value = {}};
  put32(field.value, value);
  return field;
}

Field rationals(std::uint16_t tag, std::span<const std::array<std::uint32_t, 2>> values) {
  Field field{.tag = tag, .type = 5, .count = static_cast<std::uint32_t>(values.size()), .value = {}};
  for (const std::array<std::uint32_t, 2>& value : values) {
    put32(field.value, value[0]);
    put32(field.value, value[1]);
  }
  return field;
}

}  // namespace

std::span<const std::uint16_t, 4096> blue_noise(void) noexcept { return blue_noise_ranks; }

LevelsArray levels_of(Picture<const double> picture, std::int32_t bits, Dither dither, bool apart) {
  const std::size_t rows = picture.extent(0);
  const std::size_t columns = picture.extent(1);
  const std::size_t channels = picture.extent(2);
  LevelsArray result(picture.extents());
  const Picture<std::uint16_t> out = result.view();
  const double top = bits == 16 ? 65535.0 : 255.0;
  const double scale = bits == 16 ? 257.0 : 1.0;
  const std::array<double, side * side>& table = limits();
  std::vector<double> row(columns);
  std::vector<double> products(columns);
  std::vector<double> errors(columns);
  std::vector<double> bounds(columns, 0.5);
  std::vector<double> levels(columns);
  for (std::size_t c = 0; c < channels; ++c) {
    const std::array<std::size_t, 2> shift = apart ? shifts[c % shifts.size()] : std::array<std::size_t, 2>{0, 0};
    for (std::size_t i = 0; i < rows; ++i) {
      for (std::size_t j = 0; j < columns; ++j) row[j] = picture[i, j, c];
      scaled(scale, row, products, errors);
      if (dither == Dither::blue_noise) {
        const std::size_t matrix_row = ((i + shift[0]) % side) * side;
        for (std::size_t j = 0; j < columns; ++j) bounds[j] = table[matrix_row + (j + shift[1]) % side];
      }
      leveled(products, errors, bounds, top, levels);
      for (std::size_t j = 0; j < columns; ++j) out[i, j, c] = static_cast<std::uint16_t>(levels[j]);
    }
  }
  return result;
}

bool turns(std::int32_t orientation) noexcept { return orientation >= 2 && orientation <= 8; }

std::vector<std::uint8_t> tiff_bytes(Picture<const double> picture, Photometric photometric, bool alpha,
                                     std::span<const std::uint8_t> icc_profile) {
  const std::size_t rows = picture.extent(0);
  const std::size_t columns = picture.extent(1);
  const std::size_t channels = picture.extent(2);
  const std::size_t samples = rows * columns * channels;
  constexpr std::size_t most = std::numeric_limits<std::uint32_t>::max();
  if (samples == 0u || samples > (most - 4096u - icc_profile.size()) / 8u) return {};
  const auto count = static_cast<std::uint16_t>(channels);
  std::vector<Field> fields;
  fields.push_back(longs(256, static_cast<std::uint32_t>(columns)));
  fields.push_back(longs(257, static_cast<std::uint32_t>(rows)));
  const std::vector<std::uint16_t> bits_per_sample(channels, 64);
  fields.push_back(shorts(258, bits_per_sample));
  fields.push_back(shorts(259, std::array<std::uint16_t, 1>{1}));
  const std::uint16_t interpretation = photometric == Photometric::grey ? 1 : photometric == Photometric::rgb ? 2 : 6;
  fields.push_back(shorts(262, std::array<std::uint16_t, 1>{interpretation}));
  const std::size_t strip_offset_index = fields.size();
  fields.push_back(longs(273, 0));
  fields.push_back(shorts(277, std::array<std::uint16_t, 1>{count}));
  fields.push_back(longs(278, static_cast<std::uint32_t>(rows)));
  fields.push_back(longs(279, static_cast<std::uint32_t>(samples * 8u)));
  fields.push_back(shorts(284, std::array<std::uint16_t, 1>{1}));
  if (alpha) fields.push_back(shorts(338, std::array<std::uint16_t, 1>{2}));
  const std::vector<std::uint16_t> formats(channels, 3);
  fields.push_back(shorts(339, formats));
  if (photometric == Photometric::ycbcr) {
    // JFIF's coefficients, no subsampling, and the full range.
    constexpr std::array<std::array<std::uint32_t, 2>, 3> coefficients = {{{299, 1000}, {587, 1000}, {114, 1000}}};
    fields.push_back(rationals(529, coefficients));
    fields.push_back(shorts(530, std::array<std::uint16_t, 2>{1, 1}));
    constexpr std::array<std::array<std::uint32_t, 2>, 6> range = {
        {{0, 1}, {255, 1}, {128, 1}, {255, 1}, {128, 1}, {255, 1}}};
    fields.push_back(rationals(532, range));
  }
  if (!icc_profile.empty()) {
    Field profile{.tag = 34675, .type = 7, .count = static_cast<std::uint32_t>(icc_profile.size()), .value = {}};
    profile.value.assign(icc_profile.begin(), icc_profile.end());
    fields.push_back(std::move(profile));
  }
  // The header, the entries, the values that do not fit in them, then the samples.
  const std::size_t entries_size = 2u + 12u * fields.size() + 4u;
  std::size_t extra = 8u + entries_size;
  std::vector<std::size_t> offsets(fields.size(), 0);
  for (std::size_t k = 0; k < fields.size(); ++k) {
    if (fields[k].value.size() > 4u) {
      offsets[k] = extra;
      extra += fields[k].value.size() + (fields[k].value.size() % 2u);
    }
  }
  const std::size_t data_offset = extra;
  fields[strip_offset_index].value.clear();
  put32(fields[strip_offset_index].value, static_cast<std::uint32_t>(data_offset));
  std::vector<std::uint8_t> bytes;
  bytes.reserve(data_offset + samples * 8u);
  bytes.push_back('I');
  bytes.push_back('I');
  put16(bytes, 42);
  put32(bytes, 8);
  put16(bytes, static_cast<std::uint16_t>(fields.size()));
  for (std::size_t k = 0; k < fields.size(); ++k) {
    const Field& field = fields[k];
    put16(bytes, field.tag);
    put16(bytes, field.type);
    put32(bytes, field.count);
    if (field.value.size() > 4u) {
      put32(bytes, static_cast<std::uint32_t>(offsets[k]));
    } else {
      std::array<std::uint8_t, 4> inline_value{};
      std::ranges::copy(field.value, inline_value.begin());
      bytes.insert(bytes.end(), inline_value.begin(), inline_value.end());
    }
  }
  put32(bytes, 0);
  for (const Field& field : fields) {
    if (field.value.size() <= 4u) continue;
    bytes.insert(bytes.end(), field.value.begin(), field.value.end());
    if (field.value.size() % 2u != 0u) bytes.push_back(0);
  }
  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t j = 0; j < columns; ++j) {
      for (std::size_t c = 0; c < channels; ++c) put64(bytes, std::bit_cast<std::uint64_t>(picture[i, j, c]));
    }
  }
  return bytes;
}

std::vector<std::uint8_t> pnm_bytes(Picture<const std::uint16_t> levels, std::int32_t bits) {
  const std::size_t rows = levels.extent(0);
  const std::size_t columns = levels.extent(1);
  const std::size_t channels = levels.extent(2);
  const std::string header = std::string(channels == 1u ? "P5" : "P6") + "\n" + std::to_string(columns) + " " +
                             std::to_string(rows) + "\n" + (bits == 16 ? "65535" : "255") + "\n";
  std::vector<std::uint8_t> bytes(header.begin(), header.end());
  bytes.reserve(header.size() + rows * columns * channels * (bits == 16 ? 2u : 1u));
  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t j = 0; j < columns; ++j) {
      for (std::size_t c = 0; c < channels; ++c) {
        const std::uint16_t level = levels[i, j, c];
        if (bits == 16) bytes.push_back(static_cast<std::uint8_t>(level >> 8u));
        bytes.push_back(static_cast<std::uint8_t>(level & 0xFFu));
      }
    }
  }
  return bytes;
}

}  // namespace unround::output
