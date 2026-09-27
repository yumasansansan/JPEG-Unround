// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Tests of the DCT (docs/math.md, 1.1): every entry of the basis is the double
// nearest to its value (conformance/references/basis.txt); the blocks of the
// references transform, forward and back, to within the bounds of their rounding
// of the exact values; and a round trip returns to within the bounds of both
// transforms.

#include "unround/dct.hpp"

#include "unround/arrays.hpp"

#include "support/check.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

using unround::Block8x8;
using unround::dct::Block;
using unround::test::Exact;
using unround::test::gamma;
using unround::test::u;

// The bound of a transform's rounding, relatively to |C| |X| |C|^T: the even and odd
// halves round each output of an 8-point transform within e1 = (1 + u)^2 (1 +
// gamma_4) - 1 of its exact value times sum |C| |x|, which is within gamma_6; two of
// them within e1 (2 + e1) (docs/math.md, 1.1 and 9.3).
const double epsilon = gamma(6) * (2.0 + gamma(6));

// The bounds are sums of products of at most 64 terms, each with at most 3 roundings:
// computed within gamma_200 of their values, by which they are raised.
const double bound_rounding = 1.0 + gamma(200);

// |C| |X| |C|^T, entry (v, u), and |C|^T |Y| |C|, entry (m, n).
Block forward_magnitudes(const Block& samples) {
  const unround::dct::Basis& c = unround::dct::basis();
  const Block8x8<const double> x(samples.data());
  Block result{};
  const Block8x8<double> out(result.data());
  for (std::size_t v = 0; v < 8u; ++v) {
    for (std::size_t w = 0; w < 8u; ++w) {
      double sum = 0.0;
      for (std::size_t m = 0; m < 8u; ++m) {
        for (std::size_t n = 0; n < 8u; ++n) sum += std::abs(c[v][m]) * std::abs(c[w][n]) * std::abs(x[m, n]);
      }
      out[v, w] = sum;
    }
  }
  return result;
}

Block inverse_magnitudes(const Block& coefficients) {
  const unround::dct::Basis& c = unround::dct::basis();
  const Block8x8<const double> y(coefficients.data());
  Block result{};
  const Block8x8<double> out(result.data());
  for (std::size_t m = 0; m < 8u; ++m) {
    for (std::size_t n = 0; n < 8u; ++n) {
      double sum = 0.0;
      for (std::size_t v = 0; v < 8u; ++v) {
        for (std::size_t w = 0; w < 8u; ++w) sum += std::abs(c[v][m]) * std::abs(c[w][n]) * std::abs(y[v, w]);
      }
      out[m, n] = sum;
    }
  }
  return result;
}

// Whether a computed value lies within `bound` of the exact hi + lo. The error is
// taken in two subtractions, each within u of its value, so that the comparison asks
// for a hair less than the bound.
bool within(double value, const Exact& exact, double bound) {
  const double error = std::abs((value - exact.hi) - exact.lo);
  return error <= bound * (1.0 - 4.0 * u);
}

void basis(const std::filesystem::path& path) {
  const unround::dct::Basis& c = unround::dct::basis();
  int entries = 0;
  for (const std::vector<std::string>& fields : unround::test::lines_of(path)) {
    const auto k = static_cast<std::size_t>(std::stoul(fields.at(0)));
    const auto n = static_cast<std::size_t>(std::stoul(fields.at(1)));
    const Exact exact = unround::test::exact_of(fields.at(2));
    // hi is the double nearest to the entry: the reference is exact to 2^-106 of it.
    CHECK(std::bit_cast<std::uint64_t>(c[k][n]) == std::bit_cast<std::uint64_t>(exact.hi));
    ++entries;
  }
  CHECK(entries == 64);
}

Block block_of(const std::vector<std::string>& fields) {
  Block values{};
  for (std::size_t k = 0; k < 64u; ++k) values[k] = unround::test::number(fields.at(k + 1u));
  return values;
}

void transforms(const std::filesystem::path& path) {
  const std::vector<std::vector<std::string>> lines = unround::test::lines_of(path);
  int forwards = 0;
  int inverses = 0;
  for (std::size_t index = 0; index + 1u < lines.size(); ++index) {
    const std::vector<std::string>& given = lines[index];
    const std::vector<std::string>& exact = lines[index + 1u];
    if (exact.at(0) != "exact" || (given.at(0) != "forward" && given.at(0) != "inverse")) continue;
    const Block input = block_of(given);
    const bool forward = given.at(0) == "forward";
    const Block output = forward ? unround::dct::forward(input) : unround::dct::inverse(input);
    const Block magnitudes = forward ? forward_magnitudes(input) : inverse_magnitudes(input);
    for (std::size_t k = 0; k < 64u; ++k) {
      CHECK(within(output[k], unround::test::exact_of(exact.at(k + 1u)), epsilon * magnitudes[k] * bound_rounding));
    }
    ++(forward ? forwards : inverses);
  }
  CHECK(forwards == 5 && inverses == 3);
}

void round_trip(void) {
  unround::test::Numbers numbers(5);
  for (int trial = 0; trial < 200; ++trial) {
    Block x{};
    for (double& sample : x) sample = static_cast<double>(numbers.between(-128, 127));
    const Block y = unround::dct::forward(x);
    const Block z = unround::dct::inverse(y);
    // The inverse's own rounding, and the forward's carried through it:
    // |C|^T (epsilon |C| |X| |C|^T) |C|, each entry of the inverse within 1 + epsilon of
    // its exact map.
    const Block own = inverse_magnitudes(y);
    const Block carried = inverse_magnitudes(forward_magnitudes(x));
    for (std::size_t k = 0; k < 64u; ++k) {
      const double bound = (epsilon * own[k] + epsilon * (1.0 + epsilon) * carried[k]) * bound_rounding;
      CHECK(std::abs(z[k] - x[k]) <= bound * (1.0 - 2.0 * u));
    }
  }
  // The canvas transforms are the block ones, block by block, to the last bit: the
  // blocks at the top left of a canvas wider and taller than they are.
  const std::size_t rows = 2;
  const std::size_t columns = 3;
  unround::GridArray<double> canvas(unround::grid_extents(8u * rows + 3u, 8u * columns + 5u));
  for (double& sample : canvas.entries()) sample = static_cast<double>(numbers.between(0, 255));
  unround::BlocksArray<double> coefficients(unround::block_extents(rows, columns));
  unround::dct::forward(canvas.view(), coefficients.view());
  unround::GridArray<double> back(canvas.extents(), -1.0);
  unround::dct::inverse(std::as_const(coefficients).view(), back.view());
  for (std::size_t by = 0; by < rows; ++by) {
    for (std::size_t bx = 0; bx < columns; ++bx) {
      Block samples{};
      const Block8x8<double> into(samples.data());
      for (std::size_t m = 0; m < 8u; ++m) {
        for (std::size_t n = 0; n < 8u; ++n) into[m, n] = canvas.view()[8u * by + m, 8u * bx + n];
      }
      const Block transformed = unround::dct::forward(samples);
      const Block8x8<const double> expected(transformed.data());
      Block given{};
      const Block8x8<double> given_view(given.data());
      // (w for the horizontal frequency: u is the unit roundoff here.)
      for (std::size_t v = 0; v < 8u; ++v) {
        for (std::size_t w = 0; w < 8u; ++w) {
          CHECK(std::bit_cast<std::uint64_t>(coefficients.view()[by, bx, v, w]) ==
                std::bit_cast<std::uint64_t>(expected[v, w]));
          given_view[v, w] = coefficients.view()[by, bx, v, w];
        }
      }
      const Block restored = unround::dct::inverse(given);
      const Block8x8<const double> restored_samples(restored.data());
      for (std::size_t m = 0; m < 8u; ++m) {
        for (std::size_t n = 0; n < 8u; ++n) {
          CHECK(std::bit_cast<std::uint64_t>(back.view()[8u * by + m, 8u * bx + n]) ==
                std::bit_cast<std::uint64_t>(restored_samples[m, n]));
        }
      }
    }
  }
  // The samples beyond the blocks are left as they were.
  for (std::size_t i = 0; i < back.extent(0); ++i) {
    for (std::size_t j = 0; j < back.extent(1); ++j) {
      if (i >= 8u * rows || j >= 8u * columns) CHECK(back.view()[i, j] == -1.0);
    }
  }
}

int run(std::span<char* const> arguments) {
  if (arguments.size() < 2u) return 2;
  const std::filesystem::path references = arguments[1];
  basis(references / "basis.txt");
  transforms(references / "dct.txt");
  round_trip();
  return unround::test::finish("dct");
}

}  // namespace

int main(int argc, char** argv) {
  // A test that throws fails with what it says, rather than ending the program without
  // a word.
  try {
    return run(std::span<char* const>{argv, static_cast<std::size_t>(argc)});
  } catch (const std::exception& error) {
    (void)std::fputs(error.what(), stderr);
    (void)std::fputc('\n', stderr);
    return EXIT_FAILURE;
  }
}
