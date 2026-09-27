// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// What the tests of the C++ library share: checks that count and report where
// they fail, the unit roundoff and the bounds of k roundings, a fixed generator
// of numbers, and the reading of the exact references (conformance/references),
// whose values are pairs of doubles hi:lo that sum to the value to about 2^-106
// of it.
#ifndef UNROUND_TESTS_CHECK_HPP
#define UNROUND_TESTS_CHECK_HPP

#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <print>
#include <source_location>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace unround::test {

inline int& failures(void) noexcept {
  static int count = 0;
  return count;
}

inline int& checks(void) noexcept {
  static int count = 0;
  return count;
}

inline bool check(bool ok, std::string_view what, std::source_location where = std::source_location::current()) {
  ++checks();
  if (!ok) {
    ++failures();
    std::println(stderr, "{}:{}: check failed: {}", where.file_name(), where.line(), what);
  }
  return ok;
}

// The result of a test program: its checks, and 0 where every one of them held.
inline int finish(std::string_view name) {
  std::println("{}: {} checks, {} failed", name, checks(), failures());
  return failures() == 0 ? 0 : 1;
}

// The unit roundoff of binary64, and gamma_k = k u / (1 - k u) (Higham, 3.1).
inline constexpr double u = 0x1p-53;
[[nodiscard]] constexpr double gamma(int k) noexcept {
  const double ku = static_cast<double>(k) * u;
  return ku / (1.0 - ku);
}

// SplitMix64: numbers from a seed, the same on every system.
class Numbers {
 public:
  explicit Numbers(std::uint64_t seed) noexcept : state_(seed) {}

  std::uint64_t next(void) noexcept {
    state_ += 0x9E3779B97F4A7C15ull;
    std::uint64_t z = state_;
    z = (z ^ (z >> 30u)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27u)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31u);
  }

  // An integer in [low, high].
  std::int64_t between(std::int64_t low, std::int64_t high) noexcept {
    const auto span = static_cast<std::uint64_t>(high - low) + 1u;
    return low + static_cast<std::int64_t>(next() % span);
  }

  // A double in [low, high), of 53 random bits.
  double uniform(double low, double high) noexcept {
    const double unit = static_cast<double>(next() >> 11u) * 0x1p-53;
    return low + (high - low) * unit;
  }

 private:
  std::uint64_t state_;
};

// An exact value of the references: hi + lo.
struct Exact {
  double hi = 0.0;
  double lo = 0.0;
};

[[nodiscard]] inline double number(std::string_view text) {
  double value = 0.0;
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc{} || end != text.data() + text.size()) {
    std::println(stderr, "not a number: '{}'", text);
    ++failures();
  }
  return value;
}

[[nodiscard]] inline Exact exact_of(std::string_view text) {
  const std::size_t colon = text.find(':');
  if (colon == std::string_view::npos) return Exact{.hi = number(text), .lo = 0.0};
  return Exact{.hi = number(text.substr(0, colon)), .lo = number(text.substr(colon + 1u))};
}

// The fields of every line of a file that is not a comment, split at spaces.
[[nodiscard]] inline std::vector<std::vector<std::string>> lines_of(const std::filesystem::path& path) {
  std::vector<std::vector<std::string>> lines;
  std::ifstream file(path);
  if (!file) {
    std::println(stderr, "cannot read {}", path.string());
    ++failures();
    return lines;
  }
  std::string line;
  while (std::getline(file, line)) {
    if (line.ends_with('\r')) line.pop_back();
    if (line.empty() || line.starts_with('#')) continue;
    std::vector<std::string> fields;
    std::size_t start = 0;
    while (start < line.size()) {
      const std::size_t end = line.find(' ', start);
      const std::size_t stop = end == std::string::npos ? line.size() : end;
      if (stop > start) fields.emplace_back(line.substr(start, stop - start));
      start = stop + 1u;
    }
    lines.push_back(std::move(fields));
  }
  return lines;
}

}  // namespace unround::test

#define CHECK(...) ::unround::test::check(static_cast<bool>(__VA_ARGS__), #__VA_ARGS__)

#endif  // UNROUND_TESTS_CHECK_HPP
