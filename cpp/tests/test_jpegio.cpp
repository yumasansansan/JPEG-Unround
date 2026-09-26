// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Tests of the C++ face of the C layer: a read through it returns what the C
// layer returns; what it returns owns its memory, which a move hands over and
// nothing frees twice; a failure comes back as a value; and several threads
// read at once, each getting what one thread alone gets. A PNG file written
// through it reads back with libpng as the picture that went in, with its ICC
// profile, or without one the check refuses; a wrong picture is refused.

#include "unround/jpegio.hpp"

#include "test_jpeg.h"
#include "test_png.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <print>
#include <source_location>
#include <span>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

std::atomic<int> failures{0};
std::atomic<int> checks{0};

bool check(bool ok, std::string_view what, std::source_location where = std::source_location::current()) {
  checks.fetch_add(1, std::memory_order_relaxed);
  if (!ok) {
    failures.fetch_add(1, std::memory_order_relaxed);
    std::println(stderr, "{}:{}: check failed: {}", where.file_name(), where.line(), what);
  }
  return ok;
}

#define CHECK(...) check(static_cast<bool>(__VA_ARGS__), #__VA_ARGS__)

namespace jpegio = unround::jpegio;

// A file written by libjpeg's encoder, with the coefficients that went in.
struct File {
  std::vector<std::uint8_t> data;
  std::vector<std::vector<std::int16_t>> coefficients;
};

File make_file(std::uint32_t width, std::uint32_t height, std::int32_t color_space, std::int32_t luma_factor,
               std::uint32_t seed, bool progressive) {
  test_jpeg spec{};
  spec.width = width;
  spec.height = height;
  spec.color_space = color_space;
  spec.h_samp_factor[0] = luma_factor;
  spec.v_samp_factor[0] = luma_factor;
  spec.quant_table_slot[1] = 1;
  spec.quant_table_slot[2] = 1;
  spec.progressive = progressive ? 1 : 0;
  for (std::size_t slot = 0; slot < 2; ++slot) {
    for (std::size_t k = 0; k < 64; ++k) {
      spec.quant_tables[slot][k] = static_cast<std::uint16_t>(1 + (seed + 7 * k + 13 * slot) % 50);
    }
  }
  File file;
  const std::int32_t components = test_jpeg_components(&spec);
  file.coefficients.resize(static_cast<std::size_t>(components));
  for (std::int32_t ci = 0; ci < components; ++ci) {
    const std::size_t count = static_cast<std::size_t>(test_jpeg_blocks_wide(&spec, ci)) *
                              static_cast<std::size_t>(test_jpeg_blocks_high(&spec, ci)) * 64;
    std::vector<std::int16_t>& coefficients = file.coefficients[static_cast<std::size_t>(ci)];
    coefficients.resize(count);
    for (std::size_t i = 0; i < count; ++i) {
      const std::size_t k = i % 64;
      const std::uint32_t mixed = (static_cast<std::uint32_t>(i) * 2654435761U) ^ (seed * 40503U);
      if (k == 0) {
        coefficients[i] = static_cast<std::int16_t>(static_cast<std::int32_t>(mixed % 401U) - 200);
      } else if (k < 12 && mixed % 3U == 0) {
        coefficients[i] = static_cast<std::int16_t>(static_cast<std::int32_t>(mixed % 41U) - 20);
      }
    }
    spec.coefficients[ci] = coefficients.data();
  }
  std::uint8_t* data = nullptr;
  std::size_t size = 0;
  char message[256];
  if (!CHECK(test_jpeg_write(&spec, &data, &size, message, sizeof message) == 0)) {
    std::println(stderr, "  {}", message);
    return file;
  }
  file.data.assign(data, data + size);
  std::free(data);
  return file;
}

bool same_coefficients(const jpegio::Image& image, const File& file) {
  if (image.components().size() != file.coefficients.size()) return false;
  for (std::size_t ci = 0; ci < file.coefficients.size(); ++ci) {
    if (!std::ranges::equal(jpegio::coefficients(image.components()[ci]), file.coefficients[ci])) return false;
  }
  return true;
}

void test_read(void) {
  const File file = make_file(37, 21, TEST_JPEG_YCBCR, 2, 1, false);
  auto image = jpegio::Image::read(file.data);
  if (!CHECK(image.has_value())) return;
  CHECK(image->width() == 37 && image->height() == 21);
  CHECK(image->color_space() == jpegio::ColorSpace::ycbcr);
  CHECK(!image->progressive() && !image->arithmetic());
  CHECK(image->warnings() == 0 && image->warning().empty());
  CHECK(image->icc_profile().empty());
  CHECK(image->components().size() == 3);
  CHECK(same_coefficients(*image, file));

  const unround_jpegio_component& luma = image->components()[0];
  CHECK(luma.h_samp_factor == 2 && luma.v_samp_factor == 2);
  CHECK(luma.width_in_blocks == 5 && luma.height_in_blocks == 3);
  CHECK(jpegio::coefficients(luma).size() == std::size_t{5} * 3 * 64);
  CHECK(std::ranges::equal(jpegio::block(luma, 4, 2),
                           std::span{file.coefficients[0]}.subspan(std::size_t{2 * 5 + 4} * 64, 64)));
  CHECK(jpegio::quant_table(luma)[0] == static_cast<std::uint16_t>(1 + 1 % 50));
  CHECK(image->components()[1].quant_table_slot == 1);
}

void test_ownership(void) {
  const File file = make_file(16, 16, TEST_JPEG_GRAYSCALE, 1, 2, true);
  auto read = jpegio::Image::read(file.data);
  if (!CHECK(read.has_value())) return;
  jpegio::Image first = std::move(*read);
  CHECK(first.components().size() == 1 && first.progressive());
  jpegio::Image second = std::move(first);
  // A moved-from image is empty, which is what is checked.
  CHECK(first.components().empty());  // NOLINT(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  CHECK(second.components().size() == 1 && same_coefficients(second, file));
  first = std::move(second);
  CHECK(second.components().empty());  // NOLINT(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  CHECK(same_coefficients(first, file));
  jpegio::Image empty;
  CHECK(empty.components().empty() && empty.width() == 0);
  first = std::move(empty);
  CHECK(first.components().empty());
}

void test_errors(void) {
  static constexpr std::uint8_t text[] = {'n', 'o', 't', ' ', 'a', ' ', 'J', 'P', 'E', 'G'};
  auto image = jpegio::Image::read(text);
  if (CHECK(!image.has_value())) {
    CHECK(image.error().code == jpegio::Errc::decode);
    CHECK(image.error().message.contains("Not a JPEG file"));
  }
  auto nothing = jpegio::Image::read({});
  CHECK(!nothing.has_value() && nothing.error().code == jpegio::Errc::decode);

  const File file = make_file(40, 30, TEST_JPEG_YCBCR, 1, 3, false);
  auto limited = jpegio::Image::read(file.data, jpegio::Options{.max_pixels = std::uint64_t{40} * 30 - 1});
  CHECK(!limited.has_value() && limited.error().code == jpegio::Errc::limit);
  const auto allowed = jpegio::Image::read(file.data, jpegio::Options{.max_pixels = std::uint64_t{40} * 30});
  CHECK(allowed.has_value());

  // Cut short: a warning, or an error when warnings are errors.
  const std::span<const std::uint8_t> cut = std::span{file.data}.first(file.data.size() - 40);
  auto warned = jpegio::Image::read(cut);
  if (CHECK(warned.has_value())) CHECK(warned->warnings() > 0 && warned->warning().contains("Premature end"));
  auto strict = jpegio::Image::read(cut, jpegio::Options{.warnings_are_errors = true});
  CHECK(!strict.has_value() && strict.error().code == jpegio::Errc::decode);
}

void test_planes(void) {
  const File file = make_file(33, 17, TEST_JPEG_YCBCR, 2, 4, false);
  auto planes = jpegio::Planes::decode(file.data);
  if (!CHECK(planes.has_value())) return;
  CHECK(planes->planes().size() == 3);
  const unround_jpegio_plane& luma = planes->planes()[0];
  CHECK(luma.width == 40 && luma.height == 24);
  CHECK(jpegio::samples(luma).size() == std::size_t{luma.stride} * luma.height);
  const unround_jpegio_plane& chroma = planes->planes()[1];
  CHECK(chroma.width == 24 && chroma.height == 16);
  const jpegio::Planes moved = std::move(*planes);
  // Moved-from planes are empty, which is what is checked.
  CHECK(planes->planes().empty());  // NOLINT(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  CHECK(moved.planes().size() == 3);
}

// Threads read files at once, each a different one in turn, and each result has
// to be what a read on one thread gives.
void test_threads(void) {
  constexpr std::size_t thread_count = 8;
  constexpr int rounds = 20;
  std::vector<File> files;
  files.reserve(4);
  for (std::uint32_t i = 0; i < 4; ++i) {
    files.push_back(make_file(24 + 8 * i, 16 + 5 * i, i % 2 == 0 ? TEST_JPEG_YCBCR : TEST_JPEG_GRAYSCALE,
                              i % 3 == 0 ? 2 : 1, 10 + i, i % 2 == 1));
  }
  std::barrier start{static_cast<std::ptrdiff_t>(thread_count)};
  std::atomic<int> mismatches{0};
  {
    std::vector<std::jthread> threads;
    threads.reserve(thread_count);
    for (std::size_t t = 0; t < thread_count; ++t) {
      threads.emplace_back([&files, &start, &mismatches, t](void) {
        start.arrive_and_wait();
        for (int r = 0; r < rounds; ++r) {
          const File& file = files[(t + static_cast<std::size_t>(r)) % files.size()];
          auto image = jpegio::Image::read(file.data);
          auto planes = jpegio::Planes::decode(file.data);
          if (!image.has_value() || !same_coefficients(*image, file) || !planes.has_value() ||
              planes->planes().size() != file.coefficients.size()) {
            mismatches.fetch_add(1, std::memory_order_relaxed);
          }
        }
      });
    }
  }
  CHECK(mismatches.load() == 0);
}

// A well-formed ICC profile of version 2, a monitor's, of the data color space
// given: its header, one tag, and the tag's data.
std::vector<std::uint8_t> profile_of(std::size_t size, std::string_view space) {
  std::vector<std::uint8_t> profile(size, 0);
  const auto put = [&profile](std::size_t at, std::uint32_t value) {
    for (std::size_t i = 0; i < 4; ++i) profile[at + i] = static_cast<std::uint8_t>(value >> (24 - 8 * i));
  };
  const auto text = [&profile](std::size_t at, std::string_view four) {
    for (std::size_t i = 0; i < 4; ++i) profile[at + i] = static_cast<std::uint8_t>(four[i]);
  };
  put(0, static_cast<std::uint32_t>(size));
  profile[8] = 2;
  text(12, "mntr");
  text(16, space);
  text(20, "XYZ ");
  text(36, "acsp");
  put(68, 0x0000F6D6U);  // the illuminant D50
  put(72, 0x00010000U);
  put(76, 0x0000D32DU);
  put(128, 1);
  text(132, "desc");
  put(136, 144);
  put(140, static_cast<std::uint32_t>(size - 144));
  for (std::size_t i = 144; i < size; ++i) profile[i] = static_cast<std::uint8_t>(i * 13U);
  return profile;
}

// The file read back by libpng; checks that it is the picture, and returns its
// profile.
template <typename Sample>
std::vector<std::uint8_t> read_back(const jpegio::PngFile& file, std::uint32_t width, std::uint32_t height,
                                    std::int32_t channels, std::span<const Sample> samples) {
  test_png back{};
  std::array<char, 256> message{};
  std::vector<std::uint8_t> profile;
  if (!CHECK(test_png_read(file.bytes.data(), file.bytes.size(), &back, message.data(), message.size()) == 0)) {
    std::println(stderr, "  {}", message.data());
    return profile;
  }
  CHECK(back.width == width && back.height == height && back.channels == channels);
  CHECK(back.bits == static_cast<std::int32_t>(8 * sizeof(Sample)));
  const std::span<const Sample> read{static_cast<const Sample*>(back.samples), samples.size()};
  CHECK(std::ranges::equal(read, samples));
  CHECK(back.warning[0] == '\0');
  if (back.icc_profile != nullptr) {
    profile.assign(back.icc_profile, back.icc_profile + back.icc_profile_size);
  }
  test_png_free(&back);
  return profile;
}

void test_png_files(void) {
  constexpr std::size_t pixels = std::size_t{7} * 5;
  std::vector<std::uint8_t> eight(pixels * 3);
  std::vector<std::uint16_t> sixteen(pixels * 3);
  for (std::size_t i = 0; i < eight.size(); ++i) {
    eight[i] = static_cast<std::uint8_t>(i * 37U);
    sixteen[i] = static_cast<std::uint16_t>(i * 4099U);
  }
  for (const std::int32_t channels : {1, 3}) {
    const std::size_t count = pixels * static_cast<std::size_t>(channels);
    const auto small = std::span<const std::uint8_t>{eight}.first(count);
    const auto large = std::span<const std::uint16_t>{sixteen}.first(count);
    const auto file8 = jpegio::write_png(7, 5, channels, small);
    if (CHECK(file8.has_value())) {
      CHECK(file8->warning.empty());
      CHECK(read_back(*file8, 7, 5, channels, small).empty());
    }
    const auto file16 = jpegio::write_png(7, 5, channels, large, {.compression = 9, .icc_profile = {}});
    if (CHECK(file16.has_value())) CHECK(read_back(*file16, 7, 5, channels, large).empty());
  }

  // The profile goes with the picture of its color space, and is left out of
  // another, with the check's reason.
  const std::vector<std::uint8_t> rgb = profile_of(500, "RGB ");
  CHECK(jpegio::check_icc(rgb, 3).has_value());
  const auto refused = jpegio::check_icc(rgb, 1);
  CHECK(!refused.has_value() && !refused.error().empty());
  const auto small = std::span<const std::uint8_t>{eight};
  const auto with = jpegio::write_png(7, 5, 3, small, {.compression = -1, .icc_profile = rgb});
  if (CHECK(with.has_value())) {
    CHECK(with->warning.empty());
    CHECK(read_back(*with, 7, 5, 3, small) == rgb);
  }
  const auto gray = small.first(pixels);
  const auto without = jpegio::write_png(7, 5, 1, gray, {.compression = -1, .icc_profile = rgb});
  if (CHECK(without.has_value())) {
    CHECK(without->warning == "the ICC profile is not written: " + refused.error());
    CHECK(read_back(*without, 7, 5, 1, gray).empty());
  }

  // Wrong pictures.
  const auto short_samples = jpegio::write_png(7, 5, 3, small.first(100));
  CHECK(!short_samples.has_value() && short_samples.error().code == jpegio::Errc::argument);
  const auto two = jpegio::write_png(5, 7, 2, small.first(70));
  CHECK(!two.has_value() && two.error().code == jpegio::Errc::argument && !two.error().message.empty());
  const auto level = jpegio::write_png(7, 5, 3, small, {.compression = 10, .icc_profile = {}});
  CHECK(!level.has_value() && level.error().code == jpegio::Errc::argument);
  const auto huge = jpegio::write_png(0xFFFFFFFFU, 0xFFFFFFFFU, 3, small);
  CHECK(!huge.has_value() && huge.error().code == jpegio::Errc::argument);
  CHECK(jpegio::libpng_version().starts_with("libpng 1.6."));
}

}  // namespace

int main(void) {
  // A test that throws -- out of memory, or a thread that cannot start -- fails
  // with what it says, rather than ending the program without a word.
  try {
    test_read();
    test_ownership();
    test_errors();
    test_planes();
    test_threads();
    test_png_files();
    std::println(stderr, "{} of {} checks failed", failures.load(), checks.load());
  } catch (const std::exception& error) {
    (void)std::fputs(error.what(), stderr);
    (void)std::fputc('\n', stderr);
    return EXIT_FAILURE;
  }
  return failures.load() == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
