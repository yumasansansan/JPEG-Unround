// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The C layer (unround/jpegio.h) as C++ sees it: what a read returns owns its
// memory and frees it when it goes, a PNG file comes back as its bytes, and a
// failure comes back as an Error in a std::expected rather than as a status and
// a buffer. The structures stay the C layer's own; spans give their arrays a
// length.
#ifndef UNROUND_JPEGIO_HPP
#define UNROUND_JPEGIO_HPP

#include "unround/jpegio.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace unround::jpegio {

enum class Errc : std::int32_t {
  argument = UNROUND_JPEGIO_ERROR_ARGUMENT,
  decode = UNROUND_JPEGIO_ERROR_DECODE,
  unsupported = UNROUND_JPEGIO_ERROR_UNSUPPORTED,
  limit = UNROUND_JPEGIO_ERROR_LIMIT,
  memory = UNROUND_JPEGIO_ERROR_MEMORY,
  encode = UNROUND_JPEGIO_ERROR_ENCODE,
};

struct Error {
  Errc code;
  std::string message;
};

enum class ColorSpace : std::int32_t {
  grayscale = UNROUND_JPEGIO_GRAYSCALE,
  ycbcr = UNROUND_JPEGIO_YCBCR,
  rgb = UNROUND_JPEGIO_RGB,
};

// The limits of unround_jpegio_options; zero means the C layer's default.
struct Options {
  std::uint64_t max_pixels = 0;
  std::int32_t max_scans = 0;
  bool warnings_are_errors = false;
};

namespace detail {

inline constexpr std::size_t message_capacity = 256;
using Message = std::array<char, message_capacity>;

[[nodiscard]] inline unround_jpegio_options to_c(const Options& options) noexcept {
  return unround_jpegio_options{.max_pixels = options.max_pixels,
                                .max_scans = options.max_scans,
                                .warnings_are_errors = options.warnings_are_errors ? 1 : 0};
}

[[nodiscard]] inline std::string to_string(const Message& message) { return std::string{message.data()}; }

}  // namespace detail

// The coefficients of a component: height_in_blocks * width_in_blocks blocks of
// 64, natural order.
[[nodiscard]] inline std::span<const std::int16_t> coefficients(const unround_jpegio_component& component) noexcept {
  const std::size_t blocks =
      static_cast<std::size_t>(component.width_in_blocks) * static_cast<std::size_t>(component.height_in_blocks);
  if (component.coefficients == nullptr) return {};
  return {component.coefficients, blocks * UNROUND_JPEGIO_BLOCK_SIZE};
}

// The 64 coefficients of one block of a component.
[[nodiscard]] inline std::span<const std::int16_t, UNROUND_JPEGIO_BLOCK_SIZE> block(
    const unround_jpegio_component& component, std::uint32_t block_x, std::uint32_t block_y) noexcept {
  const std::size_t index = static_cast<std::size_t>(block_y) * component.width_in_blocks + block_x;
  return coefficients(component).subspan(index * UNROUND_JPEGIO_BLOCK_SIZE).first<UNROUND_JPEGIO_BLOCK_SIZE>();
}

[[nodiscard]] inline std::span<const std::uint16_t, UNROUND_JPEGIO_BLOCK_SIZE> quant_table(
    const unround_jpegio_component& component) noexcept {
  return std::span<const std::uint16_t, UNROUND_JPEGIO_BLOCK_SIZE>{component.quant_table};
}

// The samples of a plane, stride * height of them.
[[nodiscard]] inline std::span<const std::uint8_t> samples(const unround_jpegio_plane& plane) noexcept {
  if (plane.samples == nullptr) return {};
  return {plane.samples, static_cast<std::size_t>(plane.stride) * static_cast<std::size_t>(plane.height)};
}

// What unround_jpegio_read() returns.
class Image {
 public:
  Image(void) noexcept = default;
  Image(const Image&) = delete;
  Image& operator=(const Image&) = delete;
  Image(Image&& other) noexcept
      : image_{std::exchange(other.image_, unround_jpegio_image{})}, warning_{std::move(other.warning_)} {}
  Image& operator=(Image&& other) noexcept {
    if (this != &other) {
      unround_jpegio_image_free(&image_);
      image_ = std::exchange(other.image_, unround_jpegio_image{});
      warning_ = std::move(other.warning_);
    }
    return *this;
  }
  ~Image(void) { unround_jpegio_image_free(&image_); }

  [[nodiscard]] static std::expected<Image, Error> read(std::span<const std::uint8_t> data,
                                                        const Options& options = {}) {
    Image image;
    const unround_jpegio_options c_options = detail::to_c(options);
    detail::Message message{};
    const unround_jpegio_status status =
        unround_jpegio_read(data.data(), data.size(), &c_options, &image.image_, message.data(), message.size());
    if (status != UNROUND_JPEGIO_OK) {
      return std::unexpected{Error{.code = static_cast<Errc>(status), .message = detail::to_string(message)}};
    }
    image.warning_ = detail::to_string(message);
    return image;
  }

  [[nodiscard]] std::uint32_t width(void) const noexcept { return image_.width; }
  [[nodiscard]] std::uint32_t height(void) const noexcept { return image_.height; }
  [[nodiscard]] ColorSpace color_space(void) const noexcept { return static_cast<ColorSpace>(image_.color_space); }
  [[nodiscard]] bool progressive(void) const noexcept { return image_.progressive != 0; }
  [[nodiscard]] bool arithmetic(void) const noexcept { return image_.arithmetic != 0; }
  [[nodiscard]] std::int32_t exif_orientation(void) const noexcept { return image_.exif_orientation; }
  [[nodiscard]] std::int32_t warnings(void) const noexcept { return image_.warnings; }
  // The first corrupt-data warning, or an empty string.
  [[nodiscard]] const std::string& warning(void) const noexcept { return warning_; }

  [[nodiscard]] std::span<const unround_jpegio_component> components(void) const noexcept {
    return std::span<const unround_jpegio_component>{image_.components}.first(
        static_cast<std::size_t>(image_.num_components));
  }

  [[nodiscard]] std::span<const std::uint8_t> icc_profile(void) const noexcept {
    if (image_.icc_profile == nullptr) return {};
    return {image_.icc_profile, static_cast<std::size_t>(image_.icc_profile_size)};
  }

  [[nodiscard]] const unround_jpegio_image& raw(void) const noexcept { return image_; }

 private:
  unround_jpegio_image image_{};
  std::string warning_;
};

// What unround_jpegio_decode_planes() returns.
class Planes {
 public:
  Planes(void) noexcept = default;
  Planes(const Planes&) = delete;
  Planes& operator=(const Planes&) = delete;
  Planes(Planes&& other) noexcept
      : planes_{std::exchange(other.planes_, unround_jpegio_planes{})}, warning_{std::move(other.warning_)} {}
  Planes& operator=(Planes&& other) noexcept {
    if (this != &other) {
      unround_jpegio_planes_free(&planes_);
      planes_ = std::exchange(other.planes_, unround_jpegio_planes{});
      warning_ = std::move(other.warning_);
    }
    return *this;
  }
  ~Planes(void) { unround_jpegio_planes_free(&planes_); }

  [[nodiscard]] static std::expected<Planes, Error> decode(std::span<const std::uint8_t> data,
                                                           const Options& options = {}) {
    Planes planes;
    const unround_jpegio_options c_options = detail::to_c(options);
    detail::Message message{};
    const unround_jpegio_status status = unround_jpegio_decode_planes(data.data(), data.size(), &c_options,
                                                                      &planes.planes_, message.data(), message.size());
    if (status != UNROUND_JPEGIO_OK) {
      return std::unexpected{Error{.code = static_cast<Errc>(status), .message = detail::to_string(message)}};
    }
    planes.warning_ = detail::to_string(message);
    return planes;
  }

  [[nodiscard]] std::span<const unround_jpegio_plane> planes(void) const noexcept {
    return std::span<const unround_jpegio_plane>{planes_.planes}.first(
        static_cast<std::size_t>(planes_.num_components));
  }

  [[nodiscard]] const std::string& warning(void) const noexcept { return warning_; }

 private:
  unround_jpegio_planes planes_{};
  std::string warning_;
};

// The versions of the libpng and the zlib the C layer is built with.
[[nodiscard]] inline std::string_view libpng_version(void) noexcept { return unround_jpegio_libpng_version(); }

// Whether an ICC profile goes with a picture of the channels (1: gray, 3: RGB),
// as libpng takes the profile of an iCCP chunk; why not, where it does not.
[[nodiscard]] inline std::expected<void, std::string> check_icc(std::span<const std::uint8_t> profile,
                                                                std::int32_t channels) {
  detail::Message reason{};
  if (unround_jpegio_check_icc(profile.data(), profile.size(), channels, reason.data(), reason.size()) == 1) return {};
  return std::unexpected{detail::to_string(reason)};
}

// A PNG file that the C layer wrote, and the first warning of its writing (such
// as an ICC profile that is not written, and why), or an empty string.
struct PngFile {
  std::vector<std::uint8_t> bytes;
  std::string warning;
};

// How a PNG file is written: zlib's level for the samples and the ICC profile, 0
// to 9, or -1 for zlib's default; and an ICC profile for the iCCP chunk, none
// where empty (see unround_jpegio_png).
struct PngOptions {
  std::int32_t compression = -1;
  std::span<const std::uint8_t> icc_profile;
};

namespace detail {

// Whether a * b * c is the count, without overflowing.
[[nodiscard]] inline bool product_is(std::uint64_t a, std::uint64_t b, std::uint64_t c, std::size_t count) noexcept {
  constexpr std::uint64_t most = std::numeric_limits<std::uint64_t>::max();
  if (a != 0 && b > most / a) return false;
  const std::uint64_t ab = a * b;
  if (ab != 0 && c > most / ab) return false;
  return ab * c == static_cast<std::uint64_t>(count);
}

// The bytes a writing allocated, freed when this goes.
class WrittenBytes {
 public:
  WrittenBytes(void) noexcept = default;
  WrittenBytes(const WrittenBytes&) = delete;
  WrittenBytes& operator=(const WrittenBytes&) = delete;
  WrittenBytes(WrittenBytes&&) = delete;
  WrittenBytes& operator=(WrittenBytes&&) = delete;
  ~WrittenBytes(void) { unround_jpegio_bytes_free(&bytes_); }

  [[nodiscard]] unround_jpegio_bytes* get(void) noexcept { return &bytes_; }
  [[nodiscard]] std::span<const std::uint8_t> view(void) const noexcept {
    if (bytes_.data == nullptr) return {};
    return {bytes_.data, static_cast<std::size_t>(bytes_.size)};
  }

 private:
  unround_jpegio_bytes bytes_{};
};

[[nodiscard]] inline std::expected<PngFile, Error> write_png(std::uint32_t width, std::uint32_t height,
                                                             std::int32_t channels, std::int32_t bits,
                                                             const void* samples, std::size_t count,
                                                             const PngOptions& options) {
  if (channels < 0 || !product_is(width, height, static_cast<std::uint64_t>(channels), count)) {
    return std::unexpected{
        Error{.code = Errc::argument, .message = "the samples are not height * width * channels of them"}};
  }
  const unround_jpegio_png png{
      .width = width,
      .height = height,
      .channels = channels,
      .bits = bits,
      .compression = options.compression,
      .reserved = 0,
      .samples = samples,
      .icc_profile = options.icc_profile.empty() ? nullptr : options.icc_profile.data(),
      .icc_profile_size = options.icc_profile.size(),
  };
  WrittenBytes file;
  Message message{};
  const unround_jpegio_status status = unround_jpegio_write_png(&png, file.get(), message.data(), message.size());
  if (status != UNROUND_JPEGIO_OK) {
    return std::unexpected{Error{.code = static_cast<Errc>(status), .message = to_string(message)}};
  }
  const std::span<const std::uint8_t> bytes = file.view();
  return PngFile{.bytes = std::vector<std::uint8_t>(bytes.begin(), bytes.end()), .warning = to_string(message)};
}

}  // namespace detail

// Writes a picture as a PNG file: height * width pixels of 1 (gray) or 3 (RGB)
// samples of 8 bits, rows from the top, the samples of a pixel together.
[[nodiscard]] inline std::expected<PngFile, Error> write_png(std::uint32_t width, std::uint32_t height,
                                                             std::int32_t channels,
                                                             std::span<const std::uint8_t> samples,
                                                             const PngOptions& options = {}) {
  return detail::write_png(width, height, channels, 8, samples.data(), samples.size(), options);
}

// The same with samples of 16 bits.
[[nodiscard]] inline std::expected<PngFile, Error> write_png(std::uint32_t width, std::uint32_t height,
                                                             std::int32_t channels,
                                                             std::span<const std::uint16_t> samples,
                                                             const PngOptions& options = {}) {
  return detail::write_png(width, height, channels, 16, samples.data(), samples.size(), options);
}

}  // namespace unround::jpegio

#endif
