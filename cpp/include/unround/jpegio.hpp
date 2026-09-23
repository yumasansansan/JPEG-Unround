// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The C layer (unround/jpegio.h) as C++ sees it: what a read returns owns its
// memory and frees it when it goes, and a failure comes back as an Error in a
// std::expected rather than as a status and a buffer. The structures stay the
// C layer's own; spans give their arrays a length.
#ifndef UNROUND_JPEGIO_HPP
#define UNROUND_JPEGIO_HPP

#include "unround/jpegio.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <utility>

namespace unround::jpegio {

enum class Errc : std::int32_t {
  argument = UNROUND_JPEGIO_ERROR_ARGUMENT,
  decode = UNROUND_JPEGIO_ERROR_DECODE,
  unsupported = UNROUND_JPEGIO_ERROR_UNSUPPORTED,
  limit = UNROUND_JPEGIO_ERROR_LIMIT,
  memory = UNROUND_JPEGIO_ERROR_MEMORY,
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

}  // namespace unround::jpegio

#endif
