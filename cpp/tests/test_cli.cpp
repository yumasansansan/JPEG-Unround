// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Tests of the command line (docs/cli.md), run in the process on files of the test
// encoder: TIFF, PNG and PNM files that are the library's result turned upright as the
// EXIF orientation says, to the last bit; the orientation kept; the Y, Cb and Cr of
// --ycbcr; the name of the output next to the input; a file that exists, written over
// only with --overwrite; a PNG file with alpha, refused as PNM; the report; the
// verbose records; an ICC profile left out with a warning; a file cut short, with the
// warning of its reading; --version and --help; the exit statuses of what is refused
// and of what is not read; and with --project-only, the library's projection of the
// picture given, taken as turned as the result is, its excess said and reported, and
// the pictures and options that a projection does not take.

#include "unround/cli.hpp"

#include "unround/arrays.hpp"
#include "unround/decode.hpp"
#include "unround/jpegio.hpp"
#include "unround/output.hpp"
#include "unround/picture.hpp"
#include "unround/settings.hpp"

#include "support/check.hpp"
#include "test_jpeg.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <memory>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;

// An EXIF APP1 payload with the orientation 6, and bytes that are no ICC profile.
constexpr std::array<std::uint8_t, 32> exif = {'E',  'x',  'i', 'f', 0, 0, 'I', 'I', 42, 0, 8, 0, 0, 0, 1, 0,
                                               0x12, 0x01, 3,   0,   1, 0, 0,   0,   6,  0, 0, 0, 0, 0, 0, 0};
constexpr std::array<std::uint8_t, 12> not_a_profile = {'n', 'o', 't', ' ', 'a', ' ', 'p', 'r', 'o', 'f', 'i', 'l'};

std::vector<std::uint8_t> jpeg_of(std::uint32_t width, std::uint32_t height, std::int32_t color_space,
                                  std::int32_t luma_factor, bool tagged) {
  test_jpeg spec{};
  spec.width = width;
  spec.height = height;
  spec.color_space = color_space;
  spec.h_samp_factor[0] = luma_factor;
  spec.v_samp_factor[0] = luma_factor;
  spec.quant_table_slot[1] = 1;
  spec.quant_table_slot[2] = 1;
  if (tagged) {
    spec.app1 = exif.data();
    spec.app1_size = static_cast<std::uint32_t>(exif.size());
    spec.icc_profile = not_a_profile.data();
    spec.icc_profile_size = static_cast<std::uint32_t>(not_a_profile.size());
  }
  for (std::size_t slot = 0; slot < 2; ++slot) {
    for (std::size_t k = 0; k < 64; ++k) spec.quant_tables[slot][k] = static_cast<std::uint16_t>(3 + (7 * k) % 30);
  }
  const std::int32_t components = test_jpeg_components(&spec);
  std::vector<std::vector<std::int16_t>> coefficients(static_cast<std::size_t>(components));
  unround::test::Numbers numbers(81);
  for (std::int32_t ci = 0; ci < components; ++ci) {
    const std::size_t count = static_cast<std::size_t>(test_jpeg_blocks_wide(&spec, ci)) *
                              static_cast<std::size_t>(test_jpeg_blocks_high(&spec, ci)) * 64;
    std::vector<std::int16_t>& levels = coefficients[static_cast<std::size_t>(ci)];
    levels.resize(count);
    for (std::size_t i = 0; i < count; ++i) {
      if (i % 64 == 0) {
        levels[i] = static_cast<std::int16_t>(numbers.between(-30, 30));
      } else if (i % 64 < 8 && numbers.between(0, 2) == 0) {
        levels[i] = static_cast<std::int16_t>(numbers.between(-5, 5));
      }
    }
    spec.coefficients[ci] = levels.data();
  }
  std::uint8_t* data = nullptr;
  std::size_t size = 0;
  char message[256];
  if (test_jpeg_write(&spec, &data, &size, message, sizeof message) != 0) {
    std::println(stderr, "the test encoder failed: {}", message);
    std::abort();
  }
  std::vector<std::uint8_t> file(data, data + size);
  std::free(data);
  return file;
}

struct CloseFile {
  void operator()(std::FILE* file) const noexcept { (void)std::fclose(file); }
};
using File = std::unique_ptr<std::FILE, CloseFile>;

void write_file(const fs::path& path, std::span<const std::uint8_t> bytes) {
  const File file{std::fopen(path.string().c_str(), "wb")};
  if (file == nullptr || std::fwrite(bytes.data(), 1, bytes.size(), file.get()) != bytes.size()) std::abort();
}

std::vector<std::uint8_t> read_file(const fs::path& path) {
  std::ifstream file(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

// A run of the command line, and what it wrote on its two streams.
struct Ran {
  int status = 0;
  std::string out;
  std::string err;
};

std::string text_of(std::FILE* stream) {
  if (std::fseek(stream, 0, SEEK_SET) != 0) std::abort();
  std::string text;
  std::array<char, 4096> buffer{};
  std::size_t count = 0;
  do {
    count = std::fread(buffer.data(), 1, buffer.size(), stream);
    text.append(buffer.data(), count);
  } while (count == buffer.size());
  return text;
}

using Arguments = std::vector<std::string_view>;

// The command line, its two streams files of the test's directory (tmpfile() is not
// taken: on Windows it makes its file in the root of the drive).
class Runner {
 public:
  explicit Runner(fs::path directory) : directory_(std::move(directory)) {}

  Ran operator()(const Arguments& arguments) const {
    const File out{std::fopen((directory_ / "out.txt").string().c_str(), "w+b")};
    const File err{std::fopen((directory_ / "err.txt").string().c_str(), "w+b")};
    if (out == nullptr || err == nullptr) std::abort();
    Ran ran;
    ran.status = unround::cli::run(arguments, out.get(), err.get());
    ran.out = text_of(out.get());
    ran.err = text_of(err.get());
    return ran;
  }

 private:
  fs::path directory_;
};

unround::Decoded decoded_of(std::span<const std::uint8_t> file, const Arguments& arguments) {
  auto settings = unround::parse_settings(arguments);
  if (!settings) std::abort();
  auto decoded = unround::decode(file, *settings);
  if (!decoded) std::abort();
  return std::move(*decoded);
}

bool contains(std::string_view text, std::string_view part) { return text.find(part) != std::string_view::npos; }

// Whether every line of a text is an error of the command line.
bool only_errors(std::string_view text) {
  if (text.empty()) return false;
  while (!text.empty()) {
    if (!text.starts_with("unround: ")) return false;
    const std::size_t end = text.find('\n');
    text.remove_prefix(end == std::string_view::npos ? text.size() : end + 1u);
  }
  return true;
}

void outputs(const fs::path& directory) {
  const Runner run(directory);
  const std::vector<std::uint8_t> colour = jpeg_of(21, 13, TEST_JPEG_YCBCR, 2, true);
  const fs::path input = directory / "colour.jpg";
  write_file(input, colour);
  const unround::Decoded decoded = decoded_of(colour, {"--method", "mmse"});
  CHECK(decoded.exif_orientation == 6);
  const unround::output::PictureArray upright =
      unround::output::oriented(std::as_const(decoded.picture).view(), decoded.exif_orientation);
  CHECK(upright.extent(0) == 21u && upright.extent(1) == 13u);
  const std::string in = input.string();
  // TIFF: the result upright, to the last bit; its profile is not one, and is left out.
  {
    const std::string target = (directory / "out.tif").string();
    const Ran ran = run({"--method", "mmse", in, target});
    CHECK(ran.status == 0 && contains(ran.err, "the ICC profile is not written"));
    const std::vector<std::uint8_t> expected =
        unround::output::tiff_bytes(upright.view(), unround::output::Photometric::rgb, false, {});
    CHECK(read_file(target) == expected);
  }
  // The orientation kept: the picture as stored.
  {
    const std::string target = (directory / "kept.tif").string();
    CHECK(run({"--method=mmse", "--orientation", "keep", "--no-icc", in, target}).status == 0);
    const std::vector<std::uint8_t> expected = unround::output::tiff_bytes(
        std::as_const(decoded.picture).view(), unround::output::Photometric::rgb, false, {});
    CHECK(read_file(target) == expected);
  }
  // Y, Cb and Cr: the canvas cut to the picture, upright.
  {
    const std::string target = (directory / "ycbcr.tif").string();
    CHECK(run({"--method", "mmse", "--ycbcr", "--quiet", in, target}).status == 0);
    const unround::output::PictureArray cut =
        unround::picture_of(decoded.canvas, decoded.height, decoded.width, unround::ColourSpace::rgb);
    const unround::output::PictureArray turned = unround::output::oriented(std::as_const(cut).view(), 6);
    CHECK(read_file(target) ==
          unround::output::tiff_bytes(turned.view(), unround::output::Photometric::ycbcr, false, {}));
  }
  // PNG: the dithered levels of the result.
  {
    const std::string target = (directory / "out.png").string();
    CHECK(run({"--method", "mmse", "--quiet", in, target}).status == 0);
    const auto image = unround::jpegio::PngImage::read(read_file(target));
    const unround::output::LevelsArray levels =
        unround::output::levels_of(upright.view(), 8, unround::output::Dither::blue_noise, true);
    CHECK(image.has_value() && image->bits() == 8 &&
          std::ranges::equal(std::span<const std::uint16_t>(image->levels().data_handle(), image->levels().size()),
                             levels.entries()));
  }
  // PNM of 16 bits, rounded, named next to the input.
  {
    const Ran ran = run({"--method", "mmse", "--format", "pnm", "--bits", "16", "--dither", "none", "-q", in});
    CHECK(ran.status == 0 && ran.err.empty());
    const unround::output::LevelsArray levels =
        unround::output::levels_of(upright.view(), 16, unround::output::Dither::none, true);
    CHECK(read_file(directory / "colour.pnm") == unround::output::pnm_bytes(levels.view(), 16));
    // It exists now: not written again without --overwrite.
    const Ran again = run({"--method", "mmse", "--format", "pnm", "-q", in});
    CHECK(again.status == unround::cli::not_written && contains(again.err, "exists"));
    CHECK(run({"--method", "mmse", "--format", "pnm", "-q", "--overwrite", in}).status == 0);
  }
  // The MMSE decoder solves nothing: its report has no iterations, stop or records.
  {
    const Ran ran =
        run({"--method", "mmse", "-q", "--report", "-", "--overwrite", in, (directory / "mmse.tif").string()});
    CHECK(ran.status == 0 && ran.err.empty() && contains(ran.out, "\"settings\"") && !contains(ran.out, "\"stop\"") &&
          ran.out.ends_with("]\n}\n"));
  }
  // The report, to standard output, and the records on standard error.
  {
    const Ran ran = run({"--method", "tv", "--iterations", "20", "--tolerance", "0", "--record-every", "5", "-v",
                         "--report", "-", "--overwrite", in, (directory / "tv.tif").string()});
    CHECK(ran.status == 0);
    for (const std::string_view key :
         {"\"version\": \"unround 0.1.0 (C++)\"", "\"method\": \"tv\"", "\"iterations\": 20",
          "\"stop\": \"iterations\"", "\"oriented\": true", "\"exif_orientation\": 6", "\"color_space\": \"ycbcr\"",
          "\"dither\": null", "\"icc_profile\": false", "\"--iterations=20\"", "\"partial_gap\""}) {
      if (!CHECK(contains(ran.out, key))) std::println(stderr, "not in the report: {}", key);
    }
    // Five records, at 0, 5, 10, 15 and 20, each a line of -v; the profile's warning;
    // and the summary.
    CHECK(std::ranges::count(ran.err, '\n') == 5 + 1 + 1);
  }
}

void png_input(const fs::path& directory) {
  const Runner run(directory);
  unround::MdArray<std::uint8_t, unround::PictureExtents> rgba(unround::PictureExtents{6, 7, 4});
  for (std::size_t k = 0; k < rgba.size(); ++k) rgba.entries()[k] = static_cast<std::uint8_t>((k * 37u) % 256u);
  const auto file = unround::jpegio::write_png(std::as_const(rgba).view());
  if (!CHECK(file.has_value())) return;
  const fs::path input = directory / "alpha.png";
  write_file(input, file->bytes);
  const std::string in = input.string();
  // The colours' solver and the alpha's, each summed up.
  const std::string target = (directory / "alpha.tif").string();
  const Ran tiff =
      run({"--iterations", "10", "--tolerance", "0", "--report", (directory / "alpha.json").string(), in, target});
  CHECK(tiff.status == 0 && tiff.err == std::format("{0}: 10 iterations, iterations\n"
                                                    "{0}: alpha: 10 iterations, iterations\n",
                                                    target));
  std::ifstream report(directory / "alpha.json");
  const std::string text{std::istreambuf_iterator<char>(report), std::istreambuf_iterator<char>()};
  CHECK(contains(text, "\"input_format\": \"png\"") && contains(text, "\"alpha\": true") &&
        contains(text, "\"alpha_solver\""));
  // PNM holds no alpha.
  const Ran pnm = run({"--iterations", "10", in, (directory / "alpha.pnm").string()});
  CHECK(pnm.status == unround::cli::not_written);
  // A PNG file refuses --ycbcr and the data term's other options.
  CHECK(run({"--ycbcr", in, (directory / "a.tif").string()}).status == unround::cli::refused);
  CHECK(run({"--centres", "mmse", in, (directory / "b.tif").string()}).status == unround::cli::refused);
}

void refusals(const fs::path& directory) {
  const Runner run(directory);
  const std::vector<std::uint8_t> grey = jpeg_of(30, 21, TEST_JPEG_GRAYSCALE, 1, false);
  const fs::path input = directory / "grey.jpg";
  write_file(input, grey);
  const std::string in = input.string();
  const std::string out = (directory / "r.tif").string();
  // Each refused before anything is solved (-v would show the records): --ycbcr of a
  // grey file from its header.
  const std::vector<Arguments> cases = {
      {},
      {"--unknown", in, out},
      {"--bits", "8", in, out},
      {"--compression", "5", "--format", "pnm", in, out},
      {"--ycbcr", "--format", "png", in, out},
      {"--dither", "none", "--dither-channels", "together", "--format", "png", in, out},
      {"--format", "bmp", in, out},
      {in, out, out},
      {"--alpha", "0", in, out},
      {"--format", "png", "--format", "png", in, out},
      {"--ycbcr", "-v", in, out},
      {"-x", in, out},
  };
  for (const Arguments& arguments : cases) {
    const Ran ran = run(arguments);
    if (!CHECK(ran.status == unround::cli::refused && only_errors(ran.err))) {
      std::println(stderr, "status {}: {}", ran.status, ran.err);
    }
  }
  CHECK(run({(directory / "none.jpg").string(), out}).status == unround::cli::not_read);
  const fs::path garbage = directory / "garbage.jpg";
  write_file(garbage, std::array<std::uint8_t, 5>{1, 2, 3, 4, 5});
  CHECK(run({garbage.string(), out}).status == unround::cli::not_read);
  // A file cut short: written, with the warning of its reading; not read where warnings
  // are errors.
  const fs::path cut = directory / "cut.jpg";
  write_file(cut, std::span<const std::uint8_t>(grey).first(grey.size() - 40u));
  const std::string cut_out = (directory / "cut.tif").string();
  const Ran warned = run({"--method", "mmse", cut.string(), cut_out});
  CHECK(warned.status == 0 && contains(warned.err, std::format("unround: {}: Premature end of JPEG file", cut_out)));
  CHECK(run({"--method", "mmse", "--warnings-are-errors", "--overwrite", cut.string(), cut_out}).status ==
        unround::cli::not_read);
  const Ran version = run({"--version"});
  CHECK(version.status == 0 && version.out == "unround 0.1.0 (C++)\n");
  const Ran help = run({"-h"});
  CHECK(help.status == 0 && contains(help.out, "unround [OPTIONS] INPUT [OUTPUT]"));
}

unround::Source source_of(std::span<const std::uint8_t> file, const Arguments& arguments) {
  auto settings = unround::parse_settings(arguments);
  if (!settings) std::abort();
  auto source = unround::read_source(file, *settings);
  if (!source) std::abort();
  return std::move(*source);
}

// The picture of 8-bit levels as doubles, which are the levels themselves.
unround::output::PictureArray values_of(const unround::MdArray<std::uint8_t, unround::PictureExtents>& levels) {
  unround::output::PictureArray values(levels.extents());
  std::ranges::transform(levels.entries(), values.entries().begin(),
                         [](std::uint8_t level) { return static_cast<double>(level); });
  return values;
}

// What standard error says of each component's excess begins with the counts that the
// library's summary gives.
bool says_excess(std::string_view err, std::string_view target, std::span<const std::string_view> names,
                 const unround::picture::Projection& projection) {
  bool said = true;
  for (std::size_t c = 0; c < names.size(); ++c) {
    const unround::picture::Summary summary = unround::picture::summary(projection.excess[c].view());
    const std::string named = names[c].empty() ? std::string() : std::format("{}: ", names[c]);
    const std::string line =
        std::format("{}: {}{} of {} coefficients of the picture lay outside their intervals{}", target, named,
                    summary.outside, summary.coefficients, summary.outside > 0u ? ", by at most " : "\n");
    said = said && contains(err, line);
  }
  return said;
}

void projection(const fs::path& directory) {
  const Runner run(directory);
  const std::vector<std::uint8_t> colour = jpeg_of(21, 13, TEST_JPEG_YCBCR, 2, true);
  const fs::path input = directory / "projected.jpg";
  write_file(input, colour);
  const std::string in = input.string();
  // A picture made elsewhere: the MMSE decoder's in 8 bits, as the file stores it, three
  // samples moved far.
  const unround::Decoded decoded = decoded_of(colour, {"--method", "mmse"});
  const unround::output::LevelsArray rounded =
      unround::output::levels_of(std::as_const(decoded.picture).view(), 8, unround::output::Dither::none, true);
  unround::MdArray<std::uint8_t, unround::PictureExtents> stored(rounded.extents());
  std::ranges::transform(rounded.entries(), stored.entries().begin(),
                         [](std::uint16_t level) { return static_cast<std::uint8_t>(level); });
  stored.view()[4, 6, 0] = 255;
  stored.view()[4, 6, 2] = 0;
  stored.view()[9, 15, 1] = 0;
  const unround::Source source = source_of(colour, {});
  const unround::output::PictureArray values = values_of(stored);
  const auto projected = unround::picture::project(source, values.view());
  if (!CHECK(projected.has_value())) return;
  const unround::output::PictureArray upright =
      unround::output::oriented(std::as_const(projected->decoded.picture).view(), 6);
  // Upright, as the result is: TIFF, to the last bit, with the excess said and reported.
  const unround::MdArray<std::uint8_t, unround::PictureExtents> shown =
      unround::output::oriented(std::as_const(stored).view(), 6);
  const fs::path upright_png = directory / "upright.png";
  const fs::path stored_png = directory / "stored.png";
  {
    const auto shown_file = unround::jpegio::write_png(shown.view());
    const auto stored_file = unround::jpegio::write_png(std::as_const(stored).view());
    if (!CHECK(shown_file.has_value() && stored_file.has_value())) return;
    write_file(upright_png, shown_file->bytes);
    write_file(stored_png, stored_file->bytes);
  }
  {
    const std::string target = (directory / "projected.tif").string();
    const std::string report = (directory / "projected.json").string();
    const Ran ran = run({"--project-only", upright_png.string(), "--report", report, in, target});
    CHECK(ran.status == 0);
    const std::array<std::string_view, 3> names = {"Y", "Cb", "Cr"};
    CHECK(says_excess(ran.err, target, names, *projected));
    CHECK(contains(ran.err, "the ICC profile is not written"));
    CHECK(read_file(target) ==
          unround::output::tiff_bytes(upright.view(), unround::output::Photometric::rgb, false, {}));
    std::ifstream file(report);
    const std::string text{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
    const unround::picture::Summary luma = unround::picture::summary(projected->excess[0].view());
    for (const std::string& key :
         {std::string("\"method\": \"projection\""), std::format("\"picture\": \"{}\"", upright_png.string()),
          std::string("\"oriented\": true"), std::string("\"settings\": [\"--slack=0\"]"),
          std::format("\"excess\": [{{\"coefficients\": {}, \"outside\": {}, \"largest\": {}}}", luma.coefficients,
                      luma.outside, luma.largest)}) {
      if (!CHECK(contains(text, key))) std::println(stderr, "not in the report: {}", key);
    }
    CHECK(luma.outside > 0u && !contains(text, "\"records\""));
  }
  // As stored, with --orientation keep: the Y, Cb and Cr of the projected canvas.
  {
    const std::string target = (directory / "projected-kept.tif").string();
    const Ran ran =
        run({"--project-only", stored_png.string(), "--orientation", "keep", "--ycbcr", "--quiet", in, target});
    CHECK(ran.status == 0 && ran.err.empty());
    const unround::output::PictureArray cut =
        unround::picture_of(projected->decoded.canvas, 13, 21, unround::ColourSpace::rgb);
    CHECK(read_file(target) == unround::output::tiff_bytes(cut.view(), unround::output::Photometric::ycbcr, false, {}));
  }
  // The picture stored is not the upright one, nor the upright one the stored.
  {
    const Ran stored_run = run({"--project-only", stored_png.string(), in, (directory / "x.tif").string()});
    CHECK(stored_run.status == unround::cli::not_read &&
          contains(stored_run.err, "a picture of the file is 21 x 13 pixels, not 13 x 21, as the file stores it"));
    const Ran upright_run =
        run({"--project-only", upright_png.string(), "--orientation=keep", in, (directory / "x.tif").string()});
    CHECK(upright_run.status == unround::cli::not_read &&
          contains(upright_run.err, "turned upright, which --orientation apply takes"));
  }
  // PNG of 16 bits, rounded, of the projection with a slack.
  {
    const unround::Source widened = source_of(colour, {"--slack", "0.5"});
    const auto loose = unround::picture::project(widened, values.view());
    if (!CHECK(loose.has_value())) return;
    const std::string target = (directory / "slack.png").string();
    const Ran ran = run({"--project-only", upright_png.string(), "--slack", "0.5", "--bits", "16", "--dither", "none",
                         "--quiet", in, target});
    CHECK(ran.status == 0);
    const unround::output::PictureArray turned =
        unround::output::oriented(std::as_const(loose->decoded.picture).view(), 6);
    const unround::output::LevelsArray levels =
        unround::output::levels_of(turned.view(), 16, unround::output::Dither::none, true);
    const auto image = unround::jpegio::PngImage::read(read_file(target));
    CHECK(image.has_value() && image->bits() == 16 &&
          std::ranges::equal(std::span<const std::uint16_t>(image->levels().data_handle(), image->levels().size()),
                             levels.entries()));
  }
}

void grey_projection(const fs::path& directory) {
  const Runner run(directory);
  const std::vector<std::uint8_t> grey = jpeg_of(30, 21, TEST_JPEG_GRAYSCALE, 1, false);
  const fs::path input = directory / "grey-projected.jpg";
  write_file(input, grey);
  const std::string in = input.string();
  // 16 bits, level v standing for v / 257.
  unround::MdArray<std::uint16_t, unround::PictureExtents> levels(unround::PictureExtents{21, 30, 1});
  unround::MdArray<std::uint16_t, unround::PictureExtents> rgb(unround::PictureExtents{21, 30, 3});
  unround::output::PictureArray values(unround::PictureExtents{21, 30, 1});
  for (std::size_t k = 0; k < levels.size(); ++k) {
    const auto level = static_cast<std::uint16_t>((k * 2654435761u) >> 16u);
    levels.entries()[k] = level;
    for (std::size_t c = 0; c < 3u; ++c) rgb.entries()[3u * k + c] = level;
    values.entries()[k] = static_cast<double>(level) / 257.0;
  }
  const auto projected = unround::picture::project(source_of(grey, {}), std::as_const(values).view());
  if (!CHECK(projected.has_value())) return;
  const std::vector<std::uint8_t> expected = unround::output::tiff_bytes(
      std::as_const(projected->decoded.picture).view(), unround::output::Photometric::grey, false, {});
  const auto grey_file = unround::jpegio::write_png(std::as_const(levels).view());
  const auto rgb_file = unround::jpegio::write_png(std::as_const(rgb).view());
  if (!CHECK(grey_file.has_value() && rgb_file.has_value())) return;
  const fs::path grey_png = directory / "grey16.png";
  const fs::path rgb_png = directory / "rgb16.png";
  write_file(grey_png, grey_file->bytes);
  write_file(rgb_png, rgb_file->bytes);
  for (const fs::path& picture_path : {grey_png, rgb_png}) {
    const std::string target = (directory / (picture_path.stem().string() + ".tif")).string();
    const Ran ran = run({"--project-only", picture_path.string(), in, target});
    const std::array<std::string_view, 1> names = {""};
    CHECK(ran.status == 0 && says_excess(ran.err, target, names, *projected));
    CHECK(read_file(target) == expected);
  }
  // Pictures that are not taken.
  unround::MdArray<std::uint16_t, unround::PictureExtents> unlike = rgb;
  unlike.entries()[4] ^= 1u;
  unround::MdArray<std::uint8_t, unround::PictureExtents> alpha(unround::PictureExtents{21, 30, 2});
  const unround::MdArray<std::uint8_t, unround::PictureExtents> small(unround::PictureExtents{20, 30, 1});
  const auto unlike_file = unround::jpegio::write_png(std::as_const(unlike).view());
  const auto alpha_file = unround::jpegio::write_png(std::as_const(alpha).view());
  const auto small_file = unround::jpegio::write_png(small.view());
  if (!CHECK(unlike_file.has_value() && alpha_file.has_value() && small_file.has_value())) return;
  struct Case {
    std::string_view name;
    std::span<const std::uint8_t> bytes;
    std::string_view said;
  };
  const std::array<Case, 4> cases = {
      Case{.name = "unlike.png",
           .bytes = unlike_file->bytes,
           .said = "a greyscale file's picture is grey, or RGB of three channels alike"},
      Case{.name = "alpha.png", .bytes = alpha_file->bytes, .said = "a picture with alpha is not a JPEG file's"},
      Case{.name = "small.png",
           .bytes = small_file->bytes,
           .said = "a picture of the file is 21 x 30 pixels, not 20 x 30"},
      Case{.name = "not.png", .bytes = grey, .said = "the picture to project is a PNG file"},
  };
  for (const Case& refused_case : cases) {
    const fs::path picture_path = directory / refused_case.name;
    write_file(picture_path, refused_case.bytes);
    const Ran ran = run({"--project-only", picture_path.string(), in, (directory / "x.tif").string()});
    if (!CHECK(ran.status == unround::cli::not_read && contains(ran.err, refused_case.said))) {
      std::println(stderr, "{}: status {}: {}", refused_case.name, ran.status, ran.err);
    }
  }
  // A colour file's picture is RGB.
  const fs::path colour = directory / "colour-projected.jpg";
  write_file(colour, jpeg_of(21, 13, TEST_JPEG_YCBCR, 2, false));
  const Ran colour_run = run({"--project-only", grey_png.string(), colour.string(), (directory / "x.tif").string()});
  CHECK(colour_run.status == unround::cli::not_read && contains(colour_run.err, "a colour file's picture is RGB"));
  // What a projection does not take.
  const Ran method = run({"--project-only", grey_png.string(), "--method", "tv", "--iterations", "3", in, "-"});
  CHECK(method.status == unround::cli::refused && contains(method.err, "--method is not taken with --project-only") &&
        contains(method.err, "--iterations is not taken with --project-only"));
  const Ran both = run({"--project-only", "-", "-"});
  CHECK(both.status == unround::cli::refused && contains(both.err, "not both the standard input"));
  const Ran png = run({"--project-only", grey_png.string(), grey_png.string(), (directory / "x.tif").string()});
  CHECK(png.status == unround::cli::refused && contains(png.err, "not of a PNG file"));
}

int run_tests(std::span<char* const> arguments) {
  if (arguments.size() < 2u) return 2;
  const fs::path directory = arguments[1];
  std::error_code code;
  fs::remove_all(directory, code);
  fs::create_directories(directory);
  outputs(directory);
  png_input(directory);
  refusals(directory);
  projection(directory);
  grey_projection(directory);
  return unround::test::finish("cli");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run_tests(std::span<char* const>{argv, static_cast<std::size_t>(argc)});
  } catch (const std::exception& error) {
    (void)std::fputs(error.what(), stderr);
    (void)std::fputc('\n', stderr);
    return EXIT_FAILURE;
  }
}
