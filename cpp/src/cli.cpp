// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The command line of docs/cli.md: the arguments read, the file reconstructed (a JPEG
// file by decode, a PNG file by samples::reconstruct) or a picture made elsewhere
// projected onto a JPEG file's intervals (picture::project), turned upright, written as
// TIFF, PNG or PNM, and what was done said and reported.

#include "unround/cli.hpp"

#include "unround/arrays.hpp"
#include "unround/decode.hpp"
#include "unround/error.hpp"
#include "unround/frames.hpp"
#include "unround/jpegio.hpp"
#include "unround/output.hpp"
#include "unround/pdhg.hpp"
#include "unround/picture.hpp"
#include "unround/results.hpp"
#include "unround/samples.hpp"
#include "unround/settings.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <expected>
#include <filesystem>
#include <format>
#include <memory>
#include <optional>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

namespace unround::cli {

namespace {

constexpr std::string_view version_text = "unround 0.1.0 (C++)";

constexpr std::string_view help_text = R"(unround [OPTIONS] INPUT [OUTPUT]
unround [OPTIONS] --project-only PICTURE INPUT [OUTPUT]

Reconstructs the JPEG file INPUT within its quantization intervals (a PNG file within
the intervals of its rounded samples) and writes the result to OUTPUT: TIFF of binary64
samples by default, PNG or PNM of 8 or 16 bits. With --project-only, projects the PNG
file PICTURE, a picture of the JPEG file INPUT made elsewhere, onto INPUT's intervals
instead. '-' is standard input or output.

Input and output:
  --format tiff|png|pnm     the format (default: OUTPUT's extension, else TIFF)
  --overwrite               write over an OUTPUT that exists
  --bits 8|16               PNG and PNM: the bits of a sample (default 8)
  --compression N           PNG: zlib's level, 0 to 9 (default 6)
  --ycbcr                   TIFF of a file in YCbCr: Y, Cb and Cr instead of RGB
  --dither blue-noise|none  PNG and PNM: the ordered dither, or rounding
  --dither-channels apart|together
  --orientation apply|keep  turn the result upright as the EXIF orientation says
  --no-icc                  write no ICC profile
  --max-pixels N  --max-scans N  --warnings-are-errors

Projection:
  --project-only PICTURE    with --slack and the options of reading alone

The method and the model:
  --method mmse|tv|tgv|subgradient  --alpha A  --alpha1 A  --alpha0 A
  --channel-weights G,...  --channels coupled|apart
  --mu M|rule  --mu-scale S  --mu-power P  --mu-chroma C  --weight-power P
  --dc-weight W  --centres mmse|midpoint  --slack S  --slack-cost C
  --flat-weight F  --flat-levels N  --flat-dc D|any

The primal-dual method:
  --iterations N  --tolerance T  --relative-tolerance T  --partial-tolerance T
  --partial-radius R  --step-ratio K  --relaxation R  --step-product P
  --norm-squared L2  --no-weight-scaling  --record-every N  --free-radius R
  --threads N

The subgradient method:
  --subgradient-iterations N  --subgradient-step S  --subgradient-decay D
  --no-momentum  --subgradient-record-every N

Reports:
  -v, --verbose  -q, --quiet  --report PATH|-  --version  -h, --help

docs/cli.md says what each does.
)";

enum class Format : std::int32_t { tiff, png, pnm };

// What the command line's own options say.
struct Command {
  std::vector<std::string> positional;
  std::optional<Format> format;
  bool overwrite = false;
  std::optional<std::int32_t> bits;
  std::optional<std::int32_t> compression;
  bool ycbcr = false;
  std::optional<output::Dither> dither;
  std::optional<bool> together;
  bool keep = false;
  bool no_icc = false;
  bool verbose = false;
  bool quiet = false;
  std::optional<std::string> report;
  // The picture made elsewhere that --project-only projects, where it is given.
  std::optional<std::string> picture;
  bool version = false;
  bool help = false;
  // The settings' options, each --name or --name=value.
  std::vector<std::string> settings;
  std::vector<std::string> errors;
  std::vector<std::string> given;
};

constexpr std::array<std::string_view, 8> own_valued = {
    "format", "bits", "compression", "dither", "dither-channels", "orientation", "report", "project-only"};
constexpr std::array<std::string_view, 7> own_flags = {"overwrite", "ycbcr",   "no-icc", "verbose",
                                                       "quiet",     "version", "help"};

std::optional<std::int64_t> integer_of(std::string_view text) noexcept {
  std::int64_t value = 0;
  const char* const first = text.data();
  const char* const last = first + text.size();
  const auto [stop, error] = std::from_chars(first, last, value);
  if (error != std::errc{} || stop != last || text.empty()) return std::nullopt;
  return value;
}

void set_value(Command& command, std::string_view name, std::string_view value) {
  const auto refuse = [&command, name, value](std::string_view range) {
    command.errors.push_back(std::format("--{} is {}, not {}", name, range, value));
  };
  if (name == "format") {
    if (value == "tiff") {
      command.format = Format::tiff;
    } else if (value == "png") {
      command.format = Format::png;
    } else if (value == "pnm") {
      command.format = Format::pnm;
    } else {
      refuse("tiff, png or pnm");
    }
  } else if (name == "bits") {
    if (value == "8" || value == "16") {
      command.bits = value == "8" ? 8 : 16;
    } else {
      refuse("8 or 16");
    }
  } else if (name == "compression") {
    const std::optional<std::int64_t> level = integer_of(value);
    if (level && *level >= 0 && *level <= 9) {
      command.compression = static_cast<std::int32_t>(*level);
    } else {
      refuse("an integer from 0 to 9");
    }
  } else if (name == "dither") {
    if (value == "blue-noise" || value == "none") {
      command.dither = value == "none" ? output::Dither::none : output::Dither::blue_noise;
    } else {
      refuse("blue-noise or none");
    }
  } else if (name == "dither-channels") {
    if (value == "apart" || value == "together") {
      command.together = value == "together";
    } else {
      refuse("apart or together");
    }
  } else if (name == "orientation") {
    if (value == "apply" || value == "keep") {
      command.keep = value == "keep";
    } else {
      refuse("apply or keep");
    }
  } else if (name == "report") {
    command.report = std::string(value);
  } else if (name == "project-only") {
    command.picture = std::string(value);
  }
}

void set_flag(Command& command, std::string_view name) {
  if (name == "overwrite") command.overwrite = true;
  if (name == "ycbcr") command.ycbcr = true;
  if (name == "no-icc") command.no_icc = true;
  if (name == "verbose") command.verbose = true;
  if (name == "quiet") command.quiet = true;
  if (name == "version") command.version = true;
  if (name == "help") command.help = true;
}

Command command_of(std::span<const std::string_view> arguments) {
  Command command;
  const auto once = [&command](std::string_view name) {
    if (std::ranges::find(command.given, name) != command.given.end()) {
      command.errors.push_back(std::format("--{} is given twice", name));
      return false;
    }
    command.given.emplace_back(name);
    return true;
  };
  for (std::size_t i = 0; i < arguments.size(); ++i) {
    const std::string_view argument = arguments[i];
    if (argument == "-" || !argument.starts_with('-')) {
      command.positional.emplace_back(argument);
      continue;
    }
    std::string_view name;
    if (argument == "-v" || argument == "-q" || argument == "-h") {
      name = argument == "-v" ? "verbose" : argument == "-q" ? "quiet" : "help";
    } else if (!argument.starts_with("--")) {
      command.errors.push_back(std::format("{} is not an option", argument));
      continue;
    } else {
      const std::size_t equals = argument.find('=');
      name = argument.substr(2, equals == std::string_view::npos ? std::string_view::npos : equals - 2u);
      const bool has_value = equals != std::string_view::npos;
      if (std::ranges::find(own_valued, name) != own_valued.end()) {
        std::string_view value;
        if (has_value) {
          value = argument.substr(equals + 1u);
        } else if (i + 1u < arguments.size()) {
          value = arguments[++i];
        } else {
          command.errors.push_back(std::format("--{} needs a value", name));
          continue;
        }
        if (once(name)) set_value(command, name, value);
        continue;
      }
      if (std::ranges::find(own_flags, name) == own_flags.end()) {
        // An option of the settings, with its value where it takes one; or one that is
        // not an option, which the settings refuse.
        if (!has_value && setting_takes_value(name) && i + 1u < arguments.size()) {
          command.settings.push_back(std::format("{}={}", argument, arguments[++i]));
        } else {
          command.settings.emplace_back(argument);
        }
        continue;
      }
      if (has_value) {
        command.errors.push_back(std::format("--{} takes no value", name));
        continue;
      }
    }
    if (once(name)) set_flag(command, name);
  }
  return command;
}

std::string extension_of(Format format) {
  switch (format) {
    case Format::tiff:
      return ".tif";
    case Format::png:
      return ".png";
    case Format::pnm:
      return ".pnm";
  }
  return ".tif";
}

std::optional<Format> format_of_name(std::string_view path) {
  const std::string extension = std::filesystem::path(std::string(path)).extension().string();
  std::string lower;
  for (const char c : extension) lower.push_back(c >= 'A' && c <= 'Z' ? static_cast<char>(c - 'A' + 'a') : c);
  if (lower == ".tif" || lower == ".tiff") return Format::tiff;
  if (lower == ".png") return Format::png;
  if (lower == ".pgm" || lower == ".ppm" || lower == ".pnm") return Format::pnm;
  return std::nullopt;
}

// Closes a file that this program opened.
struct CloseFile {
  void operator()(std::FILE* file) const noexcept { (void)std::fclose(file); }
};

// The bytes of a stream, to its end; none where it fails.
std::optional<std::vector<std::uint8_t>> bytes_of(std::FILE* stream) {
  std::vector<std::uint8_t> bytes;
  std::array<std::uint8_t, 65536> buffer{};
  while (true) {
    const std::size_t count = std::fread(buffer.data(), 1, buffer.size(), stream);
    bytes.insert(bytes.end(), buffer.begin(), buffer.begin() + static_cast<std::ptrdiff_t>(count));
    if (count < buffer.size()) break;
  }
  if (std::ferror(stream) != 0) return std::nullopt;
  return bytes;
}

// Files are opened by their names in UTF-8, which fopen takes on every system this
// builds for: on Windows, through the manifest that makes UTF-8 the program's code page.
std::expected<std::vector<std::uint8_t>, std::string> read_all(const std::string& path) {
  if (path == "-") {
#ifdef _WIN32
    (void)_setmode(_fileno(stdin), _O_BINARY);
#endif
    std::optional<std::vector<std::uint8_t>> bytes = bytes_of(stdin);
    if (!bytes) return std::unexpected(std::string("standard input could not be read"));
    return std::move(*bytes);
  }
  const std::unique_ptr<std::FILE, CloseFile> file{std::fopen(path.c_str(), "rb")};
  if (file == nullptr) return std::unexpected(std::format("{}: the file could not be opened", path));
  std::optional<std::vector<std::uint8_t>> bytes = bytes_of(file.get());
  if (!bytes) return std::unexpected(std::format("{}: the file could not be read", path));
  return std::move(*bytes);
}

std::expected<void, std::string> write_all(const std::string& path, std::span<const std::byte> bytes, bool overwrite,
                                           std::FILE* out) {
  if (path == "-") {
#ifdef _WIN32
    (void)_setmode(_fileno(out), _O_BINARY);
#endif
    if (std::fwrite(bytes.data(), 1, bytes.size(), out) != bytes.size() || std::fflush(out) != 0) {
      return std::unexpected(std::string("standard output could not be written"));
    }
    return {};
  }
  std::error_code code;
  if (!overwrite && std::filesystem::exists(std::filesystem::path(path), code)) {
    return std::unexpected(std::format("{}: the file exists (--overwrite writes over it)", path));
  }
  std::unique_ptr<std::FILE, CloseFile> file{std::fopen(path.c_str(), "wb")};
  if (file == nullptr) return std::unexpected(std::format("{}: the file could not be opened for writing", path));
  const bool wrote = std::fwrite(bytes.data(), 1, bytes.size(), file.get()) == bytes.size();
  // Closed here, where what it says is heard: fclose writes what is still buffered.
  if (std::fclose(file.release()) != 0 || !wrote) {
    return std::unexpected(std::format("{}: the file could not be written", path));
  }
  return {};
}

// JSON.
std::string json_string(std::string_view text) {
  std::string quoted = "\"";
  for (const char c : text) {
    const auto byte = static_cast<unsigned char>(c);
    if (c == '"' || c == '\\') {
      quoted += '\\';
      quoted += c;
    } else if (byte < 0x20u) {
      quoted += std::format("\\u{:04x}", static_cast<unsigned>(byte));
    } else {
      quoted += c;
    }
  }
  return quoted + "\"";
}

std::string json_number(double value) {
  if (!std::isfinite(value)) return "null";
  return std::format("{}", value);
}

std::string json_optional(const std::optional<double>& value) { return value ? json_number(*value) : "null"; }

template <typename T, typename Text>
std::string json_list(const std::vector<T>& items, const Text& text) {
  std::string list = "[";
  for (std::size_t k = 0; k < items.size(); ++k) list += (k == 0u ? "" : ", ") + text(items[k]);
  return list + "]";
}

std::string_view stop_name(Stop stop) noexcept {
  switch (stop) {
    case Stop::converged:
      return "converged";
    case Stop::iterations:
      return "iterations";
    case Stop::observer:
      return "observer";
    case Stop::stationary:
      return "stationary";
  }
  return "iterations";
}

std::string records_json(const std::vector<Record>& records) {
  return std::format(
      "{{\n    \"iterations\": {},\n    \"seconds\": {},\n    \"primal\": {},\n    \"dual\": {},\n    \"scaling\": "
      "{},\n    \"partial_gap\": {}\n  }}",
      json_list(records, [](const Record& record) { return std::to_string(record.iteration); }),
      json_list(records, [](const Record& record) { return json_number(record.seconds); }),
      json_list(records, [](const Record& record) { return json_number(record.primal); }),
      json_list(records, [](const Record& record) { return json_number(record.dual); }),
      json_list(records, [](const Record& record) { return json_optional(record.scaling); }),
      json_list(records, [](const Record& record) { return json_optional(record.partial_gap); }));
}

std::string_view space_name(ColourSpace space) noexcept {
  switch (space) {
    case ColourSpace::grayscale:
      return "grayscale";
    case ColourSpace::ycbcr:
      return "ycbcr";
    case ColourSpace::rgb:
      return "rgb";
  }
  return "grayscale";
}

std::string_view method_name(Method method) noexcept {
  switch (method) {
    case Method::mmse:
      return "mmse";
    case Method::tv:
      return "tv";
    case Method::tgv:
      return "tgv";
    case Method::subgradient:
      return "subgradient";
  }
  return "tgv";
}

// What was reconstructed, whichever the input.
struct Made {
  MdArray<double, PictureExtents> picture;
  output::Photometric photometric = output::Photometric::grey;
  bool alpha = false;
  std::size_t height = 0;
  std::size_t width = 0;
  ColourSpace space = ColourSpace::grayscale;
  std::int32_t exif_orientation = 0;
  std::vector<std::uint8_t> icc_profile;
  std::vector<Record> records;
  std::uint64_t iterations = 0;
  Stop stop = Stop::iterations;
  bool png = false;
  std::int32_t png_bits = 8;
  std::uint64_t alpha_iterations = 0;
  Stop alpha_stop = Stop::iterations;
  std::vector<std::string> warnings;
  // Of a projection: the picture's excess, component by component.
  bool projection = false;
  std::vector<picture::Summary> excess;
};

int status_of(Errc code) noexcept {
  switch (code) {
    case Errc::options:
      return refused;
    case Errc::io:
      return not_written;
    case Errc::read:
    case Errc::unsupported:
    case Errc::internal:
      return not_read;
  }
  return not_read;
}

// What a reconstruction of a JPEG file made, or a projection onto its intervals: its
// picture, or with --ycbcr its canvas cut to the picture (Y, Cb and Cr, which a picture
// of RGB's space is not converted from), and what the file says.
void take(Made& made, Decoded decoded, bool ycbcr) {
  made.height = decoded.height;
  made.width = decoded.width;
  made.space = decoded.space;
  made.exif_orientation = decoded.exif_orientation;
  made.icc_profile = std::move(decoded.icc_profile);
  made.records = std::move(decoded.records);
  made.iterations = decoded.iterations;
  made.stop = decoded.stop;
  if (!decoded.warning.empty()) made.warnings.push_back(decoded.warning);
  if (ycbcr) {
    made.picture = picture_of(decoded.canvas, decoded.height, decoded.width, ColourSpace::rgb);
    made.photometric = output::Photometric::ycbcr;
  } else {
    made.picture = std::move(decoded.picture);
    made.photometric = decoded.space == ColourSpace::grayscale ? output::Photometric::grey : output::Photometric::rgb;
  }
}

// The settings that a projection takes besides the options of the output.
constexpr std::array<std::string_view, 4> projection_settings = {"slack", "max-pixels", "max-scans",
                                                                 "warnings-are-errors"};

// What went wrong: the exit status, and what is said.
struct Failure {
  int status = not_read;
  std::string message;
};

// The picture that --project-only gives: its samples as the file stores the picture,
// and the first warning of its reading.
struct Given {
  MdArray<double, PictureExtents> samples;
  std::string warning;
};

// The samples of a PNG file's levels on the scale of 0 to 255: level v of `bits` bits is
// the double nearest to v 255 / (2^bits - 1), a quotient of integers that doubles hold,
// which the division rounds once.
void scale(std::span<const std::uint16_t> levels, std::int32_t bits, std::span<double> samples) noexcept {
  const double top = static_cast<double>((1u << static_cast<unsigned>(bits)) - 1u);
  const std::size_t count = levels.size();
  for (std::size_t k = 0; k < count; ++k) samples[k] = static_cast<double>(255u * levels[k]) / top;
}

// PICTURE as a projection takes it (docs/cli.md): a PNG file of the file's size and
// channels (or RGB of three channels alike, of a greyscale file), without alpha, turned
// as the result is; status 1 where it is not read, or not taken.
std::expected<Given, Failure> picture_given(const std::string& path, const Source& source, bool keep,
                                            std::uint64_t max_pixels) {
  const auto not_taken = [&path](std::string_view message) {
    return std::unexpected(Failure{.status = not_read, .message = std::format("{}: {}", path, message)});
  };
  const std::expected<std::vector<std::uint8_t>, std::string> bytes = read_all(path);
  if (!bytes) return std::unexpected(Failure{.status = not_read, .message = bytes.error()});
  if (!samples::is_png(*bytes)) return not_taken("the picture to project is a PNG file");
  const std::expected<jpegio::PngImage, jpegio::Error> png = jpegio::PngImage::read(*bytes, max_pixels);
  if (!png) return not_taken(std::format("the picture is not read: {}", png.error().message));
  const Frame& frame = source.frame;
  const std::size_t channels = picture::channels_of(frame);
  const Picture<const std::uint16_t> stored = png->levels();
  const std::size_t height = stored.extent(0);
  const std::size_t width = stored.extent(1);
  const std::size_t own = stored.extent(2);
  const std::span<const std::uint16_t> all(stored.data_handle(), stored.size());
  constexpr std::string_view grey = "a greyscale file's picture is grey, or RGB of three channels alike";
  if (own == 2u || own == 4u) return not_taken("a picture with alpha is not a JPEG file's");
  std::vector<std::uint16_t> levels;
  if (own == channels) {
    levels.assign(all.begin(), all.end());
  } else if (own == 3u && channels == 1u) {
    bool alike = true;
    for (std::size_t k = 0; k < all.size(); k += 3u) alike = alike && all[k] == all[k + 1u] && all[k] == all[k + 2u];
    if (!alike) return not_taken(grey);
    levels.reserve(all.size() / 3u);
    for (std::size_t k = 0; k < all.size(); k += 3u) levels.push_back(all[k]);
  } else {
    return not_taken(channels == 1u ? grey : std::string_view("a colour file's picture is RGB"));
  }
  // Turned as the result is.
  const std::int32_t orientation = source.decoded.exif_orientation;
  const bool turn = !keep && output::turns(orientation);
  const bool transposed = orientation >= 5 && orientation <= 8;
  const std::size_t rows = turn && transposed ? frame.width() : frame.height();
  const std::size_t columns = turn && transposed ? frame.height() : frame.width();
  if (height != rows || width != columns) {
    std::string_view hint;
    if (transposed && height == columns && width == rows) {
      hint = turn ? ", as the file stores it, which --orientation keep takes"
                  : ", turned upright, which --orientation apply takes";
    }
    return not_taken(
        std::format("a picture of the file is {} x {} pixels, not {} x {}{}", rows, columns, height, width, hint));
  }
  MdArray<double, PictureExtents> samples(PictureExtents{height, width, channels});
  scale(levels, png->bits(), samples.entries());
  if (turn) samples = output::oriented(std::as_const(samples).view(), output::inverse(orientation));
  return Given{.samples = std::move(samples), .warning = png->warning()};
}

// The settings that a projection took, as its report says them: the slack, and the
// options of reading that were given.
std::vector<std::string> projection_options(const Settings& settings) {
  std::vector<std::string> options;
  for (std::string& option : options_of(settings)) {
    if (option.starts_with("--slack=")) options.push_back(std::move(option));
  }
  const auto given = [&settings](std::string_view name) {
    return std::ranges::find(settings.given, name) != settings.given.end();
  };
  if (given("max-pixels")) options.push_back(std::format("--max-pixels={}", settings.reading.max_pixels));
  if (given("max-scans")) options.push_back(std::format("--max-scans={}", settings.reading.max_scans));
  if (settings.reading.warnings_are_errors) options.emplace_back("--warnings-are-errors");
  return options;
}

// A number as the reference implementation writes it on standard error: three decimals
// in scientific notation, the exponent without a plus sign or leading zeros (7.300e-1).
std::string scientific(double value) {
  std::string text = std::format("{:.3e}", value);
  const std::size_t e = text.find('e');
  if (e == std::string::npos) return text;
  const char* first = text.data() + e + 1u;
  const char* const last = text.data() + text.size();
  if (first != last && *first == '+') ++first;
  std::int32_t exponent = 0;
  const auto [stop, error] = std::from_chars(first, last, exponent);
  if (error != std::errc{} || stop != last) return text;
  return std::format("{}e{}", text.substr(0, e), exponent);
}

// The names of a file's components on standard error: none of a greyscale file's.
std::array<std::string_view, 3> component_names(ColourSpace space) noexcept {
  switch (space) {
    case ColourSpace::ycbcr:
      return {"Y", "Cb", "Cr"};
    case ColourSpace::rgb:
      return {"R", "G", "B"};
    case ColourSpace::grayscale:
      break;
  }
  return {"", "", ""};
}

// What standard error says of a component's excess.
std::string excess_line(std::string_view target, std::string_view name, const picture::Summary& excess) {
  std::string line = std::format("{}: ", target);
  if (!name.empty()) line += std::format("{}: ", name);
  line += std::format("{} of {} coefficients of the picture lay outside their intervals", excess.outside,
                      excess.coefficients);
  if (excess.outside > 0u) line += std::format(", by at most {} steps", scientific(excess.largest));
  return line;
}

std::string excess_json(const picture::Summary& excess) {
  return std::format("{{\"coefficients\": {}, \"outside\": {}, \"largest\": {}}}", excess.coefficients, excess.outside,
                     json_number(excess.largest));
}

}  // namespace

int run(std::span<const std::string_view> arguments, std::FILE* out, std::FILE* err) {
  Command command = command_of(arguments);
  if (command.help) {
    std::print(out, "{}", help_text);
    return written;
  }
  if (command.version) {
    std::println(out, "{}", version_text);
    return written;
  }
  std::vector<std::string_view> setting_arguments(command.settings.begin(), command.settings.end());
  const std::expected<Settings, Error> parsed = parse_settings(setting_arguments);
  if (!parsed) command.errors.push_back(parsed.error().message);
  if (command.positional.empty()) command.errors.emplace_back("INPUT is not given");
  if (command.positional.size() > 2u) {
    command.errors.push_back(std::format("INPUT and OUTPUT, not {} paths", command.positional.size()));
  }
  if (command.picture && parsed) {
    for (const std::string& name : parsed->given) {
      if (std::ranges::find(projection_settings, name) == projection_settings.end()) {
        command.errors.push_back(
            std::format("--{} is not taken with --project-only, which takes --slack and the options of reading", name));
      }
    }
  }
  if (command.picture == "-" && !command.positional.empty() && command.positional[0] == "-") {
    command.errors.emplace_back("INPUT and PICTURE are not both the standard input");
  }
  // The format: --format, else OUTPUT's extension, else TIFF.
  Format format = Format::tiff;
  if (command.format) {
    format = *command.format;
  } else if (command.positional.size() == 2u && command.positional[1] != "-") {
    format = format_of_name(command.positional[1]).value_or(Format::tiff);
  }
  if (format == Format::tiff && command.bits) command.errors.emplace_back("--bits is for PNG and PNM");
  if (format != Format::png && command.compression) command.errors.emplace_back("--compression is for PNG");
  if (format != Format::tiff && command.ycbcr) command.errors.emplace_back("--ycbcr is for TIFF");
  if (format == Format::tiff && (command.dither || command.together)) {
    command.errors.emplace_back("--dither and --dither-channels are for PNG and PNM");
  }
  if (command.together && command.dither == output::Dither::none) {
    command.errors.emplace_back("--dither-channels is for --dither blue-noise");
  }
  if (!command.errors.empty()) {
    for (const std::string& error : command.errors) std::println(err, "unround: {}", error);
    std::println(err, "unround: --help gives the options");
    return refused;
  }
  const Settings& settings = *parsed;
  const std::string& input = command.positional[0];
  const std::expected<std::vector<std::uint8_t>, std::string> bytes = read_all(input);
  if (!bytes) {
    std::println(err, "unround: {}", bytes.error());
    return not_read;
  }
  const bool verbose = command.verbose && !command.quiet;
  const pdhg::Observer observer = [verbose, err](const Record& record, std::span<const GridArray<double>> canvas) {
    if (verbose) {
      std::size_t samples = 0;
      for (const GridArray<double>& channel : canvas) samples += channel.size();
      const double gap = (record.primal - record.dual) / static_cast<double>(std::max<std::size_t>(samples, 1u));
      std::println(err, "{:>8} {:.3e} {} {} {:.3f}", record.iteration, gap, record.primal, record.dual, record.seconds);
    }
    return false;
  };
  Made made;
  if (command.picture) {
    if (samples::is_png(*bytes)) {
      std::println(err, "unround: {}: --project-only projects onto the intervals of a JPEG file, not of a PNG file",
                   input);
      return refused;
    }
    const std::expected<Source, Error> source = read_source(*bytes, settings);
    if (!source) {
      std::println(err, "unround: {}: {}", input, source.error().message);
      return status_of(source.error().code);
    }
    if (command.ycbcr && source->decoded.space != ColourSpace::ycbcr) {
      std::println(err, "unround: --ycbcr is for files in YCbCr");
      return refused;
    }
    const std::expected<Given, Failure> given =
        picture_given(*command.picture, *source, command.keep, settings.reading.max_pixels);
    if (!given) {
      std::println(err, "unround: {}", given.error().message);
      return given.error().status;
    }
    std::expected<picture::Projection, Error> projection = picture::project(*source, given->samples.view());
    if (!projection) {
      std::println(err, "unround: {}: {}", input, projection.error().message);
      return status_of(projection.error().code);
    }
    made.projection = true;
    for (const BlocksArray<double>& excess : projection->excess) made.excess.push_back(picture::summary(excess.view()));
    take(made, std::move(projection->decoded), command.ycbcr);
    // After the file's own, and before the writing's.
    if (!given->warning.empty()) made.warnings.push_back(std::format("{}: {}", *command.picture, given->warning));
  } else if (samples::is_png(*bytes)) {
    if (command.ycbcr) {
      std::println(err, "unround: --ycbcr is for JPEG files in YCbCr");
      return refused;
    }
    std::expected<samples::Reconstructed, Error> result = samples::reconstruct(*bytes, settings, observer);
    if (!result) {
      std::println(err, "unround: {}: {}", input, result.error().message);
      return status_of(result.error().code);
    }
    made.png = true;
    made.png_bits = result->bits;
    made.alpha = result->alpha;
    made.picture = std::move(result->picture);
    made.height = made.picture.extent(0);
    made.width = made.picture.extent(1);
    const std::size_t colours = made.picture.extent(2) - (made.alpha ? 1u : 0u);
    made.space = colours == 1u ? ColourSpace::grayscale : ColourSpace::rgb;
    made.photometric = colours == 1u ? output::Photometric::grey : output::Photometric::rgb;
    made.icc_profile = std::move(result->icc_profile);
    made.records = std::move(result->colours.records);
    made.iterations = result->colours.iterations;
    made.stop = result->colours.stop;
    made.alpha_iterations = result->alphas.iterations;
    made.alpha_stop = result->alphas.stop;
    if (!result->warning.empty()) made.warnings.push_back(result->warning);
  } else {
    if (command.ycbcr) {
      // Refused before the solve, from the file's header: only a file in YCbCr has a Y,
      // Cb and Cr. A file that is not read is left to decode, which says why.
      const jpegio::Options reading{.max_pixels = settings.reading.max_pixels,
                                    .max_scans = settings.reading.max_scans,
                                    .warnings_are_errors = settings.reading.warnings_are_errors};
      const std::expected<jpegio::Image, jpegio::Error> image = jpegio::Image::read(*bytes, reading);
      if (image && image->color_space() != jpegio::ColorSpace::ycbcr) {
        std::println(err, "unround: --ycbcr is for files in YCbCr");
        return refused;
      }
    }
    std::expected<Decoded, Error> decoded = decode(*bytes, settings, observer);
    if (!decoded) {
      std::println(err, "unround: {}: {}", input, decoded.error().message);
      return status_of(decoded.error().code);
    }
    if (command.ycbcr && decoded->space != ColourSpace::ycbcr) {
      std::println(err, "unround: --ycbcr is for files in YCbCr");
      return refused;
    }
    take(made, std::move(*decoded), command.ycbcr);
  }
  // Upright, unless kept as stored.
  const bool oriented = !command.keep && output::turns(made.exif_orientation);
  if (oriented) made.picture = output::oriented(std::as_const(made.picture).view(), made.exif_orientation);
  const std::string target = command.positional.size() == 2u ? command.positional[1]
                             : input == "-"
                                 ? std::string("-")
                                 : std::filesystem::path(input).replace_extension(extension_of(format)).string();
  // The ICC profile, where it goes with the picture.
  std::span<const std::uint8_t> profile;
  if (!command.no_icc && !made.icc_profile.empty()) {
    if (format == Format::pnm) {
      made.warnings.emplace_back("PNM holds no ICC profile: the file's is not written");
    } else if (const auto accepted =
                   jpegio::check_icc(made.icc_profile, static_cast<std::int32_t>(made.picture.extent(2)));
               accepted) {
      profile = made.icc_profile;
    } else {
      made.warnings.push_back(std::format("the ICC profile is not written: {}", accepted.error()));
    }
  }
  const std::int32_t bits = command.bits.value_or(8);
  const output::Dither dither = command.dither.value_or(output::Dither::blue_noise);
  const bool apart = !command.together.value_or(false);
  std::vector<std::uint8_t> file;
  switch (format) {
    case Format::tiff:
      file = output::tiff_bytes(std::as_const(made.picture).view(), made.photometric, made.alpha, profile);
      if (file.empty()) {
        std::println(err, "unround: the picture is too large for a TIFF file");
        return not_written;
      }
      break;
    case Format::png: {
      const output::LevelsArray levels = output::levels_of(std::as_const(made.picture).view(), bits, dither, apart);
      const jpegio::PngOptions options{.compression = command.compression.value_or(-1), .icc_profile = profile};
      std::expected<jpegio::PngFile, jpegio::Error> png;
      if (bits == 8) {
        MdArray<std::uint8_t, PictureExtents> eight(levels.extents());
        std::ranges::transform(levels.entries(), eight.entries().begin(),
                               [](std::uint16_t level) { return static_cast<std::uint8_t>(level); });
        png = jpegio::write_png(std::as_const(eight).view(), options);
      } else {
        png = jpegio::write_png(levels.view(), options);
      }
      if (!png) {
        std::println(err, "unround: {}", png.error().message);
        return not_written;
      }
      if (!png->warning.empty()) made.warnings.push_back(png->warning);
      file = std::move(png->bytes);
      break;
    }
    case Format::pnm: {
      if (made.alpha) {
        std::println(err, "unround: PNM holds no alpha: write the picture as TIFF or PNG");
        return not_written;
      }
      const output::LevelsArray levels = output::levels_of(std::as_const(made.picture).view(), bits, dither, apart);
      file = output::pnm_bytes(levels.view(), bits);
      break;
    }
  }
  const bool solved = !made.projection && (settings.method != Method::mmse || made.png);
  if (const std::expected<void, std::string> wrote =
          write_all(target, std::as_bytes(std::span(file)), command.overwrite, out);
      !wrote) {
    std::println(err, "unround: {}", wrote.error());
    return not_written;
  }
  if (command.report) {
    std::string report = "{\n";
    report += std::format("  \"version\": {},\n", json_string(version_text));
    report += std::format("  \"input\": {},\n", json_string(input));
    if (made.png) report += "  \"input_format\": \"png\",\n";
    if (made.projection && command.picture) {
      report += std::format("  \"picture\": {},\n", json_string(*command.picture));
    }
    report += std::format("  \"height\": {},\n  \"width\": {},\n", made.height, made.width);
    report += std::format("  \"color_space\": {},\n", json_string(space_name(made.space)));
    if (made.png) report += std::format("  \"bits\": {},\n  \"alpha\": {},\n", made.png_bits, made.alpha);
    report += std::format("  \"exif_orientation\": {},\n  \"oriented\": {},\n", made.exif_orientation, oriented);
    report += std::format("  \"icc_profile\": {},\n", !profile.empty());
    const std::string dither_name = format == Format::tiff           ? "null"
                                    : dither == output::Dither::none ? "\"none\""
                                    : apart                          ? "\"blue-noise\""
                                                                     : "\"blue-noise-together\"";
    report += std::format("  \"dither\": {},\n", dither_name);
    report += std::format("  \"warnings\": {},\n", json_list(made.warnings, json_string));
    report += std::format("  \"method\": {},\n",
                          json_string(made.projection ? std::string_view("projection") : method_name(settings.method)));
    report +=
        std::format("  \"settings\": {}",
                    json_list(made.projection ? projection_options(settings) : options_of(settings), json_string));
    if (made.png && made.alpha) {
      report += std::format(",\n  \"alpha_solver\": {{\"iterations\": {}, \"stop\": {}}}", made.alpha_iterations,
                            json_string(stop_name(made.alpha_stop)));
    }
    // The solver's, where one solved: not the MMSE decoder.
    if (solved) {
      report += std::format(",\n  \"iterations\": {},\n  \"stop\": {},\n  \"records\": {}", made.iterations,
                            json_string(stop_name(made.stop)), records_json(made.records));
    }
    // A projection's, in place of a solver's: the picture's excess.
    if (made.projection) report += std::format(",\n  \"excess\": {}", json_list(made.excess, excess_json));
    report += "\n}\n";
    if (const auto wrote = write_all(*command.report, std::as_bytes(std::span(report)), true, out); !wrote) {
      std::println(err, "unround: {}", wrote.error());
      return not_written;
    }
  }
  if (!command.quiet) {
    for (const std::string& warning : made.warnings) std::println(err, "unround: {}: {}", target, warning);
    if (solved && target != "-") {
      std::println(err, "{}: {} iterations, {}", target, made.iterations, stop_name(made.stop));
      if (made.png && made.alpha) {
        std::println(err, "{}: alpha: {} iterations, {}", target, made.alpha_iterations, stop_name(made.alpha_stop));
      }
    }
    if (made.projection && target != "-") {
      const std::array<std::string_view, 3> names = component_names(made.space);
      for (std::size_t c = 0; c < made.excess.size() && c < names.size(); ++c) {
        std::println(err, "{}", excess_line(target, names[c], made.excess[c]));
      }
    }
  }
  return written;
}

}  // namespace unround::cli
