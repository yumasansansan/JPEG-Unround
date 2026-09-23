// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Checks that a program of this project gets its arguments in UTF-8 and opens
// files by UTF-8 names, which the command line of the C++ implementation relies
// on. On Windows that needs UTF-8 to be the code page of the process, which the
// manifest that every program is linked with names (cmake/utf8.manifest); on
// Linux and macOS, arguments and names are the bytes that they are given.
//
//   unround_cpp_utf8_arguments_test <text> <directory>
//
// <text> has to be the text of `expected` below, which CTest gives
// (cpp/CMakeLists.txt). The test makes a directory of that name in <directory>
// and a file of that name in it with fopen(), finds the file under its name
// with std::filesystem, reads it back with std::ifstream, and removes both.

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <ios>
#include <iterator>
#include <memory>
#include <print>
#include <span>
#include <string>
#include <string_view>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace {

// Kanji, Latin letters beyond ASCII, and a character beyond the Basic
// Multilingual Plane, which UTF-16 writes in two units: "日本語 ×ü 🙂".
constexpr std::string_view expected{"\xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E \xC3\x97\xC3\xBC \xF0\x9F\x99\x82"};

constexpr std::string_view contents{"JPEG-Unround\n"};

int failures = 0;

bool check(bool ok, std::string_view what) {
  if (!ok) {
    ++failures;
    std::println(stderr, "check failed: {}", what);
  }
  return ok;
}

// The bytes of a text in hexadecimal, for a message.
std::string hex(std::string_view text) {
  static constexpr std::string_view digits{"0123456789ABCDEF"};
  std::string result;
  for (const char c : text) {
    const unsigned int byte = static_cast<unsigned char>(c);
    if (!result.empty()) {
      result += ' ';
    }
    result += digits[static_cast<std::size_t>(byte >> 4U)];
    result += digits[static_cast<std::size_t>(byte & 0x0FU)];
  }
  return result;
}

struct CloseFile {
  void operator()(std::FILE* file) const noexcept { (void)std::fclose(file); }
};

int run(std::span<char* const> arguments) {
#if defined(_WIN32)
  // Without the manifest, the code page is the system language's, and the
  // arguments come in it.
  const UINT code_page = GetACP();
  check(code_page == CP_UTF8, std::format("the code page of the process is {}, not UTF-8 ({})", code_page, CP_UTF8));
#endif
  if (arguments.size() != 3) {
    std::println(stderr, "usage: unround_cpp_utf8_arguments_test <text> <directory>");
    return EXIT_FAILURE;
  }
  const std::string_view text{arguments[1]};
  if (!check(text == expected, std::format("the argument came as the bytes {}, not {}", hex(text), hex(expected)))) {
    return EXIT_FAILURE;
  }

  // A name given as char is read in the code page, as the argument was.
  const std::string directory = std::string{arguments[2]} + "/" + std::string{text};
  const std::filesystem::path directory_path{directory};
  std::filesystem::remove_all(directory_path);
  std::filesystem::create_directories(directory_path);
  const std::string name = directory + "/" + std::string{text} + ".txt";

  // C's stdio writes the file by its name.
  std::unique_ptr<std::FILE, CloseFile> file{std::fopen(name.c_str(), "wb")};
  if (!check(file != nullptr, std::format("fopen() opens {} to write", hex(name)))) {
    return EXIT_FAILURE;
  }
  const std::size_t written = std::fwrite(contents.data(), 1, contents.size(), file.get());
  const int closed = std::fclose(file.release());
  check(written == contents.size() && closed == 0, "fwrite() and fclose() write the file");

  // std::filesystem finds it under that name, which it gives as UTF-8.
  bool found = false;
  for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator{directory_path}) {
    const std::u8string stem = entry.path().stem().u8string();
    found = found || std::ranges::equal(stem, expected, [](char8_t a, char b) { return a == static_cast<char8_t>(b); });
  }
  check(found, "std::filesystem lists the file under its name");

  // A file stream opens it by the same name and reads what was written.
  std::ifstream stream{name, std::ios::binary};
  check(stream.is_open(), "std::ifstream opens the file");
  const std::string read_back{std::istreambuf_iterator<char>{stream}, std::istreambuf_iterator<char>{}};
  stream.close();
  check(read_back == contents, "std::ifstream reads what fwrite() wrote");

  check(std::filesystem::remove(std::filesystem::path{name}), "std::filesystem removes the file by its name");
  check(std::filesystem::remove(directory_path), "std::filesystem removes the directory by its name");
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

}  // namespace

int main(int argc, char** argv) {
  // A test that throws -- a directory that cannot be made, say -- fails with
  // what it says, rather than ending the program without a word.
  try {
    return run(std::span<char* const>{argv, static_cast<std::size_t>(argc)});
  } catch (const std::exception& error) {
    (void)std::fputs(error.what(), stderr);
    (void)std::fputc('\n', stderr);
    return EXIT_FAILURE;
  }
}
