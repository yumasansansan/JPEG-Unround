// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The command line of the C++ implementation, unround (docs/cli.md).

#include "unround/cli.hpp"

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <span>
#include <string_view>
#include <vector>

int main(int argc, char** argv) {
  try {
    const std::span<char* const> given{argv, static_cast<std::size_t>(argc)};
    std::vector<std::string_view> arguments;
    for (const char* argument : given.subspan(given.empty() ? 0u : 1u)) arguments.emplace_back(argument);
    return unround::cli::run(arguments, stdout, stderr);
  } catch (const std::exception& error) {
    (void)std::fputs("unround: ", stderr);
    (void)std::fputs(error.what(), stderr);
    (void)std::fputc('\n', stderr);
    return EXIT_FAILURE;
  }
}
