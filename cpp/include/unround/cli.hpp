// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The command line of docs/cli.md: unround [OPTIONS] INPUT [OUTPUT].
#ifndef UNROUND_CLI_HPP
#define UNROUND_CLI_HPP

#include <cstdio>
#include <span>
#include <string_view>

namespace unround::cli {

// The exit statuses of docs/cli.md.
inline constexpr int written = 0;
inline constexpr int not_read = 1;
inline constexpr int refused = 2;
inline constexpr int not_written = 3;

// Runs the command line on its arguments (those after the program's name): reads the
// input, reconstructs it, writes the result and the report, and says what it did on
// `err` (standard output, for '-', is `out`). Returns the exit status.
[[nodiscard]] int run(std::span<const std::string_view> arguments, std::FILE* out, std::FILE* err);

}  // namespace unround::cli

#endif  // UNROUND_CLI_HPP
