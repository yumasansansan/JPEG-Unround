// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// What goes wrong, as the library's functions return it: a kind and a message,
// in a std::expected. The kinds are those of the command line's exit statuses
// and of the C interface of the reference implementation (docs/cli.md).
#ifndef UNROUND_ERROR_HPP
#define UNROUND_ERROR_HPP

#include <cstdint>
#include <string>

namespace unround {

enum class Errc : std::int32_t {
  options,      // options, or arrays, out of their ranges
  read,         // a file that the C layer could not read, or would not
  unsupported,  // a file whose layout JPEG-Unround does not take
  io,           // a file that could not be written or read
  internal,     // an error of JPEG-Unround itself
};

struct Error {
  Errc code = Errc::internal;
  std::string message;
};

}  // namespace unround

#endif  // UNROUND_ERROR_HPP
