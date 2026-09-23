// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// A file that includes standard headers (as the headers of a C library such as
// libjpeg do) and imports the std module as well.

#include <cstdio>
#include <string>
#include <vector>
import std;

int main(void) {
  std::vector<std::string> words{"a", "b"};
  std::printf("%zu\n", words.size());
  std::println("mixed: {}", words);
  return words.size() == 2uz ? 0 : 1;
}
