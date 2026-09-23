// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import std;

int main(void) {
  std::vector<int> v{1, 2, 3};
  std::println("import std: {}", v);
  return std::ranges::fold_left(v, 0, std::plus{}) == 6 ? 0 : 1;
}
