// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

module;
#include <cstddef>
export module unround.probe;

export namespace unround::probe {
constexpr std::size_t block_size(void) { return 8uz; }
int twice(int x) { return 2 * x; }
}  // namespace unround::probe
