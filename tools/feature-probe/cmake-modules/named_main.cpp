// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import unround.probe;

int main(void) { return unround::probe::twice(21) == 42 && unround::probe::block_size() == 8uz ? 0 : 1; }
