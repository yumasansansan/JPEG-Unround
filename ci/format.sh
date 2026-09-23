#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
#
#   ci/format.sh
#
# Checks that the project's own C and C++ (c/, cpp/, tests/) are laid out as
# .clang-format says, with the clang-format of the toolchain; it changes nothing.
# `clang-format -i <file>` lays a file out. The Rust and the Python are checked by
# ci/rust.sh and ci/python.sh, with rustfmt and ruff.
set -euo pipefail

mapfile -t files < <(git ls-files -- c cpp tests | grep -E '\.(c|h|cpp|hpp)$')
if [ ${#files[@]} -eq 0 ]; then
  echo "error: no C or C++ files to read" >&2
  exit 1
fi
echo "== the layout of ${#files[@]} files, by $(clang-format --version)"
clang-format --dry-run --Werror "${files[@]}"
echo "Every file is laid out as .clang-format says."
