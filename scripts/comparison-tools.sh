#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
#
#   scripts/comparison-tools.sh
#
# Builds the tools that JPEG-Unround is compared against -- jpeg2png, jpegqs,
# knusperli, and mozjpeg's cjpeg -- from their sources, which it fetches into
# third_party/src/ (scripts/comparison-tools/CMakeLists.txt says which), with
# the toolchain of CMakePresets.json; runs each on a small JPEG file; and
# installs them into third_party/<system>-<processor>/bin.
#
# The build directory is build/comparison-tools/<system>-<processor>. On Linux
# the LLVM of the build comes first on PATH, as for the build of JPEG-Unround
# (/usr/lib/llvm-23/bin on Ubuntu).
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)

case "$(uname -s)" in
  Linux) system=linux ;;
  Darwin) system=macos ;;
  MINGW* | MSYS* | CYGWIN*) system=windows ;;
  *) echo "error: this script does not know the system $(uname -s)" >&2; exit 1 ;;
esac
case "$(uname -m)" in
  x86_64 | amd64) processor=x86_64 ;;
  arm64 | aarch64) processor=arm64 ;;
  *) echo "error: this script does not know the processor $(uname -m)" >&2; exit 1 ;;
esac
build=$root/build/comparison-tools/$system-$processor

cmake -S "$root/scripts/comparison-tools" -B "$build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_LINKER_TYPE=LLD \
  -DCMAKE_AR=llvm-ar \
  -DCMAKE_RANLIB=llvm-ranlib
cmake --build "$build"
ctest --test-dir "$build" --output-on-failure
cmake --install "$build"

prefix=$root/third_party/$system-$processor
echo "== installed into $prefix"
ls "$prefix/bin"
cat "$prefix/sources.txt"
