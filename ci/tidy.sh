#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
#
#   ci/tidy.sh <preset>
#
# Runs clang-tidy over the project's own C and C++ (c/, cpp/, tests/), with the
# checks that .clang-tidy names, on the compile commands of build/<preset>. Each
# file that the build compiles is read once, however many targets compile it,
# and the headers of those directories are read along with them. Nothing of
# libjpeg-turbo, libpng or zlib-ng is reported: their headers are included from
# the build directory's installations, which the pattern of what is ours leaves
# out.
#
# clang-tidy reads each file with the very options the build compiles it with,
# so the preset is configured here, and libjpeg-turbo, zlib-ng and libpng are
# built and installed, since the C layer includes their headers; nothing of this
# project's own code is compiled. .clang-tidy makes every check an error, so anything reported fails
# the script.
set -euo pipefail

preset=${1:-}
if [ -z "$preset" ]; then
  echo "usage: ci/tidy.sh <preset>" >&2
  exit 2
fi

build=build/$preset
cmake --preset "$preset" > /dev/null
cmake --build --preset "$preset" --target libjpeg-turbo libpng > /dev/null

# A file that several targets compile has a compile command for each of them,
# and clang-tidy would read it once for every command; the checks say the same
# thing every time, so the commands are reduced to the first of each file.
commands=$(mktemp -d)
trap 'rm -rf "$commands"' EXIT
python3 - "$build/compile_commands.json" "$commands/compile_commands.json" <<'REDUCE'
import json, sys
first = {}
for entry in json.load(open(sys.argv[1], encoding="utf-8")):
    first.setdefault(entry["file"], entry)
json.dump(list(first.values()), open(sys.argv[2], "w", encoding="utf-8"))
print("== %d compile commands, %d files" % (len(json.load(open(sys.argv[1], encoding="utf-8"))), len(first)))
REDUCE

root=$(pwd -P)
ours="^$root/(c|cpp|tests)/"
jobs=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)
echo "== $(clang-tidy --version | sed -n 's/^.*LLVM version/LLVM/p' | head -n 1), $jobs files at a time, on $build"
run-clang-tidy -p "$commands" -j "$jobs" -quiet -use-color=0 \
  -config-file="$root/.clang-tidy" -header-filter="$ours" "$ours"
