#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
#
#   ci/probe.sh <report directory>
#
# Measures what the toolchain of this runner can do, which decides the language
# and library features JPEG-Unround uses: the language and library probes
# (tools/feature-probe/run.py), C++ modules through CMake
# (tools/feature-probe/cmake-modules) and Rust linked with LLD against a C23
# static library (tools/feature-probe/rust-link). A probe that fails is a
# result, not an error: this script fails only when a probe cannot be run at
# all. macOS builds are for macOS 26.0 and later, and the SDK's libc++ offers
# some features only to recent systems, so everything is measured for that
# deployment target.
set -euo pipefail

out=$1
mkdir -p "$out"
case "$(uname -s)" in
  MINGW* | MSYS* | CYGWIN*) system=windows python=python exe=.exe ;;
  Darwin) system=macos python=python3 exe= ;;
  *) system=linux python=python3 exe= ;;
esac
macos_target=26.0

if [ "$system" = macos ]; then
  "$python" tools/feature-probe/run.py --out "$out" --macos-min "$macos_target"
else
  "$python" tools/feature-probe/run.py --out "$out"
fi

# Each program is built on its own, so that one that fails to build does not
# hide whether the others build. The log of a failed build keeps its last lines.
modules=$out/modules-$system.md
{
  echo "## C++ modules through CMake ($system)"
  echo
  echo '```'
  args=(-S tools/feature-probe/cmake-modules -B build/probe-modules -G Ninja -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_LINKER_TYPE=LLD)
  if [ "$system" = macos ]; then
    args+=("-DCMAKE_OSX_DEPLOYMENT_TARGET=$macos_target")
  fi
  if cmake "${args[@]}" > build-probe-modules.log 2>&1; then
    grep -E "std module|C\+\+ library|import std is not probed" build-probe-modules.log || true
    for program in probe_named probe_import_std probe_mixed; do
      if ! grep -q "build $program$exe:" build/probe-modules/build.ninja; then
        echo "$program: not configured (no std module to import)"
      elif cmake --build build/probe-modules --target "$program" > "build-$program.log" 2>&1; then
        if "build/probe-modules/$program$exe" > /dev/null; then
          echo "$program: pass"
        else
          echo "$program: exit status $?"
        fi
      else
        echo "$program: build failed"
        grep -E "error|FAILED" "build-$program.log" | head -n 5 || true
      fi
    done
  else
    echo "configure failed"
    tail -n 20 build-probe-modules.log
  fi
  if [ "$system" = macos ]; then
    # Where a std module for the SDK's libc++ could come from, if Xcode ships one.
    echo
    echo "std module sources in Xcode and the SDK:"
    developer=$(xcode-select -p)
    find "$developer/Toolchains/XcodeDefault.xctoolchain/usr" "$(xcrun --show-sdk-path)/usr" \
      \( -name 'libc++.modules.json' -o -name 'std.cppm' -o -name 'std.compat.cppm' \) -print 2> /dev/null ||
      true
  fi
  echo '```'
} > "$modules" || true
rm -f build-probe-modules.log build-probe_*.log

rust=$out/rust-$system.md
{
  echo "## Rust with LLD and a C23 static library ($system)"
  echo
  echo '```'
  if command -v cargo > /dev/null; then
    rustc --version
    # rustc and Clang both read the deployment target from the environment.
    if [ "$system" = macos ]; then
      export MACOSX_DEPLOYMENT_TARGET=$macos_target
    fi
    (cd tools/feature-probe/rust-link && cargo run --release 2>&1) || echo "failed"
  else
    echo "no Rust on this runner"
  fi
  echo '```'
} > "$rust" || true

if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
  cat "$out"/report-*.md "$modules" "$rust" >> "$GITHUB_STEP_SUMMARY"
fi
