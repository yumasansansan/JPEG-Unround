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
# static library (tools/feature-probe/rust-link). A probe that
# fails is a result, not an error: this script fails only when a probe cannot
# be run at all. On macOS the probes run for two deployment targets, since the
# SDK's libc++ offers some features only to newer systems.
set -euo pipefail

out=$1
mkdir -p "$out"
case "$(uname -s)" in
  MINGW* | MSYS* | CYGWIN*) system=windows python=python exe=.exe ;;
  Darwin) system=macos python=python3 exe= ;;
  *) system=linux python=python3 exe= ;;
esac

if [ "$system" = macos ]; then
  for target in 14.0 26.0; do
    "$python" tools/feature-probe/run.py --out "$out" --label "macos-arm64-min$target" --macos-min "$target"
  done
else
  "$python" tools/feature-probe/run.py --out "$out"
fi

modules=$out/modules-$system.md
{
  echo "## C++ modules through CMake ($system)"
  echo
  args=(-S tools/feature-probe/cmake-modules -B build/probe-modules -G Ninja -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_LINKER_TYPE=LLD)
  if [ "$system" = macos ]; then
    args+=(-DCMAKE_OSX_DEPLOYMENT_TARGET=14.0)
  fi
  echo '```'
  if cmake "${args[@]}" 2>&1 && cmake --build build/probe-modules 2>&1; then
    for program in probe_named probe_import_std probe_mixed; do
      if [ -x "build/probe-modules/$program$exe" ]; then
        if "build/probe-modules/$program$exe" > /dev/null; then
          echo "$program: pass"
        else
          echo "$program: exit status $?"
        fi
      else
        echo "$program: not built"
      fi
    done
  else
    echo "configure or build failed"
  fi
  echo '```'
} > "$modules" || true

rust=$out/rust-$system.md
{
  echo "## Rust with LLD and a C23 static library ($system)"
  echo
  echo '```'
  if command -v cargo > /dev/null; then
    rustc --version
    # rustc and Clang both read the deployment target from the environment.
    if [ "$system" = macos ]; then
      export MACOSX_DEPLOYMENT_TARGET=14.0
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
