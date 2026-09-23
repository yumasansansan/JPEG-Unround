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
# all.
#
# On macOS, UNROUND_MACOS_TARGETS names the deployment targets to measure for,
# separated by spaces (26.0 when it is empty or unset). The SDK's libc++ marks
# some features as available only from a given version of macOS, and a newer
# SDK brings headers of its own, so the reports and results are named after
# both the SDK and the deployment target (report-macos-arm64-sdk27.0-min26.0.md).
set -euo pipefail

out=$1
mkdir -p "$out"
root=$(pwd)
case "$(uname -s)" in
  MINGW* | MSYS* | CYGWIN*) system=windows python=python exe=.exe ;;
  Darwin) system=macos python=python3 exe= ;;
  *) system=linux python=python3 exe= ;;
esac

# The runs of this system: an empty name on Windows and Linux, and one per
# deployment target on macOS.
runs=("")
if [ "$system" = macos ]; then
  targets=${UNROUND_MACOS_TARGETS:-}
  if [ -z "${targets// /}" ]; then
    targets=26.0
  fi
  sdk=$(xcrun --show-sdk-version)
  runs=()
  for target in $targets; do
    runs+=("$target")
  done
  echo "macOS SDK $sdk ($(xcrun --show-sdk-path)); deployment targets: ${runs[*]}"
fi

# Each program is built on its own, so that one that fails to build does not
# hide whether the others build. The log of a failed build keeps its last lines.
probe_modules() {  # build-directory [cmake option...]
  local build=$1
  shift
  local args=(-S tools/feature-probe/cmake-modules -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=Release
              -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_LINKER_TYPE=LLD "$@")
  if cmake "${args[@]}" > build-probe-modules.log 2>&1; then
    grep -E "std module|C\+\+ library|import std is not probed" build-probe-modules.log || true
    for program in probe_named probe_import_std probe_mixed; do
      if ! grep -q "build $program$exe:" "$build/build.ninja"; then
        echo "$program: not configured (no std module to import)"
      elif cmake --build "$build" --target "$program" > "build-$program.log" 2>&1; then
        if "$build/$program$exe" > /dev/null; then
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
  rm -f build-probe-modules.log build-probe_*.log
}

summaries=()
for target in "${runs[@]}"; do
  if [ "$system" = macos ]; then
    label=macos-arm64-sdk$sdk-min$target
    "$python" tools/feature-probe/run.py --out "$out" --label "$label" --macos-min "$target"
  else
    label=$system
    "$python" tools/feature-probe/run.py --out "$out"
  fi

  modules=$out/modules-$label.md
  {
    echo "## C++ modules through CMake ($label)"
    echo
    echo '```'
    # Build trees are named after the run, so that one working tree can be probed
    # from Windows and from WSL without either reading the other's cache.
    if [ "$system" = macos ]; then
      probe_modules "build/probe-modules-$label" "-DCMAKE_OSX_DEPLOYMENT_TARGET=$target"
      # Where a std module for the SDK's libc++ could come from, if Xcode ships one.
      echo
      echo "std module sources in Xcode and the SDK:"
      developer=$(xcode-select -p)
      find "$developer/Toolchains/XcodeDefault.xctoolchain/usr" "$(xcrun --show-sdk-path)/usr" \
        \( -name 'libc++.modules.json' -o -name 'std.cppm' -o -name 'std.compat.cppm' \) -print 2> /dev/null ||
        true
    else
      probe_modules "build/probe-modules-$label"
    fi
    echo '```'
  } > "$modules" || true
  summaries+=("$modules")
done

# Rust is measured once, for the last deployment target on macOS.
rust=$out/rust-$label.md
{
  echo "## Rust with LLD and a C23 static library ($label)"
  echo
  echo '```'
  if command -v cargo > /dev/null; then
    rustc --version
    # rustc and Clang both read the deployment target from the environment.
    if [ "$system" = macos ]; then
      export MACOSX_DEPLOYMENT_TARGET=$target
    fi
    (cd tools/feature-probe/rust-link &&
      CARGO_TARGET_DIR="$root/build/rust-link-$label" cargo run --release 2>&1) || echo "failed"
  else
    echo "no Rust on this runner"
  fi
  echo '```'
} > "$rust" || true

if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
  cat "$out"/report-*.md "${summaries[@]}" "$rust" >> "$GITHUB_STEP_SUMMARY"
fi
