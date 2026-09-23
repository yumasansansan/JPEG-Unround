#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
#
#   ci/build.sh <preset> [<cmake option>...]
#
# Configures a CMake preset (CMakePresets.json) and builds it, with every
# command shown in full. The configuration checks the toolchain
# (cmake/LLVMToolchain.cmake), and this script shows what it checked, in the log
# and, in GitHub Actions, in the summary of the job.
#
# Then it lists the libraries that each program of the build asks the system for
# when it is loaded. The C++ library and the C runtime are the only ones
# JPEG-Unround takes from the system, and everything else is linked in, so for a
# build without sanitizers anything else on the list is an error. (A sanitizer's
# runtime is a library of its own on Windows and macOS, and brings libraries of
# the system with it on Linux; those builds are for testing and are only listed.)
set -euo pipefail

preset=$1
shift
build=build/$preset

cmake --preset "$preset" "$@"

toolchain=$build/llvm-toolchain.txt
echo "The toolchain, as the configuration checked it:"
cat "$toolchain"
if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
  {
    echo "### Toolchain of $preset"
    echo
    echo "| part | version | program |"
    echo "| --- | --- | --- |"
    while IFS=$'\t' read -r part version program; do
      echo "| $part | $version | \`$program\` |"
    done < "$toolchain"
    echo
  } >> "$GITHUB_STEP_SUMMARY"
fi

cmake --build --preset "$preset" --verbose

sanitized=0
if grep -q '^UNROUND_SANITIZERS:STRING=.\+' "$build/CMakeCache.txt"; then
  sanitized=1
fi

case "$(uname -s)" in
  Linux)
    list_libraries() { llvm-readelf --dynamic-table "$1" | sed -n 's/.*(NEEDED) *Shared library: \[\(.*\)\]/\1/p'; }
    allowed='^(libstdc\+\+\.so\.6|libm\.so\.6|libgcc_s\.so\.1|libc\.so\.6)$'
    ;;
  Darwin)
    list_libraries() { llvm-objdump --macho --dylibs-used "$1" | tail -n +2 | sed 's/^[[:space:]]*//; s/ (.*//'; }
    allowed='^(/usr/lib/libc\+\+\.1\.dylib|/usr/lib/libSystem\.B\.dylib)$'
    ;;
  MINGW* | MSYS* | CYGWIN*)
    list_libraries() { llvm-objdump --private-headers "$1" | sed -n 's/.*DLL Name: //p' | tr -d '\r'; }
    allowed='^(kernel32\.dll|msvcp140(_atomic_wait)?\.dll|vcruntime140(_1)?\.dll|api-ms-win-crt-[a-z0-9-]+\.dll)$'
    ;;
  *)
    echo "error: this script does not know the system $(uname -s)" >&2
    exit 1
    ;;
esac

status=0
programs=$(find "$build/c" "$build/cpp" -type f \( -name 'unround_*.exe' -o \( -name 'unround_*' ! -name '*.*' \) \) | sort)
for program in $programs; do
  echo "$program asks the system for:"
  libraries=$(list_libraries "$program")
  for library in $libraries; do
    echo "  $library"
    name=$library
    if [ "$(uname -s)" != Darwin ]; then
      name=$(printf '%s' "$library" | tr '[:upper:]' '[:lower:]')
    fi
    if [ "$sanitized" -eq 0 ] && ! printf '%s\n' "$name" | grep -Eq "$allowed"; then
      echo "error: $program asks the system for $library, which JPEG-Unround links statically or not at all" >&2
      status=1
    fi
  done
done
exit "$status"
