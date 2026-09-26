#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
#
#   ci/avx512.sh [test]
#
# Builds the Rust command line (unround) and C API (unround_capi) for x86-64
# processors with AVX-512, into rust/target/avx512/<triple>/release: the code of
# the default build, which needs AVX2 and FMA (rust/.cargo/config.toml), compiled
# for x86-64-v4 with vectors of eight doubles. For x86-64-v4 LLVM keeps to vectors
# of 256 bits unless told otherwise, which -prefer-256-bit turned off does (rustc
# passes the feature to LLVM, and warns that it does not know it). The target is
# named, so that the build scripts, which run here, are built for this processor.
#
# With `test`, the tests of the build run too, where the processor has AVX-512F.
set -euo pipefail

case "$(uname -s)" in
MINGW* | MSYS* | CYGWIN*) triple=x86_64-pc-windows-msvc ;;
Linux) triple=x86_64-unknown-linux-gnu ;;
*)
    echo "error: the build for AVX-512 is for x86-64 Windows and Linux" >&2
    exit 1
    ;;
esac
flags='["-C", "target-cpu=x86-64-v4", "-C", "target-feature=-prefer-256-bit"]'
arguments=(--release --locked --target "$triple" --target-dir target/avx512 --config "target.$triple.rustflags=$flags")

cd rust
cargo build "${arguments[@]}" -p jpeg-unround -p unround-capi
ls -l "target/avx512/$triple/release/"
if [[ "${1:-}" == test ]]; then
    if grep -qw avx512f /proc/cpuinfo 2>/dev/null; then
        cargo test "${arguments[@]}" --workspace
    else
        echo "note: this processor has no AVX-512F; the build is not tested"
    fi
fi
