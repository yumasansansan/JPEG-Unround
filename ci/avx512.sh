#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
#
#   ci/avx512.sh [test | measure]
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
# With `measure`, where the processor has AVX-512F, the speed of the solvers and of
# their parts (rust/unround/tests/throughput.rs) is measured in the build shipped
# and in this one, twice each, one after the other, on the same machine; under
# GitHub Actions the solvers' figures go into the summary of the run too.
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
if [[ "${1:-}" == measure ]]; then
    if ! grep -qw avx512f /proc/cpuinfo 2>/dev/null; then
        echo "note: this processor has no AVX-512F; nothing is measured"
        exit 0
    fi
    grep -m 1 "model name" /proc/cpuinfo
    measured=$(mktemp -d)
    for round in 1 2; do
        for build in shipped avx512; do
            if [[ $build == shipped ]]; then
                build_arguments=(--release --locked)
            else
                build_arguments=("${arguments[@]}")
            fi
            echo "== $build, round $round"
            cargo test "${build_arguments[@]}" -p jpeg-unround --test throughput -- --ignored --nocapture \
                --test-threads 1 2>&1 | tee "$measured/$build-$round.txt"
        done
    done
    if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
        {
            echo "### The solvers, in ns per sample and iteration"
            echo
            echo "$(grep -m 1 "model name" /proc/cpuinfo | sed 's/.*: //'), $(nproc) threads"
            echo
            echo '```'
            for round in 1 2; do
                for build in shipped avx512; do
                    echo "== $build, round $round"
                    grep "ns per sample and iteration" "$measured/$build-$round.txt" || true
                done
            done
            echo '```'
        } >> "$GITHUB_STEP_SUMMARY"
    fi
fi
