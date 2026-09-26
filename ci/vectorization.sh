#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
#
#   ci/vectorization.sh
#
# Checks that the Rust code is vector code of the processor's full width
# (ci/vectorization.py), for each build that is shipped: on x86-64, the default
# build (x86-64-v3, AVX2, vectors of 256 bits, rust/.cargo/config.toml) and the
# build for AVX-512 (ci/avx512.sh, vectors of 512 bits); on AArch64, NEON (vectors
# of 128 bits). The assembly of the library, and of
# rust/unround/examples/vectorization.rs with the kernels one by one, is compiled
# as the builds compile the code, with the flags of the build given as ci/avx512.sh
# gives them, after those of rust/.cargo/config.toml; but with the vectorization of
# the remainders of loops at a narrower width turned off, so that what is narrower
# than the full width is code that the compiler kept from it.
set -euo pipefail

root=$(pwd)
python=$(command -v python3 || command -v python)
triple=$(rustc -vV | sed -n 's/^host: //p')
case "$triple" in
x86_64-*)
    builds=(
        "avx2 256 []"
        'avx512 512 ["-C", "target-cpu=x86-64-v4", "-C", "target-feature=-prefer-256-bit"]'
    )
    neon=()
    ;;
aarch64-*)
    builds=("neon 128 []")
    neon=(--neon)
    ;;
*)
    echo "error: no build to check for $triple" >&2
    exit 1
    ;;
esac

cd rust
for build in "${builds[@]}"; do
    read -r name width flags <<<"$build"
    target="target/vectorization-$name"
    echo "== $name: vectors of $width bits"
    arguments=(--release --locked -p jpeg-unround --target "$triple" --target-dir "$target"
        --config "target.$triple.rustflags=$flags")
    listing=(-- --emit asm -C debuginfo=1 -C llvm-args=-enable-epilogue-vectorization=false)
    cargo rustc "${arguments[@]}" --lib "${listing[@]}"
    cargo rustc "${arguments[@]}" --example vectorization "${listing[@]}"
    output="$target/$triple/release"
    # The newest listing of each: the names have a hash on some systems.
    library=$(ls -t "$output"/deps/jpeg_unround-*.s | head -n 1)
    kernels=$(ls -t "$output"/examples/vectorization*.s | head -n 1)
    "$python" "$root/ci/vectorization.py" kernels "$kernels" --width "$width" "${neon[@]}"
    "$python" "$root/ci/vectorization.py" library "$library" --width "$width" "${neon[@]}" \
        --allowed "$root/ci/vectorization-allowed.toml"
done
