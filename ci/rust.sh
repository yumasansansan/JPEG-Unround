#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
#
#   ci/rust.sh
#
# Checks and tests the Rust implementation (rust/), with the toolchain that
# rust/rust-toolchain.toml pins: the layout (rustfmt), clippy with every warning
# an error (rust/Cargo.toml names the lints), and the tests in the dev and the
# release profile. Building the jpegio-sys crate builds the C layer through the
# CMake presets (rust/jpegio-sys/build.rs), so the LLVM toolchain has to be first
# on PATH, as ci/setup.sh puts it. Nothing is fetched from crates.io
# (--locked keeps it that way).
set -euo pipefail

cd rust
rustc --version
cargo --version
cargo fmt --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
cargo test --workspace --locked --release
