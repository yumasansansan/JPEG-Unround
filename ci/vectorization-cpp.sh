#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
#
#   ci/vectorization-cpp.sh BUILD
#
# Checks that the C++ library is vector code of the processor's full width
# (ci/vectorization.py cxx): each source of cpp/src compiled as the CMake build
# BUILD compiles it (its compile_commands.json; a Release build, which the
# library's hardening does not slow), for each target the library is built for:
# on x86-64, x86-64-v3 (AVX2, vectors of 256 bits, as the library's own build)
# and x86-64-v4 with vectors of 512 bits preferred (AVX-512); on AArch64, NEON
# (vectors of 128 bits).
set -euo pipefail

build=${1:?usage: ci/vectorization-cpp.sh BUILD}
root=$(pwd)
# Windows calls it python; there, python3 may be the Store's stub.
python=$(command -v python3 || command -v python)
if [ "${RUNNER_OS:-}" = Windows ] || [ "${OS:-}" = Windows_NT ]; then python=$(command -v python); fi
allowed="$root/ci/vectorization-cpp-allowed.toml"

case "$(uname -m)" in
x86_64 | AMD64 | amd64)
    "$python" ci/vectorization.py cxx "$build" --width 256 --flags=-march=x86-64-v3 --allowed "$allowed"
    "$python" ci/vectorization.py cxx "$build" --width 512 \
        "--flags=-march=x86-64-v4 -mprefer-vector-width=512" --allowed "$allowed"
    ;;
arm64 | aarch64)
    "$python" ci/vectorization.py cxx "$build" --width 128 --neon --allowed "$allowed"
    ;;
*)
    echo "error: no build to check on $(uname -m)" >&2
    exit 1
    ;;
esac
