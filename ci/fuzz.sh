#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
#
#   ci/fuzz.sh <seconds> <corpus directory> <crash directory>
#
# Fuzzes the C layer with libFuzzer for the given time, in a build of the fuzz
# preset that ci/build.sh made. The corpus starts from the JPEG files and the
# ICC profiles that the smoke test writes (c/jpegio/fuzz/smoke.c) and from
# whatever the directory already holds; an input that fails is written to the
# crash directory, and the job fails with it.
set -euo pipefail

seconds=$1
corpus=$2
crashes=$3
build=build/fuzz

mkdir -p "$corpus" "$crashes"
"$build/c/jpegio/unround_jpegio_fuzz_smoke" --seeds "$corpus"
"$build/c/jpegio/unround_jpegio_fuzz" \
  -max_total_time="$seconds" -timeout=10 -rss_limit_mb=2048 -print_final_stats=1 \
  -artifact_prefix="$crashes/" "$corpus"
