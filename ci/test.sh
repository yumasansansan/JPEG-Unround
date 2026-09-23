#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
#
#   ci/test.sh <preset>
#
# Runs the tests of a preset that ci/build.sh built, showing the output of each
# test in full: the tests say what they checked, and a sanitizer says what it
# found, only there.
set -euo pipefail

preset=$1
ctest --preset "$preset" --verbose
