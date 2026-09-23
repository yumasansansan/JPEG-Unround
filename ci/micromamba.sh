#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
#
#   ci/micromamba.sh
#
# Sets up a GitHub Actions runner for the Python implementation: micromamba of
# one pinned version, from the binaries mamba-org publishes, checked by their
# SHA-256 (GitHub lists it as the asset's digest, and each binary has a .sha256
# beside it that says the same), and the environment of environment.yml, created
# at a prefix of its own. micromamba goes first on PATH for the later steps, and
# UNROUND_PYTHON_PREFIX names the environment for ci/python.sh.
#
# To move to another version of micromamba, change the values below.
set -euo pipefail

version=2.9.0-0
case "${RUNNER_OS:-}" in
  Linux)
    platform=linux-64
    sha256=366cd9cd8be14df1ab8ed50352a82111082a36686b2d389fdb79a92c3fafb3e3
    executable=micromamba
    ;;
  macOS)
    platform=osx-arm64
    sha256=ec2a072f028e1a7cf20f3e2e74d5a8127cf5a5f27636375b5359811565f4e5be
    executable=micromamba
    ;;
  Windows)
    platform=win-64
    sha256=a6d804394b2418991c4e29562853eaace2f2ce9d9da661a98e74e02e8dbb44b0
    executable=micromamba.exe
    ;;
  *)
    echo "error: unsupported runner '${RUNNER_OS:-}'" >&2
    exit 1
    ;;
esac

temp=$RUNNER_TEMP
if command -v cygpath > /dev/null; then
  temp=$(cygpath --unix "$RUNNER_TEMP")
fi
bin=$temp/micromamba-bin
mkdir -p "$bin"
curl --fail --location --silent --show-error --retry 3 --retry-all-errors \
  --output "$bin/$executable" \
  "https://github.com/mamba-org/micromamba-releases/releases/download/$version/micromamba-$platform"
if command -v sha256sum > /dev/null; then
  actual=$(sha256sum "$bin/$executable")
else
  actual=$(shasum -a 256 "$bin/$executable")
fi
actual=${actual%% *}
if [ "$actual" != "$sha256" ]; then
  echo "error: micromamba $version for $platform has SHA-256 $actual, expected $sha256" >&2
  exit 1
fi
chmod +x "$bin/$executable"
"$bin/$executable" --version

root=$temp/micromamba
prefix=$root/envs/jpeg-unround
if command -v cygpath > /dev/null; then
  root=$(cygpath --windows "$root")
  prefix=$(cygpath --windows "$prefix")
fi
export MAMBA_ROOT_PREFIX=$root
"$bin/$executable" create --yes --file environment.yml --prefix "$prefix"

if command -v cygpath > /dev/null; then
  cygpath --windows "$bin"
else
  echo "$bin"
fi >> "$GITHUB_PATH"
{
  echo "MAMBA_ROOT_PREFIX=$root"
  echo "UNROUND_PYTHON_PREFIX=$prefix"
} >> "$GITHUB_ENV"
