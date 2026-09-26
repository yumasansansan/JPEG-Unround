#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
#
#   ci/python.sh <preset>
#
# Builds the shared C layer with a CMake preset (-DUNROUND_WITH_PYTHON=ON) and
# checks that it exports its API and nothing else; then checks and tests the
# Python implementation (python/) in the micromamba environment of
# environment.yml, with the library it just built: ruff (the rules and the
# layout), mypy (strict) and pytest. The Python outside the package (scripts/,
# experiments/, ci/ and conformance/, with ruff.toml) is held to the same rules
# and types, and the exact references of conformance/ are checked against their
# definitions.
#
# The environment is the one named jpeg-unround, which `micromamba create -f
# environment.yml` makes, or the one at the prefix in UNROUND_PYTHON_PREFIX, which
# ci/micromamba.sh makes in CI.
set -euo pipefail

preset=${1:-}
if [ -z "$preset" ]; then
  echo "usage: ci/python.sh <preset>" >&2
  exit 2
fi

cmake --preset "$preset" -DUNROUND_WITH_PYTHON=ON
cmake --build --preset "$preset" --target unround_jpegio_shared
ctest --preset "$preset" --tests-regex '^c\.jpegio\.exports$'

case "$(uname -s)" in
  Linux) name=libunround_jpegio.so ;;
  Darwin) name=libunround_jpegio.dylib ;;
  MINGW* | MSYS* | CYGWIN*) name=unround_jpegio.dll ;;
  *) echo "error: this script does not know the system $(uname -s)" >&2; exit 1 ;;
esac
library=$(pwd -P)/build/$preset/python/$name
if command -v cygpath > /dev/null; then
  library=$(cygpath --windows "$library")
fi
export UNROUND_JPEGIO_LIBRARY=$library
echo "== UNROUND_JPEGIO_LIBRARY=$UNROUND_JPEGIO_LIBRARY"

if [ -n "${UNROUND_PYTHON_PREFIX:-}" ]; then
  environment=(--prefix "$UNROUND_PYTHON_PREFIX")
else
  environment=(--name jpeg-unround)
fi
in_environment() {
  micromamba run "${environment[@]}" "$@"
}

in_environment python -c "import sys, sysconfig; gil = 'free-threaded' if sysconfig.get_config_var('Py_GIL_DISABLED') else 'with the GIL'; print(sys.version.split()[0], gil)"

echo "== python/"
(
  cd python
  in_environment ruff check
  in_environment ruff format --check
  in_environment mypy
  in_environment python -X utf8 -m pytest
)

echo "== scripts/, experiments/, ci/ and conformance/"
in_environment ruff check
in_environment ruff format --check
MYPYPATH=python/src in_environment mypy --config-file python/pyproject.toml scripts experiments ci conformance
in_environment python conformance/references.py --check
