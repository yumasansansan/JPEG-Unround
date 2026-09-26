# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Where the shared libraries of the package are: at an environment variable's path, or in unround/_native."""

import os
import sys
from pathlib import Path
from typing import Final

__all__ = ["C_LAYER", "RUST", "bundled", "configured", "file_name"]

C_LAYER: Final = "unround_jpegio"
"""The C layer alone (unround/jpegio.h), which CMake builds with -DUNROUND_WITH_PYTHON=ON."""

RUST: Final = "unround_capi"
"""The C interface of the reference implementation (unround.h), which passes the C layer on too."""

_AFFIXES: Final = {"win32": ("", ".dll"), "darwin": ("lib", ".dylib")}


def file_name(stem: str) -> str:
    """The name of the shared library of the stem on this system."""
    prefix, suffix = _AFFIXES.get(sys.platform, ("lib", ".so"))
    return f"{prefix}{stem}{suffix}"


def configured(variable: str) -> Path | None:
    """The path in the environment variable, where it is set, whether or not a file is there."""
    value = os.environ.get(variable)
    return Path(value) if value else None


def bundled(stem: str) -> Path | None:
    """The package's own library of the stem, in unround/_native, where it is there."""
    path = Path(__file__).parent / "_native" / file_name(stem)
    return path if path.is_file() else None
