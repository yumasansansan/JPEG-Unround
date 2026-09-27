# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""python -m unround: the command line of the reference implementation (docs/cli.md), through its C interface."""

import sys

from unround import native

if __name__ == "__main__":
    sys.exit(native.main())
