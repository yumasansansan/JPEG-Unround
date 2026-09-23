# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""JPEG decoding that removes compression artifacts within the quantization intervals.

The Python implementation of JPEG-Unround. unround.jpegio reads JPEG files through
the C layer that all three implementations share.
"""

from unround import jpegio

__all__ = ["jpegio"]
__version__ = "0.1.0"
