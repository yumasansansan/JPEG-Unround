# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""JPEG decoding that removes compression artifacts within the quantization intervals.

The Python package of JPEG-Unround: the reference implementation (Rust), called
through its C interface, with NumPy arrays in and out. The mathematics is
docs/math.md's, and the options are the command line's (docs/cli.md).

  decode    a file, its bytes, or its components as read, reconstructed (unround.native.decode)
  read      a file's quantized coefficients, tables and sampling factors (unround.jpegio.read)
  settings  the method, the model and the solvers' options
  native    the reference implementation: reconstruction, its writers (TIFF in binary64, PNG,
            PNM), EXIF's orientations, JFIF's colour conversions, and the command line
  jpegio    the C layer that reads JPEG files

python -m unround runs the command line.
"""

from unround import jpegio, native, settings
from unround.jpegio import read
from unround.native import (
    Decoded,
    Record,
    UnroundError,
    UnroundWarning,
    decode,
    oriented,
    png_bytes,
    pnm_bytes,
    solve,
    tiff_bytes,
    to_rgb,
    to_ycbcr,
)
from unround.settings import TGV, TV, DataTerm, PdhgOptions, Settings, SubgradientOptions

__all__ = [
    "TGV",
    "TV",
    "DataTerm",
    "Decoded",
    "PdhgOptions",
    "Record",
    "Settings",
    "SubgradientOptions",
    "UnroundError",
    "UnroundWarning",
    "decode",
    "jpegio",
    "native",
    "oriented",
    "png_bytes",
    "pnm_bytes",
    "read",
    "settings",
    "solve",
    "tiff_bytes",
    "to_rgb",
    "to_ycbcr",
]
__version__ = "0.1.0"
