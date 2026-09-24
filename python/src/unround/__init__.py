# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""JPEG decoding that removes compression artifacts within the quantization intervals.

The Python implementation of JPEG-Unround, the reference of the other two. The
mathematics is docs/math.md's. unround.jpegio reads JPEG files through the C layer
that all three implementations share; unround.decode reconstructs a greyscale file.

  dct          the 8x8 block DCT of JPEG
  laplace      the Laplace model of AC coefficients and their MMSE centres
  operators    finite differences, their adjoints, and pointwise norms
  model        the quantization constraint set, the data term, the objectives and gaps
  pdhg         the primal-dual method for TV and TGV
  subgradient  a subgradient method of jpeg2png's kind, to compare with
  results      what the solvers return
  metrics      PSNR, PSNR-B, and consistency with the intervals
  decode       a greyscale file, reconstructed by one of the methods
  tiff         the result in binary64, written bit for bit
"""

from unround import dct, decode, jpegio, laplace, metrics, model, operators, pdhg, results, subgradient, tiff

__all__ = [
    "dct",
    "decode",
    "jpegio",
    "laplace",
    "metrics",
    "model",
    "operators",
    "pdhg",
    "results",
    "subgradient",
    "tiff",
]
__version__ = "0.1.0"
