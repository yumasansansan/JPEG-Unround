# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""The YCbCr of JFIF and its inverse, in binary64 (docs/math.md, 8).

The coefficients are the rationals that K_R = 0.299 and K_B = 0.114 make, each the
double nearest to it, and not the six-digit decimals that T.871 prints: those would
move a converted block's coefficients by more than 10^-4 of a small step. Each
operation is rounded to the nearest in the order docs/math.md gives, none fused, so
that the RGB of a solution is the same to the last bit wherever it is computed.
"""

from fractions import Fraction
from typing import Final

import numpy as np
import numpy.typing as npt

__all__ = ["to_rgb", "to_ycbcr"]

type Array = npt.NDArray[np.float64]

_CENTRE: Final = 128.0
_CHANNELS: Final = 3

_RED_FROM_CR: Final = float(Fraction(701, 500))  # 2 (1 - K_R) = 1.402
_BLUE_FROM_CB: Final = float(Fraction(443, 250))  # 2 (1 - K_B) = 1.772
_GREEN_FROM_CB: Final = float(Fraction(25251, 73375))  # 2 K_B (1 - K_B) / K_G
_GREEN_FROM_CR: Final = float(Fraction(209599, 293500))  # 2 K_R (1 - K_R) / K_G

_Y_FROM_RED: Final = float(Fraction(299, 1000))
_Y_FROM_GREEN: Final = float(Fraction(587, 1000))
_Y_FROM_BLUE: Final = float(Fraction(114, 1000))
_CB_FROM_BLUE: Final = float(Fraction(250, 443))  # 1 / 1.772
_CR_FROM_RED: Final = float(Fraction(500, 701))  # 1 / 1.402


def to_rgb(planes: npt.ArrayLike) -> Array:
    """The RGB of JFIF, (height, width, 3), of Y, Cb and Cr planes, (3, height, width), in binary64.

    R = Y + c_R (Cr - 128), B = Y + c_B (Cb - 128), G = (Y - c_GB (Cb - 128)) - c_GR (Cr - 128).
    """
    luma, blue_difference, red_difference = _planes(planes)
    cb = blue_difference - _CENTRE
    cr = red_difference - _CENTRE
    red = luma + _RED_FROM_CR * cr
    blue = luma + _BLUE_FROM_CB * cb
    green = (luma - _GREEN_FROM_CB * cb) - _GREEN_FROM_CR * cr
    return np.stack((red, green, blue), axis=-1)


def to_ycbcr(picture: npt.ArrayLike) -> Array:
    """The Y, Cb and Cr planes of JFIF, (3, height, width), of an RGB picture, (height, width, 3), in binary64.

    Y = (k_R R + k_G G) + k_B B, Cb = 128 + k_Cb (B - Y), Cr = 128 + k_Cr (R - Y).
    """
    samples = np.asarray(picture, dtype=np.float64)
    if samples.ndim != _CHANNELS or samples.shape[2] != _CHANNELS:
        message = f"an RGB picture is (height, width, 3), not {samples.shape}"
        raise ValueError(message)
    red, green, blue = samples[:, :, 0], samples[:, :, 1], samples[:, :, 2]
    luma = (_Y_FROM_RED * red + _Y_FROM_GREEN * green) + _Y_FROM_BLUE * blue
    blue_difference = _CENTRE + _CB_FROM_BLUE * (blue - luma)
    red_difference = _CENTRE + _CR_FROM_RED * (red - luma)
    return np.stack((luma, blue_difference, red_difference))


def _planes(planes: npt.ArrayLike) -> tuple[Array, Array, Array]:
    samples = np.asarray(planes, dtype=np.float64)
    if samples.ndim != _CHANNELS or samples.shape[0] != _CHANNELS:
        message = f"YCbCr planes are (3, height, width), not {samples.shape}"
        raise ValueError(message)
    return samples[0], samples[1], samples[2]
