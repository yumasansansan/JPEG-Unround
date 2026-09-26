# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""The conversion of JFIF (docs/math.md, 8).

Its rationals are checked exactly: they are what K_R = 0.299 and K_B = 0.114 make, and
the inverse undoes the forward conversion in rational numbers. Its order of operations
is checked to the last bit against the same operations on Python's floats. The round
trip in binary64 is checked against a running bound of its rounding, which each
operation adds to by U times its computed result, and each constant by its distance
from its rational.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import pytest

from unround import colour

U = Fraction(1, 2**53)
K_R, K_B = Fraction(299, 1000), Fraction(114, 1000)
K_G = 1 - K_R - K_B
RED_FROM_CR, BLUE_FROM_CB = Fraction(701, 500), Fraction(443, 250)
GREEN_FROM_CB, GREEN_FROM_CR = Fraction(25251, 73375), Fraction(209599, 293500)
CB_FROM_BLUE, CR_FROM_RED = 1 / BLUE_FROM_CB, 1 / RED_FROM_CR


def test_the_rationals_are_those_of_the_weights() -> None:
    assert Fraction(587, 1000) == 1 - K_R - K_B
    assert 2 * (1 - K_R) == RED_FROM_CR
    assert 2 * (1 - K_B) == BLUE_FROM_CB
    assert 2 * K_B * (1 - K_B) / K_G == GREEN_FROM_CB
    assert 2 * K_R * (1 - K_R) / K_G == GREEN_FROM_CR
    assert Fraction(250, 443) == 1 / (2 * (1 - K_B))
    assert Fraction(500, 701) == 1 / (2 * (1 - K_R))
    # In rationals, the inverse undoes the forward conversion.
    rng = np.random.default_rng(60)
    for _ in range(100):
        red, green, blue = (Fraction(int(value), 7) for value in rng.integers(-300, 2100, size=3))
        luma = K_R * red + K_G * green + K_B * blue
        cb = 128 + CB_FROM_BLUE * (blue - luma)
        cr = 128 + CR_FROM_RED * (red - luma)
        assert luma + RED_FROM_CR * (cr - 128) == red
        assert luma + BLUE_FROM_CB * (cb - 128) == blue
        assert luma - GREEN_FROM_CB * (cb - 128) - GREEN_FROM_CR * (cr - 128) == green


def test_the_order_of_the_operations() -> None:
    rng = np.random.default_rng(61)
    planes = rng.uniform(-20.0, 280.0, size=(3, 5, 7))
    rgb = colour.to_rgb(planes)
    back = colour.to_ycbcr(rgb)
    c_r, c_b, c_gb, c_gr = (float(value) for value in (RED_FROM_CR, BLUE_FROM_CB, GREEN_FROM_CB, GREEN_FROM_CR))
    k_r, k_g, k_b = float(K_R), float(K_G), float(K_B)
    k_cb, k_cr = float(CB_FROM_BLUE), float(CR_FROM_RED)
    for index in np.ndindex(5, 7):
        luma, cb, cr = (float(planes[(channel, *index)]) for channel in range(3))
        red = luma + c_r * (cr - 128.0)
        blue = luma + c_b * (cb - 128.0)
        green = (luma - c_gb * (cb - 128.0)) - c_gr * (cr - 128.0)
        assert (float(rgb[(*index, 0)]), float(rgb[(*index, 1)]), float(rgb[(*index, 2)])) == (red, green, blue)
        luma_back = (k_r * red + k_g * green) + k_b * blue
        cb_back = 128.0 + k_cb * (blue - luma_back)
        cr_back = 128.0 + k_cr * (red - luma_back)
        assert (float(back[(0, *index)]), float(back[(1, *index)]), float(back[(2, *index)])) == (
            luma_back,
            cb_back,
            cr_back,
        )
    # A grey, with Cb and Cr at 128, is R = G = B = Y exactly.
    grey = np.stack((planes[0], np.full((5, 7), 128.0), np.full((5, 7), 128.0)))
    rgb = colour.to_rgb(grey)
    for channel in range(3):
        np.testing.assert_array_equal(rgb[:, :, channel], planes[0])


@dataclass(frozen=True)
class Tracked:
    """A double the conversion computed, and a bound of its distance from the exact value."""

    value: float
    bound: Fraction

    @staticmethod
    def constant(rational: Fraction) -> Tracked:
        value = float(rational)
        return Tracked(value, abs(Fraction(value) - rational))

    @staticmethod
    def exact(value: float) -> Tracked:
        return Tracked(value, Fraction(0))

    def _rounded(self, value: float, carried: Fraction) -> Tracked:
        # |fl(z) - z| <= U |z| <= U |fl(z)| / (1 - U).
        return Tracked(value, carried + U / (1 - U) * abs(Fraction(value)))

    def __add__(self, other: Tracked) -> Tracked:
        return self._rounded(self.value + other.value, self.bound + other.bound)

    def __sub__(self, other: Tracked) -> Tracked:
        return self._rounded(self.value - other.value, self.bound + other.bound)

    def __mul__(self, other: Tracked) -> Tracked:
        a, b = abs(Fraction(self.value)), abs(Fraction(other.value))
        carried = a * other.bound + b * self.bound + self.bound * other.bound
        return self._rounded(self.value * other.value, carried)


def test_the_round_trip_is_within_its_rounding() -> None:
    # YCbCr to RGB and back, and RGB to YCbCr and back, in the order of docs/math.md, 8: the
    # exact conversions are inverses, so what the computed ones leave of their input is
    # within the running bound of their rounding.
    rng = np.random.default_rng(62)
    planes = rng.uniform(-20.0, 280.0, size=(3, 40))
    rgb = colour.to_rgb(planes[:, np.newaxis, :])
    back = colour.to_ycbcr(rgb)
    centre = Tracked.exact(128.0)
    c_r, c_b = Tracked.constant(RED_FROM_CR), Tracked.constant(BLUE_FROM_CB)
    c_gb, c_gr = Tracked.constant(GREEN_FROM_CB), Tracked.constant(GREEN_FROM_CR)
    k_r, k_g, k_b = Tracked.constant(K_R), Tracked.constant(K_G), Tracked.constant(K_B)
    k_cb, k_cr = Tracked.constant(CB_FROM_BLUE), Tracked.constant(CR_FROM_RED)
    for sample in range(40):
        luma, cb, cr = (Tracked.exact(float(planes[channel, sample])) for channel in range(3))
        red = luma + c_r * (cr - centre)
        blue = luma + c_b * (cb - centre)
        green = (luma - c_gb * (cb - centre)) - c_gr * (cr - centre)
        luma_back = (k_r * red + k_g * green) + k_b * blue
        cb_back = centre + k_cb * (blue - luma_back)
        cr_back = centre + k_cr * (red - luma_back)
        for channel, tracked in enumerate((luma_back, cb_back, cr_back)):
            computed = float(back[channel, 0, sample])
            assert computed == tracked.value
            assert abs(Fraction(computed) - Fraction(float(planes[channel, sample]))) <= tracked.bound
            # And the bound is of the order of the rounding: a few hundred units of 2^-53 of 256.
            assert tracked.bound <= 1024 * U * 256
    pictures = rng.uniform(-20.0, 280.0, size=(1, 40, 3))
    again = colour.to_rgb(colour.to_ycbcr(pictures))
    for sample in range(40):
        red, green, blue = (Tracked.exact(float(pictures[0, sample, channel])) for channel in range(3))
        luma = (k_r * red + k_g * green) + k_b * blue
        cb = centre + k_cb * (blue - luma)
        cr = centre + k_cr * (red - luma)
        red_back = luma + c_r * (cr - centre)
        green_back = (luma - c_gb * (cb - centre)) - c_gr * (cr - centre)
        blue_back = luma + c_b * (cb - centre)
        for channel, tracked in enumerate((red_back, green_back, blue_back)):
            computed = float(again[0, sample, channel])
            assert computed == tracked.value
            assert abs(Fraction(computed) - Fraction(float(pictures[0, sample, channel]))) <= tracked.bound


def test_the_shapes_are_checked() -> None:
    with pytest.raises(ValueError, match="YCbCr planes"):
        colour.to_rgb(np.zeros((5, 7, 3)))
    with pytest.raises(ValueError, match="an RGB picture"):
        colour.to_ycbcr(np.zeros((3, 5, 7)))
