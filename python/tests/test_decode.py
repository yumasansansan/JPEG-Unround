# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Decoding a greyscale file with each method: the picture's size, and the intervals kept."""

import io

import numpy as np
import pytest
from PIL import Image

import synthetic
from unround import dct, decode, jpegio, model, pdhg, subgradient

METHODS: list[decode.Method] = ["mmse", "tv", "tgv", "subgradient"]


def grey_file(height: int = 21, width: int = 30, quality: int = 30) -> bytes:
    samples = np.clip(synthetic.picture(height, width, seed=50), 0.0, 255.0).astype(np.uint8)
    stream = io.BytesIO()
    Image.fromarray(samples).save(stream, format="JPEG", quality=quality)
    return stream.getvalue()


@pytest.mark.parametrize("method", METHODS)
def test_each_method_keeps_the_intervals(method: decode.Method) -> None:
    data = grey_file()
    settings = decode.Settings(
        method=method, pdhg=pdhg.Options(iterations=30), subgradient=subgradient.Options(iterations=10)
    )
    decoded = decode.decode(data, settings)
    assert decoded.picture.shape == (21, 30)
    assert decoded.coefficients.shape == (3, 4, 8, 8)
    assert model.excess(decoded.problem, decoded.coefficients).max() == 0.0
    assert (decoded.result is None) == (method == "mmse")
    component = jpegio.read(data).components[0]
    lower = (component.coefficients - 0.5) * component.quant_table
    lower[:, :, 0, 0] += 1024.0
    np.testing.assert_array_equal(decoded.problem.lower, lower)
    # The picture is the solver's canvas itself, cut to the picture: no operation between.
    canvas = dct.inverse(decoded.coefficients)
    np.testing.assert_array_equal(decoded.picture.view(np.uint64), canvas[:21, :30].view(np.uint64))


def test_the_mmse_decoder_is_the_centres() -> None:
    decoded = decode.decode(grey_file(), decode.Settings(method="mmse"))
    np.testing.assert_array_equal(decoded.coefficients, decoded.problem.centres)


def test_slack_widens_the_intervals() -> None:
    data = grey_file()
    decoded = decode.decode(data, decode.Settings(method="mmse", slack=0.5))
    component = jpegio.read(data).components[0]
    upper = (component.coefficients + 1.0) * component.quant_table
    upper[:, :, 0, 0] += 1024.0
    np.testing.assert_array_equal(decoded.problem.upper, upper)


def test_colour_files_are_not_decoded_yet() -> None:
    stream = io.BytesIO()
    Image.new("RGB", (16, 16), (200, 30, 90)).save(stream, format="JPEG")
    with pytest.raises(ValueError, match="greyscale"):
        decode.decode(stream.getvalue())
