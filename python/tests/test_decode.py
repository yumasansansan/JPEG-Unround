# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Decoding a file with each method: the picture's size, and the intervals kept."""

import io

import numpy as np
import pytest
from PIL import Image

import synthetic
from unround import colour, dct, decode, frames, jpegio, model, pdhg, subgradient
from unround.model import TGV, TV, DataTerm

METHODS: list[decode.Method] = ["mmse", "tv", "tgv", "subgradient"]
SUBSAMPLINGS = {"4:4:4": 0, "4:2:2": 1, "4:2:0": 2}  # Pillow's names for them
INVARIANT = 1e-4  # in steps: every output's coefficients are within their intervals to this


def grey_file(height: int = 21, width: int = 30, quality: int = 30) -> bytes:
    samples = np.clip(synthetic.picture(height, width, seed=50), 0.0, 255.0).astype(np.uint8)
    stream = io.BytesIO()
    Image.fromarray(samples).save(stream, format="JPEG", quality=quality)
    return stream.getvalue()


def colour_file(subsampling: str, height: int = 21, width: int = 30, quality: int = 30) -> bytes:
    samples = np.clip(synthetic.colour_picture(height, width, seed=51), 0.0, 255.0).astype(np.uint8)
    stream = io.BytesIO()
    Image.fromarray(samples).save(stream, format="JPEG", quality=quality, subsampling=SUBSAMPLINGS[subsampling])
    return stream.getvalue()


@pytest.mark.parametrize("method", METHODS)
def test_each_method_keeps_the_intervals(method: decode.Method) -> None:
    data = grey_file()
    settings = decode.Settings(
        method=method, pdhg=pdhg.Options(iterations=30), subgradient=subgradient.Options(iterations=10)
    )
    decoded = decode.decode(data, settings)
    assert decoded.picture.shape == (21, 30)
    assert decoded.planes.shape == (1, 21, 30)
    assert decoded.color_space is jpegio.ColorSpace.GRAYSCALE
    (coefficients,) = decoded.coefficients
    problem = decoded.frame.channels[0].problem
    assert coefficients.shape == (3, 4, 8, 8)
    assert model.excess(problem, coefficients).max() == 0.0
    assert (decoded.result is None) == (method == "mmse")
    component = jpegio.read(data).components[0]
    lower = (component.coefficients - 0.5) * component.quant_table
    lower[:, :, 0, 0] += 1024.0
    np.testing.assert_array_equal(problem.lower, lower)
    # The picture is the solver's canvas itself, cut to the picture: no operation between.
    canvas = dct.inverse(coefficients)
    np.testing.assert_array_equal(decoded.picture.view(np.uint64), canvas[:21, :30].view(np.uint64))


def test_the_mmse_decoder_is_the_centres() -> None:
    decoded = decode.decode(grey_file(), decode.Settings(method="mmse"))
    np.testing.assert_array_equal(decoded.coefficients[0], decoded.frame.channels[0].problem.centres)


def test_slack_widens_the_intervals() -> None:
    data = grey_file()
    decoded = decode.decode(data, decode.Settings(method="mmse", data=DataTerm(slack=0.5)))
    component = jpegio.read(data).components[0]
    upper = (component.coefficients + 1.0) * component.quant_table
    upper[:, :, 0, 0] += 1024.0
    np.testing.assert_array_equal(decoded.frame.channels[0].problem.upper, upper)


def test_the_settings_reach_the_model() -> None:
    # With the middles as centres, the decoder of the centres is the plain decoder, q Q
    # without rounding; DC's weight is mu / Q^2 times the weight given.
    data = grey_file()
    decoded = decode.decode(data, decode.Settings(method="mmse", data=DataTerm(dc_weight=3.0, centres="midpoint")))
    component = jpegio.read(data).components[0]
    middles = component.coefficients.astype(np.float64) * component.quant_table.astype(np.float64)
    middles[:, :, 0, 0] += 1024.0
    problem = decoded.frame.channels[0].problem
    np.testing.assert_array_equal(problem.centres, middles)
    np.testing.assert_array_equal(decoded.coefficients[0], middles)
    step = float(component.quant_table[0, 0])
    assert problem.weights[0, 0] == (1e-3 / (step * step)) * 3.0


@pytest.mark.parametrize("subsampling", list(SUBSAMPLINGS))
@pytest.mark.parametrize("method", ["mmse", "tv", "tgv"])
def test_colour_files_keep_their_intervals(method: decode.Method, subsampling: str) -> None:
    data = colour_file(subsampling)
    settings = decode.Settings(method=method, pdhg=pdhg.Options(iterations=30))
    decoded = decode.decode(data, settings)
    image = jpegio.read(data)
    assert decoded.color_space is jpegio.ColorSpace.YCBCR
    assert decoded.picture.shape == (21, 30, 3)
    assert decoded.planes.shape == (3, 21, 30)
    frame = decoded.frame
    for index, (component, channel) in enumerate(zip(image.components, frame.channels, strict=True)):
        own = decoded.coefficients[index]
        assert own.shape == component.coefficients.shape
        assert model.excess(channel.problem, own).max() == 0.0
    # The canvas's own coefficients are within their intervals: to rounding, which is far
    # below the invariant (docs/math.md, 1.2); where the solver took a step, the canvas is
    # its proximal step's output.
    canvas = frames.start(frame).canvas if decoded.result is None else decoded.result.primal.canvas
    for excess in frames.excess(frame, canvas):
        assert excess.max() <= INVARIANT
    # The planes are the canvas itself, cut to the picture; the picture is their RGB.
    np.testing.assert_array_equal(decoded.planes.view(np.uint64), canvas[:, :21, :30].view(np.uint64))
    np.testing.assert_array_equal(decoded.picture.view(np.uint64), colour.to_rgb(decoded.planes).view(np.uint64))
    # And the RGB of the whole canvas, converted back, is within its intervals as well.
    back = colour.to_ycbcr(colour.to_rgb(canvas))
    for excess in frames.excess(frame, back):
        assert excess.max() <= INVARIANT


def test_the_frame_of_a_colour_file() -> None:
    # 4:2:0 of 21 x 30: the canvas is 32 x 32, whole MCUs of 16; Y has 3 x 4 blocks, the
    # chroma 2 x 2 of cells of 2 x 2.
    frame = decode.frame_of(jpegio.read(colour_file("4:2:0")))
    assert frame.shape == (32, 32)
    assert [channel.ratio for channel in frame.channels] == [(1, 1), (2, 2), (2, 2)]
    assert [channel.problem.shape for channel in frame.channels] == [(24, 32), (16, 16), (16, 16)]
    assert [frame.free(channel) for channel in frame.channels] == [True, True, True]
    frame = decode.frame_of(jpegio.read(colour_file("4:4:4")))
    assert frame.shape == (24, 32)
    assert [frame.free(channel) for channel in frame.channels] == [False, False, False]


def test_options_of_g_for_each_component() -> None:
    data = colour_file("4:2:0")
    each = (DataTerm(), DataTerm(slack=0.5), DataTerm(centres="midpoint"))
    decoded = decode.decode(data, decode.Settings(method="mmse", data=each))
    image = jpegio.read(data)
    problems = [channel.problem for channel in decoded.frame.channels]
    upper = (image.components[1].coefficients + 1.0) * image.components[1].quant_table
    upper[:, :, 0, 0] += 1024.0
    np.testing.assert_array_equal(problems[1].upper, upper)
    middles = image.components[2].coefficients.astype(np.float64) * image.components[2].quant_table
    middles[:, :, 0, 0] += 1024.0
    np.testing.assert_array_equal(problems[2].centres, middles)
    with pytest.raises(ValueError, match="each of the 3 components"):
        decode.decode(data, decode.Settings(method="mmse", data=each[:2]))


def test_the_channels_are_coupled_or_not_as_the_weights_say() -> None:
    data = colour_file("4:2:0")
    options = pdhg.Options(iterations=20)
    coupled = decode.decode(data, decode.Settings(method="tv", pdhg=options))
    apart = decode.decode(data, decode.Settings(method="tv", tv=TV(coupled=False), pdhg=options))
    weighted = decode.decode(
        data, decode.Settings(method="tgv", tgv=TGV(channel_weights=(1.0, 0.5, 0.5)), pdhg=options)
    )
    assert not np.array_equal(coupled.planes, apart.planes)
    assert weighted.result is not None
    with pytest.raises(ValueError, match="each of the 3 channels"):
        decode.decode(data, decode.Settings(method="tv", tv=TV(channel_weights=(1.0, 1.0)), pdhg=options))


def test_the_subgradient_method_is_for_greyscale_files() -> None:
    with pytest.raises(ValueError, match="greyscale"):
        decode.decode(colour_file("4:4:4"), decode.Settings(method="subgradient"))
