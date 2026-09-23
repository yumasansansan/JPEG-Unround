# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests of the C layer as Python reads it.

Files are written by Pillow (conda-forge's libjpeg-turbo) and read both through
the C layer and through jpeglib, which carries libjpeg builds of its own: the
coefficients, the quantization tables and the sampling factors have to agree
exactly. The planes have to be the inverse DCT of the coefficients to within one
level. Then the metadata, the errors, threads, and arbitrary bytes.
"""

import ctypes
import io
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple

import jpeglib
import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given
from hypothesis import strategies as st
from PIL import Image as PILImage
from scipy.fft import idctn

from unround import jpegio


def picture(seed: int, height: int, width: int, channels: int) -> npt.NDArray[np.uint8]:
    """Waves, steps and noise: something between an illustration and a photograph."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width].astype(np.float64)
    layers = []
    for channel in range(channels):
        phase = rng.uniform(0.0, 2.0 * np.pi)
        value = 128.0 + 70.0 * np.sin(x / (9.0 + channel) + phase) * np.cos(y / 7.0)
        value += 60.0 * (((x // 13) + (y // 11)) % 3 == 0)
        value += rng.integers(-24, 25, size=(height, width))
        layers.append(np.clip(value, 0, 255))
    stacked = np.stack(layers, axis=-1).astype(np.uint8)
    return stacked[..., 0] if channels == 1 else stacked


def encode(pixels: npt.NDArray[np.uint8], **options: object) -> bytes:
    buffer = io.BytesIO()
    PILImage.fromarray(pixels).save(buffer, format="JPEG", **options)
    return buffer.getvalue()


def test_structures_have_the_c_layout() -> None:
    assert ctypes.sizeof(jpegio._Options) == 16
    assert ctypes.sizeof(jpegio._Component) == 168
    assert jpegio._Component.quant_table.offset == 32
    assert jpegio._Component.coefficients.offset == 160
    assert ctypes.sizeof(jpegio._Image) == 728
    assert jpegio._Image.components.offset == 40
    assert jpegio._Image.icc_profile.offset == 712
    assert ctypes.sizeof(jpegio._Plane) == 24
    assert ctypes.sizeof(jpegio._Planes) == 104
    assert jpegio.abi_version() == jpegio.ABI_VERSION
    assert jpegio.libjpeg_version() == "libjpeg-turbo 3.2.0 (libjpeg API 62)"


class Case(NamedTuple):
    """A file for Pillow to write. Its subsampling 0, 1 and 2 are 4:4:4, 4:2:2 and 4:2:0."""

    mode: str
    subsampling: int
    quality: int
    progressive: bool
    height: int
    width: int

    def __str__(self) -> str:
        coding = "progressive" if self.progressive else "sequential"
        return f"{self.mode}-{self.subsampling}-q{self.quality}-{coding}-{self.height}x{self.width}"


CASES = [
    Case(mode, subsampling, quality, progressive, height, width)
    for mode, subsampling in [("L", 0), ("RGB", 0), ("RGB", 1), ("RGB", 2)]
    for quality in (10, 50, 90)
    for progressive in (False, True)
    for height, width in [(1, 1), (8, 8), (21, 37), (48, 33)]
]


@pytest.mark.parametrize("case", CASES, ids=str)
def test_coefficients_match_jpeglib(tmp_path: Path, case: Case) -> None:
    channels = 1 if case.mode == "L" else 3
    seed = zlib.crc32(str(case).encode())
    data = encode(
        picture(seed, case.height, case.width, channels),
        quality=case.quality,
        subsampling=case.subsampling,
        progressive=case.progressive,
    )
    path = tmp_path / "file.jpg"
    path.write_bytes(data)

    ours = jpegio.read(data)
    theirs = jpeglib.read_dct(str(path))
    assert (ours.width, ours.height) == (theirs.width, theirs.height)
    assert ours.progressive == case.progressive == bool(theirs.progressive_mode)
    assert ours.color_space == (jpegio.ColorSpace.GRAYSCALE if channels == 1 else jpegio.ColorSpace.YCBCR)
    assert ours.warnings == 0
    assert ours.warning == ""
    their_components = [theirs.Y] if channels == 1 else [theirs.Y, theirs.Cb, theirs.Cr]
    assert len(ours.components) == len(their_components)
    for component, coefficients, slot, (v, h) in zip(
        ours.components, their_components, theirs.quant_tbl_no, theirs.samp_factor, strict=True
    ):
        assert component.coefficients.dtype == np.int16
        np.testing.assert_array_equal(component.coefficients, coefficients)
        assert component.quant_table_slot == slot
        np.testing.assert_array_equal(component.quant_table, theirs.qt[slot])
        assert (component.h_samp_factor, component.v_samp_factor) == (h, v)
        assert component.height_in_blocks == -(-component.height // 8)
        assert component.width_in_blocks == -(-component.width // 8)


@pytest.mark.parametrize("subsampling", [0, 1, 2])
def test_planes_are_the_inverse_dct_of_the_coefficients(subsampling: int) -> None:
    data = encode(picture(subsampling, 45, 70, 3), quality=40, subsampling=subsampling)
    image = jpegio.read(data)
    planes = jpegio.decode_planes(data)
    assert planes.warning == ""
    assert len(planes.planes) == len(image.components)
    for component, plane in zip(image.components, planes.planes, strict=True):
        rows, columns = component.height_in_blocks, component.width_in_blocks
        assert plane.shape == (rows * 8, columns * 8)
        assert plane.dtype == np.uint8
        dequantized = component.coefficients.astype(np.float64) * component.quant_table
        expected = np.clip(np.rint(idctn(dequantized, axes=(2, 3), norm="ortho") + 128.0), 0, 255)
        blocks = plane.reshape(rows, 8, columns, 8).transpose(0, 2, 1, 3).astype(np.float64)
        assert np.abs(blocks - expected).max() <= 1


def owner(array: npt.NDArray[np.generic]) -> object:
    """The object at the end of an array's chain of bases: what holds its memory."""
    base: object = array
    while isinstance(base, np.ndarray) and base.base is not None:
        base = base.base
    return base


def test_arrays_are_copies_and_read_only() -> None:
    # The C layer's memory is freed before read() and decode_planes() return, so
    # every array has to hold memory of its own; a view of the C layer's would
    # point at nothing. A plane whose stride is its width is where NumPy would
    # have made a view of it without being told otherwise.
    data = encode(picture(3, 16, 16, 3))
    component = jpegio.read(data).components[0]
    plane = jpegio.decode_planes(data).planes[0]
    for array in (component.coefficients, component.quant_table, plane):
        root = owner(array)
        assert isinstance(root, np.ndarray)
        assert root.flags.owndata
        with pytest.raises(ValueError, match="read-only"):
            array[0] = 0


def test_exif_orientation_and_icc_profile() -> None:
    exif = PILImage.Exif()
    exif[0x0112] = 6
    profile = bytes(np.random.default_rng(4).integers(0, 256, size=140_000, dtype=np.uint8))
    image = jpegio.read(encode(picture(5, 24, 32, 3), exif=exif.tobytes(), icc_profile=profile))
    assert image.exif_orientation == 6
    assert image.icc_profile == profile
    plain = jpegio.read(encode(picture(5, 24, 32, 3)))
    assert plain.exif_orientation == 0
    assert plain.icc_profile is None


def test_what_cannot_be_read_raises() -> None:
    with pytest.raises(jpegio.JpegioError) as garbage:
        jpegio.read(b"this is not a JPEG file")
    assert garbage.value.kind is jpegio.ErrorKind.DECODE
    assert "Not a JPEG file" in garbage.value.message
    with pytest.raises(jpegio.JpegioError) as empty:
        jpegio.decode_planes(b"")
    assert empty.value.kind is jpegio.ErrorKind.DECODE

    buffer = io.BytesIO()
    PILImage.new("CMYK", (16, 16), (10, 20, 30, 40)).save(buffer, format="JPEG")
    cmyk = buffer.getvalue()
    with pytest.raises(jpegio.JpegioError) as unsupported:
        jpegio.read(cmyk)
    assert unsupported.value.kind is jpegio.ErrorKind.UNSUPPORTED

    data = encode(picture(6, 30, 40, 3))
    with pytest.raises(jpegio.JpegioError) as limited:
        jpegio.read(data, max_pixels=30 * 40 - 1)
    assert limited.value.kind is jpegio.ErrorKind.LIMIT
    assert jpegio.read(data, max_pixels=30 * 40).width == 40

    progressive = encode(picture(7, 30, 40, 3), progressive=True)
    with pytest.raises(jpegio.JpegioError) as scans:
        jpegio.read(progressive, max_scans=1)
    assert scans.value.kind is jpegio.ErrorKind.LIMIT


def test_a_cut_file_warns_or_fails() -> None:
    data = encode(picture(8, 64, 64, 3), quality=95)
    cut = data[: len(data) * 3 // 4]
    image = jpegio.read(cut)
    assert image.warnings > 0
    assert "Premature end" in image.warning
    assert "Premature end" in jpegio.decode_planes(cut).warning
    with pytest.raises(jpegio.JpegioError) as strict:
        jpegio.read(cut, warnings_are_errors=True)
    assert strict.value.kind is jpegio.ErrorKind.DECODE


def test_limits_are_checked_before_they_reach_c() -> None:
    data = encode(picture(9, 8, 8, 1))
    with pytest.raises(ValueError, match="max_pixels"):
        jpegio.read(data, max_pixels=-1)
    with pytest.raises(ValueError, match="max_pixels"):
        jpegio.read(data, max_pixels=2**64)
    with pytest.raises(ValueError, match="max_scans"):
        jpegio.read(data, max_scans=2**31)


def test_buffers_of_every_kind_are_read() -> None:
    data = encode(picture(10, 16, 24, 3))
    expected = jpegio.read(data).components[0].coefficients
    for buffer in (bytearray(data), memoryview(data)):
        np.testing.assert_array_equal(jpegio.read(buffer).components[0].coefficients, expected)


def test_threads_read_at_once() -> None:
    files = [encode(picture(11 + index, 40 + 8 * index, 56, 3), subsampling=index % 3) for index in range(4)]
    expected = [jpegio.read(data) for data in files]

    def read(index: int) -> bool:
        image = jpegio.read(files[index % len(files)])
        reference = expected[index % len(files)]
        return all(
            np.array_equal(ours.coefficients, theirs.coefficients)
            for ours, theirs in zip(image.components, reference.components, strict=True)
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert all(pool.map(read, range(64)))


VALID = encode(picture(12, 24, 40, 3), quality=60, progressive=True)


# What a file can make the C layer say: never a wrong argument or a lack of memory.
REFUSALS = {jpegio.ErrorKind.DECODE, jpegio.ErrorKind.UNSUPPORTED, jpegio.ErrorKind.LIMIT}


@given(st.binary(max_size=2048))
def test_any_bytes_give_an_image_or_an_error(data: bytes) -> None:
    kind = None
    try:
        image = jpegio.read(data, max_pixels=1 << 20)
    except jpegio.JpegioError as error:
        kind = error.kind
    else:
        assert image.width * image.height <= 1 << 20
    assert kind is None or kind in REFUSALS


@given(st.lists(st.tuples(st.integers(0, len(VALID) - 1), st.integers(0, 255)), min_size=1, max_size=8))
def test_damaged_files_give_an_image_or_an_error(changes: list[tuple[int, int]]) -> None:
    data = bytearray(VALID)
    for position, value in changes:
        data[position] = value
    kind = None
    try:
        image = jpegio.read(data, max_pixels=1 << 20)
        planes = jpegio.decode_planes(data, max_pixels=1 << 20)
    except jpegio.JpegioError as error:
        kind = error.kind
    else:
        for component, plane in zip(image.components, planes.planes, strict=True):
            assert plane.shape == (component.height_in_blocks * 8, component.width_in_blocks * 8)
    assert kind is None or kind in REFUSALS
