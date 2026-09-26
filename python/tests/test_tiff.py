# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Binary64 output: TIFF files whose samples are the picture's to the last bit.

tifffile, independent of the writer, reads the files back; the samples are compared
as the integers of their bits, so that -0.0 differs from 0.0 and nothing rounds.
"""

import io
import math
from pathlib import Path

import numpy as np
import pytest
import tifffile
from PIL import Image

import synthetic
from unround import decode, tiff


def bits(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.float64).view(np.uint64)


def awkward(shape: tuple[int, ...]) -> np.ndarray:
    """Random binary64 samples, with the values that a careless conversion would change."""
    rng = np.random.default_rng(60)
    samples = rng.uniform(-10.0, 265.0, size=shape)
    special = [-0.0, 0.0, 5e-324, -5e-324, 2.2250738585072014e-308, 1.7976931348623157e308, 1.0 / 3.0, 255.0]
    special.append(math.nextafter(128.0, math.inf))
    flat = samples.reshape(-1)  # a view: the samples are contiguous
    count = min(len(special), flat.size)
    flat[:count] = special[:count]
    return samples


@pytest.mark.parametrize("shape", [(1, 1), (7, 13), (21, 30), (5, 6, 3)])
def test_the_samples_are_written_bit_for_bit(tmp_path: Path, shape: tuple[int, ...]) -> None:
    picture = awkward(shape)
    path = tmp_path / "picture.tif"
    tiff.write_float64(path, picture)
    with tifffile.TiffFile(path) as file:
        page = file.pages[0]
        assert isinstance(page, tifffile.TiffPage)
        assert page.dtype == np.float64
        assert page.compression == 1
        assert page.samplesperpixel == (3 if len(shape) == 3 else 1)
        read = page.asarray()
    assert read.shape == shape
    np.testing.assert_array_equal(bits(read), bits(picture))


def test_a_decoded_picture_is_written_as_it_is(tmp_path: Path) -> None:
    # The output of decode, which is the solver's canvas itself, through the file and back.
    samples = np.clip(synthetic.picture(21, 30, seed=61), 0.0, 255.0).astype(np.uint8)
    stream = io.BytesIO()
    Image.fromarray(samples).save(stream, format="JPEG", quality=20)
    decoded = decode.decode(stream.getvalue(), decode.Settings(method="mmse"))
    path = tmp_path / "decoded.tif"
    tiff.write_float64(path, decoded.picture)
    np.testing.assert_array_equal(bits(tifffile.imread(path)), bits(decoded.picture))


def test_ycbcr_is_declared_as_it_is(tmp_path: Path) -> None:
    # The planes of a colour file, as the solver left them, through the file and back, which
    # says they are JFIF's YCbCr: no subsampling, and JFIF's coefficients.
    samples = np.clip(synthetic.colour_picture(21, 30, seed=62), 0.0, 255.0).astype(np.uint8)
    stream = io.BytesIO()
    Image.fromarray(samples).save(stream, format="JPEG", quality=20, subsampling=2)
    decoded = decode.decode(stream.getvalue(), decode.Settings(method="mmse"))
    planes = np.moveaxis(decoded.planes, 0, -1)
    path = tmp_path / "planes.tif"
    tiff.write_float64(path, planes, ycbcr=True)
    with tifffile.TiffFile(path) as file:
        page = file.pages[0]
        assert isinstance(page, tifffile.TiffPage)
        assert page.photometric == tifffile.PHOTOMETRIC.YCBCR
        assert page.tags["YCbCrSubSampling"].value == (1, 1)
        assert page.tags["YCbCrCoefficients"].value == (299, 1000, 587, 1000, 114, 1000)
        assert page.tags["ReferenceBlackWhite"].value == (0, 1, 255, 1, 128, 1, 255, 1, 128, 1, 255, 1)
        read = page.asarray()
    np.testing.assert_array_equal(bits(read.reshape(planes.shape)), bits(planes))
    with pytest.raises(ValueError, match="three samples"):
        tiff.write_float64(tmp_path / "grey.tif", planes[:, :, 0], ycbcr=True)


def test_other_types_and_shapes_are_refused(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="binary64"):
        tiff.write_float64(tmp_path / "a.tif", np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="height, width"):
        tiff.write_float64(tmp_path / "b.tif", np.zeros((2, 2, 2)))
    with pytest.raises(ValueError, match="empty"):
        tiff.write_float64(tmp_path / "c.tif", np.zeros((0, 3)))
