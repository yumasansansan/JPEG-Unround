# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pictures in binary64, written as TIFF with 64-bit IEEE floating-point samples.

The samples go into the file as they are, bit for bit: no conversion, no scaling
(they are in the units of the samples, 0 to 255 for 8-bit JPEG, not normalized to
0-1), and no compression that could change them. The file is baseline TIFF 6.0 in
little-endian byte order, one strip, with SampleFormat 3 (IEEE floating point) and
BitsPerSample 64: libtiff, tifffile and the like read it as it was written.

The three samples of a colour picture are R, G and B, or JFIF's Y, Cb and Cr, which
the file then declares: PhotometricInterpretation YCbCr, without subsampling, with
JFIF's coefficients and full range (docs/math.md, 8).

A classic TIFF addresses at most 4 GiB, which is a little under 2^29 greyscale samples
in binary64.
"""

import struct
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt

__all__ = ["write_float64"]

_SHORT: Final = 3
_LONG: Final = 4
_RATIONAL: Final = 5
_HEADER: Final = 8
_ENTRY: Final = 12
_IFD: Final = 2 + 4  # the count of entries and the next IFD's offset, besides the entries
_LIMIT: Final = 2**32
_GREY: Final = 2  # axes of a greyscale picture: height and width
_COLOUR: Final = 3  # axes of a colour picture, and its channels
_BITS: Final = 64
_IEEE: Final = 3  # SampleFormat: IEEE floating point
_BLACK_IS_ZERO: Final = 1
_RGB: Final = 2
_YCBCR: Final = 6
_JFIF_COEFFICIENTS: Final = (299, 1000, 587, 1000, 114, 1000)  # K_R, K_G and K_B, as rationals
_FULL_RANGE: Final = (0, 1, 255, 1, 128, 1, 255, 1, 128, 1, 255, 1)  # black and white of Y, Cb and Cr


def _entry(tag: int, kind: int, count: int, value: bytes) -> bytes:
    """An IFD entry whose value, of at most four bytes, is held in the entry itself."""
    return struct.pack("<HHI", tag, kind, count) + value.ljust(4, b"\0")


def write_float64(path: str | Path, picture: npt.ArrayLike, *, ycbcr: bool = False) -> None:
    """Writes a picture of binary64 samples, (height, width) or (height, width, 3), exactly.

    The picture's values are the file's samples to the last bit; a picture of another type
    is refused rather than converted. The three samples of a pixel are R, G and B, or with
    ycbcr, JFIF's Y, Cb and Cr.
    """
    samples = np.asarray(picture)
    if samples.dtype != np.float64:
        message = f"the samples are binary64, not {samples.dtype}: they are not converted"
        raise TypeError(message)
    if samples.ndim == _GREY:
        samples = samples[:, :, np.newaxis]
    if samples.ndim != _COLOUR or samples.shape[2] not in {1, _COLOUR} or 0 in samples.shape:
        message = f"a picture is (height, width) or (height, width, 3), and not empty: {samples.shape}"
        raise ValueError(message)
    height, width, channels = (int(size) for size in samples.shape)
    if ycbcr and channels != _COLOUR:
        message = f"Y, Cb and Cr are three samples to a pixel, not {channels}"
        raise ValueError(message)
    data = np.ascontiguousarray(samples, dtype="<f8").tobytes()

    # The layout: the header, the image data, the values that do not fit in an entry, and
    # the IFD. Entries are in ascending order of their tags, as TIFF requires.
    bits_offset = _HEADER + len(data)
    bits = struct.pack(f"<{channels}H", *([_BITS] * channels)) if channels > 1 else b""
    format_offset = bits_offset + len(bits)
    formats = struct.pack(f"<{channels}H", *([_IEEE] * channels)) if channels > 1 else b""
    resolution_offset = format_offset + len(formats)
    resolution = struct.pack("<II", 1, 1)
    coefficients_offset = resolution_offset + len(resolution)
    coefficients = struct.pack("<6I", *_JFIF_COEFFICIENTS) if ycbcr else b""
    reference_offset = coefficients_offset + len(coefficients)
    reference = struct.pack("<12I", *_FULL_RANGE) if ycbcr else b""
    values = bits + formats + resolution + coefficients + reference
    ifd_offset = reference_offset + len(reference)
    ifd_offset += ifd_offset % 2  # an IFD starts on a word boundary
    photometric = _YCBCR if ycbcr else (_RGB if channels == _COLOUR else _BLACK_IS_ZERO)
    entries = [
        _entry(256, _LONG, 1, struct.pack("<I", width)),  # ImageWidth
        _entry(257, _LONG, 1, struct.pack("<I", height)),  # ImageLength
        _entry(258, _SHORT, channels, struct.pack("<I", bits_offset) if bits else struct.pack("<H", _BITS)),
        _entry(259, _SHORT, 1, struct.pack("<H", 1)),  # Compression: none
        _entry(262, _SHORT, 1, struct.pack("<H", photometric)),  # PhotometricInterpretation
        _entry(273, _LONG, 1, struct.pack("<I", _HEADER)),  # StripOffsets
        _entry(277, _SHORT, 1, struct.pack("<H", channels)),  # SamplesPerPixel
        _entry(278, _LONG, 1, struct.pack("<I", height)),  # RowsPerStrip: one strip
        _entry(279, _LONG, 1, struct.pack("<I", len(data))),  # StripByteCounts
        _entry(282, _RATIONAL, 1, struct.pack("<I", resolution_offset)),  # XResolution: 1/1
        _entry(283, _RATIONAL, 1, struct.pack("<I", resolution_offset)),  # YResolution: 1/1
        _entry(284, _SHORT, 1, struct.pack("<H", 1)),  # PlanarConfiguration: chunky
        _entry(296, _SHORT, 1, struct.pack("<H", 1)),  # ResolutionUnit: none
        _entry(339, _SHORT, channels, struct.pack("<I", format_offset) if formats else struct.pack("<H", _IEEE)),
    ]
    if ycbcr:
        entries += [
            _entry(529, _RATIONAL, 3, struct.pack("<I", coefficients_offset)),  # YCbCrCoefficients: JFIF's
            _entry(530, _SHORT, 2, struct.pack("<HH", 1, 1)),  # YCbCrSubSampling: none
            _entry(531, _SHORT, 1, struct.pack("<H", 1)),  # YCbCrPositioning: centred
            _entry(532, _RATIONAL, 6, struct.pack("<I", reference_offset)),  # ReferenceBlackWhite: full range
        ]
    if ifd_offset + _IFD + _ENTRY * len(entries) > _LIMIT:
        message = f"a picture of {height} x {width} x {channels} binary64 samples does not fit a classic TIFF"
        raise ValueError(message)
    ifd = struct.pack("<H", len(entries)) + b"".join(entries) + struct.pack("<I", 0)
    header = b"II" + struct.pack("<HI", 42, ifd_offset)
    padding = b"\0" * (ifd_offset - _HEADER - len(data) - len(values))
    Path(path).write_bytes(header + data + values + padding + ifd)
