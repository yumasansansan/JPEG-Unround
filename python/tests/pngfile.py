# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""PNG files read with zlib alone, from the specification, for the tests.

What the writers here write: gray or RGB, 8 or 16 bits, not interlaced, with an ICC
profile or without. Every chunk's CRC is checked, the rows are unfiltered with the
five filters of the specification, and the samples come back as they are, 16-bit
ones as the integers of their two bytes, most significant first.
"""

import struct
import zlib
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

SIGNATURE = b"\x89PNG\r\n\x1a\n"
_CHANNELS = {0: 1, 2: 3}  # colour types: gray, RGB


@dataclass(frozen=True)
class Png:
    """A PNG file's samples, (height, width, channels), its depth, and its ICC profile."""

    samples: npt.NDArray[np.int64]
    bits: int
    icc_profile: bytes | None
    icc_name: bytes | None


def _paeth(a: int, b: int, c: int) -> int:
    estimate = a + b - c
    distances = (abs(estimate - a), abs(estimate - b), abs(estimate - c))
    if distances[0] <= distances[1] and distances[0] <= distances[2]:
        return a
    return b if distances[1] <= distances[2] else c


def _unfiltered(kind: int, line: list[int], previous: list[int], step: int) -> list[int]:
    row = [0] * len(line)
    for index, value in enumerate(line):
        left = row[index - step] if index >= step else 0
        up = previous[index]
        corner = previous[index - step] if index >= step else 0
        predictor = [0, left, up, (left + up) // 2, _paeth(left, up, corner)][kind]
        row[index] = (value + predictor) % 256
    return row


def read(data: bytes) -> Png:
    """The file's samples and profile; an AssertionError where it is not such a file."""
    assert data[:8] == SIGNATURE
    at, header, compressed, profile, name = 8, None, [], None, None
    while at < len(data):
        (length,) = struct.unpack(">I", data[at : at + 4])
        kind, body = data[at + 4 : at + 8], data[at + 8 : at + 8 + length]
        (crc,) = struct.unpack(">I", data[at + 8 + length : at + 12 + length])
        assert zlib.crc32(kind + body) == crc, kind
        if kind == b"IHDR":
            header = struct.unpack(">IIBBBBB", body)
        elif kind == b"iCCP":
            name, rest = body.split(b"\0", 1)
            assert rest[0] == 0  # deflate
            profile = zlib.decompress(rest[1:])
        elif kind == b"IDAT":
            compressed.append(body)
        at += 12 + length
        if kind == b"IEND":
            break
    assert header is not None
    assert at == len(data)
    width, height, bits, colour, method, filtering, interlace = header
    assert (method, filtering, interlace) == (0, 0, 0)
    assert bits in {8, 16}
    channels = _CHANNELS[colour]
    step = channels * bits // 8
    stride = width * step
    raw = zlib.decompress(b"".join(compressed))
    assert len(raw) == height * (stride + 1)
    rows, previous = [], [0] * stride
    for y in range(height):
        start = y * (stride + 1)
        previous = _unfiltered(raw[start], list(raw[start + 1 : start + 1 + stride]), previous, step)
        rows.append(previous)
    values = np.array(rows, dtype=np.int64).reshape(height, width * channels * bits // 8)
    if bits == 16:
        values = values[:, 0::2] * 256 + values[:, 1::2]
    return Png(values.reshape(height, width, channels), bits, profile, name)
