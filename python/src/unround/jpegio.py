# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""The C layer (unround/jpegio.h) for Python, through ctypes.

The C layer reads what the reconstruction needs from a JPEG file -- the quantized
DCT coefficients of every component, the quantization table each was quantized
with, the sampling factors, the ICC profile and the EXIF orientation -- and
decodes the component planes as libjpeg's standard decoder does, before
upsampling and color conversion. Here the results are copied into NumPy arrays,
which are read-only, and the C layer's memory is freed before a function returns.
Its check of an ICC profile, libpng's for the iCCP chunk of PNG, is here too; its
writing of PNG files is the reference implementation's (unround.native.png_bytes).

The C layer is in two libraries: its own, which CMake builds with
-DUNROUND_WITH_PYTHON=ON into python/ of the build directory, and the reference
implementation's (unround.native), which passes its functions on under names of its
own. The one used is the one at the path in the environment variable
UNROUND_JPEGIO_LIBRARY, or else UNROUND_LIBRARY, or else the one in unround/_native,
where a wheel carries it. ctypes releases the GIL while the library reads, so
threads read files at once.
"""

import ctypes
import enum
import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt

from unround import _loading

__all__ = [
    "ABI_VERSION",
    "ColorSpace",
    "Component",
    "ErrorKind",
    "Image",
    "JpegioError",
    "LibraryNotFoundError",
    "Planes",
    "abi_version",
    "check_icc",
    "decode_planes",
    "libjpeg_version",
    "libpng_version",
    "read",
]

ABI_VERSION: Final = 2
"""UNROUND_JPEGIO_ABI_VERSION of the C layer whose structures this module mirrors."""

_MAX_COMPONENTS: Final = 4
_BLOCK_SIZE: Final = 64
_MESSAGE_SIZE: Final = 256
_UINT64_MAX: Final = 2**64 - 1
_INT32_MAX: Final = 2**31 - 1


class _Options(ctypes.Structure):
    _fields_ = (
        ("max_pixels", ctypes.c_uint64),
        ("max_scans", ctypes.c_int32),
        ("warnings_are_errors", ctypes.c_int32),
    )


class _Component(ctypes.Structure):
    _fields_ = (
        ("id", ctypes.c_int32),
        ("h_samp_factor", ctypes.c_int32),
        ("v_samp_factor", ctypes.c_int32),
        ("quant_table_slot", ctypes.c_int32),
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("width_in_blocks", ctypes.c_uint32),
        ("height_in_blocks", ctypes.c_uint32),
        ("quant_table", ctypes.c_uint16 * _BLOCK_SIZE),
        ("coefficients", ctypes.POINTER(ctypes.c_int16)),
    )


class _Image(ctypes.Structure):
    _fields_ = (
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("color_space", ctypes.c_int32),
        ("num_components", ctypes.c_int32),
        ("max_h_samp_factor", ctypes.c_int32),
        ("max_v_samp_factor", ctypes.c_int32),
        ("progressive", ctypes.c_int32),
        ("arithmetic", ctypes.c_int32),
        ("exif_orientation", ctypes.c_int32),
        ("warnings", ctypes.c_int32),
        ("components", _Component * _MAX_COMPONENTS),
        ("icc_profile", ctypes.POINTER(ctypes.c_uint8)),
        ("icc_profile_size", ctypes.c_uint64),
    )


class _Plane(ctypes.Structure):
    _fields_ = (
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("stride", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("samples", ctypes.POINTER(ctypes.c_uint8)),
    )


class _Planes(ctypes.Structure):
    _fields_ = (
        ("num_components", ctypes.c_int32),
        ("reserved", ctypes.c_int32),
        ("planes", _Plane * _MAX_COMPONENTS),
    )


# The sizes the C layer's structures have on the 64-bit systems JPEG-Unround
# supports; the Rust implementation asserts the same numbers.
_SIZES: Final = ((_Options, 16), (_Component, 168), (_Image, 728), (_Plane, 24), (_Planes, 104))


class ErrorKind(enum.IntEnum):
    """What went wrong, as unround_jpegio_status says it."""

    ARGUMENT = 1
    DECODE = 2
    UNSUPPORTED = 3
    LIMIT = 4
    MEMORY = 5
    ENCODE = 6


class ColorSpace(enum.IntEnum):
    """The color space of the components, as the file declares it."""

    GRAYSCALE = 1
    YCBCR = 2
    RGB = 3


class JpegioError(Exception):
    """A JPEG file the C layer could not read, or would not."""

    def __init__(self, kind: ErrorKind, message: str) -> None:
        super().__init__(f"{message} ({kind.name.lower()})")
        self.kind = kind
        self.message = message


class LibraryNotFoundError(RuntimeError):
    """The shared C layer is not where this module looks for it."""


@dataclass(frozen=True, slots=True, eq=False)
class Component:
    """One component of a JPEG file.

    quant_table is the 8x8 table the component was quantized with, and
    coefficients holds its blocks, block rows from the top and blocks from the
    left, each 8x8 in natural order: element [by, bx, v, u] is the coefficient of
    vertical frequency v and horizontal frequency u, and times quant_table[v, u]
    it is within half a step of the orthonormal DCT-II of the encoder's samples,
    level-shifted by 128. The dummy blocks that complete an MCU are not included.
    """

    id: int
    h_samp_factor: int
    v_samp_factor: int
    quant_table_slot: int
    width: int
    height: int
    quant_table: npt.NDArray[np.uint16]
    coefficients: npt.NDArray[np.int16]

    @property
    def width_in_blocks(self) -> int:
        """ceil(width / 8): the blocks that carry image data."""
        return int(self.coefficients.shape[1])

    @property
    def height_in_blocks(self) -> int:
        """ceil(height / 8)."""
        return int(self.coefficients.shape[0])


@dataclass(frozen=True, slots=True, eq=False)
class Image:
    """What read() returns."""

    width: int
    height: int
    color_space: ColorSpace
    max_h_samp_factor: int
    max_v_samp_factor: int
    progressive: bool
    arithmetic: bool
    exif_orientation: int
    """1 to 8 from the EXIF APP1 marker, or 0 when there is none."""
    warnings: int
    """Corrupt-data warnings libjpeg reported while reading."""
    warning: str
    """The first of those warnings, or an empty string."""
    components: tuple[Component, ...]
    icc_profile: bytes | None


@dataclass(frozen=True, slots=True, eq=False)
class Planes:
    """What decode_planes() returns: one array of 8-bit samples per component.

    Each plane is the inverse DCT of its component's coefficients with libjpeg's
    accurate integer method, level-shifted and clamped to 0-255, without block
    smoothing, upsampling or color conversion, and padded to whole blocks:
    height_in_blocks * 8 rows of width_in_blocks * 8 samples.
    """

    planes: tuple[npt.NDArray[np.uint8], ...]
    warning: str
    """The first corrupt-data warning libjpeg reported while decoding, or an empty string."""


def _check_layout() -> None:
    for structure, size in _SIZES:
        if ctypes.sizeof(structure) != size:
            message = f"{structure.__name__} is {ctypes.sizeof(structure)} bytes, and the C layer's is {size}"
            raise RuntimeError(message)


# The libraries that hold the C layer, where they are looked for in this order: its
# own, and the reference implementation's, which passes its functions on under names
# of its own.
_SOURCES: Final = (
    ("UNROUND_JPEGIO_LIBRARY", _loading.C_LAYER, "unround_jpegio_"),
    ("UNROUND_LIBRARY", _loading.RUST, "unround_jpeg_"),
)


def _find() -> tuple[Path, str]:
    for variable, _, prefix in _SOURCES:
        path = _loading.configured(variable)
        if path is not None:
            if not path.is_file():
                message = f"{variable} names {path}, where there is no file"
                raise LibraryNotFoundError(message)
            return path, prefix
    for _, stem, prefix in _SOURCES:
        path = _loading.bundled(stem)
        if path is not None:
            return path, prefix
    message = (
        "no library holds the C layer: build the reference implementation's with cargo build --release "
        "-p unround-capi in rust/ (it is written into rust/target/release) and set UNROUND_LIBRARY to it, or "
        "the C layer's own with cmake --preset release -DUNROUND_WITH_PYTHON=ON (into build/release/python) "
        "and set UNROUND_JPEGIO_LIBRARY to it"
    )
    raise LibraryNotFoundError(message)


@functools.cache
def _library() -> tuple[ctypes.CDLL, str]:
    """The library that holds the C layer, and the prefix of the names of its functions."""
    _check_layout()
    path, prefix = _find()
    library = ctypes.CDLL(str(path))
    message = (ctypes.c_char_p, ctypes.c_size_t)
    signatures: tuple[tuple[str, object, tuple[object, ...]], ...] = (
        ("abi_version", ctypes.c_int32, ()),
        ("libjpeg_version", ctypes.c_char_p, ()),
        ("libpng_version", ctypes.c_char_p, ()),
        ("check_icc", ctypes.c_int32, (ctypes.c_char_p, ctypes.c_uint64, ctypes.c_int32, *message)),
        (
            "read",
            ctypes.c_int32,
            (ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(_Options), ctypes.POINTER(_Image), *message),
        ),
        ("image_free", None, (ctypes.POINTER(_Image),)),
        (
            "decode_planes",
            ctypes.c_int32,
            (ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(_Options), ctypes.POINTER(_Planes), *message),
        ),
        ("planes_free", None, (ctypes.POINTER(_Planes),)),
    )
    for name, result, arguments in signatures:
        function = getattr(library, prefix + name)
        function.argtypes = arguments
        function.restype = result
    version = int(getattr(library, prefix + "abi_version")())
    if version != ABI_VERSION:
        message_text = f"{path} holds the C layer of ABI version {version}, and this module is of version {ABI_VERSION}"
        raise LibraryNotFoundError(message_text)
    return library, prefix


def _call(name: str, *arguments: object) -> int:
    """A function of the C layer, by its name without the prefix: its status, or its value."""
    library, prefix = _library()
    return int(getattr(library, prefix + name)(*arguments))


def _free(name: str, *arguments: object) -> None:
    library, prefix = _library()
    getattr(library, prefix + name)(*arguments)


def abi_version() -> int:
    """UNROUND_JPEGIO_ABI_VERSION of the library that is loaded."""
    return _call("abi_version")


def libjpeg_version() -> str:
    """The name and version of the libjpeg the C layer is built with."""
    library, prefix = _library()
    version: bytes = getattr(library, prefix + "libjpeg_version")()
    return version.decode("ascii")


def libpng_version() -> str:
    """The versions of the libpng and the zlib the C layer is built with."""
    library, prefix = _library()
    version: bytes = getattr(library, prefix + "libpng_version")()
    return version.decode("ascii")


def check_icc(profile: bytes, channels: int) -> str | None:
    """Why an ICC profile does not go with a picture of that many channels (1 gray, 3 RGB), or None.

    The checks are those libpng makes of the profile of an iCCP chunk when it reads one, and drops the
    profile for; the writers of TIFF and PNG leave out a profile that fails them.
    """
    if not 0 <= channels <= _INT32_MAX:
        message = f"channels is {channels}, not 0 to 2**31 - 1"
        raise ValueError(message)
    reason = ctypes.create_string_buffer(_MESSAGE_SIZE)
    accepted = _call("check_icc", profile, len(profile), channels, reason, _MESSAGE_SIZE)
    return None if accepted == 1 else reason.value.decode("utf-8", "replace")


def _options(max_pixels: int, max_scans: int, *, warnings_are_errors: bool) -> _Options:
    if not 0 <= max_pixels <= _UINT64_MAX:
        message = f"max_pixels is {max_pixels}, not 0 to 2**64 - 1"
        raise ValueError(message)
    if not 0 <= max_scans <= _INT32_MAX:
        message = f"max_scans is {max_scans}, not 0 to 2**31 - 1"
        raise ValueError(message)
    return _Options(max_pixels, max_scans, 1 if warnings_are_errors else 0)


def _raise_for(status: int, message: str) -> None:
    if status == 0:
        return
    if status in ErrorKind:
        raise JpegioError(ErrorKind(status), message)
    text = f"the C layer returned status {status}: {message}"
    raise RuntimeError(text)


def _read_only[T: np.generic](array: npt.NDArray[T]) -> npt.NDArray[T]:
    array.flags.writeable = False
    return array


def _component(raw: _Component) -> Component:
    height_in_blocks = int(raw.height_in_blocks)
    width_in_blocks = int(raw.width_in_blocks)
    count = height_in_blocks * width_in_blocks * _BLOCK_SIZE
    # Copies: the C layer's memory is freed once the image is made.
    coefficients = np.array(np.ctypeslib.as_array(raw.coefficients, shape=(count,)), dtype=np.int16, copy=True)
    quant_table = np.array(np.ctypeslib.as_array(raw.quant_table), dtype=np.uint16, copy=True)
    return Component(
        id=int(raw.id),
        h_samp_factor=int(raw.h_samp_factor),
        v_samp_factor=int(raw.v_samp_factor),
        quant_table_slot=int(raw.quant_table_slot),
        width=int(raw.width),
        height=int(raw.height),
        quant_table=_read_only(quant_table.reshape(8, 8)),
        coefficients=_read_only(coefficients.reshape(height_in_blocks, width_in_blocks, 8, 8)),
    )


def read(
    data: bytes | bytearray | memoryview,
    *,
    max_pixels: int = 0,
    max_scans: int = 0,
    warnings_are_errors: bool = False,
) -> Image:
    """Read the coefficients and the metadata of a JPEG file in memory.

    max_pixels and max_scans limit what the file may declare (0 means the C
    layer's defaults, 1 << 28 pixels and 500 scans), and warnings_are_errors
    makes a corrupt-data warning fail the read. A file that cannot be read
    raises JpegioError.
    """
    buffer = bytes(data)
    options = _options(max_pixels, max_scans, warnings_are_errors=warnings_are_errors)
    raw = _Image()
    message = ctypes.create_string_buffer(_MESSAGE_SIZE)
    status = _call("read", buffer, len(buffer), ctypes.byref(options), ctypes.byref(raw), message, _MESSAGE_SIZE)
    text = message.value.decode("utf-8", "replace")
    _raise_for(status, text)
    try:
        components = tuple(_component(raw.components[index]) for index in range(int(raw.num_components)))
        icc_profile = ctypes.string_at(raw.icc_profile, int(raw.icc_profile_size)) if raw.icc_profile else None
        return Image(
            width=int(raw.width),
            height=int(raw.height),
            color_space=ColorSpace(int(raw.color_space)),
            max_h_samp_factor=int(raw.max_h_samp_factor),
            max_v_samp_factor=int(raw.max_v_samp_factor),
            progressive=bool(raw.progressive),
            arithmetic=bool(raw.arithmetic),
            exif_orientation=int(raw.exif_orientation),
            warnings=int(raw.warnings),
            warning=text,
            components=components,
            icc_profile=icc_profile,
        )
    finally:
        _free("image_free", ctypes.byref(raw))


def decode_planes(
    data: bytes | bytearray | memoryview,
    *,
    max_pixels: int = 0,
    max_scans: int = 0,
    warnings_are_errors: bool = False,
) -> Planes:
    """Decode the component planes of a JPEG file in memory, with the limits of read()."""
    buffer = bytes(data)
    options = _options(max_pixels, max_scans, warnings_are_errors=warnings_are_errors)
    raw = _Planes()
    message = ctypes.create_string_buffer(_MESSAGE_SIZE)
    status = _call(
        "decode_planes", buffer, len(buffer), ctypes.byref(options), ctypes.byref(raw), message, _MESSAGE_SIZE
    )
    text = message.value.decode("utf-8", "replace")
    _raise_for(status, text)
    try:
        planes = []
        for index in range(int(raw.num_components)):
            plane = raw.planes[index]
            height, width, stride = int(plane.height), int(plane.width), int(plane.stride)
            samples = np.ctypeslib.as_array(plane.samples, shape=(height * stride,)).reshape(height, stride)
            # A copy, always: the C layer's memory is freed below, and a view of
            # it -- which is what ascontiguousarray would return for a plane whose
            # stride is its width -- would be left pointing at nothing.
            planes.append(_read_only(np.array(samples[:, :width], dtype=np.uint8, copy=True)))
        return Planes(planes=tuple(planes), warning=text)
    finally:
        _free("planes_free", ctypes.byref(raw))
