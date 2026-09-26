# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""The reference implementation of JPEG-Unround (Rust), through its C interface.

rust/capi/include/unround.h is the interface, and ctypes calls it. The settings go
over as the command line's options (docs/cli.md), each number as the shortest
decimal that reads back as the same double; the results come back as NumPy arrays,
copied, and the library's memory is freed before a function returns. The settings
are the dataclasses of the Python implementation (unround.decode.Settings and what
it holds), so that the two take the same values.

The library is the one at the path in the environment variable UNROUND_LIBRARY, or
else the one in unround/_native, where a wheel carries it; cargo build --release -p
unround-capi, in rust/, writes it into rust/target/release. ctypes releases the GIL
while the library computes, so threads reconstruct files at once; an observer is
called back with the GIL held.
"""

import ctypes
import enum
import functools
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from unround import _loading, jpegio, pdhg, results, subgradient
from unround.decode import Settings
from unround.model import TGV, TV, DataTerm

__all__ = [
    "ABI_VERSION",
    "Decoded",
    "LibraryNotFoundError",
    "Problem",
    "Record",
    "Solution",
    "Status",
    "Stop",
    "UnroundError",
    "decode",
    "held_options",
    "main",
    "options",
    "pnm_bytes",
    "solve",
    "tiff_bytes",
    "to_rgb",
    "to_ycbcr",
    "version",
]

type Array = npt.NDArray[np.float64]
type Observer = Callable[["Record"], bool | None]

ABI_VERSION: Final = 1
"""UNROUND_ABI_VERSION of the interface this module mirrors."""

_MESSAGE_SIZE: Final = 4096
_BLOCK_SIZE: Final = 64
_CHANNELS: Final = 3


class _Record(ctypes.Structure):
    _fields_ = (
        ("iteration", ctypes.c_uint64),
        ("gap", ctypes.c_double),
        ("primal", ctypes.c_double),
        ("dual", ctypes.c_double),
        ("channels", ctypes.c_uint64),
        ("height", ctypes.c_uint64),
        ("width", ctypes.c_uint64),
        ("canvas", ctypes.POINTER(ctypes.c_double)),
    )


class _Component(ctypes.Structure):
    _fields_ = (
        ("levels", ctypes.POINTER(ctypes.c_int16)),
        ("rows", ctypes.c_uint32),
        ("columns", ctypes.c_uint32),
        ("h_samp_factor", ctypes.c_int32),
        ("v_samp_factor", ctypes.c_int32),
        ("quant_table", ctypes.c_uint16 * _BLOCK_SIZE),
    )


class _Input(ctypes.Structure):
    _fields_ = (
        ("height", ctypes.c_uint32),
        ("width", ctypes.c_uint32),
        ("color_space", ctypes.c_int32),
        ("count", ctypes.c_uint32),
        ("components", ctypes.POINTER(_Component)),
    )


# The sizes the interface's structures have on the 64-bit systems JPEG-Unround
# supports; the header and the Rust crate assert the same numbers.
_SIZES: Final = ((_Record, 64), (_Component, 152), (_Input, 24))

_OBSERVER: Final = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p, ctypes.POINTER(_Record))


class Status(enum.IntEnum):
    """What a function of the interface reports: unround_status."""

    OK = 0
    ARGUMENT = 1
    READ = 2
    UNSUPPORTED = 3
    OPTIONS = 4
    IO = 5
    INTERNAL = 6


class Stop(enum.IntEnum):
    """Why a solver stopped."""

    CONVERGED = 1
    """A tolerance was met."""
    ITERATIONS = 2
    """The solver took the most iterations."""
    OBSERVER = 3
    """The observer stopped it."""
    STATIONARY = 4
    """The subgradient method found a subgradient of 0."""


class UnroundError(Exception):
    """A file or arrays that were not reconstructed: the status of the interface, and its message."""

    def __init__(self, status: Status, message: str) -> None:
        super().__init__(f"{message} ({status.name.lower()})")
        self.status = status
        self.message = message


class LibraryNotFoundError(RuntimeError):
    """The library is not where this module looks for it, or is not of its version."""


def _declare(library: ctypes.CDLL, name: str, result: object, *arguments: object) -> None:
    function = getattr(library, name)
    function.argtypes = arguments
    function.restype = result


def _declare_all(library: ctypes.CDLL) -> None:
    size, handle, u64, i32 = ctypes.c_size_t, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int32
    doubles, u64_out, i32_out = ctypes.POINTER(ctypes.c_double), ctypes.POINTER(u64), ctypes.POINTER(i32)
    strings, bytes_out = ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8))
    message = (ctypes.c_char_p, size)
    made = (_OBSERVER, handle, ctypes.POINTER(handle), *message)
    _declare(library, "unround_abi_version", i32)
    _declare(library, "unround_version", ctypes.c_char_p)
    _declare(library, "unround_settings_new", handle, size, strings, *message)
    _declare(library, "unround_settings_free", None, handle)
    _declare(library, "unround_settings_options", size, handle, ctypes.c_char_p, size)
    _declare(library, "unround_decode", i32, ctypes.c_char_p, size, handle, *made)
    _declare(library, "unround_solve", i32, ctypes.POINTER(_Input), handle, *made)
    _declare(library, "unround_result_free", None, handle)
    _declare(library, "unround_result_picture", doubles, handle, u64_out, u64_out, u64_out)
    _declare(library, "unround_result_planes", doubles, handle, u64_out)
    _declare(library, "unround_result_color_space", i32, handle)
    _declare(library, "unround_result_canvas", doubles, handle, u64_out, u64_out, u64_out)
    _declare(library, "unround_result_components", u64, handle)
    _declare(library, "unround_result_coefficients", doubles, handle, u64, u64_out)
    _declare(library, "unround_result_problem", doubles, handle, u64, i32, u64_out)
    _declare(library, "unround_result_component", i32, handle, u64, u64_out, u64_out, u64_out, u64_out)
    _declare(library, "unround_result_solver", i32, handle, u64_out, i32_out)
    _declare(library, "unround_result_record_iterations", u64_out, handle, u64_out)
    _declare(library, "unround_result_record_values", doubles, handle, i32, u64_out)
    _declare(library, "unround_result_icc_profile", ctypes.POINTER(ctypes.c_uint8), handle, u64_out)
    _declare(library, "unround_result_exif_orientation", i32, handle)
    written = (doubles, u64, u64, u64, i32, bytes_out, u64_out, *message)
    _declare(library, "unround_tiff", i32, *written)
    _declare(library, "unround_pnm", i32, *written)
    _declare(library, "unround_bytes_free", None, ctypes.POINTER(ctypes.c_uint8), u64)
    _declare(library, "unround_to_rgb", i32, doubles, u64, u64, doubles)
    _declare(library, "unround_to_ycbcr", i32, doubles, u64, u64, doubles)
    _declare(library, "unround_main", i32, size, strings)


@functools.cache
def _library() -> ctypes.CDLL:
    for structure, size in _SIZES:
        if ctypes.sizeof(structure) != size:
            message = f"{structure.__name__} is {ctypes.sizeof(structure)} bytes, and the interface's is {size}"
            raise RuntimeError(message)
    path = _loading.configured("UNROUND_LIBRARY") or _loading.bundled(_loading.RUST)
    if path is None or not path.is_file():
        where = f"UNROUND_LIBRARY names {path}, where there is no file" if path else "the package holds no library"
        message = (
            f"{where}: build it with cargo build --release -p unround-capi in rust/ "
            f"(it is written into rust/target/release) and set UNROUND_LIBRARY to it"
        )
        raise LibraryNotFoundError(message)
    library = ctypes.CDLL(str(path))
    _declare_all(library)
    found = int(library.unround_abi_version())
    if found != ABI_VERSION:
        message = f"{path} is of ABI version {found}, and this module of version {ABI_VERSION}"
        raise LibraryNotFoundError(message)
    return library


def version() -> str:
    """The version of JPEG-Unround and of the implementation, as the library says them."""
    text: bytes = _library().unround_version()
    return text.decode("utf-8")


def _number(value: float) -> str:
    """The shortest decimal that reads back as the same double: repr, which Rust reads correctly rounded."""
    return repr(float(value))


def _data_options(data: DataTerm | tuple[DataTerm, ...]) -> list[str]:
    """The options of G: one value where every component has it, or else one for each."""
    terms = (data,) if isinstance(data, DataTerm) else data
    fields: tuple[tuple[str, Callable[[DataTerm], str]], ...] = (
        ("--mu", lambda term: "rule" if term.mu is None else _number(term.mu)),
        ("--mu-scale", lambda term: _number(term.mu_scale)),
        ("--mu-power", lambda term: _number(term.mu_power)),
        ("--weight-power", lambda term: _number(term.power)),
        ("--dc-weight", lambda term: _number(term.dc_weight)),
        ("--centres", lambda term: term.centres),
        ("--slack", lambda term: _number(term.slack)),
        ("--slack-cost", lambda term: _number(term.slack_cost)),
    )
    result = []
    for name, value in fields:
        values = [value(term) for term in terms]
        result += [name, values[0] if len(set(values)) == 1 else ",".join(values)]
    return result


def _pdhg_options(solver: pdhg.Options) -> list[str]:
    result = [
        "--relative-tolerance",
        _number(solver.relative_tolerance),
        "--partial-tolerance",
        _number(solver.partial_tolerance),
        "--step-product",
        _number(solver.step_product),
        "--record-every",
        str(solver.record_every),
        "--free-radius",
        _number(solver.free_radius),
    ]
    if solver.iterations is not None:
        result += ["--iterations", str(solver.iterations)]
    for name, value in (
        ("--tolerance", solver.tolerance),
        ("--step-ratio", solver.step_ratio),
        ("--relaxation", solver.relaxation),
        ("--norm-squared", solver.norm_squared),
        ("--partial-radius", solver.partial_radius),
    ):
        if value is not None:
            result += [name, _number(value)]
    if not solver.scale_with_weight:
        result.append("--no-weight-scaling")
    return result


def _subgradient_options(method: subgradient.Options) -> list[str]:
    result = [
        "--subgradient-iterations",
        str(method.iterations),
        "--subgradient-record-every",
        str(method.record_every),
        "--subgradient-step",
        _number(method.step),
        "--subgradient-decay",
        _number(method.decay),
    ]
    if not method.momentum:
        result.append("--no-momentum")
    return result


def options(
    settings: Settings | None = None,
    *,
    max_pixels: int = 0,
    max_scans: int = 0,
    warnings_are_errors: bool = False,
) -> list[str]:
    """The options of the command line that are the settings, and the limits of reading (docs/cli.md).

    The weights of the channels, and whether they are coupled, are those of the method's
    model: TGV's for TGV, and TV's otherwise. The limits are those of unround.jpegio.read,
    0 for the C layer's own.
    """
    settings = Settings() if settings is None else settings
    weights: TV | TGV = settings.tgv if settings.method == "tgv" else settings.tv
    result = ["--method", settings.method, "--alpha", _number(settings.tv.alpha)]
    result += ["--alpha1", _number(settings.tgv.alpha1), "--alpha0", _number(settings.tgv.alpha0)]
    if weights.channel_weights is not None:
        result += ["--channel-weights", ",".join(_number(value) for value in weights.channel_weights)]
    result += ["--channels", "coupled" if weights.coupled else "apart"]
    result += _data_options(settings.data)
    result += _pdhg_options(settings.pdhg)
    result += _subgradient_options(settings.subgradient)
    if max_pixels < 0 or max_scans < 0:
        message = f"the limits of reading are at least 0, not {max_pixels} pixels and {max_scans} scans"
        raise ValueError(message)
    if max_pixels:
        result += ["--max-pixels", str(max_pixels)]
    if max_scans:
        result += ["--max-scans", str(max_scans)]
    if warnings_are_errors:
        result.append("--warnings-are-errors")
    return result


class _Settings:
    """Settings of the interface, made from options, and freed with the object."""

    def __init__(self, given: list[str]) -> None:
        self._library = _library()
        encoded = [option.encode("utf-8") for option in given]
        array = (ctypes.c_char_p * max(len(encoded), 1))(*encoded)
        message = ctypes.create_string_buffer(_MESSAGE_SIZE)
        self.handle: int | None = self._library.unround_settings_new(len(encoded), array, message, _MESSAGE_SIZE)
        if not self.handle:
            raise UnroundError(Status.OPTIONS, message.value.decode("utf-8", errors="replace"))

    def __del__(self) -> None:
        handle = getattr(self, "handle", None)
        if handle:
            self._library.unround_settings_free(handle)
            self.handle = None


def held_options(
    settings: Settings | None = None,
    *,
    max_pixels: int = 0,
    max_scans: int = 0,
    warnings_are_errors: bool = False,
) -> list[str]:
    """The settings as the library holds them, once it has taken them: every value, as options.

    Each is --name=value or --flag, each number the shortest decimal that reads back as
    the double the library holds; values not given are the library's defaults.
    """
    library = _library()
    handle = _Settings(
        options(settings, max_pixels=max_pixels, max_scans=max_scans, warnings_are_errors=warnings_are_errors)
    )
    return _held(library, handle)


def _held(library: ctypes.CDLL, handle: _Settings) -> list[str]:
    length = int(library.unround_settings_options(handle.handle, None, 0))
    buffer = ctypes.create_string_buffer(length + 1)
    if int(library.unround_settings_options(handle.handle, buffer, length + 1)) != length:
        message = "the library's options changed between two calls"
        raise RuntimeError(message)
    return buffer.value.decode("ascii").split(" ")


@dataclass(frozen=True, slots=True, eq=False)
class Record:
    """A record of the solver, as an observer sees it.

    gap is the duality gap per sample, which the tolerance is compared with; primal and
    dual are the values; canvas is a copy of the solver's canvas, (C, H, W).
    """

    iteration: int
    gap: float
    primal: float
    dual: float
    canvas: Array


@dataclass(frozen=True, slots=True, eq=False)
class Problem:
    """A component's model, as the solver took it.

    ratio is the canvas's samples per sample of the component, down and across. lower,
    upper and centres have the shape of the component's coefficients, (rows, columns, 8,
    8): the ends of the intervals and the centres of the data term. steps are the
    quantization steps and weights the data term's, (8, 8).
    """

    ratio: tuple[int, int]
    lower: Array
    upper: Array
    centres: Array
    steps: Array
    weights: Array


@dataclass(frozen=True, slots=True, eq=False)
class Solution:
    """What the solver did: the iterations it took, why it stopped, and its records."""

    iterations: int
    stop: Stop
    history: results.History

    @property
    def converged(self) -> bool:
        """Whether a tolerance was met."""
        return self.stop is Stop.CONVERGED


@dataclass(frozen=True, slots=True, eq=False)
class Decoded:
    """A file, or arrays, reconstructed by the reference implementation.

    picture is (height, width) of greyscale, or (height, width, 3) of RGB, in binary64,
    neither clamped nor rounded: of a file in YCbCr, JFIF's RGB of the planes. planes are
    the canvas cut to the picture, (C, height, width): the Y, Cb and Cr of a file in
    YCbCr; canvas is the whole of it, (C, H, W). coefficients are each component's,
    (rows, columns, 8, 8), within their intervals, and problems their models. result is
    the solver's, None for the decoder of the centres.
    """

    picture: Array
    planes: Array
    canvas: Array
    color_space: jpegio.ColorSpace
    coefficients: tuple[Array, ...]
    problems: tuple[Problem, ...]
    result: Solution | None
    icc_profile: bytes | None
    exif_orientation: int


def _copy(pointer: object, shape: tuple[int, ...]) -> Array:
    """A copy of doubles of the library, in the shape."""
    count = int(np.prod(shape, dtype=np.int64))
    if count == 0:
        return np.zeros(shape, dtype=np.float64)
    if not pointer:
        message = "the library gave no values where it has some"
        raise RuntimeError(message)
    values = np.ctypeslib.as_array(pointer, shape=(count,))
    return np.array(values, dtype=np.float64, copy=True).reshape(shape)


def _sizes(*count: int) -> tuple[ctypes.c_uint64, ...]:
    return tuple(ctypes.c_uint64() for _ in count)


def _checked(pointer: object, count: ctypes.c_uint64, shape: tuple[int, ...]) -> Array:
    if int(count.value) != int(np.prod(shape, dtype=np.int64)):
        message = f"the library gave {count.value} values where {shape} holds {int(np.prod(shape))}"
        raise RuntimeError(message)
    return _copy(pointer, shape)


def _problem(library: ctypes.CDLL, result: ctypes.c_void_p, index: int) -> tuple[Array, Problem]:
    rows, columns, down, across = _sizes(0, 1, 2, 3)
    status = library.unround_result_component(
        result, index, ctypes.byref(rows), ctypes.byref(columns), ctypes.byref(down), ctypes.byref(across)
    )
    if status != Status.OK:
        message = f"the result has no component {index}"
        raise RuntimeError(message)
    shape = (int(rows.value), int(columns.value), 8, 8)
    count = ctypes.c_uint64()
    coefficients = _checked(library.unround_result_coefficients(result, index, ctypes.byref(count)), count, shape)
    arrays = []
    for which, wanted in ((0, shape), (1, shape), (2, shape), (3, (8, 8)), (4, (8, 8))):
        arrays.append(
            _checked(library.unround_result_problem(result, index, which, ctypes.byref(count)), count, wanted)
        )
    lower, upper, centres, steps, weights = arrays
    ratio = (int(down.value), int(across.value))
    return coefficients, Problem(ratio, lower, upper, centres, steps, weights)


def _solution(library: ctypes.CDLL, result: ctypes.c_void_p) -> Solution | None:
    iterations, stop = ctypes.c_uint64(), ctypes.c_int32()
    if not library.unround_result_solver(result, ctypes.byref(iterations), ctypes.byref(stop)):
        return None
    count = ctypes.c_uint64()
    recorded = library.unround_result_record_iterations(result, ctypes.byref(count))
    size = int(count.value)
    steps = np.array(np.ctypeslib.as_array(recorded, shape=(size,)), dtype=np.int64) if size else np.zeros(0, np.int64)
    values = [
        _checked(library.unround_result_record_values(result, which, ctypes.byref(count)), count, (size,))
        for which in range(5)
    ]
    seconds, primal, dual, scaling, partial_gap = values
    history = results.History(steps, seconds, primal, dual, scaling, partial_gap)
    return Solution(int(iterations.value), Stop(int(stop.value)), history)


def _decoded(library: ctypes.CDLL, result: ctypes.c_void_p) -> Decoded:
    height, width, channels = _sizes(0, 1, 2)
    samples = library.unround_result_picture(result, ctypes.byref(height), ctypes.byref(width), ctypes.byref(channels))
    rows, columns, depth = int(height.value), int(width.value), int(channels.value)
    picture = _copy(samples, (rows, columns) if depth == 1 else (rows, columns, depth))
    planes_count, canvas_channels, canvas_height, canvas_width = _sizes(0, 1, 2, 3)
    canvas_pointer = library.unround_result_canvas(
        result, ctypes.byref(canvas_channels), ctypes.byref(canvas_height), ctypes.byref(canvas_width)
    )
    planes_pointer = library.unround_result_planes(result, ctypes.byref(planes_count))
    count = int(canvas_channels.value)
    planes = _checked(planes_pointer, planes_count, (count, rows, columns))
    canvas = _copy(canvas_pointer, (count, int(canvas_height.value), int(canvas_width.value)))
    components = [_problem(library, result, index) for index in range(int(library.unround_result_components(result)))]
    profile_size = ctypes.c_uint64()
    profile = library.unround_result_icc_profile(result, ctypes.byref(profile_size))
    return Decoded(
        picture=picture,
        planes=planes,
        canvas=canvas,
        color_space=jpegio.ColorSpace(int(library.unround_result_color_space(result))),
        coefficients=tuple(coefficients for coefficients, _ in components),
        problems=tuple(problem for _, problem in components),
        result=_solution(library, result),
        icc_profile=ctypes.string_at(profile, int(profile_size.value)) if profile else None,
        exif_orientation=int(library.unround_result_exif_orientation(result)),
    )


class _Watcher:
    """The observer as the interface calls it: each record turned into a Record, and an exception kept."""

    def __init__(self, observer: Observer | None) -> None:
        self._observer = observer
        self.error: BaseException | None = None
        # A null pointer of the function's type where there is no observer: ctypes takes no None for it.
        self.callback = _OBSERVER() if observer is None else _OBSERVER(self._call)

    def _call(self, _user: int | None, pointer: ctypes._Pointer[_Record]) -> int:
        if self._observer is None or self.error is not None:
            return 1
        try:
            record = pointer.contents
            shape = (int(record.channels), int(record.height), int(record.width))
            seen = Record(
                iteration=int(record.iteration),
                gap=float(record.gap),
                primal=float(record.primal),
                dual=float(record.dual),
                canvas=_copy(record.canvas, shape),
            )
            return 1 if self._observer(seen) else 0
        except BaseException as error:  # noqa: BLE001 -- raised again once the library has returned
            self.error = error
            return 1


type _Call = Callable[[object, "ctypes._CArgObject", ctypes.Array[ctypes.c_char]], int]


def _reconstruct(call: _Call, observer: Observer | None) -> Decoded:
    """Calls unround_decode or unround_solve, and reads what they made."""
    library = _library()
    watcher = _Watcher(observer)
    result = ctypes.c_void_p()
    message = ctypes.create_string_buffer(_MESSAGE_SIZE)
    status = call(watcher.callback, ctypes.byref(result), message)
    try:
        if watcher.error is not None:
            raise watcher.error
        if status != Status.OK:
            known = status in Status.__members__.values()
            raise UnroundError(Status(status) if known else Status.INTERNAL, message.value.decode("utf-8", "replace"))
        return _decoded(library, result)
    finally:
        library.unround_result_free(result)


def decode(  # noqa: PLR0913 -- the file, the settings, the observer, and the limits of jpegio.read
    data: bytes | bytearray | memoryview,
    settings: Settings | None = None,
    *,
    observer: Observer | None = None,
    max_pixels: int = 0,
    max_scans: int = 0,
    warnings_are_errors: bool = False,
) -> Decoded:
    """Reconstructs a JPEG file in memory.

    observer, where given, is called with each record after the first, and stops the
    solver where it returns True; an exception it raises stops the solver too, and is
    raised here. The limits of reading are those of unround.jpegio.read. A file that is
    not reconstructed raises UnroundError, and options out of their ranges too.
    """
    library = _library()
    handle = _Settings(
        options(settings, max_pixels=max_pixels, max_scans=max_scans, warnings_are_errors=warnings_are_errors)
    )
    buffer = bytes(data)

    def call(callback: object, result: ctypes._CArgObject, message: ctypes.Array[ctypes.c_char]) -> int:
        size = len(buffer)
        return int(library.unround_decode(buffer, size, handle.handle, callback, None, result, message, _MESSAGE_SIZE))

    return _reconstruct(call, observer)


def solve(image: jpegio.Image, settings: Settings | None = None, *, observer: Observer | None = None) -> Decoded:
    """Reconstructs the components of an image as unround.jpegio.read gives them, or changed.

    What decode does with a file, it does with the image's size, color space, and each
    component's levels, quantization table and sampling factors.
    """
    library = _library()
    handle = _Settings(options(settings))
    count = len(image.components)
    levels = [np.ascontiguousarray(component.coefficients, dtype=np.int16) for component in image.components]
    raw = (_Component * max(count, 1))()
    for index, (component, values) in enumerate(zip(image.components, levels, strict=True)):
        expected = 4
        if values.ndim != expected or values.shape[2:] != (8, 8):
            message = f"the levels of component {index} are (rows, columns, 8, 8), not {values.shape}"
            raise ValueError(message)
        raw[index].levels = values.ctypes.data_as(ctypes.POINTER(ctypes.c_int16))
        raw[index].rows, raw[index].columns = int(values.shape[0]), int(values.shape[1])
        raw[index].h_samp_factor = int(component.h_samp_factor)
        raw[index].v_samp_factor = int(component.v_samp_factor)
        table = np.asarray(component.quant_table, dtype=np.uint16).reshape(_BLOCK_SIZE)
        raw[index].quant_table = (ctypes.c_uint16 * _BLOCK_SIZE)(*(int(step) for step in table))
    given = _Input(int(image.height), int(image.width), int(image.color_space), count, raw)

    def call(callback: object, result: ctypes._CArgObject, message: ctypes.Array[ctypes.c_char]) -> int:
        pointer = ctypes.byref(given)
        return int(library.unround_solve(pointer, handle.handle, callback, None, result, message, _MESSAGE_SIZE))

    decoded = _reconstruct(call, observer)
    del levels  # kept alive until the library has read them
    return decoded


def _samples(picture: npt.ArrayLike) -> tuple[Array, int, int, int]:
    samples = np.ascontiguousarray(picture, dtype=np.float64)
    if samples.ndim == 2:  # noqa: PLR2004
        return samples, int(samples.shape[0]), int(samples.shape[1]), 1
    if samples.ndim == 3:  # noqa: PLR2004
        return samples, int(samples.shape[0]), int(samples.shape[1]), int(samples.shape[2])
    message = f"a picture is (height, width) or (height, width, channels), not {samples.shape}"
    raise ValueError(message)


def _written(name: str, picture: npt.ArrayLike, flag: int) -> bytes:
    library = _library()
    samples, height, width, channels = _samples(picture)
    pointer = ctypes.POINTER(ctypes.c_uint8)()
    size = ctypes.c_uint64()
    message = ctypes.create_string_buffer(_MESSAGE_SIZE)
    doubles = samples.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    status = getattr(library, name)(
        doubles, height, width, channels, flag, ctypes.byref(pointer), ctypes.byref(size), message, _MESSAGE_SIZE
    )
    if status != Status.OK:
        raise UnroundError(Status(status), message.value.decode("utf-8", "replace"))
    try:
        return ctypes.string_at(pointer, int(size.value))
    finally:
        library.unround_bytes_free(pointer, size)


def tiff_bytes(picture: npt.ArrayLike, *, ycbcr: bool = False) -> bytes:
    """A TIFF of the picture's samples in binary64, bit for bit: (height, width), or (height, width, 3).

    With ycbcr, the three samples of a pixel are JFIF's Y, Cb and Cr, and the file says so.
    """
    return _written("unround_tiff", picture, 1 if ycbcr else 0)


def pnm_bytes(picture: npt.ArrayLike, *, sixteen: bool = False) -> bytes:
    """A PNM file of the picture's samples, rounded to 8 bits, or to 16 (docs/cli.md)."""
    return _written("unround_pnm", picture, 1 if sixteen else 0)


def to_rgb(planes: npt.ArrayLike) -> Array:
    """JFIF's RGB, (height, width, 3), of Y, Cb and Cr planes, (3, height, width) (docs/math.md, 8)."""
    values = np.ascontiguousarray(planes, dtype=np.float64)
    if values.ndim != _CHANNELS or values.shape[0] != _CHANNELS:
        message = f"the planes are (3, height, width), not {values.shape}"
        raise ValueError(message)
    height, width = int(values.shape[1]), int(values.shape[2])
    picture = np.empty((height, width, _CHANNELS), dtype=np.float64)
    doubles = ctypes.POINTER(ctypes.c_double)
    status = _library().unround_to_rgb(values.ctypes.data_as(doubles), height, width, picture.ctypes.data_as(doubles))
    if status != Status.OK:
        raise UnroundError(Status(status), "the planes were not converted")
    return picture


def to_ycbcr(picture: npt.ArrayLike) -> Array:
    """JFIF's Y, Cb and Cr planes, (3, height, width), of an RGB picture, (height, width, 3)."""
    values = np.ascontiguousarray(picture, dtype=np.float64)
    if values.ndim != _CHANNELS or values.shape[2] != _CHANNELS:
        message = f"the picture is (height, width, 3), not {values.shape}"
        raise ValueError(message)
    height, width = int(values.shape[0]), int(values.shape[1])
    planes = np.empty((_CHANNELS, height, width), dtype=np.float64)
    doubles = ctypes.POINTER(ctypes.c_double)
    status = _library().unround_to_ycbcr(values.ctypes.data_as(doubles), height, width, planes.ctypes.data_as(doubles))
    if status != Status.OK:
        raise UnroundError(Status(status), "the picture was not converted")
    return planes


def main(arguments: Sequence[str] | None = None) -> int:
    """The command line (docs/cli.md) on the arguments, or the process's where None: its exit status."""
    given = [argument.encode("utf-8") for argument in (sys.argv[1:] if arguments is None else arguments)]
    array = (ctypes.c_char_p * max(len(given), 1))(*given)
    return int(_library().unround_main(len(given), array))
