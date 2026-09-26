# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Decoding a JPEG file: its components, reconstructed within their intervals (docs/math.md, 1.3 and 8).

The components are reconstructed on one canvas (unround.frames) by one of the
methods, and the picture is cut from it. The methods are the decoder of the data
term's centres, "mmse" (the MMSE decoder with the default centres, docs/math.md,
2.3), TV and TGV by the primal-dual method (5), and, for greyscale files, TV by the
subgradient method of jpeg2png's kind (7), which is there to be compared with. A
file in YCbCr is solved in YCbCr, and its picture is the RGB that JFIF's
conversion gives (8); that of a file in RGB, or greyscale, is its canvas.
"""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt

from unround import colour, frames, jpegio, pdhg, subgradient
from unround.frames import Frame
from unround.model import TGV, TV, DataTerm, Problem, make_problem
from unround.results import FrameResult

__all__ = ["Decoded", "Method", "Settings", "component_problem", "decode", "frame_of"]

type Array = npt.NDArray[np.float64]
type Method = Literal["mmse", "tv", "tgv", "subgradient"]


@dataclass(frozen=True, slots=True)
class Settings:
    """The method, the model, and the solvers' options.

    data are the options of G (unround.model.DataTerm), one for every component or one
    for each; tv and tgv are the weights of the models, with how they take the channels;
    and pdhg and subgradient the options of the solvers.
    """

    method: Method = "tgv"
    data: DataTerm | tuple[DataTerm, ...] = field(default_factory=DataTerm)
    tv: TV = field(default_factory=TV)
    tgv: TGV = field(default_factory=TGV)
    pdhg: pdhg.Options = field(default_factory=pdhg.Options)
    subgradient: subgradient.Options = field(default_factory=subgradient.Options)


@dataclass(frozen=True, slots=True, eq=False)
class Decoded:
    """A reconstructed file.

    picture is in binary64: (height, width) of a greyscale file, and (height, width, 3),
    RGB, of a colour one. Of a greyscale file or one in RGB, it is the solver's canvas
    itself, cut to the picture; of one in YCbCr, the RGB that JFIF's conversion gives of
    that (docs/math.md, 8). It is neither clamped nor rounded (unround.tiff writes it bit
    for bit). planes are the canvas cut to the picture, (C, height, width), as the solver
    left it: the Y, Cb and Cr of a file in YCbCr. coefficients are every component's,
    each within its intervals. result is the solver's, None for the MMSE decoder.
    """

    picture: Array
    planes: Array
    color_space: jpegio.ColorSpace
    coefficients: tuple[Array, ...]
    frame: Frame
    result: FrameResult | None


def component_problem(component: jpegio.Component, settings: Settings | None = None, index: int = 0) -> Problem:
    """The problem of a component, the index-th of its file, with the model of the settings."""
    settings = Settings() if settings is None else settings
    data = settings.data if isinstance(settings.data, DataTerm) else settings.data[index]
    return make_problem(component.coefficients, component.quant_table, data)


def frame_of(image: jpegio.Image, settings: Settings | None = None) -> Frame:
    """The frame of a file's components (docs/math.md, 1.3), with the model of the settings."""
    settings = Settings() if settings is None else settings
    count = len(image.components)
    if isinstance(settings.data, tuple) and len(settings.data) != count:
        message = f"options of G for each of the {count} components, not {len(settings.data)}"
        raise ValueError(message)
    problems = [component_problem(component, settings, index) for index, component in enumerate(image.components)]
    factors = [(component.h_samp_factor, component.v_samp_factor) for component in image.components]
    return frames.make_frame(problems, factors, (image.height, image.width))


def _subgradient(frame: Frame, settings: Settings) -> FrameResult:
    if len(frame.channels) != 1:
        message = "the subgradient method is for greyscale files"
        raise ValueError(message)
    result = subgradient.solve_tv(frame.channels[0].problem, settings.tv, settings.subgradient)
    return FrameResult(
        primal=frames.Primal(coefficients=(result.primal.coefficients,), canvas=result.primal.canvas[np.newaxis]),
        dual=None,
        iterations=result.iterations,
        converged=result.converged,
        history=result.history,
    )


def decode(data: bytes, settings: Settings | None = None) -> Decoded:
    """Reconstructs a JPEG file, greyscale or in colour."""
    settings = Settings() if settings is None else settings
    image = jpegio.read(data)
    frame = frame_of(image, settings)
    result: FrameResult | None
    match settings.method:
        case "mmse":
            result = None
            point = frames.start(frame)
        case "tv":
            result = pdhg.solve_frame_tv(frame, settings.tv, settings.pdhg)
            point = result.primal
        case "tgv":
            result = pdhg.solve_frame_tgv(frame, settings.tgv, settings.pdhg)
            point = result.primal
        case "subgradient":
            result = _subgradient(frame, settings)
            point = result.primal
    planes = point.canvas[:, : image.height, : image.width]
    if image.color_space is jpegio.ColorSpace.YCBCR:
        picture = colour.to_rgb(planes)
    elif len(frame.channels) == 1:
        picture = planes[0]
    else:
        picture = planes.transpose(1, 2, 0)
    return Decoded(
        picture=picture,
        planes=planes,
        color_space=image.color_space,
        coefficients=point.coefficients,
        frame=frame,
        result=result,
    )
