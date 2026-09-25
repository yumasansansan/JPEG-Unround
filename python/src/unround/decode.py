# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Decoding a greyscale JPEG file: its one component, reconstructed within its intervals.

The component is reconstructed on its canvas of whole blocks by one of the
methods, and the picture is cut from it. The methods are the decoder of the
data term's centres, "mmse" (the MMSE decoder with the default centres,
docs/math.md, 2.3), TV and TGV by the primal-dual method (5), and TV by the
subgradient method of jpeg2png's kind (7), which is there to be compared with.
Colour files, with chroma subsampling, are still to come.
"""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt

from unround import jpegio, pdhg, subgradient
from unround.model import TGV, TV, DataTerm, Problem, make_problem
from unround.results import Result

__all__ = ["Decoded", "Method", "Settings", "component_problem", "decode", "decode_component"]

type Array = npt.NDArray[np.float64]
type Method = Literal["mmse", "tv", "tgv", "subgradient"]


@dataclass(frozen=True, slots=True)
class Settings:
    """The method, the model, and the solvers' options.

    data are the options of G (unround.model.DataTerm), tv and tgv the weights of the
    models, and pdhg and subgradient the options of the solvers.
    """

    method: Method = "tgv"
    data: DataTerm = field(default_factory=DataTerm)
    tv: TV = field(default_factory=TV)
    tgv: TGV = field(default_factory=TGV)
    pdhg: pdhg.Options = field(default_factory=pdhg.Options)
    subgradient: subgradient.Options = field(default_factory=subgradient.Options)


@dataclass(frozen=True, slots=True, eq=False)
class Decoded:
    """A reconstructed component.

    picture is the picture's samples, (height, width), in binary64: the solver's canvas
    itself, cut to the picture, neither clamped nor rounded, nor changed by any operation
    (unround.tiff writes it bit for bit). coefficients are those of the whole canvas, each
    within its interval. result is the solver's, None for the MMSE decoder.
    """

    picture: Array
    coefficients: Array
    problem: Problem
    result: Result | None


def component_problem(component: jpegio.Component, settings: Settings | None = None) -> Problem:
    """The problem of one component, with the model of the settings."""
    settings = Settings() if settings is None else settings
    return make_problem(component.coefficients, component.quant_table, settings.data)


def decode_component(component: jpegio.Component, settings: Settings | None = None) -> Decoded:
    """Reconstructs one component without chroma subsampling."""
    settings = Settings() if settings is None else settings
    problem = component_problem(component, settings)
    result: Result | None
    match settings.method:
        case "mmse":
            result = None
            point = pdhg.start(problem)
        case "tv":
            result = pdhg.solve_tv(problem, settings.tv, settings.pdhg)
            point = result.primal
        case "tgv":
            result = pdhg.solve_tgv(problem, settings.tgv, settings.pdhg)
            point = result.primal
        case "subgradient":
            result = subgradient.solve_tv(problem, settings.tv, settings.subgradient)
            point = result.primal
    picture = point.canvas[: component.height, : component.width]
    return Decoded(picture=picture, coefficients=point.coefficients, problem=problem, result=result)


def decode(data: bytes, settings: Settings | None = None) -> Decoded:
    """Reconstructs a greyscale JPEG file."""
    image = jpegio.read(data)
    if len(image.components) != 1:
        message = f"only greyscale files are decoded so far, and this one has {len(image.components)} components"
        raise ValueError(message)
    return decode_component(image.components[0], settings)
