#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""The conformance cases: their inputs, the Python implementation's results, and the bounds of their rounding.

    python conformance/generate.py [--out conformance/cases] [--check] [--native] [case ...]

A case is the quantized coefficients of a picture -- one component, or three in 4:2:0,
4:2:2 or 4:4:4, in YCbCr or in RGB -- and the settings of a reconstruction, as the
command line's options. Its file (conformance/cases/<name>.txt) holds them and what the
Python implementation made of them: the model's intervals, weights and centres, the steps
of the primal-dual method, its records, why it stopped, and the canvas of its last point;
and the exact centres of the Laplace model, in 60-digit decimals. Every implementation
reads the same files and has to reproduce them: what is rational to the last bit, and the
rest within the tolerances that bound the rounding of both implementations (docs/math.md,
9), which conformance/bounds.py computes from what the file holds of the reference run.

To bound the rounding of the reference run, its iterations are run again here, operation
for operation as unround.pdhg runs them (each step compared with the library's to the
last bit), and the norms of the magnitudes that bound each operation's rounding are kept,
iteration by iteration. The pictures are drawn with fixed seeds.

--check compares what would be written with the files; the Python implementation's
results may differ in their last bits on other systems, so the files are made, and
checked, on one. --native also runs the reference implementation (Rust, through its C
interface, UNROUND_LIBRARY) on every case and prints how far it lies from the expected
values, against the tolerances.
"""

import argparse
import math
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt

import bounds
import references
from unround import dct, decode, frames, jpegio, model, native, pdhg, subgradient
from unround.model import TGV, TV, DataTerm, Problem
from unround.operators import div, div2, grad, sym_grad
from unround.results import FrameResult

type Array = npt.NDArray[np.float64]

ROOT: Final = Path(__file__).resolve().parent
U: Final = 2.0**-53

ANNEX_K_LUMINANCE: Final = (
    *(16, 11, 10, 16, 24, 40, 51, 61),
    *(12, 12, 14, 19, 26, 58, 60, 55),
    *(14, 13, 16, 24, 40, 57, 69, 56),
    *(14, 17, 22, 29, 51, 87, 80, 62),
    *(18, 22, 37, 56, 68, 109, 103, 77),
    *(24, 35, 55, 64, 81, 104, 113, 92),
    *(49, 64, 78, 87, 103, 121, 120, 101),
    *(72, 92, 95, 98, 112, 100, 103, 99),
)
ANNEX_K_CHROMINANCE: Final = (
    *(17, 18, 24, 47, 99, 99, 99, 99),
    *(18, 21, 26, 66, 99, 99, 99, 99),
    *(24, 26, 56, 99, 99, 99, 99, 99),
    *(47, 66, 99, 99, 99, 99, 99, 99),
    *(99,) * 32,
)

# The subgradient method has no bound of its rounding (docs/math.md, 9.6). Its case is a
# regression check: measured on Windows (x86-64), the reference implementation's canvas lay
# 3.6e-11 from the Python one's in the Euclidean norm, and its records' primal values within
# 4.3e-15 of their size; the tolerances are more than two hundred times as much.
SUBGRADIENT_CANVAS: Final = 1e-8
SUBGRADIENT_PRIMAL: Final = 1e-12

# A case that stops at a tolerance runs this many iterations at most while one is chosen, and
# stops at the first record after STOP_AFTER iterations whose gap lies below it, and every
# earlier gap above it, by the bound of the gaps' difference and a factor of STOP_ROOM more.
STOP_SEARCH: Final = 600
STOP_AFTER: Final = 200
STOP_ROOM: Final = 1.02


@dataclass(frozen=True, slots=True)
class Case:
    """A picture's components and the settings of its reconstruction.

    size is the picture's rows and columns, factors the sampling factors (h, v) of each
    component, scale that of Annex K's tables (the luminance's for the first component, the
    chrominance's for the others, or the luminance's for every one in RGB). stop asks for a
    tolerance of TV chosen so that the method stops, at a record where every implementation
    has to stop too. step_from makes the case one iteration from the state that the method,
    unrelaxed, reaches after that many: given as the start, as its coefficients and dual
    fields, which every implementation takes alike where no sample is free.
    """

    name: str
    colour: jpegio.ColorSpace
    size: tuple[int, int]
    factors: tuple[tuple[int, int], ...]
    scale: float
    seed: int
    settings: decode.Settings
    stop: bool = False
    step_from: int = 0


def _pdhg(
    iterations: int,
    record_every: int,
    *,
    relaxation: float | None = None,
    step_ratio: float | None = None,
    partial_radius: float | None = None,
) -> pdhg.Options:
    """Options that run the iterations given, whatever the gap, and record every record_every."""
    return pdhg.Options(
        iterations=iterations,
        tolerance=0.0,
        record_every=record_every,
        relaxation=relaxation,
        step_ratio=step_ratio,
        partial_radius=partial_radius,
    )


GREY: Final = ((1, 1),)
C420: Final = ((2, 2), (1, 1), (1, 1))
C422: Final = ((2, 1), (1, 1), (1, 1))
C444: Final = ((1, 1), (1, 1), (1, 1))
GRAYSCALE, YCBCR, RGB = jpegio.ColorSpace.GRAYSCALE, jpegio.ColorSpace.YCBCR, jpegio.ColorSpace.RGB

CASES: Final = (
    Case("grey-mmse", GRAYSCALE, (21, 30), GREY, 2.0, 1, decode.Settings(method="mmse")),
    Case("grey-tv", GRAYSCALE, (21, 30), GREY, 2.0, 1, decode.Settings(method="tv", pdhg=_pdhg(100, 10))),
    Case(
        "grey-tv-unrelaxed",
        GRAYSCALE,
        (21, 30),
        GREY,
        2.0,
        1,
        decode.Settings(method="tv", pdhg=_pdhg(60, 20, relaxation=1.0, step_ratio=4.0)),
    ),
    Case(
        "grey-tv-model",
        GRAYSCALE,
        (21, 30),
        GREY,
        2.0,
        2,
        decode.Settings(
            method="tv",
            data=DataTerm(mu=0.02, centres="midpoint", dc_weight=0.5, power=1.0, slack=0.5, slack_cost=3.0),
            pdhg=_pdhg(60, 20),
        ),
    ),
    Case(
        "grey-tv-rule",
        GRAYSCALE,
        (24, 17),
        GREY,
        3.0,
        3,
        decode.Settings(
            method="tv", data=DataTerm(mu=None, mu_scale=2e-4, mu_power=1.0), tv=TV(alpha=0.5), pdhg=_pdhg(40, 10)
        ),
    ),
    Case(
        "grey-tv-stop",
        GRAYSCALE,
        (21, 30),
        GREY,
        2.0,
        1,
        decode.Settings(method="tv", pdhg=pdhg.Options(iterations=STOP_SEARCH, record_every=10)),
        stop=True,
    ),
    Case(
        "grey-tgv",
        GRAYSCALE,
        (21, 30),
        GREY,
        2.0,
        1,
        decode.Settings(method="tgv", pdhg=_pdhg(100, 10, partial_radius=20.0)),
    ),
    Case(
        "grey-subgradient",
        GRAYSCALE,
        (21, 30),
        GREY,
        2.0,
        1,
        decode.Settings(method="subgradient", subgradient=subgradient.Options(iterations=15)),
    ),
    Case("colour420-mmse", YCBCR, (16, 24), C420, 1.5, 4, decode.Settings(method="mmse")),
    Case("colour420-tv", YCBCR, (16, 24), C420, 1.5, 4, decode.Settings(method="tv", pdhg=_pdhg(60, 20))),
    Case("colour420-tgv", YCBCR, (16, 24), C420, 1.5, 4, decode.Settings(method="tgv", pdhg=_pdhg(60, 20))),
    Case(
        "colour422-tv-apart",
        YCBCR,
        (16, 24),
        C422,
        1.5,
        5,
        decode.Settings(method="tv", tv=TV(channel_weights=(1.0, 0.5, 0.75), coupled=False), pdhg=_pdhg(40, 20)),
    ),
    Case(
        "colour444-tgv-apart",
        YCBCR,
        (16, 24),
        C444,
        1.5,
        6,
        decode.Settings(method="tgv", tgv=TGV(channel_weights=(1.0, 0.5, 0.5), coupled=False), pdhg=_pdhg(40, 20)),
    ),
    Case("rgb-tv", RGB, (16, 24), C444, 1.0, 7, decode.Settings(method="tv", pdhg=_pdhg(40, 20))),
    Case(
        "grey-tv-step",
        GRAYSCALE,
        (21, 30),
        GREY,
        2.0,
        1,
        decode.Settings(method="tv", pdhg=_pdhg(1, 1, relaxation=1.0, step_ratio=4.0)),
        step_from=30,
    ),
    Case(
        "grey-tgv-step",
        GRAYSCALE,
        (21, 30),
        GREY,
        2.0,
        1,
        decode.Settings(method="tgv", pdhg=_pdhg(1, 1, relaxation=1.0)),
        step_from=30,
    ),
    Case(
        "colour444-tv-step",
        YCBCR,
        (16, 24),
        C444,
        1.5,
        6,
        decode.Settings(method="tv", pdhg=_pdhg(1, 1, relaxation=1.0)),
        step_from=30,
    ),
)


# The input.


def table(values: Sequence[int], scale: float) -> npt.NDArray[np.uint16]:
    """A table of Annex K scaled, each step rounded to the nearest integer within 1 to 255, 8 x 8."""
    steps = [min(255, max(1, round(value * scale))) for value in values]
    return np.asarray(steps, dtype=np.uint16).reshape(8, 8)


def plane(rng: np.random.Generator, rows: int, columns: int, *, luma: bool) -> Array:
    """Samples of a component: waves, and for luma steps and noise, within 0 to 255."""
    y, x = np.mgrid[0:rows, 0:columns].astype(np.float64)
    phase = rng.uniform(0.0, 2.0 * math.pi, size=2)
    if luma:
        value = 128.0 + 55.0 * np.sin(x / 4.3 + phase[0]) * np.cos(y / 3.7 + phase[1])
        value += 45.0 * ((x // 6 + y // 5) % 3 == 0) - 20.0 + rng.normal(0.0, 5.0, size=(rows, columns))
    else:
        value = 128.0 + 35.0 * np.sin(x / 7.1 + phase[0]) * np.sin(y / 5.3 + phase[1])
        value += rng.normal(0.0, 2.0, size=(rows, columns))
    return np.asarray(np.clip(value, 0.0, 255.0), dtype=np.float64)


def image_of(case: Case) -> jpegio.Image:
    """The case's components, quantized from planes drawn with its seed, as the C layer would read them."""
    rng = np.random.default_rng(case.seed)
    rows, columns = case.size
    most_h = max(h for h, _ in case.factors)
    most_v = max(v for _, v in case.factors)
    components = []
    for index, (h, v) in enumerate(case.factors):
        down, across = most_v // v, most_h // h
        blocks = (-(-rows // (8 * down)), -(-columns // (8 * across)))
        samples = plane(rng, 8 * blocks[0], 8 * blocks[1], luma=index == 0 or case.colour is RGB)
        steps = table(ANNEX_K_LUMINANCE if index == 0 or case.colour is RGB else ANNEX_K_CHROMINANCE, case.scale)
        levels = np.rint(dct.forward(samples - 128.0) / steps.astype(np.float64)).astype(np.int16)
        components.append(
            jpegio.Component(
                id=index + 1,
                h_samp_factor=h,
                v_samp_factor=v,
                quant_table_slot=min(index, 1),
                width=-(-columns * h // most_h),
                height=-(-rows * v // most_v),
                quant_table=steps,
                coefficients=levels,
            )
        )
    return jpegio.Image(
        width=columns,
        height=rows,
        color_space=case.colour,
        max_h_samp_factor=most_h,
        max_v_samp_factor=most_v,
        progressive=False,
        arithmetic=False,
        exif_orientation=0,
        warnings=0,
        warning="",
        components=tuple(components),
        icc_profile=None,
    )


# The exact centres (docs/math.md, 2.2), in decimals.


def exact_centres(levels: npt.NDArray[np.int16], steps: npt.NDArray[np.uint16], data: DataTerm) -> list[Decimal]:
    """The centre of every coefficient, (rows, columns, 8, 8) flattened: the mean of its bin, or its middle."""
    rows, columns = levels.shape[:2]
    exact = [Decimal(0)] * (rows * columns * 64)
    with localcontext() as context:
        context.prec = 60
        for v in range(8):
            for u in range(8):
                step = int(steps[v, u])
                column = [int(levels[b // columns, b % columns, v, u]) for b in range(rows * columns)]
                if (v, u) == (0, 0) or data.centres == "midpoint":
                    shift = 1024 if (v, u) == (0, 0) else 0
                    for b, q in enumerate(column):
                        exact[64 * b + 8 * v + u] = Decimal(q * step + shift)
                    continue
                nonzero = [abs(q) for q in column if q != 0]
                odd = sum(2 * q - 1 for q in nonzero)
                if odd == 0:
                    continue
                scale = references.exact_scale(len(column) - len(nonzero), len(nonzero), odd, step)
                for b, q in enumerate(column):
                    if q == 0:
                        continue
                    lower = (abs(q) - Decimal("0.5")) * step
                    mean = references.mean_by_antiderivative(lower, lower + step, scale)
                    exact[64 * b + 8 * v + u] = mean if q > 0 else -mean
    return exact


def tested_centre_bounds(levels: npt.NDArray[np.int16], steps: npt.NDArray[np.uint16], data: DataTerm) -> Array:
    """How far a conforming implementation's centres may lie from the exact ones (docs/math.md, 9.2).

    (2|q| + 20) u Q for an AC coefficient of a level q other than 0 and MMSE centres; 0 elsewhere, where the
    centre is exact.
    """
    magnitude = np.abs(levels.astype(np.float64))
    bound = (2.0 * magnitude + 20.0) * U * steps.astype(np.float64)
    bound[magnitude == 0.0] = 0.0
    bound[:, :, 0, 0] = 0.0
    if data.centres == "midpoint":
        bound[...] = 0.0
    return np.asarray(bound, dtype=np.float64)


# Magnitudes.

ABS_BASIS: Final = np.abs(dct.BASIS)
TENSOR_WEIGHTS: Final = np.asarray([1.0, 1.0, 2.0]).reshape(3, 1, 1, 1)


def norm(values: Array) -> float:
    """The Euclidean norm."""
    return float(np.sqrt(np.sum(values * values)))


def tensor_norm(values: Array) -> float:
    """The Euclidean norm of a tensor field (3, C, H, W), the off-diagonal entry counted twice."""
    return float(np.sqrt(np.sum(TENSOR_WEIGHTS * values * values)))


def forward_magnitude(samples: Array) -> Array:
    """|C| |X| |C|^T of every block of a canvas of a component."""
    return dct.from_blocks(np.asarray(ABS_BASIS @ dct.blocks(np.abs(samples)) @ ABS_BASIS.T, dtype=np.float64))


def inverse_magnitude(coefficients: Array) -> Array:
    """|C|^T |Z| |C| of every block of coefficients (rows, columns, 8, 8)."""
    return dct.from_blocks(np.asarray(ABS_BASIS.T @ np.abs(coefficients) @ ABS_BASIS, dtype=np.float64))


def backward_terms(f: Array, axis: int) -> Array:
    """The magnitudes of the terms of the backward difference along an axis (-1 across, -2 down), added."""
    result = np.zeros_like(f)
    head = [slice(None)] * f.ndim
    tail = [slice(None)] * f.ndim
    head[axis], tail[axis] = slice(None, -1), slice(1, None)
    result[tuple(head)] += f[tuple(head)]
    result[tuple(tail)] += f[tuple(head)]
    return result


def forward_terms(f: Array, axis: int) -> Array:
    """The magnitudes of the terms of the forward difference along an axis, added: 0 at the far end."""
    result = np.zeros_like(f)
    head = [slice(None)] * f.ndim
    tail = [slice(None)] * f.ndim
    head[axis], tail[axis] = slice(None, -1), slice(1, None)
    result[tuple(head)] = f[tuple(tail)] + f[tuple(head)]
    return result


def div_terms(p: Array) -> Array:
    """The magnitudes of the terms of div p, p (2, C, H, W) of magnitudes."""
    return backward_terms(p[0], -1) + backward_terms(p[1], -2)


def div2_terms(r: Array) -> Array:
    """The magnitudes of the terms of div2 r, r (3, C, H, W) of magnitudes."""
    return np.stack(
        (forward_terms(r[0], -1) + forward_terms(r[2], -2), forward_terms(r[2], -1) + forward_terms(r[1], -2))
    )


def sym_grad_terms(w: Array) -> Array:
    """The magnitudes of the terms of the symmetrized gradient of w (2, C, H, W) of magnitudes."""
    halved = (backward_terms(w[0], -2) + backward_terms(w[1], -1)) / 2.0
    return np.stack((backward_terms(w[0], -1), backward_terms(w[1], -2), halved))


# The reference's iterations, run again with the magnitudes of their rounding.


@dataclass(frozen=True, slots=True)
class Model:
    """What the bounds need of a case's model, besides its frame."""

    frame: frames.Frame
    gammas: Array
    tested: tuple[Array, ...]
    reference: tuple[Array, ...]
    exact: tuple[list[Decimal], ...]


@dataclass(slots=True)
class State:
    """A state of the primal-dual method, and the tilde point it last made."""

    x: Array
    p: Array
    w: Array | None = None
    r: Array | None = None
    coefficients: tuple[Array, ...] = ()
    canvas: Array | None = None
    p_out: Array | None = None
    w_out: Array | None = None
    r_out: Array | None = None


def traced_prox(model_: Model, v: Array, tau: float) -> tuple[tuple[Array, ...], Array, dict[str, float]]:
    """frames.prox, step for step, with the norms of the magnitudes of its rounding (docs/math.md, 9.3)."""
    frame = model_.frame
    coefficients: list[Array] = []
    result = np.empty_like(v)
    squares = dict.fromkeys(("Mf", "Mi", "Mphi", "McenA", "McenR", "Mmean", "Mput"), 0.0)
    for index, channel in enumerate(frame.channels):
        n = channel.cells
        problem = channel.problem
        own = frames.means(channel, v[index])
        e = dct.forward(own)
        chosen = model.prox(problem, e, tau / n)
        coefficients.append(chosen)
        result[index] = frames._put(frame, channel, v[index], own, chosen)
        t = (tau / n) * problem.weights
        phi = (np.abs(e) + t * np.abs(problem.centres)) / (1.0 + t)
        if problem.costs is not None:
            phi = phi + (tau / n) * problem.costs / (1.0 + t)
        squares["Mf"] += n * norm(forward_magnitude(own)) ** 2
        squares["Mi"] += n * norm(inverse_magnitude(chosen)) ** 2
        squares["Mphi"] += n * norm(phi) ** 2
        squares["McenA"] += n * norm(t * model_.tested[index] / (1.0 + t)) ** 2
        squares["McenR"] += n * norm(t * model_.reference[index] / (1.0 + t)) ** 2
        if n > 1:
            squares["Mmean"] += n * norm(frames.means(channel, np.abs(v[index]))) ** 2
            change = np.abs(dct.inverse(chosen) - own)
            rows, columns = channel.extent
            put = frames.spread(frame, channel, change)[:rows, :columns] + np.abs(result[index][:rows, :columns])
            squares["Mput"] += norm(put) ** 2
    library = frames.prox(frame, v, tau)
    same = all(np.array_equal(one, other) for one, other in zip(library[0], coefficients, strict=True))
    if not (np.array_equal(library[1], result) and same):
        message = "the proximal map run again differs from unround.frames.prox"
        raise RuntimeError(message)
    return tuple(coefficients), result, {key: math.sqrt(value) for key, value in squares.items()}


def relaxed(rho: float, new: Array, old: Array) -> Array:
    return np.asarray(rho * new + (1.0 - rho) * old, dtype=np.float64)


def tv_step(model_: Model, weights: TV, plan: pdhg.Plan, s: State) -> tuple[State, dict[str, float]]:
    """One iteration of unround.pdhg.solve_frame_tv, and the norms of its magnitudes."""
    tau, sigma = pdhg.steps(plan.norm_squared, plan.step_ratio, plan.step_product)
    gammas, rho = model_.gammas, plan.relaxation
    v = s.x + tau * (gammas * div(s.p))
    coefficients, canvas, m = traced_prox(model_, v, tau)
    extrapolated = 2.0 * canvas - s.x
    gradient = grad(extrapolated)
    ascent = s.p + sigma * (gammas * gradient)
    p_out = frames.project_vectors(ascent, weights.alpha, coupled=weights.coupled)
    m |= {
        "Mx": norm(s.x),
        "Md": norm(tau * gammas * div_terms(np.abs(s.p))),
        "Mq": norm(sigma * gammas * np.abs(extrapolated)),
        "Mg": norm(sigma * gammas * np.abs(gradient)),
        "Ma": norm(ascent),
        "Mp": norm(p_out),
    }
    if rho == 1.0:
        new = State(x=canvas, p=p_out)
    else:
        new = State(x=relaxed(rho, canvas, s.x), p=relaxed(rho, p_out, s.p))
        m["Mrx"] = norm(rho * np.abs(canvas) + abs(1.0 - rho) * np.abs(s.x))
        m["Mrp"] = norm(rho * np.abs(p_out) + abs(1.0 - rho) * np.abs(s.p))
    new.coefficients, new.canvas, new.p_out = coefficients, canvas, p_out
    return new, m


def tgv_step(model_: Model, weights: TGV, plan: pdhg.Plan, s: State) -> tuple[State, dict[str, float]]:
    """One iteration of unround.pdhg.solve_frame_tgv, and the norms of its magnitudes."""
    if s.w is None or s.r is None:
        message = "a state of TGV has w and r"
        raise ValueError(message)
    tau, sigma = pdhg.steps(plan.norm_squared, plan.step_ratio, plan.step_product)
    gammas, rho, coupled = model_.gammas, plan.relaxation, weights.coupled
    v = s.x + tau * (gammas * div(s.p))
    coefficients, canvas, m = traced_prox(model_, v, tau)
    w_out = s.w + tau * (gammas * (s.p + div2(s.r)))
    extrapolated, extrapolated_w = 2.0 * canvas - s.x, 2.0 * w_out - s.w
    gradient = grad(extrapolated)
    ascent = s.p + sigma * (gammas * (gradient - extrapolated_w))
    p_out = frames.project_vectors(ascent, weights.alpha1, coupled=coupled)
    lifted = s.r + sigma * (gammas * sym_grad(extrapolated_w))
    r_out = frames.project_tensors(lifted, weights.alpha0, coupled=coupled)
    m |= {
        "Mx": norm(s.x),
        "Md": norm(tau * gammas * div_terms(np.abs(s.p))),
        "Mw": norm(s.w),
        "Mh": norm(tau * gammas * (np.abs(s.p) + div2_terms(np.abs(s.r)))),
        "Mq": norm(sigma * gammas * np.abs(extrapolated)),
        "Mwbar": norm(sigma * gammas * np.abs(extrapolated_w)),
        "Mg": norm(sigma * gammas * (np.abs(gradient) + np.abs(extrapolated_w))),
        "Ma": norm(ascent),
        "Mp": norm(p_out),
        "ME": tensor_norm(sigma * gammas * sym_grad_terms(np.abs(extrapolated_w))),
        "Mb": tensor_norm(lifted),
        "Mr": tensor_norm(r_out),
    }
    if rho == 1.0:
        new = State(x=canvas, p=p_out, w=w_out, r=r_out)
    else:
        new = State(
            x=relaxed(rho, canvas, s.x),
            p=relaxed(rho, p_out, s.p),
            w=relaxed(rho, w_out, s.w),
            r=relaxed(rho, r_out, s.r),
        )
        m["Mrx"] = norm(rho * np.abs(canvas) + abs(1.0 - rho) * np.abs(s.x))
        m["Mrp"] = norm(rho * np.abs(p_out) + abs(1.0 - rho) * np.abs(s.p))
        m["Mrw"] = norm(rho * np.abs(w_out) + abs(1.0 - rho) * np.abs(s.w))
        m["Mrr"] = tensor_norm(rho * np.abs(r_out) + abs(1.0 - rho) * np.abs(s.r))
    new.coefficients, new.canvas, new.p_out, new.w_out, new.r_out = coefficients, canvas, p_out, w_out, r_out
    return new, m


# The records' values, and what bounds their differences.


def conjugate_terms(problem: Problem, s: Array) -> Array:
    """The magnitudes of the terms of G*'s value at s = D xi, coefficient by coefficient (unround.model.conjugate)."""
    weights = np.broadcast_to(problem.weights, s.shape)
    quadratic = weights > 0.0
    m = np.where(quadratic, weights, 1.0)
    if problem.costs is None or problem.inner_lower is None or problem.inner_upper is None:
        linear = np.abs(np.maximum(s * problem.lower, s * problem.upper))
        best = np.clip(problem.centres + s / m, problem.lower, problem.upper)
        curved = np.abs(s * best) + 0.5 * m * (best - problem.centres) ** 2
        return np.asarray(np.where(quadratic, curved, linear), dtype=np.float64)
    cost = np.broadcast_to(problem.costs, s.shape)
    lower, upper = problem.inner_lower, problem.inner_upper
    top = problem.centres + s / m
    up = np.minimum(np.maximum(problem.centres + (s - cost) / m, upper), problem.upper)
    down = np.maximum(np.minimum(problem.centres + (s + cost) / m, lower), problem.lower)
    curved = np.where(top > upper, up, np.where(top < lower, down, top))
    straight = np.where(s > cost, problem.upper, np.where(s > 0.0, upper, np.where(s < -cost, problem.lower, lower)))
    best = np.where(quadratic, curved, straight)
    charged = cost * model.beyond(problem, best)
    return np.asarray(np.abs(s * best) + 0.5 * weights * (best - problem.centres) ** 2 + charged, dtype=np.float64)


def certified(frame: frames.Frame) -> bool:
    """Whether the gap is certified: no free samples, whose partial gap assumes a box (docs/math.md, 6.6)."""
    return not any(frame.free(channel) for channel in frame.channels)


def record_bounds(
    model_: Model, coefficients: Sequence[Array], xi_terms: Array | None, xi: Array | None, primal: float
) -> dict[str, float]:
    """What bounds the differences of a record's values (docs/math.md, 9.4)."""
    frame = model_.frame
    lzeta = lcen = 0.0
    for channel, own in zip(frame.channels, coefficients, strict=True):
        problem = channel.problem
        weight = np.broadcast_to(problem.weights, own.shape)
        cost = 0.0 if problem.costs is None else np.broadcast_to(problem.costs, own.shape)
        distance = weight * np.abs(own - problem.centres)
        lzeta += norm(distance + cost) ** 2
        lcen += norm(distance) ** 2
    result = {"Lzeta": math.sqrt(lzeta), "Lcen": math.sqrt(lcen), "P": primal, "Dabs": -1.0, "Zdot": 0.0}
    if xi is None or xi_terms is None or not certified(frame):
        return result
    absolute = zdot = 0.0
    for index, channel in enumerate(frame.channels):
        problem = channel.problem
        s = dct.forward(xi[index])
        absolute += float(np.sum(conjugate_terms(problem, s)))
        reach = dct.blocks(forward_magnitude(xi_terms[index]))
        zdot += float(np.sum(np.maximum(np.abs(problem.lower), np.abs(problem.upper)) * reach))
    result["Dabs"], result["Zdot"] = absolute, zdot
    return result


def tv_dual(model_: Model, p: Array) -> tuple[Array, Array]:
    """xi = gamma div p of TV's dual value, and the magnitudes of its terms."""
    return model_.gammas * div(p), model_.gammas * div_terms(np.abs(p))


def tgv_dual(model_: Model, r: Array, theta: float) -> tuple[Array, Array]:
    """xi = gamma div(-theta div2 r) of TGV's feasible dual, and the magnitudes of its terms."""
    return model_.gammas * div(-theta * div2(r)), model_.gammas * theta * div_terms(div2_terms(np.abs(r)))


# The case's constants.


def root_bound(tau: float, sigma: float, gamma_max: float, method: str) -> float:
    """An upper bound of sqrt(sigma tau ||K||^2), with the bound of ||K||^2 of docs/math.md, 3.3."""
    with localcontext() as context:
        context.prec = 50
        bound = Decimal(8) if method == "tv" else (Decimal(17) + Decimal(33).sqrt()) / 2
        exact = Fraction(tau) * Fraction(sigma) * Fraction(gamma_max) ** 2
        product = Decimal(exact.numerator) / Decimal(exact.denominator) * bound
        root = product.sqrt()
    return math.nextafter(float(root), math.inf)


def centre_norms(model_: Model) -> tuple[float, float]:
    tested = math.sqrt(sum(norm(values) ** 2 for values in model_.tested))
    reference = math.sqrt(sum(norm(values) ** 2 for values in model_.reference))
    return tested, reference


def constants(case: Case, model_: Model, plan: pdhg.Plan | None) -> dict[str, str]:
    """The constants of a case's bounds (conformance/bounds.py), as the text of their values."""
    frame = model_.frame
    settings = case.settings
    channels, (height, width) = len(frame.channels), frame.shape
    cells = max(channel.cells for channel in frame.channels)
    tested, reference = centre_norms(model_)
    values: dict[str, float] = {"centres_tested": tested, "centres_reference": reference}
    whole: dict[str, int] = {"cells": cells}
    if plan is not None:
        weights: TV | TGV = settings.tgv if settings.method == "tgv" else settings.tv
        gamma_max = float(np.max(model_.gammas))
        tau, sigma = pdhg.steps(plan.norm_squared, plan.step_ratio, plan.step_product)
        coupled = weights.coupled and channels > 1
        norms = height * width * (1 if coupled else channels)
        # The dual fields lie in their balls, one at each pixel coupled and one for each channel apart: the
        # field xi whose conjugate the dual value takes, gamma div p for TV and -gamma div div2 (theta r) for
        # TGV, is at most gamma sqrt 8 times alpha (alpha1) sqrt(norms), and the magnitudes of its terms at most
        # gamma 2 sqrt 2 alpha sqrt(norms) (8 gamma alpha0 sqrt(norms)), whatever the implementation.
        if isinstance(weights, TV):
            lipschitz = weights.alpha * math.sqrt(norms) * gamma_max * bounds.SQRT8
            xi = gamma_max * bounds.SQRT8 * weights.alpha * math.sqrt(norms)
            xi_terms = gamma_max * bounds.SQRT8 * weights.alpha * math.sqrt(norms)
        else:
            lipschitz = (3.0 * weights.alpha1 + weights.alpha0 * bounds.SQRT8) * math.sqrt(norms) * gamma_max
            xi = gamma_max * bounds.SQRT8 * weights.alpha1 * math.sqrt(norms)
            xi_terms = 8.0 * gamma_max * weights.alpha0 * math.sqrt(norms)
        reach = spans = fixed = 0.0
        coefficients = 0
        for channel in frame.channels:
            problem = channel.problem
            reach += float(np.sum(np.maximum(np.abs(problem.lower), np.abs(problem.upper)) ** 2))
            interval = problem.upper - problem.lower
            weight = np.broadcast_to(problem.weights, problem.lower.shape)
            spans += norm(weight * interval) ** 2
            cost = 0.0 if problem.costs is None else np.broadcast_to(problem.costs, problem.lower.shape)
            fixed += float(np.sum(0.5 * weight * interval * interval + cost * interval))
            coefficients += problem.lower.size
        values |= {
            "tau": tau,
            "sigma": sigma,
            "rho": plan.relaxation,
            "root": root_bound(tau, sigma, gamma_max, settings.method),
            "gamma_max": gamma_max,
            "lipschitz": lipschitz,
            # Where the gap is certified, every cell is one sample and nothing lies beyond the blocks.
            "dual_lipschitz": gamma_max * bounds.SQRT8 * math.sqrt(reach),
            "dual_centres": math.sqrt(spans),
            "weight_max": max(float(np.max(channel.problem.weights)) for channel in frame.channels),
            "dual_reach": math.sqrt(reach),
            "dual_fixed": fixed,
            "xi_bound": xi,
            "xi_terms_bound": xi_terms,
        }
        whole |= {
            "vector_terms": 2 * channels if coupled else 2,
            "tensor_terms": 3 * channels if coupled else 3,
            "primal_terms": 2 * channels * height * width + coefficients + 4 * channels + 16,
            "dual_terms": coefficients + 8,
            "dual_rounding": 4 if settings.method == "tv" else 2 * channels + 18,
        }
    if settings.method == "subgradient":
        values |= {"regression_canvas": SUBGRADIENT_CANVAS, "regression_primal": SUBGRADIENT_PRIMAL}
    if case.step_from:
        values["one_step"] = 1.0
    return {
        "method": settings.method,
        **{key: repr(value) for key, value in values.items()},
        **{key: str(value) for key, value in whole.items()},
    }


# Running a case.


@dataclass(slots=True)
class Run:
    """What the reference made of a case, and what bounds its rounding."""

    canvas: Array
    iterations: int
    converged: bool
    history: list[tuple[int, float, float, float, float]]
    start: dict[str, float]
    majorants: list[dict[str, float]]
    records: dict[int, dict[str, float]]
    settings: decode.Settings
    first: pdhg.FrameInitial | None = None


def model_of(image: jpegio.Image, settings: decode.Settings) -> Model:
    frame = decode.frame_of(image, settings)
    weights: TV | TGV = settings.tgv if settings.method == "tgv" else settings.tv
    data = settings.data
    tested, reference, exact = [], [], []
    for index, component in enumerate(image.components):
        term = data if isinstance(data, DataTerm) else data[index]
        levels, steps = component.coefficients, component.quant_table
        centres = exact_centres(levels, steps, term)
        computed = frame.channels[index].problem.centres
        difference = [
            abs(Decimal(value) - truth) for value, truth in zip(computed.ravel().tolist(), centres, strict=True)
        ]
        tested.append(tested_centre_bounds(levels, steps, term))
        reference.append(np.asarray([float(value) for value in difference]).reshape(computed.shape) * (1.0 + 2.0**-50))
        exact.append(centres)
    gammas = frames.channel_weights(frame, weights.channel_weights)
    return Model(frame=frame, gammas=gammas, tested=tuple(tested), reference=tuple(reference), exact=tuple(exact))


def start_norms(model_: Model, start: frames.Primal) -> dict[str, float]:
    frame = model_.frame
    si0 = scen_a = scen_r = 0.0
    for index, channel in enumerate(frame.channels):
        n = channel.cells
        si0 += n * norm(inverse_magnitude(start.coefficients[index])) ** 2
        scen_a += n * norm(model_.tested[index]) ** 2
        scen_r += n * norm(model_.reference[index]) ** 2
    return {
        "SI0": math.sqrt(si0),
        "ScenA": math.sqrt(scen_a),
        "ScenR": math.sqrt(scen_r),
        "Sgx0": norm(grad(start.canvas)),
    }


def history_of(result: FrameResult) -> list[tuple[int, float, float, float, float]]:
    h = result.history
    return [
        (int(n), float(p), float(d), float(s), float(g))
        for n, p, d, s, g in zip(h.iterations, h.primal, h.dual, h.scaling, h.partial_gap, strict=True)
    ]


def run_pdhg(case: Case, model_: Model, settings: decode.Settings) -> Run:
    """The reference's run of TV or TGV, and the bounds' norms of every iteration and record."""
    frame = model_.frame
    tgv = settings.method == "tgv"
    weights: TV | TGV = settings.tgv if tgv else settings.tv
    plan = pdhg.plan(settings.pdhg, weights)
    result = (
        pdhg.solve_frame_tgv(frame, settings.tgv, settings.pdhg)
        if tgv
        else pdhg.solve_frame_tv(frame, settings.tv, settings.pdhg)
    )
    history = history_of(result)
    due = {n for n, *_ in history}
    begin = frames.start(frame)
    s = State(x=begin.canvas, p=np.zeros((2, len(frame.channels), *frame.shape)))
    if tgv:
        s.w, s.r = grad(begin.canvas), np.zeros((3, len(frame.channels), *frame.shape))
    theta = {n: scaling for n, _, _, scaling, _ in history}
    primal = {n: p for n, p, *_ in history}

    def dual_of(n: int, s: State) -> tuple[Array, Array]:
        if isinstance(weights, TGV):
            r = s.r if n == 0 else s.r_out
            if r is None:
                message = "a state of TGV has r"
                raise ValueError(message)
            return tgv_dual(model_, r, theta[n])
        p = s.p if n == 0 else s.p_out
        if p is None:
            message = "a state of TV has p"
            raise ValueError(message)
        return tv_dual(model_, p)

    xi, terms = dual_of(0, s)
    records = {0: record_bounds(model_, begin.coefficients, terms, xi, primal[0])}
    majorants = []
    for n in range(1, result.iterations + 1):
        if isinstance(weights, TGV):
            s, m = tgv_step(model_, weights, plan, s)
        else:
            s, m = tv_step(model_, weights, plan, s)
        majorants.append(m)
        if n in due:
            xi, terms = dual_of(n, s)
            records[n] = record_bounds(model_, s.coefficients, terms, xi, primal[n])
    if s.canvas is None or not np.array_equal(s.canvas, result.primal.canvas):
        message = f"{case.name}: the iterations run again differ from the library's"
        raise RuntimeError(message)
    return Run(
        canvas=result.primal.canvas,
        iterations=result.iterations,
        converged=result.converged,
        history=history,
        start=start_norms(model_, begin),
        majorants=majorants,
        records=records,
        settings=settings,
    )


def chosen_tolerance(case: Case, model_: Model, run: Run) -> float:
    """A tolerance at which TV stops at a record where every implementation stops too (docs/math.md, 9.5).

    The record is the first after STOP_AFTER iterations whose gap, and those of every record before it, lie
    far enough from the tolerance that the gaps of two implementations, within the bound of their difference,
    fall on the same sides of it: every earlier gap above it by that bound, and the record's below it by that
    bound, with room to spare. The tolerance is the geometric mean of the two sides.
    """
    text = "\n".join(bound_lines(case, model_, run)) + "\n"
    allowed = bounds.computed(text).gap
    samples = model_.frame.samples
    gaps = {n: primal - dual for n, primal, dual, _, _ in run.history}
    records = sorted(gaps)
    for index, n in enumerate(records):
        if n < STOP_AFTER:
            continue
        above = min(gaps[j] - allowed[j] for j in records[:index])
        below = gaps[n] + allowed[n]
        if below * STOP_ROOM < above:
            return math.sqrt(below * above) / samples
    message = f"{case.name}: no tolerance stops TV with room for the rounding"
    raise RuntimeError(message)


def solve(frame: frames.Frame, settings: decode.Settings, first: pdhg.FrameInitial | None = None) -> FrameResult:
    if settings.method == "tgv":
        return pdhg.solve_frame_tgv(frame, settings.tgv, settings.pdhg, first=first)
    return pdhg.solve_frame_tv(frame, settings.tv, settings.pdhg, first=first)


def run_step(case: Case, model_: Model, settings: decode.Settings) -> Run:
    """One iteration from the reference's state after case.step_from iterations (docs/math.md, 9.3)."""
    frame = model_.frame
    before = replace(settings, pdhg=replace(settings.pdhg, iterations=case.step_from, record_every=case.step_from))
    earlier = solve(frame, before)
    if earlier.dual is None or settings.pdhg.relaxation != 1.0 or not certified(frame):
        message = f"{case.name}: one iteration starts from an unrelaxed state without free samples"
        raise ValueError(message)
    tgv = settings.method == "tgv"
    first = pdhg.FrameInitial(
        coefficients=earlier.primal.coefficients,
        p=earlier.dual.p,
        w=earlier.primal.w if tgv else None,
        r=earlier.dual.r if tgv else None,
    )
    result = solve(frame, settings, first)
    begin = frames.start(frame, first.coefficients)
    if isinstance(first.p, np.ndarray) and not tgv:
        s = State(x=begin.canvas, p=frames.project_vectors(first.p, settings.tv.alpha, coupled=settings.tv.coupled))
    elif first.p is not None and first.w is not None and first.r is not None:
        coupled = settings.tgv.coupled
        s = State(
            x=begin.canvas,
            p=frames.project_vectors(first.p, settings.tgv.alpha1, coupled=coupled),
            w=np.array(first.w),
            r=frames.project_tensors(first.r, settings.tgv.alpha0, coupled=coupled),
        )
    else:
        message = f"{case.name}: the state to start from is incomplete"
        raise ValueError(message)
    history = history_of(result)
    theta = {n: scaling for n, _, _, scaling, _ in history}
    primal = {n: p for n, p, *_ in history}
    plan = pdhg.plan(settings.pdhg, settings.tgv if tgv else settings.tv)
    if s.r is not None:
        xi, terms = tgv_dual(model_, s.r, theta[0])
    else:
        xi, terms = tv_dual(model_, s.p)
    records = {0: record_bounds(model_, begin.coefficients, terms, xi, primal[0])}
    if tgv:
        s, majorant = tgv_step(model_, settings.tgv, plan, s)
    else:
        s, majorant = tv_step(model_, settings.tv, plan, s)
    if s.r_out is not None:
        xi, terms = tgv_dual(model_, s.r_out, theta[1])
    elif s.p_out is not None:
        xi, terms = tv_dual(model_, s.p_out)
    records[1] = record_bounds(model_, s.coefficients, terms, xi, primal[1])
    if s.canvas is None or not np.array_equal(s.canvas, result.primal.canvas):
        message = f"{case.name}: the iteration run again differs from the library's"
        raise RuntimeError(message)
    start = {
        "SI0": math.sqrt(
            sum(
                channel.cells * norm(inverse_magnitude(begin.coefficients[index])) ** 2
                for index, channel in enumerate(frame.channels)
            )
        ),
        "Sp0": norm(first.p),
        "Sr0": 0.0 if first.r is None else tensor_norm(first.r),
    }
    return Run(
        canvas=result.primal.canvas,
        iterations=result.iterations,
        converged=result.converged,
        history=history,
        start=start,
        majorants=[majorant],
        records=records,
        settings=settings,
        first=first,
    )


def run_case(case: Case, image: jpegio.Image) -> tuple[Model, Run]:
    settings = case.settings
    model_ = model_of(image, settings)
    frame = model_.frame
    if settings.method == "mmse":
        begin = frames.start(frame)
        run = Run(
            canvas=begin.canvas,
            iterations=0,
            converged=False,
            history=[],
            start=start_norms(model_, begin),
            majorants=[],
            records={},
            settings=settings,
        )
        return model_, run
    if settings.method == "subgradient":
        result = decode._subgradient(frame, settings)
        records = {n: {"P": p} for n, p, *_ in history_of(result)}
        run = Run(
            result.primal.canvas, result.iterations, result.converged, history_of(result), {}, [], records, settings
        )
        return model_, run
    if case.step_from:
        return model_, run_step(case, model_, settings)
    if not case.stop:
        return model_, run_pdhg(case, model_, settings)
    search = replace(settings, pdhg=replace(settings.pdhg, tolerance=0.0))
    long = run_pdhg(case, model_, search)
    tolerance = chosen_tolerance(case, model_, long)
    stopping = replace(settings, pdhg=replace(settings.pdhg, tolerance=tolerance))
    run = run_pdhg(case, model_, stopping)
    if not run.converged:
        message = f"{case.name}: TV did not stop at the tolerance chosen"
        raise RuntimeError(message)
    return model_, run


# The file.


def numbers(values: Iterable[float]) -> str:
    return " ".join(repr(float(value)) for value in values)


def input_lines(case: Case, image: jpegio.Image, options: list[str]) -> list[str]:
    rows, columns = case.size
    lines = [
        f"name {case.name}",
        f"picture {rows} {columns} {case.colour.name.lower()}",
    ]
    for index, component in enumerate(image.components):
        levels = component.coefficients
        lines.append(
            f"component {index} {component.h_samp_factor} {component.v_samp_factor} {levels.shape[0]} {levels.shape[1]}"
        )
        lines.append(f"table {index} " + " ".join(str(int(value)) for value in component.quant_table.ravel()))
        lines += [
            f"levels {index} {row} " + " ".join(str(int(value)) for value in levels[row].ravel())
            for row in range(levels.shape[0])
        ]
    lines.append("options " + " ".join(options))
    return lines


def model_lines(model_: Model, plan: pdhg.Plan | None) -> list[str]:
    lines = []
    if plan is not None:
        tau, sigma = pdhg.steps(plan.norm_squared, plan.step_ratio, plan.step_product)
        lines.append(f"steps {tau!r} {sigma!r}")
    for index, channel in enumerate(model_.frame.channels):
        problem = channel.problem
        lines.append(f"weights {index} {numbers(problem.weights.ravel())}")
        for name, values in (("lower", problem.lower), ("upper", problem.upper), ("centres", problem.centres)):
            lines += [f"{name} {index} {row} {numbers(values[row].ravel())}" for row in range(values.shape[0])]
        exact = model_.exact[index]
        per_row = 64 * problem.lower.shape[1]
        lines += [
            f"exact-centres {index} {row} "
            + " ".join(references.pair(value) for value in exact[row * per_row : (row + 1) * per_row])
            for row in range(problem.lower.shape[0])
        ]
    return lines


def first_lines(first: pdhg.FrameInitial | None) -> list[str]:
    """The state an iteration starts from: coefficients by block row, and fields by entry, channel and row."""
    if first is None:
        return []
    lines = []
    for index, coefficients in enumerate(first.coefficients or ()):
        lines += [
            f"first-coefficients {index} {row} {numbers(coefficients[row].ravel())}"
            for row in range(coefficients.shape[0])
        ]
    for name, field in (("first-p", first.p), ("first-w", first.w), ("first-r", first.r)):
        if field is None:
            continue
        entries, channels, height = field.shape[:3]
        lines += [
            f"{name} {e} {c} {row} {numbers(field[e, c, row])}"
            for e in range(entries)
            for c in range(channels)
            for row in range(height)
        ]
    return lines


def result_lines(run: Run) -> list[str]:
    lines = [f"record {n} {p!r} {d!r} {s!r} {g!r}" for n, p, d, s, g in run.history]
    lines.append(f"stop {run.iterations} {'converged' if run.converged else 'iterations'}")
    channels, height, _ = run.canvas.shape
    lines += [f"canvas {c} {row} {numbers(run.canvas[c, row])}" for c in range(channels) for row in range(height)]
    return lines


def bound_lines(case: Case, model_: Model, run: Run) -> list[str]:
    settings = run.settings
    plan = None
    if settings.method in {"tv", "tgv"}:
        plan = pdhg.plan(settings.pdhg, settings.tgv if settings.method == "tgv" else settings.tv)
    fields = constants(case, model_, plan)
    lines = ["constants " + " ".join(f"{key}={value}" for key, value in fields.items())]
    if run.start:
        lines.append("start " + " ".join(f"{key}={value!r}" for key, value in run.start.items()))
    lines += [
        f"majorant {k} " + " ".join(f"{key}={value!r}" for key, value in sorted(m.items()))
        for k, m in enumerate(run.majorants)
    ]
    lines += [
        f"recordbound {n} " + " ".join(f"{key}={value!r}" for key, value in sorted(r.items()))
        for n, r in sorted(run.records.items())
    ]
    lines.append(f"stop {run.iterations} {'converged' if run.converged else 'iterations'}")
    return lines


HEADER: Final = """\
# A conformance case of JPEG-Unround, written by conformance/generate.py: the quantized
# coefficients of a picture and the options of its reconstruction; what the Python
# implementation made of them; the exact centres; and what bounds the rounding of that
# run, from which conformance/bounds.py computes the tolerances at the end (docs/math.md, 9).
# Each line is a keyword and its values. Numbers are decimal, each double the shortest
# decimal that reads back as it; an exact centre is hi:lo, two doubles whose sum is the
# value to about 2^-106 of it."""


def text_of(case: Case, image: jpegio.Image) -> tuple[str, decode.Settings]:
    """The case's file, and the settings it was run with (a stop's with the tolerance chosen)."""
    model_, run = run_case(case, image)
    settings = run.settings
    plan = None
    if settings.method in {"tv", "tgv"}:
        plan = pdhg.plan(settings.pdhg, settings.tgv if settings.method == "tgv" else settings.tv)
    lines = [
        HEADER,
        *input_lines(case, image, native.options(settings)),
        *model_lines(model_, plan),
        *first_lines(run.first),
        *[line for line in result_lines(run) if not line.startswith("stop ")],
        *bound_lines(case, model_, run),
    ]
    return bounds.with_tolerances("\n".join(lines) + "\n"), settings


# The reference implementation against the expected values.


def compare(case: Case, image: jpegio.Image, settings: decode.Settings, text: str) -> str:
    """How far the reference implementation (Rust) lies from the expected values, against the tolerances.

    Its C interface starts where the method does by default: a case of one iteration from a state given is left
    to the tests of the implementations, which give it that state.
    """
    if case.step_from:
        return f"{case.name:22} starts from a state given, which the C interface does not take"
    tolerances = bounds.computed(text)
    decoded = native.solve(image, settings)
    lines = text.splitlines()
    expected = np.asarray([[float(v) for v in line.split()[3:]] for line in lines if line.startswith("canvas ")])
    distance = norm(decoded.canvas - expected.reshape(decoded.canvas.shape))
    worst = 0.0
    if decoded.result is not None and tolerances.primal:
        recorded = {int(line.split()[1]): float(line.split()[2]) for line in lines if line.startswith("record ")}
        history = decoded.result.history
        for n, primal in zip(history.iterations, history.primal, strict=True):
            worst = max(worst, abs(float(primal) - recorded[int(n)]) / tolerances.primal[int(n)])
    return (
        f"{case.name:22} canvas {distance:9.2e} of the {tolerances.canvas:9.2e} allowed; "
        f"primal values at {worst:.2e} of theirs"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", help="cases, by name (all by default)")
    parser.add_argument("--out", type=Path, default=ROOT / "cases")
    parser.add_argument("--check", action="store_true", help="compare with the files rather than write them")
    parser.add_argument("--native", action="store_true", help="compare the reference implementation with the cases")
    options = parser.parse_args()
    chosen = [case for case in CASES if not options.names or case.name in options.names]
    failures = 0
    for case in chosen:
        path = options.out / f"{case.name}.txt"
        image = image_of(case)
        text, settings = text_of(case, image)
        if options.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != text:
                print(f"error: {path.as_posix()} differs from what the Python implementation gives")
                failures += 1
        else:
            options.out.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8", newline="\n")
            print(f"{path.as_posix()}: {len(text.splitlines())} lines")
        if options.native:
            print(compare(case, image, settings, text))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
