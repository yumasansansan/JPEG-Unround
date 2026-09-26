#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""The tolerances of the conformance cases, from bounds of rounding error (docs/math.md, 9).

    python conformance/bounds.py [--check] [case ...]

A case file holds, besides the input and the expected values, what the bounds of its
rounding depend on: the constants of the case ("constants"), the norms of the magnitudes
that bound the rounding of the start ("start") and of every iteration of the reference run
("majorant"), and what the records' values depend on ("recordbound"). From these, this
module computes how far a conforming implementation may lie from the expected values: its
canvas after the last iteration, the primal value of every record, and, where the gap is
certified, how far one implementation's dual value may lie above the other's primal
value, and how far the two gaps may lie apart. conformance/generate.py appends them to a
case as "tolerance" lines; --check computes them again and compares.

The subgradient method has no such bound (docs/math.md, 9.6): its cases hold a tolerance
measured as a regression check, which is kept as it is.

The norms are those of the reference's values. The fields of an implementation under test
lie within DIFFERENCE of the reference's, as every bound computed here confirms, and so its
norms within MARGIN of the reference's, as magnitudes() confirms. This module's own
arithmetic is binary64; the factor INFLATION covers its rounding.
"""

import argparse
import math
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final

ROOT: Final = Path(__file__).resolve().parent
U: Final = 2.0**-53
SQRT8: Final = math.sqrt(8.0) * (1.0 + 2.0**-50)
SQRT2: Final = math.sqrt(2.0) * (1.0 + 2.0**-50)
INFLATION: Final = 1.0 + 1e-9
"""Covers the rounding of this module's sums, products and square roots, and of the norms of the case."""
DIFFERENCE: Final = 1e-3
"""The largest difference, in the Euclidean norm of any field, from the reference's values that the bounds assume."""
MARGIN: Final = 1.0
"""What the norms of the magnitudes of an implementation under test may exceed the reference's by: at least
magnitudes() times DIFFERENCE."""


def gamma(k: int) -> float:
    """Higham's gamma_k = k u / (1 - k u), the relative bound of k roundings; 0 for k <= 0."""
    return 0.0 if k <= 0 else k * U / (1.0 - k * U)


EPS2_REFERENCE: Final = gamma(9) * (2.0 + gamma(9))
"""The block DCT of the reference: NumPy's products with the 8 x 8 basis, each 1-D output a sum of eight
products in any order, the basis within u of the exact one."""
EPS2_TESTED: Final = gamma(6) * (2.0 + gamma(6))
"""The block DCT of docs/math.md, 1.1, by the even and odd halves, which a conforming implementation computes."""


@dataclass(frozen=True, slots=True)
class Constants:
    """What a case's bounds depend on besides the norms of its start and iterations."""

    method: str
    tau: float = 1.0
    sigma: float = 1.0
    rho: float = 1.0
    root: float = 0.0
    """An upper bound of sqrt(sigma tau ||K||^2), the square root of the product of the steps."""
    gamma_max: float = 1.0
    vector_terms: int = 2
    """The terms of the squared norm of a projected vector: 2 C coupled, 2 apart."""
    tensor_terms: int = 3
    """Those of a projected tensor: 3 C coupled, 3 apart."""
    cells: int = 1
    """The most samples of a cell."""
    lipschitz: float = 0.0
    """Of the primal value in the canvas (and TGV's field), jointly, in the Euclidean norm."""
    dual_lipschitz: float = 0.0
    """Of TV's dual value in the dual field p."""
    dual_centres: float = 0.0
    """Of the dual value in the centres."""
    weight_max: float = 0.0
    """The largest weight mu omega of the data term."""
    centres_tested: float = 0.0
    """A bound of the Euclidean distance of a tested implementation's centres from the exact ones."""
    centres_reference: float = 0.0
    """The Euclidean distance of the reference's centres from the exact ones."""
    primal_terms: int = 0
    dual_terms: int = 0
    dual_rounding: int = 0
    """The roundings of the field whose DCT the dual value takes, relative to its magnitudes."""
    dual_reach: float = 0.0
    """The Euclidean norm of max(|a|, |b|) over the coefficients."""
    dual_fixed: float = 0.0
    """The sum of mu omega (b - a)^2 / 2 and lambda (b - a) over the coefficients."""
    xi_bound: float = 0.0
    """A bound of the Euclidean norm of the field whose conjugate the dual value takes, for any dual field."""
    xi_terms_bound: float = 0.0
    """A bound of that of the magnitudes of its terms."""
    regression_canvas: float = 0.0
    regression_primal: float = 0.0
    one_step: float = 0.0
    """1 for a case of one iteration from a state given (docs/math.md, 9.3), 0 otherwise."""


WHOLE: Final = frozenset({"vector_terms", "tensor_terms", "cells", "primal_terms", "dual_terms", "dual_rounding"})


def constants_of(fields: Mapping[str, str]) -> Constants:
    """The constants of a "constants" line's fields."""
    changes: dict[str, Any] = {
        key: int(value) if key in WHOLE else float(value) for key, value in fields.items() if key != "method"
    }
    return replace(Constants(method=fields["method"]), **changes)


@dataclass(frozen=True, slots=True)
class Local:
    """The Euclidean bounds of the local errors of one iteration of one implementation (docs/math.md, 9.3).

    x, w, p and r are those of the proximal outputs, the tilde point; zeta that of the proximal map's
    output coefficients; state and tilde the bounds, in the norm of the method's metric, of the next
    state's error and of the tilde point's.
    """

    x: float
    w: float
    p: float
    r: float
    zeta: float
    state: float
    tilde: float


def local(c: Constants, m: Mapping[str, float], *, tested: bool) -> Local:
    """The bounds of one iteration of an implementation, from the norms m of the reference's values."""
    extra = MARGIN if tested else 0.0

    def g(key: str) -> float:
        return m.get(key, 0.0) + extra

    eps2 = EPS2_TESTED if tested else EPS2_REFERENCE
    centres = m["McenA"] if tested else m["McenR"]
    v = U * g("Mx") + gamma(6) * g("Md")
    zeta = v + eps2 * g("Mf") + gamma(8) * g("Mphi") + centres + 2.0 * gamma(c.cells - 1) * g("Mmean")
    x = zeta + eps2 * g("Mi") + U * g("Mput")
    if c.method == "tv":
        w = r = 0.0
        p = (
            2.0 * c.sigma * c.gamma_max * SQRT8 * x
            + SQRT8 * U * g("Mq")
            + U * g("Ma")
            + gamma(3) * g("Mg")
            + gamma(c.vector_terms + 4) * g("Mp")
        )
    else:
        w = U * g("Mw") + gamma(7) * g("Mh")
        p = (
            c.sigma * c.gamma_max * (2.0 * SQRT8 * x + 2.0 * w)
            + SQRT8 * U * g("Mq")
            + U * g("Mwbar")
            + gamma(4) * g("Mg")
            + U * g("Ma")
            + gamma(c.vector_terms + 4) * g("Mp")
        )
        r = (
            c.sigma * c.gamma_max * 2.0 * SQRT8 * w
            + SQRT8 * U * g("Mwbar")
            + gamma(5) * g("ME")
            + U * g("Mb")
            + gamma(c.tensor_terms + 4) * g("Mr")
        )
    if c.rho == 1.0:
        relaxed = (x, w, p, r)
    else:
        relaxed = (
            c.rho * x + gamma(2) * g("Mrx"),
            c.rho * w + gamma(2) * g("Mrw"),
            c.rho * p + gamma(2) * g("Mrp"),
            c.rho * r + gamma(2) * g("Mrr"),
        )
    state = metric(c, math.hypot(relaxed[0], relaxed[1]), math.hypot(relaxed[2], relaxed[3]))
    tilde = metric(c, math.hypot(x, w), math.hypot(p, r))
    return Local(x=x, w=w, p=p, r=r, zeta=zeta, state=state, tilde=tilde)


def magnitudes(c: Constants) -> float:
    """How far a norm of magnitudes of an iteration may move per unit of the largest difference of the states,
    the tilde points and the coefficients (docs/math.md, 9.3).

    Each is the norm of a map of nonnegative weights, of norm at most 8 (the DCT's, |C| (x) |C|),
    tau gamma (1 + 2 sqrt 2) (the terms of the divergences) or sigma gamma sqrt 8 (those of the symmetrized
    gradient, and the weights), applied to the magnitudes of a field that moves by at most 3 + tau gamma sqrt 8
    (v, the input of the proximal map, and the change put on the canvas), 3 sqrt 8 + 3 (the extrapolations and
    their gradients), 1 + sigma gamma (3 sqrt 8 + 3) (the ascents) or rho + |1 - rho| (the relaxed states)
    times that difference.
    """
    tg, sg = c.tau * c.gamma_max, c.sigma * c.gamma_max
    maps = max(8.0, tg * (1.0 + 2.0 * SQRT2), sg * SQRT8)
    moves = max(3.0 + tg * SQRT8, 3.0 * SQRT8 + 3.0, 1.0 + sg * (3.0 * SQRT8 + 3.0), c.rho + abs(1.0 - c.rho))
    return maps * moves * INFLATION


def metric(c: Constants, primal: float, dual: float) -> float:
    """A bound of the metric's norm of (z, y) from bounds of ||z|| and ||y||: with a = ||z|| / sqrt(tau) and
    b = ||y|| / sqrt(sigma), ||(z, y)||_M^2 = a^2 + b^2 - 2 <K z, y> <= a^2 + b^2 + 2 sqrt(theta) a b."""
    a, b = primal / math.sqrt(c.tau), dual / math.sqrt(c.sigma)
    return math.sqrt(a * a + b * b + 2.0 * c.root * a * b) * INFLATION


@dataclass(frozen=True, slots=True)
class Start:
    """How far the two implementations' starts may lie apart: the canvas, TGV's field, the metric's norm."""

    x: float
    w: float
    metric: float


def start(c: Constants, s: Mapping[str, float]) -> Start:
    """docs/math.md, 9.2."""
    canvas, field_w, norm = 0.0, 0.0, 0.0
    for tested in (True, False):
        eps2 = EPS2_TESTED if tested else EPS2_REFERENCE
        x0 = eps2 * s["SI0"] + (s["ScenA"] if tested else s["ScenR"])
        w0 = SQRT8 * x0 + U * s["Sgx0"] if c.method == "tgv" else 0.0
        canvas += x0
        field_w += w0
        norm += metric(c, math.hypot(x0, w0), 0.0)
    return Start(x=canvas * INFLATION, w=field_w * INFLATION, metric=norm * INFLATION)


@dataclass(frozen=True, slots=True)
class Tolerances:
    """What a case allows: the canvas after the last iteration, and the values of its records."""

    canvas: float
    primal: dict[int, float] = field(default_factory=dict)
    bracket: dict[int, float] = field(default_factory=dict)
    gap: dict[int, float] = field(default_factory=dict)
    largest: float = 0.0
    """The largest difference of any field that the bounds allow, which must stay within DIFFERENCE."""


def evaluation_primal(c: Constants, primal: float) -> float:
    """How far the two computed primal values, sums of terms of at least 0, may lie from their exact ones together:
    the tested one within 1 of the reference's, as its tolerance, at most 1, allows."""
    return gamma(c.primal_terms) * (2.0 * abs(primal) + 1.0) * INFLATION


def evaluation_dual(c: Constants, record: Mapping[str, float]) -> float:
    """How far the two computed dual values may each lie from their exact ones (docs/math.md, 9.4): the
    reference's by the magnitudes of its terms, and a tested implementation's by their bounds for any dual
    field in its balls, |s c*| at most |s| max(|a|, |b|) and the rest at most mu omega (b - a)^2 / 2 and
    lambda (b - a)."""
    rounding = gamma(c.dual_rounding)
    reference = gamma(c.dual_terms) * record["Dabs"] + (EPS2_REFERENCE + rounding) * record["Zdot"]
    tested = (
        gamma(c.dual_terms) * (c.dual_reach * c.xi_bound + c.dual_fixed)
        + (EPS2_TESTED + rounding) * 8.0 * c.dual_reach * c.xi_terms_bound
    )
    return max(reference, tested) * INFLATION


def tolerances(
    c: Constants,
    beginning: Mapping[str, float],
    majorants: Mapping[int, Mapping[str, float]],
    records: Mapping[int, Mapping[str, float]],
    iterations: int,
) -> Tolerances:
    """The tolerances of a case (docs/math.md, 9.2 to 9.5)."""
    if c.method == "subgradient":
        regression = {n: c.regression_primal * (abs(record["P"]) + 1.0) for n, record in records.items()}
        return Tolerances(canvas=c.regression_canvas, primal=regression)
    if c.one_step:
        return one_step(c, beginning, majorants[0], records)
    first = start(c, beginning)
    if c.method == "mmse":
        return Tolerances(canvas=first.x, largest=first.x)
    # ||z||^2 <= tau ||(z, y)||_M^2 / (1 - theta), and ||y||^2 <= sigma ||(z, y)||_M^2 / (1 - theta):
    # the least of the metric's norm over y for a given z is at y = sigma K z (docs/math.md, 9.1).
    down = 1.0 / math.sqrt(1.0 - c.root * c.root)
    # state[n] bounds the metric's norm of the two states' difference after n iterations, and
    # tilde[n] that of the two tilde points of iteration n, computed from the states after n - 1.
    state = [first.metric]
    tilde: dict[int, float] = {}
    zeta: dict[int, float] = {}
    for k in range(iterations):
        tested, reference = local(c, majorants[k], tested=True), local(c, majorants[k], tested=False)
        tilde[k + 1] = (state[k] + tested.tilde + reference.tilde) * INFLATION
        # The input of the proximal map, x + tau gamma div p, moves by at most ||dx|| + tau gamma sqrt(8) ||dp||,
        # which is (1 + sqrt(theta)) sqrt(tau / (1 - theta)) times the states' distance in the metric.
        moved = (1.0 + c.root) * math.sqrt(c.tau) * down * state[k]
        zeta[k + 1] = (moved + tested.zeta + reference.zeta) * INFLATION
        state.append((state[k] + tested.state + reference.state) * INFLATION)
    centres = (c.centres_tested + c.centres_reference) * INFLATION
    result = Tolerances(canvas=0.0)
    largest = first.x + first.w
    for n, record in records.items():
        if n == 0:
            joint, coefficients, dual_field = math.hypot(first.x, first.w), centres, 0.0
        else:
            joint = math.sqrt(c.tau) * down * tilde[n]
            coefficients = zeta[n]
            dual_field = math.sqrt(c.sigma) * down * tilde[n]
        primal = (
            c.lipschitz * joint
            + (record["Lzeta"] + c.weight_max * coefficients) * coefficients
            + (record["Lcen"] + c.weight_max * centres) * centres
            + evaluation_primal(c, record["P"])
        ) * INFLATION
        result.primal[n] = primal
        if record["Dabs"] >= 0.0:
            dual = evaluation_dual(c, record)
            result.bracket[n] = (evaluation_primal(c, record["P"]) + dual) * INFLATION
            if c.method == "tv":
                moved = c.dual_lipschitz * dual_field + c.dual_centres * centres + 2.0 * dual
                result.gap[n] = (primal + moved) * INFLATION
        largest = max(largest, joint, coefficients, dual_field)
    canvas = math.sqrt(c.tau) * down * tilde[iterations] * INFLATION if iterations > 0 else first.x
    spread = math.sqrt(max(c.tau, c.sigma, 1.0)) * down * max(state) * 8.0
    return Tolerances(
        canvas=canvas,
        primal=result.primal,
        bracket=result.bracket,
        gap=result.gap,
        largest=max(largest, canvas, spread),
    )


def one_step(
    c: Constants, s: Mapping[str, float], m: Mapping[str, float], records: Mapping[int, Mapping[str, float]]
) -> Tolerances:
    """The tolerances of one iteration from a state given (docs/math.md, 9.3): the implementations start
    within the rounding of the inverse DCT of its coefficients and of the projections of its dual fields,
    and each proximal map, projection and difference of the iteration is Lipschitz with the constant 1, or
    that of the operator, in the Euclidean norm."""
    tested, reference = local(c, m, tested=True), local(c, m, tested=False)
    x0 = (EPS2_TESTED + EPS2_REFERENCE) * s["SI0"]
    p0 = 2.0 * gamma(c.vector_terms + 4) * s["Sp0"]
    r0 = 2.0 * gamma(c.tensor_terms + 4) * s["Sr0"]
    v = x0 + c.tau * c.gamma_max * SQRT8 * p0
    x1 = v + tested.x + reference.x
    zeta1 = v + tested.zeta + reference.zeta
    # w is given exactly; w~ moves with p and r, and w-bar = 2 w~ - w twice as much.
    moved = c.tau * c.gamma_max * (p0 + SQRT8 * r0) if c.method == "tgv" else 0.0
    w1 = moved + tested.w + reference.w if c.method == "tgv" else 0.0
    p1 = p0 + c.sigma * c.gamma_max * (SQRT8 * (2.0 * v + x0) + 2.0 * moved) + tested.p + reference.p
    r1 = r0 + c.sigma * c.gamma_max * SQRT8 * 2.0 * w1 + tested.r + reference.r if c.method == "tgv" else 0.0
    result = Tolerances(canvas=x1 * INFLATION)
    for n, record in records.items():
        joint, coefficients, dual_field = (
            (x0, 0.0, math.hypot(p0, r0))
            if n == 0
            else (
                math.hypot(x1, w1),
                zeta1,
                math.hypot(p1, r1),
            )
        )
        primal = (
            c.lipschitz * joint
            + (record["Lzeta"] + c.weight_max * coefficients) * coefficients
            + evaluation_primal(c, record["P"])
        ) * INFLATION
        result.primal[n] = primal
        if record["Dabs"] >= 0.0:
            dual = evaluation_dual(c, record)
            result.bracket[n] = (evaluation_primal(c, record["P"]) + dual) * INFLATION
            if c.method == "tv":
                result.gap[n] = (primal + c.dual_lipschitz * dual_field + 2.0 * dual) * INFLATION
    return Tolerances(
        canvas=result.canvas,
        primal=result.primal,
        bracket=result.bracket,
        gap=result.gap,
        largest=max(x1, w1, zeta1, p1, r1),
    )


@dataclass(slots=True)
class Parsed:
    """What a case file holds for its bounds, and its lines without the tolerances."""

    constants: Constants | None = None
    beginning: dict[str, float] = field(default_factory=dict)
    majorants: dict[int, dict[str, float]] = field(default_factory=dict)
    records: dict[int, dict[str, float]] = field(default_factory=dict)
    iterations: int = 0
    kept: list[str] = field(default_factory=list)


def parse(text: str) -> Parsed:
    parsed = Parsed()
    for line in text.splitlines():
        fields = line.split()
        if fields and fields[0] == "tolerance":
            continue
        parsed.kept.append(line)
        if not fields:
            continue
        pairs = dict(item.split("=", 1) for item in fields[1:] if "=" in item)
        match fields[0]:
            case "constants":
                parsed.constants = constants_of(pairs)
            case "start":
                parsed.beginning = {key: float(value) for key, value in pairs.items()}
            case "majorant":
                parsed.majorants[int(fields[1])] = {key: float(value) for key, value in pairs.items()}
            case "recordbound":
                parsed.records[int(fields[1])] = {key: float(value) for key, value in pairs.items()}
            case "stop":
                parsed.iterations = int(fields[1])
            case _:
                pass
    return parsed


def tolerance_lines(result: Tolerances) -> list[str]:
    lines = [f"tolerance canvas {result.canvas!r}"]
    for n in sorted(result.primal):
        bracket = result.bracket.get(n, -1.0)
        gap = result.gap.get(n, -1.0)
        lines.append(f"tolerance record {n} {result.primal[n]!r} {bracket!r} {gap!r}")
    return lines


def computed(text: str) -> Tolerances:
    parsed = parse(text)
    if parsed.constants is None:
        message = "a case file has its constants"
        raise ValueError(message)
    c = parsed.constants
    result = tolerances(c, parsed.beginning, parsed.majorants, parsed.records, parsed.iterations)
    if not result.largest <= DIFFERENCE:
        message = f"the bounds allow differences of {result.largest:.3e}, beyond the {DIFFERENCE} they assume"
        raise ValueError(message)
    if c.method in {"tv", "tgv"} and not magnitudes(c) * DIFFERENCE <= MARGIN:
        message = f"the norms of magnitudes may move by {magnitudes(c) * DIFFERENCE:.3e}, beyond the {MARGIN} added"
        raise ValueError(message)
    if not all(value <= 1.0 for value in result.primal.values()):
        message = "a primal value's tolerance exceeds the 1 that the bounds of its evaluation assume"
        raise ValueError(message)
    return result


def with_tolerances(text: str) -> str:
    """The case file's text with its tolerance lines computed again, at its end."""
    result = computed(text)
    kept = parse(text).kept
    while kept and not kept[-1]:
        kept.pop()
    return "\n".join([*kept, *tolerance_lines(result)]) + "\n"


def cases(names: Iterable[str]) -> list[Path]:
    folder = ROOT / "cases"
    chosen = list(names)
    if chosen:
        return [folder / f"{name}.txt" for name in chosen]
    return sorted(folder.glob("*.txt"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", help="cases, by name (all by default)")
    parser.add_argument("--check", action="store_true", help="compare with the files rather than write them")
    options = parser.parse_args()
    failures = 0
    for path in cases(options.names):
        text = path.read_text(encoding="utf-8")
        again = with_tolerances(text)
        if options.check:
            if again != text:
                print(f"error: {path.as_posix()} differs from the tolerances its bounds give")
                failures += 1
        elif again != text:
            path.write_text(again, encoding="utf-8", newline="\n")
            print(f"{path.as_posix()}: tolerances written")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
