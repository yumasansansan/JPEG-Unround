#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Phase 2: the primal-dual method relaxed, and its steps and tolerance chosen again on the tuning images.

    python experiments/phase2_solver.py --build build/release [--data data] [--workers 10]
        [--cache data/phase2/solver] [--out experiments/results] [stage ...]

The files are those of experiments/phase1_tuning.py: the greyscale tuning images as
libjpeg-turbo encoded them, all 48 for TV and the 24 of the first image of each kind
for TGV, with the weights of unround.decode.Settings. The stages:

  variants    On three of the files, with Phase 1's ratios of the steps: the method
              relaxed with the rho of RELAXATIONS (docs/math.md, 5); TGV started from
              TV's solution (w = 0, TV's p); and adaptive steps (Goldstein, Li and Yuan),
              with the proportions of SCALES. The last two are not in the package: they
              are written here, as they were tried.
  references  Each file for long, relaxed: its last point stands for the least point.
  stops       Each file with each variant of VARIANTS (a ratio of the steps and a
              relaxation), until the gap per sample is within the least of TOLERANCES, or
              for the references' iterations. The result is measured exactly where the gap
              first falls within each tolerance.
  report      The tables and the figures, and the variant and the tolerance they choose.
              Stopping at a tolerance changes the result, against the reference's last
              point: the criterion is on the distribution of those changes over the files
              (LIMITS), for tolerances that every file reaches and that the references can
              tell (their median last gap at most a tenth of it). Phase 1 held every file to
              0.01 dB and 1e-4; against these references, which are nearer the least points
              and not on the stops' own paths, no tolerance tried met that, and the largest
              changes did not fall as the tolerance did (docs/math.md, 6.4).
  test        With the package's defaults, which are the choice: TV and TGV on
              libjpeg-turbo's greyscale test files, as experiments/phase1_comparison.py
              runs them, and the time of an iteration, one run after another in this
              process. The report sets them beside Phase 1's (--phase1-test).

A run's results are kept as JSON in --cache, and a stage skips the runs whose results
are there already. No random number is drawn: the runs are deterministic.
"""

import argparse
import dataclasses
import math
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import numpy.typing as npt

import phase1_common as common
import phase1_comparison as comparison
import phase1_tuning as tuning

if TYPE_CHECKING:
    # For the annotations alone: the package is imported once its path is set.
    from unround.decode import Settings
    from unround.model import Problem

type Array = npt.NDArray[np.float64]
type Variant = tuple[float, float]  # the ratio of the steps, and the relaxation

MODELS: Final = tuning.MODELS
MODEL_NAMES: Final = tuning.MODEL_NAMES

# The variants stage: three files, Phase 1's ratios and how long.
VARIANT_FILES: Final = (("chart-0-grey", 10), ("text-0-grey", 50), ("illustration-0-grey", 30))
VARIANT_RATIOS: Final = {"tv": 30.0, "tgv": 3.0}
VARIANT_ITERATIONS: Final = {"tv": 2500, "tgv": 4000}
RELAXATIONS: Final = (1.0, 1.5, 1.9)
WARM_RELAXATIONS: Final = (1.0, 1.9)
SCALES: Final = (0.3, 1.0, 3.0)
# The adaptive steps: the first change, its decay, and the band of proportions left alone.
ADAPT_FIRST: Final = 0.5
ADAPT_DECAY: Final = 0.95
ADAPT_BAND: Final = 1.5
VARIANT_MARKS: Final = (1000, 2000, 3000, 4000)
VARIANT_TOLERANCES: Final = {"tv": (1e-3, 5e-4), "tgv": (2e-2, 1e-2)}

# The references, the variants whose stops are measured, and the tolerances.
REFERENCES: Final[dict[str, Variant]] = {"tv": (30.0, 1.5), "tgv": (3.0, 1.9)}
REFERENCE_ITERATIONS: Final = {"tv": 8000, "tgv": 12000}
VARIANTS: Final[dict[str, tuple[Variant, ...]]] = {
    "tv": ((10.0, 1.5), (10.0, 1.9), (30.0, 1.5), (30.0, 1.9)),
    "tgv": ((3.0, 1.5), (3.0, 1.9), (10.0, 1.9)),
}
TOLERANCES: Final = {"tv": (1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4), "tgv": (5e-2, 2e-2, 1e-2, 5e-3, 2e-3)}

# The criterion: the median and the 90th percentile over the files of the change that
# stopping makes, of the PSNR of the binary64 result (dB) and of the SSIM of its 8-bit samples.
LIMITS: Final = {"psnr": (1e-3, 5e-3), "ssim_8": (1e-5, 5e-5)}
PERCENTILE: Final = 90.0
TELL: Final = 10.0  # a tolerance is told when it is this many times the references' median last gap

STAGES: Final = ["variants", "references", "stops", "test", "report"]
RECORD_EVERY: Final = 10


@dataclasses.dataclass(frozen=True, slots=True)
class Trial:
    """A run of the variants stage: a file, a model, and what is tried."""

    case: common.Case
    model: str
    kind: str  # "relaxed", "warm" (TGV from TV's solution) or "adaptive"
    parameter: float  # the relaxation, or the adaptive steps' proportion

    @property
    def name(self) -> str:
        """The name of its results."""
        return f"{self.model}-{self.kind}-{self.parameter:g}-{self.case.name}"


def record_psnr(original: npt.NDArray[np.uint8], canvas: Array, height: int, width: int) -> float:
    """The PSNR of a canvas's picture against the original."""
    from unround import metrics  # noqa: PLC0415

    return metrics.psnr(original, canvas[:height, :width])


def adapt(steps: tuple[float, float, float], primal: float, dual: float, scale: float) -> tuple[float, float, float]:
    """Goldstein, Li and Yuan's change of the steps (tau, sigma, a) from the residuals of an iteration.

    Where the primal residual is more than ADAPT_BAND times scale times the dual one, tau
    grows by 1 / (1 - a) and sigma shrinks by 1 - a; where it is less than scale /
    ADAPT_BAND times, the other way; either way a then shrinks by ADAPT_DECAY. tau sigma
    stays what it was.
    """
    tau, sigma, a = steps
    if primal > ADAPT_BAND * scale * dual:
        return tau / (1.0 - a), sigma * (1.0 - a), a * ADAPT_DECAY
    if primal < scale * dual / ADAPT_BAND:
        return tau * (1.0 - a), sigma / (1.0 - a), a * ADAPT_DECAY
    return steps


@dataclasses.dataclass(slots=True)
class Trace:
    """What a trial records: the values every RECORD_EVERY iterations, and the PSNR at the marks."""

    iterations: list[int] = dataclasses.field(default_factory=list)
    primal: list[float] = dataclasses.field(default_factory=list)
    dual: list[float] = dataclasses.field(default_factory=list)
    psnr: dict[str, float] = dataclasses.field(default_factory=dict)
    ratios: list[float] = dataclasses.field(default_factory=list)


def adaptive_tv(problem: Problem, original: npt.NDArray[np.uint8], height: int, width: int, scale: float) -> Trace:
    """TV by the unrelaxed method (docs/math.md, 5), its steps adapted every RECORD_EVERY iterations."""
    from unround import dct, pdhg  # noqa: PLC0415
    from unround.model import Dual, Primal, prox, tv_values  # noqa: PLC0415
    from unround.operators import div, grad, project_vectors  # noqa: PLC0415

    settings = decode_settings()
    tau, sigma = pdhg.steps(pdhg.TV_NORM_SQUARED, VARIANT_RATIOS["tv"])
    state = (tau, sigma, ADAPT_FIRST)
    x = pdhg.start(problem).canvas
    p = np.zeros((2, *problem.shape))
    trace = Trace()
    for iteration in range(1, VARIANT_ITERATIONS["tv"] + 1):
        tau, sigma, _ = state
        coefficients = prox(problem, dct.forward(x + tau * div(p)), tau)
        canvas = dct.inverse(coefficients)
        p_out = project_vectors(p + sigma * grad(2.0 * canvas - x), settings.tv.alpha)
        if iteration % RECORD_EVERY == 0:
            values = tv_values(problem, settings.tv, Primal(coefficients, canvas), Dual(p_out))
            trace.iterations.append(iteration)
            trace.primal.append(values[0])
            trace.dual.append(values[1])
            if iteration in VARIANT_MARKS:
                trace.psnr[str(iteration)] = record_psnr(original, canvas, height, width)
            dx, dp = x - canvas, p - p_out
            primal = math.sqrt(float(np.sum((dx / tau + div(dp)) ** 2)))
            dual = math.sqrt(float(np.sum((dp / sigma - grad(dx)) ** 2)))
            state = adapt(state, primal, dual, scale)
            trace.ratios.append(state[0] / state[1])
        x, p = canvas, p_out
    return trace


def adaptive_tgv(problem: Problem, original: npt.NDArray[np.uint8], height: int, width: int, scale: float) -> Trace:
    """TGV by the unrelaxed method, its steps adapted every RECORD_EVERY iterations."""
    from unround import dct, pdhg  # noqa: PLC0415
    from unround.model import Dual, Primal, prox, tgv_values  # noqa: PLC0415
    from unround.operators import (  # noqa: PLC0415
        div,
        div2,
        grad,
        project_tensors,
        project_vectors,
        sym_grad,
        tensor_inner,
    )

    settings = decode_settings()
    weights = settings.tgv
    tau, sigma = pdhg.steps(pdhg.TGV_NORM_SQUARED, VARIANT_RATIOS["tgv"])
    state = (tau, sigma, ADAPT_FIRST)
    x = pdhg.start(problem).canvas
    w = grad(x)
    p, r = np.zeros((2, *problem.shape)), np.zeros((3, *problem.shape))
    trace = Trace()
    for iteration in range(1, VARIANT_ITERATIONS["tgv"] + 1):
        tau, sigma, _ = state
        coefficients = prox(problem, dct.forward(x + tau * div(p)), tau)
        canvas = dct.inverse(coefficients)
        w_out = w + tau * (p + div2(r))
        x_bar, w_bar = 2.0 * canvas - x, 2.0 * w_out - w
        p_out = project_vectors(p + sigma * (grad(x_bar) - w_bar), weights.alpha1)
        r_out = project_tensors(r + sigma * sym_grad(w_bar), weights.alpha0)
        if iteration % RECORD_EVERY == 0:
            values = tgv_values(problem, weights, Primal(coefficients, canvas, w_out), Dual(p_out, r_out))
            trace.iterations.append(iteration)
            trace.primal.append(values[0])
            trace.dual.append(values[1])
            if iteration in VARIANT_MARKS:
                trace.psnr[str(iteration)] = record_psnr(original, canvas, height, width)
            dx, dw, dp, dr = x - canvas, w - w_out, p - p_out, r - r_out
            primal_x, primal_w = dx / tau + div(dp), dw / tau + dp + div2(dr)
            dual_p, dual_r = dp / sigma - (grad(dx) - dw), dr / sigma - sym_grad(dw)
            primal = math.sqrt(float(np.sum(primal_x**2) + np.sum(primal_w**2)))
            dual = math.sqrt(float(np.sum(dual_p**2)) + tensor_inner(dual_r, dual_r))
            state = adapt(state, primal, dual, scale)
            trace.ratios.append(state[0] / state[1])
        x, w, p, r = canvas, w_out, p_out, r_out
    return trace


def warm_tgv(problem: Problem, original: npt.NDArray[np.uint8], height: int, width: int, rho: float) -> Trace:
    """TGV from TV's solution (by the package's defaults): x and p as TV left them, w and r at 0."""
    from unround import dct, pdhg  # noqa: PLC0415
    from unround.model import Dual, Primal, prox, tgv_values  # noqa: PLC0415
    from unround.operators import div, div2, grad, project_tensors, project_vectors, sym_grad  # noqa: PLC0415

    settings = decode_settings()
    weights = settings.tgv
    tv = pdhg.solve_tv(problem, settings.tv, pdhg.Options(relaxation=1.0))
    if tv.dual is None:
        message = "the primal-dual method returned no dual point"
        raise RuntimeError(message)
    tau, sigma = pdhg.steps(pdhg.TGV_NORM_SQUARED, VARIANT_RATIOS["tgv"])
    x = tv.primal.canvas
    w = np.zeros((2, *problem.shape))
    p, r = tv.dual.p.copy(), np.zeros((3, *problem.shape))
    trace = Trace()
    trace.ratios.append(float(tv.iterations))  # the iterations TV took first
    for iteration in range(1, VARIANT_ITERATIONS["tgv"] + 1):
        coefficients = prox(problem, dct.forward(x + tau * div(p)), tau)
        canvas = dct.inverse(coefficients)
        w_out = w + tau * (p + div2(r))
        x_bar, w_bar = 2.0 * canvas - x, 2.0 * w_out - w
        p_out = project_vectors(p + sigma * (grad(x_bar) - w_bar), weights.alpha1)
        r_out = project_tensors(r + sigma * sym_grad(w_bar), weights.alpha0)
        if iteration % RECORD_EVERY == 0:
            values = tgv_values(problem, weights, Primal(coefficients, canvas, w_out), Dual(p_out, r_out))
            trace.iterations.append(iteration)
            trace.primal.append(values[0])
            trace.dual.append(values[1])
            if iteration in VARIANT_MARKS:
                trace.psnr[str(iteration)] = record_psnr(original, canvas, height, width)
        x = rho * canvas + (1.0 - rho) * x
        w = rho * w_out + (1.0 - rho) * w
        p = rho * p_out + (1.0 - rho) * p
        r = rho * r_out + (1.0 - rho) * r
    return trace


def decode_settings() -> Settings:
    """unround.decode.Settings(): the weights of the model."""
    from unround import decode  # noqa: PLC0415

    return decode.Settings()


def run_trial(trial: Trial) -> dict[str, Any]:
    """Runs a trial of the variants stage."""
    from unround import decode, jpegio  # noqa: PLC0415

    settings = decode_settings()
    component = jpegio.read(trial.case.jpeg.read_bytes()).components[0]
    problem = decode.component_problem(component, settings)
    original = common.read_original(trial.case.original)
    height, width = component.height, component.width
    started = time.perf_counter()
    outcome: dict[str, Any] = {
        "image": trial.case.image,
        "quality": trial.case.quality,
        "model": trial.model,
        "kind": trial.kind,
        "parameter": trial.parameter,
        "samples": problem.samples,
    }
    if trial.kind == "relaxed":
        run = tuning.Run(
            trial.case,
            trial.model,
            VARIANT_RATIOS[trial.model],
            VARIANT_ITERATIONS[trial.model],
            relaxation=trial.parameter,
        )
        result = tuning.solve(run)
        history = result["history"]
        trace = Trace(
            iterations=list(history["iterations"][1:]),
            primal=list(history["primal"][1:]),
            dual=list(history["dual"][1:]),
            psnr={str(point["iteration"]): float(point["psnr"]) for point in result["checkpoints"]},
        )
    elif trial.kind == "warm":
        trace = warm_tgv(problem, original, height, width, trial.parameter)
    elif trial.model == "tv":
        trace = adaptive_tv(problem, original, height, width, trial.parameter)
    else:
        trace = adaptive_tgv(problem, original, height, width, trial.parameter)
    outcome["trace"] = dataclasses.asdict(trace)
    outcome["wall_seconds"] = time.perf_counter() - started
    return outcome


def solved(model: str, cases: list[common.Case]) -> list[common.Case]:
    """The files a model is solved on: every one for TV, the first image of each kind for TGV."""
    return [case for case in cases if model == "tv" or case.image in tuning.TGV_IMAGES]


def variant_jobs(cases: list[common.Case], models: list[str], cache: Path) -> list[tuple[Path, Trial]]:
    """The variants stage's trials."""
    trials = []
    for case in cases:
        if (case.image, case.quality) not in VARIANT_FILES:
            continue
        for model in models:
            trials += [Trial(case, model, "relaxed", rho) for rho in RELAXATIONS]
            trials += [Trial(case, model, "adaptive", scale) for scale in SCALES]
            if model == "tgv":
                trials += [Trial(case, model, "warm", rho) for rho in WARM_RELAXATIONS]
    return [(cache / "variants" / f"{trial.name}.json", trial) for trial in trials]


def reference_jobs(cases: list[common.Case], models: list[str], cache: Path) -> list[tuple[Path, tuning.Run]]:
    """The references stage's runs."""
    jobs = []
    for model in models:
        ratio, rho = REFERENCES[model]
        for case in solved(model, cases):
            run = tuning.Run(case, model, ratio, REFERENCE_ITERATIONS[model], relaxation=rho)
            jobs.append((cache / "references" / f"{model}-{case.name}.json", run))
    return tuning.longest_first(jobs)


def stop_jobs(cases: list[common.Case], models: list[str], cache: Path) -> list[tuple[Path, tuning.Run]]:
    """The stops stage's runs."""
    jobs = []
    for model in models:
        tolerances = TOLERANCES[model]
        for case in solved(model, cases):
            for ratio, rho in VARIANTS[model]:
                run = tuning.Run(
                    case,
                    model,
                    ratio,
                    REFERENCE_ITERATIONS[model],
                    tolerance=min(tolerances),
                    crossings=tolerances,
                    relaxation=rho,
                )
                jobs.append((cache / "stops" / f"{model}-r{ratio:g}-p{rho:g}-{case.name}.json", run))
    return tuning.longest_first(jobs)


type Files = dict[tuple[str, int], dict[str, Any]]


def by_file(directory: Path, pattern: str) -> Files:
    """The results kept in a directory whose names match, by image and quality."""
    found: Files = {}
    for path in sorted(directory.glob(pattern)):
        result = common.load(path)
        found[(str(result["image"]), int(result["quality"]))] = result
    return found


def exact_stops(runs: Files, references: Files, tolerance: float) -> list[tuning.Stop]:
    """Where every file's run first brought its gap per sample within a tolerance, and the change there.

    Against the reference's last point; the files whose run did not reach it are left out.
    """
    stops = []
    for key, run in runs.items():
        crossing = run["crossings"].get(f"{tolerance:g}")
        if crossing is None:
            continue
        last = references[key]["last"]
        stops.append(
            tuning.Stop(
                iterations=int(crossing["iteration"]),
                psnr=abs(float(crossing["psnr"]) - float(last["psnr"])),
                psnr_8=abs(float(crossing["psnr_8"]) - float(last["psnr_8"])),
                ssim_8=abs(float(crossing["ssim_8"]) - float(last["ssim_8"])),
            )
        )
    return stops


@dataclasses.dataclass(frozen=True, slots=True)
class Choice:
    """The variant and the tolerance chosen for a model, the most iterations, and whether it was accepted."""

    variant: Variant
    tolerance: float
    iterations: int
    stops: list[tuning.Stop]
    accepted: bool = True


def spread(values: list[float]) -> tuple[float, float, float]:
    """The median, the PERCENTILE-th percentile and the largest of values."""
    return float(np.median(values)), float(np.percentile(values, PERCENTILE)), max(values)


def acceptable(stops: list[tuning.Stop]) -> bool:
    """Whether the changes of stopping keep within LIMITS, in the median and the percentile over the files."""
    for measure, (median_limit, percentile_limit) in LIMITS.items():
        median, percentile, _ = spread([float(getattr(stop, measure)) for stop in stops])
        if median > median_limit or percentile > percentile_limit:
            return False
    return True


def told(references: Files) -> float:
    """The least tolerance the references can tell: TELL times their median last gap per sample."""
    return TELL * float(np.median([float(tuning.gaps_of(run)[-1]) for run in references.values()]))


def choose(model: str, runs: dict[Variant, Files], references: Files) -> Choice | None:
    """The variant and the tolerance that stop the files acceptably in the fewest iterations in all.

    For each variant, over the tolerances that every file reached and that the references
    tell, the largest that is acceptable and whose smaller ones are all acceptable too. Where
    no variant has one, the choice is not accepted: the least of those tolerances, with the
    variant that got there in the fewest iterations in all.
    """
    least = told(references)
    best: Choice | None = None
    fallback: Choice | None = None
    for variant, files in runs.items():
        chosen: tuple[float, list[tuning.Stop]] | None = None
        for tolerance in sorted(TOLERANCES[model]):
            found = exact_stops(files, references, tolerance)
            if len(found) < len(files) or tolerance < least:
                continue
            total = tuning.total_of(found)
            if (
                fallback is None
                or tolerance < fallback.tolerance
                or (tolerance == fallback.tolerance and total < tuning.total_of(fallback.stops))
            ):
                fallback = Choice(variant, tolerance, tuning.cap_of(found), found, accepted=False)
            if not acceptable(found):
                break
            chosen = (tolerance, found)
        if chosen is None:
            continue
        tolerance, found = chosen
        if best is None or tuning.total_of(found) < tuning.total_of(best.stops):
            best = Choice(variant, tolerance, tuning.cap_of(found), found)
    return best if best is not None else fallback


def variant_label(variant: Variant) -> str:
    """A variant as the tables show it."""
    return f"$\\tau/\\sigma = {variant[0]:g}$, $\\rho = {variant[1]:g}$"


def stop_rows(model: str, runs: dict[Variant, Files], references: Files) -> list[list[str]]:
    """What stopping exactly at each tolerance does, with each variant, against the references."""
    rows = []
    least = told(references)
    for variant, files in runs.items():
        for tolerance in TOLERANCES[model]:
            found = exact_stops(files, references, tolerance)
            reached = f"{len(found)} of {len(files)}"
            if not found:
                rows.append([variant_label(variant), f"{tolerance:g}", reached, "", "", "", "", ""])
                continue
            psnr = spread([stop.psnr for stop in found])
            ssim = spread([stop.ssim_8 for stop in found])
            if len(found) < len(files):
                verdict = "not reached"
            elif tolerance < least:
                verdict = "not told"
            else:
                verdict = "yes" if acceptable(found) else "no"
            rows.append(
                [
                    variant_label(variant),
                    f"{tolerance:g}",
                    reached,
                    tuning.median_and_range([float(stop.iterations) for stop in found], ".0f"),
                    str(tuning.total_of(found)),
                    " / ".join(f"{value:.4f}" for value in psnr),
                    " / ".join(f"{value:.1e}" for value in ssim),
                    verdict,
                ]
            )
    return rows


def phase1_rows(phase1_tuning: Path, references: Files) -> list[list[str]]:
    """Phase 1's stops of TV (unrelaxed, tau / sigma = 30), against these references, at its tolerances."""
    rows = []
    runs = by_file(phase1_tuning / "stops", "tv-r30-*.json")
    for tolerance in (5e-4, 1e-3):
        found = exact_stops(runs, references, tolerance)
        if not found:
            continue
        rows.append(
            [
                f"{tolerance:g}",
                f"{len(found)} of {len(runs)}",
                " / ".join(f"{value:.4f}" for value in spread([stop.psnr for stop in found])),
                " / ".join(f"{value:.1e}" for value in spread([stop.ssim_8 for stop in found])),
            ]
        )
    return rows


def settling_row(model: str, references: Files) -> list[str]:
    """How much the references' results changed over their last sixth: their own uncertainty."""
    changes_psnr, changes_ssim = [], []
    for run in references.values():
        before = [point for point in run["checkpoints"] if point["iteration"] <= run["iterations"] * 5 // 6][-1]
        changes_psnr.append(abs(float(before["psnr"]) - float(run["last"]["psnr"])))
        changes_ssim.append(abs(float(before["ssim_8"]) - float(run["last"]["ssim_8"])))
    return [
        f"{REFERENCE_ITERATIONS[model] * 5 // 6} to {REFERENCE_ITERATIONS[model]}",
        tuning.median_and_range(changes_psnr, ".4f"),
        tuning.median_and_range(changes_ssim, ".1e"),
    ]


def variant_rows(model: str, trials: list[dict[str, Any]], references: Files) -> list[list[str]]:
    """Each trial of a model: the first iterations within some tolerances, and the PSNR's change at some marks."""
    rows = []
    order = {"relaxed": 0, "adaptive": 1, "warm": 2}
    for trial in sorted(trials, key=lambda t: (order[t["kind"]], t["parameter"], t["image"], t["quality"])):
        trace = trial["trace"]
        gaps = (np.asarray(trace["primal"]) - np.asarray(trace["dual"])) / float(trial["samples"])
        firsts = []
        for tolerance in VARIANT_TOLERANCES[model]:
            stop = common.first_within(trace["iterations"], gaps, tolerance)
            firsts.append(str(stop) if stop is not None else f"> {trace['iterations'][-1]}")
        last = references[(trial["image"], trial["quality"])]["last"]["psnr"]
        changes = [
            f"{trace['psnr'][str(mark)] - last:+.3f}" if str(mark) in trace["psnr"] else ""
            for mark in VARIANT_MARKS
            if mark <= VARIANT_ITERATIONS[model]
        ]
        kind = {"relaxed": "relaxed, $\\rho$", "adaptive": "adaptive, proportion", "warm": "from TV, $\\rho$"}
        before = f" (after {int(trace['ratios'][0])} of TV)" if trial["kind"] == "warm" else ""
        rows.append(
            [
                f"{kind[trial['kind']]} = {trial['parameter']:g}",
                f"{trial['image']} q{trial['quality']}{before}",
                *firsts,
                *changes,
            ]
        )
    return rows


def figure(directory: Path, runs: dict[str, dict[Variant, Files]], choices: dict[str, Choice]) -> list[str]:
    """The median gap per sample of the stops' variants against iterations, and the markdown that shows it."""
    import matplotlib as mpl  # noqa: PLC0415

    mpl.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    directory.mkdir(parents=True, exist_ok=True)
    models = [model for model in MODELS if runs.get(model)]
    shown_figure, axes = plt.subplots(1, len(models), figsize=(5.0 * len(models), 3.8), squeeze=False)
    for axis, model in zip(axes[0], models, strict=True):
        for variant, files in runs[model].items():
            curves = [tuning.gaps_of(run) for run in files.values()]
            length = min(len(curve) for curve in curves)
            records = np.asarray(next(iter(files.values()))["history"]["iterations"][:length], dtype=np.float64)
            median = np.median(np.stack([curve[:length] for curve in curves]), axis=0)
            axis.loglog(records[1:], median[1:], label=variant_label(variant))
        if model in choices:
            axis.axhline(choices[model].tolerance, color="black", linewidth=0.8, linestyle="--")
        axis.set_title(MODEL_NAMES[model])
        axis.set_xlabel("iterations")
        axis.set_ylabel("gap per sample (median over the files)")
        axis.grid(visible=True, which="major", linewidth=0.3)
        axis.legend(fontsize="small")
    shown_figure.tight_layout()
    shown_figure.savefig(directory / "solver-gap.png", dpi=100, metadata={"Software": None})
    plt.close(shown_figure)
    return [
        "![the gap per sample of the variants, to the least tolerance or the first file's stop](phase2/solver-gap.png)"
    ]


def test_jobs(data: Path, cache: Path) -> list[tuple[Path, comparison.Task]]:
    """The test stage's runs: TV and TGV on libjpeg-turbo's greyscale test files."""
    return [
        (cache / "test" / f"{method}-{case.name}.json", comparison.Task(case, method))
        for case in common.cases(data, "libjpeg-turbo", "test")
        for method in ("tgv", "tv")
    ]


def timing_jobs(data: Path, cache: Path) -> list[tuple[Path, common.Case]]:
    """The test stage's timings: the quality-50 file of each test image."""
    return [
        (cache / "timing" / f"{case.image}.json", case)
        for case in common.cases(data, "libjpeg-turbo", "test")
        if case.quality == comparison.TIMED_QUALITY
    ]


def test_rows(model: str, cache: Path, phase1: Path) -> list[list[str]]:
    """Phase 1's defaults and these on the test images: iterations, time, and the PSNR and SSIM of the result."""
    rows = []
    now_costs = {path.stem: common.load(path) for path in sorted((cache / "timing").glob("*.json"))}
    then_costs = {path.stem: common.load(path) for path in sorted((phase1 / "timing").glob("*.json"))}
    for quality in common.QUALITIES:
        pairs = []
        for path in sorted((cache / "test").glob(f"{model}-libjpeg-turbo-*-q{quality}.json")):
            then_path = phase1 / "run" / path.name
            if then_path.exists():
                pairs.append((common.load(then_path), common.load(path)))
        if not pairs:
            continue
        then_iterations = [float(then["iterations"]) for then, _ in pairs]
        now_iterations = [float(now["iterations"]) for _, now in pairs]
        then_seconds = [float(then["iterations"]) * float(then_costs[then["image"]][model]) for then, _ in pairs]
        now_seconds = [float(now["iterations"]) * float(now_costs[now["image"]][model]) for _, now in pairs]
        psnr = [float(now["last"]["psnr"]) - float(then["last"]["psnr"]) for then, now in pairs]
        ssim = [float(now["last"]["ssim_8"]) - float(then["last"]["ssim_8"]) for then, now in pairs]
        rows.append(
            [
                str(quality),
                f"{np.median(then_iterations):.0f} to {np.median(now_iterations):.0f}",
                f"{max(then_iterations):.0f} to {max(now_iterations):.0f}",
                f"{np.median(then_seconds):.1f} to {np.median(now_seconds):.1f}",
                f"{np.mean(psnr):+.4f} ({min(psnr):+.4f} to {max(psnr):+.4f})",
                f"{np.mean(ssim):+.1e}",
            ]
        )
    return rows


def report(cache: Path, out: Path, phase1: Path, phase1_tuning: Path) -> dict[str, Choice]:
    """Writes the tables and the figure, and returns the choices."""
    from importlib import metadata  # noqa: PLC0415

    lines = [
        "<!-- Written by experiments/phase2_solver.py; do not edit by hand. -->",
        "",
        "# Phase 2: the primal-dual method relaxed",
        "",
        "- Files: those of phase1-tuning.md, the greyscale tuning images as libjpeg-turbo encoded them: TV on "
        "all 48, TGV on the 24 of the first image of each kind. The weights of `unround.decode.Settings`.",
        "- The method is the relaxed one of docs/math.md, 5, with the ratio of the steps $\\tau/\\sigma$ and the "
        "relaxation $\\rho$.",
        "- References: "
        + "; ".join(
            f"{MODEL_NAMES[model]} with {variant_label(REFERENCES[model])}, {REFERENCE_ITERATIONS[model]} iterations"
            for model in MODELS
        )
        + ".",
        "- Stopping where the gap per sample first falls within a tolerance changes the result, against the "
        "reference's last point. A tolerance is acceptable when, over the files, the change of the PSNR of the "
        f"binary64 result is at most {LIMITS['psnr'][0]:g} dB in the median and {LIMITS['psnr'][1]:g} dB in the "
        f"{PERCENTILE:g}th percentile, and that of the SSIM of its 8-bit samples at most {LIMITS['ssim_8'][0]:g} "
        f"and {LIMITS['ssim_8'][1]:g}; every file has to reach it, and the references have to tell it: their "
        f"median last gap per sample at most 1/{TELL:g} of it. The largest changes are shown.",
        f"- NumPy {np.__version__}, SciPy {metadata.version('scipy')}, "
        f"scikit-image {metadata.version('scikit-image')}.",
        "",
    ]
    choices: dict[str, Choice] = {}
    all_runs: dict[str, dict[Variant, Files]] = {}
    for model in MODELS:
        references = by_file(cache / "references", f"{model}-*.json")
        if not references:
            continue
        runs = {
            variant: by_file(cache / "stops", f"{model}-r{variant[0]:g}-p{variant[1]:g}-*.json")
            for variant in VARIANTS[model]
        }
        runs = {variant: files for variant, files in runs.items() if files}
        all_runs[model] = runs
        lines += [f"## {MODEL_NAMES[model]}", "", "### Stopping at a tolerance", ""]
        lines += common.table(
            [
                "variant",
                "gap per sample ≤",
                "files that reached it",
                "iterations: median (least to largest)",
                "iterations in all",
                f"\\|Δ PSNR\\|, binary64, dB: median / {PERCENTILE:g}th / largest",
                f"\\|Δ SSIM\\|, 8 bits: median / {PERCENTILE:g}th / largest",
                "acceptable",
            ],
            stop_rows(model, runs, references),
        )
        choice = choose(model, runs, references) if runs else None
        if choice is not None:
            choices[model] = choice
            chosen = (
                f"{variant_label(choice.variant)}, stopping at a gap per sample of {choice.tolerance:g}, and after "
                f"at most {choice.iterations} iterations (twice the most a tuning file took, rounded up)"
            )
            if choice.accepted:
                lines += [f"Chosen: {chosen}.", ""]
            else:
                psnr = spread([stop.psnr for stop in choice.stops])
                lines += [
                    "No variant has a tolerance that every file reached, that the references tell and that is "
                    f"acceptable. Chosen, the least of those that every file reached: {chosen}. Stopping there "
                    f"changed the PSNR by {psnr[0]:.4f} dB in the median, {psnr[1]:.4f} dB in the "
                    f"{PERCENTILE:g}th percentile and {psnr[2]:.4f} dB at most.",
                    "",
                ]
        lines += [
            "The references' own change over their last sixth: median (least to largest) over the files. Their "
            f"median last gap per sample is {told(references) / TELL:.1e}, so that they tell tolerances of "
            f"{told(references):.1e} and more.",
            "",
            *common.table(
                ["iterations", "\\|Δ PSNR\\|, binary64, dB", "\\|Δ SSIM\\|, 8 bits"], [settling_row(model, references)]
            ),
        ]
        phase1_rows_found = phase1_rows(phase1_tuning, references) if model == "tv" else []
        if phase1_rows_found:
            lines += [
                "Phase 1's stops (unrelaxed, $\\tau/\\sigma = 30$) against these references: Phase 1 measured "
                "them against its own long runs, which went on from the stops along the same path, and found at "
                "most 0.0083 dB at 5e-4.",
                "",
                *common.table(
                    [
                        "gap per sample ≤",
                        "files",
                        f"\\|Δ PSNR\\|, binary64, dB: median / {PERCENTILE:g}th / largest",
                        f"\\|Δ SSIM\\|, 8 bits: median / {PERCENTILE:g}th / largest",
                    ],
                    phase1_rows_found,
                ),
            ]
        trials = [common.load(path) for path in sorted((cache / "variants").glob(f"{model}-*.json"))]
        if trials:
            marks = [mark for mark in VARIANT_MARKS if mark <= VARIANT_ITERATIONS[model]]
            lines += [
                "### What else was tried",
                "",
                f"On three files, with Phase 1's ratio $\\tau/\\sigma = {VARIANT_RATIOS[model]:g}$: the first "
                "iteration whose gap per sample is within each tolerance, and the change of the PSNR of the "
                "binary64 result, against the reference's last point, at some iterations.",
                "",
                *common.table(
                    [
                        "method",
                        "file",
                        *[f"gap ≤ {tolerance:g}" for tolerance in VARIANT_TOLERANCES[model]],
                        *[f"Δ PSNR at {mark}" for mark in marks],
                    ],
                    variant_rows(model, trials, references),
                ),
            ]
    if (cache / "test").is_dir():
        lines += [
            "## The test images",
            "",
            "libjpeg-turbo's greyscale test files, solved with Phase 1's defaults (phase1-comparison.md) and "
            "with these: the median and the largest number of iterations, the median time (the iterations "
            "times the time an iteration took, one run after another in one process), and the change of the "
            "PSNR of the binary64 result and of the SSIM of its 8-bit samples.",
            "",
        ]
        for model in MODELS:
            rows = test_rows(model, cache, phase1)
            if rows:
                lines += [
                    f"**{MODEL_NAMES[model]}**",
                    "",
                    *common.table(
                        [
                            "quality",
                            "iterations: median",
                            "largest",
                            "seconds: median",
                            "Δ PSNR, binary64, dB: mean (least to largest)",
                            "Δ SSIM, 8 bits: mean",
                        ],
                        rows,
                    ),
                ]
    if all_runs:
        lines += ["## Figure", "", *figure(out / "phase2", all_runs, choices), ""]
    out.mkdir(parents=True, exist_ok=True)
    (out / "phase2-solver.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return choices


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--build", type=Path, required=True, help="a build directory with -DUNROUND_WITH_PYTHON=ON")
    parser.add_argument("--data", type=Path, default=common.ROOT / "data")
    parser.add_argument("--cache", type=Path, default=common.ROOT / "data" / "phase2" / "solver")
    parser.add_argument("--out", type=Path, default=common.ROOT / "experiments" / "results")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument(
        "--phase1-test",
        type=Path,
        default=common.ROOT / "data" / "phase1" / "test",
        help="phase1_comparison.py's cache",
    )
    phase1_tuning = common.ROOT / "data" / "phase1" / "tuning"
    parser.add_argument("--phase1-tuning", type=Path, default=phase1_tuning, help="phase1_tuning.py's cache")
    parser.add_argument("--models", nargs="*", default=list(MODELS), choices=MODELS)
    parser.add_argument("stages", nargs="*", default=STAGES, choices=STAGES)
    options = parser.parse_args()
    common.use_build(options.build)

    cases = common.cases(options.data, "libjpeg-turbo", "tuning")
    if "variants" in options.stages:
        common.run_all(run_trial, variant_jobs(cases, options.models, options.cache), options.workers, label="variants")
    if "references" in options.stages:
        jobs = reference_jobs(cases, options.models, options.cache)
        common.run_all(tuning.solve, jobs, options.workers, label="references")
    if "stops" in options.stages:
        common.run_all(tuning.solve, stop_jobs(cases, options.models, options.cache), options.workers, label="stops")
    if "test" in options.stages:
        common.run_all(comparison.run, test_jobs(options.data, options.cache), options.workers, label="test")
        common.run_all(comparison.timing, timing_jobs(options.data, options.cache), 1, label="timing")
    if "report" in options.stages:
        for model, choice in report(options.cache, options.out, options.phase1_test, options.phase1_tuning).items():
            accepted = "" if choice.accepted else " (not accepted)"
            print(
                f"{MODEL_NAMES[model]}: tau / sigma = {choice.variant[0]:g}, rho = {choice.variant[1]:g}, "
                f"tolerance {choice.tolerance:g}, at most {choice.iterations} iterations{accepted}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
