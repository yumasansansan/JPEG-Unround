#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Phase 1: the ratio of the steps and the stopping tolerance of the primal-dual method, on the tuning images.

    python experiments/phase1_tuning.py --build build/release [--data data] [--workers 8]
        [--cache data/phase1/tuning] [--out experiments/results] [stage ...]

The files are the greyscale tuning images (data/synthetic/tuning) as libjpeg-turbo
encoded them, at every quality: 8 images at 6 qualities. They are solved for TV, and
the first image of each kind for TGV (docs/math.md, 4.2 and 4.3), with the weights of
unround.decode.Settings, by the primal-dual method (5), in these stages:

  runs    Each file with each ratio of the steps tau / sigma of RATIOS, for ITERATIONS;
          with LONG_RATIO, for LONG_ITERATIONS: its last point stands for the least
          point. The gap is recorded every 10 iterations, and the canvases at
          iterations about 25 percent apart are measured: against the original (the
          PSNR of the binary64 picture and of its 8-bit samples, and their SSIM), and
          against the run's last canvas. The distances of the start from the last
          point, in the primal and in the dual, give the ratio that the bound of the
          method's rate would choose (docs/math.md, 5).
  start   TGV with its field w starting at the gradient of the start and at 0, on two
          files, with four ratios.
  stops   Each file with each ratio of STOP_RATIOS, until the gap per sample is within
          the least of STOP_TOLERANCES (or for LONG_ITERATIONS): the result is measured
          exactly where the gap first falls within each of them.
  report  The tables and the figures, and the ratio and the tolerance they choose: the
          tolerance at which stopping changes the result by at most PSNR_LIMIT and
          SSIM_LIMIT, against the last point of the long run, on every file.

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
from scipy import integrate

import phase1_common as common

if TYPE_CHECKING:
    # For the annotations alone: the package is imported once its path is set.
    from unround.results import Result

type Array = npt.NDArray[np.float64]

MODELS: Final = ("tv", "tgv")
MODEL_NAMES: Final = {"tv": "TV", "tgv": "TGV"}

# The tolerances of the gap per sample at which the report shows how fast each ratio got there.
REACH_TOLERANCES: Final = {"tv": (2e-2, 1e-2, 5e-3, 2e-3, 1e-3), "tgv": (1e-1, 5e-2, 2e-2, 1e-2)}

# What stopping may change of the result, against the last point of a long run: the PSNR of
# the binary64 picture, and the SSIM of its 8-bit samples. The PSNR of 8-bit samples is shown
# but not held to a limit: rounding moves it by about 0.01 dB between one iterate and the next.
PSNR_LIMIT: Final = 0.01
SSIM_LIMIT: Final = 1e-4

# The ratios of the steps tau / sigma tried, how long each run goes, and the run of each file
# that goes on for long. TGV is solved on the first image of each kind.
RATIOS: Final = {"tv": (1.0, 3.0, 10.0, 30.0, 100.0), "tgv": (1.0, 3.0, 10.0)}
ITERATIONS: Final = {"tv": 2000, "tgv": 2500}
LONG_RATIO: Final = {"tv": 30.0, "tgv": 3.0}
LONG_ITERATIONS: Final = {"tv": 6000, "tgv": 8000}
TGV_IMAGES: Final = ("chart-0-grey", "illustration-0-grey", "lineart-0-grey", "text-0-grey")

# The stops: with these ratios, each file is solved until its gap per sample is within the
# least of the tolerances (or for the long run's iterations), and the result is measured where
# the gap first falls within each.
STOP_RATIOS: Final = {"tv": (10.0, 20.0, 30.0), "tgv": (3.0,)}
STOP_TOLERANCES: Final = {"tv": (1e-2, 5e-3, 2e-3, 1e-3, 5e-4), "tgv": (5e-2, 2e-2, 1e-2, 5e-3)}

# The start of w: two files, four ratios, 3000 iterations.
START_FILES: Final = (("chart-0-grey", 10), ("text-0-grey", 50))
START_RATIOS: Final = (3.0, 300.0, 1000.0, 3000.0)
START_ITERATIONS: Final = 3000

STAGES: Final = ["runs", "start", "stops", "report"]
RECORD_EVERY: Final = 10

# Iterations at which the report shows the primal values, and the drift of the long runs.
PRIMAL_MARKS: Final = (100, 300, 1000, 2000)
DRIFT_MARKS: Final = (300, 1000, 2000, 5000)

# Iterations at which a run keeps its canvas: about 25 percent apart, at records.
CHECKPOINTS: Final = frozenset(
    [10, 20, 30, 40, 50, 60, 80]
    + [int(mantissa * 10**exponent) for exponent in (2, 3, 4) for mantissa in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8)]
)


@dataclasses.dataclass(frozen=True, slots=True)
class Run:
    """One solver run: a file, a model, the ratio of the steps, and how long."""

    case: common.Case
    model: str
    ratio: float
    iterations: int
    tolerance: float = 0.0
    w_start: str = "gradient"  # TGV's w at the start: the gradient of the start's canvas, or zero
    crossings: tuple[float, ...] = ()  # tolerances at whose first crossing the result is measured
    relaxation: float = 1.0  # the rho of the relaxed steps; Phase 1's runs were not relaxed


def moments(a: float) -> tuple[float, float]:
    """The mean and the mean square of s on [0, 1] with the density proportional to e^(-a s), a > 0.

    From a = 1 on, the closed forms, whose terms then cancel by at most a factor of about
    twelve; below, by quadrature.
    """
    if a >= 1.0:
        decay = math.exp(-a)
        mass = -math.expm1(-a)
        first = (1.0 - decay * (1.0 + a)) / (a * mass)
        second = (2.0 - decay * (a * a + 2.0 * a + 2.0)) / (a * a * mass)
        return first, second
    mass = integrate.quad(lambda s: math.exp(-a * s), 0.0, 1.0)[0]
    first = integrate.quad(lambda s: s * math.exp(-a * s), 0.0, 1.0)[0] / mass
    second = integrate.quad(lambda s: s * s * math.exp(-a * s), 0.0, 1.0)[0] / mass
    return first, second


def posterior_rms(coefficients: npt.ArrayLike, quant_table: npt.ArrayLike) -> float:
    """The RMS spread of the coefficients within their intervals under the Laplace model.

    The square root of the mean, over every coefficient, of its variance given its level:
    a truncated Laplace distribution of the frequency's scale (unround.laplace.scales) on
    the interval, and a uniform distribution for DC. It is what the MMSE centres leave of
    the coefficients, on average, in the model.
    """
    from unround import laplace  # noqa: PLC0415

    levels = np.asarray(coefficients, dtype=np.int64)
    table = np.asarray(quant_table, dtype=np.int64)
    scale = laplace.scales(levels, table)
    blocks = int(levels.shape[0] * levels.shape[1])
    total = 0.0
    for v in range(8):
        for u in range(8):
            step = float(table[v, u])
            if v == 0 and u == 0:
                total += blocks * step * step / 12.0
                continue
            beta = float(scale[v, u])
            if beta == 0.0:
                continue  # every level is 0, and the model puts every coefficient at 0
            zeros = int(np.count_nonzero(levels[:, :, v, u] == 0))
            half = step / 2.0
            _, zero_second = moments(half / beta)
            first, second = moments(step / beta)
            total += zeros * half * half * zero_second + (blocks - zeros) * step * step * (second - first * first)
    return math.sqrt(total / (blocks * 64))


def measure_checkpoints(
    kept: dict[int, Array], last: Array, original: npt.NDArray[np.uint8], gaps: dict[int, float]
) -> list[dict[str, float]]:
    """The kept pictures against the last and against the original, with the gap per sample of each.

    gaps are the gaps per sample of the records, by iteration.
    """
    from unround import metrics  # noqa: PLC0415

    last_8 = metrics.quantize(last)
    measured = []
    for iteration in sorted(kept):
        picture = kept[iteration]
        picture_8 = metrics.quantize(picture)
        difference = picture - last
        measured.append(
            {
                "iteration": iteration,
                "gap_per_sample": gaps[iteration],
                "rms": common.root_mean_square(difference),
                "largest": float(np.max(np.abs(difference))),
                "changed_8": float(np.count_nonzero(picture_8 != last_8)) / last_8.size,
                "psnr": metrics.psnr(original, picture),
                "psnr_8": metrics.psnr(original, picture_8),
                "ssim_8": common.ssim(original, picture_8),
            }
        )
    return measured


def start_distances(run: Run, start: Array, result: Result) -> tuple[float, float]:
    """The distances of the start from the last point: in the primal, (x, w) for TGV, and in the dual.

    The primal starts at the canvas start and, for TGV, w at its gradient or at 0; the dual at 0.
    """
    from unround.operators import grad, tensor_inner  # noqa: PLC0415

    dual = result.dual
    if dual is None:
        message = "the primal-dual method returned no dual point"
        raise RuntimeError(message)
    primal = float(np.sum((start - result.primal.canvas) ** 2))
    dual_squared = float(np.sum(dual.p * dual.p))
    if run.model == "tgv":
        if result.primal.w is None or dual.r is None:
            message = "TGV's run returned no w or no r"
            raise RuntimeError(message)
        start_w = np.zeros_like(result.primal.w) if run.w_start == "zero" else grad(start)
        primal += float(np.sum((start_w - result.primal.w) ** 2))
        dual_squared += tensor_inner(dual.r, dual.r)
    return math.sqrt(primal), math.sqrt(dual_squared)


def solve(run: Run) -> dict[str, Any]:
    """Solves a file, and measures what the run kept."""
    from unround import decode, jpegio, metrics, pdhg  # noqa: PLC0415
    from unround.model import Primal  # noqa: PLC0415

    settings = decode.Settings()
    component = jpegio.read(run.case.jpeg.read_bytes()).components[0]
    problem = decode.component_problem(component, settings)
    original = common.read_original(run.case.original)
    height, width = component.height, component.width
    kept: dict[int, Array] = {}
    crossed: dict[float, dict[str, float]] = {}

    def observe(iteration: int, point: Primal, gap: float) -> None:
        if iteration in CHECKPOINTS:
            kept[iteration] = point.canvas[:height, :width].copy()
        for tolerance in run.crossings:
            if tolerance not in crossed and gap <= tolerance:
                picture = point.canvas[:height, :width]
                eight = metrics.quantize(picture)
                crossed[tolerance] = {
                    "iteration": iteration,
                    "gap_per_sample": gap,
                    "psnr": metrics.psnr(original, picture),
                    "psnr_8": metrics.psnr(original, eight),
                    "ssim_8": common.ssim(original, eight),
                }

    options = pdhg.Options(
        iterations=run.iterations,
        tolerance=run.tolerance,
        step_ratio=run.ratio,
        relaxation=run.relaxation,
        record_every=RECORD_EVERY,
    )
    started = time.perf_counter()
    if run.model == "tv":
        result = pdhg.solve_tv(problem, settings.tv, options, observe=observe)
    else:
        first = pdhg.Initial(w=np.zeros((2, *problem.shape))) if run.w_start == "zero" else None
        result = pdhg.solve_tgv(problem, settings.tgv, options, first=first, observe=observe)
    wall = time.perf_counter() - started

    outcome: dict[str, Any] = {
        "encoder": run.case.encoder,
        "image": run.case.image,
        "quality": run.case.quality,
        "model": run.model,
        "ratio": run.ratio,
        "relaxation": run.relaxation,
        "tolerance": run.tolerance,
        "w_start": run.w_start,
        "samples": problem.samples,
        "iterations": result.iterations,
        "converged": result.converged,
        "wall_seconds": wall,
        "history": common.history_of(result),
        "crossings": {f"{tolerance:g}": crossed[tolerance] for tolerance in sorted(crossed)},
    }
    start = pdhg.start(problem)
    kept[0] = start.canvas[:height, :width]
    last = result.primal.canvas[:height, :width]
    gaps = dict(zip(outcome["history"]["iterations"], (float(gap) for gap in gaps_of(outcome)), strict=True))
    outcome["checkpoints"] = measure_checkpoints(kept, last, original, gaps)
    outcome["last"] = common.measure(original, last, problem, result.primal.canvas)
    outcome["primal_distance"], outcome["dual_distance"] = start_distances(run, start.canvas, result)
    outcome["rms_step"] = math.sqrt(float(np.mean(problem.steps**2)) / 12.0)
    outcome["posterior_rms"] = posterior_rms(component.coefficients, component.quant_table)
    return outcome


def ideal_ratio(result: dict[str, Any]) -> float:
    """The ratio tau / sigma that the bound of the rate chooses: the start's squared distances, primal over dual."""
    return float(result["primal_distance"] / result["dual_distance"]) ** 2


type Runs = dict[tuple[str, int], dict[float, dict[str, Any]]]


def runs_of(directory: Path, model: str) -> Runs:
    """The runs of a model kept in a directory: by image and quality, then by ratio."""
    found: Runs = {}
    for path in sorted(directory.glob(f"{model}-*.json")):
        result = common.load(path)
        found.setdefault((str(result["image"]), int(result["quality"])), {})[float(result["ratio"])] = result
    return found


def gaps_of(result: dict[str, Any]) -> Array:
    """The gap per sample at each record of a run."""
    history = result["history"]
    return np.asarray(
        (np.asarray(history["primal"]) - np.asarray(history["dual"])) / float(result["samples"]), dtype=np.float64
    )


@dataclasses.dataclass(frozen=True, slots=True)
class Stop:
    """Where a run stops at a tolerance, and how much its result then differs from the long run's last."""

    iterations: int
    psnr: float
    psnr_8: float
    ssim_8: float


type Stops = dict[float, dict[tuple[str, int], dict[str, Any]]]


def stops_of(directory: Path, model: str) -> Stops:
    """The stops runs of a model: by ratio, then by image and quality."""
    found: Stops = {}
    for path in sorted(directory.glob(f"{model}-*.json")):
        result = common.load(path)
        found.setdefault(float(result["ratio"]), {})[(str(result["image"]), int(result["quality"]))] = result
    return found


def exact_stops(model: str, runs: dict[tuple[str, int], dict[str, Any]], files: Runs, tolerance: float) -> list[Stop]:
    """Where every file's run first brought its gap per sample within a tolerance, and the result's change there.

    The change is against the last point of the file's long run. A file whose run did not
    reach the tolerance is left out.
    """
    stops = []
    for key, run in runs.items():
        crossing = run["crossings"].get(f"{tolerance:g}")
        if crossing is None:
            continue
        last = files[key][LONG_RATIO[model]]["last"]
        stops.append(
            Stop(
                iterations=int(crossing["iteration"]),
                psnr=abs(float(crossing["psnr"]) - float(last["psnr"])),
                psnr_8=abs(float(crossing["psnr_8"]) - float(last["psnr_8"])),
                ssim_8=abs(float(crossing["ssim_8"]) - float(last["ssim_8"])),
            )
        )
    return stops


def acceptable(stops: list[Stop]) -> bool:
    """Whether stopping changed neither the PSNR nor the SSIM beyond their limits, on any file."""
    return all(stop.psnr <= PSNR_LIMIT and stop.ssim_8 <= SSIM_LIMIT for stop in stops)


@dataclasses.dataclass(frozen=True, slots=True)
class Choice:
    """The ratio and the tolerance chosen for a model, the most iterations to allow, and whether it was accepted."""

    ratio: float
    tolerance: float
    iterations: int
    stops: list[Stop]
    accepted: bool = True


def one_two_five_above(value: float) -> int:
    """The least number of the series 1, 2, 5, 10, 20, ... that is at least value."""
    power = 1
    while True:
        for mantissa in (1, 2, 5):
            if mantissa * power >= value:
                return mantissa * power
        power *= 10


def choose(model: str, stops: Stops, files: Runs) -> Choice | None:
    """The ratio and the tolerance that stop every tuning file acceptably in the fewest iterations in all.

    For each ratio, the tolerance is the largest that every file reached and whose stops,
    and those of every smaller tolerance that every file reached, are acceptable. The most
    iterations allowed is twice the most any tuning file took, rounded up to 1, 2 or 5 times
    a power of ten. Where no ratio has an acceptable tolerance, the choice is not accepted:
    it is the least tolerance that every file reached, with the ratio that got there in the
    fewest iterations in all.
    """
    best: Choice | None = None
    fallback: Choice | None = None
    for ratio, runs in sorted(stops.items()):
        chosen: tuple[float, list[Stop]] | None = None
        for tolerance in sorted(STOP_TOLERANCES[model]):
            found = exact_stops(model, runs, files, tolerance)
            if len(found) < len(runs):
                continue
            if (
                fallback is None
                or tolerance < fallback.tolerance
                or (tolerance == fallback.tolerance and total_of(found) < total_of(fallback.stops))
            ):
                fallback = Choice(ratio, tolerance, cap_of(found), found, accepted=False)
            if not acceptable(found):
                break
            chosen = (tolerance, found)
        if chosen is None:
            continue
        tolerance, found = chosen
        if best is None or total_of(found) < total_of(best.stops):
            best = Choice(ratio=ratio, tolerance=tolerance, iterations=cap_of(found), stops=found)
    return best if best is not None else fallback


def total_of(stops: list[Stop]) -> int:
    """The iterations of the stops, in all."""
    return sum(stop.iterations for stop in stops)


def cap_of(stops: list[Stop]) -> int:
    """The most iterations to allow: twice the most of the stops, rounded up to 1, 2 or 5 times a power of ten."""
    return one_two_five_above(2 * max(stop.iterations for stop in stops))


def median_and_range(values: list[float], form: str) -> str:
    """The median of values, and their least and largest."""
    return f"{np.median(values):{form}} ({min(values):{form}} to {max(values):{form}})"


def stop_rows(model: str, stops: Stops, files: Runs) -> list[list[str]]:
    """What stopping exactly at each tolerance does, with each ratio, against the long runs' last points."""
    rows = []
    for ratio, runs in sorted(stops.items()):
        for tolerance in STOP_TOLERANCES[model]:
            found = exact_stops(model, runs, files, tolerance)
            reached = f"{len(found)} of {len(runs)}"
            if not found:
                rows.append([f"{ratio:g}", f"{tolerance:g}", reached, "", "", "", "", "", ""])
                continue
            rows.append(
                [
                    f"{ratio:g}",
                    f"{tolerance:g}",
                    reached,
                    median_and_range([float(stop.iterations) for stop in found], ".0f"),
                    str(sum(stop.iterations for stop in found)),
                    f"{max(stop.psnr for stop in found):.4f}",
                    f"{max(stop.psnr_8 for stop in found):.4f}",
                    f"{max(stop.ssim_8 for stop in found):.1e}",
                    "yes" if len(found) == len(runs) and acceptable(found) else "no",
                ]
            )
    return rows


def reach_rows(files: Runs, model: str) -> list[list[str]]:
    """How many iterations each ratio took to bring the gap per sample within some tolerances, in the runs stage."""
    rows = []
    budget = ITERATIONS[model]
    for tolerance in REACH_TOLERANCES[model]:
        cells = [f"{tolerance:g}"]
        for ratio in RATIOS[model]:
            reach = []
            for by_ratio in files.values():
                run = by_ratio[ratio]
                stop = common.first_within(run["history"]["iterations"], gaps_of(run), tolerance)
                reach.append(float(stop) if stop is not None and stop <= budget else math.inf)
            median = float(np.median(reach))
            count = sum(1 for value in reach if math.isfinite(value))
            cells.append(f"{median:.0f} ({count})" if math.isfinite(median) else f"> {budget} ({count})")
        rows.append(cells)
    return rows


def settling_row(files: Runs, model: str) -> list[str]:
    """How much the long runs' results changed over their last sixth: the reference's own uncertainty."""
    changes_psnr, changes_ssim = [], []
    for by_ratio in files.values():
        run = by_ratio[LONG_RATIO[model]]
        before = [point for point in run["checkpoints"] if point["iteration"] <= run["iterations"] * 5 // 6][-1]
        changes_psnr.append(abs(float(before["psnr"]) - float(run["last"]["psnr"])))
        changes_ssim.append(abs(float(before["ssim_8"]) - float(run["last"]["ssim_8"])))
    return [
        f"{LONG_ITERATIONS[model] * 5 // 6} to {LONG_ITERATIONS[model]}",
        median_and_range(changes_psnr, ".4f"),
        median_and_range(changes_ssim, ".1e"),
    ]


def best_duals(files: Runs) -> dict[tuple[str, int], float]:
    """The greatest dual value of any run of each file: a lower bound of its least value."""
    return {key: max(max(run["history"]["dual"]) for run in by_ratio.values()) for key, by_ratio in files.items()}


def primal_excess(files: Runs, ratio: float, iterations: int) -> list[float]:
    """(P - D) / N at an iteration for every file's run with a ratio, D the file's best dual value."""
    duals = best_duals(files)
    excess = []
    for key, by_ratio in files.items():
        run = by_ratio[ratio]
        index = run["history"]["iterations"].index(iterations)
        excess.append((float(run["history"]["primal"][index]) - duals[key]) / float(run["samples"]))
    return excess


def ratio_rows(files: Runs, model: str) -> list[list[str]]:
    """How far each ratio's primal values are above the best dual value, by iteration."""
    rows = []
    for ratio in RATIOS[model]:
        cells = [f"{ratio:g}"]
        cells += [
            median_and_range(primal_excess(files, ratio, iterations), ".1e")
            for iterations in PRIMAL_MARKS
            if iterations <= ITERATIONS[model]
        ]
        rows.append(cells)
    return rows


def bound_rows(files: Runs, model: str) -> list[list[str]]:
    """The ratio that the bound of the rate chooses, from each long run's last point, by quality."""
    rows = []
    for quality in common.QUALITIES:
        chosen = [by_ratio[LONG_RATIO[model]] for (_, q), by_ratio in files.items() if q == quality]
        if not chosen:
            continue
        rows.append(
            [
                str(quality),
                median_and_range([ideal_ratio(run) for run in chosen], ".3g"),
                median_and_range([run["primal_distance"] / math.sqrt(run["samples"]) for run in chosen], ".3g"),
                median_and_range([run["dual_distance"] / math.sqrt(run["samples"]) for run in chosen], ".3g"),
                median_and_range([float(gaps_of(run)[-1]) for run in chosen], ".1e"),
            ]
        )
    return rows


def drift_rows(files: Runs, model: str) -> list[list[str]]:
    """How far the long runs' canvases still were from their last, at some iterations."""
    rows = []
    for iterations in DRIFT_MARKS:
        distances = []
        for by_ratio in files.values():
            points = {point["iteration"]: point for point in by_ratio[LONG_RATIO[model]]["checkpoints"]}
            if iterations in points:
                distances.append(float(points[iterations]["rms"]))
        if distances:
            rows.append([str(iterations), median_and_range(distances, ".2e")])
    return rows


def start_rows(directory: Path) -> list[list[str]]:
    """TGV from w at the gradient of the start and at 0: the bound's ratio, and the iterations to two gaps."""
    rows = []
    for path in sorted(directory.glob("tgv-*.json")):
        run = common.load(path)
        gaps = gaps_of(run)
        iterations = run["history"]["iterations"]
        reached = [common.first_within(iterations, gaps, tolerance) for tolerance in (1e-1, 3e-2)]
        rows.append(
            [
                f"{run['image']} q{run['quality']}",
                str(run["w_start"]),
                f"{run['ratio']:g}",
                f"{ideal_ratio(run):.3g}",
                *[str(value) if value is not None else f"> {run['iterations']}" for value in reached],
                f"{float(run['last']['psnr']):.3f}",
            ]
        )
    return rows


def median_curve(files: Runs, ratio: float, iterations: int, values: str) -> tuple[Array, Array]:
    """The median over the files of the gap per sample, or of (P - D) / N, at each record up to some iterations."""
    duals = best_duals(files)
    curves = []
    records: Array | None = None
    for key, by_ratio in files.items():
        run = by_ratio[ratio]
        history = run["history"]
        count = history["iterations"].index(iterations) + 1
        records = np.asarray(history["iterations"][:count], dtype=np.float64)
        if values == "gap":
            curves.append(gaps_of(run)[:count])
        else:
            primal = np.asarray(history["primal"][:count], dtype=np.float64)
            curves.append((primal - duals[key]) / float(run["samples"]))
    if records is None:
        message = "no runs to take a median of"
        raise ValueError(message)
    return records, np.asarray(np.median(np.stack(curves), axis=0), dtype=np.float64)


def figures(directory: Path, by_model: dict[str, Runs], choices: dict[str, Choice]) -> list[str]:
    """Draws the figures of the report, and returns the markdown that shows them."""
    import matplotlib as mpl  # noqa: PLC0415

    mpl.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    directory.mkdir(parents=True, exist_ok=True)
    models = [model for model in MODELS if model in by_model]
    shown = []
    for name, values, label in (
        ("tuning-gap.png", "gap", "gap per sample (median over the files)"),
        ("tuning-primal.png", "primal", "(P - D_best) / N (median over the files)"),
    ):
        figure, axes = plt.subplots(1, len(models), figsize=(5.0 * len(models), 3.8), squeeze=False)
        for axis, model in zip(axes[0], models, strict=True):
            for ratio in RATIOS[model]:
                records, medians = median_curve(by_model[model], ratio, ITERATIONS[model], values)
                axis.loglog(records[1:], medians[1:], label=f"$\\tau/\\sigma = {ratio:g}$")
            if values == "gap" and model in choices:
                axis.axhline(choices[model].tolerance, color="black", linewidth=0.8, linestyle="--")
            axis.set_title(MODEL_NAMES[model])
            axis.set_xlabel("iterations")
            axis.set_ylabel(label)
            axis.grid(visible=True, which="major", linewidth=0.3)
            axis.legend(fontsize="small")
        figure.tight_layout()
        figure.savefig(directory / name, dpi=100, metadata={"Software": None})
        plt.close(figure)
        shown.append(f"![{label}](phase1/{name})")

    figure, axes = plt.subplots(1, len(models), figsize=(5.0 * len(models), 3.8), squeeze=False)
    for axis, model in zip(axes[0], models, strict=True):
        files = by_model[model]
        qualities = [quality for (_, quality) in files]
        ideals = [ideal_ratio(by_ratio[LONG_RATIO[model]]) for by_ratio in files.values()]
        axis.semilogy(qualities, ideals, "o", markersize=4, label="the bound's ratio, from each file's long run")
        if model in choices:
            axis.axhline(choices[model].ratio, color="black", linewidth=0.8, linestyle="--", label="chosen")
        axis.set_title(MODEL_NAMES[model])
        axis.set_xlabel("quality")
        axis.set_ylabel("$\\tau/\\sigma$")
        axis.grid(visible=True, which="major", linewidth=0.3)
        axis.legend(fontsize="small")
    figure.tight_layout()
    figure.savefig(directory / "tuning-ratio.png", dpi=100, metadata={"Software": None})
    plt.close(figure)
    shown.append("![the ratio of the steps the bound chooses](phase1/tuning-ratio.png)")
    return shown


def report(cache: Path, out: Path) -> dict[str, Choice]:
    """Writes the tables and the figures, and returns the choices."""
    from importlib import metadata  # noqa: PLC0415

    by_model = {model: runs_of(cache / "runs", model) for model in MODELS}
    by_model = {model: files for model, files in by_model.items() if files}
    lines = [
        "<!-- Written by experiments/phase1_tuning.py; do not edit by hand. -->",
        "",
        "# Phase 1 tuning: the ratio of the steps and the stopping tolerance",
        "",
        "- Files: the greyscale tuning images (`data/synthetic/tuning`) as libjpeg-turbo encoded them at "
        f"qualities {', '.join(str(quality) for quality in common.QUALITIES)}: TV on all "
        f"{len(by_model.get('tv', {}))} files, TGV on the {len(by_model.get('tgv', {}))} of the first image of "
        "each kind.",
        "- Model: the weights of `unround.decode.Settings`: $\\mu = 10^{-3}$; TV $\\alpha = 1$; TGV "
        "$\\alpha_1 = 1$, $\\alpha_0 = 2$.",
        "- Runs: "
        + "; ".join(
            f"{MODEL_NAMES[model]} with $\\tau/\\sigma$ = {', '.join(f'{ratio:g}' for ratio in RATIOS[model])} for "
            f"{ITERATIONS[model]} iterations, and {LONG_RATIO[model]:g} for {LONG_ITERATIONS[model]} (the long run)"
            for model in MODELS
        )
        + ". The gap is recorded every 10 iterations; the canvases are measured at iterations about 25 "
        "percent apart.",
        "- Stops: "
        + "; ".join(
            f"{MODEL_NAMES[model]} with $\\tau/\\sigma$ = {', '.join(f'{ratio:g}' for ratio in STOP_RATIOS[model])}, "
            f"until the gap per sample is within {min(STOP_TOLERANCES[model]):g} or for "
            f"{LONG_ITERATIONS[model]} iterations"
            for model in MODELS
        )
        + ". The result is measured exactly where the gap first falls within each tolerance.",
        f"- A tolerance is acceptable when stopping at it changes, on every file, the PSNR of the binary64 "
        f"result by at most {PSNR_LIMIT:g} dB and the SSIM of its 8-bit samples by at most {SSIM_LIMIT:g}, "
        "against the last point of the file's long run. The PSNR of the 8-bit samples is shown, not held to a "
        "limit: rounding moves it by about 0.01 dB from one iterate to the next.",
        f"- NumPy {np.__version__}, SciPy {metadata.version('scipy')}, "
        f"scikit-image {metadata.version('scikit-image')}.",
        "",
    ]
    choices: dict[str, Choice] = {}
    for model, files in by_model.items():
        name = MODEL_NAMES[model]
        stops = stops_of(cache / "stops", model)
        lines += [f"## {name}", "", "### Stopping at a tolerance", ""]
        if stops:
            lines += common.table(
                [
                    "$\\tau/\\sigma$",
                    "gap per sample ≤",
                    "files that reached it",
                    "iterations: median (least to largest)",
                    "iterations in all",
                    "largest \\|Δ PSNR\\|, binary64, dB",
                    "largest \\|Δ PSNR\\|, 8 bits, dB",
                    "largest \\|Δ SSIM\\|, 8 bits",
                    "acceptable",
                ],
                stop_rows(model, stops, files),
            )
            choice = choose(model, stops, files)
            if choice is None:
                lines += ["No tolerance was reached by every file.", ""]
            else:
                choices[model] = choice
                chosen = (
                    f"$\\tau/\\sigma = {choice.ratio:g}$, stopping at a gap per sample of {choice.tolerance:g}, "
                    f"and after at most {choice.iterations} iterations (twice the most a tuning file took, rounded up)"
                )
                if choice.accepted:
                    lines += [f"Chosen: {chosen}.", ""]
                else:
                    lines += [
                        "No ratio has a tolerance that every file reached and that is acceptable. Chosen, the "
                        f"least tolerance that every file reached: {chosen}. Stopping there changed the PSNR "
                        f"by up to {max(stop.psnr for stop in choice.stops):.3f} dB.",
                        "",
                    ]
        lines += [
            "The long runs' own change over their last sixth, against which the stops are measured: median "
            "(least to largest) over the files.",
            "",
            *common.table(
                ["iterations", "\\|Δ PSNR\\|, binary64, dB", "\\|Δ SSIM\\|, 8 bits"], [settling_row(files, model)]
            ),
            "### How fast each ratio brings the gap down",
            "",
            f"The iterations to the first gap per sample within a tolerance: median over the files (how many "
            f"reached it within {ITERATIONS[model]}).",
            "",
            *common.table(
                ["gap per sample ≤", *[f"$\\tau/\\sigma = {ratio:g}$" for ratio in RATIOS[model]]],
                reach_rows(files, model),
            ),
        ]
        marks = [iterations for iterations in PRIMAL_MARKS if iterations <= ITERATIONS[model]]
        lines += [
            "### The primal values",
            "",
            "$(P - D)/N$ at some iterations, with D the greatest dual value any run of the file reached: an "
            "upper bound of how far the primal value is from the least, tighter than a run's own gap. Median "
            "(least to largest) over the files.",
            "",
        ]
        header = ["$\\tau/\\sigma$", *[f"{iterations}" for iterations in marks]]
        lines += common.table(header, ratio_rows(files, model))
        lines += [
            "### The ratio that the bound of the rate chooses",
            "",
            "$(\\lVert z^\\ast - z^0 \\rVert / \\lVert y^\\ast \\rVert)^2$ from the start and the last "
            "point of each long run (docs/math.md, 5), with the distances per square root of the number of "
            "samples. Median (least to largest) over the images.",
            "",
        ]
        lines += common.table(
            [
                "quality",
                "$(\\lVert z^\\ast - z^0 \\rVert / \\lVert y^\\ast \\rVert)^2$",
                "$\\lVert z^\\ast - z^0 \\rVert / \\sqrt{N}$",
                "$\\lVert y^\\ast \\rVert / \\sqrt{N}$",
                "last gap per sample",
            ],
            bound_rows(files, model),
        )
        lines += [
            "### How far the iterates still move",
            "",
            "The RMS difference, in grey levels, of the canvas of each long run at some iterations from its "
            "last. Median (least to largest) over the files.",
            "",
        ]
        lines += common.table(["iterations", "RMS to the last"], drift_rows(files, model))
    start = start_rows(cache / "start")
    if start:
        lines += [
            "## TGV: the start of w",
            "",
            "w starting at the gradient of the start's canvas, or at 0; the ratio the bound would choose from "
            "the start and the last point; the first iteration whose gap per sample is within 0.1 and 0.03; "
            "and the PSNR of the last point.",
            "",
        ]
        lines += common.table(
            ["file", "w starts at", "$\\tau/\\sigma$", "the bound's ratio", "gap ≤ 0.1", "gap ≤ 0.03", "PSNR"],
            start,
        )
    if by_model:
        lines += ["## Figures", ""]
        lines += [*figures(out / "phase1", by_model, choices), ""]
    out.mkdir(parents=True, exist_ok=True)
    (out / "phase1-tuning.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return choices


COST: Final = {"tv": 1.0, "tgv": 2.5}  # the time of an iteration, relative to TV's


def solved(model: str, cases: list[common.Case]) -> list[common.Case]:
    """The files a model is solved on: every one for TV, the first image of each kind for TGV."""
    return [case for case in cases if model == "tv" or case.image in TGV_IMAGES]


def longest_first(jobs: list[tuple[Path, Run]]) -> list[tuple[Path, Run]]:
    """The jobs, the longest first, so that the workers finish together."""
    return sorted(jobs, key=lambda job: -job[1].iterations * COST[job[1].model])


def run_jobs(cases: list[common.Case], models: list[str], cache: Path) -> list[tuple[Path, Run]]:
    """The runs stage: every ratio, and the long run."""
    jobs = []
    for model in models:
        for case in solved(model, cases):
            for ratio in RATIOS[model]:
                length = LONG_ITERATIONS[model] if ratio == LONG_RATIO[model] else ITERATIONS[model]
                jobs.append((cache / "runs" / f"{model}-r{ratio:g}-{case.name}.json", Run(case, model, ratio, length)))
    return longest_first(jobs)


def start_jobs(cases: list[common.Case], cache: Path) -> list[tuple[Path, Run]]:
    """The start stage: TGV from w at the gradient and at 0."""
    return [
        (
            cache / "start" / f"tgv-{w_start}-r{ratio:g}-{case.name}.json",
            Run(case, "tgv", ratio, START_ITERATIONS, w_start=w_start),
        )
        for case in cases
        if (case.image, case.quality) in START_FILES
        for w_start in ("gradient", "zero")
        for ratio in START_RATIOS
    ]


def stop_jobs(cases: list[common.Case], models: list[str], cache: Path) -> list[tuple[Path, Run]]:
    """The stops stage: to the least tolerance, measuring where the gap first falls within each."""
    jobs = []
    for model in models:
        tolerances = STOP_TOLERANCES[model]
        for case in solved(model, cases):
            for ratio in STOP_RATIOS[model]:
                run = Run(case, model, ratio, LONG_ITERATIONS[model], tolerance=min(tolerances), crossings=tolerances)
                jobs.append((cache / "stops" / f"{model}-r{ratio:g}-{case.name}.json", run))
    return longest_first(jobs)


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--build", type=Path, required=True, help="a build directory with -DUNROUND_WITH_PYTHON=ON")
    parser.add_argument("--data", type=Path, default=common.ROOT / "data")
    parser.add_argument("--cache", type=Path, default=common.ROOT / "data" / "phase1" / "tuning")
    parser.add_argument("--out", type=Path, default=common.ROOT / "experiments" / "results")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--images", nargs="*", help="only these images (by name, such as chart-0-grey)")
    parser.add_argument("--qualities", type=int, nargs="*", help="only these qualities")
    parser.add_argument("--models", nargs="*", default=list(MODELS), choices=MODELS)
    parser.add_argument("stages", nargs="*", default=STAGES, choices=STAGES)
    options = parser.parse_args()
    common.use_build(options.build)

    chosen = [
        case
        for case in common.cases(options.data, "libjpeg-turbo", "tuning")
        if (not options.images or case.image in options.images)
        and (not options.qualities or case.quality in options.qualities)
    ]
    if "runs" in options.stages:
        common.run_all(solve, run_jobs(chosen, options.models, options.cache), options.workers, label="runs")
    if "start" in options.stages:
        common.run_all(solve, start_jobs(chosen, options.cache), options.workers, label="start")
    if "stops" in options.stages:
        common.run_all(solve, stop_jobs(chosen, options.models, options.cache), options.workers, label="stops")
    if "report" in options.stages:
        for model, choice in report(options.cache, options.out).items():
            print(
                f"{MODEL_NAMES[model]}: tau / sigma = {choice.ratio:g}, tolerance {choice.tolerance:g}, "
                f"at most {choice.iterations} iterations"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
