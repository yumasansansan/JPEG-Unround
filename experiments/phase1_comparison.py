#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Phase 1: the methods compared on the test images.

    python experiments/phase1_comparison.py --build build/release [--data data] [--workers 8]
        [--cache data/phase1/test] [--out experiments/results] [stage ...]

The files are the greyscale test images (data/synthetic/test) as libjpeg-turbo and
mozjpeg encoded them (scripts/test_images.py): 12 images at 6 qualities, for each
encoder. The model's weights, the ratio of the steps and the stopping tolerances are
those of unround.decode.Settings, which the tuning images chose
(experiments/phase1_tuning.py). The stages:

  run     On each file: the standard decoder (libjpeg's accurate integer inverse DCT,
          through the C layer); (d) the MMSE decoder; (a) the subgradient method of
          jpeg2png's kind for 500 iterations, and its picture at 50 as well
          (docs/math.md, 7); and (b) TV and (c) TGV by the primal-dual method, until
          the tolerance. (b)'s objective, recorded every 10 iterations, is compared
          with (a)'s.
  timing  The time an iteration of (a), (b) and (c) takes, on the quality-50 file of
          each image, the runs one after another in this process.
  report  The tables and the figures.

A run's results are kept as JSON in --cache, and a stage skips the runs whose results
are there already. No random number is drawn: the runs are deterministic.
"""

import argparse
import dataclasses
import math
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import numpy.typing as npt

import phase1_common as common

if TYPE_CHECKING:
    # For the annotations alone: the package is imported once its path is set.
    from unround.results import Result

type Array = npt.NDArray[np.float64]

ENCODERS: Final = ("libjpeg-turbo", "mozjpeg")
METHODS: Final = ("standard", "subgradient", "tv", "tgv")
# TGV is left out on mozjpeg's files: thousands of iterations on each, for what TV shows of
# them already (their intervals, which trellis quantization does not keep to).
METHODS_OF: Final = {"libjpeg-turbo": METHODS, "mozjpeg": ("standard", "subgradient", "tv")}
SUBGRADIENT_ITERATIONS: Final = 500
SUBGRADIENT_EARLY: Final = 50  # jpeg2png's default
RECORD_EVERY: Final = 10
TIMED_ITERATIONS: Final = 100
TIMED_REPEATS: Final = 3
TIMED_QUALITY: Final = 50
STAGES: Final = ["run", "timing", "report"]


@dataclasses.dataclass(frozen=True, slots=True)
class Task:
    """One method on one file."""

    case: common.Case
    method: str


def run(task: Task) -> dict[str, Any]:
    """Decodes a file by a method, and measures the picture."""
    from unround import decode, jpegio, pdhg, subgradient  # noqa: PLC0415
    from unround.model import Primal  # noqa: PLC0415

    settings = decode.Settings()
    data = task.case.jpeg.read_bytes()
    component = jpegio.read(data).components[0]
    problem = decode.component_problem(component, settings)
    original = common.read_original(task.case.original)
    height, width = component.height, component.width
    outcome: dict[str, Any] = {
        "encoder": task.case.encoder,
        "image": task.case.image,
        "quality": task.case.quality,
        "method": task.method,
        "samples": problem.samples,
    }
    if task.method == "standard":
        plane = jpegio.decode_planes(data).planes[0].astype(np.float64)
        outcome["standard"] = common.measure(original, plane[:height, :width], problem, plane)
        start = pdhg.start(problem)
        outcome["mmse"] = common.measure(original, start.canvas[:height, :width], problem, start.canvas)
        return outcome

    kept: dict[int, Array] = {}

    def observe(iteration: int, point: Primal, gap: float) -> None:  # noqa: ARG001
        if iteration == SUBGRADIENT_EARLY:
            kept[iteration] = point.canvas.copy()

    started = time.perf_counter()
    if task.method == "subgradient":
        options = subgradient.Options(iterations=SUBGRADIENT_ITERATIONS, record_every=RECORD_EVERY)
        result = subgradient.solve_tv(problem, settings.tv, options, observe=observe)
    elif task.method == "tv":
        result = pdhg.solve_tv(problem, settings.tv, settings.pdhg)
    else:
        result = pdhg.solve_tgv(problem, settings.tgv, settings.pdhg)
    outcome["wall_seconds"] = time.perf_counter() - started
    outcome["iterations"] = result.iterations
    outcome["converged"] = result.converged
    outcome["history"] = common.history_of(result)
    canvas = result.primal.canvas
    outcome["last"] = common.measure(original, canvas[:height, :width], problem, canvas)
    for iteration, canvas in kept.items():
        outcome[f"at_{iteration}"] = common.measure(original, canvas[:height, :width], problem, canvas)
    return outcome


def timing(case: common.Case) -> dict[str, Any]:
    """The least time an iteration of each solver took, over a few runs without records."""
    from unround import decode, jpegio, pdhg, subgradient  # noqa: PLC0415

    settings = decode.Settings()
    component = jpegio.read(case.jpeg.read_bytes()).components[0]
    problem = decode.component_problem(component, settings)
    timed = dataclasses.replace(settings.pdhg, iterations=TIMED_ITERATIONS, tolerance=0.0, record_every=0)
    sub = subgradient.Options(iterations=TIMED_ITERATIONS, record_every=0)
    solvers: dict[str, Callable[[], Result]] = {
        "subgradient": lambda: subgradient.solve_tv(problem, settings.tv, sub),
        "tv": lambda: pdhg.solve_tv(problem, settings.tv, timed),
        "tgv": lambda: pdhg.solve_tgv(problem, settings.tgv, timed),
    }
    outcome: dict[str, Any] = {"image": case.image, "quality": case.quality, "samples": problem.samples}
    for name, solver in solvers.items():
        seconds = []
        for _ in range(TIMED_REPEATS):
            result = solver()
            seconds.append(float(result.history.seconds[-1]) / result.iterations)
        outcome[name] = min(seconds)
    return outcome


type Files = dict[tuple[str, str, int], dict[str, dict[str, Any]]]

# The results shown, by the method and the measures they come from.
SHOWN: Final = (
    ("standard", "standard", "standard"),
    ("(d) MMSE", "standard", "mmse"),
    ("(a) 50", "subgradient", f"at_{SUBGRADIENT_EARLY}"),
    ("(a) 500", "subgradient", "last"),
    ("(b) TV", "tv", "last"),
    ("(c) TGV", "tgv", "last"),
)
FIGURE_FILES: Final = (("text-2-grey", 10), ("text-2-grey", 50), ("chart-2-grey", 10), ("chart-2-grey", 50))


def files_of(directory: Path) -> Files:
    """The results kept in a directory: by encoder, image and quality, then by method."""
    found: Files = {}
    for path in sorted(directory.glob("*.json")):
        result = common.load(path)
        key = (str(result["encoder"]), str(result["image"]), int(result["quality"]))
        found.setdefault(key, {})[str(result["method"])] = result
    return found


def measures(by_method: dict[str, dict[str, Any]], method: str, entry: str) -> dict[str, float]:
    """The measures of one result shown."""
    measured: dict[str, float] = by_method[method][entry]
    return measured


def quality_table(files: Files, encoder: str, measure: str, form: str) -> list[str]:
    """The mean of a measure over the images, by quality and by the results shown."""
    rows = []
    for quality in common.QUALITIES:
        chosen = [by_method for (e, _, q), by_method in files.items() if e == encoder and q == quality]
        if not chosen:
            continue
        cells = [str(quality)]
        for _, method, entry in SHOWN:
            values = [float(measures(by_method, method, entry)[measure]) for by_method in chosen if method in by_method]
            cells.append(f"{np.mean(values):{form}}" if values else "")
        rows.append(cells)
    return common.table(["quality", *[name for name, _, _ in SHOWN]], rows)


def consistency_rows(files: Files, encoder: str) -> list[list[str]]:
    """How far each result shown lies from the intervals, over every file of an encoder."""
    rows = []
    for name, method, entry in SHOWN:
        chosen = [
            measures(by_method, method, entry)
            for (e, _, _), by_method in files.items()
            if e == encoder and method in by_method
        ]
        if not chosen:
            continue
        cells = [name]
        for share, largest in (
            ("canvas_inside", "canvas_outside_largest"),
            ("whole_inside", "whole_outside_largest"),
            ("picture_inside", "picture_outside_largest"),
            ("whole_inside_8", "whole_outside_largest_8"),
            ("picture_inside_8", "picture_outside_largest_8"),
        ):
            least = min(float(measured[share]) for measured in chosen)
            most = max(float(measured[largest]) for measured in chosen)
            cells.append(f"{100.0 * least:.3f} / {most:.2e}")
        rows.append(cells)
    return rows


def iteration_rows(files: Files, encoder: str, method: str, per_iteration: dict[str, float]) -> list[list[str]]:
    """How many iterations a primal-dual method took to stop, whether it reached its tolerance, and the time."""
    rows = []
    for quality in common.QUALITIES:
        chosen = [
            (image, by_method[method])
            for (e, image, q), by_method in files.items()
            if e == encoder and q == quality and method in by_method
        ]
        if not chosen:
            continue
        iterations = [float(result["iterations"]) for _, result in chosen]
        reached = sum(1 for _, result in chosen if result["converged"])
        seconds = [float(result["iterations"]) * per_iteration[image] for image, result in chosen]
        rows.append(
            [
                str(quality),
                median_and_range(iterations, ".0f"),
                f"{reached} of {len(chosen)}",
                median_and_range(seconds, ".2f"),
            ]
        )
    return rows


def first_at_most(result: dict[str, Any], value: float) -> int | None:
    """The first recorded iteration of a run whose primal value is at most value."""
    for iteration, primal in zip(result["history"]["iterations"], result["history"]["primal"], strict=True):
        if float(primal) <= value:
            return int(iteration)
    return None


def objective_rows(files: Files, encoder: str, early: int) -> tuple[list[list[str]], int, int]:
    """The iterations (b) took to reach (a)'s objective after some iterations, by quality.

    (b) is the default TV run, until its tolerance. Returns the rows, and how many files (b)
    reached it on in fewer iterations than (a), of how many.
    """
    rows = []
    fewer = total = 0
    for quality in common.QUALITIES:
        chosen = [by_method for (e, _, q), by_method in files.items() if e == encoder and q == quality]
        if not chosen:
            continue
        needed = []
        for by_method in chosen:
            history = by_method["subgradient"]["history"]
            target = float(history["primal"][history["iterations"].index(early)])
            reached = first_at_most(by_method["tv"], target)
            needed.append(math.inf if reached is None else float(reached))
            fewer += 1 if reached is not None and reached < early else 0
            total += 1
        finite = [value for value in needed if math.isfinite(value)]
        rows.append(
            [
                str(quality),
                median_and_range(finite, ".0f") if finite else "",
                f"{len(needed) - len(finite)}",
                f"{sum(1 for value in needed if value < early)} of {len(needed)}",
            ]
        )
    return rows, fewer, total


def median_and_range(values: list[float], form: str) -> str:
    """The median of values, and their least and largest."""
    return f"{np.median(values):{form}} ({min(values):{form}} to {max(values):{form}})"


def comparison_figures(directory: Path, files: Files, per_iteration: dict[str, float]) -> list[str]:
    """Draws the objective of (a) and (b) against iterations and against time, and returns the markdown."""
    import matplotlib as mpl  # noqa: PLC0415

    mpl.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    directory.mkdir(parents=True, exist_ok=True)
    shown = []
    chosen = [
        (key, files[key])
        for key in (("libjpeg-turbo", image, quality) for image, quality in FIGURE_FILES)
        if key in files
    ]
    for name, against_time in (("comparison-objective.png", False), ("comparison-objective-time.png", True)):
        figure, axes = plt.subplots(2, 2, figsize=(10.0, 7.6), squeeze=False)
        for axis, ((_, image, quality), by_method) in zip(axes.ravel(), chosen, strict=False):
            lower = max(by_method["tv"]["history"]["dual"])
            samples = float(by_method["tv"]["samples"])
            for method, label, cost in (
                ("subgradient", "(a) subgradient", per_iteration.get(f"{image}/subgradient", 1.0)),
                ("tv", "(b) primal-dual", per_iteration.get(f"{image}/tv", 1.0)),
            ):
                history = by_method[method]["history"]
                iterations = np.asarray(history["iterations"][1:], dtype=np.float64)
                excess = (np.asarray(history["primal"][1:], dtype=np.float64) - lower) / samples
                axis.loglog(iterations * cost if against_time else iterations, excess, label=label)
            axis.set_title(f"{image} q{quality}")
            axis.set_xlabel("seconds" if against_time else "iterations")
            axis.set_ylabel("(P - D_best) / N")
            axis.grid(visible=True, which="major", linewidth=0.3)
            axis.legend(fontsize="small")
        for axis in axes.ravel()[len(chosen) :]:
            axis.set_visible(False)
        figure.tight_layout()
        figure.savefig(directory / name, dpi=100, metadata={"Software": None})
        plt.close(figure)
        what = "time" if against_time else "iterations"
        shown.append(f"![the objective of (a) and (b) against {what}](phase1/{name})")
    return shown


def comparison_report(cache: Path, out: Path) -> None:
    """Writes the tables and the figures of the comparison."""
    from importlib import metadata  # noqa: PLC0415

    from unround import decode, pdhg  # noqa: PLC0415

    files = files_of(cache / "run")
    timings = [common.load(path) for path in sorted((cache / "timing").glob("*.json"))]
    per_iteration: dict[str, float] = {}
    for timed in timings:
        per_iteration[str(timed["image"])] = float(timed["tv"])
        for method in ("subgradient", "tv", "tgv"):
            per_iteration[f"{timed['image']}/{method}"] = float(timed[method])
    settings = decode.Settings()
    tv, tgv = pdhg.plan(settings.pdhg, settings.tv), pdhg.plan(settings.pdhg, settings.tgv)
    encoders = sorted({encoder for encoder, _, _ in files})
    lines = [
        "<!-- Written by experiments/phase1_comparison.py; do not edit by hand. -->",
        "",
        "# Phase 1: the methods compared on the test images",
        "",
        f"- Files: the greyscale test images (`data/synthetic/test`), {len(files)} files of "
        f"{', '.join(encoders)} at qualities {', '.join(str(quality) for quality in common.QUALITIES)}.",
        "- Methods: the standard decoder (libjpeg's accurate integer inverse DCT, 8 bits); (d) the MMSE "
        f"decoder; (a) the subgradient method of jpeg2png's kind after {SUBGRADIENT_EARLY} and "
        f"{SUBGRADIENT_ITERATIONS} iterations; (b) TV and (c) TGV by the primal-dual method with the defaults "
        f"of `unround.decode.Settings`: for TV $\\tau/\\sigma = {tv.step_ratio:g}$, stopping at a gap per "
        f"sample of {tv.tolerance:g} or after {tv.iterations} iterations; for TGV $\\tau/\\sigma = "
        f"{tgv.step_ratio:g}$, {tgv.tolerance:g}, {tgv.iterations}.",
        "- Model: $\\mu = 10^{-3}$; TV $\\alpha = 1$; TGV $\\alpha_1 = 1$, $\\alpha_0 = 2$.",
        "- Measures, against the original: the PSNR of the binary64 result and of its 8-bit samples (rounded "
        "half away from zero, clamped to 0-255), and the SSIM (Gaussian weights of 1.5) and PSNR-B of the "
        "8-bit samples. The means are over the images.",
        f"- NumPy {np.__version__}, SciPy {metadata.version('scipy')}, "
        f"scikit-image {metadata.version('scikit-image')}.",
        "",
    ]
    for encoder in encoders:
        lines += [f"## Quality: `{encoder}`", ""]
        for measure, title, form in (
            ("psnr_8", "PSNR of the 8-bit samples, dB", ".3f"),
            ("ssim_8", "SSIM of the 8-bit samples", ".5f"),
            ("psnr_b_8", "PSNR-B of the 8-bit samples, dB", ".3f"),
            ("psnr", "PSNR of the binary64 result, dB", ".3f"),
        ):
            lines += [f"**{title}**", "", *quality_table(files, encoder, measure, form)]
    lines += [
        "## Consistency with the file",
        "",
        "For each result, over every file: the least share of the coefficients within their intervals, in "
        "percent, and the largest excess beyond them, in steps. The canvas is the solver's own (for the "
        "standard decoder, its planes of whole blocks); the picture is the result cut to the size of the "
        "image and padded again as libjpeg's encoder pads it, in binary64 and in 8 bits: whether a file that "
        "encoded it would read as the same file. The blocks wholly within the picture are those whose "
        "samples the padding does not touch; at the right and bottom edges of a picture that is not whole "
        "blocks, the padding repeats the picture's last samples where the canvas has its own. A coefficient "
        "that the solvers leave at an end of its interval is outside by the rounding of the round trip of "
        "the DCT half the time, which the shares count.",
        "",
    ]
    for encoder in encoders:
        lines += [
            f"**`{encoder}`**",
            "",
            *common.table(
                [
                    "result",
                    "canvas",
                    "picture, binary64: whole blocks",
                    "every block",
                    "picture, 8 bits: whole blocks",
                    "every block",
                ],
                consistency_rows(files, encoder),
            ),
        ]
    for encoder in encoders:
        lines += [f"## Iterations and time: `{encoder}`", ""]
        for method, name in (("tv", "(b) TV"), ("tgv", "(c) TGV")):
            if method not in METHODS_OF[encoder]:
                continue
            costs = {str(timed["image"]): float(timed[method]) for timed in timings}
            lines += [f"**{name}**", ""]
            lines += common.table(
                [
                    "quality",
                    "iterations: median (least to largest)",
                    "within the tolerance",
                    "seconds: median (least to largest)",
                ],
                iteration_rows(files, encoder, method, costs),
            )
    if timings:
        lines += [
            "The seconds are the iterations times the time an iteration took on the image, measured one run "
            "after another in one process (`timing`): per iteration, "
            + "; ".join(
                f"{method} {np.median([1e3 * float(timed[method]) for timed in timings]):.1f} ms"
                for method in ("subgradient", "tv", "tgv")
            )
            + " (medians over the images).",
            "",
        ]
    for encoder in encoders:
        lines += [f"## The objective: (b) against (a), `{encoder}`", ""]
        for early in (SUBGRADIENT_EARLY, SUBGRADIENT_ITERATIONS):
            rows, fewer, total = objective_rows(files, encoder, early)
            lines += [
                f"**The iterations (b) took to reach (a)'s objective after {early} iterations**: (b) needed fewer "
                f"on {fewer} of {total} files.",
                "",
                *common.table(
                    ["quality", "iterations: median (least to largest)", "not before its stop", f"fewer than {early}"],
                    rows,
                ),
            ]
    lines += ["## Figures", "", *comparison_figures(out / "phase1", files, per_iteration), ""]
    out.mkdir(parents=True, exist_ok=True)
    (out / "phase1-comparison.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--build", type=Path, required=True, help="a build directory with -DUNROUND_WITH_PYTHON=ON")
    parser.add_argument("--data", type=Path, default=common.ROOT / "data")
    parser.add_argument("--cache", type=Path, default=common.ROOT / "data" / "phase1" / "test")
    parser.add_argument("--out", type=Path, default=common.ROOT / "experiments" / "results")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--images", nargs="*", help="only these images (by name, such as chart-2-grey)")
    parser.add_argument("--qualities", type=int, nargs="*", help="only these qualities")
    parser.add_argument("--encoders", nargs="*", default=list(ENCODERS), choices=ENCODERS)
    parser.add_argument("stages", nargs="*", default=STAGES, choices=STAGES)
    options = parser.parse_args()
    common.use_build(options.build)

    chosen = [
        case
        for encoder in options.encoders
        for case in common.cases(options.data, encoder, "test")
        if (not options.images or case.image in options.images)
        and (not options.qualities or case.quality in options.qualities)
    ]
    if "run" in options.stages:
        cost = {"tgv": 4.0, "tv": 2.0, "subgradient": 1.0, "standard": 0.0}
        tasks = [
            (options.cache / "run" / f"{method}-{case.name}.json", Task(case, method))
            for case in chosen
            for method in METHODS_OF[case.encoder]
        ]
        tasks.sort(key=lambda task: -cost[task[1].method])
        common.run_all(run, tasks, options.workers, label="run")
    if "timing" in options.stages:
        timed = [
            (options.cache / "timing" / f"{case.image}.json", case)
            for case in chosen
            if case.encoder == "libjpeg-turbo" and case.quality == TIMED_QUALITY
        ]
        common.run_all(timing, timed, 1, label="timing")
    if "report" in options.stages:
        comparison_report(options.cache, options.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
