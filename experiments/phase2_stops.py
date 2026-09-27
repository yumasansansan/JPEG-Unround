#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Phase 2: when the primal-dual method stops, chosen again for the data term of the defaults.

    python experiments/phase2_stops.py --rust rust/target/release [--data data] [--workers 10]
        [--cache data/phase2/stops] [--out experiments/results] [stage ...]

The reconstructions are the reference implementation's, through the package's binding
(experiments/common.py), with the library's defaults of the model (docs/math.md, 4.1):
the defaults of the stop were chosen under the former data term (6.5). The files are the
tuning images of scripts/test_images.py as libjpeg-turbo encoded them, of four sets: the
synthetic greyscale images, the photographs in grey, and the synthetic and the
photographed colour images in 4:2:0 and 4:4:4. The stages:

  ratios      Two files of each set (RATIO_FILES), TV and TGV with each ratio of the steps
              of RATIOS, relaxed by RELAXATION, until the gap per sample is within
              RATIO_FLOORS or for RATIO_ITERATIONS: the iterations to each tolerance of
              TOLERANCES.
  references  The files of study(), four images of each set (two for TGV) at the qualities
              10, 50 and 90, with REFERENCES (a relaxation that no stop takes), until
              REFERENCE_FLOORS or for MOST iterations: the last point stands for the least
              point.
  stops       The same files with each variant of VARIANTS, until STOP_FLOORS or for MOST
              iterations: the result where each tolerance is first reached.
  report      The tables, and the variant and the tolerance that the criterion of
              docs/math.md, 6.5, chooses: stopping changes the PSNR of the binary64
              result, against the reference's last point, by at most LIMITS in the median
              and in the 90th percentile over the files, and the SSIM of its 8-bit samples
              likewise; every file reaches the tolerance; and the references tell it, their
              median last gap at most a tenth of it.

A run's results are kept as JSON in --cache, and a stage skips the runs whose results
are there already. No random number is drawn: the runs are deterministic.
"""

import argparse
import dataclasses
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt

import common

type Array = npt.NDArray[np.float64]
type Variant = tuple[float, float]  # the ratio of the steps, and the relaxation

MODELS: Final = ("tv", "tgv")
MODEL_NAMES: Final = {"tv": "TV", "tgv": "TGV"}
SETS: Final = ("synthetic, grey", "photographs, grey", "synthetic, colour", "photographs, colour")

# The ratios stage: two files of each set, as (set, image, quality, sampling), the image by
# its place among the set's sorted names.
RATIO_FILES: Final = (
    ("synthetic, grey", 0, 20, "grey"),
    ("synthetic, grey", 2, 70, "grey"),
    ("photographs, grey", 3, 50, "grey"),
    ("photographs, grey", 10, 10, "grey"),
    ("synthetic, colour", 1, 50, "420"),
    ("synthetic, colour", 3, 90, "444"),
    ("photographs, colour", 5, 30, "444"),
    ("photographs, colour", 12, 70, "420"),
)
RATIOS: Final = {"tv": (3.0, 10.0, 30.0, 100.0), "tgv": (1.0, 3.0, 10.0, 30.0)}
RELAXATION: Final = 1.9
RATIO_ITERATIONS: Final = 6000

# The tolerances of the gap per sample, the floors where the runs stop, and the criterion:
# the median and the PERCENTILE-th percentile over the files of the change that stopping
# makes, of the PSNR of the binary64 result (dB) and of the SSIM of its 8-bit samples.
TOLERANCES: Final = {
    "tv": (1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5),
    "tgv": (1e-1, 5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4),
}
RATIO_FLOORS: Final = {"tv": 1e-6, "tgv": 1e-5}
LIMITS: Final = {"psnr": (1e-3, 5e-3), "ssim_8": (1e-5, 5e-5)}
PERCENTILE: Final = 90.0
TELL: Final = 10.0  # a tolerance is told when it is this many times the references' median last gap

# The references and the stops: of each set, four images (by their places among the set's
# sorted names, with their samplings), TGV the first two of them, at STUDY_QUALITIES; the
# ratios of the steps that the ratios stage found fastest.
STUDY_IMAGES: Final = {
    "synthetic, grey": ((0, "grey"), (2, "grey"), (4, "grey"), (6, "grey")),
    "photographs, grey": ((0, "grey"), (6, "grey"), (12, "grey"), (18, "grey")),
    "synthetic, colour": ((0, "420"), (2, "444"), (4, "420"), (10, "444")),
    "photographs, colour": ((0, "420"), (6, "444"), (12, "420"), (18, "444")),
}
TGV_IMAGES: Final = 2
STUDY_QUALITIES: Final = (10, 50, 90)
REFERENCES: Final[dict[str, Variant]] = {"tv": (3.0, 1.5), "tgv": (3.0, 1.5)}
REFERENCE_FLOORS: Final = {"tv": 2e-6, "tgv": 1e-4}
VARIANTS: Final[dict[str, tuple[Variant, ...]]] = {"tv": ((3.0, 1.9), (10.0, 1.9)), "tgv": ((3.0, 1.9), (10.0, 1.9))}
STOP_FLOORS: Final = {"tv": 5e-5, "tgv": 1e-3}
MOST: Final = 8000
# The defaults of the stop before (docs/math.md, 6.5), to compare with.
FORMER: Final = {"tv": ((30.0, 1.9), 2e-4), "tgv": ((10.0, 1.9), 1e-2)}

STAGES: Final = ["ratios", "references", "stops", "report"]


@dataclasses.dataclass(frozen=True, slots=True)
class Run:
    """One run: a file of a set, the model, the steps' ratio and relaxation, and where it stops.

    measured runs measure the result where each tolerance is first reached, and at the end.
    """

    case: common.Case
    group: str
    model: str
    ratio: float
    relaxation: float
    floor: float
    most: int
    measured: bool

    @property
    def name(self) -> str:
        """The name of its results."""
        return f"{self.model}-r{self.ratio:g}-p{self.relaxation:g}-{self.case.name}"


def picture_of(canvas: Array, height: int, width: int) -> Array:
    """The picture of a canvas, (C, H, W): its one channel, or JFIF's RGB of its Y, Cb and Cr."""
    from unround import native  # noqa: PLC0415

    if canvas.shape[0] == 1:
        return canvas[0, :height, :width]
    return native.to_rgb(canvas[:, :height, :width])


def solve(run: Run) -> dict[str, Any]:
    """Solves a file until the floor or the most iterations: the records' gaps, and where each tolerance is reached."""
    from unround import jpegio, native  # noqa: PLC0415
    from unround.settings import PdhgOptions, Settings  # noqa: PLC0415

    case = run.case
    data = case.jpeg.read_bytes()
    image = jpegio.read(data)
    height, width = image.height, image.width
    original = common.read_original(case.original) if run.measured else None
    tolerances = sorted(TOLERANCES[run.model], reverse=True)
    iterations: list[int] = []
    gaps: list[float] = []
    crossings: dict[str, dict[str, float]] = {}

    def observer(record: native.Record) -> bool:
        iterations.append(record.iteration)
        gaps.append(record.gap)
        reached = [
            tolerance for tolerance in tolerances if f"{tolerance:g}" not in crossings and record.gap <= tolerance
        ]
        if reached:
            point: dict[str, float] = {"iteration": record.iteration, "gap": record.gap}
            if original is not None:
                point |= common.measure(original, picture_of(record.canvas, height, width))
            for tolerance in reached:
                crossings[f"{tolerance:g}"] = point
        return False

    options = PdhgOptions(iterations=run.most, tolerance=run.floor, step_ratio=run.ratio, relaxation=run.relaxation)
    settings = Settings(method=run.model, pdhg=options)  # type: ignore[arg-type]
    started = time.perf_counter()
    decoded = native.decode(data, settings, observer=observer)
    seconds = time.perf_counter() - started
    if decoded.result is None:
        message = f"{case.name}: the solver recorded nothing"
        raise RuntimeError(message)
    last: dict[str, float] = {"iteration": decoded.result.iterations, "gap": gaps[-1]}
    if original is not None:
        last |= common.measure(original, decoded.picture)
    return {
        "image": case.image,
        "quality": case.quality,
        "sampling": case.sampling,
        "set": run.group,
        "model": run.model,
        "ratio": run.ratio,
        "relaxation": run.relaxation,
        "samples": int(decoded.canvas.size),
        "iterations": decoded.result.iterations,
        "stop": decoded.result.stop.name.lower(),
        "seconds": seconds,
        "records": iterations,
        "gaps": gaps,
        "crossings": crossings,
        "last": last,
    }


type Sets = dict[str, list[common.Case]]


def sets_of(data: Path) -> Sets:
    """The four sets of tuning files."""
    colour = ("420", "444")
    return {
        "synthetic, grey": common.cases(data, "libjpeg-turbo", "tuning"),
        "photographs, grey": common.cases(
            data, "libjpeg-turbo", "tuning", jpeg="jpeg-photos", originals="photo-originals"
        ),
        "synthetic, colour": common.cases(data, "libjpeg-turbo", "tuning", samplings=colour),
        "photographs, colour": common.cases(
            data, "libjpeg-turbo", "tuning", samplings=colour, jpeg="jpeg-photos", originals="photo-originals"
        ),
    }


def ratio_cases(sets: Sets) -> list[tuple[str, common.Case]]:
    """The files of the ratios stage, with their sets."""
    chosen = []
    for group, place, quality, sampling in RATIO_FILES:
        images = sorted({case.image for case in sets[group]})
        chosen += [
            (group, case)
            for case in sets[group]
            if case.image == images[place] and case.quality == quality and case.sampling == sampling
        ]
    return chosen


def ratio_runs(sets: Sets) -> list[Run]:
    """Every file of the ratios stage with every ratio, for both models."""
    return [
        Run(case, group, model, ratio, RELAXATION, RATIO_FLOORS[model], RATIO_ITERATIONS, measured=False)
        for group, case in ratio_cases(sets)
        for model in MODELS
        for ratio in RATIOS[model]
    ]


def study(sets: Sets, model: str) -> list[tuple[str, common.Case]]:
    """The files of the references and the stops of a model, with their sets."""
    chosen = []
    for group, images in STUDY_IMAGES.items():
        names = sorted({case.image for case in sets[group]})
        for place, sampling in images[: TGV_IMAGES if model == "tgv" else len(images)]:
            chosen += [
                (group, case)
                for case in sets[group]
                if case.image == names[place] and case.sampling == sampling and case.quality in STUDY_QUALITIES
            ]
    return chosen


def reference_runs(sets: Sets) -> list[Run]:
    """Every file of study() with the reference's steps, for both models."""
    return [
        Run(case, group, model, *REFERENCES[model], REFERENCE_FLOORS[model], MOST, measured=True)
        for model in MODELS
        for group, case in study(sets, model)
    ]


def stop_runs(sets: Sets) -> list[Run]:
    """Every file of study() with every variant, for both models."""
    return [
        Run(case, group, model, ratio, relaxation, STOP_FLOORS[model], MOST, measured=True)
        for model in MODELS
        for group, case in study(sets, model)
        for ratio, relaxation in VARIANTS[model]
    ]


def jobs_of(runs: Sequence[Run], cache: Path, directory: str) -> list[tuple[Path, Run]]:
    """The runs' jobs, the largest canvases and TGV first, so that the longest start first."""
    ordered = sorted(runs, key=lambda run: (run.model != "tgv", run.group.endswith("grey"), run.name))
    return [(cache / directory / f"{run.name}.json", run) for run in ordered]


type Results = list[dict[str, Any]]


def load_all(cache: Path, directory: str) -> Results:
    """A stage's results."""
    return [common.load(path) for path in sorted((cache / directory).glob("*.json"))]


def ratio_section(results: Results) -> list[str]:
    """For each model and tolerance, the iterations each ratio takes to reach it, in all and at most over the files."""
    lines = [
        "## The ratio of the steps",
        "",
        f"Two files of each set, with the relaxation {RELAXATION:g}, until the gap per sample is within the floor or "
        f"for {RATIO_ITERATIONS} iterations: for each tolerance, the iterations the files took to reach it, in all "
        "and the most of one file; where a file did not reach it, how many did.",
        "",
    ]
    for model in MODELS:
        runs = [result for result in results if result["model"] == model]
        if not runs:
            continue
        ratios = sorted({float(result["ratio"]) for result in runs})
        rows = []
        for tolerance in TOLERANCES[model]:
            row = [f"{tolerance:g}"]
            for ratio in ratios:
                chosen = [result for result in runs if float(result["ratio"]) == ratio]
                reached = [result["crossings"].get(f"{tolerance:g}") for result in chosen]
                found = [int(point["iteration"]) for point in reached if point is not None]
                if len(found) == len(chosen):
                    row.append(f"{sum(found)} ({max(found)})")
                else:
                    row.append(f"{len(found)} of {len(chosen)}")
            rows.append(row)
        header = ["gap per sample ≤", *(f"$\\tau/\\sigma = {ratio:g}$" for ratio in ratios)]
        lines += [f"### {MODEL_NAMES[model]}", "", *common.table(header, rows)]
        seconds = [
            float(np.median([float(r["seconds"]) / max(int(r["iterations"]), 1) * 1e3 for r in runs if r["set"] == s]))
            for s in SETS
        ]
        lines += [
            "Milliseconds an iteration, the median of each set with 10 runs at a time: "
            + ", ".join(f"{group} {value:.1f}" for group, value in zip(SETS, seconds, strict=True))
            + ".",
            "",
        ]
    return lines


type Key = tuple[str, str, int, str]


def key_of(result: dict[str, Any]) -> Key:
    """A result's model, image, quality and sampling."""
    return (str(result["model"]), str(result["image"]), int(result["quality"]), str(result["sampling"]))


@dataclasses.dataclass(frozen=True, slots=True)
class Stop:
    """Where a file's run first reached a tolerance, and what stopping there changed against its reference."""

    group: str
    iterations: int
    psnr: float
    ssim_8: float


def stops_at(runs: Results, references: dict[Key, dict[str, Any]], tolerance: float) -> list[Stop]:
    """The stops of the runs that reached a tolerance, against their references' last points."""
    found = []
    for run in runs:
        crossing = run["crossings"].get(f"{tolerance:g}")
        if crossing is None:
            continue
        last = references[key_of(run)]["last"]
        found.append(
            Stop(
                str(run["set"]),
                int(crossing["iteration"]),
                abs(float(crossing["psnr"]) - float(last["psnr"])),
                abs(float(crossing["ssim_8"]) - float(last["ssim_8"])),
            )
        )
    return found


def spread(values: Sequence[float]) -> tuple[float, float, float]:
    """The median, the PERCENTILE-th percentile and the largest of values."""
    return float(np.median(values)), float(np.percentile(values, PERCENTILE)), float(max(values))


def acceptable(stops: Sequence[Stop]) -> bool:
    """Whether the changes of stopping keep within LIMITS, in the median and the percentile over the files."""
    for measure, (median_limit, percentile_limit) in LIMITS.items():
        median, percentile, _ = spread([float(getattr(stop, measure)) for stop in stops])
        if median > median_limit or percentile > percentile_limit:
            return False
    return True


def told(references: Results) -> float:
    """The least tolerance the references tell: TELL times their median last gap per sample."""
    return TELL * float(np.median([float(run["last"]["gap"]) for run in references]))


def cap_of(stops: Sequence[Stop]) -> int:
    """The most iterations: twice the most a file took, rounded up to 1, 2 or 5 times a power of ten."""
    most = 2 * max(stop.iterations for stop in stops)
    power = 1
    while True:
        for step in (1, 2, 5):
            if step * power >= most:
                return step * power
        power *= 10


@dataclasses.dataclass(frozen=True, slots=True)
class Choice:
    """The variant and the tolerance chosen for a model, and whether the criterion accepted it."""

    variant: Variant
    tolerance: float
    stops: list[Stop]
    accepted: bool


def tolerances_of(model: str) -> list[float]:
    """The tolerances the stops reach down to, from the least."""
    return sorted(tolerance for tolerance in TOLERANCES[model] if tolerance >= STOP_FLOORS[model])


def choose(
    model: str, runs: dict[Variant, Results], references: dict[Key, dict[str, Any]], least: float
) -> Choice | None:
    """The variant and the tolerance that stop the files acceptably in the fewest iterations in all.

    For each variant, over the tolerances that every file reached and that the references
    tell, the largest that is acceptable and whose smaller ones are all acceptable too. Where
    no variant has one, the choice is not accepted: the least of those tolerances, with the
    variant that got there in the fewest iterations in all.
    """
    best: Choice | None = None
    fallback: Choice | None = None
    for variant, files in runs.items():
        chosen: tuple[float, list[Stop]] | None = None
        for tolerance in tolerances_of(model):
            found = stops_at(files, references, tolerance)
            if len(found) < len(files) or tolerance < least:
                continue
            total = sum(stop.iterations for stop in found)
            if (
                fallback is None
                or tolerance < fallback.tolerance
                or (tolerance == fallback.tolerance and total < sum(stop.iterations for stop in fallback.stops))
            ):
                fallback = Choice(variant, tolerance, found, accepted=False)
            if not acceptable(found):
                break
            chosen = (tolerance, found)
        if chosen is None:
            continue
        tolerance, found = chosen
        if best is None or sum(stop.iterations for stop in found) < sum(stop.iterations for stop in best.stops):
            best = Choice(variant, tolerance, found, accepted=True)
    return best if best is not None else fallback


def variant_label(variant: Variant) -> str:
    """A variant as the tables show it."""
    return f"$\\tau/\\sigma = {variant[0]:g}$, $\\rho = {variant[1]:g}$"


def verdict_of(stops: Sequence[Stop], files: int, tolerance: float, least: float) -> str:
    """Whether the criterion accepts a tolerance, or why it is not judged."""
    if len(stops) < files:
        return "not reached"
    if tolerance < least:
        return "not told"
    return "yes" if acceptable(stops) else "no"


def stop_section(cache: Path) -> tuple[list[str], dict[str, Choice]]:
    """The references, what stopping at each tolerance changes, and the choice, for each model."""
    references_all = load_all(cache, "references")
    stops_all = load_all(cache, "stops")
    choices: dict[str, Choice] = {}
    if not references_all or not stops_all:
        return [], choices
    lines = [
        "## Stopping at a tolerance",
        "",
        f"Of each set, four images (two for TGV) at the qualities {', '.join(map(str, STUDY_QUALITIES))}. Stopping "
        "where the gap per sample first falls within a tolerance changes the result, against the reference's last "
        "point. A tolerance is acceptable when, over the files, the change of the PSNR of the binary64 result is at "
        f"most {LIMITS['psnr'][0]:g} dB in the median and {LIMITS['psnr'][1]:g} dB in the {PERCENTILE:g}th "
        f"percentile, and that of the SSIM of its 8-bit samples at most {LIMITS['ssim_8'][0]:g} and "
        f"{LIMITS['ssim_8'][1]:g}; every file has to reach it, and the references have to tell it: their median last "
        f"gap per sample at most 1/{TELL:g} of it. For each variant, the largest acceptable tolerance whose smaller "
        "ones are acceptable too; the variant is the one that then stops the files in the fewest iterations in all, "
        "and the most iterations twice the most a file took, rounded up to 1, 2 or 5 times a power of ten.",
        "",
    ]
    for model in MODELS:
        references_list = [result for result in references_all if result["model"] == model]
        if not references_list:
            continue
        references = {key_of(result): result for result in references_list}
        runs = {
            variant: [
                result
                for result in stops_all
                if result["model"] == model and (float(result["ratio"]), float(result["relaxation"])) == variant
            ]
            for variant in VARIANTS[model]
        }
        runs = {
            variant: files for variant, files in runs.items() if files and all(key_of(f) in references for f in files)
        }
        least = told(references_list)
        last_gaps = [float(result["last"]["gap"]) for result in references_list]
        at_most = sum(int(result["iterations"]) >= MOST for result in references_list)
        lines += [
            f"### {MODEL_NAMES[model]}",
            "",
            f"References: {variant_label(REFERENCES[model])}, until the gap per sample is within "
            f"{REFERENCE_FLOORS[model]:g} or for {MOST} iterations ({at_most} of {len(references_list)} files took "
            f"them all); their last gap per sample {np.median(last_gaps):.1e} in the median and {max(last_gaps):.1e} "
            f"at most: they tell the tolerances from {least:.1e}.",
            "",
        ]
        rows = []
        for variant, files in runs.items():
            for tolerance in reversed(tolerances_of(model)):
                found = stops_at(files, references, tolerance)
                reached = f"{len(found)} of {len(files)}"
                if not found:
                    rows.append([variant_label(variant), f"{tolerance:g}", reached, "", "", "", "", ""])
                    continue
                iterations = [float(stop.iterations) for stop in found]
                rows.append(
                    [
                        variant_label(variant),
                        f"{tolerance:g}",
                        reached,
                        f"{np.median(iterations):.0f} ({min(iterations):.0f} to {max(iterations):.0f})",
                        f"{sum(iterations):.0f}",
                        " / ".join(f"{value:.4f}" for value in spread([stop.psnr for stop in found])),
                        " / ".join(f"{value:.1e}" for value in spread([stop.ssim_8 for stop in found])),
                        verdict_of(found, len(files), tolerance, least),
                    ]
                )
        header = [
            "variant",
            "gap per sample ≤",
            "files that reached it",
            "iterations: median (least to largest)",
            "in all",
            "PSNR change (dB): median / p90 / largest",
            "SSIM change: median / p90 / largest",
            "acceptable",
        ]
        lines += common.table(header, rows)
        choice = choose(model, runs, references, least)
        if choice is None:
            continue
        choices[model] = choice
        verdict = "accepted" if choice.accepted else "not accepted by the criterion: the least tolerance told"
        lines += [
            f"The choice ({verdict}): {variant_label(choice.variant)}, a gap per sample of {choice.tolerance:g}, "
            f"and at most {cap_of(choice.stops)} iterations.",
            "",
        ]
        by_set = []
        former_variant, former_tolerance = FORMER[model]
        former = stops_at(runs.get(former_variant, []), references, former_tolerance)
        for group in SETS:
            chosen = [stop for stop in choice.stops if stop.group == group]
            if not chosen:
                continue
            before = [stop for stop in former if stop.group == group]
            changes = [stop.psnr for stop in chosen]
            row = [
                group,
                str(len(chosen)),
                f"{np.median([stop.iterations for stop in chosen]):.0f}",
                f"{max(stop.iterations for stop in chosen)}",
                " / ".join(f"{value:.4f}" for value in spread(changes)),
            ]
            if former:
                row.append(f"{np.median([stop.iterations for stop in before]):.0f}" if before else "")
            by_set.append(row)
        header = ["set", "files", "iterations: median", "most", "PSNR change (dB): median / p90 / largest"]
        if former:
            header.append(f"iterations at the former stop ({variant_label(former_variant)} at {former_tolerance:g})")
        lines += ["By set, at the choice:", "", *common.table(header, by_set)]
    return lines, choices


def comparison_section(cache: Path, choices: dict[str, Choice]) -> list[str]:
    """TGV against TV at their choices, on TGV's files: iterations, and time by the ratios stage's iterations."""
    if set(choices) != set(MODELS):
        return []
    ratios = load_all(cache, "ratios")
    stops_all = load_all(cache, "stops")
    rows = []
    for group in SETS:
        milliseconds = {
            model: float(
                np.median(
                    [
                        float(r["seconds"]) / max(int(r["iterations"]), 1) * 1e3
                        for r in ratios
                        if r["model"] == model and r["set"] == group
                    ]
                )
            )
            for model in MODELS
        }
        files = {key_of(r)[1:] for r in stops_all if r["model"] == "tgv" and r["set"] == group}
        medians = {}
        for model in MODELS:
            variant, tolerance = choices[model].variant, choices[model].tolerance
            reached = [
                int(r["crossings"][f"{tolerance:g}"]["iteration"])
                for r in stops_all
                if r["model"] == model
                and (float(r["ratio"]), float(r["relaxation"])) == variant
                and key_of(r)[1:] in files
                and f"{tolerance:g}" in r["crossings"]
            ]
            medians[model] = float(np.median(reached)) if reached else float("nan")
        seconds = {model: medians[model] * milliseconds[model] for model in MODELS}
        rows.append(
            [
                group,
                f"{medians['tgv']:.0f}",
                f"{medians['tv']:.0f}",
                f"{milliseconds['tgv']:.1f}",
                f"{milliseconds['tv']:.1f}",
                f"{seconds['tgv'] / seconds['tv']:.1f}",
            ]
        )
    return [
        "## TGV against TV",
        "",
        "At their choices, on the files that TGV takes: the median iterations of each, the milliseconds an iteration "
        "of the ratios stage (10 runs at a time), and the time of TGV over that of TV.",
        "",
        *common.table(
            [
                "set",
                "TGV: iterations",
                "TV: iterations",
                "TGV: ms an iteration",
                "TV: ms an iteration",
                "time, TGV over TV",
            ],
            rows,
        ),
    ]


def report(cache: Path, out: Path) -> None:
    """Writes the tables."""
    from importlib import metadata  # noqa: PLC0415

    from unround import native  # noqa: PLC0415

    lines = [
        "<!-- Written by experiments/phase2_stops.py; do not edit by hand. -->",
        "",
        "# Phase 2: when the primal-dual method stops, under the data term of the defaults",
        "",
        "- Files: the tuning images as libjpeg-turbo encoded them, of four sets: the synthetic greyscale images, the "
        "photographs in grey (BSDS500's train split), and the synthetic and the photographed colour images in 4:2:0 "
        "and 4:4:4.",
        "- The model is the library's default (docs/math.md, 4.1); the method is the relaxed one of 5, with the "
        "ratio of the steps $\\tau/\\sigma$ and the relaxation $\\rho$; the gap per sample is that of 6.4 (the partial "
        "gap of 6.6 where samples are free).",
        f"- The reference implementation, {native.version()}; NumPy {np.__version__}, scikit-image "
        f"{metadata.version('scikit-image')}.",
        "",
    ]
    lines += ratio_section(load_all(cache, "ratios"))
    stops, choices = stop_section(cache)
    lines += stops
    lines += comparison_section(cache, choices)
    out.mkdir(parents=True, exist_ok=True)
    (out / "phase2-stops.md").write_text("\n".join(lines).rstrip("\n") + "\n", encoding="utf-8", newline="\n")
    print(f"{(out / 'phase2-stops.md').as_posix()} written")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rust", type=Path, required=True, help="where cargo wrote unround_capi")
    parser.add_argument("--data", type=Path, default=common.ROOT / "data")
    parser.add_argument("--cache", type=Path, default=common.ROOT / "data" / "phase2" / "stops")
    parser.add_argument("--out", type=Path, default=common.ROOT / "experiments" / "results")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("stages", nargs="*", default=STAGES, choices=STAGES)
    options = parser.parse_args()
    common.use_library(options.rust)
    sets = sets_of(options.data)
    stages: dict[str, list[Run]] = {
        "ratios": ratio_runs(sets),
        "references": reference_runs(sets),
        "stops": stop_runs(sets),
    }
    for stage in STAGES:
        if stage not in options.stages:
            continue
        if stage == "report":
            report(options.cache, options.out)
        elif stage in stages:
            common.run_all(solve, jobs_of(stages[stage], options.cache, stage), options.workers, label=stage)
    return 0


if __name__ == "__main__":
    sys.exit(main())
