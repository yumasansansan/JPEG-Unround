#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Phase 2: colour -- the components of a file on one canvas, coupled or each on its own.

    python experiments/phase2_colour.py --build build/release [--data data] [--workers 10]
        [--cache data/phase2/colour] [--out experiments/results] [stage ...]

The files are the colour tuning images as libjpeg-turbo encoded them. The model is
that of unround.decode.Settings: TV and TGV with the channels coupled, pixel by
pixel, or each on its own (docs/math.md, 4.4), solved in YCbCr, and measured in the
RGB of JFIF's conversion (8). The stages:

  baselines   Every colour tuning file (4:2:0, 4:2:2 and 4:4:4 at every quality):
              libjpeg-turbo's own decoder, as Pillow calls it, and the decoders of the
              centres (docs/math.md, 2.3), the MMSE centres and the middles of the
              intervals, each chroma sample repeated over its cell.
  agreement   Every 4:4:4 tuning file: the decoder of the middles against libjpeg-
              turbo's, sample by sample in 8 bits, with Y, Cb and Cr clamped to 0-255
              before the conversion, as libjpeg does, and without.
  references  TV coupled on every image in SAMPLINGS at QUALITIES, for long and
              relaxed: its last point stands for the least point.
  stops       The same files, TV coupled with the package's defaults, until the gap per
              sample is within the least of TOLERANCES; the result is measured where the
              gap first falls within each. Where samples are free (4:2:0), the gap is the
              partial gap of 6.6.
  apart       The same files, TV with the channels each on its own, for APART_ITERATIONS;
              and TGV coupled and apart on the first image of each kind in TGV_SAMPLINGS,
              for TGV_ITERATIONS: the models compared where their PSNR has settled.
  paths       On the three files other than gradients whose stops changed most at the
              default tolerance, and on the gradient whose stop changed most: TV coupled
              along the defaults' path and the references', for PATH_ITERATIONS, with the
              PSNR of RGB and of Y every PATH_EVERY iterations. Where the two paths come
              together, the stops' changes are slow convergence; where they do not, the
              result depends on the path.
  report      The tables and the figure. The stops are held to the criterion of
              experiments/phase2_solver.py: the change that stopping makes against the
              references, in its median and percentile over the files.

The measures are those of RGB: the PSNR of the binary64 result, and the PSNR and the
SSIM (the mean over R, G and B) of its 8-bit samples, rounded and clamped to 0-255 as
libjpeg-turbo's are. The binary64 result is not clamped: its PSNR, which the stopping
criterion holds, counts what lies beyond 0-255, and so the decoders are compared in 8
bits. Consistency is measured on the
canvas, on the RGB of the whole canvas converted back, and on the 8-bit picture over
the blocks wholly within it. A run's results are kept as JSON in --cache, and a stage
skips the runs whose results are there already. No random number is drawn.
"""

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import numpy.typing as npt
from PIL import Image
from skimage.metrics import structural_similarity

import phase1_common as common
import phase2_solver as solver

if TYPE_CHECKING:
    # For the annotations alone: the package is imported once its path is set.
    from unround.frames import Frame

type Array = npt.NDArray[np.float64]
type Files = dict[tuple[str, str, int], dict[str, Any]]

SAMPLINGS: Final = ("420", "444")
QUALITIES: Final = (20, 50)
BASELINE_SAMPLINGS: Final = ("420", "422", "444")
REFERENCE: Final = (30.0, 1.5)  # TV's ratio of the steps and relaxation, off the defaults' path
REFERENCE_ITERATIONS: Final = 8000
TOLERANCES: Final = (1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4)
STOP_ITERATIONS: Final = 8000
APART_ITERATIONS: Final = 3000
TGV_IMAGES: Final = ("chart-0", "gradient-0", "illustration-0", "lineart-0", "text-0", "ui-0")
TGV_SAMPLINGS: Final = ("420",)
TGV_ITERATIONS: Final = 4000
RECORD_EVERY: Final = 10
DEFAULT_TOLERANCE: Final = 2e-4  # unround.pdhg.TV_TOLERANCE, which the stops are measured against

PATH_FILES: Final = ("chart-0-q50-444", "chart-0-q50-420", "text-0-q50-420", "gradient-1-q50-444")
PATH_VARIANTS: Final = ((30.0, 1.9), (30.0, 1.5))  # the defaults' path, and the references'
PATH_ITERATIONS: Final = 16000
PATH_EVERY: Final = 250
PATH_MARKS: Final = (2000, 4000, 8000, 16000)

STAGES: Final = ["baselines", "agreement", "references", "stops", "apart", "paths", "report"]


@dataclasses.dataclass(frozen=True, slots=True)
class Case:
    """A colour JPEG file of a synthetic image, and the image it was encoded from."""

    image: str
    quality: int
    sampling: str
    jpeg: Path
    original: Path

    @property
    def name(self) -> str:
        """The image, the quality and the sampling, as in the names of the results."""
        return f"{self.image}-q{self.quality}-{self.sampling}"


def colour_cases(data: Path, samplings: tuple[str, ...], qualities: tuple[int, ...]) -> list[Case]:
    """The colour tuning files that libjpeg-turbo encoded, in these samplings and at these qualities."""
    manifest = json.loads((data / "jpeg" / "libjpeg-turbo" / "manifest.json").read_text(encoding="utf-8"))
    found = []
    for entry in manifest["files"]:
        file, original = str(entry["file"]), str(entry["original"])
        sampling, quality = str(entry["sampling"]), int(entry["quality"])
        if not file.startswith("tuning/") or sampling not in samplings or quality not in qualities:
            continue
        found.append(
            Case(
                image=Path(original).name.removesuffix(".ppm"),
                quality=quality,
                sampling=sampling,
                jpeg=data / "jpeg" / "libjpeg-turbo" / file,
                original=data / "synthetic" / original,
            )
        )
    return sorted(found, key=lambda case: (case.image, case.sampling, case.quality))


@dataclasses.dataclass(frozen=True, slots=True)
class Run:
    """One solver run: a file, a model, whether its channels are coupled, and how long."""

    case: Case
    model: str
    coupled: bool
    iterations: int
    ratio: float | None = None  # None: the package's default, as is the relaxation
    relaxation: float | None = None
    crossings: tuple[float, ...] = ()  # tolerances at whose first crossing the result is measured

    @property
    def name(self) -> str:
        """The name of its results."""
        return f"{self.model}-{'coupled' if self.coupled else 'apart'}-{self.case.name}"


def read_rgb(path: Path) -> npt.NDArray[np.uint8]:
    """The samples of a colour original (8-bit PPM), (height, width, 3)."""
    with Image.open(path) as image:
        if image.mode != "RGB":
            message = f"{path} is not an 8-bit RGB picture: {image.mode}"
            raise ValueError(message)
        return np.asarray(image, dtype=np.uint8)


def ssim(reference: npt.NDArray[np.generic], image: npt.NDArray[np.generic]) -> float:
    """SSIM as common.ssim takes it, for each of R, G and B, and their mean."""
    # scikit-image declares no types.
    return float(
        structural_similarity(  # type: ignore[no-untyped-call]
            reference.astype(np.float64),
            image.astype(np.float64),
            data_range=255.0,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
            channel_axis=-1,
        )
    )


def quality(original: npt.NDArray[np.uint8], planes: Array) -> dict[str, float]:
    """The PSNR of the RGB of planes (Y, Cb, Cr cut to the picture), and the PSNR and SSIM of its 8 bits."""
    from unround import colour, metrics  # noqa: PLC0415

    rgb = colour.to_rgb(planes)
    eight = metrics.quantize(rgb)
    return {
        "psnr": metrics.psnr(original, rgb),
        "psnr_8": metrics.psnr(original, eight),
        "ssim_8": ssim(original, eight),
    }


def whole_blocks(frame: Frame, eight: npt.NDArray[np.uint8], height: int, width: int) -> dict[str, float]:
    """How consistent an 8-bit RGB picture is with the file, over the blocks wholly within the picture.

    Its YCbCr (JFIF's, in binary64) is filled out to the canvas by repeating the last row
    and column; the blocks whose cells are all within the picture do not depend on that.
    """
    from unround import colour, frames  # noqa: PLC0415

    planes = colour.to_ycbcr(eight.astype(np.float64))
    rows, columns = frame.shape
    padded = np.pad(planes, ((0, 0), (0, rows - height), (0, columns - width)), mode="edge")
    inside, total, largest = 0, 0, 0.0
    for excess, channel in zip(frames.excess(frame, padded), frame.channels, strict=True):
        whole = excess[: height // (8 * channel.ratio[0]), : width // (8 * channel.ratio[1])]
        inside += int(np.count_nonzero(whole == 0.0))
        total += whole.size
        largest = max(largest, float(whole.max()) if whole.size else 0.0)
    return {"whole_inside_8": inside / max(total, 1), "whole_outside_largest_8": largest}


def consistency(frame: Frame, canvas: Array, height: int, width: int) -> dict[str, float]:
    """The largest excess of the canvas, and of the RGB of the whole canvas converted back; and whole_blocks."""
    from unround import colour, frames, metrics  # noqa: PLC0415

    back = colour.to_ycbcr(colour.to_rgb(canvas))
    eight = metrics.quantize(colour.to_rgb(canvas[:, :height, :width]))
    return {
        "canvas_outside_largest": max(float(excess.max()) for excess in frames.excess(frame, canvas)),
        "back_outside_largest": max(float(excess.max()) for excess in frames.excess(frame, back)),
        **whole_blocks(frame, eight, height, width),
    }


def baseline(case: Case) -> dict[str, Any]:
    """libjpeg-turbo's decoder, and the decoders of the MMSE centres and of the middles."""
    from unround import decode, frames, jpegio, metrics  # noqa: PLC0415
    from unround.model import DataTerm  # noqa: PLC0415

    data = case.jpeg.read_bytes()
    image = jpegio.read(data)
    height, width = image.height, image.width
    original = read_rgb(case.original)
    frame = decode.frame_of(image)
    with Image.open(case.jpeg) as decoded:
        libjpeg = np.asarray(decoded.convert("RGB"), dtype=np.uint8)
    result: dict[str, Any] = {
        "image": case.image,
        "quality": case.quality,
        "sampling": case.sampling,
        "libjpeg": {
            "psnr_8": metrics.psnr(original, libjpeg),
            "ssim_8": ssim(original, libjpeg),
            **whole_blocks(frame, libjpeg, height, width),
        },
    }
    for name, terms in (("mmse", DataTerm()), ("midpoint", DataTerm(centres="midpoint"))):
        own = decode.frame_of(image, decode.Settings(data=terms))
        canvas = frames.start(own).canvas
        result[name] = {**quality(original, canvas[:, :height, :width]), **consistency(own, canvas, height, width)}
    return result


def agreement(case: Case) -> dict[str, Any]:
    """The decoder of the middles against libjpeg-turbo's own, in 8 bits, sample by sample.

    libjpeg clamps Y, Cb and Cr to 0-255 before converting them; the decoder of the middles
    is compared so clamped, and as it is.
    """
    from unround import colour, decode, frames, jpegio, metrics  # noqa: PLC0415
    from unround.model import DataTerm  # noqa: PLC0415

    image = jpegio.read(case.jpeg.read_bytes())
    frame = decode.frame_of(image, decode.Settings(data=DataTerm(centres="midpoint")))
    planes = frames.start(frame).canvas[:, : image.height, : image.width]
    with Image.open(case.jpeg) as decoded:
        theirs = np.asarray(decoded.convert("RGB"), dtype=np.int64)
    clamped = metrics.quantize(colour.to_rgb(np.clip(planes, 0.0, 255.0))).astype(np.int64) - theirs
    unclamped = metrics.quantize(colour.to_rgb(planes)).astype(np.int64) - theirs
    return {
        "image": case.image,
        "quality": case.quality,
        "sampling": case.sampling,
        "largest": int(np.max(np.abs(clamped))),
        "within_1": float(np.count_nonzero(np.abs(clamped) <= 1)) / clamped.size,
        "largest_unclamped": int(np.max(np.abs(unclamped))),
    }


@dataclasses.dataclass(frozen=True, slots=True)
class Trace:
    """A run of the paths stage: a file, and the ratio of the steps and the relaxation."""

    case: Case
    ratio: float
    relaxation: float

    @property
    def name(self) -> str:
        """The name of its results."""
        return f"tv-r{self.ratio:g}-p{self.relaxation:g}-{self.case.name}"


def trace(run: Trace) -> dict[str, Any]:
    """TV coupled along a path for PATH_ITERATIONS, with the gap and the PSNR of RGB and of Y every PATH_EVERY."""
    from unround import colour, decode, frames, jpegio, metrics, pdhg  # noqa: PLC0415
    from unround.model import TV  # noqa: PLC0415

    case = run.case
    image = jpegio.read(case.jpeg.read_bytes())
    frame = decode.frame_of(image)
    height, width = image.height, image.width
    original = read_rgb(case.original)
    luma = colour.to_ycbcr(original.astype(np.float64))[0]
    records: list[dict[str, float]] = []

    def observe(iteration: int, point: frames.Primal, gap: float) -> None:
        planes = point.canvas[:, :height, :width]
        records.append(
            {
                "iteration": iteration,
                "gap": gap,
                "psnr": metrics.psnr(original, colour.to_rgb(planes)),
                "psnr_y": metrics.psnr(luma, planes[0]),
            }
        )

    options = pdhg.Options(
        iterations=PATH_ITERATIONS,
        tolerance=0.0,
        step_ratio=run.ratio,
        relaxation=run.relaxation,
        record_every=PATH_EVERY,
    )
    pdhg.solve_frame_tv(frame, TV(), options, observe=observe)
    return {
        "image": case.image,
        "quality": case.quality,
        "sampling": case.sampling,
        "ratio": run.ratio,
        "relaxation": run.relaxation,
        "records": records,
    }


def solve(run: Run) -> dict[str, Any]:
    """Solves a file, and measures its result, and where the gap first fell within each tolerance."""
    from unround import decode, frames, jpegio, pdhg  # noqa: PLC0415
    from unround.model import TGV, TV  # noqa: PLC0415

    case = run.case
    image = jpegio.read(case.jpeg.read_bytes())
    frame = decode.frame_of(image)
    height, width = image.height, image.width
    original = read_rgb(case.original)
    crossings: dict[str, dict[str, float]] = {}
    pending = sorted(run.crossings, reverse=True)

    def observe(iteration: int, point: frames.Primal, gap: float) -> None:
        while pending and gap <= pending[0]:
            tolerance = pending.pop(0)
            crossings[f"{tolerance:g}"] = {
                "iteration": iteration,
                **quality(original, point.canvas[:, :height, :width]),
            }

    options = pdhg.Options(
        iterations=run.iterations,
        tolerance=min(run.crossings, default=0.0),
        step_ratio=run.ratio,
        relaxation=run.relaxation,
        record_every=RECORD_EVERY,
    )
    if run.model == "tv":
        result = pdhg.solve_frame_tv(frame, TV(coupled=run.coupled), options, observe=observe)
    else:
        result = pdhg.solve_frame_tgv(frame, TGV(coupled=run.coupled), options, observe=observe)
    canvas = result.primal.canvas
    history = result.history
    return {
        "image": case.image,
        "quality": case.quality,
        "sampling": case.sampling,
        "model": run.model,
        "coupled": run.coupled,
        "samples": frame.samples,
        "iterations": result.iterations,
        "converged": result.converged,
        "seconds": float(history.seconds[-1]),
        "records": [int(value) for value in history.iterations],
        "gaps": [float(value) for value in history.gap / frame.samples],
        "last": {**quality(original, canvas[:, :height, :width]), **consistency(frame, canvas, height, width)},
        "crossings": crossings,
    }


def by_file(directory: Path, pattern: str) -> Files:
    """The results kept in a directory whose names match, by image, sampling and quality."""
    found: Files = {}
    for path in sorted(directory.glob(pattern)):
        result = common.load(path)
        found[(str(result["image"]), str(result["sampling"]), int(result["quality"]))] = result
    return found


def baseline_jobs(data: Path, cache: Path) -> list[tuple[Path, Case]]:
    """The baselines stage: every colour tuning file."""
    cases = colour_cases(data, BASELINE_SAMPLINGS, common.QUALITIES)
    return [(cache / "baselines" / f"{case.name}.json", case) for case in cases]


def path_jobs(data: Path, cache: Path) -> list[tuple[Path, Trace]]:
    """The paths stage's runs."""
    cases = [case for case in colour_cases(data, SAMPLINGS, QUALITIES) if case.name in PATH_FILES]
    runs = [Trace(case, ratio, rho) for case in cases for ratio, rho in PATH_VARIANTS]
    return [(cache / "paths" / f"{run.name}.json", run) for run in runs]


def path_rows(cache: Path) -> list[list[str]]:
    """The two paths of each file at PATH_MARKS: their gaps, their PSNR of RGB and how far apart, and of Y."""
    rows = []
    for name in PATH_FILES:
        traces = []
        for ratio, rho in PATH_VARIANTS:
            found = cache / "paths" / f"tv-r{ratio:g}-p{rho:g}-{name}.json"
            if found.exists():
                traces.append({int(record["iteration"]): record for record in common.load(found)["records"]})
        if len(traces) != len(PATH_VARIANTS):
            continue
        for mark in PATH_MARKS:
            first, second = traces[0][mark], traces[1][mark]
            rows.append(
                [
                    name,
                    str(mark),
                    f"{first['gap']:.1e} / {second['gap']:.1e}",
                    f"{first['psnr']:.4f} / {second['psnr']:.4f}",
                    f"{abs(first['psnr'] - second['psnr']):.4f}",
                    f"{first['psnr_y']:.4f} / {second['psnr_y']:.4f}",
                ]
            )
    return rows


def agreement_jobs(data: Path, cache: Path) -> list[tuple[Path, Case]]:
    """The agreement stage: every 4:4:4 tuning file."""
    cases = colour_cases(data, ("444",), common.QUALITIES)
    return [(cache / "agreement" / f"{case.name}.json", case) for case in cases]


def run_jobs(data: Path, cache: Path, stage: str) -> list[tuple[Path, Run]]:
    """The references, stops or apart stage's runs, the longest first."""
    cases = colour_cases(data, SAMPLINGS, QUALITIES)
    runs: list[Run] = []
    if stage == "references":
        runs = [
            Run(case, "tv", coupled=True, iterations=REFERENCE_ITERATIONS, ratio=REFERENCE[0], relaxation=REFERENCE[1])
            for case in cases
        ]
    elif stage == "stops":
        runs = [Run(case, "tv", coupled=True, iterations=STOP_ITERATIONS, crossings=TOLERANCES) for case in cases]
    else:
        runs = [Run(case, "tv", coupled=False, iterations=APART_ITERATIONS) for case in cases]
        for case in colour_cases(data, TGV_SAMPLINGS, QUALITIES):
            if case.image in TGV_IMAGES:
                runs += [Run(case, "tgv", coupled=coupled, iterations=TGV_ITERATIONS) for coupled in (True, False)]
    cost = {"tv": 1.0, "tgv": 2.5}
    runs.sort(key=lambda run: -run.iterations * cost[run.model])
    return [(cache / stage / f"{run.name}.json", run) for run in runs]


def spread(values: list[float]) -> tuple[float, float, float]:
    """The median, the percentile of the criterion and the largest of values."""
    return float(np.median(values)), float(np.percentile(values, solver.PERCENTILE)), max(values)


def stop_changes(stops: Files, references: Files, tolerance: float) -> list[dict[str, float]]:
    """Where every file's run first brought its gap within a tolerance, and the change there against its reference."""
    found = []
    for key, run in stops.items():
        crossing = run["crossings"].get(f"{tolerance:g}")
        if crossing is None or key not in references:
            continue
        last = references[key]["last"]
        found.append(
            {
                "iterations": float(crossing["iteration"]),
                "psnr": abs(float(crossing["psnr"]) - float(last["psnr"])),
                "ssim_8": abs(float(crossing["ssim_8"]) - float(last["ssim_8"])),
            }
        )
    return found


def acceptable(changes: list[dict[str, float]]) -> bool:
    """Whether the changes keep within the criterion of phase2_solver.py, in the median and the percentile."""
    for measure, (median_limit, percentile_limit) in solver.LIMITS.items():
        median, percentile, _ = spread([change[measure] for change in changes])
        if median > median_limit or percentile > percentile_limit:
            return False
    return True


def stop_rows(stops: Files, references: Files) -> tuple[list[list[str]], float | None]:
    """The table of the stops, and the tolerance the criterion chooses (None where none is acceptable).

    The choice is the largest tolerance that every file reached, that the references tell,
    and that is acceptable with every smaller one that meets the same.
    """
    told = solver.TELL * float(np.median([run["gaps"][-1] for run in references.values()]))
    rows, chosen, failed = [], None, False
    for tolerance in sorted(TOLERANCES):
        changes = stop_changes(stops, references, tolerance)
        if not changes:
            continue
        psnr = spread([change["psnr"] for change in changes])
        similarity = spread([change["ssim_8"] for change in changes])
        iterations = [change["iterations"] for change in changes]
        valid = len(changes) == len(stops) and tolerance >= told
        verdict = acceptable(changes)
        if valid and not verdict:
            failed = True
        if valid and verdict and not failed:
            chosen = tolerance
        rows.append(
            [
                f"{tolerance:g}",
                f"{len(changes)} of {len(stops)}",
                f"{np.median(iterations):.0f} ({min(iterations):.0f} to {max(iterations):.0f})",
                f"{psnr[0]:.4f} / {psnr[1]:.4f} / {psnr[2]:.4f}",
                f"{similarity[0]:.1e} / {similarity[1]:.1e} / {similarity[2]:.1e}",
                ("yes" if verdict else "no") + ("" if tolerance >= told else " (not told)"),
            ]
        )
    return rows, chosen


def medians(files: Files, path: tuple[str, str]) -> dict[tuple[str, int], float]:
    """The median over the images of a measure, by sampling and quality."""
    grouped: dict[tuple[str, int], list[float]] = {}
    for (_, sampling, quality_), result in files.items():
        grouped.setdefault((sampling, quality_), []).append(float(result[path[0]][path[1]]))
    return {key: float(np.median(values)) for key, values in grouped.items()}


def quality_rows(baselines: Files, references: Files, apart: Files) -> list[list[str]]:
    """The median 8-bit PSNR and SSIM of the baselines and of TV, coupled and apart, on TV's files."""
    shared = {key: result for key, result in baselines.items() if key in references}
    columns = [
        medians(shared, ("libjpeg", "psnr_8")),
        medians(shared, ("mmse", "psnr_8")),
        medians(references, ("last", "psnr_8")),
        medians(apart, ("last", "psnr_8")),
        medians(shared, ("libjpeg", "ssim_8")),
        medians(references, ("last", "ssim_8")),
        medians(apart, ("last", "ssim_8")),
    ]
    rows = []
    for key in sorted(columns[0]):
        values = [column.get(key, float("nan")) for column in columns]
        rows.append(
            [
                sampling_name(key[0]),
                str(key[1]),
                *(f"{value:.2f}" for value in values[:4]),
                *(f"{value:.4f}" for value in values[4:]),
            ]
        )
    return rows


def baseline_rows(baselines: Files) -> list[list[str]]:
    """The median 8-bit PSNR and SSIM of the three baselines on every file."""
    columns = [
        medians(baselines, ("libjpeg", "psnr_8")),
        medians(baselines, ("midpoint", "psnr_8")),
        medians(baselines, ("mmse", "psnr_8")),
        medians(baselines, ("libjpeg", "ssim_8")),
        medians(baselines, ("mmse", "ssim_8")),
    ]
    rows = []
    for key in sorted(columns[0]):
        values = [column[key] for column in columns]
        rows.append(
            [
                sampling_name(key[0]),
                str(key[1]),
                *(f"{value:.2f}" for value in values[:3]),
                *(f"{value:.4f}" for value in values[3:]),
            ]
        )
    return rows


def sampling_name(sampling: str) -> str:
    """A sampling as the tables show it: 4:2:0 for 420."""
    return ":".join(sampling)


def median_and_range(values: list[float], form: str) -> str:
    """The median of values, and their least and largest."""
    return f"{np.median(values):{form}} ({min(values):{form}} to {max(values):{form}})"


def difference_rows(first: Files, second: Files, label: str) -> list[list[str]]:
    """The difference of the last 8-bit PSNR and SSIM of two sets of runs, file by file: median (least to largest)."""
    rows = []
    for sampling in sorted({key[1] for key in first}):
        keys = [key for key in first if key[1] == sampling and key in second]
        if not keys:
            continue
        psnr = [float(first[key]["last"]["psnr_8"]) - float(second[key]["last"]["psnr_8"]) for key in keys]
        similarity = [float(first[key]["last"]["ssim_8"]) - float(second[key]["last"]["ssim_8"]) for key in keys]
        rows.append(
            [
                label,
                sampling_name(sampling),
                str(len(keys)),
                median_and_range(psnr, "+.3f"),
                median_and_range(similarity, "+.4f"),
            ]
        )
    return rows


def gain_rows(baselines: Files, runs: dict[str, Files]) -> list[list[str]]:
    """The 8-bit PSNR and SSIM of each set of runs less libjpeg-turbo's, file by file, by sampling and quality."""
    rows = []
    for name, files in runs.items():
        for sampling, quality_ in sorted({(key[1], key[2]) for key in files}):
            keys = [key for key in files if key[1:] == (sampling, quality_) and key in baselines]
            psnr = [float(files[key]["last"]["psnr_8"]) - float(baselines[key]["libjpeg"]["psnr_8"]) for key in keys]
            similarity = [
                float(files[key]["last"]["ssim_8"]) - float(baselines[key]["libjpeg"]["ssim_8"]) for key in keys
            ]
            better = sum(1 for value in psnr if value > 0.0)
            rows.append(
                [
                    name,
                    sampling_name(sampling),
                    str(quality_),
                    f"{better} of {len(keys)}",
                    median_and_range(psnr, "+.2f"),
                    median_and_range(similarity, "+.4f"),
                ]
            )
    return rows


def consistency_rows(baselines: Files, runs: dict[str, Files]) -> list[list[str]]:
    """The largest excess of the canvases and of the RGB converted back, and the 8-bit pictures' share inside."""
    rows = []
    for name in ("libjpeg", "mmse", "midpoint"):
        values = [result[name] for result in baselines.values()]
        rows.append(consistency_row(name, values))
    for name, files in runs.items():
        rows.append(consistency_row(name, [result["last"] for result in files.values()]))
    return rows


def consistency_row(name: str, values: list[dict[str, float]]) -> list[str]:
    """One row of consistency_rows."""
    canvas = [value["canvas_outside_largest"] for value in values if "canvas_outside_largest" in value]
    back = [value["back_outside_largest"] for value in values if "back_outside_largest" in value]
    inside = [value["whole_inside_8"] for value in values]
    return [
        name,
        str(len(values)),
        f"{max(canvas):.1e}" if canvas else "",
        f"{max(back):.1e}" if back else "",
        f"{100.0 * float(np.median(inside)):.2f} ({100.0 * min(inside):.2f})",
    ]


def time_rows(runs: dict[str, Files]) -> list[list[str]]:
    """The median milliseconds of an iteration, by the runs and the sampling, while the others ran."""
    rows = []
    for name, files in runs.items():
        for sampling in sorted({key[1] for key in files}):
            costs = [
                1e9 * float(result["seconds"]) / max(int(result["iterations"]), 1) / float(result["samples"])
                for key, result in files.items()
                if key[1] == sampling
            ]
            rows.append([name, sampling_name(sampling), f"{np.median(costs):.1f}"])
    return rows


def figure(directory: Path, baselines: Files, references: Files, apart: Files) -> list[str]:
    """The gain of MMSE and of TV over libjpeg-turbo's decoder, by quality, and the markdown that shows it."""
    import matplotlib as mpl  # noqa: PLC0415

    mpl.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    directory.mkdir(parents=True, exist_ok=True)
    samplings = sorted({key[1] for key in references})
    shown, axes = plt.subplots(1, len(samplings), figsize=(4.5 * len(samplings), 3.6), squeeze=False)
    for axis, sampling in zip(axes[0], samplings, strict=True):
        for label, files, path in (
            ("MMSE", baselines, ("mmse", "psnr_8")),
            ("TV apart", apart, ("last", "psnr_8")),
            ("TV coupled", references, ("last", "psnr_8")),
        ):
            gains: dict[int, list[float]] = {}
            for key, result in files.items():
                if key[1] != sampling or key not in baselines:
                    continue
                base = float(baselines[key]["libjpeg"]["psnr_8"])
                gains.setdefault(key[2], []).append(float(result[path[0]][path[1]]) - base)
            if gains:
                qualities = sorted(gains)
                axis.plot(qualities, [np.median(gains[q]) for q in qualities], marker="o", label=label)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(sampling_name(sampling))
        axis.set_xlabel("quality")
        axis.set_ylabel("8-bit PSNR over libjpeg-turbo's (dB, median)")
        axis.grid(visible=True, linewidth=0.3)
        axis.legend(fontsize="small")
    shown.tight_layout()
    shown.savefig(directory / "colour-gain.png", dpi=100, metadata={"Software": None})
    plt.close(shown)
    return ["![the gain in PSNR over libjpeg-turbo's decoder, by quality](phase2/colour-gain.png)", ""]


def report(cache: Path, out: Path) -> float | None:
    """Writes the tables and the figure, and returns the tolerance the stops choose."""
    from importlib import metadata  # noqa: PLC0415

    baselines = by_file(cache / "baselines", "*.json")
    references = by_file(cache / "references", "tv-coupled-*.json")
    stops = by_file(cache / "stops", "tv-coupled-*.json")
    apart = by_file(cache / "apart", "tv-apart-*.json")
    tgv_coupled = by_file(cache / "apart", "tgv-coupled-*.json")
    tgv_apart = by_file(cache / "apart", "tgv-apart-*.json")
    lines = [
        "<!-- Written by experiments/phase2_colour.py; do not edit by hand. -->",
        "",
        "# Phase 2: colour",
        "",
        f"- Files: the colour tuning images as libjpeg-turbo encoded them. Baselines on all {len(baselines)}; TV "
        f"on the {len(references)} in {', '.join(SAMPLINGS)} at the qualities {', '.join(map(str, QUALITIES))}; "
        f"TGV on the {len(tgv_coupled)} of the first image of each kind in {', '.join(TGV_SAMPLINGS)}.",
        "- The model of `unround.decode.Settings`, solved in YCbCr (docs/math.md, 4.4), with the channels coupled "
        "pixel by pixel or each on its own; measured in the RGB of JFIF's conversion (8): the PSNR of the binary64 "
        "result, and the SSIM of its 8-bit samples, the mean over R, G and B.",
        f"- References: TV coupled with $\\tau/\\sigma = {REFERENCE[0]:g}$, $\\rho = {REFERENCE[1]:g}$, "
        f"{REFERENCE_ITERATIONS} iterations. Stops: the package's defaults. TV apart: {APART_ITERATIONS} "
        f"iterations; TGV: {TGV_ITERATIONS}, with the defaults.",
        f"- NumPy {np.__version__}, scikit-image {metadata.version('scikit-image')}, Pillow "
        f"{metadata.version('pillow')}.",
        "",
        "## The baselines, on every file",
        "",
        "Median PSNR (dB) and SSIM of 8-bit pictures over the images: libjpeg-turbo's decoder, and the decoders of "
        "the middles of the intervals and of the MMSE centres, each chroma sample repeated over its cell, rounded and "
        "clamped.",
        "",
    ]
    lines += common.table(
        ["sampling", "quality", "libjpeg", "middles", "MMSE", "SSIM libjpeg", "SSIM MMSE"], baseline_rows(baselines)
    )
    agreements = by_file(cache / "agreement", "*.json")
    if agreements:
        largest = max(int(result["largest"]) for result in agreements.values())
        within = [float(result["within_1"]) for result in agreements.values()]
        unclamped = max(int(result["largest_unclamped"]) for result in agreements.values())
        lines += [
            f"In 4:4:4, the decoder of the middles is libjpeg-turbo's but for their arithmetic: over the "
            f"{len(agreements)} files, with Y, Cb and Cr clamped to 0-255 before the conversion as libjpeg does, their "
            f"8-bit samples differ by at most {largest}, and by at most 1 in {100.0 * min(within):.2f} per cent of the "
            f"samples or more (median {100.0 * float(np.median(within)):.2f}). Unclamped, as the result here is, they "
            f"differ by up to {unclamped} where Y overshoots 255 or 0 (docs/math.md, 8).",
            "",
        ]
    lines += [
        "## TV, coupled and apart",
        "",
        "Median PSNR (dB) and SSIM of 8-bit pictures over the images, on TV's files: libjpeg-turbo's decoder, the "
        "MMSE decoder, and TV's results, coupled (the references' last points) and apart.",
        "",
    ]
    lines += common.table(
        [
            "sampling",
            "quality",
            "libjpeg",
            "MMSE",
            "TV coupled",
            "TV apart",
            "SSIM libjpeg",
            "SSIM coupled",
            "SSIM apart",
        ],
        quality_rows(baselines, references, apart),
    )
    lines += figure(out / "phase2", baselines, references, apart)
    lines += [
        "Less libjpeg-turbo's decoder, file by file, in 8 bits: the files where the PSNR is higher, and the PSNR "
        "and the SSIM, median (least to largest).",
        "",
    ]
    gains = {"TV coupled": references, "TV apart": apart, "TGV coupled": tgv_coupled, "TGV apart": tgv_apart}
    lines += common.table(["model", "sampling", "quality", "higher", "PSNR (dB)", "SSIM"], gain_rows(baselines, gains))
    lines += ["Coupled less apart, in 8 bits, file by file: median (least to largest).", ""]
    rows = difference_rows(references, apart, "TV")
    rows += difference_rows(tgv_coupled, tgv_apart, "TGV")
    lines += common.table(["model", "sampling", "files", "PSNR (dB)", "SSIM"], rows)
    lines += ["TGV coupled less TV coupled, on TGV's files.", ""]
    lines += common.table(
        ["model", "sampling", "files", "PSNR (dB)", "SSIM"], difference_rows(tgv_coupled, references, "TGV less TV")
    )
    table_rows, chosen = stop_rows(stops, references)
    lines += [
        "## Stopping TV coupled at a tolerance",
        "",
        "Where the gap per sample (the partial gap of docs/math.md, 6.6, where samples are free) first falls within "
        "each tolerance, the change against the reference's last point: median / "
        f"{solver.PERCENTILE:g}th percentile / largest, over the files. Acceptable as phase2_solver.py has it: PSNR "
        f"{solver.LIMITS['psnr'][0]:g} and {solver.LIMITS['psnr'][1]:g} dB, SSIM {solver.LIMITS['ssim_8'][0]:g} and "
        f"{solver.LIMITS['ssim_8'][1]:g}; told where at least {solver.TELL:g} times the references' median last gap.",
        "",
    ]
    lines += common.table(["gap per sample ≤", "files", "iterations", "PSNR (dB)", "SSIM", "acceptable"], table_rows)
    lines += [
        f"The criterion chooses {chosen:g}." if chosen is not None else "No tolerance tried is acceptable and told.",
        f"The package's default is {DEFAULT_TOLERANCE:g}.",
        "",
    ]
    rows = path_rows(cache)
    if rows:
        lines += [
            "### Two paths, for long",
            "",
            f"TV coupled along the defaults' path ($\\tau/\\sigma = {PATH_VARIANTS[0][0]:g}$, "
            f"$\\rho = {PATH_VARIANTS[0][1]:g}$) and the references' ({PATH_VARIANTS[1][0]:g}, "
            f"{PATH_VARIANTS[1][1]:g}), on the three files other than gradients whose stops changed most at "
            f"{DEFAULT_TOLERANCE:g}, and on the gradient whose stop changed most: the gap per sample, the PSNR of RGB "
            "(binary64) and how far apart the two paths are, and the PSNR of Y against the original's.",
            "",
        ]
        lines += common.table(["file", "iterations", "gaps", "PSNR RGB (dB)", "apart", "PSNR Y (dB)"], rows)
    lines += [
        "## Consistency",
        "",
        "The largest excess of the canvas's coefficients, and of those of the RGB of the whole canvas converted back, "
        "in steps; and the share of the coefficients within their intervals of the 8-bit picture, over the blocks "
        "wholly within it: median (least), per cent.",
        "",
    ]
    runs = {"TV coupled": references, "TV apart": apart, "TGV coupled": tgv_coupled, "TGV apart": tgv_apart}
    lines += common.table(
        ["decoder", "files", "canvas", "RGB back", "8-bit, whole blocks"], consistency_rows(baselines, runs)
    )
    lines += [
        "## Time",
        "",
        "Median milliseconds of an iteration per million samples of the canvas (three channels), while the other "
        "runs of the stage went on in parallel.",
        "",
    ]
    lines += common.table(["runs", "sampling", "ms per 10^6 samples"], time_rows({"TV coupled (stops)": stops, **runs}))
    out.mkdir(parents=True, exist_ok=True)
    (out / "phase2-colour.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--build", type=Path, required=True, help="a build directory with -DUNROUND_WITH_PYTHON=ON")
    parser.add_argument("--data", type=Path, default=common.ROOT / "data")
    parser.add_argument("--cache", type=Path, default=common.ROOT / "data" / "phase2" / "colour")
    parser.add_argument("--out", type=Path, default=common.ROOT / "experiments" / "results")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("stages", nargs="*", default=STAGES, choices=STAGES)
    options = parser.parse_args()
    common.use_build(options.build)

    if "baselines" in options.stages:
        common.run_all(baseline, baseline_jobs(options.data, options.cache), options.workers, label="baselines")
    if "agreement" in options.stages:
        common.run_all(agreement, agreement_jobs(options.data, options.cache), options.workers, label="agreement")
    for stage in ("references", "stops", "apart"):
        if stage in options.stages:
            common.run_all(solve, run_jobs(options.data, options.cache, stage), options.workers, label=stage)
    if "paths" in options.stages:
        common.run_all(trace, path_jobs(options.data, options.cache), options.workers, label="paths")
    if "report" in options.stages:
        chosen = report(options.cache, options.out)
        print("TV coupled: " + (f"tolerance {chosen:g}" if chosen is not None else "no tolerance acceptable"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
