#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""The weight of flat blocks (docs/math.md, 4.1): TGV with the other defaults, its flat blocks weighing
a factor of the others' weights, against the factor 1, on the tuning files.

    python experiments/flat_blocks.py --rust rust/target/release [--data data] [--workers 8]
        [--cache data/flat-blocks] [--out experiments/results] [--images N] [stage ...]

The reconstructions are the reference implementation's, through the package's binding
(experiments/common.py). A variant is a factor of the weights of flat blocks and the most AC
levels not 0 of a flat block (--flat-weight, --flat-levels); the baseline is the factor 1,
which changes nothing: the model before there were flat blocks (the default is now 0.1, with a
bound of DC of 1). The stop is TGV's default when the runs were made (STOP; docs/math.md,
6.5). The variants of COMPONENT_VARIANTS give each component of a colour
file its own factor, Y's and then Cb's and Cr's; those of DC_VARIANTS take a block for flat
only where its DC level is also within a bound of those of the blocks beside it (--flat-dc).
The files are the tuning images of scripts/test_images.py as libjpeg-turbo encoded them, at
the qualities of QUALITIES: the synthetic images in grey and in colour (4:2:0), and the
photographs in colour (4:2:0), the first --images of each set's images by name (all of them
where it is not given). The stages:

  grid    Each file with the baseline and with each variant of VARIANTS and DC_VARIANTS, and
          each colour file with each of COMPONENT_VARIANTS.
  report  The paired differences of each variant from the baseline, file by file, of the
          8-bit PSNR and SSIM against the picture encoded: their median and their 10th
          percentile over the files of a set and a quality, and of the gradients alone; and the
          iterations.

A run's results are kept as JSON in --cache, and a stage skips the runs whose results are there
already. No random number is drawn: the runs are deterministic.
"""

import argparse
import dataclasses
import sys
import time
from pathlib import Path
from typing import Any, Final

import numpy as np

import common

# The factors of the weights of flat blocks, one for every component or one for each of Y, Cb
# and Cr; the most AC levels not 0 of a flat block; and the most its DC level differs from those
# beside it, or None.
type Variant = tuple[tuple[float, ...], int, int | None]

BASELINE: Final[Variant] = ((1.0,), 0, None)
VARIANTS: Final[tuple[Variant, ...]] = (
    ((0.3,), 0, None),
    ((0.1,), 0, None),
    ((0.03,), 0, None),
    ((0.1,), 1, None),
    ((0.1,), 3, None),
)
DC_VARIANTS: Final[tuple[Variant, ...]] = (((0.3,), 0, 1), ((0.3,), 0, 2), ((0.1,), 0, 1))
COMPONENT_VARIANTS: Final[tuple[Variant, ...]] = (
    ((0.3, 1.0, 1.0), 0, None),
    ((0.1, 1.0, 1.0), 0, None),
    ((1.0, 0.3, 0.3), 0, None),
)
# TGV's ratio of the steps, relaxation, tolerance of the gap per sample and most iterations.
STOP: Final = (3.0, 1.9, 5e-3, 5000)
QUALITIES: Final = (10, 30, 70)
SETS: Final = ("synthetic, grey", "synthetic, colour", "photographs, colour")
STAGES: Final = ["grid", "report"]


@dataclasses.dataclass(frozen=True, slots=True)
class Run:
    """One run: a file of a set, and the weight of its flat blocks."""

    case: common.Case
    group: str
    variant: Variant

    @property
    def name(self) -> str:
        """The name of its results."""
        weights, levels, dc = self.variant
        bound = "" if dc is None else f"-d{dc}"
        return f"f{'-'.join(f'{weight:g}' for weight in weights)}-n{levels}{bound}-{self.case.name}"


def solve(run: Run) -> dict[str, Any]:
    """Reconstructs a file by TGV with the other defaults and its flat blocks' weights: how good the result is."""
    from unround import native  # noqa: PLC0415
    from unround.settings import DataTerm, PdhgOptions, Settings  # noqa: PLC0415

    case = run.case
    data = case.jpeg.read_bytes()
    weights, levels, dc = run.variant
    terms = tuple(DataTerm(flat_weight=weight, flat_levels=levels, flat_dc=dc) for weight in weights)
    ratio, relaxation, tolerance, most = STOP
    stop = PdhgOptions(iterations=most, tolerance=tolerance, step_ratio=ratio, relaxation=relaxation)
    settings = Settings(method="tgv", data=terms[0] if len(terms) == 1 else terms, pdhg=stop)
    started = time.perf_counter()
    decoded = native.decode(data, settings)
    seconds = time.perf_counter() - started
    if decoded.result is None:
        message = f"{case.name}: the solver recorded nothing"
        raise RuntimeError(message)
    original = common.read_original(case.original)
    return {
        "image": case.image,
        "quality": case.quality,
        "sampling": case.sampling,
        "set": run.group,
        "flat_weight": list(weights),
        "flat_levels": levels,
        "flat_dc": dc,
        "iterations": decoded.result.iterations,
        "stop": decoded.result.stop.name.lower(),
        "seconds": seconds,
        "measured": common.measure(original, decoded.picture),
    }


type Sets = dict[str, list[common.Case]]


def sets_of(data: Path, images: int | None) -> Sets:
    """The three sets of tuning files at the qualities studied, the first images of each by name."""
    found = {
        "synthetic, grey": common.cases(data, "libjpeg-turbo", "tuning"),
        "synthetic, colour": common.cases(data, "libjpeg-turbo", "tuning", samplings=("420",)),
        "photographs, colour": common.cases(
            data, "libjpeg-turbo", "tuning", samplings=("420",), jpeg="jpeg-photos", originals="photo-originals"
        ),
    }
    chosen: Sets = {}
    for group, cases in found.items():
        names = sorted({case.image for case in cases})
        kept = set(names if images is None else names[:images])
        chosen[group] = [case for case in cases if case.image in kept and case.quality in QUALITIES]
    return chosen


def variants_of(group: str) -> tuple[Variant, ...]:
    """The variants of a set: those of each component too where it is in colour."""
    return (*VARIANTS, *DC_VARIANTS, *COMPONENT_VARIANTS) if "colour" in group else (*VARIANTS, *DC_VARIANTS)


def runs_of(sets: Sets) -> list[Run]:
    """Every file with the baseline and with each variant of its set."""
    return [
        Run(case, group, variant)
        for group, cases in sets.items()
        for case in cases
        for variant in (BASELINE, *variants_of(group))
    ]


def variant_label(variant: Variant) -> str:
    weights, levels, dc = variant
    bound = "" if dc is None else f", d = {dc}"
    if len(weights) == 1:
        return f"{weights[0]:g}, n = {levels}{bound}"
    return f"Y {weights[0]:g}, CbCr {weights[1]:g}, n = {levels}{bound}"


def quantiles(values: list[float]) -> str:
    """The median and the 10th percentile."""
    if not values:
        return "-"
    return f"{np.median(values):+.3f} / {np.percentile(values, 10):+.3f}"


type Results = dict[tuple[str, str, int, str, Variant], dict[str, Any]]


def results_of(cache: Path) -> Results:
    """The kept results by set, image, quality, sampling and variant."""
    found: Results = {}
    for path in sorted(cache.glob("*.json")):
        result = common.load(path)
        weights = result["flat_weight"]
        factors = tuple(float(weight) for weight in weights) if isinstance(weights, list) else (float(weights),)
        dc = result.get("flat_dc")
        variant: Variant = (factors, int(result["flat_levels"]), None if dc is None else int(dc))
        found[(result["set"], result["image"], int(result["quality"]), result["sampling"], variant)] = result
    return found


def paired_rows(results: Results, group: str, metric: str, scale: float) -> list[list[str]]:
    """For each quality, the files and each variant's differences from the baseline."""
    rows = []
    for quality in QUALITIES:
        bases = [(key, result) for key, result in results.items() if key[0] == group and key[2] == quality]
        bases = [(key, result) for key, result in bases if key[4] == BASELINE]
        cells = [str(quality), str(len(bases))]
        for variant in variants_of(group):
            differences = [
                scale * (results[(*key[:4], variant)]["measured"][metric] - base["measured"][metric])
                for key, base in bases
                if (*key[:4], variant) in results
            ]
            cells.append(quantiles(differences))
        rows.append(cells)
    return rows


def iteration_rows(results: Results, group: str) -> list[list[str]]:
    """For each quality, each variant's median ratio of iterations to the baseline's."""
    rows = []
    for quality in QUALITIES:
        bases = [
            (key, result)
            for key, result in results.items()
            if key[0] == group and key[2] == quality and key[4] == BASELINE
        ]
        cells = [str(quality)]
        for variant in variants_of(group):
            ratios = [
                results[(*key[:4], variant)]["iterations"] / max(base["iterations"], 1)
                for key, base in bases
                if (*key[:4], variant) in results
            ]
            cells.append(f"{np.median(ratios):.2f}" if ratios else "-")
        rows.append(cells)
    return rows


def gradient_rows(results: Results) -> list[list[str]]:
    """Each gradient's PSNR with the baseline, and each variant's differences of PSNR and SSIM."""
    rows = []
    for key, base in sorted(results.items(), key=lambda item: item[0][:4]):
        if key[4] != BASELINE or "gradient" not in key[1]:
            continue
        cells = [f"{key[1]} q{key[2]} {key[3]}", f"{base['measured']['psnr_8']:.2f}"]
        for variant in variants_of(key[0]):
            other = results.get((*key[:4], variant))
            if other is None:
                cells.append("-")
                continue
            psnr = other["measured"]["psnr_8"] - base["measured"]["psnr_8"]
            ssim = 1000.0 * (other["measured"]["ssim_8"] - base["measured"]["ssim_8"])
            cells.append(f"{psnr:+.3f}, {ssim:+.2f}")
        rows.append(cells)
    return rows


def report(cache: Path, out: Path) -> None:
    """The paired differences from the baseline, by set and quality, and of the gradients alone."""
    results = results_of(cache)
    lines = [
        "# The weight of flat blocks",
        "",
        "`experiments/flat_blocks.py`: TGV with the other defaults (docs/math.md, 4.1, and the stop of 6.5",
        "when the runs were made: a ratio of the steps of 3, a gap per sample of 5e-3, at most 5000",
        "iterations), its flat blocks weighing a factor of the others' weights (a block is flat where at",
        "most n of its AC levels are not 0 and, where d is given, its DC level differs from those of the",
        "blocks above, below, left and right of it by at most d; Y and CbCr give the luma and the chroma",
        "factors of their own), against the factor 1, which changes nothing (the model before there were",
        "flat blocks), file by file, on the tuning files of libjpeg-turbo: the synthetic images in grey and",
        "in colour (4:2:0) and twelve photographs in colour (4:2:0). The cells are the median and the 10th",
        "percentile, over the files, of the difference from the factor 1 of the 8-bit PSNR (dB) and of the",
        "8-bit SSIM (x 1000), and the median ratio of the iterations.",
        "",
    ]
    for group in SETS:
        labels = [variant_label(variant) for variant in variants_of(group)]
        for metric, scale, title in (("psnr_8", 1.0, "PSNR"), ("ssim_8", 1000.0, "SSIM x 1000")):
            rows = paired_rows(results, group, metric, scale)
            lines += [f"## {group}: {title}", "", *common.table(["quality", "files", *labels], rows)]
        lines += [
            f"## {group}: iterations, as a ratio of the factor 1's",
            "",
            *common.table(["quality", *labels], iteration_rows(results, group)),
        ]
    lines += [
        "## The gradients: the PSNR with the factor 1, and each variant's differences (PSNR, SSIM x 1000)",
        "",
        *common.table(
            ["file", "factor 1", *[variant_label(variant) for variant in variants_of("synthetic, colour")]],
            gradient_rows(results),
        ),
    ]
    out.mkdir(parents=True, exist_ok=True)
    (out / "flat-blocks.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rust", type=Path, required=True, help="the directory of the C API's library")
    parser.add_argument("--data", type=Path, default=common.ROOT / "data")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cache", type=Path, default=common.ROOT / "data" / "flat-blocks")
    parser.add_argument("--out", type=Path, default=common.ROOT / "experiments" / "results")
    parser.add_argument("--images", type=int, help="the first images of each set by name (all where not given)")
    parser.add_argument("stages", nargs="*", choices=STAGES, default=STAGES)
    options = parser.parse_args()
    common.use_library(options.rust)
    common.add_package()
    sets = sets_of(options.data, options.images)
    if "grid" in options.stages:
        runs = runs_of(sets)
        jobs = [(options.cache / f"{run.name}.json", run) for run in runs]
        common.run_all(solve, jobs, options.workers, label="grid")
    if "report" in options.stages:
        report(options.cache, options.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
