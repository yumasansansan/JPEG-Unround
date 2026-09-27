#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Phase 2: the weights of the model, chosen for the quantization.

    python experiments/phase2_weights.py --rust rust/target/release [--data data] [--workers 10]
        [--cache data/phase2/weights] [--out experiments/results] [stage ...]

The reconstructions are the reference implementation's, through the package's binding
(experiments/common.py). The files are the tuning images of scripts/test_images.py, as
libjpeg-turbo encoded them unless a stage says otherwise. The model is TV with
alpha = 1 unless a stage says otherwise: only mu / alpha shapes the least point
(docs/math.md, 4.1 and 4.2). Each run stops as the package's defaults did when these runs
were made (STOPS; docs/math.md, 6.5, has chosen them again since). The stages:

  grid     The synthetic greyscale images (48 files), with the data term's centres the
           MMSE ones or the middles of the intervals, the weights mu / Q^2, and mu of
           MUS; and at the former default, MMSE centres and mu = 1e-3.
  photos   The photographs (24 of BSDS500's train split, in grey, 144 files), likewise
           with mu of PHOTO_MUS.
  variants The synthetic greyscale images with the middles as centres: the weights mu / Q
           (the power 1) with mu of POWER_MUS, and the weights mu / Q^2 with DC weighted
           as AC, with mu of DC_MUS.
  tgv      TGV (alpha1 = 1, alpha0 = 2) on the synthetic greyscale images, with the
           middles and mu of TGV_MUS.
  colour   The synthetic colour images in 4:2:0 and 4:4:4 (144 files), TV with the
           channels coupled, with the centres of both kinds and mu of COLOUR_MUS, the same
           for every component.
  rule     The rule mu = RULE_SCALE Qbar^RULE_POWER (Qbar the mean step of the
           component's table), with the middles: on the files of grid, photos and colour,
           and on those of mozjpeg's synthetic greyscale images.
  chroma   The synthetic colour images of colour, with the middles and mu by the rule, each
           component's of its own table, the chroma's times CHROMA_FACTORS.
  slack    mozjpeg's synthetic greyscale images (48 files), and libjpeg-turbo's to
           compare with, with the middles and mu by the rule: the intervals widened by
           SLACKS steps on each side, at the prices per step of SLACK_COSTS.
  photochroma  The photographs in 4:2:0 and 4:4:4 (288 files), likewise with the chroma's
           mu times PHOTO_CHROMA_FACTORS.
  dc       The files of rule but mozjpeg's, by the rule with DC weighted as AC.
  proposed The defaults that the stages above propose -- the middles, mu by the rule, DC
           weighted as AC, and in colour files the chroma's mu PROPOSED_CHROMA times the
           rule's -- on the files of grid, photos, colour and photochroma, and the former
           ones (MMSE centres, mu = 1e-3, DC unweighted) on those of grid and photos, with
           TV and with TGV.
  report   The tables: the PSNR and the SSIM of the 8-bit result against the standard
           decoder's, by quality; the mu that is best for each file, and the line of
           log mu on log Qbar; and the rule, the chroma's weights, the weight of DC, the
           proposed defaults and the slack.

A run's results are kept as JSON in --cache, and a stage skips the runs whose results
are there already. No random number is drawn.
"""

import argparse
import dataclasses
import math
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt
from PIL import Image

import common
import measures

type Array = npt.NDArray[np.float64]

MUS: Final = (10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0)
PHOTO_MUS: Final = (10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0)
POWER_MUS: Final = (1.0, 3.0, 10.0, 30.0, 100.0)
DC_MUS: Final = (100.0, 300.0, 1000.0)
TGV_MUS: Final = (30.0, 100.0, 300.0, 1000.0)
COLOUR_MUS: Final = (30.0, 100.0, 300.0, 1000.0, 3000.0)
COLOUR_SAMPLINGS: Final = ("420", "444")
CENTRES: Final = ("mmse", "midpoint")
FORMER: Final = ("mmse", 1e-3)  # the default the grid is set beside
# The rule, from the line of log mu on log Qbar of the best mu of grid's files with the
# middles (the report of grid and photos).
RULE_SCALE: Final = 9.0
RULE_POWER: Final = 0.9
CHROMA_FACTORS: Final = (0.1, 0.3, 1.0, 3.0, 10.0)
PHOTO_CHROMA_FACTORS: Final = (0.1, 0.3, 1.0)
PROPOSED_CHROMA: Final = 0.3
SLACKS: Final = (0.5, 1.0, 2.0)
SLACK_COSTS: Final = (0.0, 0.3, 3.0, 30.0)
# How the runs stop, for each model: the ratio of the steps, the relaxation, the gap per
# sample and the most iterations, the package's defaults when the runs were made.
STOPS: Final = {"tv": (30.0, 1.9, 2e-4, 20000), "tgv": (10.0, 1.9, 1e-2, 10000)}
STAGES: Final = [
    "grid",
    "photos",
    "variants",
    "tgv",
    "colour",
    "rule",
    "chroma",
    "slack",
    "photochroma",
    "dc",
    "proposed",
    "report",
]


@dataclasses.dataclass(frozen=True, slots=True)
class Run:
    """One run: a file, the model, and its data term.

    mu None is the rule, RULE_SCALE Qbar^RULE_POWER of each component's own table; chroma
    is the rule's scale in the chroma of colour files, a factor of luma's (mu_chroma).
    """

    case: common.Case
    centres: str
    mu: float | None
    power: float = 2.0
    dc_weight: float = 0.0
    model: str = "tv"
    chroma: float = 1.0
    slack: float = 0.0
    slack_cost: float = 0.0

    @property
    def name(self) -> str:
        """The name of its results."""
        weight = "rule" if self.mu is None else f"mu{self.mu:g}"
        parts = [self.model, self.centres, weight, f"p{self.power:g}", f"dc{self.dc_weight:g}"]
        if self.chroma != 1.0:
            parts.append(f"k{self.chroma:g}")
        if self.slack:
            parts += [f"s{self.slack:g}", f"c{self.slack_cost:g}"]
        return "-".join([*parts, self.case.name])


def data_terms(run: Run) -> Any:  # noqa: ANN401 -- a DataTerm
    """The data term of a run, every value given: the library's defaults are not the runs'."""
    from unround.settings import DataTerm  # noqa: PLC0415

    if run.mu is not None and run.chroma != 1.0:
        message = "a factor of the chroma is the rule's, and mu is given"
        raise ValueError(message)
    return DataTerm(
        mu=run.mu,
        mu_scale=RULE_SCALE,
        mu_power=RULE_POWER,
        mu_chroma=run.chroma,
        centres=run.centres,  # type: ignore[arg-type]
        power=run.power,
        dc_weight=run.dc_weight,
        slack=run.slack,
        slack_cost=run.slack_cost,
    )


def standard(path: Path) -> npt.NDArray[np.uint8]:
    """The standard decoder's 8-bit picture: libjpeg-turbo's, as Pillow calls it."""
    with Image.open(path) as decoded:
        return np.asarray(decoded, dtype=np.uint8)


def solve(run: Run) -> dict[str, Any]:
    """Solves a file with the run's model and weights, and measures the result against the original."""
    from unround import jpegio, native  # noqa: PLC0415
    from unround.settings import PdhgOptions, Settings  # noqa: PLC0415

    case = run.case
    data = case.jpeg.read_bytes()
    image = jpegio.read(data)
    ratio, relaxation, tolerance, most = STOPS[run.model]
    stop = PdhgOptions(iterations=most, tolerance=tolerance, step_ratio=ratio, relaxation=relaxation)
    settings = Settings(method=run.model, data=data_terms(run), pdhg=stop)  # type: ignore[arg-type]
    decoded = native.decode(data, settings)
    if decoded.result is None:
        message = f"{case.name}: the solver recorded nothing"
        raise RuntimeError(message)
    original = common.read_original(case.original)
    base = standard(case.jpeg)
    eight = measures.quantize(decoded.picture)
    table = image.components[0].quant_table.astype(np.float64)
    measured = {
        "psnr": measures.psnr(original, decoded.picture),
        "psnr_8": measures.psnr(original, eight),
        "ssim_8": common.ssim(original, eight),
        "standard_psnr_8": measures.psnr(original, base),
        "standard_ssim_8": common.ssim(original, base),
    }
    if eight.ndim == 2:
        measured["psnr_b_8"] = measures.psnr_b(original, eight)
    return {
        "image": case.image,
        "quality": case.quality,
        "sampling": case.sampling,
        "encoder": case.encoder,
        "origin": origin(case),
        "model": run.model,
        "centres": run.centres,
        "mu": run.mu,
        "power": run.power,
        "dc_weight": run.dc_weight,
        "chroma": run.chroma,
        "slack": run.slack,
        "slack_cost": run.slack_cost,
        "iterations": decoded.result.iterations,
        "stop": decoded.result.stop.name.lower(),
        "gap": float(decoded.result.history.gap[-1]) / decoded.canvas.size,
        "table_mean": float(np.mean(table)),
        **measured,
    }


def chroma_of(case: common.Case) -> float:
    """The proposed chroma factor of a file: PROPOSED_CHROMA in colour, and 1 (none) in grey."""
    return 1.0 if case.sampling == "grey" else PROPOSED_CHROMA


def origin(case: common.Case) -> str:
    """Where a file's picture comes from: "photos", the photographs, or "synthetic", the synthetic images."""
    return "photos" if "jpeg-photos" in case.jpeg.parts else "synthetic"


def jobs_of(runs: Iterable[Run], cache: Path, directory: str) -> list[tuple[Path, Run]]:
    """The runs' jobs, the heaviest weights first, so that the longest start first."""
    ordered = sorted(runs, key=lambda run: -(run.mu if run.mu is not None else 1e9))
    return [(cache / directory / f"{run.name}.json", run) for run in ordered]


def grid_runs(cases: Sequence[common.Case], mus: Sequence[float]) -> list[Run]:
    """Every centres and mu, and the former default."""
    runs = [Run(case, centres, mu) for case in cases for centres in CENTRES for mu in mus]
    return runs + [Run(case, FORMER[0], FORMER[1]) for case in cases]


type Files = dict[tuple[str, int, str, str], dict[str, Any]]
type Grid = dict[tuple[str, float | None], Files]


def load(cache: Path, directory: str, **wanted: float | str | None) -> Grid:
    """A stage's results with the values given: by centres and mu, then by image, quality, sampling and encoder."""
    grid: Grid = {}
    defaults: dict[str, float | str | None] = {
        "model": "tv",
        "power": 2.0,
        "dc_weight": 0.0,
        "chroma": 1.0,
        "slack": 0.0,
        "slack_cost": 0.0,
    }
    chosen = defaults | wanted
    for path in sorted((cache / directory).glob("*.json")):
        result = common.load(path)
        if any(result[key] != value for key, value in chosen.items()):
            continue
        mu = None if result["mu"] is None else float(result["mu"])
        key = (str(result["image"]), int(result["quality"]), str(result["sampling"]), str(result["encoder"]))
        grid.setdefault((str(result["centres"]), mu), {})[key] = result
    return grid


def gains(results: Iterable[dict[str, Any]], measure: str) -> list[float]:
    return [float(result[measure]) - float(result[f"standard_{measure}"]) for result in results]


def by_quality_rows(grid: Grid, measure: str, form: str) -> list[list[str]]:
    """The median of a measure over the files, less the standard decoder's, by quality and by centres and mu."""
    keys = sorted(grid, key=lambda key: (key[0], -1.0 if key[1] is None else key[1]))
    rows = []
    for quality in common.QUALITIES:
        row = [str(quality)]
        for key in keys:
            values = gains((result for (_, q, *_), result in grid[key].items() if q == quality), measure)
            row.append(f"{np.median(values):{form}}" if values else "")
        if any(row[1:]):
            rows.append(row)
    return rows


def header_of(grid: Grid) -> list[str]:
    keys = sorted(grid, key=lambda key: (key[0], -1.0 if key[1] is None else key[1]))
    return ["quality", *(f"{kind} {'rule' if mu is None else f'{mu:g}'}" for kind, mu in keys)]


def best(grid: Grid, centres: str) -> tuple[list[list[str]], list[tuple[float, float]]]:
    """For each quality, the mu that is best for each file (by the 8-bit PSNR), and the points of mean step and mu."""
    mus = sorted(mu for kind, mu in grid if kind == centres and mu is not None and (kind, mu) != FORMER)
    files = sorted({key for mu in mus for key in grid[(centres, mu)]})
    rows, points = [], []
    for quality in common.QUALITIES:
        chosen, won, steps = [], [], []
        for key in files:
            if key[1] != quality:
                continue
            values = [(float(grid[(centres, mu)][key]["psnr_8"]), mu) for mu in mus if key in grid[(centres, mu)]]
            top, mu = max(values)
            chosen.append(mu)
            former = grid.get(FORMER, {}).get(key)
            if former is not None:
                won.append(top - float(former["psnr_8"]))
            mean = float(grid[(centres, mu)][key]["table_mean"])
            steps.append(mean)
            points.append((mean, mu))
        if not chosen:
            continue
        counts = ", ".join(f"{mu:g}: {chosen.count(mu)}" for mu in mus if chosen.count(mu))
        rows.append([str(quality), f"{np.median(steps):.1f}", counts, f"{np.median(won):+.2f}" if won else ""])
    return rows, points


def fit(points: list[tuple[float, float]]) -> tuple[float, float]:
    """The line of log mu on log mean step, by least squares: its slope, and mu at a mean step of 1."""
    x = np.log(np.array([point[0] for point in points]))
    y = np.log(np.array([point[1] for point in points]))
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), math.exp(float(intercept))


def sections(grid: Grid, title: str) -> list[str]:
    """The tables of one grid."""
    lines = [
        f"## {title}: the 8-bit PSNR against the standard decoder",
        "",
        "Median over the files, by quality, of the 8-bit PSNR less the standard decoder's (dB).",
        "",
        *common.table(header_of(grid), by_quality_rows(grid, "psnr_8", "+.2f")),
        f"## {title}: the SSIM against the standard decoder",
        "",
        "Likewise, of the SSIM of the 8-bit result.",
        "",
        *common.table(header_of(grid), by_quality_rows(grid, "ssim_8", "+.4f")),
    ]
    for centres in CENTRES:
        rows, points = best(grid, centres)
        if not rows or len(points) < 2:
            continue
        slope, scale = fit(points)
        lines += [
            f"## {title}: the best mu, with the {centres} centres",
            "",
            "For each quality: the median mean step of the tables, how many files each mu is best for (by the 8-bit "
            "PSNR), and the median gain of the best over the former default (MMSE centres, $\\mu = 10^{-3}$).",
            "",
            *common.table(["quality", "mean step", "best mu: files", "gain (dB)"], rows),
            f"The line of $\\log \\mu$ on $\\log \\bar Q$ over the files: slope {slope:.2f}, and "
            f"$\\mu = {scale:.3g}$ at a mean step of 1.",
            "",
        ]
    return lines


def rule_rows(rule: Grid, grid: Grid) -> list[list[str]]:
    """By quality: the rule's median gain over the standard decoder, and what it gives up to each file's best mu."""
    by_file = rule.get(("midpoint", None), {})
    mus = sorted(mu for kind, mu in grid if kind == "midpoint" and mu is not None)
    rows = []
    for quality in common.QUALITIES:
        files = [key for key in by_file if key[1] == quality]
        if not files:
            continue
        won = gains((by_file[key] for key in files), "psnr_8")
        given_up = []
        for key in files:
            values = [
                float(grid[("midpoint", mu)][key]["psnr_8"]) for mu in mus if key in grid.get(("midpoint", mu), {})
            ]
            if values:
                given_up.append(max(values) - float(by_file[key]["psnr_8"]))
        ssim = gains((by_file[key] for key in files), "ssim_8")
        rows.append(
            [
                str(quality),
                str(len(files)),
                f"{np.median(won):+.2f}",
                f"{np.median(ssim):+.4f}",
                f"{np.median(given_up):+.2f} / {max(given_up):+.2f}" if given_up else "",
            ]
        )
    return rows


def rule_section(cache: Path) -> list[str]:
    lines = [
        "## The rule",
        "",
        f"$\\mu = {RULE_SCALE:g}\\, \\bar Q^{{{RULE_POWER:g}}}$ with the middles as centres, by quality: the files, "
        "the median gain of the 8-bit PSNR and SSIM over the standard decoder, and what the rule gives up against each "
        "file's best $\\mu$ of the grid (median / largest, dB), where there is a grid.",
        "",
    ]
    for title, grid_directory, source, encoder, sampling in (
        ("Synthetic images, grey", "grid", "synthetic", "libjpeg-turbo", "grey"),
        ("Synthetic images, grey, mozjpeg", "", "synthetic", "mozjpeg", "grey"),
        ("Photographs, grey", "photos", "photos", "libjpeg-turbo", "grey"),
        ("Synthetic images, 4:2:0", "colour", "synthetic", "libjpeg-turbo", "420"),
        ("Synthetic images, 4:4:4", "colour", "synthetic", "libjpeg-turbo", "444"),
    ):
        rule = only(load(cache, "rule"), source, encoder, sampling)
        if not rule:
            continue
        grid = only(load(cache, grid_directory), source, encoder, sampling) if grid_directory else {}
        header = ["quality", "files", "PSNR", "SSIM", "given up"]
        lines += [f"### {title}", "", *common.table(header, rule_rows(rule, grid))]
    return lines


type Curves = list[tuple[Array, Array, float]]


def curves(grid: Grid) -> Curves:
    """For each file of a grid with the middles: log mu, the 8-bit PSNR less the standard decoder's, the mean step."""
    mus = sorted(mu for kind, mu in grid if kind == "midpoint" and mu is not None)
    found: Curves = []
    for key in sorted({key for mu in mus for key in grid[("midpoint", mu)]}):
        points = [(mu, grid[("midpoint", mu)][key]) for mu in mus if key in grid[("midpoint", mu)]]
        logs = np.log(np.array([mu for mu, _ in points]))
        won = np.array(gains((result for _, result in points), "psnr_8"))
        found.append((logs, won, float(points[0][1]["table_mean"])))
    return found


def predicted(files: Curves, scale: float, power: float) -> Array:
    """What a rule gains on each file, read off the file's curve between the mu of the grid (flat beyond them)."""
    return np.array([np.interp(math.log(scale) + power * math.log(step), logs, won) for logs, won, step in files])


def rule_fit_rows(sets: Sequence[tuple[str, Curves]]) -> list[list[str]]:
    """For each set of files, and for them all: the mean gain of the rule and its largest loss to the file's best
    point of the grid, and the rule of scale RULE_SCALES and power RULE_POWERS that gains most on the mean."""
    scales = np.geomspace(1.0, 300.0, 50)
    powers = np.linspace(0.3, 1.2, 37)
    rows = []
    together = [(title, files) for title, files in sets if files]
    for title, files in [*together, ("every set, each a weight of 1", [])]:
        members = [files] if files else [chosen for _, chosen in together]

        def score(scale: float, power: float, members: list[Curves] = members) -> float:
            return float(np.mean([np.mean(predicted(chosen, scale, power)) for chosen in members]))

        best_scale, best_power = max(((s, p) for s in scales for p in powers), key=lambda pair: score(*pair))
        losses = np.concatenate(
            [
                np.array([won.max() for _, won, _ in chosen]) - predicted(chosen, RULE_SCALE, RULE_POWER)
                for chosen in members
            ]
        )
        rows.append(
            [
                title,
                str(sum(len(chosen) for chosen in members)),
                f"{score(RULE_SCALE, RULE_POWER):+.3f}",
                f"{np.median(losses):.3f} / {losses.max():.3f}",
                f"{best_scale:.3g}, {best_power:.3g}",
                f"{score(best_scale, best_power):+.3f}",
            ]
        )
    return rows


def rule_fit_section(cache: Path) -> list[str]:
    sets = [
        ("synthetic, grey", curves(only(load(cache, "grid"), "synthetic", "libjpeg-turbo", "grey"))),
        ("photographs, grey", curves(only(load(cache, "photos"), "photos", "libjpeg-turbo", "grey"))),
        ("synthetic, 4:2:0", curves(only(load(cache, "colour"), "synthetic", "libjpeg-turbo", "420"))),
        ("synthetic, 4:4:4", curves(only(load(cache, "colour"), "synthetic", "libjpeg-turbo", "444"))),
    ]
    header = ["files", "count", f"mean gain at {RULE_SCALE:g}, {RULE_POWER:g}", "loss (median / largest)"]
    header += ["best scale, power", "its mean gain"]
    return [
        "## The rule, read off the grids",
        "",
        "Each file's gain of the 8-bit PSNR over the standard decoder, with the middles, as a function of "
        "$\\log \\mu$, piecewise linear between the $\\mu$ of its grid and flat beyond them. A rule "
        "$\\mu = s \\bar Q^r$ gains what the curves give at its $\\mu$: its mean over the files, and its loss to each "
        "file's best point of the grid (median / largest, dB). The best rule is that of the largest mean, over "
        "$s$ from 1 to 300 and $r$ from 0.3 to 1.2; for every set together, the mean of the sets' means.",
        "",
        *common.table(header, rule_fit_rows(sets)),
    ]


def only(grid: Grid, source: str, encoder: str, sampling: str) -> Grid:
    """The grid's results of the pictures of one origin, one encoder and one sampling."""
    kept: Grid = {}
    for key, files in grid.items():
        chosen = {
            name: result
            for name, result in files.items()
            if result["origin"] == source and result["encoder"] == encoder and name[2] == sampling
        }
        if chosen:
            kept[key] = chosen
    return kept


def chroma_rows(cache: Path, source: str, directory: str, factors: Sequence[float]) -> list[list[str]]:
    """By sampling and quality, the median gain of the 8-bit PSNR over the standard decoder, for each chroma factor.

    Of the synthetic images, the factor 1 is the rule's own run.
    """
    rows = []
    for sampling in COLOUR_SAMPLINGS:
        grids = {
            factor: only(
                load(cache, "rule" if factor == 1.0 and source == "synthetic" else directory, chroma=factor),
                source,
                "libjpeg-turbo",
                sampling,
            )
            for factor in factors
        }
        for quality in common.QUALITIES:
            row = [sampling, str(quality)]
            for factor in factors:
                results = [
                    result for files in grids[factor].values() for key, result in files.items() if key[1] == quality
                ]
                row.append(f"{np.median(gains(results, 'psnr_8')):+.2f}" if results else "")
            if any(row[2:]):
                rows.append(row)
    return rows


def chroma_section(cache: Path) -> list[str]:
    lines = [
        "## The chroma's weight",
        "",
        "With the middles and $\\mu$ by the rule, each component's of its own table, the chroma's times a factor: by "
        "sampling, quality and factor, the median gain of the 8-bit PSNR over the standard decoder (dB).",
        "",
    ]
    for title, source, directory, factors in (
        ("Synthetic images", "synthetic", "chroma", CHROMA_FACTORS),
        ("Photographs", "photos", "photochroma", PHOTO_CHROMA_FACTORS),
    ):
        rows = chroma_rows(cache, source, directory, factors)
        if rows:
            header = ["sampling", "quality", *(f"{factor:g}" for factor in factors)]
            lines += [f"### {title}", "", *common.table(header, rows)]
    return lines


def dc_section(cache: Path) -> list[str]:
    """By set and quality: the median gains and iterations by the rule, with no weight on DC and with DC's as AC's."""
    lines = [
        "## The weight of DC",
        "",
        "With the middles and $\\mu$ by the rule, DC unweighted (0) and weighted as AC (1): by quality, the median "
        "gain of the 8-bit PSNR over the standard decoder (dB), and the median iterations.",
        "",
    ]
    for title, source, sampling in (
        ("Synthetic images, grey", "synthetic", "grey"),
        ("Photographs, grey", "photos", "grey"),
        ("Synthetic images, 4:2:0", "synthetic", "420"),
        ("Synthetic images, 4:4:4", "synthetic", "444"),
    ):
        without = only(load(cache, "rule"), source, "libjpeg-turbo", sampling).get(("midpoint", None), {})
        weighted = only(load(cache, "dc", dc_weight=1.0), source, "libjpeg-turbo", sampling).get(("midpoint", None), {})
        rows = []
        for quality in common.QUALITIES:
            row = [str(quality)]
            for files in (without, weighted):
                results = [result for key, result in files.items() if key[1] == quality]
                row.append(f"{np.median(gains(results, 'psnr_8')):+.2f}" if results else "")
            for files in (without, weighted):
                results = [result for key, result in files.items() if key[1] == quality]
                row.append(f"{np.median([int(result['iterations']) for result in results]):.0f}" if results else "")
            if any(row[1:]):
                rows.append(row)
        if weighted:
            header = ["quality", "PSNR, DC 0", "PSNR, DC 1", "iterations, DC 0", "iterations, DC 1"]
            lines += [f"### {title}", "", *common.table(header, rows)]
    return lines


def proposed_section(cache: Path) -> list[str]:
    """By set, model and quality: the median gains of the proposed defaults and of the former ones, and iterations."""
    lines = [
        "## The proposed defaults",
        "",
        "The middles, $\\mu$ by the rule, DC weighted as AC, and in colour files the chroma's $\\mu$ "
        f"{PROPOSED_CHROMA:g} times the rule's; of the greyscale files, against the former defaults (MMSE centres, "
        "$\\mu = 10^{-3}$, DC unweighted). By quality, the median gain of the 8-bit PSNR and of the SSIM over the "
        "standard decoder, and the median iterations, of each.",
        "",
    ]
    for title, source, sampling in (
        ("Synthetic images, grey", "synthetic", "grey"),
        ("Photographs, grey", "photos", "grey"),
        ("Synthetic images, 4:2:0", "synthetic", "420"),
        ("Synthetic images, 4:4:4", "synthetic", "444"),
        ("Photographs, 4:2:0", "photos", "420"),
        ("Photographs, 4:4:4", "photos", "444"),
    ):
        for model in ("tv", "tgv"):
            chroma = PROPOSED_CHROMA if sampling != "grey" else 1.0
            proposed = only(
                load(cache, "proposed", model=model, dc_weight=1.0, chroma=chroma), source, "libjpeg-turbo", sampling
            ).get(("midpoint", None), {})
            former = {}
            if sampling == "grey":
                former = only(load(cache, "proposed", model=model), source, "libjpeg-turbo", sampling).get(FORMER, {})
            sides = (proposed, former) if former else (proposed,)
            rows = []
            for quality in common.QUALITIES:
                chosen = [[result for key, result in files.items() if key[1] == quality] for files in sides]
                if not all(chosen):
                    continue
                row = [str(quality)]
                row += [f"{np.median(gains(results, 'psnr_8')):+.2f}" for results in chosen]
                row += [f"{np.median(gains(results, 'ssim_8')):+.4f}" for results in chosen]
                row += [f"{np.median([int(result['iterations']) for result in results]):.0f}" for results in chosen]
                rows.append(row)
            if rows:
                header = ["quality", "PSNR", "former", "SSIM", "former", "iterations", "former"]
                if not former:
                    header = ["quality", "PSNR", "SSIM", "iterations"]
                lines += [f"### {title}, {model.upper()}", "", *common.table(header, rows)]
    return lines


def slack_section(cache: Path) -> list[str]:
    lines = [
        "## The slack",
        "",
        "With the middles and $\\mu$ by the rule: by quality, the median gain of the 8-bit PSNR over the standard "
        "decoder (dB), without slack and with each slack (steps on each side) and price per step.",
        "",
    ]
    for encoder in ("mozjpeg", "libjpeg-turbo"):
        without = only(load(cache, "rule"), "synthetic", encoder, "grey").get(("midpoint", None), {})
        grids = {
            (slack, cost): only(load(cache, "slack", slack=slack, slack_cost=cost), "synthetic", encoder, "grey").get(
                ("midpoint", None), {}
            )
            for slack in SLACKS
            for cost in SLACK_COSTS
        }
        header = ["quality", "none", *(f"{slack:g} at {cost:g}" for slack, cost in grids)]
        rows = []
        for quality in common.QUALITIES:
            row = [str(quality)]
            for files in (without, *grids.values()):
                results = [result for key, result in files.items() if key[1] == quality]
                row.append(f"{np.median(gains(results, 'psnr_8')):+.2f}" if results else "")
            if any(row[1:]):
                rows.append(row)
        if rows:
            lines += [f"### {encoder}", "", *common.table(header, rows)]
    return lines


def report(cache: Path, out: Path) -> None:
    """Writes the tables."""
    from importlib import metadata  # noqa: PLC0415

    from unround import native  # noqa: PLC0415

    lines = [
        "<!-- Written by experiments/phase2_weights.py; do not edit by hand. -->",
        "",
        "# Phase 2: the weights of the model",
        "",
        "- Files: the tuning images, as libjpeg-turbo encoded them unless a section says otherwise: the synthetic "
        "greyscale images (48 files), the photographs in grey (24 of BSDS500's train split, 144 files), the "
        "synthetic colour images in 4:2:0 and 4:4:4 (144 files), and the photographs in 4:2:0 and 4:4:4 (288 files).",
        "- TV with $\\alpha = 1$ unless a section says otherwise, the weights $\\mu / Q^2$ and no weight on DC, "
        "stopped as the package's defaults did then (TV: $\\tau/\\sigma = 30$, $\\rho = 1.9$, a gap per sample of "
        "$2 \\cdot 10^{-4}$, at most 20000 iterations; TGV: 10, 1.9, $10^{-2}$, 10000); the data term's centres "
        "the MMSE ones or the middles of the intervals (docs/math.md, 2.2 and 4.1).",
        "- The measures are those of the 8-bit result, rounded and clamped as the standard decoder's (libjpeg-turbo, "
        "through Pillow), against the original; of colour files, of RGB, the SSIM the mean over R, G and B.",
        f"- The reference implementation, {native.version()}; NumPy {np.__version__}, scikit-image "
        f"{metadata.version('scikit-image')}.",
        "",
    ]
    for title, directory, sampling in (
        ("Synthetic images", "grid", "grey"),
        ("Photographs", "photos", "grey"),
        ("Synthetic images, 4:2:0", "colour", "420"),
        ("Synthetic images, 4:4:4", "colour", "444"),
    ):
        grid = only(
            load(cache, directory), "photos" if directory == "photos" else "synthetic", "libjpeg-turbo", sampling
        )
        if grid:
            lines += sections(grid, title)
    for title, variant, directory, model in (
        ("Synthetic images, the weights mu / Q", {"power": 1.0}, "variants", "tv"),
        ("Synthetic images, DC weighted as AC", {"dc_weight": 1.0}, "variants", "tv"),
        ("Synthetic images, TGV", {}, "tgv", "tgv"),
    ):
        grid = load(cache, directory, model=model, **variant)
        if grid:
            lines += sections(grid, title)
    lines += rule_section(cache)
    lines += rule_fit_section(cache)
    lines += chroma_section(cache)
    lines += dc_section(cache)
    lines += proposed_section(cache)
    lines += slack_section(cache)
    out.mkdir(parents=True, exist_ok=True)
    (out / "phase2-weights.md").write_text("\n".join(lines).rstrip("\n") + "\n", encoding="utf-8", newline="\n")
    print(f"{(out / 'phase2-weights.md').as_posix()} written")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rust", type=Path, required=True, help="where cargo wrote unround_capi")
    parser.add_argument("--data", type=Path, default=common.ROOT / "data")
    parser.add_argument("--cache", type=Path, default=common.ROOT / "data" / "phase2" / "weights")
    parser.add_argument("--out", type=Path, default=common.ROOT / "experiments" / "results")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("stages", nargs="*", default=STAGES, choices=STAGES)
    options = parser.parse_args()
    common.use_library(options.rust)
    data, cache = options.data, options.cache
    grey = common.cases(data, "libjpeg-turbo", "tuning")
    photos = common.cases(data, "libjpeg-turbo", "tuning", jpeg="jpeg-photos", originals="photo-originals")
    colour = common.cases(data, "libjpeg-turbo", "tuning", samplings=COLOUR_SAMPLINGS)
    moz = common.cases(data, "mozjpeg", "tuning")
    colour_photos = common.cases(
        data, "libjpeg-turbo", "tuning", samplings=COLOUR_SAMPLINGS, jpeg="jpeg-photos", originals="photo-originals"
    )
    stages: dict[str, list[Run]] = {
        "grid": grid_runs(grey, MUS),
        "photos": grid_runs(photos, PHOTO_MUS),
        "variants": [Run(case, "midpoint", mu, power=1.0) for case in grey for mu in POWER_MUS]
        + [Run(case, "midpoint", mu, dc_weight=1.0) for case in grey for mu in DC_MUS],
        "tgv": [Run(case, "midpoint", mu, model="tgv") for case in grey for mu in TGV_MUS],
        "colour": [Run(case, centres, mu) for case in colour for centres in CENTRES for mu in COLOUR_MUS],
        "rule": [Run(case, "midpoint", None) for case in (*grey, *photos, *colour, *moz)],
        "chroma": [
            Run(case, "midpoint", None, chroma=factor) for case in colour for factor in CHROMA_FACTORS if factor != 1.0
        ],
        "photochroma": [
            Run(case, "midpoint", None, chroma=factor) for case in colour_photos for factor in PHOTO_CHROMA_FACTORS
        ],
        "dc": [Run(case, "midpoint", None, dc_weight=1.0) for case in (*grey, *photos, *colour)],
        "proposed": [
            Run(case, "midpoint", None, dc_weight=1.0, model=model, chroma=chroma_of(case))
            for case in (*grey, *photos, *colour, *colour_photos)
            for model in ("tv", "tgv")
        ]
        + [Run(case, FORMER[0], FORMER[1], model=model) for case in (*grey, *photos) for model in ("tv", "tgv")],
        "slack": [
            Run(case, "midpoint", None, slack=slack, slack_cost=cost)
            for case in (*moz, *grey)
            for slack in SLACKS
            for cost in SLACK_COSTS
        ],
    }
    for stage in STAGES:
        if stage not in options.stages:
            continue
        if stage == "report":
            report(cache, options.out)
        else:
            common.run_all(solve, jobs_of(stages[stage], cache, stage), options.workers, label=stage)
    return 0


if __name__ == "__main__":
    sys.exit(main())
