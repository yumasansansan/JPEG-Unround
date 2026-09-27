#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Phase 3: the reference implementation (Rust, through the package's binding) against the Python one, on every file.

    python experiments/phase3_agreement.py --build build/release --rust rust/target/release [--data data]
        [--workers 10] [--cache data/phase3/agreement] [--out experiments/results] [stage ...]

Before the Python implementation's numerics are removed, the reference implementation is held to them on every
JPEG file of the test images, as both encoders wrote them. The stages:

  synthetic  Every file of the synthetic images (scripts/test_images.py): the decoder of the centres; TV and TGV
             for ITERATIONS iterations, recorded every RECORD_EVERY, with the package's other defaults; and, for
             the greyscale files, the subgradient method for as many.
  photos     Every file of the photographs (scripts/test_images.py photos): the decoder of the centres.
  report     The tables.

What is compared is what the conformance cases compare (docs/math.md, 9): the intervals and the weights of the
data term, to the last bit; the MMSE centres, within (4|q| + 40) u Q of each other, each within (2|q| + 20) u Q
of the exact mean of its bin, and the other centres to the last bit; and the canvas, within the tolerance that
bounds the rounding of both implementations (conformance/bounds.py), from the magnitudes of the Python
implementation's run, which conformance/generate.py traces operation for operation. The exact map that the
bounds compare both with is the one with the Python implementation's centres: its own centres' error is then 0,
and the reference implementation's the difference of the two. The records' primal values are compared within
their tolerances too, each dual value against the other implementation's primal value where the gap is
certified, and TV's gaps. The subgradient method has no such bound (docs/math.md, 9.6): its distances are
reported. A run's results are kept as JSON in --cache, and a stage skips the files whose results are there
already. No random number is drawn.
"""

import argparse
import json
import os
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import numpy.typing as npt

import phase1_common as common

if TYPE_CHECKING:
    # For the annotations alone: the package and conformance/ are on the path once the stages run.
    from bounds import Tolerances
    from unround.results import History

type Array = npt.NDArray[np.float64]

ROOT: Final = Path(__file__).resolve().parent.parent
ENCODERS: Final = ("libjpeg-turbo", "mozjpeg")
ITERATIONS: Final = 10
RECORD_EVERY: Final = 5
U: Final = 2.0**-53
RUST_NAMES: Final = {"win32": "unround_capi.dll", "darwin": "libunround_capi.dylib"}
STAGES: Final = ["synthetic", "photos", "report"]
SAMPLINGS: Final = ("grey", "420", "422", "444")


def use_rust(directory: Path) -> None:
    """Points the package's binding at the reference implementation's C interface in the directory."""
    library = directory.resolve() / RUST_NAMES.get(sys.platform, "libunround_capi.so")
    os.environ.setdefault("UNROUND_LIBRARY", str(library))


def files(data: Path, directory: str) -> list[tuple[str, dict[str, Any]]]:
    """The files of every encoder's manifest under data/directory, with their entries."""
    found = []
    for encoder in ENCODERS:
        manifest = json.loads((data / directory / encoder / "manifest.json").read_text(encoding="utf-8"))
        found += [(encoder, dict(entry)) for entry in manifest["files"]]
    return found


def bits(values: Array) -> npt.NDArray[np.uint64]:
    """The bits of doubles, which tell 0 from -0 and every NaN apart."""
    return np.ascontiguousarray(values, dtype=np.float64).view(np.uint64)


def model_agreement(
    levels: Sequence[npt.NDArray[np.int16]], theirs: Sequence[Any], ours: Sequence[Any]
) -> dict[str, Any]:
    """The intervals, the weights and the centres of the two implementations' models of every component.

    The MMSE centres of DC and of the level 0 are exact, and so compared to the last bit, but for the sign of 0:
    the Python implementation's centre of the level 0 is -0 (0 times a negative number), the reference's +0,
    which the data term takes alike. The others lie within (4|q| + 40) u Q of each other, and "centres" is the
    largest fraction of that.
    """
    same = True
    worst = 0.0
    for own, their, our in zip(levels, theirs, ours, strict=True):
        for name in ("lower", "upper"):
            same = same and np.array_equal(bits(getattr(their, name)), bits(getattr(our, name)))
        same = same and np.array_equal(bits(np.broadcast_to(their.weights, (8, 8))), bits(our.weights))
        magnitude = np.abs(own.astype(np.float64))
        exact = magnitude == 0.0
        exact[:, :, 0, 0] = True
        same = same and np.array_equal(their.centres[exact], our.centres[exact])
        if np.all(exact):
            continue
        allowed = (4.0 * magnitude + 40.0) * U * np.asarray(our.steps, dtype=np.float64)
        difference = np.abs(np.asarray(our.centres) - np.asarray(their.centres))
        worst = max(worst, float(np.max(difference[~exact] / allowed[~exact])))
    return {"rational": bool(same), "centres": worst}


def solved(path: Path, method: str) -> dict[str, Any]:
    """One method on one file, by both implementations, and how far apart they lie against the bounds."""
    for place in (str(ROOT / "conformance"),):
        if place not in sys.path:
            sys.path.insert(0, place)
    import bounds  # noqa: PLC0415
    import generate  # noqa: PLC0415
    from unround import decode, frames, jpegio, native, pdhg, subgradient  # noqa: PLC0415

    data = path.read_bytes()
    image = jpegio.read(data)
    settings = decode.Settings(
        method=method,  # type: ignore[arg-type]
        pdhg=pdhg.Options(iterations=ITERATIONS, tolerance=0.0, record_every=RECORD_EVERY),
        subgradient=subgradient.Options(iterations=ITERATIONS, record_every=RECORD_EVERY),
    )
    frame = decode.frame_of(image, settings)
    ours = native.decode(data, settings)
    solution = ours.result
    levels = [component.coefficients for component in image.components]
    result = model_agreement(levels, [channel.problem for channel in frame.channels], ours.problems)
    if method == "subgradient":
        theirs = decode._subgradient(frame, settings)
        result["canvas"] = generate.norm(ours.canvas - theirs.primal.canvas)
        if solution is None:
            message = "the subgradient method recorded nothing"
            raise RuntimeError(message)
        values = np.asarray(theirs.history.primal)
        difference = np.abs(np.asarray(solution.history.primal) - values) / np.maximum(np.abs(values), 1.0)
        result["primal_relative"] = float(np.max(difference))
        return result
    weights = settings.tgv if method == "tgv" else settings.tv
    tested = tuple(
        np.abs(problem.centres - channel.problem.centres)
        for problem, channel in zip(ours.problems, frame.channels, strict=True)
    )
    model_ = generate.Model(
        frame=frame,
        gammas=frames.channel_weights(frame, weights.channel_weights),
        tested=tested,
        reference=tuple(np.zeros_like(values) for values in tested),
        exact=(),
    )
    case = generate.Case(path.stem, image.color_space, (image.height, image.width), (), 1.0, 0, settings)
    if method == "mmse":
        begin = frames.start(frame)
        run = generate.Run(
            canvas=begin.canvas,
            iterations=0,
            converged=False,
            history=[],
            start=generate.start_norms(model_, begin),
            majorants=[],
            records={},
            settings=settings,
        )
    else:
        run = generate.run_pdhg(case, model_, settings)
    tolerances = bounds.computed("\n".join(generate.bound_lines(case, model_, run)) + "\n")
    result["canvas"] = generate.norm(ours.canvas - run.canvas)
    result["canvas_tolerance"] = tolerances.canvas
    if method == "mmse":
        return result
    if solution is None:
        message = f"{method} recorded nothing"
        raise RuntimeError(message)
    return result | records_agreement(solution.history, run.history, tolerances)


def records_agreement(
    history: History, theirs: Sequence[tuple[int, float, float, float, float]], tolerances: Tolerances
) -> dict[str, Any]:
    """The records of the reference implementation (history) against the Python one's (theirs), within the bounds."""
    if [int(n) for n in history.iterations] != [n for n, *_ in theirs]:
        return {"records_iterations": False}
    primal_worst, gap_worst, brackets = 0.0, 0.0, True
    for k, (n, their_primal, their_dual, _, _) in enumerate(theirs):
        own_primal, own_dual = float(history.primal[k]), float(history.dual[k])
        primal_worst = max(primal_worst, abs(own_primal - their_primal) / tolerances.primal[n])
        bracket = tolerances.bracket.get(n)
        if bracket is not None:
            brackets = brackets and own_dual - their_primal <= bracket and their_dual - own_primal <= bracket
        gap = tolerances.gap.get(n)
        if gap is not None:
            gap_worst = max(gap_worst, abs((own_primal - own_dual) - (their_primal - their_dual)) / gap)
    return {
        "records_iterations": True,
        "primal": primal_worst,
        "brackets": brackets if tolerances.bracket else None,
        "gaps": gap_worst if tolerances.gap else None,
    }


def work(job: tuple[Path, tuple[str, ...]]) -> dict[str, Any]:
    """Every method of a job on its file; a method that fails keeps its error, and the others still run."""
    path, methods = job
    results: dict[str, Any] = {}
    for method in methods:
        try:
            results[method] = solved(path, method)
        except Exception:  # noqa: BLE001 -- kept, and reported
            results[method] = {"error": traceback.format_exc(limit=4)}
    return {"file": str(path), "methods": results}


def jobs(options: argparse.Namespace, stage: str) -> list[tuple[Path, tuple[Path, tuple[str, ...]]]]:
    directory = "jpeg" if stage == "synthetic" else "jpeg-photos"
    found = []
    for encoder, entry in files(options.data, directory):
        methods: tuple[str, ...] = ("mmse",)
        if stage == "synthetic":
            methods = ("mmse", "tv", "tgv", "subgradient") if entry["sampling"] == "grey" else ("mmse", "tv", "tgv")
        path = options.data / directory / encoder / str(entry["file"])
        kept = options.cache / stage / encoder / Path(str(entry["file"])).with_suffix(".json")
        found.append((kept, (path, methods)))
    return found


def summary(options: argparse.Namespace, stage: str) -> list[str]:
    """The table of a stage: by encoder, sampling and method, the worst of every comparison."""
    directory = "jpeg" if stage == "synthetic" else "jpeg-photos"
    rows = []
    failures = 0
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for encoder, entry in files(options.data, directory):
        kept = options.cache / stage / encoder / Path(str(entry["file"])).with_suffix(".json")
        for method, result in common.load(kept)["methods"].items():
            groups.setdefault((encoder, str(entry["sampling"]), method), []).append(result)
    for (encoder, sampling, method), results in sorted(
        groups.items(), key=lambda item: (item[0][0], SAMPLINGS.index(item[0][1]), item[0][2])
    ):
        errors = sum("error" in result for result in results)
        good = [result for result in results if "error" not in result]
        rational = sum(bool(result["rational"]) for result in good)
        centres = max((float(result["centres"]) for result in good), default=0.0)
        row = [encoder, sampling, method, str(len(results)), f"{rational}/{len(good)}", f"{centres:.2f}"]
        failures += errors + (len(good) - rational) + int(centres > 1.0)
        if method == "subgradient":
            canvas = max(float(result["canvas"]) for result in good)
            relative = max(float(result["primal_relative"]) for result in good)
            row += [f"{canvas:.1e} (no bound)", "", f"{relative:.1e} (relative, no bound)", "", ""]
        else:
            ratios = [float(result["canvas"]) / float(result["canvas_tolerance"]) for result in good]
            row += [f"{max(ratios):.1e}", f"{float(np.median(ratios)):.1e}"]
            failures += int(max(ratios) > 1.0)
            if method == "mmse":
                row += ["", "", ""]
            else:
                primal = max(float(result["primal"]) for result in good)
                iterations = all(bool(result["records_iterations"]) for result in good)
                checked = [result["brackets"] for result in good if result["brackets"] is not None]
                gaps = [float(result["gaps"]) for result in good if result["gaps"] is not None]
                row += [
                    f"{primal:.1e}",
                    f"{sum(bool(value) for value in checked)}/{len(checked)}",
                    f"{max(gaps):.1e}" if gaps else "",
                ]
                failures += int(not iterations) + int(primal > 1.0) + (len(checked) - sum(map(bool, checked)))
                failures += int(bool(gaps) and max(gaps) > 1.0)
        if errors:
            row[3] += f" ({errors} failed)"
        rows.append(row)
    header = [
        "Encoder",
        "Sampling",
        "Method",
        "Files",
        "Model to the bit",
        "Centres",
        "Canvas (worst)",
        "Canvas (median)",
        "Primal values",
        "Brackets",
        "Gaps",
    ]
    return [*common.table(header, rows), f"Comparisons beyond their bounds, and failures: {failures}.", ""]


def report(options: argparse.Namespace) -> None:
    lines = [
        "# Phase 3: the reference implementation against the Python one",
        "",
        "Written by `experiments/phase3_agreement.py`. The reference implementation (Rust), through the package's",
        "binding, against the Python implementation, on every JPEG file of the test images as libjpeg-turbo and",
        "mozjpeg wrote them: the synthetic images, and the photographs. Each entry is the worst over the files of",
        "the group, and each distance a fraction of its bound (1 is the bound), as docs/math.md, 9, derives the",
        "bounds and the conformance cases check them:",
        "",
        "- Model to the bit: the files whose intervals, weights of the data term, and centres other than the",
        "  inexact MMSE ones, are the same to the last bit in both.",
        "- Centres: the MMSE centres of AC coefficients of levels other than 0, whose distance is bounded by",
        "  $(4|q| + 40)\\, u Q$: each implementation's within $(2|q| + 20)\\, u Q$ of the exact mean of its bin.",
        "- Canvas: the Euclidean distance of the canvases after the decoder of the centres, or after",
        f"  {ITERATIONS} iterations of TV or TGV with the package's other defaults, against the bound of both",
        "  implementations' rounding; the median too.",
        f"- Primal values: those of the records (every {RECORD_EVERY} iterations), against their bounds.",
        "- Brackets: the records, where the gap is certified, whose dual value in each implementation is at",
        "  most the other's primal value, within the rounding of both.",
        "- Gaps: TV's gaps, where certified, against the bound of their difference.",
        "",
        "The subgradient method has no bound (docs/math.md, 9.6): its canvas's distance and its primal values'",
        "relative difference are those measured. Its direction is the gradient over its norm, and 0 where the",
        "gradient is 0: on the flat areas of the synthetic pictures, the first iterates of the two",
        "implementations lie within about $10^{-11}$ of each other, but some tens of thousands of pixels are",
        "flat to the last bit in one and not in the other, where the direction is 0 in one and of length 1 in",
        "the other, and the two runs part.",
        "",
        "## The synthetic images",
        "",
        *summary(options, "synthetic"),
        "## The photographs",
        "",
        "The decoder of the centres alone: the model, the Laplace scales and the centres, of natural pictures.",
        "",
        *summary(options, "photos"),
    ]
    out = options.out / "phase3-agreement.md"
    out.write_text("\n".join(lines).rstrip("\n") + "\n", encoding="utf-8", newline="\n")
    print(f"{out.as_posix()} written")


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build", type=Path, required=True, help="the build of the C layer for Python")
    parser.add_argument("--rust", type=Path, required=True, help="where cargo wrote unround_capi")
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--cache", type=Path, default=ROOT / "data" / "phase3" / "agreement")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "results")
    parser.add_argument("stages", nargs="*", default=STAGES, choices=STAGES)
    options = parser.parse_args(arguments)
    common.use_build(options.build)
    use_rust(options.rust)
    for stage in STAGES:
        if stage not in options.stages:
            continue
        if stage == "report":
            report(options)
        else:
            common.run_all(work, jobs(options, stage), options.workers, label=stage)
    return 0


if __name__ == "__main__":
    sys.exit(main())
