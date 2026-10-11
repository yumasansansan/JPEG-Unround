#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""The reference implementation against the tools it is compared with -- jpeg2png, jpegqs and Knusperli -- on
the test files.

    python experiments/comparison.py --rust rust/target/release [--tools third_party/<system>-<processor>/bin]
        [--data data] [--workers 6] [--cache data/comparison] [--out experiments/results]
        [--encoders libjpeg-turbo ...] [--samplings grey 420 ...] [--qualities 10 30 ...]
        [--sets synthetic photographs] [--images N] [stage ...]

The files are the test files of scripts/test_images.py, none of the tuning ones: the synthetic images and the
photographs, as each encoder encoded them, in the samplings and qualities given. Each file is decoded by each
method:

  standard      libjpeg-turbo's decoder, as Pillow calls it (the accurate integer inverse DCT, fancy upsampling)
  unround       the reference implementation with its defaults (TGV, docs/cli.md), through the package's
                binding: its binary64 picture, and the 8-bit samples that its command line writes by default,
                dithered (docs/math.md, 10)
  jpeg2png      its defaults (50 iterations), and 500 iterations (jpeg2png-500)
  jpegqs        its defaults (quality 3), and quality 6 (jpegqs-6); its JPEG file decoded as the standard
                decoder decodes
  knusperli     as it is

Each picture is measured against the original it was encoded from -- the PSNR and SSIM of its 8-bit samples,
their PSNR-B in grey, and the PSNR of unround's binary64 picture -- and against the file's intervals: the share
of the coefficients that a file encoding it would hold (docs/math.md, 12; unround.project) that lie within
them, and the largest excess, in steps. A method that refuses a file (jpeg2png reads no greyscale file, nor a
colour file of 4:2:0 of an odd size) is counted apart. Each tool runs with one thread, the files in parallel:
the seconds are those of a run beside the others, of the decoding alone for unround (in the process) and of the
program for the others (its start included), and say little more than the order of each method's.

  run     Each file by each method. A file's results are kept as JSON in --cache, and the files whose results
          are there already are skipped.
  report  For each set, encoder and sampling, the means over the files of each quality; the files on which
          each method is the best of those that decoded them, by 8-bit PSNR and by SSIM; the share of the
          coefficients within their intervals; and the medians of the seconds. Into --out, comparison.md.

Nothing random is drawn: the runs are deterministic (but for the seconds).
"""

import argparse
import dataclasses
import io
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt
from PIL import Image

import common
import measures

METHODS: Final = ("standard", "unround", "jpeg2png", "jpeg2png-500", "jpegqs", "jpegqs-6", "knusperli")
# The tools' programs and arguments besides their files: one thread each.
TOOLS: Final = {
    "jpeg2png": ("jpeg2png", ("-q", "-t", "1")),
    "jpeg2png-500": ("jpeg2png", ("-q", "-t", "1", "-i", "500")),
    "jpegqs": ("jpegqs", ("-t", "1", "-i", "0")),
    "jpegqs-6": ("jpegqs", ("-t", "1", "-i", "0", "-q", "6")),
    "knusperli": ("knusperli", ()),
}
# The directories under --data of each set's JPEG files and originals.
SETS: Final = {"synthetic": ("jpeg", "synthetic"), "photographs": ("jpeg-photos", "photo-originals")}
STAGES: Final = ["run", "report"]

type Picture = npt.NDArray[np.uint8]


@dataclasses.dataclass(frozen=True, slots=True)
class Job:
    """A file to decode by every method, of a set, with the directory of the tools."""

    case: common.Case
    group: str
    tools: Path

    @property
    def name(self) -> str:
        """The set and the file, as the name of its results."""
        return f"{self.group}-{self.case.name}"


def program(tools: Path, name: str) -> Path:
    """A tool's program in the directory."""
    windows = tools / f"{name}.exe"
    return windows if windows.exists() else tools / name


def eight_bits(picture: npt.NDArray[np.float64]) -> Picture:
    """The 8-bit samples that unround's command line writes of a picture by default, dithered, as PNM holds them."""
    from unround import native  # noqa: PLC0415 -- once the worker has the package on its path

    data = native.pnm_bytes(picture)
    with Image.open(io.BytesIO(data)) as image:
        return np.asarray(image, dtype=np.uint8)


def consistency(picture: npt.NDArray[np.generic], data: bytes) -> dict[str, float]:
    """The share of the coefficients of a file encoding the picture that lie within the file's intervals, and the
    largest excess, in steps (docs/math.md, 12)."""
    from unround import native  # noqa: PLC0415

    projected = native.project(np.asarray(picture, dtype=np.float64), data)
    if projected.excess is None:
        message = "the projection gave no excess"
        raise RuntimeError(message)
    excess = np.concatenate([values.ravel() for values in projected.excess])
    return {"inside": float(np.count_nonzero(excess == 0.0)) / excess.size, "largest": float(excess.max())}


def measured(original: Picture, eight: Picture, data: bytes) -> dict[str, float]:
    """The measures of 8-bit samples: against the original, and against the file's intervals."""
    result = {"psnr_8": measures.psnr(original, eight), "ssim_8": common.ssim(original, eight)}
    if eight.ndim == 2:
        result["psnr_b_8"] = measures.psnr_b(original, eight)
    return result | consistency(eight, data)


def tool(job: Job, method: str, scratch: Path) -> tuple[Picture | None, float, str]:
    """A tool's picture of the file, the seconds of its run, and what it said where it refused the file."""
    name, arguments = TOOLS[method]
    jpeg = str(job.case.jpeg)
    if name == "jpeg2png":
        output = scratch / f"{method}.png"
        command = [str(program(job.tools, name)), *arguments, jpeg, "-o", str(output)]
    elif name == "jpegqs":
        output = scratch / f"{method}.jpg"
        command = [str(program(job.tools, name)), *arguments, jpeg, str(output)]
    else:
        output = scratch / f"{method}.png"
        command = [str(program(job.tools, name)), *arguments, jpeg, str(output)]
    started = time.perf_counter()
    done = subprocess.run(command, capture_output=True, text=True, check=False)
    seconds = time.perf_counter() - started
    if done.returncode != 0 or not output.exists():
        said = (done.stderr or done.stdout).strip().splitlines()
        return None, seconds, said[-1] if said else f"status {done.returncode}"
    with Image.open(output) as image:
        return np.asarray(image, dtype=np.uint8), seconds, ""


def compare(job: Job) -> dict[str, Any]:
    """A file decoded by every method, and each picture measured."""
    from unround import native  # noqa: PLC0415
    from unround.settings import PdhgOptions, Settings  # noqa: PLC0415

    case = job.case
    data = case.jpeg.read_bytes()
    original = common.read_original(case.original)
    results: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    with Image.open(case.jpeg) as image:
        standard = np.asarray(image, dtype=np.uint8)
    results["standard"] = measured(original, standard, data) | {"seconds": time.perf_counter() - started}
    started = time.perf_counter()
    # One thread, as each tool has: the files go in parallel, one to a process.
    decoded = native.decode(data, Settings(pdhg=PdhgOptions(threads=1)))
    seconds = time.perf_counter() - started
    if decoded.result is None:
        message = f"{case.name}: the solver recorded nothing"
        raise RuntimeError(message)
    eight = eight_bits(decoded.picture)
    within = consistency(decoded.picture, data)
    results["unround"] = measured(original, eight, data) | {
        "seconds": seconds,
        "psnr": measures.psnr(original, decoded.picture),
        "inside_64": within["inside"],
        "largest_64": within["largest"],
        "iterations": decoded.result.iterations,
        "stop": decoded.result.stop.name.lower(),
    }
    with tempfile.TemporaryDirectory(prefix="unround-comparison-") as directory:
        for method in TOOLS:
            picture, seconds, said = tool(job, method, Path(directory))
            if picture is None:
                results[method] = {"refused": said, "seconds": seconds}
            elif picture.shape != original.shape:
                results[method] = {"refused": f"a picture of {picture.shape}, not {original.shape}", "seconds": seconds}
            else:
                results[method] = measured(original, picture, data) | {"seconds": seconds}
    return {
        "set": job.group,
        "encoder": case.encoder,
        "image": case.image,
        "quality": case.quality,
        "sampling": case.sampling,
        "height": int(original.shape[0]),
        "width": int(original.shape[1]),
        "methods": results,
    }


def jobs_of(arguments: argparse.Namespace) -> list[tuple[Path, Job]]:
    """The files of the sets, encoders, samplings and qualities asked for, and where their results go."""
    found = []
    for group in arguments.sets:
        jpeg, originals = SETS[group]
        for encoder in arguments.encoders:
            cases = common.cases(
                arguments.data, encoder, "test", samplings=arguments.samplings, jpeg=jpeg, originals=originals
            )
            images = sorted({case.image for case in cases})
            if arguments.images is not None:
                images = images[: arguments.images]
            for case in cases:
                if case.quality in arguments.qualities and case.image in images:
                    job = Job(case=case, group=group, tools=arguments.tools)
                    found.append((arguments.cache / f"{job.name}.json", job))
    return found


def mean(values: Sequence[float]) -> float:
    """The mean, summed in binary64."""
    return float(np.mean(values))


def cell(values: Sequence[float], count: int, digits: int) -> str:
    """The mean of a column's values, and how many of the files they are of where not all."""
    if not values:
        return "--"
    text = f"{mean(values):.{digits}f}"
    return text if len(values) == count else f"{text} ({len(values)}/{count})"


def quality_table(results: Sequence[dict[str, Any]], key: str, digits: int, methods: Sequence[str]) -> list[str]:
    """The means of a measure over the files of each quality, method by method."""
    rows = []
    for quality in sorted({int(result["quality"]) for result in results}):
        files = [result for result in results if result["quality"] == quality]
        row = [str(quality)]
        for method in methods:
            values = [float(file["methods"][method][key]) for file in files if key in file["methods"][method]]
            row.append(cell(values, len(files), digits))
        rows.append(row)
    return common.table(["quality", *methods], rows)


def best_counts(results: Sequence[dict[str, Any]], key: str, methods: Sequence[str]) -> list[str]:
    """For each method, on how many of the files it decoded it is the best of those that decoded them."""
    wins = dict.fromkeys(methods, 0)
    decoded = dict.fromkeys(methods, 0)
    for result in results:
        values = {method: result["methods"][method].get(key) for method in methods}
        taken = {method: value for method, value in values.items() if value is not None}
        top = max(taken.values())
        for method, value in taken.items():
            decoded[method] += 1
            wins[method] += value == top
    return common.table(
        ["method", "best", "files"], [[method, str(wins[method]), str(decoded[method])] for method in methods]
    )


def report(cache: Path, out: Path) -> None:
    """The tables of the results kept, into comparison.md."""
    results = [common.load(path) for path in sorted(cache.glob("*.json"))]
    if not results:
        print(f"no results in {cache}")
        return
    shown = ["standard", "unround", "jpeg2png", "jpeg2png-500", "jpegqs", "jpegqs-6", "knusperli"]
    lines = [
        "<!-- Written by experiments/comparison.py; its docstring says what is measured, and how. -->",
        "",
        "# The reference implementation and the tools it is compared with",
        "",
        "- Methods: the standard decoder (libjpeg-turbo's, as Pillow calls it); unround with its defaults (TGV), "
        "its 8-bit samples as its command line writes them (dithered); jpeg2png with its defaults (50 iterations) "
        "and with 500; jpegqs with its defaults (quality 3) and with quality 6; Knusperli.",
        "- Measures, against the original: the PSNR and SSIM (Gaussian weights of 1.5) of the 8-bit samples, "
        "and their PSNR-B in grey; the means are over the files. Against the file's intervals: the share of "
        "the coefficients of a file encoding the picture that lie within them.",
        f"- Files: {len(results)} test files. A cell of fewer files than the row's says how many it is of: "
        "jpeg2png reads no greyscale file, nor a 4:2:0 file of an odd size.",
        "",
    ]
    groups = sorted({(str(r["set"]), str(r["encoder"]), str(r["sampling"])) for r in results})
    for group, encoder, sampling in groups:
        chosen = [r for r in results if (r["set"], r["encoder"], r["sampling"]) == (group, encoder, sampling)]
        images = len({r["image"] for r in chosen})
        lines += [f"## {group}, {encoder}, {sampling}", "", f"{len(chosen)} files of {images} images.", ""]
        lines += ["**PSNR of the 8-bit samples, dB**", "", *quality_table(chosen, "psnr_8", 3, shown)]
        lines += ["**SSIM of the 8-bit samples**", "", *quality_table(chosen, "ssim_8", 5, shown)]
        if sampling == "grey":
            lines += ["**PSNR-B of the 8-bit samples, dB**", "", *quality_table(chosen, "psnr_b_8", 3, shown)]
        lines += [
            "**Share of the coefficients within their intervals**",
            "",
            *quality_table(chosen, "inside", 4, shown),
        ]
        lines += ["**The best by 8-bit PSNR**", "", *best_counts(chosen, "psnr_8", shown)]
        lines += ["**The best by SSIM**", "", *best_counts(chosen, "ssim_8", shown)]
    lines += ["## Seconds", "", "The medians over the files that each method decoded, in a run beside the others.", ""]
    rows = []
    for group, encoder, sampling in groups:
        chosen = [r for r in results if (r["set"], r["encoder"], r["sampling"]) == (group, encoder, sampling)]
        row = [f"{group}, {encoder}, {sampling}"]
        for method in shown:
            seconds = [float(r["methods"][method]["seconds"]) for r in chosen if "refused" not in r["methods"][method]]
            row.append(f"{statistics.median(seconds):.3f}" if seconds else "--")
        rows.append(row)
    lines += common.table(["files", *shown], rows)
    unround = [r["methods"]["unround"] for r in results]
    lines += [
        "unround's binary64 picture: the share of its coefficients within the intervals, "
        f"{mean([float(u['inside_64']) for u in unround]):.6f} on average, and the largest excess "
        f"{max(float(u['largest_64']) for u in unround):.3g} steps (the blocks that the picture covers in part, "
        "extended again by its edges, and the rounding of the DCTs: docs/math.md, 12).",
        "",
    ]
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"wrote {out / 'comparison.md'}")


def main() -> int:
    """Runs the stages asked for."""
    system = {"win32": "windows", "darwin": "macos"}.get(sys.platform, "linux")
    processor = "arm64" if platform.machine().lower() in {"arm64", "aarch64"} else "x86_64"
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rust", type=Path, required=True, help="the directory of the C API's library")
    parser.add_argument("--tools", type=Path, default=common.ROOT / "third_party" / f"{system}-{processor}" / "bin")
    parser.add_argument("--data", type=Path, default=common.ROOT / "data")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--cache", type=Path, default=common.ROOT / "data" / "comparison")
    parser.add_argument("--out", type=Path, default=common.ROOT / "experiments" / "results")
    parser.add_argument("--encoders", nargs="+", default=["libjpeg-turbo", "mozjpeg"])
    parser.add_argument("--samplings", nargs="+", default=["grey", "420"], choices=common.SAMPLINGS)
    parser.add_argument("--qualities", nargs="+", type=int, default=list(common.QUALITIES))
    parser.add_argument("--sets", nargs="+", default=list(SETS), choices=list(SETS))
    parser.add_argument("--images", type=int, help="the first images of each set by name (all where not given)")
    parser.add_argument("stages", nargs="*", choices=STAGES, default=STAGES)
    arguments = parser.parse_args()
    common.use_library(arguments.rust)
    if "run" in arguments.stages:
        common.run_all(compare, jobs_of(arguments), arguments.workers, label="comparison")
    if "report" in arguments.stages:
        report(arguments.cache, arguments.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
