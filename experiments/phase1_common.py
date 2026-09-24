# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""What the experiments of Phase 1 share: the files, the package, the measures, and runs in parallel.

The files are the greyscale ones of the synthetic images (scripts/test_images.py),
as each encoder's manifest lists them. A run's results are kept as JSON, one file
to a run, so that a report can be written again without solving anything.
"""

import dataclasses
import json
import math
import os
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import numpy.typing as npt
from PIL import Image
from skimage.metrics import structural_similarity

if TYPE_CHECKING:
    # For the annotations alone: the package is imported once use_build() has set its path.
    from unround.model import Problem
    from unround.results import Result

ROOT: Final = Path(__file__).resolve().parent.parent
LIBRARY_NAMES: Final = {"win32": "unround_jpegio.dll", "darwin": "libunround_jpegio.dylib"}
QUALITIES: Final = (10, 20, 30, 50, 70, 90)

type Array = npt.NDArray[np.float64]


def use_build(build: Path) -> None:
    """Points the package at the C layer of a build (-DUNROUND_WITH_PYTHON=ON), and Python at the package.

    BLAS gets one thread in each process: the runs go in parallel, one to a process.
    Processes started after this inherit all of it.
    """
    library = build.resolve() / "python" / LIBRARY_NAMES.get(sys.platform, "libunround_jpegio.so")
    os.environ.setdefault("UNROUND_JPEGIO_LIBRARY", str(library))
    os.environ.setdefault("UNROUND_PYTHON_SOURCE", str(ROOT / "python" / "src"))
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    add_package()


def add_package() -> None:
    """Puts the package on the path of this process, as use_build() said where it is."""
    source = os.environ["UNROUND_PYTHON_SOURCE"]
    if source not in sys.path:
        sys.path.insert(0, source)


@dataclasses.dataclass(frozen=True, slots=True)
class Case:
    """A greyscale JPEG file of a synthetic image, and the image it was encoded from."""

    encoder: str
    split: str
    image: str
    quality: int
    jpeg: Path
    original: Path

    @property
    def name(self) -> str:
        """The encoder, the image and the quality, as in the names of the results."""
        return f"{self.encoder}-{self.image}-q{self.quality}"


def cases(data: Path, encoder: str, split: str) -> list[Case]:
    """The greyscale files of an encoder and a split (tuning or test), by image and quality."""
    manifest = json.loads((data / "jpeg" / encoder / "manifest.json").read_text(encoding="utf-8"))
    found = []
    for entry in manifest["files"]:
        file, original = str(entry["file"]), str(entry["original"])
        if entry["sampling"] != "grey" or not file.startswith(f"{split}/"):
            continue
        image = Path(original).name.removesuffix(".pgm")
        found.append(
            Case(
                encoder=encoder,
                split=split,
                image=image,
                quality=int(entry["quality"]),
                jpeg=data / "jpeg" / encoder / file,
                original=data / "synthetic" / original,
            )
        )
    return sorted(found, key=lambda case: (case.image, case.quality))


def read_original(path: Path) -> npt.NDArray[np.uint8]:
    """The samples of a greyscale original (8-bit PGM)."""
    with Image.open(path) as image:
        if image.mode != "L":
            message = f"{path} is not an 8-bit greyscale picture: {image.mode}"
            raise ValueError(message)
        return np.asarray(image, dtype=np.uint8)


def ssim(reference: npt.NDArray[np.generic], image: npt.NDArray[np.generic]) -> float:
    """SSIM as Wang et al. define it: Gaussian weights of sigma 1.5, the population covariance, a range of 255."""
    # scikit-image declares no types.
    return float(
        structural_similarity(  # type: ignore[no-untyped-call]
            reference.astype(np.float64),
            image.astype(np.float64),
            data_range=255.0,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
        )
    )


def measure(
    original: npt.NDArray[np.uint8], picture: Array, problem: Problem, canvas: Array | None = None
) -> dict[str, float]:
    """How good a reconstruction is, as binary64 and as 8-bit samples, and how consistent with the file.

    picture is the reconstruction's samples, cut to the picture, and canvas, where there is
    one, the whole canvas it was cut from. The canvas's own coefficients are compared with
    their intervals: that is the constraint the solvers keep. The picture is compared as a
    file that encoded it would be (unround.metrics.picture_excess), as it is and as the
    8-bit samples that unround.metrics.quantize() rounds it to, over every block and over
    the blocks wholly within it. Consistency is the share of the coefficients within their
    intervals, and the largest excess in steps.
    """
    from unround import dct, metrics  # noqa: PLC0415
    from unround.model import excess  # noqa: PLC0415

    eight = metrics.quantize(picture)
    measured = {
        "psnr": metrics.psnr(original, picture),
        "psnr_8": metrics.psnr(original, eight),
        "ssim_8": ssim(original, eight),
        "psnr_b_8": metrics.psnr_b(original, eight),
    }
    # The blocks wholly within the picture, whose samples a padding does not touch.
    whole_rows, whole_columns = picture.shape[0] // dct.BLOCK, picture.shape[1] // dct.BLOCK
    for suffix, samples in (("", picture), ("_8", eight.astype(np.float64))):
        outside = metrics.picture_excess(problem, samples)
        measured[f"picture_inside{suffix}"] = float(np.count_nonzero(outside == 0.0)) / outside.size
        measured[f"picture_outside_largest{suffix}"] = float(outside.max())
        within = outside[:whole_rows, :whole_columns]
        measured[f"whole_inside{suffix}"] = float(np.count_nonzero(within == 0.0)) / max(within.size, 1)
        measured[f"whole_outside_largest{suffix}"] = float(within.max()) if within.size else 0.0
    if canvas is not None:
        outside = excess(problem, dct.forward(canvas))
        measured["canvas_inside"] = float(np.count_nonzero(outside == 0.0)) / outside.size
        measured["canvas_outside_largest"] = float(outside.max())
    return measured


def root_mean_square(values: Array) -> float:
    """The RMS of an array, summed in binary64."""
    return math.sqrt(float(np.mean(values * values)))


def save(path: Path, result: dict[str, Any]) -> None:
    """Keeps a run's results as JSON, written whole or not at all."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")
    partial.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
    partial.replace(path)


def load(path: Path) -> dict[str, Any]:
    """A run's results."""
    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def run_all[T](
    work: Callable[[T], dict[str, Any]], jobs: Sequence[tuple[Path, T]], workers: int, *, label: str
) -> None:
    """Runs work on each job whose results are not kept yet, in parallel, and keeps what it returns.

    jobs are the paths of the results and what to run for them. With one worker, the jobs
    run in this process, one after another: for timings.
    """
    pending = [(path, job) for path, job in jobs if not path.exists()]
    print(f"{label}: {len(pending)} of {len(jobs)} runs to go, {workers} at a time", flush=True)
    started = time.perf_counter()
    if workers <= 1:
        for done, (path, job) in enumerate(pending, start=1):
            save(path, work(job))
            print(f"  {done}/{len(pending)} {path.stem} ({time.perf_counter() - started:.0f} s)", flush=True)
        return
    with ProcessPoolExecutor(max_workers=workers, initializer=add_package) as pool:
        futures = {pool.submit(work, job): path for path, job in pending}
        for done, future in enumerate(as_completed(futures), start=1):
            path = futures[future]
            save(path, future.result())
            print(f"  {done}/{len(pending)} {path.stem} ({time.perf_counter() - started:.0f} s)", flush=True)


def history_of(result: Result) -> dict[str, list[float] | list[int]]:
    """A solver's history (unround.results.History), as lists for JSON."""
    history = result.history
    return {
        "iterations": [int(value) for value in history.iterations],
        "seconds": [float(value) for value in history.seconds],
        "primal": [float(value) for value in history.primal],
        "dual": [float(value) for value in history.dual],
        "scaling": [float(value) for value in history.scaling],
    }


def table(header: Sequence[str], rows: Iterable[Sequence[str]]) -> list[str]:
    """A markdown table, and the empty line after it."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return [*lines, ""]


def first_within(iterations: Iterable[int], gaps: Iterable[float], tolerance: float) -> int | None:
    """The first recorded iteration whose gap is within the tolerance, None if there is none."""
    for iteration, gap in zip(iterations, gaps, strict=True):
        if gap <= tolerance:
            return iteration
    return None
