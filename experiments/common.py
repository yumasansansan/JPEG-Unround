# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""What the experiments share: the files, the package, the measures, and runs in parallel.

The files are those of the test images (scripts/test_images.py), as each encoder's
manifest lists them: the synthetic images, and the photographs. The reconstructions are
the reference implementation's (Rust), through the package's binding (unround.native),
at the path that use_library() gives; the measures are those of experiments/measures.py.
A run's results are kept as JSON, one file to a run, so that a report can be written
again without solving anything.
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

import measures

if TYPE_CHECKING:
    # For the annotations alone: the package is imported once use_library() has set its path.
    from unround.native import History, Problem

ROOT: Final = Path(__file__).resolve().parent.parent
LIBRARY_NAMES: Final = {"win32": "unround_capi.dll", "darwin": "libunround_capi.dylib"}
QUALITIES: Final = (10, 20, 30, 50, 70, 90)
SAMPLINGS: Final = ("grey", "420", "422", "444")

type Array = npt.NDArray[np.float64]


def use_library(directory: Path) -> None:
    """Points the package at the reference implementation's C interface in the directory, and Python at the package.

    cargo build --release -p unround-capi, in rust/, writes it into rust/target/release;
    the package reads JPEG files through it too. BLAS gets one thread in each process: the
    runs go in parallel, one to a process. Processes started after this inherit all of it.
    """
    library = directory.resolve() / LIBRARY_NAMES.get(sys.platform, "libunround_capi.so")
    os.environ.setdefault("UNROUND_LIBRARY", str(library))
    os.environ.setdefault("UNROUND_PYTHON_SOURCE", str(ROOT / "python" / "src"))
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    add_package()


def add_package() -> None:
    """Puts the package on the path of this process, as use_library() said where it is."""
    source = os.environ["UNROUND_PYTHON_SOURCE"]
    if source not in sys.path:
        sys.path.insert(0, source)


@dataclasses.dataclass(frozen=True, slots=True)
class Case:
    """A JPEG file of a test image, and the picture it was encoded from."""

    encoder: str
    split: str
    image: str
    quality: int
    sampling: str
    jpeg: Path
    original: Path

    @property
    def name(self) -> str:
        """The encoder, the image, the quality and the sampling, as in the names of the results."""
        name = f"{self.encoder}-{self.image}-q{self.quality}"
        return name if self.sampling == "grey" else f"{name}-{self.sampling}"


def cases(  # noqa: PLR0913 -- where the files are, which of them, and the directories of photographs
    data: Path,
    encoder: str,
    split: str,
    *,
    samplings: Sequence[str] = ("grey",),
    jpeg: str = "jpeg",
    originals: str = "synthetic",
) -> list[Case]:
    """The files of an encoder and a split (tuning or test) in the samplings given, by image, quality and sampling.

    jpeg and originals are the directories under data of the JPEG files and of the
    pictures they were encoded from: those of the synthetic images, or jpeg-photos and
    photo-originals for the photographs (scripts/test_images.py).
    """
    manifest = json.loads((data / jpeg / encoder / "manifest.json").read_text(encoding="utf-8"))
    found = []
    for entry in manifest["files"]:
        file, original, sampling = str(entry["file"]), str(entry["original"]), str(entry["sampling"])
        if sampling not in samplings or not file.startswith(f"{split}/"):
            continue
        found.append(
            Case(
                encoder=encoder,
                split=split,
                image=Path(original).stem,
                quality=int(entry["quality"]),
                sampling=sampling,
                jpeg=data / jpeg / encoder / file,
                original=data / originals / original,
            )
        )
    return sorted(found, key=lambda case: (case.image, case.quality, SAMPLINGS.index(case.sampling)))


def read_original(path: Path) -> npt.NDArray[np.uint8]:
    """The samples of an original: (height, width) of an 8-bit PGM, or (height, width, 3) of an 8-bit PPM."""
    with Image.open(path) as image:
        if image.mode not in {"L", "RGB"}:
            message = f"{path} is not an 8-bit greyscale or RGB picture: {image.mode}"
            raise ValueError(message)
        return np.asarray(image, dtype=np.uint8)


def ssim(reference: npt.NDArray[np.generic], image: npt.NDArray[np.generic]) -> float:
    """SSIM as Wang et al. define it: Gaussian weights of sigma 1.5, the population covariance, a range of 255.

    Of an RGB picture, the mean over R, G and B.
    """
    if reference.ndim == 3:
        return float(np.mean([ssim(reference[:, :, channel], image[:, :, channel]) for channel in range(3)]))
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
    original: npt.NDArray[np.uint8], picture: Array, problem: Problem | None = None, canvas: Array | None = None
) -> dict[str, float]:
    """How good a reconstruction is, as binary64 and as 8-bit samples, and how consistent with the file.

    picture is the reconstruction's samples, cut to the picture: (height, width), or
    (height, width, 3) of RGB. Of a greyscale file, problem is its model (as the reference
    implementation made it), and canvas, where there is one, the whole canvas the picture
    was cut from: the canvas's own coefficients are compared with their intervals, which is
    the constraint the solvers keep; and the picture is compared as a file that encoded it
    would be (measures.picture_excess), as it is and as the 8-bit samples that
    measures.quantize() rounds it to, over every block and over the blocks wholly within
    it. Consistency is the share of the coefficients within their intervals, and the
    largest excess in steps.
    """
    eight = measures.quantize(picture)
    measured = {
        "psnr": measures.psnr(original, picture),
        "psnr_8": measures.psnr(original, eight),
        "ssim_8": ssim(original, eight),
    }
    if picture.ndim != 2 or problem is None:
        return measured
    measured["psnr_b_8"] = measures.psnr_b(original, eight)
    # The blocks wholly within the picture, whose samples a padding does not touch.
    whole_rows, whole_columns = picture.shape[0] // measures.BLOCK, picture.shape[1] // measures.BLOCK
    for suffix, samples in (("", picture), ("_8", eight.astype(np.float64))):
        outside = measures.picture_excess(problem, samples)
        measured[f"picture_inside{suffix}"] = float(np.count_nonzero(outside == 0.0)) / outside.size
        measured[f"picture_outside_largest{suffix}"] = float(outside.max())
        within = outside[:whole_rows, :whole_columns]
        measured[f"whole_inside{suffix}"] = float(np.count_nonzero(within == 0.0)) / max(within.size, 1)
        measured[f"whole_outside_largest{suffix}"] = float(within.max()) if within.size else 0.0
    if canvas is not None:
        outside = measures.excess(problem, measures.block_dct(canvas))
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


def history_of(history: History) -> dict[str, list[float] | list[int]]:
    """A solver's records (unround.native.History), as lists for JSON."""
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
