#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Phase 0: the conventions of JPEG-Unround, measured on the test images.

    python experiments/phase0_verification.py --build build/release [--data data]
        [--out experiments/results/phase0-verification.md]

For every JPEG file under data/jpeg/<encoder>/ (scripts/test_images.py writes
them, from the originals in data/synthetic/):

  0. The C layer against jpeglib. The coefficients, the quantization tables and
     the sampling factors that the C layer reads have to be jpeglib's, exactly.
  1. The DCT convention. Each 8x8 block of the decoded component planes
     (libjpeg's accurate integer inverse DCT, level-shifted and clamped, before
     any upsampling), shifted by -128 and put through the orthonormal DCT-II, is
     compared with q * Q. With the convention right, what is left is the
     rounding of the planes to 8 bits: an error of about sqrt(1/12) ~ 0.29 per
     coefficient. The acceptance criterion is every coefficient within its
     interval [(q - 1/2)Q, (q + 1/2)Q] and an RMS of at most about 0.35, in the
     blocks that the decoder did not clamp to 0 or 255: the samples of a clamped
     block are no longer the inverse DCT of q * Q.
  2. The standard decoder against the QCS. djpeg's output -- upsampled with
     libjpeg's fancy upsampling and converted to RGB, 8 bits a sample -- is
     converted back to YCbCr in floating point (JFIF), averaged down to each
     component's grid (the model's S), transformed, and compared with the
     intervals: how much of what a standard decoder shows is consistent with the
     file. Only blocks that lie wholly inside the image are counted, since the
     output has no samples beyond it.
  3. The original against the QCS, twice. (a) The image the encoder was given,
     taken through the model in floating point (YCbCr, S with the encoder's edge
     padding, DCT), against the intervals the model builds from the file: how
     often, and by how much, the picture itself lies outside them. (b) The same
     image taken through libjpeg's own integer arithmetic instead -- its
     fixed-point YCbCr, rounded to 8 bits, and its downsampling, with the
     alternating rounding bias -- which is what libjpeg's encoder quantized: what
     is left outside then is the error of its integer forward DCT. A trellis-
     quantizing encoder (mozjpeg) also chooses levels that are not the nearest on
     purpose, and mozjpeg's overshoot deringing, before its DCT, pushes the white
     (255) samples of a block that is not all white beyond 255; (b) is also
     counted over the blocks that the deringing leaves alone, where what is left
     outside is the trellis's.

The tables go to --out as markdown. Everything is deterministic: the images come
from fixed seeds, and no random number is drawn here.
"""

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import numpy.typing as npt
import scipy
from scipy.fft import dctn

if TYPE_CHECKING:
    # For the annotations alone: main() imports the package once its path is set.
    from unround.jpegio import Image, Planes

ROOT: Final = Path(__file__).resolve().parent.parent
LIBRARY_NAMES: Final = {"win32": "unround_jpegio.dll", "darwin": "libunround_jpegio.dylib"}
EXECUTABLE_SUFFIX: Final = ".exe" if sys.platform == "win32" else ""
Array = npt.NDArray[np.float64]


def read_pnm(data: bytes) -> npt.NDArray[np.uint8]:
    """The samples of a binary PPM (P6) or PGM (P5) with a maximum of 255."""
    fields: list[bytes] = []
    position = 0
    while len(fields) < 4:
        while data[position : position + 1].isspace():
            position += 1
        if data[position : position + 1] == b"#":
            position = data.index(b"\n", position) + 1
            continue
        end = position
        while not data[end : end + 1].isspace():
            end += 1
        fields.append(data[position:end])
        position = end
    position += 1  # the single white-space character after the maximum
    kind, width, height, maximum = fields[0], int(fields[1]), int(fields[2]), int(fields[3])
    if maximum != 255 or kind not in {b"P5", b"P6"}:
        message = f"not an 8-bit binary PNM: {kind!r} with maximum {maximum}"
        raise ValueError(message)
    channels = 3 if kind == b"P6" else 1
    samples = np.frombuffer(data, dtype=np.uint8, count=width * height * channels, offset=position)
    return samples.reshape(height, width, channels) if channels == 3 else samples.reshape(height, width)


def ycbcr(rgb: npt.NDArray[np.uint8]) -> list[Array]:
    """JFIF's full-range YCbCr of RGB samples, in floating point."""
    red, green, blue = (rgb[..., channel].astype(np.float64) for channel in range(3))
    luma = 0.299 * red + 0.587 * green + 0.114 * blue
    blue_difference = -0.168735892 * red - 0.331264108 * green + 0.5 * blue + 128.0
    red_difference = 0.5 * red - 0.418687589 * green - 0.081312411 * blue + 128.0
    return [luma, blue_difference, red_difference]


def libjpeg_ycbcr(rgb: npt.NDArray[np.uint8]) -> list[Array]:
    """YCbCr as libjpeg's encoder computes it: fixed point with 16 fractional bits, rounded to 8 bits."""

    def fix(value: float) -> int:
        return int(value * 65536 + 0.5)

    half, offset = 1 << 15, 128 << 16
    red, green, blue = (rgb[..., channel].astype(np.int64) for channel in range(3))
    luma = (fix(0.29900) * red + fix(0.58700) * green + fix(0.11400) * blue + half) >> 16
    blue_difference = (-fix(0.16874) * red - fix(0.33126) * green + fix(0.5) * blue + offset + half - 1) >> 16
    red_difference = (fix(0.5) * red - fix(0.41869) * green - fix(0.08131) * blue + offset + half - 1) >> 16
    return [plane.astype(np.float64) for plane in (luma, blue_difference, red_difference)]


def replicate(plane: Array, height: int, width: int) -> Array:
    """The plane grown to height x width by repeating its last row and column."""
    return np.pad(plane, ((0, height - plane.shape[0]), (0, width - plane.shape[1])), mode="edge")


def encoder_padding(plane: Array, factor_y: int, columns: int, factor_x: int) -> Array:
    """A full-resolution plane padded as libjpeg's encoder pads it before downsampling.

    Across, each row is repeated out to the width the component's blocks need
    (jcsample.c, expand_right_edge). Down, only the last group of factor_y rows is
    completed by repeating the last row (jcprepct.c); the rows of blocks below that
    are filled after downsampling, by repeating the last downsampled row.
    """
    height = -(-plane.shape[0] // factor_y) * factor_y
    return replicate(plane, height, columns * factor_x)


def average_down(plane: Array, rows: int, columns: int, factor_y: int, factor_x: int) -> Array:
    """The model's S, in floating point: the plane padded as the encoder pads it, averaged over each factor."""
    padded = encoder_padding(plane, factor_y, columns, factor_x)
    groups = padded.shape[0] // factor_y
    averaged = padded.reshape(groups, factor_y, columns, factor_x).mean(axis=(1, 3))
    return replicate(averaged, rows, columns)


def libjpeg_downsample(plane: Array, rows: int, columns: int, factor_y: int, factor_x: int) -> Array:
    """libjpeg's downsampling: the padded plane averaged in integers, with an alternating rounding bias."""
    padded = encoder_padding(plane, factor_y, columns, factor_x).astype(np.int64)
    if (factor_y, factor_x) == (1, 1):
        averaged = padded
    elif (factor_y, factor_x) == (1, 2):
        bias = np.arange(columns, dtype=np.int64) % 2  # 0, 1, 0, 1, ... across a row
        averaged = (padded[:, 0::2] + padded[:, 1::2] + bias) >> 1
    elif (factor_y, factor_x) == (2, 2):
        bias = np.where(np.arange(columns) % 2 == 0, 1, 2).astype(np.int64)  # 1, 2, 1, 2, ...
        total = padded[0::2, 0::2] + padded[0::2, 1::2] + padded[1::2, 0::2] + padded[1::2, 1::2]
        averaged = (total + bias) >> 2
    else:
        message = f"no libjpeg downsampling by {factor_y} x {factor_x} here"
        raise ValueError(message)
    return replicate(averaged.astype(np.float64), rows, columns)


def blocks_of(plane: Array) -> Array:
    """The whole 8x8 blocks of a plane, as (block rows, block columns, 8, 8)."""
    rows, columns = plane.shape[0] // 8, plane.shape[1] // 8
    return plane[: rows * 8, : columns * 8].reshape(rows, 8, columns, 8).transpose(0, 2, 1, 3)


def block_dct(plane: Array) -> Array:
    """The orthonormal DCT-II of each 8x8 block, as (block rows, block columns, 8, 8)."""
    return np.asarray(dctn(blocks_of(plane), axes=(2, 3), norm="ortho"), dtype=np.float64)


@dataclasses.dataclass
class Tally:
    """Sums over coefficients, for fractions and root mean squares."""

    count: int = 0
    inside: int = 0
    squares: float = 0.0
    worst: float = 0.0  # the largest |difference| / Q
    excess: list[float] = dataclasses.field(default_factory=list)  # (|difference| - Q/2) / Q, where positive

    def add(self, difference: Array, step: Array, *, keep_excess: bool = False) -> None:
        relative = np.abs(difference) / step
        self.count += difference.size
        self.inside += int(np.count_nonzero(relative <= 0.5 + 1e-9))
        self.squares += float(np.sum(difference * difference))
        self.worst = max(self.worst, float(relative.max(initial=0.0)))
        if keep_excess:
            outside = relative[relative > 0.5 + 1e-9] - 0.5
            self.excess.extend(float(value) for value in outside)

    def inside_percent(self) -> float:
        return 100.0 * self.inside / self.count if self.count else float("nan")

    def rms(self) -> float:
        return float(np.sqrt(self.squares / self.count)) if self.count else float("nan")


@dataclasses.dataclass(frozen=True)
class Key:
    encoder: str
    sampling: str
    quality: int
    kind: str  # "Y" for luma and grey, "C" for chroma


def tally_table() -> dict[Key, Tally]:
    return defaultdict(Tally)


@dataclasses.dataclass
class Tallies:
    """The measurements, added up for each encoder, sampling, quality and kind of component."""

    convention: dict[Key, Tally] = dataclasses.field(default_factory=tally_table)
    convention_unclamped: dict[Key, Tally] = dataclasses.field(default_factory=tally_table)
    decoded: dict[Key, Tally] = dataclasses.field(default_factory=tally_table)
    truth: dict[Key, Tally] = dataclasses.field(default_factory=tally_table)
    truth_integer: dict[Key, Tally] = dataclasses.field(default_factory=tally_table)
    # The blocks that mozjpeg's overshoot deringing leaves as they are: those
    # with no sample at 255, or with every sample at 255 (jcdctmgr.c,
    # preprocess_deringing).
    truth_integer_undisturbed: dict[Key, Tally] = dataclasses.field(default_factory=tally_table)


@dataclasses.dataclass
class Run:
    """What the tables say about the run itself."""

    files: int
    mismatches: list[str]
    manifests: dict[str, str]  # each encoder, and the version its cjpeg reports
    libjpeg: str
    jpeglib_version: str
    elapsed: float


def jpeg_files(data: Path) -> Iterator[tuple[str, dict[str, object], Path]]:
    for manifest_path in sorted((data / "jpeg").glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for entry in manifest["files"]:
            yield str(manifest["encoder"]), entry, manifest_path.parent / str(entry["file"])


def agrees_with_jpeglib(image: Image, theirs: Any) -> bool:  # noqa: ANN401 -- jpeglib has no types
    """0. Whether the C layer read the coefficients, tables and sampling factors that jpeglib reads."""
    their_components = [theirs.Y] if len(image.components) == 1 else [theirs.Y, theirs.Cb, theirs.Cr]
    for index, (component, coefficients) in enumerate(zip(image.components, their_components, strict=True)):
        slot = int(theirs.quant_tbl_no[index])
        v, h = (int(value) for value in theirs.samp_factor[index])
        if (
            not np.array_equal(component.coefficients, coefficients)
            or not np.array_equal(component.quant_table, theirs.qt[slot])
            or (component.h_samp_factor, component.v_samp_factor) != (h, v)
        ):
            return False
    return True


def measure_convention(tallies: Tallies, keys: list[Key], image: Image, planes: Planes) -> None:
    """1. The decoded planes against q * Q: overall, and in the blocks that were not clamped."""
    for key, component, plane in zip(keys, image.components, planes.planes, strict=True):
        step = component.quant_table.astype(np.float64)
        samples = plane.astype(np.float64)
        difference = block_dct(samples - 128.0) - component.coefficients.astype(np.float64) * step
        steps = np.broadcast_to(step, difference.shape)
        tallies.convention[key].add(difference, steps)
        blocks = blocks_of(samples)
        unclamped = ~((blocks == 0.0) | (blocks == 255.0)).any(axis=(2, 3))
        tallies.convention_unclamped[key].add(difference[unclamped], steps[unclamped])


def measure_against_intervals(
    tallies: Tallies, keys: list[Key], image: Image, shown: list[Array], original: npt.NDArray[np.uint8]
) -> None:
    """2 and 3. The standard decoder's output and the original, on each component's grid, against the QCS."""
    original_planes = [original.astype(np.float64)] if original.ndim == 2 else ycbcr(original)
    integer_planes = [original.astype(np.float64)] if original.ndim == 2 else libjpeg_ycbcr(original)
    for index, (key, component) in enumerate(zip(keys, image.components, strict=True)):
        factor_x = image.max_h_samp_factor // component.h_samp_factor
        factor_y = image.max_v_samp_factor // component.v_samp_factor
        step = component.quant_table.astype(np.float64)
        centre = component.coefficients.astype(np.float64) * step
        rows, columns = component.height_in_blocks * 8, component.width_in_blocks * 8

        # 2. Only the blocks wholly inside the image: the output has no samples beyond it.
        whole_rows, whole_columns = component.height // 8, component.width // 8
        if whole_rows and whole_columns:
            output = average_down(shown[index], rows, columns, factor_y, factor_x)
            difference = block_dct(output - 128.0)[:whole_rows, :whole_columns] - centre[:whole_rows, :whole_columns]
            tallies.decoded[key].add(difference, np.broadcast_to(step, difference.shape))

        given = average_down(original_planes[index], rows, columns, factor_y, factor_x)
        difference = block_dct(given - 128.0) - centre
        tallies.truth[key].add(difference, np.broadcast_to(step, difference.shape), keep_excess=True)

        quantized = libjpeg_downsample(integer_planes[index], rows, columns, factor_y, factor_x)
        difference = block_dct(quantized - 128.0) - centre
        steps = np.broadcast_to(step, difference.shape)
        tallies.truth_integer[key].add(difference, steps, keep_excess=True)
        white = blocks_of(quantized) >= 255.0
        undisturbed = ~white.any(axis=(2, 3)) | white.all(axis=(2, 3))
        tallies.truth_integer_undisturbed[key].add(difference[undisturbed], steps[undisturbed], keep_excess=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--build", type=Path, required=True, help="a build directory with -DUNROUND_WITH_PYTHON=ON")
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "results" / "phase0-verification.md")
    options = parser.parse_args()

    build = options.build.resolve()
    library = build / "python" / LIBRARY_NAMES.get(sys.platform, "libunround_jpegio.so")
    os.environ.setdefault("UNROUND_JPEGIO_LIBRARY", str(library))
    sys.path.insert(0, str(ROOT / "python" / "src"))
    # Imported here, once the paths to the package and to its library are set.
    import jpeglib  # noqa: PLC0415

    from unround import jpegio  # noqa: PLC0415

    djpeg = build / "libjpeg-turbo" / "install" / "bin" / f"djpeg{EXECUTABLE_SUFFIX}"
    started = time.perf_counter()
    tallies = Tallies()
    files = 0
    mismatches: list[str] = []
    for encoder, entry, path in jpeg_files(options.data):
        data = path.read_bytes()
        image = jpegio.read(data)
        files += 1
        if not agrees_with_jpeglib(image, jpeglib.read_dct(str(path))):
            mismatches.append(path.relative_to(options.data).as_posix())
        sampling, quality = str(entry["sampling"]), int(str(entry["quality"]))
        keys = [Key(encoder, sampling, quality, "Y" if index == 0 else "C") for index in range(len(image.components))]
        measure_convention(tallies, keys, image, jpegio.decode_planes(data))
        output = read_pnm(subprocess.run([str(djpeg), str(path)], capture_output=True, check=True).stdout)
        shown = [output.astype(np.float64)] if output.ndim == 2 else ycbcr(output)
        original = read_pnm((options.data / "synthetic" / str(entry["original"])).read_bytes())
        measure_against_intervals(tallies, keys, image, shown, original)

    manifests = {}
    for manifest_path in sorted((options.data / "jpeg").glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifests[str(manifest["encoder"])] = str(manifest["cjpeg"])
    elapsed = time.perf_counter() - started
    run = Run(files, mismatches, manifests, jpegio.libjpeg_version(), jpeglib.__version__, elapsed)
    options.out.parent.mkdir(parents=True, exist_ok=True)
    options.out.write_text("\n".join(render(run, tallies)) + "\n", encoding="utf-8", newline="\n")
    print(f"{files} files, {len(mismatches)} mismatches with jpeglib, {elapsed:.1f} s; the tables are in {options.out}")
    return 0 if not mismatches else 1


def render(run: Run, tallies: Tallies) -> list[str]:
    keys = sorted(tallies.convention, key=lambda key: (key.encoder, key.sampling, key.quality, key.kind))
    encoders = sorted({key.encoder for key in keys})
    samplings = [
        sampling for sampling in ("grey", "444", "422", "420") if any(key.sampling == sampling for key in keys)
    ]
    qualities = sorted({key.quality for key in keys})

    def table(title: str, value: Callable[[Tally], str], source: dict[Key, Tally], encoder: str) -> list[str]:
        header = ["quality"]
        columns: list[tuple[str, str]] = []
        for sampling in samplings:
            for kind in ("Y", "C"):
                if sampling == "grey" and kind == "C":
                    continue
                if any(Key(encoder, sampling, quality, kind) in source for quality in qualities):
                    columns.append((sampling, kind))
                    header.append(f"{sampling} {kind}")
        rows = [f"**{title}**", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        for quality in qualities:
            cells = [str(quality)]
            for sampling, kind in columns:
                tally = source.get(Key(encoder, sampling, quality, kind))
                cells.append(value(tally) if tally is not None and tally.count else "")
            rows.append("| " + " | ".join(cells) + " |")
        rows.append("")
        return rows

    def percent(tally: Tally) -> str:
        return f"{tally.inside_percent():.3f}"

    def rms(tally: Tally) -> str:
        return f"{tally.rms():.3f}"

    def worst(tally: Tally) -> str:
        return f"{tally.worst:.3f}"

    def tail(tally: Tally) -> str:
        if not tally.excess:
            return "0"
        return f"{np.quantile(np.asarray(tally.excess), 0.99):.3f} / {max(tally.excess):.3f}"

    excess = "(|difference| - Q/2) / Q, 99th percentile / largest"
    lines = [
        "<!-- Written by experiments/phase0_verification.py; do not edit by hand. -->",
        "",
        "# Phase 0 verification tables",
        "",
        f"- Files: {run.files} JPEG files of the synthetic test images (scripts/test_images.py), "
        f"read in {run.elapsed:.0f} s.",
        f"- The C layer: {run.libjpeg}. jpeglib {run.jpeglib_version}. "
        f"NumPy {np.__version__}, SciPy {scipy.__version__}.",
    ]
    lines += [f"- Encoder `{name}`: {version}." for name, version in sorted(run.manifests.items())]
    lines += [
        "- Columns are the chroma sampling (grey, 4:4:4, 4:2:2, 4:2:0) and the component: Y (luma, or grey) and C "
        "(both chroma components together).",
        "",
        "## 0. The C layer against jpeglib",
        "",
        f"Coefficients, quantization tables and sampling factors of every component of {run.files} files: "
        f"{len(run.mismatches)} mismatches.",
        "",
    ]
    lines += [f"- {mismatch}" for mismatch in run.mismatches[:20]]
    for encoder in encoders:
        lines += [f"## 1. The DCT convention: the decoded planes against q·Q (`{encoder}`)", ""]
        lines += table("Coefficients within their interval, percent", percent, tallies.convention, encoder)
        lines += table("RMS of DCT(plane - 128) - q·Q, in coefficient units", rms, tallies.convention, encoder)
        lines += table("The largest |DCT(plane - 128) - q·Q| / Q", worst, tallies.convention, encoder)
        unclamped = tallies.convention_unclamped
        lines += table(
            "Within their interval, in blocks with no sample at 0 or 255, percent", percent, unclamped, encoder
        )
        lines += table("RMS in blocks with no sample at 0 or 255, in coefficient units", rms, unclamped, encoder)
        lines += [f"## 2. The standard decoder's output against the QCS (`{encoder}`)", ""]
        lines += table(
            "Coefficients of djpeg's RGB output within their interval, percent", percent, tallies.decoded, encoder
        )
        lines += table("RMS of the difference from q·Q, in coefficient units", rms, tallies.decoded, encoder)
        lines += [f"## 3a. The original, in floating point, against the QCS (`{encoder}`)", ""]
        lines += table("Coefficients of the original within their interval, percent", percent, tallies.truth, encoder)
        lines += table(f"Outside coefficients: {excess}", tail, tallies.truth, encoder)
        lines += [f"## 3b. The original, in libjpeg's integer arithmetic, against the QCS (`{encoder}`)", ""]
        lines += table("Coefficients within their interval, percent", percent, tallies.truth_integer, encoder)
        lines += table(f"Outside coefficients: {excess}", tail, tallies.truth_integer, encoder)
        undisturbed = tallies.truth_integer_undisturbed
        lines += table(
            "Within their interval, in blocks that overshoot deringing leaves alone (no sample at 255, or only 255), "
            "percent",
            percent,
            undisturbed,
            encoder,
        )
        lines += table(f"Outside coefficients of those blocks: {excess}", tail, undisturbed, encoder)
    return lines


if __name__ == "__main__":
    sys.exit(main())
