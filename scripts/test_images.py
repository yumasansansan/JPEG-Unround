#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Synthetic test images for JPEG-Unround, and their JPEG files.

    python scripts/test_images.py synthesize [--out data/synthetic]
    python scripts/test_images.py encode --cjpeg <cjpeg> --encoder <name> [--source data/synthetic] [--out data/jpeg]

synthesize draws the originals: text, charts, gradients, line art, user
interfaces and flat-colour illustrations -- what JPEG-Unround is for first --
in colour (PPM), and the text, charts, line art and illustrations in grey as well
(PGM). Everything comes from NumPy's generator with fixed seeds and from Pillow's
own rasterizer and bitmap font, with no font file of the system; the arithmetic
that makes a pixel is +, -, *, / and square roots, which IEEE 754 rounds exactly
alike everywhere, and never a sine or a logarithm of the system's library. So
every system draws the same bytes:
scripts/test-images.sha256 lists them, and synthesize checks each file against
it. Lines and shapes are drawn at four times the size and averaged down, as
antialiasing; text is drawn at its size, as a screen draws it.

The originals are split, per category, into images for tuning parameters (the
first two of each) and images for testing (the other three): parameters are
never tuned on the images they are tested with.

encode writes each original as JPEG with a cjpeg -- libjpeg-turbo's, pinned by
the submodule, or mozjpeg's, whose trellis quantization is a case to study -- at
qualities 10, 20, 30, 50, 70 and 90, and for colour at 4:4:4, 4:2:2 and 4:2:0.
The files go to data/jpeg/<encoder>/<split>/<name>-q<quality>-<sampling>.jpg,
with a manifest of the encoder, the options and every file's SHA-256.

data/ is not committed.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw, ImageFont

ROOT: Final = Path(__file__).resolve().parent.parent
HASHES: Final = ROOT / "scripts" / "test-images.sha256"
SCALE: Final = 4  # shapes are drawn this many times larger and averaged down
QUALITIES: Final = (10, 20, 30, 50, 70, 90)
SAMPLINGS: Final = {"444": "1x1", "422": "2x1", "420": "2x2"}
SIZES: Final = ((512, 384), (509, 381), (640, 360), (400, 300), (333, 251))
TUNING_PER_CATEGORY: Final = 2
IMAGES_PER_CATEGORY: Final = 5
GREY_CATEGORIES: Final = ("text", "chart", "lineart", "illustration")

Pixels = npt.NDArray[np.uint8]
Colour = tuple[int, int, int]

WORD_LETTERS: Final = "abcdefghijklmnopqrstuvwxyz"

# The twelve directions at multiples of 30 degrees, from square roots alone.
_HALF_ROOT_THREE: Final = float(np.sqrt(3.0)) / 2.0
DIRECTIONS: Final = (
    (1.0, 0.0),
    (_HALF_ROOT_THREE, 0.5),
    (0.5, _HALF_ROOT_THREE),
    (0.0, 1.0),
    (-0.5, _HALF_ROOT_THREE),
    (-_HALF_ROOT_THREE, 0.5),
    (-1.0, 0.0),
    (-_HALF_ROOT_THREE, -0.5),
    (-0.5, -_HALF_ROOT_THREE),
    (0.0, -1.0),
    (0.5, -_HALF_ROOT_THREE),
    (_HALF_ROOT_THREE, -0.5),
)


def average_down(image: Image.Image, factor: int) -> Pixels:
    """The image averaged over factor x factor squares, rounded to the nearest level."""
    pixels = np.asarray(image, dtype=np.uint32)
    height, width = pixels.shape[0] // factor, pixels.shape[1] // factor
    blocks = pixels[: height * factor, : width * factor].reshape(height, factor, width, factor, -1)
    total = blocks.sum(axis=(1, 3))
    area = factor * factor
    return ((total + area // 2) // area).astype(np.uint8)


def random_colour(rng: np.random.Generator, low: int = 0, high: int = 255) -> Colour:
    red, green, blue = (int(value) for value in rng.integers(low, high + 1, size=3))
    return red, green, blue


def words(rng: np.random.Generator, count: int) -> str:
    chosen = []
    for _ in range(count):
        length = int(rng.integers(1, 10))
        chosen.append("".join(WORD_LETTERS[int(index)] for index in rng.integers(0, 26, size=length)))
    return " ".join(chosen)


def draw_text(image: Image.Image, position: tuple[int, int], text: str, colour: Colour, magnify: int) -> None:
    """Text in Pillow's bitmap font, magnified by whole pixels, as a screen draws a pixel font."""
    font = ImageFont.load_default_imagefont()
    left, top, right, bottom = font.getbbox(text)
    mask = Image.new("L", (right - left + 1, bottom - top + 1), 0)
    ImageDraw.Draw(mask).text((-left, -top), text, font=font, fill=255)
    if magnify > 1:
        mask = mask.resize((mask.width * magnify, mask.height * magnify), Image.Resampling.NEAREST)
    image.paste(Image.new("RGB", mask.size, colour), position, mask)


def text_image(rng: np.random.Generator, width: int, height: int) -> Pixels:
    pairs = [
        ((255, 255, 255), (0, 0, 0)),
        ((250, 244, 228), (20, 40, 110)),
        ((40, 42, 48), (225, 230, 235)),
        ((30, 90, 200), (255, 255, 255)),
    ]
    background, foreground = pairs[int(rng.integers(0, len(pairs)))]
    image = Image.new("RGB", (width, height), background)
    y = 4
    while y < height - 8:
        magnify = int(rng.choice([1, 1, 1, 2, 2, 3]))
        colour = foreground if rng.random() < 0.8 else random_colour(rng, 0, 200)
        line = words(rng, int(rng.integers(3, 12)))
        characters = max(1, (width - 8) // (6 * magnify))
        draw_text(image, (4, y), line[:characters], colour, magnify)
        y += 12 * magnify + int(rng.integers(0, 6))
    return np.asarray(image, dtype=np.uint8)


def chart_image(rng: np.random.Generator, width: int, height: int) -> Pixels:
    big = Image.new("RGB", (width * SCALE, height * SCALE), (255, 255, 255))
    draw = ImageDraw.Draw(big)
    left, top, right, bottom = 40 * SCALE, 16 * SCALE, (width - 12) * SCALE, (height - 30) * SCALE
    for step in range(1, 6):
        y = bottom - (bottom - top) * step // 6
        draw.line([(left, y), (right, y)], fill=(215, 215, 215), width=SCALE)
    palette = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189), (140, 86, 75)]
    bars = int(rng.integers(5, 11))
    slot = (right - left) // bars
    for index in range(bars):
        value = float(rng.uniform(0.1, 0.95))
        x0 = left + index * slot + slot // 6
        x1 = left + (index + 1) * slot - slot // 6
        y0 = bottom - int((bottom - top) * value)
        draw.rectangle([x0, y0, x1, bottom], fill=palette[index % len(palette)])
    points = []
    level = float(rng.uniform(0.3, 0.7))
    for index in range(bars + 1):
        level = min(0.95, max(0.05, level + float(rng.uniform(-0.2, 0.2))))
        points.append((left + index * slot, bottom - int((bottom - top) * level)))
    draw.line(points, fill=(20, 20, 20), width=3 * SCALE // 2)
    for x, y in points:
        radius = 3 * SCALE
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius], fill=(255, 255, 255), outline=(20, 20, 20), width=SCALE
        )
    draw.line([(left, top), (left, bottom), (right, bottom)], fill=(0, 0, 0), width=2 * SCALE)
    image = Image.fromarray(average_down(big, SCALE))
    for step in range(6):
        y = (bottom - (bottom - top) * step // 6) // SCALE - 4
        draw_text(image, (4, y), f"{step * 20:3d}", (60, 60, 60), 1)
    for index in range(bars):
        draw_text(image, ((left + index * slot + slot // 3) // SCALE, height - 24), words(rng, 1)[:4], (60, 60, 60), 1)
    return np.asarray(image, dtype=np.uint8)


def coordinates(width: int, height: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """The row and the column of every pixel."""
    grid: npt.NDArray[np.float64] = np.mgrid[0:height, 0:width].astype(np.float64)
    return grid[0], grid[1]


def gradient_image(rng: np.random.Generator, width: int, height: int) -> Pixels:
    y, x = coordinates(width, height)
    # A direction of whole numbers, so that no sine is needed.
    dx, dy = (float(value) for value in rng.integers(-4, 5, size=2))
    if dx == 0.0 and dy == 0.0:
        dx = 1.0
    length = float(np.sqrt(dx * dx + dy * dy))
    reach = (abs(dx) * width + abs(dy) * height) / length
    along = (x * dx + y * dy) / length / reach + 0.5
    start = np.array(random_colour(rng), dtype=np.float64)
    end = np.array(random_colour(rng), dtype=np.float64)
    pixels = start + np.clip(along, 0.0, 1.0)[..., None] * (end - start)
    centre_x, centre_y = float(rng.uniform(0.2, 0.8)) * width, float(rng.uniform(0.2, 0.8)) * height
    radius = np.sqrt((x - centre_x) * (x - centre_x) + (y - centre_y) * (y - centre_y))
    glow = np.clip(1.0 - radius / (0.45 * min(width, height)), 0.0, 1.0)
    glow = glow * glow
    pixels += glow[..., None] * (np.array(random_colour(rng), dtype=np.float64) - pixels) * 0.6
    # A band of a shallow slope, where banding shows, and a sharp horizon.
    band = (y > 0.62 * height) & (y < 0.8 * height)
    level = 118.0 + 10.0 * x / width
    pixels[band] = level[band][..., None] + np.array([0.0, 4.0, 8.0])
    horizon = y > 0.8 * height
    pixels[horizon] = (40.0 + 60.0 * (y[horizon] - 0.8 * height) / (0.2 * height))[..., None] * np.array(
        [0.6, 0.8, 1.0]
    )
    return np.clip(np.rint(pixels), 0, 255).astype(np.uint8)


def bezier(points: npt.NDArray[np.float64], samples: int) -> list[tuple[float, float]]:
    t = np.linspace(0.0, 1.0, samples)[:, None]
    curve = (
        (1 - t) ** 3 * points[0] + 3 * (1 - t) ** 2 * t * points[1] + 3 * (1 - t) * t**2 * points[2] + t**3 * points[3]
    )
    return [(float(px), float(py)) for px, py in curve]


def lineart_image(rng: np.random.Generator, width: int, height: int) -> Pixels:
    big = Image.new("RGB", (width * SCALE, height * SCALE), (255, 255, 255))
    draw = ImageDraw.Draw(big)
    ink = [(0, 0, 0), (0, 0, 0), (0, 0, 0), (60, 30, 20), (20, 40, 120)]
    for _ in range(int(rng.integers(12, 24))):
        points = rng.uniform([0.0, 0.0], [width * SCALE, height * SCALE], size=(4, 2))
        stroke = int(rng.integers(1, 7)) * SCALE // 2
        draw.line(bezier(points, 160), fill=ink[int(rng.integers(0, len(ink)))], width=max(1, stroke))
    for _ in range(int(rng.integers(4, 9))):
        x, y = rng.uniform(0.1, 0.9) * width * SCALE, rng.uniform(0.1, 0.9) * height * SCALE
        radius = float(rng.uniform(10, 60)) * SCALE
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], outline=(0, 0, 0), width=2 * SCALE)
    x0, y0 = int(rng.integers(0, width // 2)) * SCALE, int(rng.integers(0, height // 2)) * SCALE
    for offset in range(0, 60 * SCALE, 5 * SCALE):
        draw.line([(x0 + offset, y0), (x0, y0 + offset)], fill=(0, 0, 0), width=SCALE)
    return average_down(big, SCALE)


def box_blur(pixels: npt.NDArray[np.float64], radius: int) -> npt.NDArray[np.float64]:
    """A box blur of the given radius along both axes, with edges held."""
    result = pixels
    for axis in (0, 1):
        padded = np.concatenate(
            [
                np.repeat(np.take(result, [0], axis=axis), radius, axis=axis),
                result,
                np.repeat(np.take(result, [-1], axis=axis), radius, axis=axis),
            ],
            axis=axis,
        )
        summed = np.cumsum(padded, axis=axis)
        zero = np.zeros_like(np.take(summed, [0], axis=axis))
        summed = np.concatenate([zero, summed], axis=axis)
        size = result.shape[axis]
        upper = np.take(summed, np.arange(2 * radius + 1, 2 * radius + 1 + size), axis=axis)
        lower = np.take(summed, np.arange(0, size), axis=axis)
        result = (upper - lower) / (2 * radius + 1)
    return result


def ui_image(rng: np.random.Generator, width: int, height: int) -> Pixels:
    base = np.full((height, width, 3), 236.0)
    # A panel with a soft shadow under it.
    px0, py0 = int(rng.integers(20, width // 3)), int(rng.integers(40, height // 3))
    px1, py1 = px0 + width // 2, py0 + height // 2
    shadow = np.zeros((height, width))
    shadow[py0 + 6 : py1 + 6, px0 + 6 : px1 + 6] = 1.0
    base *= 1.0 - 0.35 * box_blur(shadow, 5)[..., None]
    base[py0:py1, px0:px1] = 252.0
    # A title bar with a vertical gradient.
    top, bottom = (
        np.array(random_colour(rng, 40, 120), dtype=np.float64),
        np.array(random_colour(rng, 120, 220), dtype=np.float64),
    )
    ramp = np.linspace(0.0, 1.0, 28)[:, None, None]
    base[0:28] = top + ramp * (bottom - top)
    image = Image.fromarray(np.clip(np.rint(base), 0, 255).astype(np.uint8))

    big = image.resize((width * SCALE, height * SCALE), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(big)
    for index in range(int(rng.integers(3, 6))):
        x0 = (px0 + 12 + index * 70) * SCALE
        y0 = (py1 - 34) * SCALE
        if x0 + 60 * SCALE > px1 * SCALE:
            break
        fill = random_colour(rng, 60, 230)
        draw.rounded_rectangle(
            [x0, y0, x0 + 60 * SCALE, y0 + 22 * SCALE], radius=6 * SCALE, fill=fill, outline=(90, 90, 90), width=SCALE
        )
    for index in range(int(rng.integers(3, 7))):
        x, y = (px0 + 16) * SCALE, (py0 + 20 + index * 22) * SCALE
        if y + 12 * SCALE > (py1 - 40) * SCALE:
            break
        draw.rectangle([x, y, x + 12 * SCALE, y + 12 * SCALE], outline=(80, 80, 80), width=SCALE, fill=(255, 255, 255))
        if rng.random() < 0.6:
            draw.line(
                [(x + 2 * SCALE, y + 6 * SCALE), (x + 5 * SCALE, y + 10 * SCALE), (x + 11 * SCALE, y + 2 * SCALE)],
                fill=(20, 120, 40),
                width=2 * SCALE,
            )
    for index in range(int(rng.integers(3, 6))):
        x, y = (width - 40 - index * 36) * SCALE, 6 * SCALE
        draw.ellipse([x, y, x + 16 * SCALE, y + 16 * SCALE], fill=random_colour(rng, 100, 255))
    image = Image.fromarray(average_down(big, SCALE))

    draw_text(image, (8, 9), words(rng, 3)[:30], (255, 255, 255), 1)
    for index in range(int(rng.integers(3, 7))):
        y = py0 + 22 + index * 22
        if y + 12 > py1 - 40:
            break
        draw_text(image, (px0 + 34, y), words(rng, 3)[:24], (30, 30, 30), 1)
    list_x = px1 + 16
    for row in range(0, height - 40, 18):
        if list_x + 60 > width:
            break
        shade = (245, 245, 245) if (row // 18) % 2 == 0 else (225, 232, 245)
        ImageDraw.Draw(image).rectangle([list_x, 34 + row, width - 8, 34 + row + 17], fill=shade)
        draw_text(image, (list_x + 4, 37 + row), words(rng, 2)[:12], (40, 40, 40), 1)
    return np.asarray(image, dtype=np.uint8)


def illustration_image(rng: np.random.Generator, width: int, height: int) -> Pixels:
    big = Image.new("RGB", (width * SCALE, height * SCALE), random_colour(rng, 150, 255))
    draw = ImageDraw.Draw(big)
    for _ in range(int(rng.integers(8, 16))):
        fill = random_colour(rng, 30, 255)
        outline_width = int(rng.integers(1, 5)) * SCALE
        if rng.random() < 0.5:
            cx, cy = rng.uniform(0.0, 1.0) * width * SCALE, rng.uniform(0.0, 1.0) * height * SCALE
            rx, ry = float(rng.uniform(15, 90)) * SCALE, float(rng.uniform(15, 90)) * SCALE
            draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=fill, outline=(0, 0, 0), width=outline_width)
        else:
            # Corners in twelve directions, a few of them left out, at random
            # distances: a star-like polygon without a sine.
            centre = rng.uniform([0.0, 0.0], [width * SCALE, height * SCALE])
            chosen = sorted(int(index) for index in rng.choice(12, size=int(rng.integers(3, 8)), replace=False))
            radii = rng.uniform(20, 100, size=len(chosen)) * SCALE
            polygon = [
                (float(centre[0] + radius * DIRECTIONS[index][0]), float(centre[1] + radius * DIRECTIONS[index][1]))
                for index, radius in zip(chosen, radii, strict=True)
            ]
            draw.polygon(polygon, fill=fill, outline=(0, 0, 0), width=outline_width)
    pixels = average_down(big, SCALE).astype(np.float64)
    # Soft shading across the picture, as an illustration's light.
    y, x = coordinates(width, height)
    light = 0.85 + 0.15 * (x / width + y / height) / 2.0
    return np.clip(np.rint(pixels * light[..., None]), 0, 255).astype(np.uint8)


GENERATORS: Final[dict[str, Callable[[np.random.Generator, int, int], Pixels]]] = {
    "text": text_image,
    "chart": chart_image,
    "gradient": gradient_image,
    "lineart": lineart_image,
    "ui": ui_image,
    "illustration": illustration_image,
}


def grey(pixels: Pixels) -> Pixels:
    """Luma as JFIF defines it, rounded."""
    weights = np.array([299, 587, 114], dtype=np.uint32)
    total = pixels.astype(np.uint32) @ weights
    return ((total + 500) // 1000).astype(np.uint8)


def write_pnm(path: Path, pixels: Pixels) -> None:
    kind = "P5" if pixels.ndim == 2 else "P6"
    header = f"{kind}\n{pixels.shape[1]} {pixels.shape[0]}\n255\n".encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + np.ascontiguousarray(pixels).tobytes())


@dataclass(frozen=True)
class Original:
    split: str
    name: str
    path: Path


def synthesize(out: Path) -> list[Original]:
    made = []
    for category_index, (category, generate) in enumerate(GENERATORS.items()):
        for index in range(IMAGES_PER_CATEGORY):
            split = "tuning" if index < TUNING_PER_CATEGORY else "test"
            seed = 1000 * (category_index + 1) + index
            width, height = SIZES[(category_index + index) % len(SIZES)]
            pixels = generate(np.random.default_rng(seed), width, height)
            name = f"{category}-{index}"
            path = out / split / f"{name}.ppm"
            write_pnm(path, pixels)
            made.append(Original(split, name, path))
            if category in GREY_CATEGORIES:
                grey_path = out / split / f"{name}-grey.pgm"
                write_pnm(grey_path, grey(pixels))
                made.append(Original(split, f"{name}-grey", grey_path))
    return made


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_hashes(originals: list[Original], out: Path, *, record: bool) -> int:
    lines = [f"{sha256(original.path)}  {original.path.relative_to(out).as_posix()}" for original in originals]
    if record:
        HASHES.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        print(f"recorded {len(lines)} hashes in {HASHES.relative_to(ROOT).as_posix()}")
        return 0
    expected = HASHES.read_text(encoding="utf-8").splitlines() if HASHES.is_file() else []
    if lines != expected:
        differing = sorted(set(lines) ^ set(expected))
        print(f"error: {len(differing)} lines differ from {HASHES.relative_to(ROOT).as_posix()}:")
        for line in differing[:20]:
            print(f"  {line}")
        return 1
    print(f"{len(lines)} originals, byte for byte as {HASHES.relative_to(ROOT).as_posix()} lists them")
    return 0


def encoder_version(cjpeg: Path) -> str:
    result = subprocess.run([str(cjpeg), "-version"], capture_output=True, text=True, check=False)
    return (result.stdout + result.stderr).strip().splitlines()[0] if (result.stdout + result.stderr).strip() else "?"


def encode(cjpeg: Path, encoder: str, source: Path, out: Path, extra: list[str]) -> int:
    originals = sorted(path for path in source.rglob("*") if path.suffix in {".ppm", ".pgm"})
    if not originals:
        print(f"error: {source} holds no originals; run synthesize first")
        return 1
    files = []
    for original in originals:
        split = original.parent.name
        samplings = {"grey": None} if original.suffix == ".pgm" else SAMPLINGS
        for quality in QUALITIES:
            for label, sample in samplings.items():
                target = out / encoder / split / f"{original.stem}-q{quality}-{label}.jpg"
                target.parent.mkdir(parents=True, exist_ok=True)
                command = [str(cjpeg), "-quality", str(quality), *extra]
                if sample is not None:
                    command += ["-sample", sample]
                command += ["-outfile", str(target), str(original)]
                subprocess.run(command, check=True)
                files.append(
                    {
                        "file": target.relative_to(out / encoder).as_posix(),
                        "original": original.relative_to(source).as_posix(),
                        "quality": quality,
                        "sampling": label,
                        "sha256": sha256(target),
                    }
                )
    manifest = {
        "encoder": encoder,
        "cjpeg": encoder_version(cjpeg),
        "options": extra,
        "files": files,
    }
    (out / encoder / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(files)} JPEG files by {manifest['cjpeg']} in {(out / encoder).as_posix()}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("synthesize", help="draw the originals")
    make.add_argument("--out", type=Path, default=ROOT / "data" / "synthetic")
    make.add_argument("--record", action="store_true", help="write the hashes instead of checking them")
    write = commands.add_parser("encode", help="write the originals as JPEG with a cjpeg")
    write.add_argument("--cjpeg", type=Path, required=True)
    write.add_argument("--encoder", required=True, help="a name for the encoder, the directory of its files")
    write.add_argument("--source", type=Path, default=ROOT / "data" / "synthetic")
    write.add_argument("--out", type=Path, default=ROOT / "data" / "jpeg")
    write.add_argument("extra", nargs="*", help="further options for cjpeg, after --")
    options = parser.parse_args()

    if options.command == "synthesize":
        originals = synthesize(options.out)
        return check_hashes(originals, options.out, record=options.record)
    return encode(options.cjpeg, options.encoder, options.source, options.out, options.extra)


if __name__ == "__main__":
    sys.exit(main())
