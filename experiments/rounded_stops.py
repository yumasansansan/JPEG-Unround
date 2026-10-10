"""When TGV should stop on pictures of rounded samples (docs/math.md, 11): synthetic pictures of 8 bits, drawn in
binary64 and rounded, reconstructed by the reference implementation's command line at several tolerances of the
gap per sample, against the pictures drawn and against a long run.

    python experiments/rounded_stops.py [--workers 4] [--out experiments/results/rounded-stops.md]

The pictures are smooth -- ramps, radial glows, and soft clouds of low-passed noise -- with sharp-edged shapes in
some, 192 x 256 RGB, from NumPy's generator with fixed seeds; none is a test image. Two sets: lively ones, whose
levels step every few samples, and dim ones, of slow ramps and faint glows in the dark, whose bands are tens of
samples wide, as a dark gradient's are. A stop is judged by the RMS distance of its result from the long run's
(tolerance 0, 20000 iterations), and by the RMS error against the picture drawn, in levels of 8 bits. The stops
are tolerances of the gap per sample, and of the gap relative to the primal value.
"""

import argparse
import json
import os
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import tifffile
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RELEASE = ROOT / "rust" / "target" / "release"
PROGRAM = RELEASE / ("unround.exe" if (RELEASE / "unround.exe").exists() else "unround")
STOPS = (
    ("5e-3", ["--tolerance", "5e-3"]),
    ("1e-3", ["--tolerance", "1e-3"]),
    ("1e-4", ["--tolerance", "1e-4"]),
    ("rel 1e-2", ["--tolerance", "0", "--relative-tolerance", "1e-2"]),
    ("rel 1e-3", ["--tolerance", "0", "--relative-tolerance", "1e-3"]),
    ("rel 1e-4", ["--tolerance", "0", "--relative-tolerance", "1e-4"]),
)
LONG = 20000
HEIGHT, WIDTH = 192, 256


def drawn(seed: int) -> npt.NDArray[np.float64]:
    """A smooth RGB picture in binary64 on 0-255: a ramp, glows, a cloud, and on odd seeds sharp shapes; from seed
    100 on, dim: slow ramps and faint glows in the dark."""
    rng = np.random.default_rng(seed)
    dim = seed >= 100
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float64)
    picture = np.zeros((HEIGHT, WIDTH, 3))
    base = rng.uniform(2.0, 12.0, size=3) if dim else rng.uniform(10.0, 60.0, size=3)
    reach = 0.02 if dim else 0.15
    slope = rng.uniform(-reach, reach, size=(2, 3))
    picture += base + slope[0] * x[..., None] + slope[1] * y[..., None]
    for _ in range(3):
        cy, cx = rng.uniform(0, HEIGHT), rng.uniform(0, WIDTH)
        radius = rng.uniform(60.0, 160.0) if dim else rng.uniform(30.0, 90.0)
        colour = rng.uniform(0.0, 12.0, size=3) if dim else rng.uniform(-40.0, 80.0, size=3)
        picture += colour * np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2.0 * radius**2))[..., None]
    noise = rng.standard_normal((HEIGHT // 16 + 2, WIDTH // 16 + 2, 3))
    cloud = np.asarray(
        Image.fromarray(((noise - noise.min()) / (noise.max() - noise.min()) * 255).astype(np.uint8)).resize(
            (WIDTH + 32, HEIGHT + 32), Image.Resampling.BICUBIC
        ),
        dtype=np.float64,
    )[16 : 16 + HEIGHT, 16 : 16 + WIDTH]
    picture += (cloud - 128.0) * (rng.uniform(0.005, 0.02) if dim else rng.uniform(0.05, 0.15))
    if seed % 2 == 1:
        for _ in range(3):
            top, left = int(rng.integers(0, HEIGHT - 40)), int(rng.integers(0, WIDTH - 60))
            picture[top : top + 40, left : left + 60] += rng.uniform(-60.0, 60.0, size=3)
    return np.clip(picture, 0.0, 255.0)


def solve(png: Path, out: Path, options: list[str]) -> dict[str, Any]:
    report = out.with_suffix(".json")
    program = os.environ.get("UNROUND_PROGRAM", str(PROGRAM))
    subprocess.run(
        [program, "--quiet", "--overwrite", "--report", str(report), *options, str(png), str(out)], check=True
    )
    found = json.loads(report.read_text(encoding="utf-8"))
    return {
        "picture": tifffile.imread(out),
        "iterations": int(found["iterations"]),
        "seconds": found["records"]["seconds"][-1],
    }


def rms(first: npt.NDArray[np.float64], second: npt.NDArray[np.float64]) -> float:
    """The root mean square of the differences."""
    return float(np.sqrt(np.mean((first - second) ** 2)))


def medians(label: str, chosen: list[dict[str, Any]]) -> str:
    """A row of the table: the medians over the pictures chosen."""
    cells = [
        label,
        f"{np.median([r['levels_error'] for r in chosen]):.4f}",
        f"{np.median([r['long_error'] for r in chosen]):.4f}",
    ]
    for name, _ in STOPS:
        found = [r["tolerances"][name] for r in chosen]
        cells.append(
            f"{np.median([f['iterations'] for f in found]):.0f}, {np.median([f['error'] for f in found]):.4f}, "
            f"{np.median([f['from_long'] for f in found]):.4f}"
        )
    return "| " + " | ".join(cells) + " |"


def work(seed: int) -> dict[str, Any]:
    truth = drawn(seed)
    levels = np.floor(truth + 0.5).astype(np.uint8)
    with tempfile.TemporaryDirectory() as scratch:
        folder = Path(scratch)
        png = folder / "rounded.png"
        Image.fromarray(levels).save(png)
        long = solve(png, folder / "long.tif", ["--tolerance", "0", "--iterations", str(LONG)])
        result: dict[str, Any] = {
            "seed": seed,
            "levels_error": rms(levels.astype(np.float64), truth),
            "long_error": rms(long["picture"], truth),
            "long_seconds": long["seconds"],
            "tolerances": {},
        }
        for index, (name, options) in enumerate(STOPS):
            run = solve(png, folder / f"t{index}.tif", [*options, "--iterations", str(LONG)])
            result["tolerances"][name] = {
                "iterations": run["iterations"],
                "error": rms(run["picture"], truth),
                "from_long": rms(run["picture"], long["picture"]),
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seeds", type=int, default=6, help="of each set")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "results" / "rounded-stops.md")
    parser.add_argument("--program", type=Path, help="the command line to run (by default the release build's)")
    arguments = parser.parse_args()
    if arguments.program is not None:
        os.environ["UNROUND_PROGRAM"] = str(arguments.program)
    seeds = [*range(arguments.seeds), *range(100, 100 + arguments.seeds)]
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        results = list(pool.map(work, seeds))
    lines = [
        "# When TGV stops on pictures of rounded samples",
        "",
        "`experiments/rounded_stops.py`: synthetic RGB pictures of 192 x 256, drawn in binary64 and rounded to 8",
        "bits, reconstructed by TGV (the defaults, alpha1 = 1, alpha0 = 2, mu = 0) at each stop -- a tolerance of the",
        "gap per sample, or of the gap relative to the primal value (rel) -- and by a long run (tolerance 0,",
        f"{LONG} iterations). Seeds below 100 are lively pictures, from 100 on dim ones. RMS in levels of 8 bits.",
        "",
        "| seed | levels' error | long run's error | "
        + " | ".join(f"{name}: iterations, error, from long" for name, _ in STOPS)
        + " |",
        "|---|---|---|" + "---|" * len(STOPS),
    ]
    for result in results:
        cells = [str(result["seed"]), f"{result['levels_error']:.4f}", f"{result['long_error']:.4f}"]
        for name, _ in STOPS:
            found = result["tolerances"][name]
            cells.append(f"{found['iterations']}, {found['error']:.4f}, {found['from_long']:.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append(medians("median, lively", [r for r in results if r["seed"] < 100]))
    lines.append(medians("median, dim", [r for r in results if r["seed"] >= 100]))
    arguments.out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
