"""The ratio of TGV's steps on pictures of rounded samples (docs/math.md, 11): the synthetic pictures of
experiments/rounded_stops.py, reconstructed by the reference implementation's command line at each ratio tau / sigma,
stopped at the default gap per sample, 1e-4, against the pictures drawn and against a long run.

    python experiments/rounded_ratios.py [--workers 4] [--ratios 0.1 0.3 1 3 10 30] [--program PATH]
        [--out experiments/results/rounded-ratios.md]

A ratio is judged by the iterations it takes to the stop, the RMS distance of its result from the long run's
(tolerance 0, 20000 iterations, at TGV's ratio on JPEG files, 3), and the RMS error against the picture drawn, in
levels of 8 bits.
"""

import argparse
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from rounded_stops import LONG, ROOT, drawn, rms, solve

RATIOS = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0)


def work(task: tuple[int, tuple[float, ...]]) -> dict[str, Any]:
    seed, ratios = task
    truth = drawn(seed)
    levels = np.floor(truth + 0.5).astype(np.uint8)
    with tempfile.TemporaryDirectory() as scratch:
        folder = Path(scratch)
        png = folder / "rounded.png"
        Image.fromarray(levels).save(png)
        long = solve(png, folder / "long.tif", ["--tolerance", "0", "--iterations", str(LONG), "--step-ratio", "3"])
        result: dict[str, Any] = {"seed": seed, "long_error": rms(long["picture"], truth), "ratios": {}}
        for ratio in ratios:
            run = solve(
                png,
                folder / f"r{ratio:g}.tif",
                ["--tolerance", "1e-4", "--iterations", str(LONG), "--step-ratio", repr(ratio)],
            )
            result["ratios"][ratio] = {
                "iterations": run["iterations"],
                "error": rms(run["picture"], truth),
                "from_long": rms(run["picture"], long["picture"]),
            }
    return result


def medians(label: str, chosen: list[dict[str, Any]], ratios: tuple[float, ...]) -> str:
    """A row of the table: the medians over the pictures chosen."""
    cells = [label, f"{np.median([r['long_error'] for r in chosen]):.4f}"]
    for ratio in ratios:
        found = [r["ratios"][ratio] for r in chosen]
        cells.append(
            f"{np.median([f['iterations'] for f in found]):.0f}, {np.median([f['error'] for f in found]):.4f}, "
            f"{np.median([f['from_long'] for f in found]):.4f}"
        )
    return "| " + " | ".join(cells) + " |"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seeds", type=int, default=6, help="of each set")
    parser.add_argument("--ratios", type=float, nargs="+", default=list(RATIOS))
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "results" / "rounded-ratios.md")
    parser.add_argument("--program", type=Path, help="the command line to run (by default the release build's)")
    arguments = parser.parse_args()
    if arguments.program is not None:
        os.environ["UNROUND_PROGRAM"] = str(arguments.program)
    ratios = tuple(arguments.ratios)
    seeds = [*range(arguments.seeds), *range(100, 100 + arguments.seeds)]
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        results = list(pool.map(work, [(seed, ratios) for seed in seeds]))
    lines = [
        "# The ratio of TGV's steps on pictures of rounded samples",
        "",
        "`experiments/rounded_ratios.py`: the synthetic RGB pictures of `experiments/rounded_stops.py`, 192 x 256,",
        "rounded to 8 bits and reconstructed by TGV (alpha1 = 1, alpha0 = 2, mu = 0, relaxation 1.9) at each ratio of",
        "the steps tau / sigma, stopped at a gap per sample of 1e-4, and by a long run (tolerance 0,",
        f"{LONG} iterations, ratio 3). Seeds below 100 are lively pictures, from 100 on dim ones. RMS in levels of 8",
        "bits.",
        "",
        "| seed | long run's error | "
        + " | ".join(f"ratio {ratio:g}: iterations, error, from long" for ratio in ratios)
        + " |",
        "|---|---|" + "---|" * len(ratios),
    ]
    for result in results:
        cells = [str(result["seed"]), f"{result['long_error']:.4f}"]
        for ratio in ratios:
            found = result["ratios"][ratio]
            cells.append(f"{found['iterations']}, {found['error']:.4f}, {found['from_long']:.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append(medians("median, lively", [r for r in results if r["seed"] < 100], ratios))
    lines.append(medians("median, dim", [r for r in results if r["seed"] >= 100], ratios))
    arguments.out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
