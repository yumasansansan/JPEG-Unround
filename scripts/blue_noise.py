#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""The blue-noise matrix of the ordered dither (docs/math.md, 10), and the source files that hold it.

    python scripts/blue_noise.py [--check]

The matrix is 64 x 64 ranks, 0 to 4095, made by Ulichney's void-and-cluster method in
integers: the energy of a place is the sum, over the points of a pattern, of a Gaussian
of their distance on the torus, each value of the Gaussian an integer (2^40 times
exp(-d^2 / (2 sigma^2)), sigma = 3/2, rounded to the nearest from 50 digits), and so is
the same everywhere. Every choice of a least or greatest energy takes the first place in
the order of rows, where several have it. The steps:

1. The initial pattern: 410 points (about a tenth), each put where the energy of those
   before it is least, the first at (0, 0); then, as long as it changes, the point of
   greatest energy is taken out and a point put where the energy is then least.
2. Ranks 409 down to 0: from the initial pattern, the point of greatest energy is taken
   out, and has the rank.
3. Ranks 410 to 4095: from the initial pattern, a point is put where the energy is least,
   and has the rank. (Ulichney's third phase takes the zeros' greatest energy among the
   zeros; on the torus the energies of the zeros and of the points add up to the same
   sum everywhere, so that is this.)

It writes rust/unround/src/blue_noise.rs; --check compares what it would write with the
file, and fails where they differ.
"""

import argparse
import sys
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt

ROOT: Final = Path(__file__).resolve().parent.parent
RUST: Final = ROOT / "rust" / "unround" / "src" / "blue_noise.rs"
SIZE: Final = 64
INITIAL: Final = 410
SCALE: Final = 2**40
# 2 sigma^2, sigma = 3/2.
SPREAD: Final = Decimal(9) / Decimal(2)


def kernel() -> npt.NDArray[np.int64]:
    """The Gaussian of the distance on the torus from (0, 0), in integers."""
    values = np.zeros((SIZE, SIZE), dtype=np.int64)
    with localcontext() as context:
        context.prec = 50
        for row in range(SIZE):
            for column in range(SIZE):
                down, across = min(row, SIZE - row), min(column, SIZE - column)
                value = Decimal(SCALE) * (-(Decimal(down * down + across * across) / SPREAD)).exp()
                values[row, column] = int(value.to_integral_value())
    return values


class Pattern:
    """Points on the torus and the energy that they give every place."""

    def __init__(self, gaussian: npt.NDArray[np.int64]) -> None:
        self.gaussian = gaussian
        self.points = np.zeros((SIZE, SIZE), dtype=bool)
        self.energy = np.zeros((SIZE, SIZE), dtype=np.int64)

    def copy(self) -> Pattern:
        other = Pattern(self.gaussian)
        other.points = self.points.copy()
        other.energy = self.energy.copy()
        return other

    def put(self, place: int) -> None:
        row, column = divmod(place, SIZE)
        self.points[row, column] = True
        self.energy += np.roll(self.gaussian, (row, column), axis=(0, 1))

    def take(self, place: int) -> None:
        row, column = divmod(place, SIZE)
        self.points[row, column] = False
        self.energy -= np.roll(self.gaussian, (row, column), axis=(0, 1))

    def tightest(self) -> int:
        """The first point of greatest energy."""
        return int(np.argmax(np.where(self.points, self.energy, -1)))

    def emptiest(self) -> int:
        """The first place without a point of least energy."""
        return int(np.argmin(np.where(self.points, np.iinfo(np.int64).max, self.energy)))


def matrix() -> npt.NDArray[np.int64]:
    """The ranks, row by row."""
    initial = Pattern(kernel())
    for _ in range(INITIAL):
        initial.put(initial.emptiest())
    for _ in range(SIZE * SIZE * 16):
        out = initial.tightest()
        initial.take(out)
        into = initial.emptiest()
        initial.put(into)
        if into == out:
            break
    else:
        message = "the initial pattern did not settle"
        raise RuntimeError(message)
    ranks = np.full(SIZE * SIZE, -1, dtype=np.int64)
    pattern = initial.copy()
    for rank in range(INITIAL - 1, -1, -1):
        place = pattern.tightest()
        pattern.take(place)
        ranks[place] = rank
    pattern = initial.copy()
    for rank in range(INITIAL, SIZE * SIZE):
        place = pattern.emptiest()
        pattern.put(place)
        ranks[place] = rank
    if sorted(ranks.tolist()) != list(range(SIZE * SIZE)):
        message = "the ranks are not each of 0 to 4095 once"
        raise RuntimeError(message)
    return ranks


def rust_source(ranks: npt.NDArray[np.int64]) -> str:
    lines = [
        "// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>",
        "// SPDX-License-Identifier: GPL-3.0-or-later",
        "",
        "//! The ranks of the blue-noise matrix of the ordered dither, 64 x 64, row by row",
        "//! (docs/math.md, 10). Written by `scripts/blue_noise.py`, which makes them by",
        "//! Ulichney's void-and-cluster method in integers; not to be edited.",
        "",
        "/// Rank `64 i + j` is that of row `i`, column `j`: each of 0 to 4095 once.",
        "#[rustfmt::skip]",
        "pub(crate) const RANKS: [u16; 4096] = [",
    ]
    for start in range(0, SIZE * SIZE, 16):
        values = ", ".join(f"{int(value):4}" for value in ranks[start : start + 16])
        lines.append(f"    {values},")
    lines.append("];")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="compare with the file instead of writing it")
    arguments = parser.parse_args()
    source = rust_source(matrix())
    if arguments.check:
        if not RUST.exists() or RUST.read_text(encoding="utf-8") != source:
            print(f"error: {RUST} is not what scripts/blue_noise.py writes", file=sys.stderr)
            return 1
        print(f"{RUST.name}: as written")
        return 0
    RUST.write_text(source, encoding="utf-8", newline="\n")
    print(f"wrote {RUST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
