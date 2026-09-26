#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Exact values that the implementations are checked against, in 60- and 70-digit decimals.

    python conformance/references.py [--out conformance/references] [--check]

Each value is computed from its definition in docs/math.md, in Python's decimals and
rationals alone, and written as a pair of doubles, hi and lo: hi is the double nearest
to the value, and lo the double nearest to what remains, so that hi + lo is the value to
about 2^-106 of it. Nothing of any implementation is used: the values are the oracle.

- basis.txt: the entries of the DCT's basis (docs/math.md, 1.1).
- dct.txt: blocks of integers, and their forward DCT; blocks of coefficients, and their
  inverse DCT.
- scale.txt: counts of levels (n0, n1, S) and steps, and the maximum-likelihood Laplace
  scale (2.1).
- shrinkage.txt: values of rho, each a double, and delta / Q = 1/2 - 1/rho + 1/(e^rho - 1)
  at it (2.2).
- centres.txt: levels, steps and scales, and the mean of the bin under the Laplace model
  (2.2).

Each file starts with lines of #, which say what its lines hold. --check compares what
would be written with the files, and fails where they differ.
"""

import argparse
import random
import sys
from collections.abc import Iterable
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).resolve().parent
DIGITS: Final = 70
# pi to 80 digits.
PI: Final = Decimal("3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986")


def pair(value: Decimal) -> str:
    """The value as hi:lo, each the repr of a double: hi nearest to it, lo nearest to what remains."""
    with localcontext() as context:
        context.prec = DIGITS + 20
        high = float(value)
        low = float(value - Decimal(high))
    return f"{high!r}:{low!r}"


def exact_cos(m: int) -> Decimal:
    """cos(pi m / 16) by its Taylor series, to about DIGITS digits."""
    with localcontext() as context:
        context.prec = DIGITS + 10
        x = PI * (m % 32) / 16
        term = total = Decimal(1)
        n = 0
        while abs(term) > Decimal(10) ** -(DIGITS + 5):
            n += 2
            term = -term * x * x / (n * (n - 1))
            total += term
        return +total


def exact_basis() -> list[list[Decimal]]:
    """The basis of the standard: C(k)/2 cos((2n + 1) k pi / 16), C(0) = 1/sqrt(2), C(k) = 1."""
    with localcontext() as context:
        context.prec = DIGITS
        first = (Decimal(1) / 8).sqrt()
        return [[first if k == 0 else exact_cos((2 * n + 1) * k) / 2 for n in range(8)] for k in range(8)]


def basis_lines(basis: list[list[Decimal]]) -> list[str]:
    lines = ["# k n exact: the entry of frequency k at position n"]
    lines += [f"{k} {n} {pair(basis[k][n])}" for k in range(8) for n in range(8)]
    return lines


def dct_lines(basis: list[list[Decimal]]) -> list[str]:
    """Random blocks and the extremes, level-shifted, and blocks of levels times a table."""
    rng = random.Random(12)
    blocks = [[rng.randrange(-128, 128) for _ in range(64)] for _ in range(2)]
    blocks += [[127] * 64, [-128] * 64, [127 if (i // 8 + i % 8) % 2 == 0 else -128 for i in range(64)]]
    lines = ["# forward s0 ... s63: a block of samples, row by row", "# exact c0 ... c63: its DCT, in natural order"]
    with localcontext() as context:
        context.prec = DIGITS
        for block in blocks:
            lines.append("forward " + " ".join(str(value) for value in block))
            exact = [
                sum((basis[v][y] * basis[u][x] * block[8 * y + x] for y in range(8) for x in range(8)), Decimal(0))
                for v in range(8)
                for u in range(8)
            ]
            lines.append("exact " + " ".join(pair(value) for value in exact))
        lines += ["# inverse c0 ... c63: a block of coefficients", "# exact s0 ... s63: its inverse DCT, row by row"]
        rng = random.Random(13)
        table = [16, 11, 10, 16, 24, 40, 51, 61]
        for _ in range(3):
            block = [rng.randrange(-60, 61) // (1 + i // 8 + i % 8) * (table[i % 8] + i // 8) for i in range(64)]
            lines.append("inverse " + " ".join(str(value) for value in block))
            exact = [
                sum((basis[v][y] * basis[u][x] * block[8 * v + u] for v in range(8) for u in range(8)), Decimal(0))
                for y in range(8)
                for x in range(8)
            ]
            lines.append("exact " + " ".join(pair(value) for value in exact))
    return lines


def exact_scale(zeros: int, nonzeros: int, odd_sum: int, step: int) -> Decimal:
    """The maximum-likelihood scale of docs/math.md, 2.1, at 60 digits."""
    with localcontext() as context:
        context.prec = 60
        a = zeros + odd_sum + 2 * nonzeros
        t = 2 * Decimal(odd_sum) / (zeros + Decimal(zeros * zeros + 4 * a * odd_sum).sqrt())
        return Decimal(step) / (-2 * t.ln())


def scale_lines() -> list[str]:
    cases = [
        (3_000_000, 1_000_000, 50_000_000_000, 1),  # D is about 2^73: beyond binary64's integers
        (4_000_000, 1, 1, 255),  # t near 0
        (1, 4_000_000, 60_000_000_000, 2),  # t near 1
        (0, 7, 7, 16),  # every level +-1
    ]
    # Counts of Laplace draws: n0, n1 and S of levels drawn with a fixed seed, by the
    # inverse of the distribution function in decimals, which every platform computes alike.
    rng = random.Random(5)
    for scale, step in ((3, 10), (10, 10), (40, 7), (1, 16), (200, 2), (5000, 1)):
        levels = []
        for _ in range(5000):
            sign = rng.choice((-1, 1))
            with localcontext() as context:
                context.prec = 30
                magnitude = -scale * Decimal(1 - rng.random()).ln() / step
            levels.append(sign * int(magnitude.to_integral_value()))
        nonzero = [abs(level) for level in levels if level != 0]
        cases.append((len(levels) - len(nonzero), len(nonzero), sum(2 * level - 1 for level in nonzero), step))
    lines = ["# n0 n1 S Q exact: the counts, the step, and the maximum-likelihood scale"]
    lines += [f"{n0} {n1} {s} {q} {pair(exact_scale(n0, n1, s, q))}" for n0, n1, s, q in cases if s > 0]
    return lines


def exact_shrinkage(rho: float) -> Decimal:
    """1/2 - 1/rho + 1/(e^rho - 1) at 60 digits, for rho as the double it is."""
    with localcontext() as context:
        context.prec = 60
        r = Decimal(rho)
        return Decimal("0.5") - 1 / r + 1 / (r.exp() - 1)


def geometric(low: int, high: int, count: int) -> Iterable[float]:
    """count doubles from 10^low to 10^high in a geometric progression, each the double nearest to its decimal."""
    with localcontext() as context:
        context.prec = 40
        for index in range(count):
            yield float(Decimal(10) ** (low + Decimal(high - low) * index / (count - 1)))


def shrinkage_lines() -> list[str]:
    rhos = [*geometric(-12, 4, 3001), 0.999999999, 1.0, 1.000000001, 745.0, 800.0]
    lines = ["# rho exact: a double rho, and delta / Q at it"]
    lines += [f"{rho!r} {pair(exact_shrinkage(rho))}" for rho in rhos]
    return lines


def mean_by_antiderivative(lower: Decimal, upper: Decimal, scale: Decimal) -> Decimal:
    """The mean of the density proportional to exp(-c / scale) on [lower, upper], 0 <= lower."""
    near, far = (-lower / scale).exp(), (-upper / scale).exp()
    return (near * (lower + scale) - far * (upper + scale)) / (near - far)


def centre_lines() -> list[str]:
    lines = ["# q Q scale exact: a level, its step, the scale, and the mean of its bin"]
    with localcontext() as context:
        context.prec = 60
        for scale in (0.5, 3.0, 11.0, 150.0, 1e5):
            for level in (-3, -2, -1, 1, 2, 5, 40):
                magnitude = abs(level)
                lower = (magnitude - Decimal("0.5")) * 12
                upper = (magnitude + Decimal("0.5")) * 12
                mean = mean_by_antiderivative(lower, upper, Decimal(scale))
                lines.append(f"{level} 12 {scale!r} {pair(mean if level > 0 else -mean)}")
    return lines


def files() -> dict[str, list[str]]:
    basis = exact_basis()
    return {
        "basis.txt": basis_lines(basis),
        "dct.txt": dct_lines(basis),
        "scale.txt": scale_lines(),
        "shrinkage.txt": shrinkage_lines(),
        "centres.txt": centre_lines(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "references")
    parser.add_argument("--check", action="store_true", help="compare with the files rather than write them")
    options = parser.parse_args()
    failures = 0
    for name, lines in files().items():
        text = "\n".join(lines) + "\n"
        path = options.out / name
        if options.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != text:
                print(f"error: {path.as_posix()} differs from what its definitions give")
                failures += 1
        else:
            options.out.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8", newline="\n")
            print(f"{path.as_posix()}: {len(lines)} lines")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
