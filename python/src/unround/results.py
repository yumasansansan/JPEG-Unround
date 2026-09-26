# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""What the solvers return: the last point, and a history of the objective as they went."""

import time
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from unround import frames
from unround.model import Dual, Primal

__all__ = ["FrameResult", "History", "Recorder", "Result"]

type Array = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True, eq=False)
class History:
    """One entry per record: the iteration, and the values at the iterate after it.

    seconds is the solver's own time up to the record, not counting the records. dual is
    the dual value, -inf where the solver has none; scaling is the theta that made TGV's
    dual feasible (docs/math.md, 6.2), and partial_gap the gap of 6.3, NaN where not taken.
    """

    iterations: npt.NDArray[np.int64]
    seconds: Array
    primal: Array
    dual: Array
    scaling: Array
    partial_gap: Array

    @property
    def gap(self) -> Array:
        """The duality gap of each record: an upper bound of the primal value's distance to its least."""
        return np.asarray(self.primal - self.dual, dtype=np.float64)


@dataclass(frozen=True, slots=True, eq=False)
class Result:
    """The last point of a solver, how many iterations it took, and what it recorded."""

    primal: Primal
    dual: Dual | None
    iterations: int
    converged: bool
    history: History


@dataclass(frozen=True, slots=True, eq=False)
class FrameResult:
    """The last point of a solver of a frame (unround.frames), and the rest as Result's."""

    primal: frames.Primal
    dual: Dual | None
    iterations: int
    converged: bool
    history: History


@dataclass(slots=True)
class Recorder:
    """Collects a History, and the solver's own time between records."""

    iterations: list[int] = field(default_factory=list)
    seconds: list[float] = field(default_factory=list)
    primal: list[float] = field(default_factory=list)
    dual: list[float] = field(default_factory=list)
    scaling: list[float] = field(default_factory=list)
    partial_gap: list[float] = field(default_factory=list)
    elapsed: float = 0.0
    _started: float = 0.0

    def resume(self) -> None:
        """Starts the solver's clock."""
        self._started = time.perf_counter()

    def pause(self) -> None:
        """Stops the solver's clock, before a record."""
        self.elapsed += time.perf_counter() - self._started

    def record(
        self, iteration: int, values: tuple[float, float], scaling: float = np.nan, partial_gap: float = np.nan
    ) -> None:
        """Keeps the primal and dual values of an iteration."""
        self.iterations.append(iteration)
        self.seconds.append(self.elapsed)
        self.primal.append(values[0])
        self.dual.append(values[1])
        self.scaling.append(scaling)
        self.partial_gap.append(partial_gap)

    def history(self) -> History:
        """The records as arrays."""
        return History(
            iterations=np.asarray(self.iterations, dtype=np.int64),
            seconds=np.asarray(self.seconds, dtype=np.float64),
            primal=np.asarray(self.primal, dtype=np.float64),
            dual=np.asarray(self.dual, dtype=np.float64),
            scaling=np.asarray(self.scaling, dtype=np.float64),
            partial_gap=np.asarray(self.partial_gap, dtype=np.float64),
        )
