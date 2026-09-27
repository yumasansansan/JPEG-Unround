# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""The settings of a reconstruction: the method, the model, and the solvers' options.

Each field is an option of the command line (docs/cli.md), under the name that
unround.native.options gives it, and the library checks their ranges. Where a value
is None, the library takes the model's default (docs/math.md, 6.5), which follows
the weight of the first-order term unless scale_with_weight is False. The defaults
of these classes are the library's own: a Settings() gives the options the library
holds of none at all.
"""

from dataclasses import dataclass, field
from typing import Literal

__all__ = ["TGV", "TV", "Centres", "DataTerm", "Method", "PdhgOptions", "Settings", "SubgradientOptions"]

type Method = Literal["mmse", "tv", "tgv", "subgradient"]
type Centres = Literal["mmse", "midpoint"]


@dataclass(frozen=True, slots=True)
class DataTerm:
    """The options of G: the weights and the centres of the data term, and the intervals' slack.

    mu weights the data term; where it is None, it is mu_scale times the mean of the
    component's 64 steps to the power mu_power, which follows the quantization, and in the
    chroma of a file in YCbCr (Cb and Cr) the scale is mu_chroma times mu_scale; a mu that is
    given is every component's. slack widens every interval by that many steps on each side
    (docs/math.md, 1.2); slack_cost is what leaving the file's own interval costs, within the
    slack, per step (0: nothing). The weight of an AC coefficient is mu / Q^power, and that of
    DC dc_weight times it (4.1). centres are the data term's: "mmse", the MMSE centres of the
    Laplace model (2.2), or "midpoint", the middles of the intervals, q Q. The defaults are
    those of docs/math.md, 4.1.
    """

    mu: float | None = None
    mu_scale: float = 9.0
    mu_power: float = 0.9
    mu_chroma: float = 0.3
    slack: float = 0.0
    dc_weight: float = 1.0
    centres: Centres = "midpoint"
    power: float = 2.0
    slack_cost: float = 0.0


@dataclass(frozen=True, slots=True)
class TV:
    """The weight of total variation (docs/math.md, 4.2), and how it takes several channels (4.4).

    channel_weights are the gamma of the channels' differences, 1 for each where None;
    coupled takes the channels together, pixel by pixel, or each on its own. Neither
    matters to one channel.
    """

    alpha: float = 1.0
    channel_weights: tuple[float, ...] | None = None
    coupled: bool = True


@dataclass(frozen=True, slots=True)
class TGV:
    """The weights of second-order total generalized variation (docs/math.md, 4.3 and 4.4).

    alpha1 weights the first-order part, ||grad x - w||, and alpha0 the second, ||E w||.
    channel_weights and coupled are those of TV.
    """

    alpha1: float = 1.0
    alpha0: float = 2.0
    channel_weights: tuple[float, ...] | None = None
    coupled: bool = True


@dataclass(frozen=True, slots=True)
class PdhgOptions:
    """When the primal-dual method stops, its steps, and how often it records (docs/math.md, 5, 6.4 and 6.6).

    iterations is the most iterations. The solver stops earlier, at the first record where
    one of these holds, each off at 0: the duality gap per sample is at most tolerance; the
    gap is at most relative_tolerance times the primal value; or, for TGV, the partial gap
    of the radius partial_radius (6.3) is at most partial_tolerance per sample. step_ratio
    is tau / sigma, relaxation the rho of the relaxed steps, in (0, 2), step_product
    sigma tau L^2, in (0, 1), and norm_squared the L^2 of the steps, a bound of ||K||^2
    (5): below ||K||^2 the method need not converge. Every record_every iterations (0:
    none but the last), and after the last, the solver records the primal and dual values,
    and checks the tolerances; with partial_radius, which only TGV takes, it records the
    partial gap too. Where a frame has free samples, every gap is the partial gap of
    free_radius, R of 6.6.
    """

    iterations: int | None = None
    tolerance: float | None = None
    relative_tolerance: float = 0.0
    partial_tolerance: float = 0.0
    step_ratio: float | None = None
    relaxation: float | None = None
    step_product: float = 0.99
    norm_squared: float | None = None
    scale_with_weight: bool = True
    record_every: int = 10
    partial_radius: float | None = None
    free_radius: float = 255.0


@dataclass(frozen=True, slots=True)
class SubgradientOptions:
    """How many iterations the subgradient method takes, its steps, and how often it records (docs/math.md, 7).

    The step after n iterations moves the canvas by step sqrt(N) / (1 + n)^decay along the
    normalized subgradient, N the number of samples, and momentum turns FISTA's
    extrapolation on. Every record_every iterations (0: none but the last), and after the
    last, the method records the objective.
    """

    iterations: int = 50
    record_every: int = 1
    step: float = 0.5
    decay: float = 0.5
    momentum: bool = True


@dataclass(frozen=True, slots=True)
class Settings:
    """The method, the model, and the solvers' options.

    method is "mmse", the MMSE decoder whatever centres the data term takes (docs/math.md,
    2.3), "tv" or "tgv" by the primal-dual method (5), or "subgradient", TV by a method of jpeg2png's kind
    (7), for greyscale files, to compare with. data are the options of G, one for every
    component or one for each; tv and tgv are the weights of the models, with how they take
    the channels; and pdhg and subgradient the options of the solvers.
    """

    method: Method = "tgv"
    data: DataTerm | tuple[DataTerm, ...] = field(default_factory=DataTerm)
    tv: TV = field(default_factory=TV)
    tgv: TGV = field(default_factory=TGV)
    pdhg: PdhgOptions = field(default_factory=PdhgOptions)
    subgradient: SubgradientOptions = field(default_factory=SubgradientOptions)
