# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""The primal-dual method: its steps, its iterates within the constraint set, and its gaps.

The iterates are checked against the constraint set within the round trip of the DCT
(rounding.py). How far the gaps fall in a number of iterations is not a theorem but a
measurement on these problems, kept as a check that the method still does as well:
each threshold is about ten times what was measured.
"""

import dataclasses
import math

import numpy as np
import pytest

import rounding
import synthetic
from unround import dct, frames, model, operators, pdhg, subgradient
from unround.model import TGV, TV, Dual, Primal

PARTIAL_RADIUS = 20.0


@pytest.mark.parametrize(
    ("norm_squared", "product"), [(pdhg.TGV_NORM_SQUARED, pdhg.STEP_PRODUCT), (8.0, 0.5), (20.0, 0.9)]
)
def test_the_steps_keep_their_ratio_and_their_product(norm_squared: float, product: float) -> None:
    # Each step is a square root of a product and a quotient: within 3U, and their quotient
    # and product within 8U.
    for ratio in (0.1, 1.0, 37.0):
        tau, sigma = pdhg.steps(norm_squared, ratio, product)
        assert abs(tau / sigma - ratio) <= 8.0 * rounding.U * ratio
        assert abs(tau * sigma * norm_squared - product) <= 8.0 * rounding.U * product


@pytest.mark.parametrize(
    ("norm_squared", "ratio", "product"),
    [
        (8.0, 0.0, 0.99),
        (8.0, math.inf, 0.99),
        (8.0, math.nan, 0.99),
        (0.0, 1.0, 0.99),
        (8.0, 1.0, 1.0),
        (8.0, 1.0, 0.0),
    ],
)
def test_the_steps_are_refused_out_of_their_ranges(norm_squared: float, ratio: float, product: float) -> None:
    with pytest.raises(ValueError, match="positive"):
        pdhg.steps(norm_squared, ratio, product)


def test_the_defaults_scale_with_the_weight() -> None:
    # What is None is the model's default: the ratio over the weight squared, the tolerance
    # times it; what is given is kept.
    tv = pdhg.plan(None, TV(1.0))
    assert (tv.iterations, tv.tolerance, tv.step_ratio) == (pdhg.TV_ITERATIONS, pdhg.TV_TOLERANCE, pdhg.TV_RATIO)
    assert tv.relaxation == pdhg.TV_RELAXATION
    tgv = pdhg.plan(pdhg.Options(), TGV(1.0, 2.0))
    assert (tgv.iterations, tgv.tolerance, tgv.step_ratio) == (pdhg.TGV_ITERATIONS, pdhg.TGV_TOLERANCE, pdhg.TGV_RATIO)
    assert tgv.relaxation == pdhg.TGV_RELAXATION
    scaled = pdhg.plan(None, TV(4.0))
    assert scaled.step_ratio == pdhg.TV_RATIO / 16.0
    assert scaled.tolerance == pdhg.TV_TOLERANCE * 4.0
    given = pdhg.plan(pdhg.Options(iterations=7, tolerance=0.0, step_ratio=2.5, record_every=3), TGV(0.5, 1.0))
    assert (given.iterations, given.tolerance, given.step_ratio, given.record_every) == (7, 0.0, 2.5, 3)
    with pytest.raises(ValueError, match="positive"):
        pdhg.plan(None, TV(0.0))
    # The relaxed steps converge for a relaxation in (0, 2) only.
    assert pdhg.plan(pdhg.Options(relaxation=1.9), TV(1.0)).relaxation == 1.9
    for relaxation in (0.0, 2.0, -1.0):
        with pytest.raises(ValueError, match="relaxation"):
            pdhg.plan(pdhg.Options(relaxation=relaxation), TV(1.0))
    # Without the scaling, the defaults are those of the weight 1, whatever the weight.
    unscaled = pdhg.plan(pdhg.Options(scale_with_weight=False), TV(4.0))
    assert (unscaled.step_ratio, unscaled.tolerance) == (pdhg.TV_RATIO, pdhg.TV_TOLERANCE)
    unscaled = pdhg.plan(pdhg.Options(scale_with_weight=False), TGV(0.5, 1.0))
    assert (unscaled.step_ratio, unscaled.tolerance) == (pdhg.TGV_RATIO, pdhg.TGV_TOLERANCE)


def test_the_other_options_have_their_defaults_and_are_kept() -> None:
    tv, tgv = pdhg.plan(None, TV()), pdhg.plan(None, TGV())
    assert (tv.norm_squared, tgv.norm_squared) == (pdhg.TV_NORM_SQUARED, pdhg.TGV_NORM_SQUARED)
    for settled in (tv, tgv):
        assert (settled.step_product, settled.relative_tolerance, settled.partial_tolerance) == (0.99, 0.0, 0.0)
        assert settled.partial_radius is None
    options = pdhg.Options(
        relative_tolerance=1e-3, partial_tolerance=2e-3, step_product=0.5, norm_squared=12.0, partial_radius=5.0
    )
    given = pdhg.plan(options, TGV())
    assert (given.relative_tolerance, given.partial_tolerance, given.step_product) == (1e-3, 2e-3, 0.5)
    assert (given.norm_squared, given.partial_radius) == (12.0, 5.0)


@pytest.mark.parametrize(
    ("options", "weights", "match"),
    [
        (pdhg.Options(iterations=-1), TV(), "most iterations"),
        (pdhg.Options(tolerance=-1e-3), TV(), "the tolerance"),
        (pdhg.Options(relative_tolerance=-1.0), TV(), "relative tolerance"),
        (pdhg.Options(relative_tolerance=math.nan), TGV(), "relative tolerance"),
        (pdhg.Options(step_ratio=0.0), TV(), "ratio"),
        (pdhg.Options(step_ratio=math.inf), TGV(), "ratio"),
        (pdhg.Options(step_product=1.0), TV(), "product"),
        (pdhg.Options(step_product=0.0), TGV(), "product"),
        (pdhg.Options(norm_squared=0.0), TV(), "L\\^2"),
        (pdhg.Options(norm_squared=math.inf), TGV(), "L\\^2"),
        (pdhg.Options(record_every=-1), TV(), "records"),
        (pdhg.Options(partial_radius=-1.0), TGV(), "radius"),
        (pdhg.Options(partial_tolerance=-1.0, partial_radius=1.0), TGV(), "partial tolerance"),
        (pdhg.Options(partial_tolerance=1e-3), TGV(), "needs the partial gap's radius"),
        (pdhg.Options(partial_radius=1.0), TV(), "TV has no partial gap"),
        (pdhg.Options(partial_tolerance=1e-3), TV(), "TV has no partial gap"),
        (None, TGV(math.nan, 2.0), "first-order"),
        (None, TGV(1.0, 0.0), "second-order"),
        (None, TV(1.0, channel_weights=(1.0, 0.0)), "channel weights"),
        (None, TGV(channel_weights=(math.inf,)), "channel weights"),
        (None, TV(1.0, channel_weights=()), "channel weights"),
        (pdhg.Options(free_radius=-1.0), TV(), "free samples"),
        (pdhg.Options(free_radius=math.inf), TGV(), "free samples"),
    ],
)
def test_the_plan_refuses_what_the_method_cannot_run_with(
    options: pdhg.Options | None, weights: TV | TGV, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        pdhg.plan(options, weights)


def test_the_plan_tells_every_value_out_of_its_range_at_once() -> None:
    with pytest.raises(ValueError, match=r"most iterations.*product.*records"):
        pdhg.plan(pdhg.Options(iterations=-1, step_product=2.0, record_every=-1), TV())


def test_the_steps_are_those_of_the_options() -> None:
    # tau and sigma depend on the product and on L^2 through their quotient alone, and a
    # quarter of the product gives the same steps as four times L^2: dividing by 4 is exact.
    # The iterates are then the same to the last bit; with the default steps they are not.
    problem, _ = synthetic.problem(seed=20)
    base = pdhg.Options(iterations=30, tolerance=0.0, record_every=10)
    quartered = pdhg.solve_tv(problem, TV(), dataclasses.replace(base, step_product=pdhg.STEP_PRODUCT / 4.0))
    widened = pdhg.solve_tv(problem, TV(), dataclasses.replace(base, norm_squared=4.0 * pdhg.TV_NORM_SQUARED))
    default = pdhg.solve_tv(problem, TV(), base)
    np.testing.assert_array_equal(quartered.primal.canvas, widened.primal.canvas)
    np.testing.assert_array_equal(quartered.history.primal, widened.history.primal)
    assert not np.array_equal(quartered.primal.canvas, default.primal.canvas)


def test_the_start_is_the_mmse_decoder() -> None:
    problem, _ = synthetic.problem(seed=11)
    point = pdhg.start(problem)
    np.testing.assert_array_equal(point.coefficients, problem.centres)
    np.testing.assert_array_equal(point.canvas, dct.inverse(problem.centres))


def within_the_set(problem: model.Problem, point: Primal) -> bool:
    """The canvas's coefficients are within their intervals, up to the round trip of the DCT."""
    reach = rounding.roundtrip_error(point.coefficients)
    return bool(np.all(model.excess(problem, dct.forward(point.canvas)) * problem.steps <= reach))


@pytest.mark.parametrize(("ratio", "threshold"), [(1.0, 4e-9), (30.0, 1e-5)])
def test_tv_converges_within_the_constraint_set(ratio: float, threshold: float) -> None:
    # Measured: a gap per sample of 3.7e-10 after 4000 iterations with the ratio 1, and 9.6e-7
    # with 30.
    problem, _ = synthetic.problem(seed=12)
    seen: list[int] = []
    gaps: list[float] = []

    def observe(iteration: int, point: Primal, gap: float) -> None:
        seen.append(iteration)
        gaps.append(gap)
        assert within_the_set(problem, point)

    options = pdhg.Options(iterations=4000, tolerance=0.0, step_ratio=ratio, record_every=200)
    result = pdhg.solve_tv(problem, TV(1.0), options, observe=observe)
    history = result.history
    assert seen == list(range(200, 4001, 200))
    # The observer is given the gap per sample of each record, as the tolerance is compared with.
    assert gaps == list(history.gap[1:] / problem.samples)
    assert np.all(model.excess(problem, result.primal.coefficients) == 0.0)
    assert np.all(np.diff(history.seconds) >= 0.0)
    assert history.gap[-1] / problem.samples < threshold
    # The gap is never below 0 by more than the rounding of the two values, bounded at the last
    # iterate, where it is least.
    assert result.dual is not None
    allowance = rounding.tv_values_error(problem, TV(1.0), result.primal, result.dual)
    assert history.gap[-1] >= -allowance
    assert np.all(history.gap[:-1] > 0.0)


@pytest.mark.parametrize(("relaxation", "threshold"), [(1.5, 3e-5), (1.9, 3e-5)])
def test_relaxed_tv_keeps_to_the_constraint_set(relaxation: float, threshold: float) -> None:
    # Measured: a gap per sample of 3.4e-6 with the relaxation 1.5 and 2.8e-6 with 1.9 after
    # 1000 iterations, against 1.0e-5 without. The current point of a relaxed step may leave
    # the constraint set; what is observed and returned, the proximal steps' outputs, may not.
    problem, _ = synthetic.problem(seed=18)

    def observe(iteration: int, point: Primal, gap: float) -> None:  # noqa: ARG001
        assert within_the_set(problem, point)

    options = pdhg.Options(iterations=1000, tolerance=0.0, step_ratio=30.0, relaxation=relaxation, record_every=100)
    result = pdhg.solve_tv(problem, TV(1.0), options, observe=observe)
    history = result.history
    assert history.gap[-1] / problem.samples < threshold
    assert np.all(model.excess(problem, result.primal.coefficients) == 0.0)
    # The dual returned is a projection's output, within the ball up to rounding: dividing by
    # a norm taken within 2U makes a vector longer by at most 3U, and taking its norm here
    # adds 2U more, to first order.
    assert result.dual is not None
    assert np.all(operators.vector_norms(result.dual.p) <= 1.0 + 6.0 * rounding.U)
    allowance = rounding.tv_values_error(problem, TV(1.0), result.primal, result.dual)
    assert history.gap[-1] >= -allowance
    assert np.all(history.gap[:-1] > 0.0)


@pytest.mark.parametrize(("relaxation", "measured"), [(1.0, 130), (1.9, 80)])
def test_tv_stops_at_its_tolerance(relaxation: float, measured: int) -> None:
    # Measured: 130 iterations unrelaxed, 80 with the relaxation 1.9.
    problem, _ = synthetic.problem(seed=13)
    options = pdhg.Options(iterations=5000, tolerance=0.05, step_ratio=10.0, relaxation=relaxation)
    result = pdhg.solve_tv(problem, TV(1.0), options)
    assert result.converged
    assert result.iterations == measured
    assert result.history.gap[-1] <= 0.05 * problem.samples


def test_tv_stops_at_its_relative_tolerance() -> None:
    # The first record whose gap is within the relative tolerance of the primal value stops
    # the solver; the gap per sample, whose tolerance is 0, stops nothing. The comparisons are
    # the solver's, of the same doubles: exact. (Measured: 280 iterations.)
    problem, _ = synthetic.problem(seed=13)
    options = pdhg.Options(iterations=5000, tolerance=0.0, relative_tolerance=1e-4, step_ratio=10.0)
    result = pdhg.solve_tv(problem, TV(1.0), options)
    history = result.history
    assert result.converged
    assert history.gap[-1] <= 1e-4 * history.primal[-1]
    assert np.all(history.gap[1:-1] > 1e-4 * history.primal[1:-1])


def test_tgv_stops_at_its_partial_tolerance() -> None:
    # As for the relative tolerance, with the partial gap per sample. The radius is more than
    # the largest |w| of the solution (measured: 5.9 after 6000 iterations), and so the partial
    # gap bounds the distance from the least value. (Measured: 380 iterations.)
    problem, _ = synthetic.problem(seed=15)
    options = pdhg.Options(
        iterations=5000, tolerance=0.0, step_ratio=10.0, partial_radius=PARTIAL_RADIUS, partial_tolerance=1e-2
    )
    result = pdhg.solve_tgv(problem, TGV(), options)
    partial = result.history.partial_gap / problem.samples
    assert result.converged
    assert partial[-1] <= 1e-2
    assert np.all(partial[1:-1] > 1e-2)


def test_tgv_converges_within_the_constraint_set() -> None:
    # Measured: a gap per sample of 1.9e-4 after 6000 iterations, from 1.6e-2 after 500, and the
    # scaling of the dual 1. The gaps here are at least 0.07, and the values about 3e3, far
    # above their rounding: a gap below 0 could only be an error of the formulas.
    problem, _ = synthetic.problem(seed=14)
    options = pdhg.Options(iterations=6000, tolerance=0.0, step_ratio=10.0, record_every=500)
    result = pdhg.solve_tgv(problem, TGV(1.0, 2.0), options)
    history = result.history
    assert np.all(history.gap > 0.0)
    assert history.gap[-1] / problem.samples < 2e-3
    assert history.gap[-1] < 0.05 * history.gap[1]
    assert history.scaling[-1] > 0.9
    assert np.all(model.excess(problem, result.primal.coefficients) == 0.0)
    assert within_the_set(problem, result.primal)
    assert result.primal.w is not None
    assert result.dual is not None
    assert result.dual.r is not None


def test_relaxed_tgv_keeps_to_the_constraint_set() -> None:
    # Measured: a gap per sample of 1.1e-7 after 6000 iterations with the relaxation 1.9,
    # against 2.8e-6 without, and the scaling of the dual 1.
    problem, _ = synthetic.problem(seed=19)

    def observe(iteration: int, point: Primal, gap: float) -> None:  # noqa: ARG001
        assert within_the_set(problem, point)

    options = pdhg.Options(iterations=6000, tolerance=0.0, step_ratio=10.0, relaxation=1.9, record_every=500)
    result = pdhg.solve_tgv(problem, TGV(1.0, 2.0), options, observe=observe)
    history = result.history
    assert np.all(history.gap > 0.0)
    assert history.gap[-1] / problem.samples < 1e-6
    assert history.scaling[-1] > 0.9
    assert np.all(model.excess(problem, result.primal.coefficients) == 0.0)


def test_tgv_starts_from_the_field_given() -> None:
    # No iteration: the one record is the objective at the start, with w as given, or the
    # gradient of the start's canvas; and the field given is not changed.
    problem, _ = synthetic.problem(seed=17)
    options = pdhg.Options(iterations=0)
    begin = pdhg.start(problem)
    zero = np.zeros((2, *problem.shape))
    result = pdhg.solve_tgv(problem, TGV(), options, first=pdhg.Initial(w=zero))
    assert result.history.primal[0] == model.tgv_objective(problem, TGV(), dataclasses.replace(begin, w=zero))
    assert result.primal.w is not None
    assert result.primal.w is not zero
    np.testing.assert_array_equal(zero, 0.0)
    default = pdhg.solve_tgv(problem, TGV(), options)
    gradient = operators.grad(begin.canvas)
    assert default.history.primal[0] == model.tgv_objective(problem, TGV(), dataclasses.replace(begin, w=gradient))
    with pytest.raises(ValueError, match="shape"):
        pdhg.solve_tgv(problem, TGV(), options, first=pdhg.Initial(w=np.zeros(problem.shape)))
    with pytest.raises(ValueError, match="TV has no field w"):
        pdhg.solve_tv(problem, TV(), options, first=pdhg.Initial(w=zero))


def test_the_dual_starts_where_given_projected_onto_its_ball() -> None:
    # No iteration: the dual returned is the start, the projection of what was given, which is
    # not changed. One iteration from it goes elsewhere than one from 0.
    problem, _ = synthetic.problem(seed=21)
    rng = np.random.default_rng(26)
    p = rng.normal(0.0, 3.0, size=(2, *problem.shape))
    r = rng.normal(0.0, 3.0, size=(3, *problem.shape))
    kept_p, kept_r = p.copy(), r.copy()
    options = pdhg.Options(iterations=0)
    tv = pdhg.solve_tv(problem, TV(1.5), options, first=pdhg.Initial(p=p))
    assert tv.dual is not None
    np.testing.assert_array_equal(tv.dual.p, operators.project_vectors(p, 1.5))
    tgv = pdhg.solve_tgv(problem, TGV(1.0, 2.0), options, first=pdhg.Initial(p=p, r=r))
    assert tgv.dual is not None
    assert tgv.dual.r is not None
    np.testing.assert_array_equal(tgv.dual.p, operators.project_vectors(p, 1.0))
    np.testing.assert_array_equal(tgv.dual.r, operators.project_tensors(r, 2.0))
    np.testing.assert_array_equal(p, kept_p)
    np.testing.assert_array_equal(r, kept_r)
    one = dataclasses.replace(options, iterations=1)
    from_given = pdhg.solve_tv(problem, TV(1.5), one, first=pdhg.Initial(p=p))
    from_zero = pdhg.solve_tv(problem, TV(1.5), one)
    assert not np.array_equal(from_given.primal.canvas, from_zero.primal.canvas)


def test_a_start_that_the_method_cannot_take_is_refused() -> None:
    problem, _ = synthetic.problem(seed=21)
    options = pdhg.Options(iterations=0)
    vectors, tensors = (2, *problem.shape), (3, *problem.shape)
    with pytest.raises(ValueError, match="TV has no field w or r"):
        pdhg.solve_tv(problem, TV(), options, first=pdhg.Initial(r=np.zeros(tensors)))
    with pytest.raises(ValueError, match="shape"):
        pdhg.solve_tgv(problem, TGV(), options, first=pdhg.Initial(r=np.zeros(vectors)))
    with pytest.raises(ValueError, match="shape"):
        pdhg.solve_tv(problem, TV(), options, first=pdhg.Initial(p=np.zeros(tensors)))
    with pytest.raises(ValueError, match="finite"):
        pdhg.solve_tv(problem, TV(), options, first=pdhg.Initial(p=np.full(vectors, np.nan)))
    with pytest.raises(ValueError, match="finite"):
        pdhg.solve_tgv(problem, TGV(), options, first=pdhg.Initial(w=np.full(vectors, np.inf)))
    with pytest.raises(ValueError, match="coefficients"):
        pdhg.solve_tv(problem, TV(), options, first=pdhg.Initial(coefficients=np.zeros((8, 8))))
    with pytest.raises(ValueError, match="coefficients"):
        pdhg.start(problem, np.full(problem.lower.shape, np.nan))


def test_the_partial_gap_is_recorded_for_tgv() -> None:
    problem, _ = synthetic.problem(seed=15)
    options = pdhg.Options(iterations=20, tolerance=0.0, record_every=10, partial_radius=100.0)
    result = pdhg.solve_tgv(problem, TGV(), options)
    assert result.history.partial_gap.shape == (3,)
    assert np.all(np.isfinite(result.history.partial_gap))
    no_radius = pdhg.solve_tv(problem, TV(), pdhg.Options(iterations=20, tolerance=0.0, record_every=10))
    assert np.all(np.isnan(no_radius.history.partial_gap))


def test_tv_reaches_below_the_subgradient_method() -> None:
    # Both minimize the same objective over the same set, so the primal-dual method's dual
    # value is a lower bound of what the subgradient method can reach (by 11.6, far above
    # rounding). And in as many iterations its primal value goes below the subgradient
    # method's (measured: 3167.97 against 3169.95).
    problem, _ = synthetic.problem(seed=16)
    weights = TV(1.0)
    options = pdhg.Options(iterations=300, tolerance=0.0, step_ratio=10.0, record_every=300)
    primal_dual = pdhg.solve_tv(problem, weights, options)
    stepped = subgradient.solve_tv(problem, weights, subgradient.Options(iterations=300, record_every=300))
    assert primal_dual.history.dual[-1] < stepped.history.primal[-1]
    assert primal_dual.history.primal[-1] < stepped.history.primal[-1]
    assert stepped.history.dual[-1] == -np.inf
    assert isinstance(primal_dual.dual, Dual)


INVARIANT = 1e-4  # in steps: every output's coefficients are within their intervals to this


def test_the_channel_weights_scale_the_bound_of_the_steps() -> None:
    # L^2 is the model's bound times the largest weight squared: 8 x 4 exactly.
    assert pdhg.plan(None, TV(channel_weights=(1.0, 2.0, 0.5))).norm_squared == 32.0
    assert pdhg.plan(None, TGV(channel_weights=(1.0, 1.0, 1.0))).norm_squared == pdhg.TGV_NORM_SQUARED
    assert pdhg.plan(pdhg.Options(norm_squared=5.0), TV(channel_weights=(3.0,))).norm_squared == 5.0
    assert pdhg.plan(None, TV()).free_radius == frames.FREE_RADIUS


def within_the_frame(frame: frames.Frame, point: frames.Primal) -> None:
    """The coefficients are within their intervals exactly, and the canvas's own within the invariant.

    The canvas's are within rounding of the coefficients (test_frames bounds it where the
    step's input is known): some units of 2^-53 of the samples, nine orders of magnitude
    below the invariant.
    """
    for index, channel in enumerate(frame.channels):
        assert np.all(model.excess(channel.problem, point.coefficients[index]) == 0.0)
    for excess in frames.excess(frame, point.canvas):
        assert excess.max() <= INVARIANT


@pytest.mark.parametrize(
    ("ratio", "weights", "threshold"),
    [((2, 2), TV(1.0), 3e-4), ((2, 2), TV(1.0, coupled=False), 2e-3), ((1, 1), TV(1.0), 5e-6)],
)
def test_tv_of_a_colour_frame_converges_within_the_set(ratio: tuple[int, int], weights: TV, threshold: float) -> None:
    # Measured after 4000 iterations: a gap per sample of 2.3e-5 coupled and 1.1e-4 apart with
    # the chroma of 4:2:0, a partial gap of the radius 255 (docs/math.md, 6.6); 4.8e-7 with
    # nothing free, where it is the gap itself. The gaps here are at least 1e-5, and the
    # values about 1e4, far above their rounding; the canvases keep within 85 to 191, in the
    # box of the radius.
    frame, _ = synthetic.colour_frame(20, 30, seed=70, ratio=ratio)

    def observe(iteration: int, point: frames.Primal, gap: float) -> None:  # noqa: ARG001
        within_the_frame(frame, point)

    options = pdhg.Options(iterations=4000, tolerance=0.0, record_every=500, relaxation=1.0, step_ratio=10.0)
    result = pdhg.solve_frame_tv(frame, weights, options, observe=observe)
    history = result.history
    assert np.all(history.gap > 0.0)
    assert history.gap[-1] / frame.samples < threshold
    assert history.gap[-1] < 1e-3 * history.gap[0]
    assert result.dual is not None
    norms = frames.vector_norms(result.dual.p, coupled=weights.coupled)
    assert np.all(norms <= 1.0 + 12.0 * rounding.U)


@pytest.mark.parametrize(("weights", "threshold"), [(TGV(1.0, 2.0), 3e-4), (TGV(1.0, 2.0, coupled=False), 6e-3)])
def test_tgv_of_a_colour_frame_converges_within_the_set(weights: TGV, threshold: float) -> None:
    # Measured after 6000 iterations, with the chroma of 4:2:0: a gap per sample of 2.8e-5
    # coupled and 5.8e-4 apart, and the scaling of the dual 1.
    frame, _ = synthetic.colour_frame(20, 30, seed=70)

    def observe(iteration: int, point: frames.Primal, gap: float) -> None:  # noqa: ARG001
        within_the_frame(frame, point)

    options = pdhg.Options(iterations=6000, tolerance=0.0, record_every=500, relaxation=1.0, step_ratio=10.0)
    result = pdhg.solve_frame_tgv(frame, weights, options, observe=observe)
    history = result.history
    assert np.all(history.gap > 0.0)
    assert history.gap[-1] / frame.samples < threshold
    assert history.scaling[-1] > 0.9
    assert result.primal.w is not None
    assert result.primal.w.shape == (2, 3, *frame.shape)


def test_a_frame_starts_where_given() -> None:
    frame, _ = synthetic.colour_frame(20, 30, seed=71)
    rng = np.random.default_rng(72)
    fields = (2, 3, *frame.shape)
    p = rng.normal(0.0, 3.0, size=fields)
    r = rng.normal(0.0, 3.0, size=(3, 3, *frame.shape))
    options = pdhg.Options(iterations=0)
    tv = pdhg.solve_frame_tv(frame, TV(1.5), options, first=pdhg.FrameInitial(p=p))
    assert tv.dual is not None
    np.testing.assert_array_equal(tv.dual.p, frames.project_vectors(p, 1.5, coupled=True))
    first = pdhg.FrameInitial(p=p, r=r, w=np.zeros(fields))
    tgv = pdhg.solve_frame_tgv(frame, TGV(1.0, 2.0, coupled=False), options, first=first)
    assert tgv.dual is not None
    assert tgv.dual.r is not None
    np.testing.assert_array_equal(tgv.dual.p, frames.project_vectors(p, 1.0, coupled=False))
    np.testing.assert_array_equal(tgv.dual.r, frames.project_tensors(r, 2.0, coupled=False))
    start = frames.start(frame)
    begin = pdhg.solve_frame_tv(frame, TV(), options, first=pdhg.FrameInitial(coefficients=start.coefficients))
    np.testing.assert_array_equal(begin.primal.canvas, start.canvas)
    with pytest.raises(ValueError, match="shape"):
        pdhg.solve_frame_tv(frame, TV(), options, first=pdhg.FrameInitial(p=np.zeros((2, *frame.shape))))
    with pytest.raises(ValueError, match="TV has no field w or r"):
        pdhg.solve_frame_tv(frame, TV(), options, first=pdhg.FrameInitial(w=np.zeros(fields)))
    with pytest.raises(ValueError, match="for each of the 3 components"):
        pdhg.solve_frame_tv(frame, TV(), options, first=pdhg.FrameInitial(coefficients=start.coefficients[:2]))
