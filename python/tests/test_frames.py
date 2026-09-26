# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Components on one canvas (docs/math.md, 1.3, 4.4 and 6.6).

The maps between the canvas and a component are checked in rational numbers, where
their identities hold exactly. The proximal map and the projection are checked
against their formulas within bounds of their rounding, computed from the arrays at
hand in exact arithmetic. One component whose canvas is its blocks is the problem of
unround.model, to the last bit.
"""

from fractions import Fraction

import numpy as np
import numpy.typing as npt
import pytest

import rounding
import synthetic
from unround import dct, frames, model, operators, pdhg
from unround.model import TGV, TV, DataTerm, Dual, Primal

type Array = npt.NDArray[np.float64]
type Objects = npt.NDArray[np.object_]

U = Fraction(1, 2**53)
RATIOS = [(1, 1), (1, 2), (2, 1), (2, 2), (1, 4)]  # the chroma's cells of 4:4:4, 4:2:2, 4:4:0, 4:2:0 and 4:1:1


def exact(values: npt.ArrayLike) -> Objects:
    """The array's doubles as Python Fractions, exactly."""
    array = np.asarray(values, dtype=np.float64)
    result = np.empty(array.shape, dtype=object)
    for index in np.ndindex(*array.shape):
        result[index] = Fraction(float(array[index]))
    return result


def rationals(shape: tuple[int, ...], seed: int) -> Objects:
    rng = np.random.default_rng(seed)
    numerators = rng.integers(-1000, 1001, size=shape)
    denominators = rng.integers(1, 97, size=shape)
    result = np.empty(shape, dtype=object)
    for index in np.ndindex(*shape):
        result[index] = Fraction(int(numerators[index]), int(denominators[index]))
    return result


def total(values: Objects) -> Fraction:
    return sum((Fraction(value) for value in values.ravel()), Fraction(0))


def gamma(n: int) -> Fraction:
    return n * U / (1 - n * U)


def blank(shape: tuple[int, int]) -> model.Problem:
    """A problem of zero levels over blocks of this shape."""
    return model.make_problem(np.zeros((shape[0] // 8, shape[1] // 8, 8, 8)), np.ones((8, 8)))


@pytest.mark.parametrize(
    ("size", "factors", "shape", "blocks"),
    [
        ((21, 30), ((2, 2), (1, 1), (1, 1)), (32, 32), ((24, 32), (16, 16), (16, 16))),
        ((21, 30), ((2, 1), (1, 1), (1, 1)), (24, 32), ((24, 32), (24, 16), (24, 16))),
        ((20, 30), ((1, 2), (1, 1), (1, 1)), (32, 32), ((24, 32), (16, 32), (16, 32))),
        ((20, 30), ((1, 1), (1, 1), (1, 1)), (24, 32), ((24, 32), (24, 32), (24, 32))),
        ((20, 30), ((4, 1), (1, 1), (1, 1)), (24, 32), ((24, 32), (24, 8), (24, 8))),
    ],
)
def test_the_canvas_is_the_picture_in_whole_mcus(
    size: tuple[int, int],
    factors: tuple[tuple[int, int], ...],
    shape: tuple[int, int],
    blocks: tuple[tuple[int, int], ...],
) -> None:
    frame = frames.make_frame([blank(own) for own in blocks], factors, size)
    assert frame.shape == shape
    most_across, most_down = max(f[0] for f in factors), max(f[1] for f in factors)
    for channel, (across, down) in zip(frame.channels, factors, strict=True):
        assert channel.ratio == (most_down // down, most_across // across)
        rows, columns = channel.extent
        assert rows <= shape[0]
        assert columns <= shape[1]
    assert frame.samples == 3 * shape[0] * shape[1]


def test_frames_that_are_not_whole_are_refused() -> None:
    with pytest.raises(ValueError, match="do not divide"):
        frames.make_frame([blank((24, 32))] * 3, [(3, 1), (2, 1), (2, 1)], (20, 30))
    with pytest.raises(ValueError, match="where a picture of"):
        frames.make_frame([blank((24, 32)), blank((24, 32)), blank((16, 16))], [(2, 2), (1, 1), (1, 1)], (21, 30))
    with pytest.raises(ValueError, match="a sampling factor for each"):
        frames.make_frame([blank((24, 32))] * 3, [(1, 1)] * 2, (20, 30))
    with pytest.raises(ValueError, match="do not tile"):
        frames.Frame(channels=(frames.Channel(blank((8, 8)), (3, 1)),), shape=(16, 16))
    with pytest.raises(ValueError, match="go beyond"):
        frames.Frame(channels=(frames.Channel(blank((16, 16)), (2, 1)),), shape=(16, 16))
    with pytest.raises(ValueError, match="a channel at least"):
        frames.Frame(channels=(), shape=(16, 16))
    # One component is its blocks, whatever its sampling factors.
    single = frames.make_frame([blank((24, 32))], [(2, 2)], (20, 30))
    assert single.shape == (24, 32)
    assert not single.free(single.channels[0])


@pytest.mark.parametrize("ratio", RATIOS)
def test_the_maps_of_a_component_are_exact(ratio: tuple[int, int]) -> None:
    # On a canvas of 16 x 32 and a component of one block (8 x 8 samples), whose cells cover
    # 8 r_v x 8 r_h: sums and spread are adjoint, the means of what spread spreads are what it
    # was given, and Pi = spread(means) is an orthogonal projection.
    frame = frames.Frame(channels=(frames.Channel(blank((8, 8)), ratio),), shape=(16, 32))
    channel = frame.channels[0]
    x, z = rationals((16, 32), 1), rationals((16, 32), 2)
    y = rationals((8, 8), 3)
    assert total(frames.spread(frame, channel, y) * x) == total(y * frames.sums(channel, x))
    np.testing.assert_array_equal(frames.means(channel, frames.spread(frame, channel, y)), y)

    def pi(canvas: Objects) -> Objects:
        return frames.spread(frame, channel, frames.means(channel, canvas))

    np.testing.assert_array_equal(pi(pi(x)), pi(x))
    assert total(pi(x) * z) == total(x * pi(z))
    # <xi, x> splits into the sums and the means, and what Pi leaves (docs/math.md, 6.6).
    assert total(z * x) == total(frames.sums(channel, z) * frames.means(channel, x)) + total((z - pi(z)) * (x - pi(x)))


def cells(channel: frames.Channel, canvas: Objects) -> Objects:
    """The canvas's blocks as cells: shape (rows, r_v, columns, r_h)."""
    rows, columns = channel.problem.shape
    down, across = channel.ratio
    return canvas[: rows * down, : columns * across].reshape(rows, down, columns, across)


def check_cells(channel: frames.Channel, before: Array, after: Array, samples: Array) -> None:
    """after is before moved by the change to samples of its cells' means: checked in exact arithmetic.

    after = fl(before + c), c = fl(samples - m), m the means of before's cells, as unround.frames
    computes them. Each after_i is within U / (1 - U) |after_i| of before_i + c, so the
    deviations from the cells' exact means move by at most that and its cell's mean; and the
    exact mean of a cell of after is within gamma(n) mean|before| (the rounding of m) plus
    U |samples - m| (that of c) plus U / (1 - U) mean|after| of the sample.
    """
    n = channel.cells
    old, new = cells(channel, exact(before)), cells(channel, exact(after))
    own = exact(frames.means(channel, before))[:, np.newaxis, :, np.newaxis]
    target = exact(samples)[:, np.newaxis, :, np.newaxis]
    old_means = old.sum(axis=(1, 3), keepdims=True) / n
    new_means = new.sum(axis=(1, 3), keepdims=True) / n
    size = np.vectorize(abs, otypes=[object])
    after_size = size(new)
    moved = size((new - new_means) - (old - old_means))
    allowed = U / (1 - U) * (after_size + after_size.sum(axis=(1, 3), keepdims=True) / n)
    assert np.all(moved <= allowed)
    reach = gamma(n) * size(old).sum(axis=(1, 3), keepdims=True) / n
    reach = reach + U * size(target - own) + U / (1 - U) * after_size.sum(axis=(1, 3), keepdims=True) / n
    assert np.all(size(new_means - target) <= reach)


@pytest.mark.parametrize("tau", [0.5, 40.0])
@pytest.mark.parametrize("ratio", [(2, 2), (1, 2), (1, 1)])
def test_the_proximal_map_moves_the_cells_and_keeps_the_free_samples(ratio: tuple[int, int], tau: float) -> None:
    # prox_{tau G}(v) is characterized by two conditions (docs/math.md, 4.4): its part that Pi
    # leaves is v's, and its coefficients are the proximal map of unround.model of v's, with
    # the step tau / n, whose optimality test_model checks. The picture of 20 x 30 leaves Y's
    # last block row free where MCUs are 16 rows.
    frame, truth = synthetic.colour_frame(20, 30, seed=40, ratio=ratio, data=DataTerm(mu=5.0))
    rng = np.random.default_rng(41)
    v = truth + rng.normal(0.0, 30.0, size=truth.shape)
    coefficients, canvas = frames.prox(frame, v, tau)
    for index, channel in enumerate(frame.channels):
        own = frames.means(channel, v[index])
        expected = model.prox(channel.problem, dct.forward(own), tau / channel.cells)
        np.testing.assert_array_equal(coefficients[index], expected)
        rows, columns = channel.extent
        beyond = np.ones(frame.shape, dtype=bool)
        beyond[:rows, :columns] = False
        np.testing.assert_array_equal(canvas[index][beyond], v[index][beyond])
        samples = dct.inverse(expected)
        if channel.cells == 1:
            np.testing.assert_array_equal(canvas[index][:rows, :columns], samples)
        else:
            check_cells(channel, v[index], canvas[index], samples)


def within(frame: frames.Frame, canvas: Array, coefficients: tuple[Array, ...], before: Array) -> list[Array]:
    """A bound, coefficient by coefficient, of |D(means of canvas) - coefficients| as dct.forward computes it.

    canvas was made from before by frames._put with these coefficients. Its cells' means,
    computed, are within gamma(n) mean|canvas| of their exact means, which are within the
    bound of check_cells of the inverse DCT of the coefficients: together within
    (gamma(n) + 4U) (mean|before| + mean|canvas|) + 2U |samples - m|, the means of the
    magnitudes computed within gamma(n) and so taken 8U larger, and the whole 8U larger for
    its own rounding. Where the cells are single samples, the means are the inverse DCT
    itself. That is within inverse_error of D^T times the coefficients; the DCT carries it
    through |C| <= (1 + U) |B|, and rounds within forward_error.
    """
    basis = np.abs(dct.BASIS) * (1.0 + 2.0 * rounding.U)
    bounds = []
    for index, channel in enumerate(frame.channels):
        n = channel.cells
        means = frames.means(channel, canvas[index])
        samples = dct.inverse(coefficients[index])
        if n > 1:
            own = frames.means(channel, before[index])
            average = np.abs(cells(channel, before[index])).mean(axis=(1, 3)) * (1.0 + 8.0 * rounding.U)
            after = np.abs(cells(channel, canvas[index])).mean(axis=(1, 3)) * (1.0 + 8.0 * rounding.U)
            spread = (rounding.gamma(n) + 4.0 * rounding.U) * (average + after)
            spread = (spread + 2.0 * rounding.U * np.abs(samples - own)) * (1.0 + 8.0 * rounding.U)
        else:
            spread = np.zeros(means.shape)
        error = spread + rounding.inverse_error(coefficients[index])
        carried = basis @ dct.blocks(error) @ basis.T
        bounds.append(np.asarray(rounding.forward_error(means) + carried, dtype=np.float64))
    return bounds


@pytest.mark.parametrize("ratio", [(2, 2), (1, 2), (1, 1)])
def test_the_projection_lands_in_the_set_and_is_idempotent(ratio: tuple[int, int]) -> None:
    frame, truth = synthetic.colour_frame(20, 30, seed=42, ratio=ratio)
    rng = np.random.default_rng(43)
    v = truth + rng.normal(0.0, 30.0, size=truth.shape)
    once = frames.project(frame, v)
    clipped = tuple(
        model.clip(channel.problem, dct.forward(frames.means(channel, v[index])))
        for index, channel in enumerate(frame.channels)
    )
    reach = within(frame, once, clipped, v)
    for index, channel in enumerate(frame.channels):
        excess = frames.excess(frame, once)[index] * channel.problem.steps
        assert np.all(excess <= reach[index])
    # A second projection moves the canvas by no more than the first left its coefficients
    # outside their intervals, carried back through the inverse DCT, with the rounding of
    # both inverse DCTs and of the sums.
    twice = frames.project(frame, once)
    basis = np.abs(dct.BASIS) * (1.0 + 2.0 * rounding.U)
    for index, channel in enumerate(frame.channels):
        again = model.clip(channel.problem, dct.forward(frames.means(channel, once[index])))
        moved = dct.from_blocks(basis.T @ reach[index] @ basis)
        moved = moved + rounding.inverse_error(again) + rounding.inverse_error(clipped[index])
        means = frames.means(channel, once[index])
        drift = np.abs(means - dct.inverse(clipped[index])) if channel.cells > 1 else np.zeros(means.shape)
        allowed = frames.spread(frame, channel, (moved + drift) * (1.0 + 4.0 * rounding.U))
        allowed = allowed + 2.0 * rounding.U * np.abs(twice[index])
        assert np.all(np.abs(twice[index] - once[index]) <= allowed)


def test_one_component_is_the_problem_of_the_model_to_the_last_bit() -> None:
    problem, canvas = synthetic.problem(seed=44)
    frame = frames.one(problem)
    rng = np.random.default_rng(45)
    v = canvas + rng.normal(0.0, 20.0, size=canvas.shape)
    coefficients, out = frames.prox(frame, v[np.newaxis], 0.7)
    expected = model.prox(problem, dct.forward(v), 0.7)
    np.testing.assert_array_equal(coefficients[0], expected)
    np.testing.assert_array_equal(out[0], dct.inverse(expected))
    np.testing.assert_array_equal(frames.project(frame, v[np.newaxis])[0], model.project(problem, v))
    np.testing.assert_array_equal(frames.start(frame).canvas[0], pdhg.start(problem).canvas)
    c = model.clip(problem, dct.forward(v))
    x = dct.inverse(c)
    w = operators.grad(x) + rng.normal(size=(2, *problem.shape))
    grey = Primal(coefficients=c, canvas=x, w=w)
    point = frames.Primal(coefficients=(c,), canvas=x[np.newaxis], w=w[:, np.newaxis])
    p = operators.project_vectors(rng.normal(0.0, 2.0, size=(2, *problem.shape)), 1.5)
    r = operators.project_tensors(rng.normal(0.0, 3.0, size=(3, *problem.shape)), 2.0)
    assert frames.tv_values(frame, TV(1.5), point, Dual(p[:, np.newaxis])) == model.tv_values(
        problem, TV(1.5), grey, Dual(p)
    )
    lifted = Dual(p[:, np.newaxis], r[:, np.newaxis])
    assert frames.tgv_values(frame, TGV(1.5, 2.0), point, lifted) == model.tgv_values(
        problem, TGV(1.5, 2.0), grey, Dual(p, r)
    )
    partial = frames.tgv_objective(frame, TGV(1.5, 2.0), point) + frames.conjugate(frame, operators.div(lifted.p))
    partial += 3.0 * frames.tgv_residual(frame, TGV(1.5, 2.0), lifted)
    assert partial == model.tgv_partial_gap(problem, TGV(1.5, 2.0), grey, Dual(p, r), 3.0)


def test_the_channels_coupled_and_apart() -> None:
    # Apart, the TV of the channels is the sum of their TVs of unround.model, operation for
    # operation; coupled, it is less, by far more than rounding, the channels' edges not all
    # being at the same pixels.
    frame, _ = synthetic.colour_frame(20, 30, seed=46)
    point = frames.start(frame)
    gradient = operators.grad(point.canvas)
    apart = frames.total(frames.vector_norms(gradient, coupled=False))
    each = [model.total_variation(point.canvas[index]) for index in range(3)]
    assert apart == (each[0] + each[1]) + each[2]
    coupled = frames.total(frames.vector_norms(gradient, coupled=True))
    assert coupled < 0.99 * apart
    data = frames.data_term(frame, point.coefficients)
    assert frames.tv_objective(frame, TV(1.5, coupled=False), point) == 1.5 * apart + data
    assert frames.tv_objective(frame, TV(1.5), point) == 1.5 * coupled + data
    # A projection onto the coupled balls: the norm over the channels at a pixel, computed
    # within (C + 1) U, the quotient by the radius within U more, the division within U, and
    # the test's own norm within (C + 1) U: (2 C + 4) U, and 2 U of slack.
    rng = np.random.default_rng(47)
    p = rng.normal(0.0, 2.0, size=(2, 3, *frame.shape))
    projected = frames.project_vectors(p, 1.5, coupled=True)
    assert np.all(frames.vector_norms(projected, coupled=True) <= 1.5 * (1.0 + 12.0 * rounding.U))
    r = rng.normal(0.0, 2.0, size=(3, 3, *frame.shape))
    projected_r = frames.project_tensors(r, 1.5, coupled=True)
    assert np.all(frames.tensor_norms(projected_r, coupled=True) <= 1.5 * (1.0 + 14.0 * rounding.U))
    inside = frames.vector_norms(p, coupled=True) <= 1.5
    np.testing.assert_array_equal(projected[:, :, inside], p[:, :, inside])


def test_the_channel_weights_scale_each_channel() -> None:
    # Doubling every difference is exact, and so doubles the norms and their sum to the last
    # bit; weights of their own weight each channel's norms apart.
    frame, _ = synthetic.colour_frame(20, 30, seed=48)
    point = frames.start(frame)
    data = frames.data_term(frame, point.coefficients)
    gradient = operators.grad(point.canvas)
    plain = frames.total(frames.vector_norms(gradient, coupled=True))
    assert frames.tv_objective(frame, TV(1.0, channel_weights=(2.0, 2.0, 2.0)), point) == 2.0 * plain + data
    gammas = np.array([1.0, 0.5, 0.25])[:, np.newaxis, np.newaxis]
    weighted = frames.total(frames.vector_norms(gammas * gradient, coupled=False))
    assert frames.tv_objective(frame, TV(1.0, channel_weights=(1.0, 0.5, 0.25), coupled=False), point) == (
        weighted + data
    )
    with pytest.raises(ValueError, match="each of the 3 channels"):
        frames.channel_weights(frame, (1.0, 1.0))
    with pytest.raises(ValueError, match="each of the 3 channels"):
        frames.channel_weights(frame, (1.0, 0.0, 1.0))


def test_the_bound_of_the_free_samples() -> None:
    # G* of a frame is unround.model's G* of the cells' sums, plus, where samples are free,
    # <zeta, m> + R ||zeta||_1 (docs/math.md, 6.6): with the radius, it grows by R times the
    # size of what Pi leaves of xi, and where nothing is free it is unround.model's.
    # Each component's term is its G* plus 128 sum(zeta beyond the blocks) plus R sum|zeta|,
    # each sum within gamma(N) of the sum of its terms' magnitudes, and the three terms and
    # the channels added with a rounding each: the difference of two radii is R times the
    # sums of |zeta|, within gamma(N + 8) of twice all those magnitudes.
    frame, _ = synthetic.colour_frame(20, 30, seed=49)
    rng = np.random.default_rng(50)
    xi = rng.normal(0.0, 1.0, size=(3, *frame.shape))
    none = frames.conjugate(frame, xi, 0.0)
    some = frames.conjugate(frame, xi, 10.0)
    free = 0.0
    magnitude = 0.0
    for index, channel in enumerate(frame.channels):
        zeta = xi[index] - frames.spread(frame, channel, frames.means(channel, xi[index]))
        size = float(np.sum(np.abs(zeta)))
        free += size
        own = model.conjugate(channel.problem, dct.forward(frames.sums(channel, xi[index])))
        magnitude += abs(own) + frames.FREE_CENTRE * size + 10.0 * size
    assert free > 0.0
    assert abs((some - none) - 10.0 * free) <= rounding.gamma(frame.samples + 8) * 2.0 * magnitude
    whole, _ = synthetic.colour_frame(24, 32, seed=49, ratio=(1, 1))
    assert not any(whole.free(channel) for channel in whole.channels)
    xi = rng.normal(0.0, 1.0, size=(3, *whole.shape))
    each = [model.conjugate(channel.problem, dct.forward(xi[index])) for index, channel in enumerate(whole.channels)]
    assert frames.conjugate(whole, xi, 10.0) == (each[0] + each[1]) + each[2]
