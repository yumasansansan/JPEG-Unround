# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""The reference implementation through its C interface (unround.native), and the package around it.

The settings are checked to reach the library exactly: it gives back, as options, the
values it holds, which are compared with the settings' own to the last bit. The
results are checked where they are exact: the intervals and the weights of the data
term, which are rationals or one division; the planes, which are the canvas cut to
the picture; the picture, which is JFIF's RGB of the planes; the records, which follow
the options; the files, which hold the samples bit for bit. The colour conversions are
checked against JFIF's exact rationals, within their rounding. What the solvers
compute is the reference implementation's own, which its tests and the conformance
cases check (conformance/); here it is taken as it comes, and checked where it is
exact: within the intervals, and the same however the file comes in.
"""

import ctypes
import dataclasses
import io
import math
import os
import struct
import subprocess
import sys
import threading
from fractions import Fraction
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
import tifffile
from hypothesis import given
from hypothesis import strategies as st
from PIL import Image

import pngfile
import synthetic
import unround
from unround import jpegio, native
from unround.settings import TGV, TV, DataTerm, Method, PdhgOptions, Settings, SubgradientOptions

METHODS: list[Method] = ["mmse", "tv", "tgv", "subgradient"]
SUBSAMPLINGS = {"4:4:4": 0, "4:2:0": 2}  # Pillow's names for them
U = 2.0**-53


def gamma(n: int) -> float:
    """Higham's gamma_n = n U / (1 - n U), the relative bound of n roundings."""
    return n * U / (1.0 - n * U)


def grey_file(height: int = 21, width: int = 30, quality: int = 30) -> bytes:
    samples = np.clip(synthetic.picture(height, width, seed=50), 0.0, 255.0).astype(np.uint8)
    stream = io.BytesIO()
    Image.fromarray(samples).save(stream, format="JPEG", quality=quality)
    return stream.getvalue()


def colour_file(subsampling: str, height: int = 21, width: int = 30, quality: int = 30, **options: object) -> bytes:
    samples = np.clip(synthetic.colour_picture(height, width, seed=51), 0.0, 255.0).astype(np.uint8)
    stream = io.BytesIO()
    image = Image.fromarray(samples)
    image.save(stream, format="JPEG", quality=quality, subsampling=SUBSAMPLINGS[subsampling], **options)
    return stream.getvalue()


def bits(values: npt.ArrayLike) -> npt.NDArray[np.uint64]:
    return np.ascontiguousarray(values, dtype=np.float64).view(np.uint64)


def same(text: str, value: float) -> bool:
    """Whether the library's decimal is the double, to the last bit: -0.0 is not 0.0."""
    return struct.pack("<d", float(text)) == struct.pack("<d", value)


def identical(value: float, other: float) -> bool:
    return struct.pack("<d", value) == struct.pack("<d", other)


def held(
    settings: Settings, *, max_pixels: int = 0, max_scans: int = 0, warnings_are_errors: bool = False
) -> dict[str, str]:
    """The settings as the library holds them, by the options' names; flags have the value ""."""
    values: dict[str, str] = {}
    limits = {"max_pixels": max_pixels, "max_scans": max_scans}
    for option in native.held_options(settings, **limits, warnings_are_errors=warnings_are_errors):
        name, _, value = option.removeprefix("--").partition("=")
        assert name not in values
        values[name] = value
    return values


def test_the_library_is_of_the_interface_s_version() -> None:
    assert native.version().startswith("unround ")
    assert int(native._library().unround_abi_version()) == native.ABI_VERSION
    for structure, size in native._SIZES:
        assert ctypes.sizeof(structure) == size


def test_the_defaults_are_the_library_s() -> None:
    # The defaults of the Python dataclasses, given as options, are what the library
    # holds of no options at all.
    library = native._library()
    assert native.held_options(Settings()) == native._held(library, native._Settings([]))


positive = st.floats(min_value=0.0, exclude_min=True, allow_infinity=False)
at_least_zero = st.floats(min_value=0.0, allow_infinity=False)
finite = st.floats(allow_nan=False, allow_infinity=False)
open_interval = st.floats(min_value=0.0, max_value=2.0, exclude_min=True, exclude_max=True)
unit_interval = st.floats(min_value=0.0, max_value=1.0, exclude_min=True, exclude_max=True)
data_terms = st.builds(
    DataTerm,
    mu=st.none() | at_least_zero,
    mu_scale=at_least_zero,
    mu_power=finite,
    slack=at_least_zero,
    dc_weight=at_least_zero,
    centres=st.sampled_from(["mmse", "midpoint"]),
    power=at_least_zero,
    slack_cost=at_least_zero,
)
gammas = st.none() | st.tuples(positive, positive, positive)
settings_drawn = st.builds(
    Settings,
    method=st.sampled_from(METHODS),
    data=data_terms | st.tuples(data_terms, data_terms, data_terms),
    tv=st.builds(TV, alpha=positive, channel_weights=gammas, coupled=st.booleans()),
    tgv=st.builds(TGV, alpha1=positive, alpha0=positive, channel_weights=gammas, coupled=st.booleans()),
    pdhg=st.builds(
        PdhgOptions,
        iterations=st.none() | st.integers(0, 2**40),
        tolerance=st.none() | at_least_zero,
        relative_tolerance=at_least_zero,
        partial_tolerance=at_least_zero,
        step_ratio=st.none() | positive,
        relaxation=st.none() | open_interval,
        step_product=unit_interval,
        norm_squared=st.none() | positive,
        scale_with_weight=st.booleans(),
        record_every=st.integers(0, 2**40),
        partial_radius=st.none() | at_least_zero,
        free_radius=at_least_zero,
    ),
    subgradient=st.builds(
        SubgradientOptions,
        iterations=st.integers(0, 2**40),
        record_every=st.integers(0, 2**40),
        step=positive,
        decay=at_least_zero,
        momentum=st.booleans(),
    ),
)


@given(
    settings=settings_drawn,
    max_pixels=st.integers(0, 2**64 - 1),
    max_scans=st.integers(0, 2**31 - 1),
    warnings_are_errors=st.booleans(),
)
def test_the_settings_reach_the_library_to_the_last_bit(
    *, settings: Settings, max_pixels: int, max_scans: int, warnings_are_errors: bool
) -> None:
    values = held(settings, max_pixels=max_pixels, max_scans=max_scans, warnings_are_errors=warnings_are_errors)
    model: TV | TGV = settings.tgv if settings.method == "tgv" else settings.tv
    assert values.pop("method") == settings.method
    assert same(values.pop("alpha"), settings.tv.alpha)
    assert same(values.pop("alpha1"), settings.tgv.alpha1)
    assert same(values.pop("alpha0"), settings.tgv.alpha0)
    weights = values.pop("channel-weights", None)
    if model.channel_weights is None:
        assert weights is None
    else:
        assert weights is not None
        items = weights.split(",")
        assert len(items) == len(model.channel_weights)
        assert all(same(item, gamma) for item, gamma in zip(items, model.channel_weights, strict=True))
    assert values.pop("channels") == ("coupled" if model.coupled else "apart")
    terms = (settings.data,) if isinstance(settings.data, DataTerm) else settings.data
    for name, field in (
        ("mu-scale", "mu_scale"),
        ("mu-power", "mu_power"),
        ("weight-power", "power"),
        ("dc-weight", "dc_weight"),
        ("slack", "slack"),
        ("slack-cost", "slack_cost"),
    ):
        items = values.pop(name).split(",")
        for index, term in enumerate(terms):
            assert same(items[index if len(items) > 1 else 0], getattr(term, field)), name
    mus, centres = values.pop("mu").split(","), values.pop("centres").split(",")
    for index, term in enumerate(terms):
        mu = mus[index if len(mus) > 1 else 0]
        assert mu == "rule" if term.mu is None else same(mu, term.mu)
        assert centres[index if len(centres) > 1 else 0] == term.centres
    solver = settings.pdhg
    for name, value in (
        ("tolerance", solver.tolerance),
        ("step-ratio", solver.step_ratio),
        ("relaxation", solver.relaxation),
        ("norm-squared", solver.norm_squared),
        ("partial-radius", solver.partial_radius),
        ("relative-tolerance", solver.relative_tolerance),
        ("partial-tolerance", solver.partial_tolerance),
        ("step-product", solver.step_product),
        ("free-radius", solver.free_radius),
    ):
        text = values.pop(name, None)
        assert (text is None) if value is None else (text is not None and same(text, value)), name
    iterations = values.pop("iterations", None)
    assert iterations == (None if solver.iterations is None else str(solver.iterations))
    assert values.pop("record-every") == str(solver.record_every)
    assert ("no-weight-scaling" in values) == (not solver.scale_with_weight)
    values.pop("no-weight-scaling", None)
    method = settings.subgradient
    assert values.pop("subgradient-iterations") == str(method.iterations)
    assert values.pop("subgradient-record-every") == str(method.record_every)
    assert same(values.pop("subgradient-step"), method.step)
    assert same(values.pop("subgradient-decay"), method.decay)
    assert ("no-momentum" in values) == (not method.momentum)
    values.pop("no-momentum", None)
    assert values.pop("max-pixels", "0") == str(max_pixels)
    assert values.pop("max-scans", "0") == str(max_scans)
    assert ("warnings-are-errors" in values) == warnings_are_errors
    values.pop("warnings-are-errors", None)
    assert values == {}


def test_settings_out_of_their_ranges_are_refused() -> None:
    with pytest.raises(native.UnroundError) as refused:
        native.held_options(Settings(tv=TV(alpha=0.0), pdhg=PdhgOptions(relaxation=2.0)))
    assert refused.value.status is native.Status.OPTIONS
    assert "--alpha is positive" in refused.value.message
    assert "--relaxation is in (0, 2)" in refused.value.message
    with pytest.raises(ValueError, match="at least 0"):
        native.options(max_pixels=-1)


def check_exact(decoded: native.Decoded, image: jpegio.Image, slack: float = 0.0) -> None:
    """What is exact of a result: the intervals and the steps, the planes, the picture, and the invariant."""
    height, width, count = image.height, image.width, len(image.components)
    assert decoded.color_space is image.color_space
    assert decoded.planes.shape == (count, height, width)
    assert decoded.canvas.shape[0] == count
    np.testing.assert_array_equal(bits(decoded.planes), bits(decoded.canvas[:, :height, :width]))
    if image.color_space is jpegio.ColorSpace.YCBCR:
        assert decoded.picture.shape == (height, width, 3)
        np.testing.assert_array_equal(bits(decoded.picture), bits(native.to_rgb(decoded.planes)))
    else:
        assert count == 1
        np.testing.assert_array_equal(bits(decoded.picture), bits(decoded.planes[0]))
    for component, coefficients, problem in zip(image.components, decoded.coefficients, decoded.problems, strict=True):
        # Halves, small integers and the slack below times steps of 16 bits: exact.
        levels = component.coefficients.astype(np.float64)
        steps = component.quant_table.astype(np.float64)
        lower, upper = (levels - 0.5 - slack) * steps, (levels + 0.5 + slack) * steps
        lower[:, :, 0, 0] += 1024.0
        upper[:, :, 0, 0] += 1024.0
        np.testing.assert_array_equal(problem.lower, lower)
        np.testing.assert_array_equal(problem.upper, upper)
        np.testing.assert_array_equal(problem.steps, steps)
        assert problem.ratio == (
            image.max_v_samp_factor // component.v_samp_factor,
            image.max_h_samp_factor // component.h_samp_factor,
        )
        assert coefficients.shape == levels.shape
        assert np.all(problem.lower <= coefficients)
        assert np.all(coefficients <= problem.upper)


@pytest.mark.parametrize("method", METHODS)
def test_a_grey_file_is_reconstructed_within_its_intervals(method: Method) -> None:
    data = grey_file()
    settings = Settings(
        method=method,
        pdhg=PdhgOptions(iterations=30, tolerance=0.0, record_every=7),
        subgradient=SubgradientOptions(iterations=10, record_every=3),
    )
    decoded = native.decode(data, settings)
    check_exact(decoded, jpegio.read(data))
    if method == "mmse":
        assert decoded.result is None
        np.testing.assert_array_equal(decoded.coefficients[0], decoded.problems[0].centres)
        return
    assert decoded.result is not None
    expected = [0, 7, 14, 21, 28, 30] if method != "subgradient" else [0, 3, 6, 9, 10]
    assert decoded.result.history.iterations.tolist() == expected
    assert decoded.result.iterations == expected[-1]
    assert decoded.result.stop is native.Stop.ITERATIONS
    assert not decoded.result.converged


@pytest.mark.parametrize("method", ["mmse", "tv", "tgv"])
@pytest.mark.parametrize("subsampling", SUBSAMPLINGS)
def test_a_colour_file_is_reconstructed_within_its_intervals(method: Method, subsampling: str) -> None:
    data = colour_file(subsampling)
    decoded = native.decode(data, Settings(method=method, pdhg=PdhgOptions(iterations=20)))
    check_exact(decoded, jpegio.read(data))


def test_the_centres_follow_the_data_term() -> None:
    # The middles are exact: q Q, and DC's with 1024. The MMSE centres lie in the halves
    # of the bins towards 0, DC's and those of 0 at the middles.
    data = colour_file("4:2:0")
    image = jpegio.read(data)
    middles = native.decode(data, Settings(method="mmse", data=DataTerm(centres="midpoint", slack=0.5)))
    check_exact(middles, image, slack=0.5)
    mmse = native.decode(data, Settings(method="mmse"))
    for component, one, other in zip(image.components, middles.problems, mmse.problems, strict=True):
        levels = component.coefficients.astype(np.float64)
        steps = component.quant_table.astype(np.float64)
        middle = levels * steps
        middle[:, :, 0, 0] += 1024.0
        np.testing.assert_array_equal(one.centres, middle)
        np.testing.assert_array_equal(other.centres[:, :, 0, 0], middle[:, :, 0, 0])
        ac = np.ones((8, 8), dtype=bool)
        ac[0, 0] = False
        magnitude, centres = np.abs(levels[:, :, ac]), other.centres[:, :, ac]
        assert np.all(centres[magnitude == 0.0] == 0.0)
        inside = magnitude > 0.0
        steps_ac = np.broadcast_to(steps[ac], magnitude.shape)
        assert np.all(np.sign(centres[inside]) == np.sign(levels[:, :, ac][inside]))
        assert np.all((magnitude - 0.5)[inside] * steps_ac[inside] <= np.abs(centres[inside]))
        assert np.all(np.abs(centres[inside]) <= magnitude[inside] * steps_ac[inside])


def test_the_weights_are_those_of_the_data_term() -> None:
    # mu / Q^p, one division by the power of the step, which is exact for p of 1 and 2,
    # and DC's times its weight (docs/math.md, 4.1). The rule's mu is mu_scale times the
    # mean of the 64 steps, whose sum is exact and so is its division by 64.
    data = colour_file("4:2:0")
    terms = (DataTerm(mu=0.25, dc_weight=3.0), DataTerm(mu=None, mu_scale=1e-3, power=1.0), DataTerm(power=2.0))
    decoded = native.decode(data, Settings(method="mmse", data=terms))
    for term, problem in zip(terms, decoded.problems, strict=True):
        steps = problem.steps
        mu = term.mu if term.mu is not None else term.mu_scale * (float(np.sum(steps)) / 64.0)
        expected = mu / (steps * steps if term.power == 2.0 else steps)
        expected[0, 0] *= term.dc_weight
        np.testing.assert_array_equal(bits(problem.weights), bits(expected))
    assert decoded.problems[0].weights[0, 0] == 0.25 / decoded.problems[0].steps[0, 0] ** 2 * 3.0


def test_an_observer_follows_the_records_and_stops_the_solver() -> None:
    data = colour_file("4:2:0")
    seen: list[native.Record] = []

    def observer(record: native.Record) -> bool:
        seen.append(record)
        return record.iteration >= 10

    settings = Settings(method="tgv", pdhg=PdhgOptions(iterations=100, tolerance=0.0, record_every=5))
    decoded = native.decode(data, settings, observer=observer)
    assert decoded.result is not None
    assert decoded.result.stop is native.Stop.OBSERVER
    assert decoded.result.iterations == 10
    history = decoded.result.history
    assert history.iterations.tolist() == [0, 5, 10]
    assert [record.iteration for record in seen] == [5, 10]
    for index, record in enumerate(seen, start=1):
        assert identical(record.primal, float(history.primal[index]))
        assert identical(record.dual, float(history.dual[index]))
        assert record.canvas.shape == decoded.canvas.shape
    # The solver stopped at the last record, and its canvas is the result's.
    np.testing.assert_array_equal(bits(seen[-1].canvas), bits(decoded.canvas))
    # The decoder of the centres has no records to show.
    assert native.decode(data, Settings(method="mmse"), observer=observer).result is None
    assert len(seen) == 2


def test_an_exception_of_the_observer_stops_the_solver_and_is_raised() -> None:
    calls = 0

    def observer(record: native.Record) -> bool:
        nonlocal calls
        calls += 1
        message = f"stopped at {record.iteration}"
        raise LookupError(message)

    settings = Settings(method="tv", pdhg=PdhgOptions(iterations=1000, tolerance=0.0, record_every=1))
    with pytest.raises(LookupError, match="stopped at 1"):
        native.decode(grey_file(), settings, observer=observer)
    assert calls == 1


@pytest.mark.parametrize("kind", ["grey", "4:2:0"])
def test_components_given_as_arrays_solve_as_their_file(kind: str) -> None:
    data = grey_file() if kind == "grey" else colour_file(kind)
    settings = Settings(method="tgv", pdhg=PdhgOptions(iterations=20))
    from_file, from_arrays = native.decode(data, settings), native.solve(jpegio.read(data), settings)
    np.testing.assert_array_equal(bits(from_arrays.picture), bits(from_file.picture))
    for one, other in zip(from_arrays.coefficients, from_file.coefficients, strict=True):
        np.testing.assert_array_equal(bits(one), bits(other))
    assert from_arrays.result is not None
    assert from_file.result is not None
    np.testing.assert_array_equal(bits(from_arrays.result.history.primal), bits(from_file.result.history.primal))


def test_changed_components_are_what_is_solved() -> None:
    image = jpegio.read(grey_file())
    (component,) = image.components
    levels = np.zeros_like(component.coefficients)
    levels[:, :, 0, 0] = component.coefficients[:, :, 0, 0]
    changed = dataclasses.replace(image, components=(dataclasses.replace(component, coefficients=levels),))
    decoded = native.solve(changed, Settings(method="mmse"))
    check_exact(decoded, changed)
    # Only DC is left: the centres of the levels of 0 are 0.
    (coefficients,) = decoded.coefficients
    assert np.all(coefficients[:, :, 1:, :] == 0.0)
    assert np.all(coefficients[:, :, 0, 1:] == 0.0)
    wrong = dataclasses.replace(component, coefficients=levels.reshape(levels.shape[0], -1))
    with pytest.raises(ValueError, match="rows, columns, 8, 8"):
        native.solve(dataclasses.replace(image, components=(wrong,)), Settings(method="mmse"))


def test_what_is_not_reconstructed_is_said() -> None:
    with pytest.raises(native.UnroundError) as garbage:
        native.decode(b"this is not a JPEG file")
    assert garbage.value.status is native.Status.READ
    with pytest.raises(native.UnroundError) as limited:
        native.decode(grey_file(), max_pixels=100)
    assert limited.value.status is native.Status.READ
    with pytest.raises(native.UnroundError) as colour_subgradient:
        native.decode(colour_file("4:4:4"), Settings(method="subgradient"))
    assert colour_subgradient.value.status is native.Status.OPTIONS
    assert "one component" in colour_subgradient.value.message
    with pytest.raises(native.UnroundError) as partial:
        native.decode(grey_file(), Settings(method="tv", pdhg=PdhgOptions(partial_radius=20.0)))
    assert partial.value.status is native.Status.OPTIONS


def test_the_profile_and_the_orientation_are_the_file_s() -> None:
    exif = Image.Exif()
    exif[0x0112] = 6
    profile = bytes(np.random.default_rng(4).integers(0, 256, size=3000, dtype=np.uint8))
    decoded = native.decode(colour_file("4:2:0", exif=exif.tobytes(), icc_profile=profile), Settings(method="mmse"))
    assert decoded.exif_orientation == 6
    assert decoded.icc_profile == profile
    plain = native.decode(grey_file(), Settings(method="mmse"))
    assert plain.exif_orientation == 0
    assert plain.icc_profile is None


def test_threads_reconstruct_at_once() -> None:
    files = [grey_file(), colour_file("4:2:0"), colour_file("4:4:4"), grey_file(quality=60)]
    settings = Settings(method="tgv", pdhg=PdhgOptions(iterations=40))
    alone = [native.decode(data, settings) for data in files]
    together: list[native.Decoded | None] = [None] * len(files)

    def work(index: int) -> None:
        together[index] = native.decode(files[index], settings)

    threads = [threading.Thread(target=work, args=(index,)) for index in range(len(files))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for one, other in zip(alone, together, strict=True):
        assert other is not None
        np.testing.assert_array_equal(bits(one.picture), bits(other.picture))


def awkward(shape: tuple[int, ...]) -> npt.NDArray[np.float64]:
    """Random binary64 samples, with the values that a careless conversion would change."""
    samples = np.random.default_rng(60).uniform(-10.0, 265.0, size=shape)
    special = [-0.0, 0.0, 5e-324, -5e-324, 2.2250738585072014e-308, 1.7976931348623157e308, 1.0 / 3.0, 255.0]
    special.append(math.nextafter(128.0, math.inf))
    flat = samples.reshape(-1)  # a view: the samples are contiguous
    count = min(len(special), flat.size)
    flat[:count] = special[:count]
    return samples


def tiff_page(data: bytes) -> tuple[npt.NDArray[np.float64], tifffile.TiffPage]:
    """The samples of a TIFF's first page, as tifffile, independent of the writer, reads them, and the page."""
    with tifffile.TiffFile(io.BytesIO(data)) as file:
        page = file.pages[0]
        assert isinstance(page, tifffile.TiffPage)
        return np.asarray(page.asarray(), dtype=np.float64), page


@pytest.mark.parametrize("shape", [(1, 1), (7, 13), (21, 30), (5, 6, 3)])
def test_the_tiff_holds_the_samples_bit_for_bit(shape: tuple[int, ...]) -> None:
    picture = awkward(shape)
    read, page = tiff_page(native.tiff_bytes(picture))
    assert page.dtype == np.float64
    assert page.compression == 1
    assert page.samplesperpixel == (3 if len(shape) == 3 else 1)
    assert read.shape == shape
    np.testing.assert_array_equal(bits(read), bits(picture))


def test_ycbcr_is_declared_as_it_is() -> None:
    # The planes of a colour file, as the solver left them, through the file and back, which
    # says they are JFIF's YCbCr: no subsampling, and JFIF's coefficients.
    decoded = native.decode(colour_file("4:2:0"), Settings(method="mmse"))
    planes = np.moveaxis(decoded.planes, 0, -1)
    read, page = tiff_page(native.tiff_bytes(planes, ycbcr=True))
    assert page.photometric == tifffile.PHOTOMETRIC.YCBCR
    assert page.tags["YCbCrSubSampling"].value == (1, 1)
    assert page.tags["YCbCrCoefficients"].value == (299, 1000, 587, 1000, 114, 1000)
    assert page.tags["ReferenceBlackWhite"].value == (0, 1, 255, 1, 128, 1, 255, 1, 128, 1, 255, 1)
    np.testing.assert_array_equal(bits(read.reshape(planes.shape)), bits(planes))
    with pytest.raises(native.UnroundError):
        native.tiff_bytes(planes[:, :, 0], ycbcr=True)
    with pytest.raises(native.UnroundError):
        native.tiff_bytes(np.zeros((2, 2, 2)))
    with pytest.raises(ValueError, match="height, width"):
        native.tiff_bytes(np.zeros(4))


def rounded(value: float, scale: int, top: int) -> int:
    """value times scale, as a double, rounded to the nearest integer, halves away from 0, and clamped."""
    exact = Fraction(value * scale)
    whole = int(abs(exact) + Fraction(1, 2)) * (1 if exact >= 0 else -1)
    return min(max(whole, 0), top)


def test_the_pnm_rounds_halves_away_from_zero() -> None:
    samples = np.array(
        [[0.5, 1.5, 2.5, 254.5, 0.49999999999999994, -0.5], [255.4, 255.5, 300.0, -7.0, 128.0, 1.0 / 257.0]]
    )
    eight = native.pnm_bytes(samples)
    header = b"P5\n6 2\n255\n"
    assert eight[: len(header)] == header
    assert list(eight[len(header) :]) == [rounded(float(value), 1, 255) for value in samples.ravel()]
    sixteen = native.pnm_bytes(samples[:, :, np.newaxis].repeat(3, axis=2), sixteen=True)
    header = b"P6\n6 2\n65535\n"
    assert sixteen[: len(header)] == header
    values = np.frombuffer(sixteen[len(header) :], dtype=">u2")
    expected = [rounded(float(value), 257, 65535) for value in samples.ravel() for _ in range(3)]
    assert values.tolist() == expected
    with pytest.raises(native.UnroundError, match="not finite"):
        native.pnm_bytes(np.array([[np.nan]]))


def icc_profile(size: int, space: bytes, seed: int = 0) -> bytes:
    """A well-formed ICC profile of version 2, a monitor's, of the data color space given: its header,
    one tag, and the tag's data."""
    profile = bytearray(np.random.default_rng(seed).integers(0, 256, size=size, dtype=np.uint8).tobytes())
    profile[:132] = bytes(132)
    struct.pack_into(">I", profile, 0, size)
    profile[8] = 2
    profile[12:24] = b"mntr" + space + b"XYZ "
    profile[36:40] = b"acsp"
    struct.pack_into(">III", profile, 68, 0x0000F6D6, 0x00010000, 0x0000D32D)  # the illuminant D50
    struct.pack_into(">I4sII", profile, 128, 1, b"desc", 144, size - 144)
    return bytes(profile)


def tiff_profile(data: bytes) -> bytes | None:
    """The ICC profile of a little-endian TIFF's first IFD (InterColorProfile, 34675), or None."""
    assert data[:4] == b"II*\x00"
    (ifd,) = struct.unpack_from("<I", data, 4)
    (count,) = struct.unpack_from("<H", data, ifd)
    for entry in range(count):
        tag, kind, length, offset = struct.unpack_from("<HHII", data, ifd + 2 + 12 * entry)
        if tag == 34675:
            assert kind == 7
            return data[offset : offset + length]
    return None


def pnm_samples(picture: npt.NDArray[np.float64], *, sixteen: bool) -> npt.NDArray[np.int64]:
    """The samples of the PNM file of the picture: those that PNG has to hold too."""
    data = native.pnm_bytes(picture, sixteen=sixteen)
    body = data.split(b"\n", 3)[3]
    values = np.frombuffer(body, dtype=">u2" if sixteen else np.uint8).astype(np.int64)
    return values.reshape(picture.shape[0], picture.shape[1], -1)


@pytest.mark.parametrize("sixteen", [False, True])
@pytest.mark.parametrize("shape", [(1, 1), (6, 9), (5, 7, 3)])
def test_the_png_holds_the_pnm_s_samples(shape: tuple[int, ...], *, sixteen: bool) -> None:
    picture = np.random.default_rng(72).uniform(-20.0, 280.0, size=shape)
    picture.flat[:3] = [0.5, 254.5, 0.49999999999999994][: picture.size]
    for compression in [None, 0, 9]:
        read = pngfile.read(native.png_bytes(picture, sixteen=sixteen, compression=compression))
        assert read.bits == (16 if sixteen else 8)
        assert read.icc_profile is None
        np.testing.assert_array_equal(read.samples, pnm_samples(picture, sixteen=sixteen))
    with pytest.raises(ValueError, match="0 to 9"):
        native.png_bytes(picture, compression=10)
    with pytest.raises(native.UnroundError, match="not finite"):
        native.png_bytes(np.full(shape, np.nan))


def test_the_icc_profile_goes_where_it_goes_with_the_picture() -> None:
    rng = np.random.default_rng(73)
    rgb, gray = icc_profile(500, b"RGB "), icc_profile(301, b"GRAY")
    assert jpegio.check_icc(rgb, 3) is None
    assert jpegio.check_icc(gray, 1) is None
    for picture, profile in [(rng.uniform(0.0, 255.0, size=(4, 5, 3)), rgb), (rng.uniform(0.0, 255.0, (4, 5)), gray)]:
        read = pngfile.read(native.png_bytes(picture, icc_profile=profile))
        assert (read.icc_profile, read.icc_name) == (profile, b"ICC profile")
        assert tiff_profile(native.tiff_bytes(picture, icc_profile=profile)) == profile
    ycbcr = native.tiff_bytes(rng.uniform(0.0, 255.0, size=(4, 5, 3)), ycbcr=True, icc_profile=rgb)
    assert tiff_profile(ycbcr) == rgb

    # A profile of another color space, or not well formed, is left out with a warning that says why.
    broken = rgb[:36] + b"ascp" + rgb[40:]
    for picture, profile in [(rng.uniform(0.0, 255.0, size=(4, 5)), rgb), (rng.uniform(0.0, 255.0, (4, 5, 3)), gray)]:
        channels = 1 if picture.ndim == 2 else 3
        for refused in (profile, broken):
            reason = jpegio.check_icc(refused, channels)
            assert reason
            with pytest.warns(native.UnroundWarning, match="the ICC profile is not written") as caught:
                data = native.png_bytes(picture, icc_profile=refused)
            assert str(caught[0].message) == f"the ICC profile is not written: {reason}"
            assert pngfile.read(data).icc_profile is None
            with pytest.warns(native.UnroundWarning, match="the ICC profile is not written"):
                assert tiff_profile(native.tiff_bytes(picture, icc_profile=refused)) is None


def turned_by_numpy(picture: npt.NDArray[np.float64], orientation: int) -> npt.NDArray[np.float64]:
    """The picture turned upright with NumPy's flips and rotations."""
    match orientation:
        case 2:
            turned = picture[:, ::-1]
        case 3:
            turned = picture[::-1, ::-1]
        case 4:
            turned = picture[::-1, :]
        case 5:
            turned = picture.swapaxes(0, 1)
        case 6:
            turned = np.rot90(picture, k=-1)
        case 7:
            turned = picture.swapaxes(0, 1)[::-1, ::-1]
        case 8:
            turned = np.rot90(picture, k=1)
        case _:
            turned = picture
    return turned


def test_a_picture_turns_as_numpy_turns_it() -> None:
    rng = np.random.default_rng(74)
    for shape in [(3, 5), (70, 66, 3), (1, 4, 3)]:
        picture = rng.uniform(-1.0, 256.0, size=shape)
        picture.flat[0] = -0.0
        for orientation in range(9):
            turned = native.oriented(picture, orientation)
            expected = turned_by_numpy(picture, orientation)
            assert turned.shape == expected.shape
            np.testing.assert_array_equal(bits(turned), bits(expected))
    with pytest.raises(native.UnroundError, match="1 to 8"):
        native.oriented(np.zeros((2, 2)), 9)


def test_the_command_line_writes_the_picture_upright_with_its_profile(tmp_path: Path) -> None:
    exif = Image.Exif()
    exif[0x0112] = 8
    profile = icc_profile(640, b"RGB ", seed=5)
    data = colour_file("4:2:0", exif=exif.tobytes(), icc_profile=profile)
    source = tmp_path / "in.jpg"
    source.write_bytes(data)
    decoded = native.decode(data, Settings(method="mmse"))
    upright = native.oriented(decoded.picture, 8)
    assert upright.shape == (30, 21, 3)
    for name, arguments in [("out.png", ["--bits", "16"]), ("out.tif", [])]:
        target = tmp_path / name
        assert native.main(["--method", "mmse", "--quiet", *arguments, str(source), str(target)]) == 0
        if name.endswith(".png"):
            read = pngfile.read(target.read_bytes())
            np.testing.assert_array_equal(read.samples, pnm_samples(upright, sixteen=True))
            assert read.icc_profile == profile
        else:
            assert target.read_bytes() == native.tiff_bytes(upright, icc_profile=profile)
    kept = tmp_path / "kept.tif"
    assert native.main(["--method", "mmse", "--orientation", "keep", "--no-icc", str(source), str(kept)]) == 0
    assert kept.read_bytes() == native.tiff_bytes(decoded.picture)


def test_the_conversions_are_jfif_s() -> None:
    # Against JFIF's exact rationals (docs/math.md, 8), in exact arithmetic, within the
    # rounding of the library, which may fuse products and sums or not. Each output is a
    # dot product of at most three terms, whose constants are below 2 and within U of the
    # rationals, with a subtraction before a product and a sum after it: gamma(8) times
    # 128 + 2 (|R| + |G| + |B|) bounds each of Y, Cb and Cr, and gamma(8) times
    # |Y| + 2 (|Cb - 128| + |Cr - 128|) each of R, G and B, of the planes it is given.
    rng = np.random.default_rng(71)
    picture = rng.uniform(-20.0, 280.0, size=(5, 7, 3))
    picture.flat[:3] = [0.0, 255.0, 128.0]
    planes = native.to_ycbcr(picture)
    assert planes.shape == (3, 5, 7)
    rgb = native.to_rgb(planes)
    assert rgb.shape == (5, 7, 3)
    rounds = Fraction(gamma(8))
    for i, j in np.ndindex(5, 7):
        r, g, b = (Fraction(float(value)) for value in picture[i, j])
        y = Fraction(299, 1000) * r + Fraction(587, 1000) * g + Fraction(114, 1000) * b
        exact = (y, 128 + Fraction(250, 443) * (b - y), 128 + Fraction(500, 701) * (r - y))
        bound = rounds * (128 + 2 * (abs(r) + abs(g) + abs(b)))
        assert all(abs(Fraction(float(planes[k, i, j])) - exact[k]) <= bound for k in range(3))
        y, cb, cr = (Fraction(float(planes[k, i, j])) for k in range(3))
        exact = (
            y + Fraction(701, 500) * (cr - 128),
            y - Fraction(25251, 73375) * (cb - 128) - Fraction(209599, 293500) * (cr - 128),
            y + Fraction(443, 250) * (cb - 128),
        )
        bound = rounds * (abs(y) + 2 * (abs(cb - 128) + abs(cr - 128)))
        assert all(abs(Fraction(float(rgb[i, j, k])) - exact[k]) <= bound for k in range(3))
    with pytest.raises(ValueError, match="3, height, width"):
        native.to_rgb(picture)
    with pytest.raises(ValueError, match="height, width, 3"):
        native.to_ycbcr(planes)


def test_the_command_line_runs_through_the_library(tmp_path: Path, capfd: pytest.CaptureFixture[str]) -> None:
    assert native.main(["--version"]) == 0
    assert "unround" in capfd.readouterr().out
    assert native.main(["--alpha", "0", "in.jpg"]) == 2
    assert "--alpha is positive" in capfd.readouterr().err
    source, target = tmp_path / "in.jpg", tmp_path / "out.tif"
    source.write_bytes(grey_file())
    assert native.main(["--method", "mmse", "--quiet", str(source), str(target)]) == 0
    decoded = native.decode(grey_file(), Settings(method="mmse"))
    assert target.read_bytes() == native.tiff_bytes(decoded.picture)


def test_the_c_layer_is_read_through_either_library(monkeypatch: pytest.MonkeyPatch) -> None:
    # The C layer's own library, where the environment names one, and the reference
    # implementation's, which passes the same functions on.
    data = colour_file("4:2:0")
    own, own_planes = jpegio.read(data), jpegio.decode_planes(data)
    profiles = [icc_profile(400, b"RGB "), icc_profile(400, b"GRAY"), b"acsp"]
    own_checks = [jpegio.check_icc(profile, channels) for profile in profiles for channels in (1, 3)]
    own_libpng = jpegio.libpng_version()
    monkeypatch.delenv("UNROUND_JPEGIO_LIBRARY", raising=False)
    jpegio._library.cache_clear()
    try:
        _, prefix = jpegio._library()
        assert prefix == "unround_jpeg_"
        through, planes = jpegio.read(data), jpegio.decode_planes(data)
        assert [jpegio.check_icc(profile, channels) for profile in profiles for channels in (1, 3)] == own_checks
        assert jpegio.libpng_version() == own_libpng
    finally:
        jpegio._library.cache_clear()
    assert own_checks[0] is not None
    assert own_checks[1] is None
    assert own_checks[2] is None
    assert own_libpng.startswith("libpng 1.6.")
    for field in ("width", "height", "color_space", "max_h_samp_factor", "max_v_samp_factor", "icc_profile"):
        assert getattr(through, field) == getattr(own, field)
    for one, other in zip(through.components, own.components, strict=True):
        np.testing.assert_array_equal(one.coefficients, other.coefficients)
        np.testing.assert_array_equal(one.quant_table, other.quant_table)
    for one_plane, other_plane in zip(planes.planes, own_planes.planes, strict=True):
        np.testing.assert_array_equal(one_plane, other_plane)


def test_the_package_decodes_a_path_bytes_or_an_image(tmp_path: Path) -> None:
    data = colour_file("4:2:0")
    source = tmp_path / "in.jpg"
    source.write_bytes(data)
    settings = Settings(method="tv", pdhg=PdhgOptions(iterations=10))
    from_bytes = unround.decode(data, settings)
    for other_source in (source, str(source), bytearray(data), unround.read(data)):
        other = unround.decode(other_source, settings)
        np.testing.assert_array_equal(bits(other.picture), bits(from_bytes.picture))
    with pytest.raises(ValueError, match="no limits"):
        unround.decode(unround.read(data), settings, max_pixels=10)


def test_python_runs_the_command_line(tmp_path: Path) -> None:
    source, target = tmp_path / "in.jpg", tmp_path / "out.tif"
    source.write_bytes(grey_file())
    package = str(Path(unround.__file__).resolve().parent.parent)
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join([package, os.environ.get("PYTHONPATH", "")])}
    arguments = [sys.executable, "-m", "unround", "--method", "mmse", "--quiet", str(source), str(target)]
    completed = subprocess.run(arguments, env=environment, capture_output=True, check=False)  # noqa: S603
    assert completed.returncode == 0, completed.stderr
    decoded = unround.decode(source, Settings(method="mmse"))
    assert target.read_bytes() == native.tiff_bytes(decoded.picture)
