// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! The sweep of the planes against the iteration of its definition in the natural
//! layout: each step over the whole canvas, one after another, with the same
//! formulas. The two compute the same operations on the same values, and so agree to
//! the last bit.

mod support;

use jpeg_unround::frames::{self, Frame, Tensor, Vector};
use jpeg_unround::kernels::{advanced, ascent, descent, extrapolated, relaxed};
use jpeg_unround::model::{DataTerm, Step};
use jpeg_unround::operators::{self, Shape};
use jpeg_unround::planar::Layout;
use jpeg_unround::sweep::{TgvOutputs, TgvPoint, TgvSweep, TvOutputs, TvPoint, TvSweep};
use support::synthetic::{self, Numbers};

/// The parameters of an iteration of TV.
struct Parameters {
    tau: f64,
    sigma: f64,
    alpha: f64,
    rho: f64,
    coupled: bool,
    gammas: Vec<f64>,
}

/// The outputs of an iteration of the definition: `x~`, its coefficients, and `p~`.
struct Outputs {
    canvas: Vec<f64>,
    coefficients: Vec<Vec<f64>>,
    p: Vector,
}

/// One iteration of TV in the natural layout, step after step.
fn iterate(frame: &Frame, steps: &[Step], given: &Parameters, x: &mut [f64], p: &mut Vector) -> Outputs {
    let shape = Shape {
        height: frame.height(),
        width: frame.width(),
    };
    let plane = frame.plane();
    let size = frame.samples();
    let mut work = vec![0.0; size];
    for (channel, &gamma) in given.gammas.iter().enumerate() {
        let range = channel * plane..(channel + 1) * plane;
        let (mut across, mut down) = (vec![0.0; plane], vec![0.0; plane]);
        operators::backward_x(&p.x[range.clone()], shape, &mut across);
        operators::backward_y(&p.y[range.clone()], shape, &mut down);
        for (((entry, &value), &a), &d) in work[range.clone()].iter_mut().zip(&x[range]).zip(&across).zip(&down) {
            *entry = descent(value, given.tau * gamma, a, d);
        }
    }
    let mut coefficients: Vec<Vec<f64>> = frame
        .channels()
        .iter()
        .map(|channel| vec![0.0; channel.problem().samples()])
        .collect();
    let mut canvas = vec![0.0; size];
    frames::prox(frame, steps, &work, &mut coefficients, &mut canvas);
    let bar: Vec<f64> = canvas
        .iter()
        .zip(x.iter())
        .map(|(&new, &old)| extrapolated(new, old))
        .collect();
    let mut dual = Vector::zeros(size);
    for (channel, &gamma) in given.gammas.iter().enumerate() {
        let range = channel * plane..(channel + 1) * plane;
        let (mut across, mut down) = (vec![0.0; plane], vec![0.0; plane]);
        operators::grad(&bar[range.clone()], shape, &mut across, &mut down);
        for (i, index) in range.enumerate() {
            dual.x[index] = ascent(p.x[index], given.sigma * gamma, across[i]);
            dual.y[index] = ascent(p.y[index], given.sigma * gamma, down[i]);
        }
    }
    frames::project_vectors(frame, &mut dual, given.alpha, given.coupled);
    let rest = 1.0 - given.rho;
    for (value, &new) in x.iter_mut().zip(&canvas) {
        *value = relaxed(given.rho, rest, new, *value);
    }
    for (value, &new) in p.x.iter_mut().zip(&dual.x) {
        *value = relaxed(given.rho, rest, new, *value);
    }
    for (value, &new) in p.y.iter_mut().zip(&dual.y) {
        *value = relaxed(given.rho, rest, new, *value);
    }
    Outputs {
        canvas,
        coefficients,
        p: dual,
    }
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// Several iterations of both, from a start of the data term's centres and a random
/// dual in its balls, compared after each.
fn agree(frame: &Frame, given: &Parameters, iterations: usize, seed: u64) {
    let layout = Layout::new(frame).expect("a layout");
    let steps = frames::steps(frame, given.tau);
    let start = frames::start(frame, None).expect("a start");
    let size = frame.samples();
    let mut numbers = Numbers::new(seed);
    let mut p = Vector {
        x: (0..size).map(|_| numbers.uniform(-2.0, 2.0)).collect(),
        y: (0..size).map(|_| numbers.uniform(-2.0, 2.0)).collect(),
    };
    frames::project_vectors(frame, &mut p, given.alpha, given.coupled);
    let mut x = start.canvas.clone();
    let mut point = TvPoint {
        x: layout.to_planes(&start.canvas),
        px: layout.to_planes(&p.x),
        py: layout.to_planes(&p.y),
    };
    let mut sweep = TvSweep::new(
        frame,
        &layout,
        &steps,
        &given.gammas,
        given.tau,
        given.sigma,
        given.alpha,
        given.coupled,
        given.rho,
    );
    let mut outputs = TvOutputs::new(&layout);
    for iteration in 0..iterations {
        let expected = iterate(frame, &steps, given, &mut x, &mut p);
        sweep.iterate(&mut point, Some(&mut outputs));
        assert_eq!(
            bits(&layout.to_natural(&point.x)),
            bits(&x),
            "x after iteration {iteration}"
        );
        assert_eq!(
            bits(&layout.to_natural(&point.px)),
            bits(&p.x),
            "p across after iteration {iteration}"
        );
        assert_eq!(
            bits(&layout.to_natural(&point.py)),
            bits(&p.y),
            "p down after iteration {iteration}"
        );
        assert_eq!(
            bits(&layout.to_natural(&outputs.canvas)),
            bits(&expected.canvas),
            "x~ at {iteration}"
        );
        assert_eq!(
            bits(&layout.to_natural(&outputs.px)),
            bits(&expected.p.x),
            "p~ at {iteration}"
        );
        assert_eq!(
            bits(&layout.to_natural(&outputs.py)),
            bits(&expected.p.y),
            "p~ at {iteration}"
        );
        for ((part, planar), natural) in layout
            .parts
            .iter()
            .zip(&outputs.coefficients)
            .zip(&expected.coefficients)
        {
            let mut converted = vec![0.0; natural.len()];
            layout.write_values_natural(part, planar, &mut converted);
            assert_eq!(bits(&converted), bits(natural), "coefficients at {iteration}");
        }
        // Without outputs, the point moves alike.
        let mut again = point.clone();
        let mut copy = TvPoint {
            x: layout.to_planes(&x),
            px: layout.to_planes(&p.x),
            py: layout.to_planes(&p.y),
        };
        std::mem::swap(&mut again, &mut copy);
        sweep.iterate(&mut copy, None);
        let mut other = again;
        sweep.iterate(&mut other, Some(&mut outputs));
        assert_eq!(bits(&copy.x), bits(&other.x), "x without outputs at {iteration}");
        assert_eq!(bits(&copy.px), bits(&other.px), "p without outputs at {iteration}");
    }
}

fn parameters(count: usize) -> Parameters {
    Parameters {
        tau: 0.37,
        sigma: 0.21,
        alpha: 1.3,
        rho: 1.7,
        coupled: true,
        gammas: vec![1.0; count],
    }
}

#[test]
fn a_grey_frame_sweeps_as_it_is_defined() {
    let frame = Frame::one(synthetic::problem(24, 40, 3, 2.0, &DataTerm::default()).0);
    agree(&frame, &parameters(1), 4, 11);
    // Blocks of one MCU across, and one down.
    let narrow = Frame::one(synthetic::problem(8, 8, 4, 2.0, &DataTerm::default()).0);
    agree(&narrow, &parameters(1), 3, 12);
}

#[test]
fn a_colour_frame_sweeps_as_it_is_defined() {
    // 4:2:0 with MCUs cut by the picture: cells of four samples, and samples beyond
    // the blocks of each component.
    let (frame, _) = synthetic::colour_frame(21, 30, 5, (2, 2), &DataTerm::default());
    agree(&frame, &parameters(3), 4, 13);
    let mut apart = parameters(3);
    apart.coupled = false;
    apart.gammas = vec![1.0, 0.5, 2.0];
    agree(&frame, &apart, 3, 14);
    let (full, _) = synthetic::colour_frame(16, 24, 6, (1, 1), &DataTerm::default());
    agree(&full, &parameters(3), 3, 15);
    let (wide, _) = synthetic::colour_frame(20, 36, 7, (1, 2), &DataTerm::default());
    agree(&wide, &parameters(3), 3, 16);
}

#[test]
fn a_slack_with_a_cost_sweeps_as_it_is_defined() {
    let data = DataTerm {
        slack: 0.5,
        slack_cost: 3.0,
        dc_weight: 2.0,
        ..DataTerm::default()
    };
    let frame = Frame::one(synthetic::problem(16, 24, 8, 2.0, &data).0);
    agree(&frame, &parameters(1), 4, 17);
    let (colour, _) = synthetic::colour_frame(21, 30, 9, (2, 2), &data);
    agree(&colour, &parameters(3), 3, 18);
}

/// The current point of TGV in the natural layout.
#[derive(Clone)]
struct TgvState {
    x: Vec<f64>,
    w: Vector,
    p: Vector,
    r: Tensor,
}

/// The outputs of an iteration of TGV's definition.
struct TgvDefined {
    canvas: Vec<f64>,
    coefficients: Vec<Vec<f64>>,
    w: Vector,
    p: Vector,
    r: Tensor,
}

/// The differences of a plane: `(forward across, forward down, backward across,
/// backward down)`.
fn differences(plane: &[f64], shape: Shape) -> [Vec<f64>; 4] {
    let size = plane.len();
    let mut result = [vec![0.0; size], vec![0.0; size], vec![0.0; size], vec![0.0; size]];
    operators::forward_x(plane, shape, &mut result[0]);
    operators::forward_y(plane, shape, &mut result[1]);
    operators::backward_x(plane, shape, &mut result[2]);
    operators::backward_y(plane, shape, &mut result[3]);
    result
}

/// One iteration of TGV in the natural layout, step after step.
fn iterate_tgv(
    frame: &Frame,
    steps: &[Step],
    given: &Parameters,
    radii: (f64, f64),
    state: &mut TgvState,
) -> TgvDefined {
    let shape = Shape {
        height: frame.height(),
        width: frame.width(),
    };
    let plane = frame.plane();
    let size = frame.samples();
    // x~ = prox(x + tau gamma div p).
    let mut work = vec![0.0; size];
    for (channel, &gamma) in given.gammas.iter().enumerate() {
        let range = channel * plane..(channel + 1) * plane;
        let px = differences(&state.p.x[range.clone()], shape);
        let py = differences(&state.p.y[range.clone()], shape);
        for (i, index) in range.enumerate() {
            work[index] = descent(state.x[index], given.tau * gamma, px[2][i], py[3][i]);
        }
    }
    let mut coefficients: Vec<Vec<f64>> = frame
        .channels()
        .iter()
        .map(|channel| vec![0.0; channel.problem().samples()])
        .collect();
    let mut canvas = vec![0.0; size];
    frames::prox(frame, steps, &work, &mut coefficients, &mut canvas);
    // w~ = w + tau gamma (p + div2 r).
    let mut w = Vector::zeros(size);
    for (channel, &gamma) in given.gammas.iter().enumerate() {
        let range = channel * plane..(channel + 1) * plane;
        let xx = differences(&state.r.xx[range.clone()], shape);
        let yy = differences(&state.r.yy[range.clone()], shape);
        let xy = differences(&state.r.xy[range.clone()], shape);
        for (i, index) in range.enumerate() {
            let step = given.tau * gamma;
            w.x[index] = advanced(state.w.x[index], step, state.p.x[index], xx[0][i], xy[1][i]);
            w.y[index] = advanced(state.w.y[index], step, state.p.y[index], xy[0][i], yy[1][i]);
        }
    }
    // The extrapolated point and field.
    let bar: Vec<f64> = canvas
        .iter()
        .zip(&state.x)
        .map(|(&new, &old)| extrapolated(new, old))
        .collect();
    let bar_w = Vector {
        x: w.x
            .iter()
            .zip(&state.w.x)
            .map(|(&new, &old)| extrapolated(new, old))
            .collect(),
        y: w.y
            .iter()
            .zip(&state.w.y)
            .map(|(&new, &old)| extrapolated(new, old))
            .collect(),
    };
    // p~ = proj(p + sigma gamma (grad bar - bar_w)), r~ = proj(r + sigma gamma E bar_w).
    let mut p = Vector::zeros(size);
    let mut r = Tensor::zeros(size);
    for (channel, &gamma) in given.gammas.iter().enumerate() {
        let range = channel * plane..(channel + 1) * plane;
        let step = given.sigma * gamma;
        let gradient = differences(&bar[range.clone()], shape);
        let field_x = differences(&bar_w.x[range.clone()], shape);
        let field_y = differences(&bar_w.y[range.clone()], shape);
        for (i, index) in range.enumerate() {
            p.x[index] = ascent(state.p.x[index], step, gradient[0][i] - bar_w.x[index]);
            p.y[index] = ascent(state.p.y[index], step, gradient[1][i] - bar_w.y[index]);
            r.xx[index] = ascent(state.r.xx[index], step, field_x[2][i]);
            r.yy[index] = ascent(state.r.yy[index], step, field_y[3][i]);
            r.xy[index] = ascent(state.r.xy[index], step, f64::midpoint(field_x[3][i], field_y[2][i]));
        }
    }
    frames::project_vectors(frame, &mut p, radii.0, given.coupled);
    frames::project_tensors(frame, &mut r, radii.1, given.coupled);
    let rest = 1.0 - given.rho;
    let relax = |values: &mut [f64], new: &[f64]| {
        for (value, &fresh) in values.iter_mut().zip(new) {
            *value = relaxed(given.rho, rest, fresh, *value);
        }
    };
    relax(&mut state.x, &canvas);
    relax(&mut state.w.x, &w.x);
    relax(&mut state.w.y, &w.y);
    relax(&mut state.p.x, &p.x);
    relax(&mut state.p.y, &p.y);
    relax(&mut state.r.xx, &r.xx);
    relax(&mut state.r.yy, &r.yy);
    relax(&mut state.r.xy, &r.xy);
    TgvDefined {
        canvas,
        coefficients,
        w,
        p,
        r,
    }
}

/// Several iterations of TGV's sweep and definition, from a start of the data term's
/// centres, `w` its gradient, and random duals in their balls, compared after each.
fn agree_tgv(frame: &Frame, given: &Parameters, iterations: usize, seed: u64) {
    let layout = Layout::new(frame).expect("a layout");
    let steps = frames::steps(frame, given.tau);
    let start = frames::start(frame, None).expect("a start");
    let size = frame.samples();
    let mut numbers = Numbers::new(seed);
    let mut draw = |scale: f64| -> Vec<f64> { (0..size).map(|_| numbers.uniform(-scale, scale)).collect() };
    let radii = (given.alpha, 2.0 * given.alpha);
    let mut p = Vector {
        x: draw(2.0),
        y: draw(2.0),
    };
    frames::project_vectors(frame, &mut p, radii.0, given.coupled);
    let mut r = Tensor {
        xx: draw(3.0),
        yy: draw(3.0),
        xy: draw(3.0),
    };
    frames::project_tensors(frame, &mut r, radii.1, given.coupled);
    let mut state = TgvState {
        x: start.canvas.clone(),
        w: frames::gradient(frame, &start.canvas),
        p,
        r,
    };
    let planes = |values: &[f64]| layout.to_planes(values);
    let mut point = TgvPoint {
        x: planes(&state.x),
        wx: planes(&state.w.x),
        wy: planes(&state.w.y),
        px: planes(&state.p.x),
        py: planes(&state.p.y),
        rxx: planes(&state.r.xx),
        ryy: planes(&state.r.yy),
        rxy: planes(&state.r.xy),
    };
    let mut sweep = TgvSweep::new(
        frame,
        &layout,
        &steps,
        &given.gammas,
        given.tau,
        given.sigma,
        radii,
        given.coupled,
        given.rho,
    );
    let mut outputs = TgvOutputs::new(&layout);
    for iteration in 0..iterations {
        let expected = iterate_tgv(frame, &steps, given, radii, &mut state);
        let copy = point.clone();
        sweep.iterate(&mut point, Some(&mut outputs));
        let natural = |values: &[f64]| bits(&layout.to_natural(values));
        for (name, ours, theirs) in [
            ("x", &point.x, &state.x),
            ("w across", &point.wx, &state.w.x),
            ("w down", &point.wy, &state.w.y),
            ("p across", &point.px, &state.p.x),
            ("p down", &point.py, &state.p.y),
            ("r11", &point.rxx, &state.r.xx),
            ("r22", &point.ryy, &state.r.yy),
            ("r12", &point.rxy, &state.r.xy),
            ("x~", &outputs.first.canvas, &expected.canvas),
            ("w~ across", &outputs.wx, &expected.w.x),
            ("w~ down", &outputs.wy, &expected.w.y),
            ("p~ across", &outputs.first.px, &expected.p.x),
            ("p~ down", &outputs.first.py, &expected.p.y),
            ("r~11", &outputs.rxx, &expected.r.xx),
            ("r~22", &outputs.ryy, &expected.r.yy),
            ("r~12", &outputs.rxy, &expected.r.xy),
        ] {
            assert_eq!(natural(ours), bits(theirs), "{name} after iteration {iteration}");
        }
        for ((part, planar), natural) in layout
            .parts
            .iter()
            .zip(&outputs.first.coefficients)
            .zip(&expected.coefficients)
        {
            let mut converted = vec![0.0; natural.len()];
            layout.write_values_natural(part, planar, &mut converted);
            assert_eq!(bits(&converted), bits(natural), "coefficients at {iteration}");
        }
        // Without outputs, the point moves alike.
        let mut again = copy;
        sweep.iterate(&mut again, None);
        assert_eq!(bits(&again.x), bits(&point.x), "x without outputs at {iteration}");
        assert_eq!(bits(&again.rxy), bits(&point.rxy), "r12 without outputs at {iteration}");
    }
}

#[test]
fn tgv_sweeps_as_it_is_defined() {
    let frame = Frame::one(synthetic::problem(24, 40, 21, 2.0, &DataTerm::default()).0);
    agree_tgv(&frame, &parameters(1), 4, 31);
    let narrow = Frame::one(synthetic::problem(8, 8, 22, 2.0, &DataTerm::default()).0);
    agree_tgv(&narrow, &parameters(1), 3, 32);
    let (colour, _) = synthetic::colour_frame(21, 30, 23, (2, 2), &DataTerm::default());
    agree_tgv(&colour, &parameters(3), 3, 33);
    let mut apart = parameters(3);
    apart.coupled = false;
    apart.gammas = vec![1.0, 0.5, 2.0];
    agree_tgv(&colour, &apart, 3, 34);
}

/// The start and its gradient in the planes are those of the natural layout, to the last
/// bit: the same centres, clipped alike, and the same inverse DCT.
fn start_agrees(frame: &Frame, given: Option<&[Vec<f64>]>) {
    let layout = Layout::new(frame).expect("a layout");
    let natural = frames::start(frame, given).expect("a start");
    let (canvas, coefficients) = jpeg_unround::sweep::start(frame, &layout, given).expect("a start");
    assert_eq!(bits(&layout.to_natural(&canvas)), bits(&natural.canvas), "the canvas");
    for ((part, planar), own) in layout.parts.iter().zip(&coefficients).zip(&natural.coefficients) {
        let mut converted = vec![0.0; own.len()];
        layout.write_values_natural(part, planar, &mut converted);
        assert_eq!(bits(&converted), bits(own), "the coefficients");
    }
    let (across, down) = jpeg_unround::sweep::gradient(&layout, &canvas);
    let gradient = frames::gradient(frame, &natural.canvas);
    assert_eq!(
        bits(&layout.to_natural(&across)),
        bits(&gradient.x),
        "the gradient across"
    );
    assert_eq!(bits(&layout.to_natural(&down)), bits(&gradient.y), "the gradient down");
}

#[test]
fn the_start_is_that_of_the_natural_layout() {
    let frame = Frame::one(synthetic::problem(24, 40, 71, 2.0, &DataTerm::default()).0);
    start_agrees(&frame, None);
    let (colour, _) = synthetic::colour_frame(21, 30, 72, (2, 2), &DataTerm::default());
    start_agrees(&colour, None);
    let midpoint = DataTerm {
        centres: jpeg_unround::model::Centres::Midpoint,
        slack: 0.5,
        ..DataTerm::default()
    };
    let (wide, _) = synthetic::colour_frame(20, 36, 73, (1, 2), &midpoint);
    start_agrees(&wide, None);
    // Coefficients given, some beyond their intervals, are clipped alike.
    let mut numbers = Numbers::new(74);
    let given: Vec<Vec<f64>> = colour
        .channels()
        .iter()
        .map(|channel| {
            let problem = channel.problem();
            problem
                .centres()
                .iter()
                .map(|&centre| centre + numbers.uniform(-40.0, 40.0))
                .collect()
        })
        .collect();
    start_agrees(&colour, Some(&given));
}

/// The conversions between the layouts put each value at its place: a sample of
/// column `c` at `(c mod P) M + c div P` of its row, and a coefficient where
/// `coefficient_place` says; the levels of each part are the problem's in the planes.
/// The values are moved, not computed, and compared to the last bit.
#[test]
fn the_conversions_put_each_value_at_its_place() {
    let count = |index: usize| f64::from(u32::try_from(index).expect("a count fits in u32"));
    for frame in [
        Frame::one(synthetic::problem(16, 40, 81, 2.0, &DataTerm::default()).0),
        synthetic::colour_frame(21, 30, 82, (2, 2), &DataTerm::default()).0,
        synthetic::colour_frame(20, 36, 83, (1, 2), &DataTerm::default()).0,
        synthetic::colour_frame(20, 36, 84, (2, 1), &DataTerm::default()).0,
    ] {
        let layout = Layout::new(&frame).expect("a layout");
        let natural: Vec<f64> = (0..frame.samples()).map(count).collect();
        let planar = layout.to_planes(&natural);
        for (row, (source, target)) in natural
            .chunks(layout.width)
            .zip(planar.chunks(layout.width))
            .enumerate()
        {
            for (column, &value) in source.iter().enumerate() {
                assert_eq!(
                    target[layout.place(column)].to_bits(),
                    value.to_bits(),
                    "row {row}, column {column}"
                );
            }
        }
        assert_eq!(bits(&layout.to_natural(&planar)), bits(&natural));
        for (part, channel) in layout.parts.iter().zip(frame.channels()) {
            let problem = channel.problem();
            let values: Vec<f64> = (1..=problem.samples()).map(count).collect();
            let planes = layout.values_to_planes(part, &values);
            let mut expected = vec![0.0; planes.len()];
            for (index, &value) in values.iter().enumerate() {
                expected[layout.coefficient_place(part, index)] = value;
            }
            assert_eq!(bits(&planes), bits(&expected));
            let mut back = vec![0.0; values.len()];
            layout.write_values_natural(part, &planes, &mut back);
            assert_eq!(bits(&back), bits(&values));
            assert_eq!(&part.levels[..], &layout.values_to_planes(part, problem.levels())[..]);
        }
    }
}
