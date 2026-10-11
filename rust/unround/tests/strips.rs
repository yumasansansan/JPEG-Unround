// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! The sweep in strips of bands, by several threads at once, against the sweep by one:
//! each strip computes what the one sweep computes of its rows, from the same values,
//! the rows beside it as they were when the iteration began, and so the point and the
//! outputs agree to the last bit, whatever the strips. The solvers on several threads,
//! their records taken on the threads too, stop where one thread stops, with the same
//! records and the same result.

mod support;

use std::ops::Range;

use jpeg_unround::frames::{self, Frame, Tensor, Vector};
use jpeg_unround::model::{DataTerm, Tgv, Tv};
use jpeg_unround::pdhg::{self, Options};
use jpeg_unround::planar::Layout;
use jpeg_unround::results::{FrameResult, Stop};
use jpeg_unround::sweep::{self, TgvOutputs, TgvPoint, TgvSweep, TvOutputs, TvPoint, TvSweep};
use support::synthetic::{self, Numbers};

const TAU: f64 = 0.37;
const SIGMA: f64 = 0.21;
const ALPHA: f64 = 1.3;
const RHO: f64 = 1.7;

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// The threads to sweep with: one, several, as many as the bands, and more.
fn thread_counts(bands: usize) -> Vec<usize> {
    vec![1, 2, 3, 4, 7, bands, bands + 3]
}

#[test]
fn the_strips_take_every_band_once_and_evenly() {
    for bands in 1..20 {
        for threads in 1..25 {
            let strips: Vec<Range<usize>> = sweep::strips(bands, threads);
            assert_eq!(strips.len(), threads.min(bands), "{bands} bands, {threads} threads");
            assert_eq!(strips[0].start, 0);
            assert_eq!(strips[strips.len() - 1].end, bands);
            for pair in strips.windows(2) {
                assert_eq!(pair[0].end, pair[1].start);
            }
            let sizes: Vec<usize> = strips.iter().map(ExactSizeIterator::len).collect();
            let (least, most) = (sizes.iter().min(), sizes.iter().max());
            assert!(least.is_some_and(|&least| least >= 1));
            assert!(most.is_some_and(|&most| most - least.copied().unwrap_or(0) <= 1));
        }
    }
}

/// A frame's TV point: its start, and a dual drawn in its balls.
fn tv_point(frame: &Frame, layout: &Layout, coupled: bool, seed: u64) -> TvPoint {
    let start = frames::start(frame, None).expect("a start");
    let size = frame.samples();
    let mut numbers = Numbers::new(seed);
    let mut p = Vector {
        x: (0..size).map(|_| numbers.uniform(-2.0, 2.0)).collect(),
        y: (0..size).map(|_| numbers.uniform(-2.0, 2.0)).collect(),
    };
    frames::project_vectors(frame, &mut p, ALPHA, coupled);
    TvPoint {
        x: layout.to_planes(&start.canvas),
        px: layout.to_planes(&p.x),
        py: layout.to_planes(&p.y),
    }
}

fn tv_sweep(frame: &Frame, layout: &Layout, gammas: &[f64], coupled: bool) -> TvSweep {
    let steps = frames::steps(frame, TAU);
    TvSweep::new(frame, layout, &steps, gammas, TAU, SIGMA, ALPHA, coupled, RHO)
}

/// TV's point and outputs after `iterations` by each count of threads, and by one
/// iteration after another, to the last bit.
fn tv_agrees(frame: &Frame, gammas: &[f64], coupled: bool, seed: u64) {
    let layout = Layout::new(frame).expect("a layout");
    let iterations = 3;
    let mut expected = tv_point(frame, &layout, coupled, seed);
    let mut expected_outputs = TvOutputs::new(&layout);
    let mut one = tv_sweep(frame, &layout, gammas, coupled);
    for done in 0..iterations {
        one.iterate(&mut expected, (done + 1 == iterations).then_some(&mut expected_outputs));
    }
    for threads in thread_counts(layout.bands()) {
        let mut point = tv_point(frame, &layout, coupled, seed);
        let mut outputs = TvOutputs::new(&layout);
        let mut sweep = tv_sweep(frame, &layout, gammas, coupled);
        // Two iterations and then one: the outputs are the last one's.
        sweep.run(&mut point, None, iterations - 1, threads);
        sweep.run(&mut point, Some(&mut outputs), 1, threads);
        for (name, ours, theirs) in [
            ("x", &point.x, &expected.x),
            ("p across", &point.px, &expected.px),
            ("p down", &point.py, &expected.py),
            ("x~", &outputs.canvas, &expected_outputs.canvas),
            ("p~ across", &outputs.px, &expected_outputs.px),
            ("p~ down", &outputs.py, &expected_outputs.py),
        ] {
            assert_eq!(bits(ours), bits(theirs), "{name}, {threads} threads");
        }
        for (ours, theirs) in outputs.coefficients.iter().zip(&expected_outputs.coefficients) {
            assert_eq!(bits(ours), bits(theirs), "coefficients, {threads} threads");
        }
    }
}

/// The data term of flat blocks of their own, with a slack that costs.
fn flat_data() -> DataTerm {
    DataTerm {
        flat_weight: 0.1,
        flat_levels: 2,
        flat_dc: None,
        slack: 0.5,
        slack_cost: 3.0,
        ..DataTerm::default()
    }
}

#[test]
fn tv_sweeps_in_strips_as_in_one() {
    // Nine bands of 8 rows, and five of 16 rows (4:2:0), the last cut by the picture.
    let grey = Frame::one(synthetic::problem(72, 40, 41, 2.0, &DataTerm::default()).0);
    tv_agrees(&grey, &[1.0], true, 51);
    let (colour, _) = synthetic::colour_frame(75, 30, 42, (2, 2), &DataTerm::default());
    tv_agrees(&colour, &[1.0; 3], true, 52);
    tv_agrees(&colour, &[1.0, 0.5, 2.0], false, 53);
    let flat = Frame::one(synthetic::problem(72, 40, 43, 2.0, &flat_data()).0);
    tv_agrees(&flat, &[1.0], true, 54);
    let (flat_colour, _) = synthetic::colour_frame(75, 30, 44, (2, 2), &flat_data());
    tv_agrees(&flat_colour, &[1.0; 3], true, 55);
}

/// A frame's TGV point: its start, the gradient of its canvas, and duals drawn in their
/// balls.
fn tgv_point(frame: &Frame, layout: &Layout, coupled: bool, seed: u64) -> TgvPoint {
    let start = frames::start(frame, None).expect("a start");
    let size = frame.samples();
    let mut numbers = Numbers::new(seed);
    let mut draw = |scale: f64| -> Vec<f64> { (0..size).map(|_| numbers.uniform(-scale, scale)).collect() };
    let mut p = Vector {
        x: draw(2.0),
        y: draw(2.0),
    };
    frames::project_vectors(frame, &mut p, ALPHA, coupled);
    let mut r = Tensor {
        xx: draw(3.0),
        yy: draw(3.0),
        xy: draw(3.0),
    };
    frames::project_tensors(frame, &mut r, 2.0 * ALPHA, coupled);
    let w = frames::gradient(frame, &start.canvas);
    let planes = |values: &[f64]| layout.to_planes(values);
    TgvPoint {
        x: planes(&start.canvas),
        wx: planes(&w.x),
        wy: planes(&w.y),
        px: planes(&p.x),
        py: planes(&p.y),
        rxx: planes(&r.xx),
        ryy: planes(&r.yy),
        rxy: planes(&r.xy),
    }
}

fn tgv_sweep(frame: &Frame, layout: &Layout, gammas: &[f64], coupled: bool) -> TgvSweep {
    let steps = frames::steps(frame, TAU);
    TgvSweep::new(
        frame,
        layout,
        &steps,
        gammas,
        TAU,
        SIGMA,
        (ALPHA, 2.0 * ALPHA),
        coupled,
        RHO,
    )
}

/// TGV's point and outputs after `iterations` by each count of threads, and by one
/// iteration after another, to the last bit.
fn tgv_agrees(frame: &Frame, gammas: &[f64], coupled: bool, seed: u64) {
    let layout = Layout::new(frame).expect("a layout");
    let iterations = 3;
    let mut expected = tgv_point(frame, &layout, coupled, seed);
    let mut expected_outputs = TgvOutputs::new(&layout);
    let mut one = tgv_sweep(frame, &layout, gammas, coupled);
    for done in 0..iterations {
        one.iterate(&mut expected, (done + 1 == iterations).then_some(&mut expected_outputs));
    }
    for threads in thread_counts(layout.bands()) {
        let mut point = tgv_point(frame, &layout, coupled, seed);
        let mut outputs = TgvOutputs::new(&layout);
        let mut sweep = tgv_sweep(frame, &layout, gammas, coupled);
        sweep.run(&mut point, None, 1, threads);
        sweep.run(&mut point, Some(&mut outputs), iterations - 1, threads);
        for (name, ours, theirs) in [
            ("x", &point.x, &expected.x),
            ("w across", &point.wx, &expected.wx),
            ("w down", &point.wy, &expected.wy),
            ("p across", &point.px, &expected.px),
            ("p down", &point.py, &expected.py),
            ("r11", &point.rxx, &expected.rxx),
            ("r22", &point.ryy, &expected.ryy),
            ("r12", &point.rxy, &expected.rxy),
            ("x~", &outputs.first.canvas, &expected_outputs.first.canvas),
            ("p~ across", &outputs.first.px, &expected_outputs.first.px),
            ("p~ down", &outputs.first.py, &expected_outputs.first.py),
            ("w~ across", &outputs.wx, &expected_outputs.wx),
            ("w~ down", &outputs.wy, &expected_outputs.wy),
            ("r~11", &outputs.rxx, &expected_outputs.rxx),
            ("r~22", &outputs.ryy, &expected_outputs.ryy),
            ("r~12", &outputs.rxy, &expected_outputs.rxy),
        ] {
            assert_eq!(bits(ours), bits(theirs), "{name}, {threads} threads");
        }
        for (ours, theirs) in outputs
            .first
            .coefficients
            .iter()
            .zip(&expected_outputs.first.coefficients)
        {
            assert_eq!(bits(ours), bits(theirs), "coefficients, {threads} threads");
        }
    }
}

#[test]
fn tgv_sweeps_in_strips_as_in_one() {
    let grey = Frame::one(synthetic::problem(72, 40, 61, 2.0, &DataTerm::default()).0);
    tgv_agrees(&grey, &[1.0], true, 71);
    let (colour, _) = synthetic::colour_frame(75, 30, 62, (2, 2), &DataTerm::default());
    tgv_agrees(&colour, &[1.0; 3], true, 72);
    tgv_agrees(&colour, &[1.0, 0.5, 2.0], false, 73);
    let (wide, _) = synthetic::colour_frame(41, 36, 63, (1, 2), &DataTerm::default());
    tgv_agrees(&wide, &[1.0; 3], true, 74);
    let flat = Frame::one(synthetic::problem(72, 40, 64, 2.0, &flat_data()).0);
    tgv_agrees(&flat, &[1.0], true, 75);
    let (flat_colour, _) = synthetic::colour_frame(75, 30, 65, (2, 2), &flat_data());
    tgv_agrees(&flat_colour, &[1.0; 3], true, 76);
}

/// Everything of a solver's result but its seconds, as bits.
fn result_bits(result: &FrameResult) -> Vec<u64> {
    let mut values = Vec::new();
    for coefficients in &result.primal.coefficients {
        values.extend(bits(coefficients));
    }
    values.extend(bits(&result.primal.canvas));
    if let Some(w) = &result.primal.w {
        values.extend(bits(&w.x));
        values.extend(bits(&w.y));
    }
    if let Some(dual) = &result.dual {
        values.extend(bits(&dual.p.x));
        values.extend(bits(&dual.p.y));
        if let Some(r) = &dual.r {
            values.extend(bits(&r.xx));
            values.extend(bits(&r.yy));
            values.extend(bits(&r.xy));
        }
    }
    let history = &result.history;
    values.extend(history.iterations.iter().copied());
    for column in [&history.primal, &history.dual, &history.scaling, &history.partial_gap] {
        values.extend(bits(column));
    }
    values.push(result.iterations);
    values
}

#[test]
fn the_solvers_on_threads_end_where_one_ends() {
    let (colour, _) = synthetic::colour_frame(75, 30, 101, (2, 2), &DataTerm::default());
    let grey = Frame::one(synthetic::problem(72, 40, 102, 2.0, &DataTerm::default()).0);
    for frame in [&colour, &grey] {
        // A tolerance that a record meets after a few, and a most that it does not.
        for (tolerance, iterations) in [(0.05, 400), (0.0, 37)] {
            let options = |threads: usize| Options {
                iterations: Some(iterations),
                tolerance: Some(tolerance),
                record_every: 7,
                threads,
                ..Options::default()
            };
            let tv = |threads: usize| pdhg::solve_tv(frame, &Tv::default(), &options(threads), None, None).expect("TV");
            let tgv =
                |threads: usize| pdhg::solve_tgv(frame, &Tgv::default(), &options(threads), None, None).expect("TGV");
            let solvers: [(&str, &dyn Fn(usize) -> FrameResult); 2] = [("TV", &tv), ("TGV", &tgv)];
            // The first stops at a record before its most, beyond iterations already run.
            let expected = if tolerance > 0.0 {
                Stop::Converged
            } else {
                Stop::Iterations
            };
            for (name, solve) in solvers {
                let one = solve(1);
                assert_eq!(one.stop, expected, "{name}");
                assert!(tolerance == 0.0 || one.iterations < iterations, "{name}");
                for threads in [2, 3, 8] {
                    let more = solve(threads);
                    assert_eq!(more.stop, one.stop, "{name}, {threads} threads");
                    assert_eq!(result_bits(&more), result_bits(&one), "{name}, {threads} threads");
                }
            }
        }
    }
}
