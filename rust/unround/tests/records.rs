// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! The values of the records in the planes (`records`) against those of the natural
//! layout (`frames`). Every term is computed by the same formula from the same values
//! in both, so the terms agree to the last bit and only the order of their sums
//! differs. A sum of `n` terms in the order of the natural layout rounds within
//! `gamma(128 + 2 ceil(log2(n / 128)))` of their magnitudes, and one in the order of
//! the planes within `gamma(ceil(W / 8) + 7 + 128 + 2 ceil(log2 runs))`, the lanes of a
//! row's run of terms and then the runs' sums; with the few additions of the parts
//! after them, `gamma(ceil(W / 8) + 300)` bounds each for any count of terms below
//! 2^64, and the two differ by at most twice that times the sum of the terms'
//! magnitudes, which the tests compute from the natural layout.

mod support;

use jpeg_unround::dct;
use jpeg_unround::frames::{self, FREE_CENTRE, Frame, Tensor, Vector};
use jpeg_unround::model::{DataTerm, Tgv, Tv};
use jpeg_unround::planar::Layout;
use jpeg_unround::records::Values;
use jpeg_unround::sweep::{TgvOutputs, TgvPoint, TgvSweep, TvOutputs, TvPoint, TvSweep};
use support::rounding::gamma;
use support::synthetic::{self, Numbers};

/// Twice the bound of either order's rounding, per unit of the terms' magnitudes.
fn tolerance(layout: &Layout) -> f64 {
    2.0 * gamma(layout.width.div_ceil(8) + 300)
}

/// The sum of the magnitudes of the terms of `G*(xi)` bounded over the box of the
/// radius, in the natural layout (`frames::conjugate`'s terms).
fn conjugate_magnitude(frame: &Frame, xi: &[f64], radius: f64) -> f64 {
    let plane = frame.plane();
    let mut total = 0.0;
    for (index, channel) in frame.channels().iter().enumerate() {
        let own = &xi[index * plane..(index + 1) * plane];
        let problem = channel.problem();
        let summed = frames::sums(channel, own, frame.width());
        let coefficients = dct::forward(&summed, problem.height(), problem.width());
        total += coefficients
            .iter()
            .enumerate()
            .map(|(place, &s)| problem.conjugate_one(place, s).abs())
            .sum::<f64>();
        if frame.free(channel) {
            let averaged = frames::spread(frame, channel, &frames::means(channel, own, frame.width()));
            let magnitude: f64 = own
                .iter()
                .zip(&averaged)
                .map(|(&sample, &average)| (sample - average).abs())
                .sum();
            total += (FREE_CENTRE + radius) * magnitude;
        }
    }
    total
}

/// `gamma div field` of every channel, in the natural layout.
fn divergence(frame: &Frame, gammas: &[f64], field: &Vector) -> Vec<f64> {
    let plane = frame.plane();
    let shape = frame.shape();
    let mut result = vec![0.0; frame.samples()];
    for (channel, &gamma) in gammas.iter().enumerate() {
        let range = channel * plane..(channel + 1) * plane;
        let target = &mut result[range.clone()];
        jpeg_unround::operators::div(&field.x[range.clone()], &field.y[range], shape, target);
        for value in target {
            *value *= gamma;
        }
    }
    result
}

fn assert_close(name: &str, ours: f64, theirs: f64, magnitude: f64, layout: &Layout) {
    let bound = tolerance(layout) * magnitude;
    assert!(
        (ours - theirs).abs() <= bound,
        "{name}: {ours} in the planes, {theirs} in the natural layout, apart by {} where the bound is {bound}",
        (ours - theirs).abs()
    );
}

/// TV's values of a few iterations of the sweep, in the planes and in the natural layout.
fn tv_agree(frame: &Frame, weights: &Tv, radius: f64, seed: u64) {
    let layout = Layout::new(frame).expect("a layout");
    let gammas = frame
        .channel_weights(weights.channel_weights.as_deref())
        .expect("weights");
    let steps = frames::steps(frame, 0.3);
    let start = frames::start(frame, None).expect("a start");
    let size = frame.samples();
    let mut numbers = Numbers::new(seed);
    let mut p = Vector {
        x: (0..size).map(|_| numbers.uniform(-2.0, 2.0)).collect(),
        y: (0..size).map(|_| numbers.uniform(-2.0, 2.0)).collect(),
    };
    frames::project_vectors(frame, &mut p, weights.alpha, weights.coupled);
    let mut point = TvPoint {
        x: layout.to_planes(&start.canvas),
        px: layout.to_planes(&p.x),
        py: layout.to_planes(&p.y),
    };
    let mut sweep = TvSweep::new(
        frame,
        &layout,
        &steps,
        &gammas,
        0.3,
        0.2,
        weights.alpha,
        weights.coupled,
        1.8,
    );
    let mut outputs = TvOutputs::new(&layout);
    let mut values = Values::new(frame, &layout, &gammas);
    for _ in 0..3 {
        sweep.iterate(&mut point, Some(&mut outputs));
    }
    let canvas = layout.to_natural(&outputs.canvas);
    let dual_field = Vector {
        x: layout.to_natural(&outputs.px),
        y: layout.to_natural(&outputs.py),
    };
    let coefficients: Vec<Vec<f64>> = layout
        .parts
        .iter()
        .zip(&outputs.coefficients)
        .map(|(part, planar)| {
            let mut natural = vec![0.0; part.rows * part.columns * 64];
            layout.write_values_natural(part, planar, &mut natural);
            natural
        })
        .collect();
    let (primal, dual) =
        frames::tv_values(frame, weights, &coefficients, &canvas, &dual_field, radius).expect("values");
    let (ours_primal, ours_dual) = values.tv(weights, &outputs, radius);
    assert_close(
        "TV's primal value",
        ours_primal,
        primal,
        primal * (1.0 + tolerance(&layout)),
        &layout,
    );
    let xi = divergence(frame, &gammas, &dual_field);
    assert_close(
        "TV's dual value",
        ours_dual,
        dual,
        conjugate_magnitude(frame, &xi, radius),
        &layout,
    );
}

/// TGV's outputs in the natural layout: the canvas, `w`, `p`, `r` and each component's
/// coefficients.
fn natural_tgv(layout: &Layout, outputs: &TgvOutputs) -> (Vec<f64>, Vector, Vector, Tensor, Vec<Vec<f64>>) {
    let natural = |values: &[f64]| layout.to_natural(values);
    let w = Vector {
        x: natural(&outputs.wx),
        y: natural(&outputs.wy),
    };
    let p = Vector {
        x: natural(&outputs.first.px),
        y: natural(&outputs.first.py),
    };
    let r = Tensor {
        xx: natural(&outputs.rxx),
        yy: natural(&outputs.ryy),
        xy: natural(&outputs.rxy),
    };
    let coefficients = layout
        .parts
        .iter()
        .zip(&outputs.first.coefficients)
        .map(|(part, planar)| {
            let mut own = vec![0.0; part.rows * part.columns * 64];
            layout.write_values_natural(part, planar, &mut own);
            own
        })
        .collect();
    (natural(&outputs.first.canvas), w, p, r, coefficients)
}

/// TGV's values of a few iterations of the sweep, in the planes and in the natural layout.
fn tgv_agree(frame: &Frame, weights: &Tgv, radius: f64, seed: u64) {
    let layout = Layout::new(frame).expect("a layout");
    let gammas = frame
        .channel_weights(weights.channel_weights.as_deref())
        .expect("weights");
    let steps = frames::steps(frame, 0.3);
    let start = frames::start(frame, None).expect("a start");
    let size = frame.samples();
    let mut numbers = Numbers::new(seed);
    let mut draw = |scale: f64| -> Vec<f64> { (0..size).map(|_| numbers.uniform(-scale, scale)).collect() };
    let mut p = Vector {
        x: draw(2.0),
        y: draw(2.0),
    };
    frames::project_vectors(frame, &mut p, weights.alpha1, weights.coupled);
    let mut r = Tensor {
        xx: draw(3.0),
        yy: draw(3.0),
        xy: draw(3.0),
    };
    frames::project_tensors(frame, &mut r, weights.alpha0, weights.coupled);
    let w = frames::gradient(frame, &start.canvas);
    let planes = |values: &[f64]| layout.to_planes(values);
    let mut point = TgvPoint {
        x: planes(&start.canvas),
        wx: planes(&w.x),
        wy: planes(&w.y),
        px: planes(&p.x),
        py: planes(&p.y),
        rxx: planes(&r.xx),
        ryy: planes(&r.yy),
        rxy: planes(&r.xy),
    };
    let mut sweep = TgvSweep::new(
        frame,
        &layout,
        &steps,
        &gammas,
        0.3,
        0.2,
        (weights.alpha1, weights.alpha0),
        weights.coupled,
        1.8,
    );
    let mut outputs = TgvOutputs::new(&layout);
    let mut values = Values::new(frame, &layout, &gammas);
    for _ in 0..3 {
        sweep.iterate(&mut point, Some(&mut outputs));
    }
    let (canvas, w_out, p_out, r_out, coefficients) = natural_tgv(&layout, &outputs);
    let (primal, dual, theta) =
        frames::tgv_values(frame, weights, &coefficients, &canvas, &w_out, &r_out, radius).expect("values");
    let (ours_primal, ours_dual, ours_theta) = values.tgv(weights, &outputs, radius);
    assert_close(
        "TGV's primal value",
        ours_primal,
        primal,
        primal * (1.0 + tolerance(&layout)),
        &layout,
    );
    // theta is the radius over the largest norm, whose terms are the same in both: a
    // largest value is exact whatever the order.
    assert_eq!(ours_theta.to_bits(), theta.to_bits(), "theta");
    let mut scaled = frames::tensor_divergence(frame, &r_out);
    for value in scaled.x.iter_mut().chain(scaled.y.iter_mut()) {
        *value *= -theta;
    }
    let xi = divergence(frame, &gammas, &scaled);
    assert_close(
        "TGV's dual value",
        ours_dual,
        dual,
        conjugate_magnitude(frame, &xi, radius),
        &layout,
    );
    let residual = frames::tgv_residual(frame, weights, &p_out, &r_out).expect("a residual");
    let bound = frames::dual_bound(frame, weights.channel_weights.as_deref(), &p_out, radius).expect("a bound");
    let (ours_bound, ours_residual) = values.tgv_partial(&outputs, radius);
    assert_close(
        "TGV's residual",
        ours_residual,
        residual,
        residual * (1.0 + tolerance(&layout)),
        &layout,
    );
    let xi = divergence(frame, &gammas, &p_out);
    assert_close(
        "TGV's dual bound",
        ours_bound,
        bound,
        conjugate_magnitude(frame, &xi, radius),
        &layout,
    );
}

#[test]
fn the_values_of_tv_are_those_of_the_natural_layout() {
    let frame = Frame::one(synthetic::problem(24, 40, 41, 2.0, &DataTerm::default()).0);
    tv_agree(&frame, &Tv::default(), 255.0, 51);
    let (colour, _) = synthetic::colour_frame(21, 30, 42, (2, 2), &DataTerm::default());
    tv_agree(&colour, &Tv::default(), 255.0, 52);
    let apart = Tv {
        alpha: 0.7,
        channel_weights: Some(vec![1.0, 0.5, 2.0]),
        coupled: false,
    };
    tv_agree(&colour, &apart, 100.0, 53);
    let charged = DataTerm {
        slack: 0.5,
        slack_cost: 3.0,
        dc_weight: 2.0,
        ..DataTerm::default()
    };
    let (slack, _) = synthetic::colour_frame(21, 30, 43, (2, 2), &charged);
    tv_agree(&slack, &Tv::default(), 255.0, 54);
}

#[test]
fn the_values_of_tgv_are_those_of_the_natural_layout() {
    let frame = Frame::one(synthetic::problem(24, 40, 44, 2.0, &DataTerm::default()).0);
    tgv_agree(&frame, &Tgv::default(), 255.0, 61);
    let (colour, _) = synthetic::colour_frame(21, 30, 45, (2, 2), &DataTerm::default());
    tgv_agree(&colour, &Tgv::default(), 255.0, 62);
    let apart = Tgv {
        alpha1: 0.7,
        alpha0: 1.5,
        channel_weights: Some(vec![1.0, 0.5, 2.0]),
        coupled: false,
    };
    tgv_agree(&colour, &apart, 100.0, 63);
}
