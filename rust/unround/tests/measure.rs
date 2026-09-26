// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! Measures what the regression checks of the solvers compare with: run by hand,
//! `cargo test --release --test measure -- --ignored --nocapture`, when a solver or
//! the synthetic problems change.

mod support;

use jpeg_unround::frames::Frame;
use jpeg_unround::model::{DataTerm, Tgv, Tv};
use jpeg_unround::pdhg::{self, Options};
use jpeg_unround::results::FrameResult;
use jpeg_unround::subgradient;
use support::synthetic;

fn per_sample(result: &FrameResult, samples: usize) -> f64 {
    let gap = result.history.gap();
    gap[gap.len() - 1] / f64::from(u32::try_from(samples).expect("fits"))
}

fn grey(seed: u64) -> Frame {
    Frame::one(synthetic::problem(16, 24, seed, 2.0, &DataTerm::default()).0)
}

fn fixed(iterations: u64, ratio: f64, relaxation: Option<f64>, record_every: u64) -> Options {
    Options {
        iterations: Some(iterations),
        tolerance: Some(0.0),
        step_ratio: Some(ratio),
        relaxation,
        record_every,
        ..Options::default()
    }
}

#[test]
#[ignore = "a measurement, run by hand"]
#[expect(
    clippy::too_many_lines,
    reason = "one measurement after another, as the tests take them"
)]
fn measure() {
    for ratio in [1.0, 30.0] {
        let frame = grey(12);
        let result =
            pdhg::solve_tv(&frame, &Tv::default(), &fixed(4000, ratio, None, 200), None, None).expect("a result");
        eprintln!(
            "tv ratio {ratio}: gap/sample {:.3e}",
            per_sample(&result, frame.samples())
        );
    }
    for relaxation in [1.0, 1.5, 1.9] {
        let frame = grey(18);
        let options = fixed(1000, 30.0, Some(relaxation), 100);
        let result = pdhg::solve_tv(&frame, &Tv::default(), &options, None, None).expect("a result");
        eprintln!(
            "tv relaxation {relaxation}: gap/sample {:.3e}",
            per_sample(&result, frame.samples())
        );
    }
    for relaxation in [1.0, 1.9] {
        let frame = grey(13);
        let options = Options {
            iterations: Some(5000),
            tolerance: Some(0.05),
            step_ratio: Some(10.0),
            relaxation: Some(relaxation),
            ..Options::default()
        };
        let result = pdhg::solve_tv(&frame, &Tv::default(), &options, None, None).expect("a result");
        eprintln!("tv stops, relaxation {relaxation}: {} iterations", result.iterations);
    }
    let frame = grey(13);
    let options = Options {
        iterations: Some(5000),
        tolerance: Some(0.0),
        relative_tolerance: 1e-4,
        step_ratio: Some(10.0),
        ..Options::default()
    };
    let result = pdhg::solve_tv(&frame, &Tv::default(), &options, None, None).expect("a result");
    eprintln!("tv relative: {} iterations", result.iterations);
    let frame = grey(15);
    let options = Options {
        iterations: Some(5000),
        tolerance: Some(0.0),
        step_ratio: Some(10.0),
        partial_radius: Some(20.0),
        partial_tolerance: 1e-2,
        ..Options::default()
    };
    let result = pdhg::solve_tgv(&frame, &Tgv::default(), &options, None, None).expect("a result");
    eprintln!("tgv partial: {} iterations", result.iterations);
    let long = fixed(6000, 10.0, None, 500);
    let result = pdhg::solve_tgv(&grey(15), &Tgv::default(), &long, None, None).expect("a result");
    let largest = result
        .primal
        .w
        .as_ref()
        .map(|w| w.x.iter().zip(&w.y).map(|(x, y)| x.hypot(*y)).fold(0.0, f64::max));
    eprintln!("tgv partial: largest |w| after 6000 iterations {largest:?}");
    for (relaxation, seed) in [(None, 14), (Some(1.9), 19)] {
        let frame = grey(seed);
        let result = pdhg::solve_tgv(&frame, &Tgv::default(), &fixed(6000, 10.0, relaxation, 500), None, None)
            .expect("a result");
        let gaps = result.history.gap();
        eprintln!(
            "tgv relaxation {relaxation:?}: gap/sample {:.3e}, first record {:.3e}, least gap {:.3e}, scaling {}",
            per_sample(&result, frame.samples()),
            gaps[1],
            gaps.iter().copied().fold(f64::INFINITY, f64::min),
            result.history.scaling[result.history.scaling.len() - 1]
        );
    }
    let frame = grey(16);
    let primal_dual =
        pdhg::solve_tv(&frame, &Tv::default(), &fixed(300, 10.0, None, 300), None, None).expect("a result");
    let stepped = subgradient::solve_tv(
        &frame,
        &Tv::default(),
        &subgradient::Options {
            iterations: 300,
            record_every: 300,
            ..subgradient::Options::default()
        },
        None,
        None,
    )
    .expect("a result");
    eprintln!(
        "tv against subgradient: dual {} primal {} subgradient {}",
        primal_dual.history.dual[1], primal_dual.history.primal[1], stepped.history.primal[1]
    );
    for (ratio, coupled) in [((2, 2), true), ((2, 2), false), ((1, 1), true)] {
        let (frame, _) = synthetic::colour_frame(20, 30, 70, ratio, &DataTerm::default());
        let weights = Tv {
            coupled,
            ..Tv::default()
        };
        let result =
            pdhg::solve_tv(&frame, &weights, &fixed(4000, 10.0, Some(1.0), 500), None, None).expect("a result");
        let gaps = result.history.gap();
        let canvas = &result.primal.canvas;
        let (low, high) = canvas
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(low, high), &value| {
                (low.min(value), high.max(value))
            });
        eprintln!(
            "colour tv {ratio:?} coupled {coupled}: gap/sample {:.3e}, least gap {:.3e}, first {:.3e}, canvas {low:.1} to {high:.1}",
            per_sample(&result, frame.samples()),
            gaps.iter().copied().fold(f64::INFINITY, f64::min),
            gaps[0]
        );
    }
    for coupled in [true, false] {
        let (frame, _) = synthetic::colour_frame(20, 30, 70, (2, 2), &DataTerm::default());
        let weights = Tgv {
            coupled,
            ..Tgv::default()
        };
        let result =
            pdhg::solve_tgv(&frame, &weights, &fixed(6000, 10.0, Some(1.0), 500), None, None).expect("a result");
        eprintln!(
            "colour tgv coupled {coupled}: gap/sample {:.3e}, scaling {}",
            per_sample(&result, frame.samples()),
            result.history.scaling[result.history.scaling.len() - 1]
        );
    }
}
