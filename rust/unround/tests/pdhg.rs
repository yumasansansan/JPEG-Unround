// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! The primal-dual method: its steps, its options, its iterates within the
//! constraint set, and its gaps (docs/math.md, 5 and 6).
//!
//! The iterates are checked against the constraint set within the round trip of the
//! DCT. How far the gaps fall in a number of iterations is not a theorem but a
//! measurement on these problems (`tests/measure.rs`), kept as a check that the
//! method still does as well: each threshold is about ten times what was measured,
//! and each count of iterations where a tolerance stops the method is the one
//! measured.

#![allow(
    clippy::float_cmp,
    reason = "the tests compare doubles to the last bit where the arithmetic is exact or the same"
)]

mod support;

use std::ops::ControlFlow;

use jpeg_unround::dct::{self, BLOCK, BLOCK_SIZE};
use jpeg_unround::frames::{self, Frame, Tensor, Vector};
use jpeg_unround::model::{DataTerm, Problem, Tgv, Tv};
use jpeg_unround::pdhg::{self, Initial, Options, Record, Weights};
use jpeg_unround::results::{FrameResult, Stop};
use jpeg_unround::subgradient;
use support::rounding::{self, U, gamma};
use support::synthetic::{self, Numbers};

fn grey(seed: u64) -> (Problem, Frame) {
    let (problem, _) = synthetic::problem(16, 24, seed, 2.0, &DataTerm::default());
    (problem.clone(), Frame::one(problem))
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

fn samples(frame: &Frame) -> f64 {
    f64::from(u32::try_from(frame.samples()).expect("fits"))
}

fn last_per_sample(result: &FrameResult, frame: &Frame) -> f64 {
    result.history.gap().last().copied().expect("records") / samples(frame)
}

#[test]
fn the_steps_keep_their_ratio_and_their_product() {
    // Each step is a square root of a product and a quotient: within 3U, and their quotient
    // and product within 8U.
    for (norm_squared, product) in [(pdhg::tgv_norm_squared(), pdhg::STEP_PRODUCT), (8.0, 0.5), (20.0, 0.9)] {
        for ratio in [0.1, 1.0, 37.0] {
            let (tau, sigma) = pdhg::steps(norm_squared, ratio, product).expect("steps");
            assert!((tau / sigma - ratio).abs() <= 8.0 * U * ratio);
            assert!((tau * sigma * norm_squared - product).abs() <= 8.0 * U * product);
        }
    }
    for (norm_squared, ratio, product) in [
        (8.0, 0.0, 0.99),
        (8.0, f64::INFINITY, 0.99),
        (8.0, f64::NAN, 0.99),
        (0.0, 1.0, 0.99),
        (8.0, 1.0, 1.0),
        (8.0, 1.0, 0.0),
    ] {
        let error = pdhg::steps(norm_squared, ratio, product).expect_err("refused");
        assert!(error.to_string().contains("positive"), "{error}");
    }
}

#[test]
fn the_defaults_scale_with_the_weight() {
    // What is not given is the model's default: the ratio over the weight squared, the
    // tolerance times it; what is given is kept.
    let tv = pdhg::plan(&Options::default(), Weights::Tv(&Tv::default())).expect("a plan");
    assert_eq!(
        (tv.iterations, tv.tolerance, tv.step_ratio, tv.relaxation),
        (
            pdhg::TV_ITERATIONS,
            pdhg::TV_TOLERANCE,
            pdhg::TV_RATIO,
            pdhg::TV_RELAXATION
        )
    );
    let tgv = pdhg::plan(&Options::default(), Weights::Tgv(&Tgv::default())).expect("a plan");
    assert_eq!(
        (tgv.iterations, tgv.tolerance, tgv.step_ratio, tgv.relaxation),
        (
            pdhg::TGV_ITERATIONS,
            pdhg::TGV_TOLERANCE,
            pdhg::TGV_RATIO,
            pdhg::TGV_RELAXATION
        )
    );
    let four = Tv {
        alpha: 4.0,
        ..Tv::default()
    };
    let scaled = pdhg::plan(&Options::default(), Weights::Tv(&four)).expect("a plan");
    assert_eq!(scaled.step_ratio, pdhg::TV_RATIO / 16.0);
    assert_eq!(scaled.tolerance, pdhg::TV_TOLERANCE * 4.0);
    let half = Tgv {
        alpha1: 0.5,
        alpha0: 1.0,
        ..Tgv::default()
    };
    let given = pdhg::plan(
        &Options {
            iterations: Some(7),
            tolerance: Some(0.0),
            step_ratio: Some(2.5),
            record_every: 3,
            ..Options::default()
        },
        Weights::Tgv(&half),
    )
    .expect("a plan");
    assert_eq!(
        (given.iterations, given.tolerance, given.step_ratio, given.record_every),
        (7, 0.0, 2.5, 3)
    );
    // Without the scaling, the defaults are those of the weight 1, whatever the weight.
    let unscaled = Options {
        scale_with_weight: false,
        ..Options::default()
    };
    let plain = pdhg::plan(&unscaled, Weights::Tv(&four)).expect("a plan");
    assert_eq!(
        (plain.step_ratio, plain.tolerance),
        (pdhg::TV_RATIO, pdhg::TV_TOLERANCE)
    );
    let plain = pdhg::plan(&unscaled, Weights::Tgv(&half)).expect("a plan");
    assert_eq!(
        (plain.step_ratio, plain.tolerance),
        (pdhg::TGV_RATIO, pdhg::TGV_TOLERANCE)
    );
}

#[test]
fn the_other_options_have_their_defaults_and_are_kept() {
    let tv = pdhg::plan(&Options::default(), Weights::Tv(&Tv::default())).expect("a plan");
    let tgv = pdhg::plan(&Options::default(), Weights::Tgv(&Tgv::default())).expect("a plan");
    for settled in [tv, tgv] {
        assert_eq!(
            (
                settled.step_product,
                settled.relative_tolerance,
                settled.partial_tolerance
            ),
            (0.99, 0.0, 0.0)
        );
        assert_eq!(settled.partial_radius, None);
        assert_eq!(settled.free_radius, frames::FREE_RADIUS);
    }
    assert_eq!(
        (tv.norm_squared, tgv.norm_squared),
        (pdhg::TV_NORM_SQUARED, pdhg::tgv_norm_squared())
    );
    let options = Options {
        relative_tolerance: 1e-3,
        partial_tolerance: 2e-3,
        step_product: 0.5,
        norm_squared: Some(12.0),
        partial_radius: Some(5.0),
        ..Options::default()
    };
    let given = pdhg::plan(&options, Weights::Tgv(&Tgv::default())).expect("a plan");
    assert_eq!(
        (given.relative_tolerance, given.partial_tolerance, given.step_product),
        (1e-3, 2e-3, 0.5)
    );
    assert_eq!((given.norm_squared, given.partial_radius), (12.0, Some(5.0)));
    // L^2 is the model's bound times the largest weight squared: 8 x 4 exactly.
    let weighted = Tv {
        channel_weights: Some(vec![1.0, 2.0, 0.5]),
        ..Tv::default()
    };
    assert_eq!(
        pdhg::plan(&Options::default(), Weights::Tv(&weighted))
            .expect("a plan")
            .norm_squared,
        32.0
    );
}

#[test]
fn the_plan_refuses_what_the_method_cannot_run_with() {
    let tv = Tv::default();
    let tgv = Tgv::default();
    let cases: Vec<(Options, Weights<'_>, &str)> = vec![
        (
            Options {
                tolerance: Some(-1e-3),
                ..Options::default()
            },
            Weights::Tv(&tv),
            "the tolerance",
        ),
        (
            Options {
                relative_tolerance: f64::NAN,
                ..Options::default()
            },
            Weights::Tgv(&tgv),
            "relative tolerance",
        ),
        (
            Options {
                step_ratio: Some(0.0),
                ..Options::default()
            },
            Weights::Tv(&tv),
            "ratio",
        ),
        (
            Options {
                step_ratio: Some(f64::INFINITY),
                ..Options::default()
            },
            Weights::Tgv(&tgv),
            "ratio",
        ),
        (
            Options {
                step_product: 1.0,
                ..Options::default()
            },
            Weights::Tv(&tv),
            "product",
        ),
        (
            Options {
                norm_squared: Some(0.0),
                ..Options::default()
            },
            Weights::Tv(&tv),
            "L^2",
        ),
        (
            Options {
                relaxation: Some(2.0),
                ..Options::default()
            },
            Weights::Tv(&tv),
            "relaxation",
        ),
        (
            Options {
                partial_radius: Some(-1.0),
                ..Options::default()
            },
            Weights::Tgv(&tgv),
            "radius",
        ),
        (
            Options {
                partial_tolerance: 1e-3,
                ..Options::default()
            },
            Weights::Tgv(&tgv),
            "needs the partial gap's radius",
        ),
        (
            Options {
                partial_radius: Some(1.0),
                ..Options::default()
            },
            Weights::Tv(&tv),
            "TV has no partial gap",
        ),
        (
            Options {
                free_radius: f64::INFINITY,
                ..Options::default()
            },
            Weights::Tgv(&tgv),
            "free samples",
        ),
    ];
    for (options, weights, fragment) in cases {
        let error = pdhg::plan(&options, weights).expect_err("refused");
        assert!(error.to_string().contains(fragment), "{error}");
    }
}

#[test]
fn the_plan_refuses_weights_that_the_method_cannot_run_with() {
    let tv = Tv::default();
    let refused = |weights: Weights<'_>, fragment: &str| {
        let error = pdhg::plan(&Options::default(), weights).expect_err("refused");
        assert!(error.to_string().contains(fragment), "{error}");
    };
    refused(
        Weights::Tgv(&Tgv {
            alpha1: f64::NAN,
            ..Tgv::default()
        }),
        "first-order",
    );
    refused(
        Weights::Tgv(&Tgv {
            alpha0: 0.0,
            ..Tgv::default()
        }),
        "second-order",
    );
    refused(
        Weights::Tv(&Tv {
            channel_weights: Some(vec![1.0, 0.0]),
            ..Tv::default()
        }),
        "channel weights",
    );
    refused(
        Weights::Tv(&Tv {
            channel_weights: Some(Vec::new()),
            ..Tv::default()
        }),
        "channel weights",
    );
    // Every value out of its range, at once.
    let all = Options {
        tolerance: Some(-1.0),
        step_product: 2.0,
        free_radius: -1.0,
        ..Options::default()
    };
    let error = pdhg::plan(&all, Weights::Tv(&tv)).expect_err("refused").to_string();
    for fragment in ["the tolerance", "product", "free samples"] {
        assert!(error.contains(fragment), "{error}");
    }
}

#[test]
fn the_steps_are_those_of_the_options() {
    // tau and sigma depend on the product and on L^2 through their quotient alone, and a
    // quarter of the product gives the same steps as four times L^2: dividing by 4 is exact.
    // The iterates are then the same to the last bit; with the default steps they are not.
    let (_, frame) = grey(20);
    let base = Options {
        iterations: Some(30),
        tolerance: Some(0.0),
        record_every: 10,
        ..Options::default()
    };
    let solve = |options: &Options| pdhg::solve_tv(&frame, &Tv::default(), options, None, None).expect("a result");
    let quartered = solve(&Options {
        step_product: pdhg::STEP_PRODUCT / 4.0,
        ..base
    });
    let widened = solve(&Options {
        norm_squared: Some(4.0 * pdhg::TV_NORM_SQUARED),
        ..base
    });
    let default = solve(&base);
    assert_eq!(quartered.primal.canvas, widened.primal.canvas);
    assert_eq!(quartered.history.primal, widened.history.primal);
    assert_ne!(quartered.primal.canvas, default.primal.canvas);
}

#[test]
fn the_start_is_the_mmse_decoder() {
    let (problem, frame) = grey(11);
    let point = frames::start(&frame, None).expect("a start");
    assert_eq!(point.coefficients[0], problem.centres());
    assert_eq!(
        point.canvas,
        dct::inverse(problem.centres(), problem.rows(), problem.columns())
    );
}

/// The canvas's coefficients are within their intervals, up to the round trip of the DCT.
fn within_the_set(problem: &Problem, coefficients: &[f64], canvas: &[f64]) -> bool {
    let excess = problem.excess(&dct::forward(canvas, problem.height(), problem.width()));
    (0..coefficients.len() / BLOCK_SIZE).all(|block| {
        let reach = rounding::roundtrip_error(&rounding::block_of(coefficients, block));
        (0..BLOCK_SIZE).all(|frequency| {
            excess[block * BLOCK_SIZE + frequency] * problem.steps()[frequency]
                <= reach[frequency / BLOCK][frequency % BLOCK]
        })
    })
}

/// A bound of the rounding of the values of TV at a point, against the exact values.
///
/// The primal value: the canvas is within the inverse DCT's bound of `D^T c`, and a sample
/// is in at most four differences, so the total variation moves by at most `4 alpha`
/// times their sum; its terms and the data term's are non-negative, each within a few
/// roundings, and summed within `gamma(2N)`. The dual value: every sample of `div p` adds
/// at most four entries, each within `alpha`, with three roundings; the DCT carries that
/// and rounds itself; and `G*` moves by at most the conjugate's bound.
fn tv_values_error(problem: &Problem, weights: &Tv, coefficients: &[f64], canvas: &[f64], p: &Vector) -> f64 {
    let frame = Frame::one(problem.clone());
    let inverse: f64 = (0..coefficients.len() / BLOCK_SIZE)
        .map(|block| {
            rounding::inverse_error(&rounding::block_of(coefficients, block))
                .iter()
                .flatten()
                .sum::<f64>()
        })
        .sum();
    let moved = 4.0 * weights.alpha * inverse;
    let primal_value = frames::tv_objective(&frame, weights, &[coefficients.to_vec()], canvas).expect("the objective");
    let primal = moved + (gamma(2 * canvas.len()) + 6.0 * U) * primal_value;
    let mut divergence = vec![0.0; canvas.len()];
    jpeg_unround::operators::div(&p.x, &p.y, frame.shape(), &mut divergence);
    let rounded = 12.0 * U * weights.alpha;
    let s = dct::forward(&divergence, problem.height(), problem.width());
    let mut error = 0.0;
    let mut reach = 0.0;
    for block in 0..s.len() / BLOCK_SIZE {
        let (row, column) = (block / problem.columns() * BLOCK, block % problem.columns() * BLOCK);
        let magnitude =
            dct::load(&divergence, problem.width(), row, column).map(|line| line.map(|v| v.abs() + rounded));
        let own = rounding::forward_error(&magnitude);
        let carried = rounding::forward_magnitude(&[[rounded; BLOCK]; BLOCK]);
        for frequency in 0..BLOCK_SIZE {
            let index = block * BLOCK_SIZE + frequency;
            let s_error = own[frequency / BLOCK][frequency % BLOCK] + carried[frequency / BLOCK][frequency % BLOCK];
            let ends = problem.lower()[index].abs().max(problem.upper()[index].abs());
            let width = problem.upper()[index] - problem.lower()[index];
            error += ends * s_error;
            reach += (s[index].abs() + s_error) * ends + 0.5 * problem.weights()[frequency] * width * width;
        }
    }
    primal + error + (gamma(s.len()) + 8.0 * U) * reach
}

#[test]
fn tv_converges_within_the_constraint_set() {
    // Measured: a gap per sample of 2.1e-6 after 4000 iterations with the ratio 1, and 9.7e-7
    // with 30.
    for (ratio, threshold) in [(1.0, 2e-5), (30.0, 1e-5)] {
        let (problem, frame) = grey(12);
        let mut seen = Vec::new();
        let mut gaps = Vec::new();
        let mut observe = |record: &Record<'_>| {
            seen.push(record.iteration);
            gaps.push(record.gap);
            assert!(within_the_set(&problem, &record.coefficients[0], record.canvas));
            ControlFlow::Continue(())
        };
        let options = fixed(4000, ratio, None, 200);
        let result = pdhg::solve_tv(&frame, &Tv::default(), &options, None, Some(&mut observe)).expect("a result");
        let history = &result.history;
        assert_eq!(seen, (200..=4000).step_by(200).collect::<Vec<u64>>());
        // The observer is given the gap per sample of each record, as the tolerance is
        // compared with.
        let expected: Vec<f64> = history.gap()[1..].iter().map(|gap| gap / samples(&frame)).collect();
        assert_eq!(gaps, expected);
        assert!(
            problem
                .excess(&result.primal.coefficients[0])
                .iter()
                .all(|&excess| excess == 0.0)
        );
        assert!(history.seconds.windows(2).all(|pair| pair[0] <= pair[1]));
        assert!(last_per_sample(&result, &frame) < threshold);
        // The gap is never below 0 by more than the rounding of the two values, bounded at the
        // last iterate, where it is least.
        let dual = result.dual.as_ref().expect("a dual");
        let allowance = tv_values_error(
            &problem,
            &Tv::default(),
            &result.primal.coefficients[0],
            &result.primal.canvas,
            &dual.p,
        );
        let all = history.gap();
        assert!(all[all.len() - 1] >= -allowance);
        assert!(all[..all.len() - 1].iter().all(|&gap| gap > 0.0));
    }
}

#[test]
fn relaxed_tv_keeps_to_the_constraint_set() {
    // Measured: a gap per sample of 2.7e-6 with the relaxation 1.5 and 2.2e-6 with 1.9 after
    // 1000 iterations, against 9.3e-6 without. The current point of a relaxed step may leave
    // the constraint set; what is observed and returned, the proximal steps' outputs, may not.
    for relaxation in [1.5, 1.9] {
        let (problem, frame) = grey(18);
        let mut observe = |record: &Record<'_>| {
            assert!(within_the_set(&problem, &record.coefficients[0], record.canvas));
            ControlFlow::Continue(())
        };
        let options = fixed(1000, 30.0, Some(relaxation), 100);
        let result = pdhg::solve_tv(&frame, &Tv::default(), &options, None, Some(&mut observe)).expect("a result");
        assert!(last_per_sample(&result, &frame) < 3e-5);
        assert!(
            problem
                .excess(&result.primal.coefficients[0])
                .iter()
                .all(|&excess| excess == 0.0)
        );
        // The dual returned is a projection's output, within the ball up to rounding: dividing
        // by a norm taken within 2U makes a vector longer by at most 3U, and taking its norm
        // here adds 2U more, to first order.
        let p = &result.dual.as_ref().expect("a dual").p;
        assert!(
            p.x.iter()
                .zip(&p.y)
                .all(|(x, y)| (x * x + y * y).sqrt() <= 1.0 + 6.0 * U)
        );
        let gaps = result.history.gap();
        assert!(gaps[..gaps.len() - 1].iter().all(|&gap| gap > 0.0));
    }
}

#[test]
fn tv_stops_at_its_tolerances() {
    // Measured: 230 iterations unrelaxed, 130 with the relaxation 1.9.
    let (_, frame) = grey(13);
    for (relaxation, measured) in [(1.0, 230), (1.9, 130)] {
        let options = Options {
            iterations: Some(5000),
            tolerance: Some(0.05),
            step_ratio: Some(10.0),
            relaxation: Some(relaxation),
            ..Options::default()
        };
        let result = pdhg::solve_tv(&frame, &Tv::default(), &options, None, None).expect("a result");
        assert_eq!(result.stop, Stop::Converged);
        assert_eq!(result.iterations, measured);
        assert!(last_per_sample(&result, &frame) <= 0.05);
    }
    // The first record whose gap is within the relative tolerance of the primal value stops
    // the solver; the gap per sample, whose tolerance is 0, stops nothing. The comparisons
    // are the solver's, of the same doubles: exact. (Measured: 340 iterations.)
    let options = Options {
        iterations: Some(5000),
        tolerance: Some(0.0),
        relative_tolerance: 1e-4,
        step_ratio: Some(10.0),
        ..Options::default()
    };
    let result = pdhg::solve_tv(&frame, &Tv::default(), &options, None, None).expect("a result");
    let (gaps, primal) = (result.history.gap(), &result.history.primal);
    assert!(result.converged());
    let last = gaps.len() - 1;
    assert!(gaps[last] <= 1e-4 * primal[last]);
    assert!((1..last).all(|index| gaps[index] > 1e-4 * primal[index]));
}

#[test]
fn tgv_stops_at_its_partial_tolerance() {
    // As for the relative tolerance, with the partial gap per sample. The radius is more than
    // the largest |w| of the solution (measured: 5.8 after 6000 iterations), and so the
    // partial gap bounds the distance from the least value. (Measured: 390 iterations.)
    let (_, frame) = grey(15);
    let options = Options {
        iterations: Some(5000),
        tolerance: Some(0.0),
        step_ratio: Some(10.0),
        partial_radius: Some(20.0),
        partial_tolerance: 1e-2,
        ..Options::default()
    };
    let result = pdhg::solve_tgv(&frame, &Tgv::default(), &options, None, None).expect("a result");
    let partial: Vec<f64> = result
        .history
        .partial_gap
        .iter()
        .map(|gap| gap / samples(&frame))
        .collect();
    assert!(result.converged());
    assert!(partial[partial.len() - 1] <= 1e-2);
    assert!(partial[1..partial.len() - 1].iter().all(|&gap| gap > 1e-2));
    // Without the radius, no partial gap is taken.
    let short = fixed(20, 10.0, None, 10);
    let plain = pdhg::solve_tv(&frame, &Tv::default(), &short, None, None).expect("a result");
    assert!(plain.history.partial_gap.iter().all(|gap| gap.is_nan()));
    let taken = Options {
        partial_radius: Some(100.0),
        ..short
    };
    let recorded = pdhg::solve_tgv(&frame, &Tgv::default(), &taken, None, None).expect("a result");
    assert_eq!(recorded.history.partial_gap.len(), 3);
    assert!(recorded.history.partial_gap.iter().all(|gap| gap.is_finite()));
}

#[test]
fn tgv_converges_within_the_constraint_set() {
    // Measured: a gap per sample of 4.0e-5 after 6000 iterations (5.2e-5 relaxed by 1.9 on
    // another file), from a gap of 5.5 after 500, and the scaling of the dual 1. The gaps
    // here are at least 0.015, and the values about 3e3, far above their rounding: a gap
    // below 0 could only be an error of the formulas.
    for (relaxation, seed, threshold) in [(None, 14, 4e-4), (Some(1.9), 19, 5e-4)] {
        let (problem, frame) = grey(seed);
        let mut observe = |record: &Record<'_>| {
            assert!(within_the_set(&problem, &record.coefficients[0], record.canvas));
            ControlFlow::Continue(())
        };
        let options = fixed(6000, 10.0, relaxation, 500);
        let result = pdhg::solve_tgv(&frame, &Tgv::default(), &options, None, Some(&mut observe)).expect("a result");
        let gaps = result.history.gap();
        assert!(gaps.iter().all(|&gap| gap > 0.0));
        assert!(last_per_sample(&result, &frame) < threshold);
        assert!(gaps[gaps.len() - 1] < 0.05 * gaps[1]);
        assert!(result.history.scaling[result.history.scaling.len() - 1] > 0.9);
        assert!(
            problem
                .excess(&result.primal.coefficients[0])
                .iter()
                .all(|&excess| excess == 0.0)
        );
        assert!(result.primal.w.is_some());
        assert!(result.dual.as_ref().is_some_and(|dual| dual.r.is_some()));
    }
}

#[test]
fn tgv_starts_from_the_field_given() {
    // No iteration: the one record is the objective at the start, with w as given, or the
    // gradient of the start's canvas.
    let (_, frame) = grey(17);
    let options = Options {
        iterations: Some(0),
        ..Options::default()
    };
    let begin = frames::start(&frame, None).expect("a start");
    let size = frame.samples();
    let zero = Vector::zeros(size);
    let first = Initial {
        w: Some(zero.clone()),
        ..Initial::default()
    };
    // The record adds the same terms as the objective in the order of the planes, whose
    // rounding and the objective's are each within gamma(ceil(W / 8) + 300) of the sum of
    // the terms, all of them at least 0 (tests/records.rs).
    let close = |value: f64, expected: f64| {
        let bound = 2.0 * gamma(frame.width().div_ceil(8) + 300) * expected * 1.01;
        assert!(
            (value - expected).abs() <= bound,
            "{value} against {expected}, the bound {bound}"
        );
    };
    let result = pdhg::solve_tgv(&frame, &Tgv::default(), &options, Some(&first), None).expect("a result");
    let at_zero = frames::tgv_objective(&frame, &Tgv::default(), &begin.coefficients, &begin.canvas, &zero)
        .expect("the objective");
    close(result.history.primal[0], at_zero);
    let default = pdhg::solve_tgv(&frame, &Tgv::default(), &options, None, None).expect("a result");
    let gradient = frames::gradient(&frame, &begin.canvas);
    let at_gradient = frames::tgv_objective(&frame, &Tgv::default(), &begin.coefficients, &begin.canvas, &gradient)
        .expect("the objective");
    close(default.history.primal[0], at_gradient);
    // The two starts differ by far more than the rounding.
    assert!((at_zero - at_gradient).abs() > 1e6 * 2.0 * gamma(frame.width().div_ceil(8) + 300) * at_zero);
    let short = Initial {
        w: Some(Vector::zeros(size - 1)),
        ..Initial::default()
    };
    let error = pdhg::solve_tgv(&frame, &Tgv::default(), &options, Some(&short), None).expect_err("refused");
    assert!(error.to_string().contains("entries"), "{error}");
    let error = pdhg::solve_tv(&frame, &Tv::default(), &options, Some(&first), None).expect_err("refused");
    assert!(error.to_string().contains("TV has no field w"), "{error}");
}

#[test]
fn the_dual_starts_where_given_projected_onto_its_ball() {
    // No iteration: the dual returned is the start, the projection of what was given. One
    // iteration from it goes elsewhere than one from 0.
    let (_, frame) = grey(21);
    let size = frame.samples();
    let mut numbers = Numbers::new(26);
    let p = Vector {
        x: (0..size).map(|_| numbers.normal(0.0, 3.0)).collect(),
        y: (0..size).map(|_| numbers.normal(0.0, 3.0)).collect(),
    };
    let r = Tensor {
        xx: (0..size).map(|_| numbers.normal(0.0, 3.0)).collect(),
        yy: (0..size).map(|_| numbers.normal(0.0, 3.0)).collect(),
        xy: (0..size).map(|_| numbers.normal(0.0, 3.0)).collect(),
    };
    let none = Options {
        iterations: Some(0),
        ..Options::default()
    };
    let weights = Tv {
        alpha: 1.5,
        ..Tv::default()
    };
    let given = Initial {
        p: Some(p.clone()),
        ..Initial::default()
    };
    let tv = pdhg::solve_tv(&frame, &weights, &none, Some(&given), None).expect("a result");
    let mut expected = p.clone();
    frames::project_vectors(&frame, &mut expected, 1.5, true);
    assert_eq!(tv.dual.as_ref().expect("a dual").p, expected);
    let both = Initial {
        p: Some(p.clone()),
        r: Some(r.clone()),
        ..Initial::default()
    };
    let tgv = pdhg::solve_tgv(&frame, &Tgv::default(), &none, Some(&both), None).expect("a result");
    let dual = tgv.dual.as_ref().expect("a dual");
    let mut expected = p;
    frames::project_vectors(&frame, &mut expected, 1.0, true);
    assert_eq!(dual.p, expected);
    let mut expected = r;
    frames::project_tensors(&frame, &mut expected, 2.0, true);
    assert_eq!(dual.r.as_ref(), Some(&expected));
    let one = Options {
        iterations: Some(1),
        ..Options::default()
    };
    let from_given = pdhg::solve_tv(&frame, &weights, &one, Some(&given), None).expect("a result");
    let from_zero = pdhg::solve_tv(&frame, &weights, &one, None, None).expect("a result");
    assert_ne!(from_given.primal.canvas, from_zero.primal.canvas);
}

#[test]
fn a_start_that_the_method_cannot_take_is_refused() {
    let (_, frame) = grey(21);
    let size = frame.samples();
    let options = Options {
        iterations: Some(0),
        ..Options::default()
    };
    let cases = [
        (
            Initial {
                r: Some(Tensor::zeros(size)),
                ..Initial::default()
            },
            true,
            "TV has no field w or r",
        ),
        (
            Initial {
                r: Some(Tensor::zeros(size + 1)),
                ..Initial::default()
            },
            false,
            "entries",
        ),
        (
            Initial {
                p: Some(Vector {
                    x: vec![f64::NAN; size],
                    y: vec![0.0; size],
                }),
                ..Initial::default()
            },
            true,
            "finite",
        ),
        (
            Initial {
                w: Some(Vector {
                    x: vec![f64::INFINITY; size],
                    y: vec![0.0; size],
                }),
                ..Initial::default()
            },
            false,
            "finite",
        ),
        (
            Initial {
                coefficients: Some(vec![vec![0.0; 64]]),
                ..Initial::default()
            },
            true,
            "coefficients",
        ),
    ];
    for (first, tv, fragment) in cases {
        let error = if tv {
            pdhg::solve_tv(&frame, &Tv::default(), &options, Some(&first), None).expect_err("refused")
        } else {
            pdhg::solve_tgv(&frame, &Tgv::default(), &options, Some(&first), None).expect_err("refused")
        };
        assert!(error.to_string().contains(fragment), "{error}");
    }
}

#[test]
fn tv_reaches_below_the_subgradient_method() {
    // Both minimize the same objective over the same set, so the primal-dual method's dual
    // value is a lower bound of what the subgradient method can reach (by 3.3 here, far above
    // rounding). And in as many iterations its primal value goes below the subgradient
    // method's (measured: 3097.97 against 3099.88).
    let (_, frame) = grey(16);
    let primal_dual =
        pdhg::solve_tv(&frame, &Tv::default(), &fixed(300, 10.0, None, 300), None, None).expect("a result");
    let options = subgradient::Options {
        iterations: 300,
        record_every: 300,
        ..subgradient::Options::default()
    };
    let stepped = subgradient::solve_tv(&frame, &Tv::default(), &options, None, None).expect("a result");
    assert!(primal_dual.history.dual[1] < stepped.history.primal[1]);
    assert!(primal_dual.history.primal[1] < stepped.history.primal[1]);
    assert_eq!(stepped.history.dual[1], f64::NEG_INFINITY);
    assert!(stepped.dual.is_none());
}

const INVARIANT: f64 = 1e-4; // in steps: every output's coefficients are within their intervals to this

/// The coefficients are within their intervals exactly, and the canvas's own within the
/// invariant: those are within rounding of the coefficients (the tests of the frames bound
/// it where the step's input is known), some units of 2^-53 of the samples, nine orders of
/// magnitude below the invariant.
fn within_the_frame(frame: &Frame, coefficients: &[Vec<f64>], canvas: &[f64]) {
    for (channel, own) in frame.channels().iter().zip(coefficients) {
        assert!(channel.problem().excess(own).iter().all(|&excess| excess == 0.0));
    }
    for excess in frames::excess(frame, canvas) {
        assert!(excess.iter().all(|&value| value <= INVARIANT));
    }
}

#[test]
fn tv_of_a_colour_frame_converges_within_the_set() {
    // Measured after 4000 iterations: a gap per sample of 1.0e-5 coupled and 6.9e-4 apart with
    // the chroma of 4:2:0, a partial gap of the radius 255 (docs/math.md, 6.6); 3.3e-7 with
    // nothing free, where it is the gap itself. The gaps here are at least 7e-4, and the
    // values about 1e4, far above their rounding; the canvases keep within 84 to 197, in the
    // box of the radius.
    for (ratio, coupled, threshold) in [((2, 2), true, 1e-4), ((2, 2), false, 7e-3), ((1, 1), true, 4e-6)] {
        let (frame, _) = synthetic::colour_frame(20, 30, 70, ratio, &DataTerm::default());
        let mut observe = |record: &Record<'_>| {
            within_the_frame(&frame, record.coefficients, record.canvas);
            ControlFlow::Continue(())
        };
        let weights = Tv {
            coupled,
            ..Tv::default()
        };
        let options = fixed(4000, 10.0, Some(1.0), 500);
        let result = pdhg::solve_tv(&frame, &weights, &options, None, Some(&mut observe)).expect("a result");
        let gaps = result.history.gap();
        assert!(gaps.iter().all(|&gap| gap > 0.0));
        assert!(last_per_sample(&result, &frame) < threshold);
        assert!(gaps[gaps.len() - 1] < 1e-3 * gaps[0]);
        let p = &result.dual.as_ref().expect("a dual").p;
        let plane = frame.plane();
        for pixel in 0..plane {
            if coupled {
                let square: f64 = (0..3)
                    .map(|channel| {
                        let index = channel * plane + pixel;
                        p.x[index] * p.x[index] + p.y[index] * p.y[index]
                    })
                    .sum();
                assert!(square.sqrt() <= 1.0 + 12.0 * U);
            }
        }
    }
}

#[test]
fn tgv_of_a_colour_frame_converges_within_the_set() {
    // Measured after 6000 iterations, with the chroma of 4:2:0: a gap per sample of 1.0e-5
    // coupled and 4.6e-4 apart, and the scaling of the dual 1.
    for (coupled, threshold) in [(true, 1e-4), (false, 5e-3)] {
        let (frame, _) = synthetic::colour_frame(20, 30, 70, (2, 2), &DataTerm::default());
        let mut observe = |record: &Record<'_>| {
            within_the_frame(&frame, record.coefficients, record.canvas);
            ControlFlow::Continue(())
        };
        let weights = Tgv {
            coupled,
            ..Tgv::default()
        };
        let options = fixed(6000, 10.0, Some(1.0), 500);
        let result = pdhg::solve_tgv(&frame, &weights, &options, None, Some(&mut observe)).expect("a result");
        assert!(result.history.gap().iter().all(|&gap| gap > 0.0));
        assert!(last_per_sample(&result, &frame) < threshold);
        assert!(result.history.scaling[result.history.scaling.len() - 1] > 0.9);
        assert_eq!(result.primal.w.as_ref().map(|w| w.x.len()), Some(frame.samples()));
    }
}

#[test]
fn an_observer_stops_the_solver() {
    let (_, frame) = grey(22);
    let mut records = 0;
    let mut observe = |_: &Record<'_>| {
        records += 1;
        if records == 3 {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    };
    let options = fixed(1000, 10.0, None, 10);
    let result = pdhg::solve_tv(&frame, &Tv::default(), &options, None, Some(&mut observe)).expect("a result");
    assert_eq!((result.iterations, result.stop), (30, Stop::Observer));
}
