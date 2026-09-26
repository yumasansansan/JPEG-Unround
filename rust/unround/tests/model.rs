// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! The model of a component: the projection onto the constraint set, the proximal
//! map, the conjugate, and weak duality (docs/math.md, 1.2, 4.1 and 6).
//!
//! What is computed in floating point is checked within bounds of its rounding,
//! derived here for the operations as the crate orders them; the comparisons that
//! decide are made in exact arithmetic where the bound is of the order of the
//! rounding itself.

#![allow(
    clippy::float_cmp,
    reason = "the tests compare doubles to the last bit where the arithmetic is exact or the same"
)]

mod support;

use jpeg_unround::dct::{self, BLOCK, BLOCK_SIZE};
use jpeg_unround::frames::{self, Frame, Tensor, Vector};
use jpeg_unround::model::{Centres, DataTerm, Problem, Tgv, Tv};
use support::exact::Dyadic;
use support::rounding::{self, U, gamma};
use support::synthetic::{self, Numbers};

fn inside(problem: &Problem, numbers: &mut Numbers) -> Vec<f64> {
    let mut values: Vec<f64> = problem
        .lower()
        .iter()
        .zip(problem.upper())
        .map(|(&lower, &upper)| numbers.uniform(lower, upper))
        .collect();
    problem.clip(&mut values);
    values
}

fn noisy(canvas: &[f64], deviation: f64, numbers: &mut Numbers) -> Vec<f64> {
    canvas
        .iter()
        .map(|&sample| sample + numbers.normal(0.0, deviation))
        .collect()
}

fn euclidean(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn forward(problem: &Problem, canvas: &[f64]) -> Vec<f64> {
    dct::forward(canvas, problem.height(), problem.width())
}

fn inverse(problem: &Problem, coefficients: &[f64]) -> Vec<f64> {
    dct::inverse(coefficients, problem.rows(), problem.columns())
}

/// A bound, sample by sample, of the rounding of the inverse DCT of coefficients.
fn inverse_errors(problem: &Problem, coefficients: &[f64]) -> Vec<f64> {
    let mut bound = vec![0.0; coefficients.len()];
    for block in 0..coefficients.len() / BLOCK_SIZE {
        let (row, column) = (block / problem.columns() * BLOCK, block % problem.columns() * BLOCK);
        let error = rounding::inverse_error(&rounding::block_of(coefficients, block));
        dct::store(&error, &mut bound, problem.width(), row, column);
    }
    bound
}

/// A bound, sample by sample, of `|project(v) - P(v)|`: the forward DCT rounds within
/// its bound, which clipping does not increase; the exact inverse carries that
/// through `|B^T| e |B|`, and the inverse rounds within its own bound.
fn projection_errors(problem: &Problem, canvas: &[f64]) -> Vec<f64> {
    let mut clipped = forward(problem, canvas);
    problem.clip(&mut clipped);
    let mut bound = inverse_errors(problem, &clipped);
    for block in 0..clipped.len() / BLOCK_SIZE {
        let (row, column) = (block / problem.columns() * BLOCK, block % problem.columns() * BLOCK);
        let error = rounding::forward_error(&dct::load(canvas, problem.width(), row, column));
        let carried = rounding::inverse_magnitude(&error);
        for (i, line) in carried.iter().enumerate() {
            for (j, &value) in line.iter().enumerate() {
                bound[(row + i) * problem.width() + column + j] += value;
            }
        }
    }
    bound
}

fn project(problem: &Problem, canvas: &[f64]) -> Vec<f64> {
    frames::project(&Frame::one(problem.clone()), canvas)
}

#[test]
fn the_projection_lands_in_the_set_is_idempotent_and_is_nearest() {
    let (problem, canvas) = synthetic::problem(16, 24, 2, 2.0, &DataTerm::default());
    let mut numbers = Numbers::new(20);
    let v = noisy(&canvas, 30.0, &mut numbers);
    let projected = project(&problem, &v);
    let mut clipped = forward(&problem, &v);
    problem.clip(&mut clipped);
    // Its coefficients are within their intervals up to the round trip of the DCT.
    let excess = problem.excess(&forward(&problem, &projected));
    for block in 0..clipped.len() / BLOCK_SIZE {
        let bound = rounding::roundtrip_error(&rounding::block_of(&clipped, block));
        for frequency in 0..BLOCK_SIZE {
            let index = block * BLOCK_SIZE + frequency;
            assert!(excess[index] * problem.steps()[frequency] <= bound[frequency / BLOCK][frequency % BLOCK]);
        }
    }
    // P is 1-Lipschitz: projecting again moves it at most twice its own error, plus the new one.
    let first = euclidean(&projection_errors(&problem, &v));
    let again = project(&problem, &projected);
    let moved: Vec<f64> = again.iter().zip(&projected).map(|(a, b)| a - b).collect();
    assert!(euclidean(&moved) <= 2.0 * first + euclidean(&projection_errors(&problem, &projected)));
    // And it is nearest: |P(v) - v| <= |z - v| for every z in the set, up to the errors of
    // the two points and of the two computed distances.
    let difference: Vec<f64> = projected.iter().zip(&v).map(|(a, b)| a - b).collect();
    let near = euclidean(&difference);
    for _ in 0..50 {
        let c = inside(&problem, &mut numbers);
        let other = inverse(&problem, &c);
        let far = euclidean(&other.iter().zip(&v).map(|(a, b)| a - b).collect::<Vec<f64>>());
        let allowance = first + euclidean(&inverse_errors(&problem, &c)) + gamma(v.len() + 1) * (near + far);
        assert!(near <= far + allowance);
    }
}

/// `1/2 |c - e|^2 + tau G(c)`, for c within the intervals.
fn prox_objective(problem: &Problem, c: &[f64], e: &[f64], tau: f64) -> f64 {
    let squares: f64 = c.iter().zip(e).map(|(a, b)| (a - b) * (a - b)).sum();
    0.5 * squares + tau * problem.data_term(c)
}

/// The proximal map of the model, as a block at a time.
fn prox(problem: &Problem, e: &[f64], tau: f64) -> Vec<f64> {
    let mut c = e.to_vec();
    problem.prox(tau, &mut c);
    c
}

/// A bound, coefficient by coefficient, of the rounding of the proximal map before it
/// is clipped, which does not increase it. The crate computes `tau w` (one rounding),
/// `1 / (1 + tau w)` (two), the numerator `e + (tau w) centre` (one or two) and their
/// product (one): relative to the exact `(e + tau w centre) / (1 + tau w)`, the
/// numerator is within `3U (|e| + tau w |centre|)` and the rest within `4U` of the
/// quotient, whose magnitude is at most `(|e| + tau w |centre|) / (1 + tau w)`: 7U of
/// that in all, and 8U with the terms of second order.
fn prox_error(problem: &Problem, e: &[f64], tau: f64) -> Vec<f64> {
    e.iter()
        .enumerate()
        .map(|(index, &value)| {
            let w = problem.weights()[index % BLOCK_SIZE];
            8.0 * U * (value.abs() + tau * w * problem.centres()[index].abs()) / (1.0 + tau * w)
        })
        .collect()
}

#[test]
fn the_prox_minimizes_its_objective() {
    for dc_weight in [0.0, 1.0] {
        for tau in [0.01, 1.0, 1e4] {
            let data = DataTerm {
                mu: Some(5.0),
                dc_weight,
                ..DataTerm::default()
            };
            let (problem, canvas) = synthetic::problem(16, 24, 3, 2.0, &data);
            let mut numbers = Numbers::new(21);
            let e = forward(&problem, &noisy(&canvas, 20.0, &mut numbers));
            let c = prox(&problem, &e, tau);
            assert!(problem.excess(&c).iter().all(|&excess| excess == 0.0));
            let delta = prox_error(&problem, &e, tau);
            // J(c) <= J(c*) + sum of |dJ(c*)| delta + (1 + tau w) delta^2 / 2, and each J is
            // evaluated within (gamma(2n) + 4U) of its value, its parts being non-negative.
            let mut suboptimal = 0.0;
            for (index, (&value, &shift)) in c.iter().zip(&delta).enumerate() {
                let w = problem.weights()[index % BLOCK_SIZE];
                let slope = (value - e[index]).abs()
                    + tau * w * (value - problem.centres()[index]).abs()
                    + (1.0 + tau * w) * shift;
                suboptimal += slope * shift + f64::midpoint(1.0, tau * w) * shift * shift;
            }
            let evaluation = gamma(2 * c.len()) + 4.0 * U;
            let best = prox_objective(&problem, &c, &e, tau);
            for _ in 0..50 {
                let other = inside(&problem, &mut numbers);
                let value = prox_objective(&problem, &other, &e, tau);
                assert!(best <= value + suboptimal + evaluation * (best + value));
            }
            // The optimality condition, coefficient by coefficient: where c* is inside its
            // interval, (e - c*) / tau is the gradient of the data term there. The residual
            // at c is within delta (1/tau + w), plus the rounding of the residual itself.
            for (index, (&value, &shift)) in c.iter().zip(&delta).enumerate() {
                let interior = value - problem.lower()[index] > shift && problem.upper()[index] - value > shift;
                if !interior {
                    continue;
                }
                let w = problem.weights()[index % BLOCK_SIZE];
                let residual = (e[index] - value) / tau - w * (value - problem.centres()[index]);
                let allowance = shift * (1.0 / tau + w)
                    + 4.0 * U * ((e[index] - value).abs() / tau + w * (value - problem.centres()[index]).abs());
                assert!(residual.abs() <= allowance, "{tau} {index}");
            }
        }
    }
}

#[test]
fn the_prox_with_a_costly_slack_meets_its_optimality_conditions() {
    // With a slack of half a step that costs 0.3 per step, the minimizer of
    // 1/2 (c - e)^2 + tau g(c) over the widened interval has, coefficient by coefficient,
    // r = (e - c) - tau m (c - centre) in tau cost times the subdifferential of the
    // distance to the file's own interval, plus its normal cone: 0 within the interval,
    // tau cost beyond it, [0, tau cost] at its upper end, and at least tau cost at the
    // widened one; and alike below. c is within delta of the exact minimizer: the
    // quotient's rounding, and three more of the shift by tau cost / (1 + tau m); r is
    // computed within 4U of its parts.
    for tau in [0.01, 1.0, 1e4] {
        let data = DataTerm {
            mu: Some(5.0),
            slack: 0.5,
            slack_cost: 0.3,
            ..DataTerm::default()
        };
        let (problem, canvas) = synthetic::problem(16, 24, 3, 2.0, &data);
        let cost = problem.cost().expect("a cost");
        let mut numbers = Numbers::new(27);
        let e = forward(&problem, &noisy(&canvas, 40.0, &mut numbers));
        let c = prox(&problem, &e, tau);
        assert!(problem.excess(&c).iter().all(|&excess| excess == 0.0));
        let quotient = prox_error(&problem, &e, tau);
        for (index, &value) in c.iter().enumerate() {
            let frequency = index % BLOCK_SIZE;
            let m = problem.weights()[frequency];
            let force = tau * cost.costs()[frequency];
            let moved = (e[index] + tau * m * problem.centres()[index]) / (1.0 + tau * m);
            let delta = quotient[index] + 4.0 * U * (moved.abs() + force / (1.0 + tau * m));
            let r = (e[index] - value) - tau * m * (value - problem.centres()[index]);
            let allowance = (1.0 + tau * m) * delta
                + 4.0 * U * ((e[index] - value).abs() + tau * m * (value - problem.centres()[index]).abs());
            let (low, high) = (cost.inner_lower()[index], cost.inner_upper()[index]);
            let (lower, upper) = (problem.lower()[index], problem.upper()[index]);
            if value > low && value < high {
                assert!(r.abs() <= allowance);
            } else if value > high && value < upper {
                assert!((r - force).abs() <= allowance);
            } else if value > lower && value < low {
                assert!((r + force).abs() <= allowance);
            } else if value == high {
                assert!(r >= -allowance && r <= force + allowance);
            } else if value == low {
                assert!(r <= allowance && r >= -force - allowance);
            } else if value == upper {
                assert!(r >= force - allowance);
            } else {
                assert!(value == lower && r <= -force + allowance);
            }
        }
    }
}

/// A bound of the sum of the magnitudes of the terms of `G*(s)` and of their parts: a
/// term is `s c* - (m/2) (c* - centre)^2` with `c*` in the interval, or
/// `max(s a, s b)`.
fn conjugate_magnitude(problem: &Problem, s: &[f64]) -> f64 {
    s.iter()
        .enumerate()
        .map(|(index, &value)| {
            let ends = problem.lower()[index].abs().max(problem.upper()[index].abs());
            let width = problem.upper()[index] - problem.lower()[index];
            value.abs() * ends + 0.5 * problem.weights()[index % BLOCK_SIZE] * width * width
        })
        .sum()
}

/// A bound of the rounding of `G*(s)`: each term within eight roundings of its parts,
/// and the sum within `gamma(n)` of their magnitudes.
fn conjugate_error(problem: &Problem, s: &[f64]) -> f64 {
    (gamma(s.len()) + 8.0 * U) * conjugate_magnitude(problem, s)
}

#[test]
fn the_conjugate_is_the_largest_of_its_objective() {
    for (mu, dc_weight) in [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0)] {
        let data = DataTerm {
            mu: Some(mu),
            dc_weight,
            ..DataTerm::default()
        };
        let (problem, _) = synthetic::problem(8, 8, 4, 2.0, &data);
        let mut numbers = Numbers::new(22);
        for _ in 0..5 {
            let s: Vec<f64> = (0..BLOCK_SIZE).map(|_| numbers.normal(0.0, 3.0)).collect();
            let value = problem.conjugate(&s);
            let own = conjugate_error(&problem, &s);
            // Fenchel-Young: no point of the set gives more than the conjugate.
            for _ in 0..200 {
                let c = inside(&problem, &mut numbers);
                let term = problem.data_term(&c);
                let pairs: f64 = s.iter().zip(&c).map(|(a, b)| a * b).sum();
                let magnitude: f64 = s.iter().zip(&c).map(|(a, b)| (a * b).abs()).sum();
                let left = pairs - term;
                assert!(left <= value + own + (gamma(2 * c.len()) + 4.0 * U) * (magnitude + term));
            }
            // And a grid over each interval comes within reach of it: its points are within
            // 4U of the ends' magnitudes; the largest on the grid is within (m/2) spacing^2
            // of the largest of the interval where that is inside it, and within the slope
            // times 4U of the ends where it is at an end.
            let mut best = 0.0;
            let mut reach = 0.0;
            for (index, &slope_at) in s.iter().enumerate() {
                let (lower, upper) = (problem.lower()[index], problem.upper()[index]);
                let (w, centre) = (problem.weights()[index], problem.centres()[index]);
                let mut largest = f64::NEG_INFINITY;
                for step in 0..=20_000u32 {
                    let c = lower + f64::from(step) / 20_000.0 * (upper - lower);
                    largest = largest.max(slope_at * c - 0.5 * w * (c - centre) * (c - centre));
                }
                best += largest;
                let spacing = (upper - lower) / 20_000.0;
                let slope = slope_at.abs() + w * (upper - lower);
                reach += 0.5 * w * spacing * spacing + slope * 4.0 * U * lower.abs().max(upper.abs());
            }
            let evaluation = (gamma(BLOCK_SIZE) + 6.0 * U) * conjugate_magnitude(&problem, &s);
            assert!((best - value).abs() <= reach + evaluation + own);
        }
        // G*(0) is minus the least of G, which is 0 at the centres, and exactly so in
        // floating point: the centres lie within their intervals, and every term is a
        // product with 0.
        assert_eq!(problem.conjugate(&[0.0; BLOCK_SIZE]), 0.0);
    }
}

#[test]
fn the_centres_lie_within_their_intervals_exactly() {
    for centres in [Centres::Mmse, Centres::Midpoint] {
        let data = DataTerm {
            centres,
            ..DataTerm::default()
        };
        let (problem, _) = synthetic::problem(16, 24, 1, 2.0, &data);
        for (index, &centre) in problem.centres().iter().enumerate() {
            assert!(problem.lower()[index] <= centre && centre <= problem.upper()[index]);
        }
    }
}

#[test]
fn the_tv_dual_value_is_never_above_the_primal_value() {
    // No allowance is made for rounding: the gaps of these random points exceed it by many
    // orders of magnitude, and so a failure can only be an error of the formulas.
    let (problem, _) = synthetic::problem(16, 24, 5, 2.0, &DataTerm::default());
    let frame = Frame::one(problem.clone());
    let weights = Tv {
        alpha: 1.5,
        ..Tv::default()
    };
    let mut numbers = Numbers::new(23);
    for _ in 0..20 {
        let c = inside(&problem, &mut numbers);
        let canvas = inverse(&problem, &c);
        let size = canvas.len();
        let mut p = Vector {
            x: (0..size).map(|_| numbers.normal(0.0, 2.0)).collect(),
            y: (0..size).map(|_| numbers.normal(0.0, 2.0)).collect(),
        };
        frames::project_vectors(&frame, &mut p, weights.alpha, true);
        let (primal, dual) =
            frames::tv_values(&frame, &weights, &[c], &canvas, &p, frames::FREE_RADIUS).expect("the values");
        assert!(dual < primal);
    }
}

#[test]
fn the_tgv_dual_is_made_feasible_and_is_never_above_the_primal_value() {
    // As for TV, the gaps here are far above any rounding.
    let (problem, _) = synthetic::problem(16, 24, 6, 2.0, &DataTerm::default());
    let frame = Frame::one(problem.clone());
    let weights = Tgv::default();
    let mut numbers = Numbers::new(24);
    for _ in 0..20 {
        let c = inside(&problem, &mut numbers);
        let canvas = inverse(&problem, &c);
        let size = canvas.len();
        let mut w = frames::gradient(&frame, &canvas);
        for value in w.x.iter_mut().chain(w.y.iter_mut()) {
            *value += numbers.normal(0.0, 1.0);
        }
        let mut r = Tensor {
            xx: (0..size).map(|_| numbers.normal(0.0, 3.0)).collect(),
            yy: (0..size).map(|_| numbers.normal(0.0, 3.0)).collect(),
            xy: (0..size).map(|_| numbers.normal(0.0, 3.0)).collect(),
        };
        frames::project_tensors(&frame, &mut r, weights.alpha0, true);
        let (primal, dual, theta) =
            frames::tgv_values(&frame, &weights, &[c], &canvas, &w, &r, frames::FREE_RADIUS).expect("the values");
        assert!(0.0 < theta && theta <= 1.0);
        // The scaled divergence is within alpha1: its norms, computed within 4U of their size.
        let mut scaled = r.clone();
        for value in scaled
            .xx
            .iter_mut()
            .chain(scaled.yy.iter_mut())
            .chain(scaled.xy.iter_mut())
        {
            *value *= theta;
        }
        let divergence = frames::tensor_divergence(&frame, &scaled);
        let largest = divergence
            .x
            .iter()
            .zip(&divergence.y)
            .map(|(x, y)| x.hypot(*y))
            .fold(0.0, f64::max);
        assert!(largest <= weights.alpha1 * (1.0 + 4.0 * U));
        assert!(dual < primal);
    }
}

#[test]
fn a_projection_of_a_field_lands_in_its_ball_and_keeps_what_is_inside() {
    // The norm of a projected vector: its computed norm within 2U, the quotient by the
    // radius within 3U, the division within U, and the test's own norm within 2U: 8U.
    // Tensors have a third term: 10U.
    let (problem, _) = synthetic::problem(8, 16, 1, 2.0, &DataTerm::default());
    let frame = Frame::one(problem);
    let size = frame.samples();
    let mut numbers = Numbers::new(8);
    let p = Vector {
        x: (0..size).map(|_| numbers.normal(0.0, 2.0)).collect(),
        y: (0..size).map(|_| numbers.normal(0.0, 2.0)).collect(),
    };
    let mut projected = p.clone();
    frames::project_vectors(&frame, &mut projected, 1.5, true);
    for index in 0..size {
        let norm = projected.x[index].hypot(projected.y[index]);
        assert!(norm <= 1.5 * (1.0 + 8.0 * U));
        if p.x[index].hypot(p.y[index]) <= 1.5 {
            assert!(projected.x[index] == p.x[index] && projected.y[index] == p.y[index]);
        }
    }
    let r = Tensor {
        xx: (0..size).map(|_| numbers.normal(0.0, 2.0)).collect(),
        yy: (0..size).map(|_| numbers.normal(0.0, 2.0)).collect(),
        xy: (0..size).map(|_| numbers.normal(0.0, 2.0)).collect(),
    };
    let mut projected = r.clone();
    frames::project_tensors(&frame, &mut projected, 1.5, true);
    for index in 0..size {
        let (a, b, c) = (projected.xx[index], projected.yy[index], projected.xy[index]);
        let norm = (a * a + b * b + 2.0 * c * c).sqrt();
        assert!(norm <= 1.5 * (1.0 + 10.0 * U));
    }
    // Projecting again leaves it within the rounding of its norm: a second projection
    // divides by at most 1 + 10U.
    let mut twice = projected.clone();
    frames::project_tensors(&frame, &mut twice, 1.5, true);
    let bound = Dyadic::from_i128(11).mul(&Dyadic::from_f64(U));
    for (&a, &b) in twice.xx.iter().zip(&projected.xx) {
        let moved = Dyadic::from_f64(a).sub(&Dyadic::from_f64(b)).abs();
        assert!(moved.at_most(&bound.mul(&Dyadic::from_f64(b).abs())));
    }
}
