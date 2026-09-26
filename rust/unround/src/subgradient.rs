// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! A subgradient method of jpeg2png's kind, for the TV model of one component
//! (docs/math.md, 7).
//!
//! Normalized subgradient steps whose length falls as a power of the iteration (by
//! default one over its square root), FISTA's extrapolation, and a projection onto
//! the quantization constraint set after every step. It is here to be compared
//! with: it has no convergence guarantee, and no gap to tell how far it is from the
//! least value.

use std::ops::ControlFlow;

use crate::dct::{self, BLOCK_SIZE};
use crate::error::Error;
use crate::exact::{self, nearest_from_u64, nearest_from_usize};
use crate::frames::{self, Frame, Primal};
use crate::model::Tv;
use crate::operators::{self, vector_square};
use crate::pdhg::{Observer, Record};
use crate::results::{FrameResult, Recorder, Stop};

/// How many iterations the method takes, its steps, and how often it records the
/// objective.
///
/// The step after `n` iterations moves the canvas by `step sqrt(N) / (1 + n)^decay`
/// along the normalized subgradient, `N` the number of samples, and `momentum` turns
/// FISTA's extrapolation on (docs/math.md, 7). Every `record_every` iterations (0:
/// none but the last), and after the last, the method records the objective.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Options {
    /// The iterations.
    pub iterations: u64,
    /// How many iterations lie between records.
    pub record_every: u64,
    /// `eta`, the first step in grey levels per sample.
    pub step: f64,
    /// `beta`, the power at which the step falls.
    pub decay: f64,
    /// Whether FISTA's extrapolation is on.
    pub momentum: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            iterations: 50,
            record_every: 1,
            step: 0.5,
            decay: 0.5,
            momentum: true,
        }
    }
}

/// A subgradient of `alpha ||grad x||_{2,1}` plus the data term, at any canvas of the
/// frame's one component. Where the gradient of `x` is 0, the subgradient of its norm
/// taken is 0.
#[must_use]
pub fn subgradient(frame: &Frame, weights: &Tv, canvas: &[f64]) -> Vec<f64> {
    let shape = frame.shape();
    let size = shape.size();
    let problem = frame.channels()[0].problem();
    let (mut across, mut down) = (vec![0.0; size], vec![0.0; size]);
    operators::grad(canvas, shape, &mut across, &mut down);
    for (x, y) in across.iter_mut().zip(&mut down) {
        let norm = vector_square(*x, *y).sqrt();
        if norm > 0.0 {
            *x /= norm;
            *y /= norm;
        } else {
            (*x, *y) = (0.0, 0.0);
        }
    }
    let mut divergence = vec![0.0; size];
    operators::div(&across, &down, shape, &mut divergence);
    let mut coefficients = dct::forward(canvas, shape.height, shape.width);
    for (index, (value, &centre)) in coefficients.iter_mut().zip(problem.centres()).enumerate() {
        *value = problem.weights()[index % BLOCK_SIZE] * (*value - centre);
    }
    let data = dct::inverse(&coefficients, problem.rows(), problem.columns());
    divergence
        .iter()
        .zip(&data)
        .map(|(&direction, &term)| -weights.alpha * direction + term)
        .collect()
}

/// `base^exponent`, and for 1/2 the square root, which is correctly rounded.
fn power(base: f64, exponent: f64) -> f64 {
    #[expect(clippy::float_cmp, reason = "the power 1/2 is taken exactly as it is given")]
    let root = exponent == 0.5;
    if root { base.sqrt() } else { base.powf(exponent) }
}

fn check(options: &Options) -> Result<(), Error> {
    let ranges = [
        (
            options.step > 0.0 && options.step.is_finite(),
            format!("the step is positive and finite, not {}", options.step),
        ),
        (
            (0.0..f64::INFINITY).contains(&options.decay),
            format!("the decay is at least 0 and finite, not {}", options.decay),
        ),
    ];
    let wrong: Vec<String> = ranges
        .into_iter()
        .filter(|(holds, _)| !holds)
        .map(|(_, message)| message)
        .collect();
    if wrong.is_empty() {
        Ok(())
    } else {
        Err(Error::Options(wrong.join("; ")))
    }
}

/// Minimizes the TV model's objective within the quantization constraint set, for a
/// frame of one component.
///
/// `start` are the coefficients to start from, the data term's centres unless
/// given. The history's dual values are `-inf`: the method has none. `observer` is
/// called with each record after the first, with an infinite gap; it can stop the
/// method.
///
/// # Errors
///
/// [`Error::Options`] for options out of their ranges, a frame of several channels,
/// or a start that the method cannot take.
pub fn solve_tv(
    frame: &Frame,
    weights: &Tv,
    options: &Options,
    start: Option<&[Vec<f64>]>,
    mut observer: Option<&mut Observer<'_>>,
) -> Result<FrameResult, Error> {
    check(options)?;
    if frame.channels().len() != 1 || frame.free(&frame.channels()[0]) {
        return Err(Error::Options(
            "the subgradient method is for one component whose canvas is its blocks".into(),
        ));
    }
    let problem = frame.channels()[0].problem();
    let shape = frame.shape();
    let first = frames::start(frame, start)?;
    let (mut coefficients, mut canvas) = (first.coefficients, first.canvas);
    let mut extrapolated = canvas.clone();
    let mut momentum: f64 = 1.0;
    let radius = options.step * nearest_from_usize(frame.samples()).sqrt();
    let mut recorder = Recorder::new();
    let objective = frames::tv_objective(frame, weights, &coefficients, &canvas)?;
    recorder.record(0, objective, f64::NEG_INFINITY, f64::NAN, f64::NAN);
    let mut stop = Stop::Iterations;
    let mut iteration: u64 = 0;
    while iteration < options.iterations {
        recorder.resume();
        let direction = subgradient(frame, weights, &extrapolated);
        let length = exact::sum(direction.iter().map(|&along| along * along)).sqrt();
        if length == 0.0 {
            recorder.pause();
            stop = Stop::Stationary;
            break;
        }
        let distance = radius / power(1.0 + nearest_from_u64(iteration), options.decay);
        iteration += 1;
        let factor = distance / length;
        let moved: Vec<f64> = extrapolated
            .iter()
            .zip(&direction)
            .map(|(&value, &along)| value - factor * along)
            .collect();
        let mut following = dct::forward(&moved, shape.height, shape.width);
        problem.clip(&mut following);
        let following_canvas = dct::inverse(&following, problem.rows(), problem.columns());
        let following_momentum = f64::midpoint(1.0, (1.0 + 4.0 * momentum * momentum).sqrt());
        if options.momentum {
            let pull = (momentum - 1.0) / following_momentum;
            for ((value, &new), &old) in extrapolated.iter_mut().zip(&following_canvas).zip(&canvas) {
                *value = new + pull * (new - old);
            }
        } else {
            extrapolated.copy_from_slice(&following_canvas);
        }
        coefficients = vec![following];
        canvas = following_canvas;
        momentum = following_momentum;
        recorder.pause();
        if iteration == options.iterations
            || (options.record_every > 0 && iteration.is_multiple_of(options.record_every))
        {
            let objective = frames::tv_objective(frame, weights, &coefficients, &canvas)?;
            recorder.record(iteration, objective, f64::NEG_INFINITY, f64::NAN, f64::NAN);
            if let Some(observe) = observer.as_deref_mut() {
                let record = Record {
                    iteration,
                    gap: f64::INFINITY,
                    primal: objective,
                    dual: f64::NEG_INFINITY,
                    coefficients: &coefficients,
                    canvas: &canvas,
                    w: None,
                };
                if observe(&record) == ControlFlow::Break(()) {
                    stop = Stop::Observer;
                    break;
                }
            }
        }
    }
    Ok(FrameResult {
        primal: Primal {
            coefficients,
            canvas,
            w: None,
        },
        dual: None,
        iterations: iteration,
        stop,
        history: recorder.finish(),
    })
}
