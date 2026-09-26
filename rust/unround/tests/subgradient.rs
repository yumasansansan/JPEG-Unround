// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! The subgradient method of jpeg2png's kind (docs/math.md, 7): its subgradient,
//! against the complex step, and its iterates.

mod support;

use std::ops::ControlFlow;

use jpeg_unround::dct::{self, BASIS, BLOCK, BLOCK_SIZE};
use jpeg_unround::frames::{self, Frame};
use jpeg_unround::model::{DataTerm, Problem, Tv};
use jpeg_unround::operators::{self, Shape};
use jpeg_unround::pdhg::Record;
use jpeg_unround::subgradient::{self, Options};
use support::rounding::{self, U, gamma};
use support::synthetic::{self, Numbers};

/// A complex number, for the complex step.
#[derive(Debug, Clone, Copy)]
struct Complex {
    re: f64,
    im: f64,
}

impl Complex {
    fn add(self, other: Self) -> Self {
        Self {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }

    fn sub(self, other: Self) -> Self {
        Self {
            re: self.re - other.re,
            im: self.im - other.im,
        }
    }

    fn mul(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }

    fn scale(self, factor: f64) -> Self {
        Self {
            re: self.re * factor,
            im: self.im * factor,
        }
    }

    /// The principal square root of a number whose real part is positive and much larger
    /// than its imaginary part: its imaginary part is `im / (2 sqrt(re))` of the root,
    /// which does not cancel.
    fn sqrt(self) -> Self {
        let modulus = self.re.hypot(self.im);
        let re = f64::midpoint(modulus, self.re).sqrt();
        Self {
            re,
            im: self.im / (2.0 * re),
        }
    }
}

/// The TV objective without the constraint, in complex arithmetic, for the complex step.
fn complex_objective(problem: &Problem, weights: &Tv, canvas: &[Complex], shape: Shape) -> Complex {
    let (height, width) = (shape.height, shape.width);
    let zero = Complex { re: 0.0, im: 0.0 };
    let mut variation = zero;
    for i in 0..height {
        for j in 0..width {
            let here = canvas[i * width + j];
            let across = if j + 1 < width {
                canvas[i * width + j + 1].sub(here)
            } else {
                zero
            };
            let down = if i + 1 < height {
                canvas[(i + 1) * width + j].sub(here)
            } else {
                zero
            };
            let square = across.mul(across).add(down.mul(down));
            if square.re > 0.0 {
                variation = variation.add(square.sqrt());
            }
        }
    }
    let mut data = zero;
    for block in 0..problem.rows() * problem.columns() {
        let (row, column) = (block / problem.columns() * BLOCK, block % problem.columns() * BLOCK);
        for (v, vertical) in BASIS.iter().enumerate() {
            for (u, horizontal) in BASIS.iter().enumerate() {
                let mut coefficient = zero;
                for (y, &down) in vertical.iter().enumerate() {
                    for (x, &across) in horizontal.iter().enumerate() {
                        let sample = canvas[(row + y) * width + column + x];
                        coefficient = coefficient.add(sample.scale(down * across));
                    }
                }
                let index = block * BLOCK_SIZE + v * BLOCK + u;
                let difference = coefficient.sub(Complex {
                    re: problem.centres()[index],
                    im: 0.0,
                });
                data = data.add(difference.mul(difference).scale(problem.weights()[v * BLOCK + u]));
            }
        }
    }
    variation.scale(weights.alpha).add(data.scale(0.5))
}

#[test]
fn the_subgradient_is_the_derivative_where_the_objective_is_smooth() {
    // The complex step, Im f(x + i h v) / h with h = 1e-20, gives the directional derivative
    // without the cancellation of a difference quotient: its error is of order h^2 and of
    // the rounding of the terms. The canvas has no zero gradient but at its last sample,
    // where both differences are 0 whatever x is, the term is constant, and the subgradient
    // taken is 0.
    let data = DataTerm {
        mu: Some(40.0),
        ..DataTerm::default()
    };
    let (problem, canvas) = synthetic::problem(8, 16, 31, 2.0, &data);
    let frame = Frame::one(problem.clone());
    let shape = frame.shape();
    let mut numbers = Numbers::new(32);
    let canvas: Vec<f64> = canvas.iter().map(|&value| value + numbers.normal(0.0, 5.0)).collect();
    let weights = Tv {
        alpha: 1.3,
        ..Tv::default()
    };
    let direction = subgradient::subgradient(&frame, &weights, &canvas);
    let h = 1e-20;
    let coefficients = dct::forward(&canvas, shape.height, shape.width);
    for _ in 0..10 {
        let v: Vec<f64> = (0..canvas.len()).map(|_| numbers.normal(0.0, 1.0)).collect();
        let stepped: Vec<Complex> = canvas
            .iter()
            .zip(&v)
            .map(|(&re, &along)| Complex { re, im: h * along })
            .collect();
        let derivative = complex_objective(&problem, &weights, &stepped, shape).im / h;
        let inner: f64 = direction.iter().zip(&v).map(|(a, b)| a * b).sum();
        // The terms' own derivatives, alpha |grad v| and w |c - centre| |D v|, bound what each
        // term contributes; each is computed within a few roundings, and both sums within
        // gamma(n) of the sums of their magnitudes.
        let (mut across, mut down) = (vec![0.0; v.len()], vec![0.0; v.len()]);
        operators::grad(&v, shape, &mut across, &mut down);
        let mut magnitude: f64 = weights.alpha * across.iter().zip(&down).map(|(a, b)| a.hypot(*b)).sum::<f64>();
        let transformed = dct::forward(&v, shape.height, shape.width);
        magnitude += coefficients
            .iter()
            .enumerate()
            .map(|(index, &c)| {
                problem.weights()[index % BLOCK_SIZE] * (c - problem.centres()[index]).abs() * transformed[index].abs()
            })
            .sum::<f64>();
        let mut allowance = (gamma(4 * canvas.len()) + 16.0 * U) * magnitude;
        allowance += gamma(canvas.len()) * direction.iter().zip(&v).map(|(a, b)| (a * b).abs()).sum::<f64>();
        assert!(
            (derivative - inner).abs() <= allowance,
            "{derivative} {inner} {allowance}"
        );
    }
}

#[test]
fn the_iterates_stay_within_the_constraint_set_and_go_down() {
    let (problem, _) = synthetic::problem(16, 24, 33, 2.0, &DataTerm::default());
    let frame = Frame::one(problem.clone());
    let mut checked = Vec::new();
    let mut observe = |record: &Record<'_>| {
        checked.push(record.iteration);
        assert!(record.gap.is_infinite()); // the method has no dual, and no gap
        let excess = problem.excess(&dct::forward(record.canvas, problem.height(), problem.width()));
        for block in 0..problem.rows() * problem.columns() {
            let reach = rounding::roundtrip_error(&rounding::block_of(&record.coefficients[0], block));
            for frequency in 0..BLOCK_SIZE {
                let index = block * BLOCK_SIZE + frequency;
                assert!(excess[index] * problem.steps()[frequency] <= reach[frequency / BLOCK][frequency % BLOCK]);
            }
        }
        ControlFlow::Continue(())
    };
    let options = Options {
        iterations: 50,
        record_every: 5,
        ..Options::default()
    };
    let result = subgradient::solve_tv(&frame, &Tv::default(), &options, None, Some(&mut observe)).expect("a result");
    assert_eq!(result.iterations, 50);
    assert_eq!(checked, (5..=50).step_by(5).collect::<Vec<u64>>());
    assert!(result.history.primal[result.history.primal.len() - 1] < result.history.primal[0]);
    assert!(!result.converged());
    assert!(result.dual.is_none());
}

#[test]
fn the_iterates_follow_the_options() {
    // The scheme of docs/math.md, 7, written out with the same operations in the same order:
    // equal to the last bit.
    let (problem, _) = synthetic::problem(16, 24, 34, 2.0, &DataTerm::default());
    let frame = Frame::one(problem.clone());
    let weights = Tv::default();
    for options in [
        Options {
            iterations: 4,
            record_every: 0,
            ..Options::default()
        },
        Options {
            iterations: 4,
            record_every: 0,
            step: 0.25,
            decay: 1.0,
            momentum: false,
        },
        Options {
            iterations: 4,
            record_every: 0,
            step: 1.0,
            decay: 0.0,
            ..Options::default()
        },
    ] {
        let result = subgradient::solve_tv(&frame, &weights, &options, None, None).expect("a result");
        let mut canvas = frames::start(&frame, None).expect("a start").canvas;
        let mut extrapolated = canvas.clone();
        let mut t: f64 = 1.0;
        let samples = f64::from(u32::try_from(canvas.len()).expect("fits"));
        for n in 0..options.iterations {
            let direction = subgradient::subgradient(&frame, &weights, &extrapolated);
            let length = jpeg_unround::exact::sum(direction.iter().map(|value| value * value)).sqrt();
            let base = 1.0 + f64::from(u32::try_from(n).expect("fits"));
            #[expect(clippy::float_cmp, reason = "the decay 1/2 is taken exactly as it is given")]
            let power = if options.decay == 0.5 {
                base.sqrt()
            } else {
                base.powf(options.decay)
            };
            let h = options.step * samples.sqrt() / power;
            let moved: Vec<f64> = extrapolated
                .iter()
                .zip(&direction)
                .map(|(&value, &along)| value - (h / length) * along)
                .collect();
            let mut coefficients = dct::forward(&moved, problem.height(), problem.width());
            problem.clip(&mut coefficients);
            let following = dct::inverse(&coefficients, problem.rows(), problem.columns());
            let t_next = f64::midpoint(1.0, (1.0 + 4.0 * t * t).sqrt());
            extrapolated = if options.momentum {
                following
                    .iter()
                    .zip(&canvas)
                    .map(|(&new, &old)| new + ((t - 1.0) / t_next) * (new - old))
                    .collect()
            } else {
                following.clone()
            };
            (canvas, t) = (following, t_next);
        }
        assert_eq!(result.iterations, options.iterations);
        assert_eq!(result.primal.canvas, canvas);
    }
}

#[test]
fn the_options_out_of_their_ranges_are_refused() {
    let (problem, _) = synthetic::problem(16, 24, 35, 2.0, &DataTerm::default());
    let frame = Frame::one(problem);
    for (options, fragment) in [
        (
            Options {
                step: 0.0,
                ..Options::default()
            },
            "step",
        ),
        (
            Options {
                step: f64::INFINITY,
                ..Options::default()
            },
            "step",
        ),
        (
            Options {
                decay: -0.5,
                ..Options::default()
            },
            "decay",
        ),
        (
            Options {
                decay: f64::NAN,
                ..Options::default()
            },
            "decay",
        ),
    ] {
        let error = subgradient::solve_tv(&frame, &Tv::default(), &options, None, None).expect_err("refused");
        assert!(error.to_string().contains(fragment), "{error}");
    }
    let (colour, _) = synthetic::colour_frame(20, 30, 36, (2, 2), &DataTerm::default());
    let error = subgradient::solve_tv(&colour, &Tv::default(), &Options::default(), None, None).expect_err("refused");
    assert!(error.to_string().contains("one component"), "{error}");
}
