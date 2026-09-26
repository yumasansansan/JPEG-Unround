// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! Components on one canvas (docs/math.md, 1.3, 4.4 and 6.6).
//!
//! The maps between the canvas and a component are checked exactly, on dyadic
//! values that they compute without rounding. The proximal map is checked against
//! its two conditions, within bounds of its rounding computed in exact arithmetic,
//! and the projection within bounds of its rounding.

#![allow(
    clippy::float_cmp,
    reason = "the tests compare doubles to the last bit where the arithmetic is exact or the same"
)]

mod support;

use std::cmp::Ordering;

use jpeg_unround::dct::{self, BLOCK, BLOCK_SIZE};
use jpeg_unround::frames::{self, Channel, Frame, Tensor, Vector};
use jpeg_unround::model::{DataTerm, Problem, Tv};
use support::exact::{self, Dyadic};
use support::rounding::{self, U, gamma};
use support::synthetic::{self, Numbers};

fn blank(height: usize, width: usize) -> Problem {
    let blocks = (height / BLOCK, width / BLOCK);
    Problem::new(
        &vec![0i16; blocks.0 * blocks.1 * BLOCK_SIZE],
        blocks.0,
        blocks.1,
        &[1; BLOCK_SIZE],
        &DataTerm::default(),
        None,
    )
    .expect("a problem")
}

#[test]
fn the_canvas_is_the_picture_in_whole_mcus() {
    type Case = ((usize, usize), [(usize, usize); 3], (usize, usize), [(usize, usize); 3]);
    let cases: [Case; 5] = [
        (
            (21, 30),
            [(2, 2), (1, 1), (1, 1)],
            (32, 32),
            [(24, 32), (16, 16), (16, 16)],
        ),
        (
            (21, 30),
            [(2, 1), (1, 1), (1, 1)],
            (24, 32),
            [(24, 32), (24, 16), (24, 16)],
        ),
        (
            (20, 30),
            [(1, 2), (1, 1), (1, 1)],
            (32, 32),
            [(24, 32), (16, 32), (16, 32)],
        ),
        (
            (20, 30),
            [(1, 1), (1, 1), (1, 1)],
            (24, 32),
            [(24, 32), (24, 32), (24, 32)],
        ),
        (
            (20, 30),
            [(4, 1), (1, 1), (1, 1)],
            (24, 32),
            [(24, 32), (24, 8), (24, 8)],
        ),
    ];
    for (size, factors, shape, blocks) in cases {
        let problems = blocks.iter().map(|&(height, width)| blank(height, width)).collect();
        let frame = Frame::of_file(problems, &factors, size).expect("a frame");
        assert_eq!((frame.height(), frame.width()), shape);
        let most_across = factors.iter().map(|factor| factor.0).max().expect("factors");
        let most_down = factors.iter().map(|factor| factor.1).max().expect("factors");
        for (channel, &(across, down)) in frame.channels().iter().zip(&factors) {
            assert_eq!(channel.ratio(), (most_down / down, most_across / across));
            let (rows, columns) = channel.extent();
            assert!(rows <= shape.0 && columns <= shape.1);
        }
        assert_eq!(frame.samples(), 3 * shape.0 * shape.1);
    }
}

#[test]
fn frames_that_are_not_whole_are_refused() {
    let refused = |result: Result<Frame, jpeg_unround::Error>, fragment: &str| {
        let error = result.expect_err("refused");
        assert!(error.to_string().contains(fragment), "{error}");
    };
    let three = || vec![blank(24, 32), blank(24, 32), blank(24, 32)];
    refused(
        Frame::of_file(three(), &[(3, 1), (2, 1), (2, 1)], (20, 30)),
        "do not divide",
    );
    refused(
        Frame::of_file(
            vec![blank(24, 32), blank(24, 32), blank(16, 16)],
            &[(2, 2), (1, 1), (1, 1)],
            (21, 30),
        ),
        "where a picture of",
    );
    refused(
        Frame::of_file(three(), &[(1, 1), (1, 1)], (20, 30)),
        "a sampling factor for each",
    );
    refused(
        Frame::new(vec![Channel::new(blank(8, 8), (3, 1))], 16, 16),
        "do not tile",
    );
    refused(
        Frame::new(vec![Channel::new(blank(16, 16), (2, 1))], 16, 16),
        "go beyond",
    );
    refused(Frame::new(Vec::new(), 16, 16), "a channel at least");
    // One component is its blocks, whatever its sampling factors.
    let single = Frame::of_file(vec![blank(24, 32)], &[(2, 2)], (20, 30)).expect("a frame");
    assert_eq!((single.height(), single.width()), (24, 32));
    assert!(!single.free(&single.channels()[0]));
}

/// Dyadic rationals `m / 2^k`, `|m| <= 1000` and `k <= 6`, which sums of four and
/// quarters of them hold exactly.
fn dyadics(size: usize, numbers: &mut Numbers) -> Vec<f64> {
    (0..size)
        .map(|_| {
            let numerator = numbers.integer(-1000, 1000);
            let power = numbers.integer(0, 6);
            f64::from(i32::try_from(numerator).expect("fits")) / f64::from(1u32 << power)
        })
        .collect()
}

fn pi(frame: &Frame, channel: &Channel, canvas: &[f64]) -> Vec<f64> {
    frames::spread(frame, channel, &frames::means(channel, canvas, frame.width()))
}

#[test]
fn the_maps_of_a_component_are_exact() {
    // On a canvas of 16 x 32 and a component of one block (8 x 8 samples), whose cells
    // cover 8 r_v x 8 r_h: sums and spread are adjoint, the means of what spread spreads
    // are what it was given, and Pi = spread(means) is an orthogonal projection.
    for ratio in [(1, 1), (1, 2), (2, 1), (2, 2), (1, 4)] {
        let frame = Frame::new(vec![Channel::new(blank(8, 8), ratio)], 16, 32).expect("a frame");
        let channel = &frame.channels()[0];
        let mut numbers = Numbers::new(1);
        let (x, z) = (dyadics(16 * 32, &mut numbers), dyadics(16 * 32, &mut numbers));
        let y = dyadics(64, &mut numbers);
        let spread = frames::spread(&frame, channel, &y);
        let sums = frames::sums(channel, &x, 32);
        assert!(exact::inner(&spread, &x).equals(&exact::inner(&y, &sums)));
        assert_eq!(frames::means(channel, &spread, 32), y);
        let pi_x = pi(&frame, channel, &x);
        assert_eq!(pi(&frame, channel, &pi_x), pi_x);
        let pi_z = pi(&frame, channel, &z);
        assert!(exact::inner(&pi_x, &z).equals(&exact::inner(&x, &pi_z)));
        // <xi, x> splits into the sums and the means, and what Pi leaves (docs/math.md, 6.6).
        let rest_z: Vec<f64> = z.iter().zip(&pi_z).map(|(a, b)| a - b).collect();
        let rest_x: Vec<f64> = x.iter().zip(&pi_x).map(|(a, b)| a - b).collect();
        let split = exact::inner(&frames::sums(channel, &z, 32), &frames::means(channel, &x, 32))
            .add(&exact::inner(&rest_z, &rest_x));
        assert!(exact::inner(&z, &x).equals(&split), "{ratio:?}");
    }
}

/// `n U / (1 - n U)`, bounded above by a dyadic: `n U (1 + 2 n U)`.
fn gamma_above(n: usize) -> Dyadic {
    let count = Dyadic::from_i128(i128::try_from(n).expect("fits"));
    let unit = Dyadic::from_f64(U);
    count
        .mul(&unit)
        .mul(&Dyadic::from_i128(1).add(&count.mul(&unit).scaled(1)))
}

/// `U / (1 - U)`, bounded above by a dyadic: `U (1 + 2U)`.
fn unit_above() -> Dyadic {
    gamma_above(1)
}

/// `after` is `before` moved by the change to `samples` of its cells' means: checked in
/// exact arithmetic.
///
/// `after = fl(before + c)`, `c = fl(samples - m)`, `m` the means of `before`'s cells as
/// the crate computes them. Each `after_i` is within `U / (1 - U) |after_i|` of
/// `before_i + c`, so the deviations from the cells' exact means move by at most that
/// and its cell's mean; and the exact mean of a cell of `after` is within
/// `gamma(n) mean|before|` (the rounding of `m`) plus `U |samples - m|` (that of `c`)
/// plus `U / (1 - U) mean|after|` of the sample.
fn check_cells(frame: &Frame, channel: &Channel, before: &[f64], after: &[f64], samples: &[f64]) {
    let (down, across) = channel.ratio();
    let n = channel.cells();
    let width = frame.width();
    let columns = channel.problem().width();
    let own = frames::means(channel, before, width);
    let count = Dyadic::from_i128(i128::try_from(n).expect("fits"));
    let unit = unit_above();
    for i in 0..channel.problem().height() {
        for j in 0..columns {
            let cell: Vec<usize> = (0..down)
                .flat_map(|a| (0..across).map(move |b| (i * down + a) * width + j * across + b))
                .collect();
            let old_sum = exact::sum(cell.iter().map(|&index| before[index]));
            let new_sum = exact::sum(cell.iter().map(|&index| after[index]));
            let old_size = exact::sum(cell.iter().map(|&index| before[index].abs()));
            let new_size = exact::sum(cell.iter().map(|&index| after[index].abs()));
            for &index in &cell {
                // n (new - old) - (new_sum - old_sum), against n times the bound.
                let moved = Dyadic::from_f64(after[index])
                    .sub(&Dyadic::from_f64(before[index]))
                    .mul(&count)
                    .sub(&new_sum.sub(&old_sum))
                    .abs();
                let allowed = unit.mul(&Dyadic::from_f64(after[index]).abs().mul(&count).add(&new_size));
                assert!(moved.at_most(&allowed), "{i} {j}");
            }
            let target = samples[i * columns + j];
            // n |new mean - target| against n times the bound.
            let reach = gamma_above(n)
                .mul(&old_size)
                .add(
                    &Dyadic::from_f64(U)
                        .mul(&Dyadic::from_f64(target - own[i * columns + j]).abs())
                        .mul(&count),
                )
                .add(&unit.mul(&new_size));
            let off = new_sum.sub(&Dyadic::from_f64(target).mul(&count)).abs();
            assert!(off.at_most(&reach), "{i} {j}");
        }
    }
}

#[test]
fn the_proximal_map_moves_the_cells_and_keeps_the_free_samples() {
    // prox_{tau G}(v) is characterized by two conditions (docs/math.md, 4.4): its part that
    // Pi leaves is v's, and its coefficients are the proximal map of the model of v's, with
    // the step tau / n, whose optimality the tests of the model check. The picture of 20 x 30
    // leaves Y's last block row free where MCUs are 16 rows.
    let data = DataTerm {
        mu: Some(5.0),
        ..DataTerm::default()
    };
    for ratio in [(2, 2), (1, 2), (1, 1)] {
        for tau in [0.5, 40.0] {
            let (frame, truth) = synthetic::colour_frame(20, 30, 40, ratio, &data);
            let mut numbers = Numbers::new(41);
            let v: Vec<f64> = truth.iter().map(|&value| value + numbers.normal(0.0, 30.0)).collect();
            let mut coefficients: Vec<Vec<f64>> = frame
                .channels()
                .iter()
                .map(|channel| vec![0.0; channel.problem().samples()])
                .collect();
            let mut canvas = vec![0.0; v.len()];
            frames::prox(&frame, &frames::steps(&frame, tau), &v, &mut coefficients, &mut canvas);
            let plane = frame.plane();
            for (index, channel) in frame.channels().iter().enumerate() {
                let problem = channel.problem();
                let source = &v[index * plane..(index + 1) * plane];
                let target = &canvas[index * plane..(index + 1) * plane];
                let own = frames::means(channel, source, frame.width());
                let mut expected = dct::forward(&own, problem.height(), problem.width());
                let cells = f64::from(u32::try_from(channel.cells()).expect("fits"));
                problem.prox(tau / cells, &mut expected);
                assert_eq!(coefficients[index], expected);
                let (rows, columns) = channel.extent();
                for i in 0..frame.height() {
                    for j in 0..frame.width() {
                        if i >= rows || j >= columns {
                            let at = i * frame.width() + j;
                            assert_eq!(target[at].to_bits(), source[at].to_bits());
                        }
                    }
                }
                let samples = dct::inverse(&expected, problem.rows(), problem.columns());
                if channel.cells() == 1 {
                    for i in 0..rows {
                        for j in 0..columns {
                            assert_eq!(
                                target[i * frame.width() + j].to_bits(),
                                samples[i * columns + j].to_bits()
                            );
                        }
                    }
                } else {
                    check_cells(&frame, channel, source, target, &samples);
                }
            }
        }
    }
}

/// `|B| values |B|^T` of every block of a component's samples, `(1 + 2U) |B|` for
/// `|C|`, as a bound carries rounding through a DCT.
fn carried(problem: &Problem, samples: &[f64]) -> Vec<f64> {
    let mut result = vec![0.0; samples.len()];
    for block in 0..problem.rows() * problem.columns() {
        let (row, column) = (block / problem.columns() * BLOCK, block % problem.columns() * BLOCK);
        let magnitude = rounding::forward_magnitude(&dct::load(samples, problem.width(), row, column));
        let scaled = magnitude.map(|line| line.map(|value| value * (1.0 + 2.0 * U) * (1.0 + 2.0 * U)));
        dct::flatten(&scaled, &mut result[block * BLOCK_SIZE..(block + 1) * BLOCK_SIZE]);
    }
    result
}

/// A bound, coefficient by coefficient, of `|D(means of canvas) - coefficients|` as the
/// DCT computes it, where `canvas` was made from `before` by the projection with these
/// coefficients.
///
/// Its cells' means, computed, are within `gamma(n) mean|canvas|` of their exact means,
/// which are within the bound of `check_cells` of the inverse DCT of the coefficients:
/// together within `(gamma(n) + 4U) (mean|before| + mean|canvas|) + 2U |samples - m|`,
/// the means of the magnitudes computed within `gamma(n)` and so taken 8U larger, and
/// the whole 8U larger for its own rounding. Where the cells are single samples, the
/// means are the inverse DCT itself. That is within the inverse DCT's bound of `D^T`
/// times the coefficients; the DCT carries it through `|C| <= (1 + U) |B|`, and rounds
/// within its own bound.
fn within(frame: &Frame, canvas: &[f64], coefficients: &[Vec<f64>], before: &[f64]) -> Vec<Vec<f64>> {
    let plane = frame.plane();
    let width = frame.width();
    frame
        .channels()
        .iter()
        .enumerate()
        .map(|(index, channel)| {
            let problem = channel.problem();
            let n = channel.cells();
            let means = frames::means(channel, &canvas[index * plane..(index + 1) * plane], width);
            let samples = dct::inverse(&coefficients[index], problem.rows(), problem.columns());
            let mut error = vec![0.0; means.len()];
            if n > 1 {
                let own = frames::means(channel, &before[index * plane..(index + 1) * plane], width);
                let size = |canvas: &[f64]| -> Vec<f64> {
                    let magnitude: Vec<f64> = canvas.iter().map(|value| value.abs()).collect();
                    frames::means(channel, &magnitude[index * plane..(index + 1) * plane], width)
                        .iter()
                        .map(|value| value * (1.0 + 8.0 * U))
                        .collect()
                };
                let (average, after) = (size(before), size(canvas));
                for (k, value) in error.iter_mut().enumerate() {
                    let spread = (gamma(n) + 4.0 * U) * (average[k] + after[k]);
                    *value = (spread + 2.0 * U * (samples[k] - own[k]).abs()) * (1.0 + 8.0 * U);
                }
            }
            let inverse = {
                let mut bound = vec![0.0; means.len()];
                for block in 0..problem.rows() * problem.columns() {
                    let (row, column) = (block / problem.columns() * BLOCK, block % problem.columns() * BLOCK);
                    let own = rounding::inverse_error(&rounding::block_of(&coefficients[index], block));
                    dct::store(&own, &mut bound, problem.width(), row, column);
                }
                bound
            };
            for (value, &extra) in error.iter_mut().zip(&inverse) {
                *value += extra;
            }
            let through = carried(problem, &error);
            let mut result = vec![0.0; coefficients[index].len()];
            for block in 0..problem.rows() * problem.columns() {
                let (row, column) = (block / problem.columns() * BLOCK, block % problem.columns() * BLOCK);
                let own = rounding::forward_error(&dct::load(&means, problem.width(), row, column));
                let target = &mut result[block * BLOCK_SIZE..(block + 1) * BLOCK_SIZE];
                dct::flatten(&own, target);
                for (value, &extra) in target
                    .iter_mut()
                    .zip(&through[block * BLOCK_SIZE..(block + 1) * BLOCK_SIZE])
                {
                    *value += extra;
                }
            }
            result
        })
        .collect()
}

#[test]
fn the_projection_lands_in_the_set_and_is_idempotent() {
    for ratio in [(2, 2), (1, 2), (1, 1)] {
        let (frame, truth) = synthetic::colour_frame(20, 30, 42, ratio, &DataTerm::default());
        let mut numbers = Numbers::new(43);
        let v: Vec<f64> = truth.iter().map(|&value| value + numbers.normal(0.0, 30.0)).collect();
        let once = frames::project(&frame, &v);
        let plane = frame.plane();
        let clipped: Vec<Vec<f64>> = frame
            .channels()
            .iter()
            .enumerate()
            .map(|(index, channel)| {
                let problem = channel.problem();
                let own = frames::means(channel, &v[index * plane..(index + 1) * plane], frame.width());
                let mut values = dct::forward(&own, problem.height(), problem.width());
                problem.clip(&mut values);
                values
            })
            .collect();
        let reach = within(&frame, &once, &clipped, &v);
        let excess = frames::excess(&frame, &once);
        for (index, channel) in frame.channels().iter().enumerate() {
            for (k, &value) in excess[index].iter().enumerate() {
                assert!(
                    value * channel.problem().steps()[k % BLOCK_SIZE] <= reach[index][k],
                    "{ratio:?}"
                );
            }
        }
        // A second projection moves the canvas by no more than the first left its
        // coefficients outside their intervals, carried back through the inverse DCT, with
        // the rounding of both inverse DCTs and of the means.
        let twice = frames::project(&frame, &once);
        for (index, channel) in frame.channels().iter().enumerate() {
            let problem = channel.problem();
            let range = index * plane..(index + 1) * plane;
            let means = frames::means(channel, &once[range.clone()], frame.width());
            let mut again = dct::forward(&means, problem.height(), problem.width());
            problem.clip(&mut again);
            let mut moved = vec![0.0; means.len()];
            for block in 0..problem.rows() * problem.columns() {
                let (row, column) = (block / problem.columns() * BLOCK, block % problem.columns() * BLOCK);
                let carried = rounding::inverse_magnitude(&rounding::block_of(&reach[index], block))
                    .map(|line| line.map(|value| value * (1.0 + 2.0 * U) * (1.0 + 2.0 * U)));
                let first = rounding::inverse_error(&rounding::block_of(&again, block));
                let second = rounding::inverse_error(&rounding::block_of(&clipped[index], block));
                for i in 0..BLOCK {
                    for j in 0..BLOCK {
                        moved[(row + i) * problem.width() + column + j] = carried[i][j] + first[i][j] + second[i][j];
                    }
                }
            }
            let exact_samples = dct::inverse(&clipped[index], problem.rows(), problem.columns());
            let drift: Vec<f64> = if channel.cells() > 1 {
                means.iter().zip(&exact_samples).map(|(a, b)| (a - b).abs()).collect()
            } else {
                vec![0.0; means.len()]
            };
            let total: Vec<f64> = moved
                .iter()
                .zip(&drift)
                .map(|(a, b)| (a + b) * (1.0 + 4.0 * U))
                .collect();
            let allowed = frames::spread(&frame, channel, &total);
            for (k, (&after, &before)) in twice[range.clone()].iter().zip(&once[range]).enumerate() {
                assert!(
                    (after - before).abs() <= allowed[k] + 2.0 * U * after.abs(),
                    "{ratio:?} {k}"
                );
            }
        }
    }
}

#[test]
fn the_channels_coupled_and_apart() {
    // Apart, the TV of the channels is the sum of their TVs, channel by channel, operation
    // for operation; coupled, it is less, by far more than rounding, the channels' edges
    // not all being at the same pixels.
    let (frame, _) = synthetic::colour_frame(20, 30, 46, (2, 2), &DataTerm::default());
    let point = frames::start(&frame, None).expect("a start");
    let data = frames::data_term(&frame, &point.coefficients);
    let apart_weights = Tv {
        alpha: 1.5,
        coupled: false,
        ..Tv::default()
    };
    let coupled_weights = Tv {
        alpha: 1.5,
        ..Tv::default()
    };
    let apart = frames::variation(&frame, &apart_weights, &point.canvas).expect("the variation");
    let coupled = frames::variation(&frame, &coupled_weights, &point.canvas).expect("the variation");
    assert!(coupled < 0.99 * apart);
    let objective = |weights: &Tv| {
        frames::tv_objective(&frame, weights, &point.coefficients, &point.canvas).expect("the objective")
    };
    assert_eq!(objective(&apart_weights).to_bits(), (1.5 * apart + data).to_bits());
    assert_eq!(objective(&coupled_weights).to_bits(), (1.5 * coupled + data).to_bits());
    let plane = frame.plane();
    let each: Vec<f64> = (0..3)
        .map(|index| {
            let problem = frame.channels()[index].problem().clone();
            let single = Frame::new(vec![Channel::new(problem, (1, 1))], frame.height(), frame.width());
            let canvas = &point.canvas[index * plane..(index + 1) * plane];
            frames::variation(&single.expect("a frame"), &Tv::default(), canvas).expect("the variation")
        })
        .collect();
    assert_eq!(apart.to_bits(), ((each[0] + each[1]) + each[2]).to_bits());
    // A projection onto the coupled balls: the norm over the channels at a pixel, computed
    // within (C + 1) U, the quotient by the radius within U more, the division within U,
    // and the test's own norm within (C + 1) U: (2 C + 4) U, and 2 U of slack.
    let mut numbers = Numbers::new(47);
    let size = frame.samples();
    let p = Vector {
        x: (0..size).map(|_| numbers.normal(0.0, 2.0)).collect(),
        y: (0..size).map(|_| numbers.normal(0.0, 2.0)).collect(),
    };
    let mut projected = p.clone();
    frames::project_vectors(&frame, &mut projected, 1.5, true);
    for pixel in 0..plane {
        let norm = |field: &Vector| {
            (0..3)
                .map(|channel| {
                    let index = channel * plane + pixel;
                    field.x[index] * field.x[index] + field.y[index] * field.y[index]
                })
                .sum::<f64>()
                .sqrt()
        };
        assert!(norm(&projected) <= 1.5 * (1.0 + 12.0 * U));
        if norm(&p) <= 1.5 {
            for channel in 0..3 {
                let index = channel * plane + pixel;
                assert!(projected.x[index] == p.x[index] && projected.y[index] == p.y[index]);
            }
        }
    }
    let r = Tensor {
        xx: (0..size).map(|_| numbers.normal(0.0, 2.0)).collect(),
        yy: (0..size).map(|_| numbers.normal(0.0, 2.0)).collect(),
        xy: (0..size).map(|_| numbers.normal(0.0, 2.0)).collect(),
    };
    let mut projected = r;
    frames::project_tensors(&frame, &mut projected, 1.5, true);
    for pixel in 0..plane {
        let square: f64 = (0..3)
            .map(|channel| {
                let index = channel * plane + pixel;
                let (a, b, c) = (projected.xx[index], projected.yy[index], projected.xy[index]);
                a * a + b * b + 2.0 * c * c
            })
            .sum();
        assert!(square.sqrt() <= 1.5 * (1.0 + 14.0 * U));
    }
}

#[test]
fn the_channel_weights_scale_each_channel() {
    // Doubling every difference is exact, and so doubles the norms and their sum to the
    // last bit.
    let (frame, _) = synthetic::colour_frame(20, 30, 48, (2, 2), &DataTerm::default());
    let point = frames::start(&frame, None).expect("a start");
    let data = frames::data_term(&frame, &point.coefficients);
    let plain = frames::variation(&frame, &Tv::default(), &point.canvas).expect("the variation");
    let doubled = Tv {
        channel_weights: Some(vec![2.0, 2.0, 2.0]),
        ..Tv::default()
    };
    let value = frames::tv_objective(&frame, &doubled, &point.coefficients, &point.canvas).expect("the objective");
    assert_eq!(value.to_bits(), (2.0 * plain + data).to_bits());
    for weights in [vec![1.0, 1.0], vec![1.0, 0.0, 1.0]] {
        let error = frame.channel_weights(Some(&weights)).expect_err("refused");
        assert!(error.to_string().contains("each of the 3 channels"), "{error}");
    }
}

#[test]
fn the_bound_of_the_free_samples() {
    // G* of a frame is the model's G* of the cells' sums, plus, where samples are free,
    // <zeta, m> + R ||zeta||_1 (docs/math.md, 6.6): with the radius, it grows by R times
    // the size of what Pi leaves of xi. Each component's term is its G* plus 128 sum(zeta
    // beyond the blocks) plus R sum|zeta|, each sum within gamma(N) of the sum of its
    // terms' magnitudes, and the three terms and the channels added with a rounding each:
    // the difference of two radii is R times the sums of |zeta|, within gamma(N + 8) of
    // twice all those magnitudes.
    let (frame, _) = synthetic::colour_frame(20, 30, 49, (2, 2), &DataTerm::default());
    let mut numbers = Numbers::new(50);
    let xi: Vec<f64> = (0..frame.samples()).map(|_| numbers.normal(0.0, 1.0)).collect();
    let none = frames::conjugate(&frame, &xi, 0.0);
    let some = frames::conjugate(&frame, &xi, 10.0);
    let plane = frame.plane();
    let (mut free, mut magnitude) = (0.0, 0.0);
    for (index, channel) in frame.channels().iter().enumerate() {
        let own = &xi[index * plane..(index + 1) * plane];
        let averaged = pi(&frame, channel, own);
        let size: f64 = own.iter().zip(&averaged).map(|(a, b)| (a - b).abs()).sum();
        free += size;
        let problem = channel.problem();
        let conjugate = problem.conjugate(&dct::forward(
            &frames::sums(channel, own, frame.width()),
            problem.height(),
            problem.width(),
        ));
        magnitude += conjugate.abs() + frames::FREE_CENTRE * size + 10.0 * size;
    }
    assert!(free > 0.0);
    assert!(((some - none) - 10.0 * free).abs() <= gamma(frame.samples() + 8) * 2.0 * magnitude);
    let (whole, _) = synthetic::colour_frame(24, 32, 49, (1, 1), &DataTerm::default());
    assert!(!whole.channels().iter().any(|channel| whole.free(channel)));
    let xi: Vec<f64> = (0..whole.samples()).map(|_| numbers.normal(0.0, 1.0)).collect();
    let each: Vec<f64> = whole
        .channels()
        .iter()
        .enumerate()
        .map(|(index, channel)| {
            let problem = channel.problem();
            let own = &xi[index * whole.plane()..(index + 1) * whole.plane()];
            problem.conjugate(&dct::forward(own, problem.height(), problem.width()))
        })
        .collect();
    let total = frames::conjugate(&whole, &xi, 10.0);
    assert_eq!(
        total.partial_cmp(&((each[0] + each[1]) + each[2])),
        Some(Ordering::Equal)
    );
}
