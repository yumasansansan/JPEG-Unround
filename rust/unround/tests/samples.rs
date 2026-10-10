// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! Pictures of rounded samples (docs/math.md, 11): the intervals exactly, the doubles
//! that round to each level and no others, the proximal map's optimality, the
//! projection's idempotence, the conjugate by Fenchel and Young, and the solvers on
//! smooth pictures rounded to levels, with their defaults of when they stop.

mod support;

use std::cmp::Ordering;

use jpeg_unround::model::{Tgv, Tv};
use jpeg_unround::output;
use jpeg_unround::pdhg::Options;
use jpeg_unround::results::Stop;
use jpeg_unround::samples::{self, Data, Method, Model};
use support::exact::Dyadic;

/// A smooth picture of `channels x height x width`, its values, and their levels of 8
/// bits (the nearest, halves away from 0).
fn smooth(channels: usize, height: usize, width: usize) -> (Vec<f64>, Vec<u16>) {
    let mut values = Vec::with_capacity(channels * height * width);
    for channel in 0..channels {
        for i in 0..height {
            for j in 0..width {
                let (c, y, x) = (
                    f64::from(u32::try_from(channel).expect("small")),
                    f64::from(u32::try_from(i).expect("small")),
                    f64::from(u32::try_from(j).expect("small")),
                );
                // A ramp across, a gentler one down, and a bend: steps of a level
                // about every 4 to 10 samples.
                values.push(40.0 + 30.0 * c + 0.23 * x + 0.11 * y + 0.002 * (x - 20.0) * (x - 20.0));
            }
        }
    }
    // The nearest levels, halves away from 0, as the library rounds 8-bit output.
    let levels = output::eight_bits(&values, 1, values.len(), 1, output::Dither::None)
        .expect("finite values")
        .into_iter()
        .map(u16::from)
        .collect();
    (values, levels)
}

#[test]
fn the_intervals_are_half_a_level_on_each_side_within_0_and_255() {
    // From half a level below, taken, to half a level above, left out: the double
    // below it, but at 255.
    let levels = [0u16, 1, 128, 254, 255];
    let model = Model::new(&levels, (1, 1, 5), 8, &Data::default()).expect("a model");
    assert_eq!(model.low(), [0.0, 0.5, 127.5, 253.5, 254.5]);
    let below = [0.5_f64, 1.5, 128.5, 254.5].map(f64::next_down);
    assert_eq!(model.high(), [below[0], below[1], below[2], below[3], 255.0]);
    assert_eq!(model.centres(), [0.0, 1.0, 128.0, 254.0, 255.0]);
    // Four bits: levels of 17, half of it on each side.
    let model = Model::new(&[0, 3, 15], (1, 1, 3), 4, &Data::default()).expect("a model");
    assert_eq!(model.low(), [0.0, 42.5, 246.5]);
    assert_eq!(model.high(), [8.5_f64.next_down(), 59.5_f64.next_down(), 255.0]);
    assert_eq!(model.centres(), [0.0, 51.0, 255.0]);
    // Sixteen bits: (2v -+ 1) 255 / 131070, rounded inward; the centre the nearest.
    let model = Model::new(&[1000], (1, 1, 1), 16, &Data::default()).expect("a model");
    assert_eq!(against(model.low()[0], 1999 * 255, 131_070), Ordering::Greater);
    assert_eq!(against(model.high()[0], 2001 * 255, 131_070), Ordering::Less);
    assert_eq!(model.centres()[0], 1000.0 * 255.0 / 65535.0);
    // A slack widens them, within 0 and 255.
    let data = Data { mu: 0.0, slack: 0.25 };
    let model = Model::new(&[0, 100, 255], (1, 1, 3), 8, &data).expect("a model");
    assert_eq!(model.low(), [0.0, 99.25, 254.25]);
    assert_eq!(model.high(), [0.75, 100.75, 255.0]);
    for wrong in [
        Model::new(&[256], (1, 1, 1), 8, &Data::default()),
        Model::new(&[1, 2], (1, 1, 1), 8, &Data::default()),
        Model::new(&[1], (1, 1, 1), 3, &Data::default()),
        Model::new(&[1], (1, 1, 1), 8, &Data { mu: -1.0, slack: 0.0 }),
    ] {
        assert!(wrong.is_err());
    }
}

/// A double against `numerator / denominator`, `denominator > 0`, exactly.
fn against(value: f64, numerator: i128, denominator: i128) -> Ordering {
    Dyadic::from_f64(value)
        .mul(&Dyadic::from_i128(denominator))
        .compare(&Dyadic::from_i128(numerator))
}

#[test]
fn every_interval_holds_the_doubles_that_round_to_its_level() {
    for bits in [1u32, 2, 4, 8, 16] {
        let top = (1u32 << bits) - 1;
        let levels: Vec<u16> = (0..=top).map(|level| u16::try_from(level).expect("16 bits")).collect();
        let model = Model::new(&levels, (1, 1, levels.len()), bits, &Data::default()).expect("a model");
        let wide = i128::from(top);
        for (level, (&low, &high)) in (0..).zip(model.low().iter().zip(model.high())) {
            // The least double at or above max(0, (v - 1/2) s).
            if level == 0 {
                assert_eq!(low, 0.0);
            } else {
                let (numerator, denominator) = ((2 * level - 1) * 255, 2 * wide);
                assert_ne!(against(low, numerator, denominator), Ordering::Less, "{bits}: {level}");
                assert_eq!(against(low.next_down(), numerator, denominator), Ordering::Less);
            }
            // The greatest double below (v + 1/2) s, or 255 at the last level.
            if level == wide {
                assert_eq!(high, 255.0);
            } else {
                let (numerator, denominator) = ((2 * level + 1) * 255, 2 * wide);
                assert_eq!(against(high, numerator, denominator), Ordering::Less, "{bits}: {level}");
                assert_ne!(against(high.next_up(), numerator, denominator), Ordering::Less);
            }
        }
        if bits != 8 && bits != 16 {
            continue;
        }
        // Rounded to the file's own bits, the ends give back their level, and the
        // doubles beyond them the levels beside it (docs/math.md, 10).
        let ends: Vec<f64> = model
            .low()
            .iter()
            .zip(model.high())
            .flat_map(|(&low, &high)| [low, high, low.next_down(), high.next_up()])
            .collect();
        let count = ends.len();
        let made: Vec<u32> = if bits == 8 {
            output::eight_bits(&ends, 1, count, 1, output::Dither::None)
                .expect("finite samples")
                .into_iter()
                .map(u32::from)
                .collect()
        } else {
            output::sixteen_bits(&ends, 1, count, 1, output::Dither::None)
                .expect("finite samples")
                .into_iter()
                .map(u32::from)
                .collect()
        };
        for (level, four) in (0..).zip(made.chunks(4)) {
            assert_eq!((four[0], four[1]), (level, level), "{bits}: {level}");
            if level > 0 {
                assert_eq!(four[2], level - 1, "{bits}: {level}");
            }
            if level < top {
                assert_eq!(four[3], level + 1, "{bits}: {level}");
            }
        }
    }
}

#[test]
#[expect(clippy::float_cmp, reason = "dyadic values compute exactly")]
fn the_proximal_map_is_optimal_and_the_projection_idempotent() {
    let levels = [10u16, 20, 30, 40];
    let plain = Model::new(&levels, (1, 1, 4), 8, &Data::default()).expect("a model");
    let values = [9.0, 20.25, 31.0, -4.0];
    let mut out = [0.0; 4];
    plain.prox(&values, 0.7, 0, &mut out);
    let top = 30.5_f64.next_down();
    assert_eq!(out, [9.5, 20.25, top, 39.5]);
    let projected = plain.project(&values);
    assert_eq!(projected, out);
    assert_eq!(plain.project(&projected), projected);
    assert_eq!(plain.excess(&values), [0.5, 0.0, 31.0 - top, 43.5]);
    // With mu tau = 1, r = 1/2 and the minimizer of (z - v)^2 / 2 + (z - c)^2 / 2 is
    // (v + c) / 2, exactly for dyadic values, then clipped: the first-order condition
    // holds where it is inside, and the gradient points outward at an end.
    let data = Data { mu: 2.0, slack: 0.0 };
    let weighted = Model::new(&levels, (1, 1, 4), 8, &data).expect("a model");
    weighted.prox(&values, 0.5, 0, &mut out);
    for (((&z, &v), &c), (&low, &high)) in out
        .iter()
        .zip(&values)
        .zip(weighted.centres())
        .zip(weighted.low().iter().zip(weighted.high()))
    {
        let gradient = (z - v) + (z - c);
        if z > low && z < high {
            assert_eq!(gradient, 0.0);
        } else if z == low {
            assert!(gradient >= 0.0);
        } else {
            assert!(gradient <= 0.0);
        }
    }
}

#[test]
fn the_conjugate_meets_fenchel_and_young() {
    let levels = [0u16, 7, 200, 255];
    for mu in [0.0, 0.5, 3.0] {
        let data = Data { mu, slack: 0.0 };
        let model = Model::new(&levels, (1, 1, 4), 8, &data).expect("a model");
        for s in [-3.0, -0.25, 0.0, 0.125, 2.0] {
            let xi = [s; 4];
            let conjugate = model.conjugate(&xi);
            // g*(s) >= s c - g(c) for every c of the intervals, with equality at some c.
            let mut best = f64::NEG_INFINITY;
            for step in 0..=64 {
                let fraction = f64::from(step) / 64.0;
                let c: Vec<f64> = model
                    .low()
                    .iter()
                    .zip(model.high())
                    .map(|(&low, &high)| low + fraction * (high - low))
                    .collect();
                let inner: f64 = c.iter().map(|&value| s * value).sum();
                best = best.max(inner - model.value(&c));
            }
            assert!(
                conjugate >= best - 1e-9 * (1.0 + best.abs()),
                "{mu} {s}: {conjugate} < {best}"
            );
            if mu == 0.0 {
                // The supremum of a linear function is at an end: the grid holds both.
                assert!(
                    (conjugate - best).abs() <= 1e-9 * (1.0 + best.abs()),
                    "{s}: {conjugate} {best}"
                );
            }
        }
    }
}

/// The root mean square of the differences.
fn rms(first: &[f64], second: &[f64]) -> f64 {
    let total: f64 = first.iter().zip(second).map(|(&a, &b)| (a - b) * (a - b)).sum();
    (total / f64::from(u32::try_from(first.len()).expect("small"))).sqrt()
}

#[test]
fn the_solvers_find_the_smooth_picture_within_the_intervals() {
    let (channels, height, width) = (3, 24, 40);
    let (values, levels) = smooth(channels, height, width);
    let model = Model::new(&levels, (channels, height, width), 8, &Data::default()).expect("a model");
    let rounded = rms(&values, model.centres());
    let options = Options {
        iterations: Some(4000),
        ..Options::default()
    };
    for method in [Method::Tv(&Tv::default()), Method::Tgv(&Tgv::default())] {
        let solution = samples::solve(&model, method, &options, None).expect("a solution");
        // Within the intervals: the result is its own projection.
        assert_eq!(model.project(&solution.picture), solution.picture);
        // Every record's dual value is at most its primal value, to rounding.
        let history = &solution.history;
        for (&primal, &dual) in history.primal.iter().zip(&history.dual) {
            assert!(dual <= primal * (1.0 + 1e-12) + 1e-9, "{dual} > {primal}");
        }
        assert!(solution.iterations > 0 && solution.stop != Stop::Observer);
        // TGV finds the ramps: its error is a fraction of the levels' (measured at the
        // defaults: 0.07 against 0.29, a regression check). TV's least value within the
        // intervals is terraced, and no nearer the picture than the levels.
        if matches!(method, Method::Tgv(_)) {
            let error = rms(&values, &solution.picture);
            assert!(error < 0.1 && error < rounded / 2.0, "{error} against {rounded}");
        }
    }
}

#[test]
fn the_solvers_take_the_defaults_of_rounded_samples() {
    let (channels, height, width) = (1, 12, 16);
    let (_, levels) = smooth(channels, height, width);
    let model = Model::new(&levels, (channels, height, width), 8, &Data::default()).expect("a model");
    // A first-order weight of 2, which the default tolerance and ratio follow.
    let tv = Tv {
        alpha: 2.0,
        ..Tv::default()
    };
    let tgv = Tgv {
        alpha1: 2.0,
        ..Tgv::default()
    };
    let solve = |method, options: &Options| samples::solve(&model, method, options, None).expect("a solution");
    for (method, ratio) in [
        (Method::Tv(&tv), samples::TV_RATIO),
        (Method::Tgv(&tgv), samples::TGV_RATIO),
    ] {
        // By default: TOLERANCE times the weight, exactly, ITERATIONS, and the
        // ratio over the weight squared.
        let found = solve(method, &Options::default());
        let given = solve(
            method,
            &Options {
                tolerance: Some(samples::TOLERANCE * 2.0),
                iterations: Some(samples::ITERATIONS),
                step_ratio: Some(ratio / (2.0 * 2.0)),
                ..Options::default()
            },
        );
        assert_eq!(found.stop, Stop::Converged);
        assert_eq!((found.iterations, &found.picture), (given.iterations, &given.picture));
        // TOLERANCE and the ratio alone where the defaults do not follow the weight.
        let found = solve(
            method,
            &Options {
                scale_with_weight: false,
                ..Options::default()
            },
        );
        let given = solve(
            method,
            &Options {
                tolerance: Some(samples::TOLERANCE),
                iterations: Some(samples::ITERATIONS),
                step_ratio: Some(ratio),
                scale_with_weight: false,
                ..Options::default()
            },
        );
        assert_eq!((found.iterations, &found.picture), (given.iterations, &given.picture));
    }
}
