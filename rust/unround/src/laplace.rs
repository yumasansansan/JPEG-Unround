// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! A Laplace model of a component's AC coefficients (docs/math.md, 2).
//!
//! For each frequency, the coefficients of all the component's blocks are taken as
//! draws from one Laplace distribution, whose scale is estimated from their bins by
//! maximum likelihood, in closed form. The MMSE centre of a coefficient is then its
//! conditional mean within its bin: the centre of its interval, moved towards 0.
//! Both are computed without cancellation: the scale is within a few units in the
//! last place of the maximum-likelihood estimate, and the shrinkage within a few
//! units of 2^-53 of its exact value.

use std::sync::OnceLock;

use crate::dct::{BLOCK_SIZE, multiply_add};
use crate::exact::{self, Ratio};

/// Below this `rho`, the shrinkage is its Taylor series.
const SERIES_BELOW: f64 = 1.0;

/// From this `t` on, `log(1/t)` is taken from `1 - t`.
const FROM_COMPLEMENT: f64 = 0.5;

/// The terms of the series: below 1, eleven leave less than 2e-19.
const SERIES_TERMS: usize = 11;

/// `B_2k / (2k)!` for `k = 1, ..., 11`, each computed exactly in rationals and
/// rounded once to the nearest double.
fn series() -> &'static [f64; SERIES_TERMS] {
    static SERIES: OnceLock<[f64; SERIES_TERMS]> = OnceLock::new();
    SERIES.get_or_init(|| {
        let numbers = exact::bernoulli(2 * SERIES_TERMS + 1);
        let mut coefficients = [0.0; SERIES_TERMS];
        for (index, coefficient) in coefficients.iter_mut().enumerate() {
            let order = 2 * (index + 1);
            let factorial = exact::factorial(u32::try_from(order).expect("the order of a term fits in u32"));
            *coefficient = (numbers[order] / Ratio::integer(factorial)).nearest();
        }
        coefficients
    })
}

/// The maximum-likelihood scale of one frequency, from the counts of its levels
/// (docs/math.md, 2.1).
///
/// `zeros` and `nonzeros` count the levels that are 0 and that are not, `odd_sum` is
/// the sum of `2|q| - 1` over the latter, and `step` is the quantization step. The
/// counts, and `A` and `D` below, are integers and are computed exactly; each
/// enters floating point once, rounded to the nearest double. The scale is 0 when
/// every level is 0.
#[must_use]
pub fn scale_from_counts(zeros: u64, nonzeros: u64, odd_sum: u64, step: u16) -> f64 {
    if odd_sum == 0 {
        return 0.0;
    }
    let (n0, n1, s) = (i128::from(zeros), i128::from(nonzeros), i128::from(odd_sum));
    let quadratic = n0 + s + 2 * n1; // A
    let discriminant = n0 * n0 + 4 * quadratic * s; // D, up to about 2^80 for a file
    let root = exact::nearest_from_i128(discriminant).sqrt();
    let zeros = exact::nearest_from_i128(n0);
    // t = exp(-Q / (2 scale)) is the root in (0, 1) of A t^2 + n0 t - S. log(1/t) is taken
    // from t itself below 1/2, and above it from 1 - t, which has a form without
    // cancellation: 8 S (n0 + n1) / ((sqrt(D) + 2S - n0) (n0 + sqrt(D))).
    let t = exact::nearest_from_i128(2 * s) / (zeros + root);
    let log_inverse = if t < FROM_COMPLEMENT {
        -t.ln()
    } else {
        let numerator = exact::nearest_from_i128(8 * s * (n0 + n1));
        let complement = numerator / ((root + exact::nearest_from_i128(2 * s - n0)) * (zeros + root));
        -(-complement).ln_1p()
    };
    f64::from(step) / (2.0 * log_inverse)
}

/// The maximum-likelihood Laplace scale of each frequency, in coefficient units.
///
/// `levels` are the quantized levels of the component's blocks, 64 to a block in
/// natural order, and `table` the steps. The scale of a frequency whose levels are
/// all 0 is 0. The DC coefficient follows no Laplace distribution: its scale is
/// infinite, which makes its MMSE centre the centre of its interval.
#[must_use]
pub fn scales(levels: &[i16], table: &[u16; BLOCK_SIZE]) -> [f64; BLOCK_SIZE] {
    let mut zeros = [0u64; BLOCK_SIZE];
    let mut sums = [0u64; BLOCK_SIZE]; // at most 2^15 for each of fewer than 2^48 blocks
    let mut blocks: u64 = 0;
    for block in levels.as_chunks::<BLOCK_SIZE>().0 {
        blocks += 1;
        for ((zero, sum), &level) in zeros.iter_mut().zip(&mut sums).zip(block) {
            let magnitude = u64::from(level.unsigned_abs());
            *zero += u64::from(magnitude == 0);
            *sum += magnitude;
        }
    }
    let mut result = [0.0; BLOCK_SIZE];
    for (((scale, &zero), &sum), &step) in result.iter_mut().zip(&zeros).zip(&sums).zip(table) {
        let nonzeros = blocks - zero;
        *scale = scale_from_counts(zero, nonzeros, 2 * sum - nonzeros, step);
    }
    result[0] = f64::INFINITY;
    result
}

/// `delta / Q = 1/2 - 1/rho + 1/(e^rho - 1)`: how far towards 0 an MMSE centre lies,
/// in steps.
///
/// `rho` is the step over the scale, from 0 (an infinite scale: `delta` is 0) to
/// infinity (a scale of 0: `delta` is 1/2). Below 1 it is the Taylor series, whose
/// terms the closed form would cancel; from 1 on, the closed form, whose terms are
/// then at most 1.
#[must_use]
pub fn shrinkage(rho: f64) -> f64 {
    if rho < SERIES_BELOW {
        let coefficients = series();
        let squared = rho * rho;
        let (rest, last) = (&coefficients[..SERIES_TERMS - 1], coefficients[SERIES_TERMS - 1]);
        let sum = rest
            .iter()
            .rev()
            .fold(last, |sum, &coefficient| multiply_add(squared, sum, coefficient));
        rho * sum
    } else {
        // 1 / (e^rho - 1) as e^-rho / (1 - e^-rho), which neither overflows nor divides by 0.
        0.5 - 1.0 / rho + (-rho).exp() / -(-rho).exp_m1()
    }
}

/// The MMSE centre of every level, in coefficient units, 64 to a block.
///
/// The centre of the level `q` with the step `Q` is `sign(q) (|q| - delta / Q) Q`,
/// where `delta` comes from the scale of its frequency. The level 0 has the centre
/// 0, and the DC coefficient the centre of its interval, `q Q`. These are the
/// coefficients of the level-shifted canvas; the model adds the level shift to DC.
/// Each centre lies within its interval in floating point too: `|q| - delta / Q`
/// rounds to within `[|q| - 1/2, |q|]`, and so does its product with `Q` to within
/// the interval.
#[must_use]
pub fn centres(levels: &[i16], table: &[u16; BLOCK_SIZE], scale: &[f64; BLOCK_SIZE]) -> Vec<f64> {
    let mut delta = [0.0; BLOCK_SIZE];
    for ((shrunk, &step), &scale) in delta.iter_mut().zip(table).zip(scale) {
        let rho = if scale > 0.0 {
            f64::from(step) / scale
        } else {
            f64::INFINITY
        };
        *shrunk = shrinkage(rho);
    }
    let mut result = vec![0.0; levels.len()];
    for (centres, block) in result
        .as_chunks_mut::<BLOCK_SIZE>()
        .0
        .iter_mut()
        .zip(levels.as_chunks::<BLOCK_SIZE>().0)
    {
        for (((centre, &level), &shrunk), &step) in centres.iter_mut().zip(block).zip(&delta).zip(table) {
            if level != 0 {
                let magnitude = (f64::from(level.unsigned_abs()) - shrunk) * f64::from(step);
                *centre = if level < 0 { -magnitude } else { magnitude };
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[expect(clippy::float_cmp, reason = "exact values are compared to the last bit")]
    fn the_ends_of_the_shrinkage() {
        assert_eq!(shrinkage(0.0), 0.0);
        assert_eq!(shrinkage(f64::INFINITY), 0.5);
        // The first coefficient is B_2 / 2! = 1/12, rounded once.
        assert_eq!(series()[0], 1.0 / 12.0);
        assert_eq!(series()[1], -1.0 / 720.0);
    }

    #[test]
    #[expect(clippy::float_cmp, reason = "exact values are compared to the last bit")]
    fn all_zeros_have_the_scale_zero_and_dc_an_infinite_one() {
        let mut levels = vec![0i16; 6 * BLOCK_SIZE];
        for block in levels.as_chunks_mut::<BLOCK_SIZE>().0 {
            block[0] = 5;
        }
        let result = scales(&levels, &[10; BLOCK_SIZE]);
        assert_eq!(result[0], f64::INFINITY);
        assert!(result[1..].iter().all(|&scale| scale == 0.0));
    }

    #[test]
    #[expect(clippy::float_cmp, reason = "exact values are compared to the last bit")]
    fn the_centres_of_zero_and_of_dc() {
        let mut levels = vec![0i16; 2 * BLOCK_SIZE];
        levels[0] = -4;
        levels[BLOCK_SIZE] = 9;
        levels[BLOCK_SIZE + 3 * 8 + 3] = 2;
        let table = [5; BLOCK_SIZE];
        let centres = centres(&levels, &table, &scales(&levels, &table));
        assert_eq!((centres[0], centres[BLOCK_SIZE]), (-20.0, 45.0));
        assert_eq!(centres[3 * 8 + 3], 0.0);
        // A frequency with a single non-zero level among two: its centre lies towards 0.
        let lone = centres[BLOCK_SIZE + 3 * 8 + 3];
        assert!((7.5..10.0).contains(&lone), "{lone}");
    }
}
