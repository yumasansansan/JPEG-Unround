// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! The DCT and the Laplace model against exact values (docs/math.md, 1.1 and 2).
//!
//! `conformance/references.py` computes the values from their definitions in 60- and
//! 70-digit decimals, and writes each as a pair of doubles `hi:lo`, `hi + lo` within
//! about 2^-106 of it. The comparisons here are made in exact arithmetic, with
//! 2^-100 of the value added for the pair's own rounding, and the tolerances are the
//! bounds of rounding error of the implementation.

mod support;

use jpeg_unround::dct::{self, BASIS, BLOCK, BLOCK_SIZE, Block};
use jpeg_unround::laplace;
use support::exact::Dyadic;
use support::rounding::{self, U};

const BASIS_FILE: &str = include_str!("../../../conformance/references/basis.txt");
const DCT_FILE: &str = include_str!("../../../conformance/references/dct.txt");
const SCALE_FILE: &str = include_str!("../../../conformance/references/scale.txt");
const SHRINKAGE_FILE: &str = include_str!("../../../conformance/references/shrinkage.txt");
const CENTRES_FILE: &str = include_str!("../../../conformance/references/centres.txt");

/// An exact value, as the pair of doubles that holds it.
#[derive(Debug, Clone, Copy)]
struct Exact {
    high: f64,
    low: f64,
}

impl Exact {
    fn parse(text: &str) -> Self {
        let (high, low) = text.split_once(':').expect("a pair hi:lo");
        Self {
            high: high.parse().expect("a double"),
            low: low.parse().expect("a double"),
        }
    }

    /// Whether `value` is within `bound` of it, in exact arithmetic, with 2^-100 of it for
    /// the pair's own rounding.
    fn within(self, value: f64, bound: &Dyadic) -> bool {
        let exact = Dyadic::from_f64(self.high).add(&Dyadic::from_f64(self.low));
        let slack = Dyadic::from_f64(self.high).abs().scaled(-100);
        Dyadic::from_f64(value).sub(&exact).abs().at_most(&bound.add(&slack))
    }
}

fn lines(file: &str) -> impl Iterator<Item = Vec<&str>> {
    file.lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .map(|line| line.split_whitespace().collect())
}

#[test]
fn every_entry_of_the_basis_is_the_double_nearest_to_it() {
    let mut count = 0;
    for fields in lines(BASIS_FILE) {
        let (k, n): (usize, usize) = (fields[0].parse().expect("k"), fields[1].parse().expect("n"));
        let exact = Exact::parse(fields[2]);
        // hi is the double nearest to the entry, and lo is within half a unit of hi's last place.
        assert_eq!(BASIS[k][n].to_bits(), exact.high.to_bits(), "{k}, {n}");
        count += 1;
    }
    assert_eq!(count, 64);
}

fn block_of(fields: &[&str]) -> Block {
    let values: Vec<f64> = fields
        .iter()
        .map(|field| f64::from(field.parse::<i32>().expect("an integer")))
        .collect();
    dct::gather(&values)
}

#[test]
fn the_dct_is_that_of_the_standard_within_its_rounding() {
    let rows: Vec<Vec<&str>> = lines(DCT_FILE).collect();
    let mut checked = 0;
    for pair in rows.as_chunks::<2>().0 {
        let (kind, exact) = (&pair[0], &pair[1]);
        assert_eq!(exact[0], "exact");
        let block = block_of(&kind[1..]);
        let (computed, bound) = match kind[0] {
            "forward" => (dct::forward_block(&block), rounding::forward_error(&block)),
            "inverse" => (dct::inverse_block(&block), rounding::inverse_error(&block)),
            other => panic!("a line of {other}"),
        };
        for (index, field) in exact[1..].iter().enumerate() {
            let (row, column) = (index / BLOCK, index % BLOCK);
            let reference = Exact::parse(field);
            let tolerance = Dyadic::from_f64(bound[row][column]);
            assert!(
                reference.within(computed[row][column], &tolerance),
                "{} {index}",
                kind[0]
            );
        }
        checked += 1;
    }
    assert_eq!(checked, 8);
}

#[test]
fn the_scale_is_within_sixteen_units_of_the_maximum_of_the_likelihood() {
    // About ten roundings, none magnified (docs/math.md, 2.1): sixteen units of 2^-53 are taken.
    let mut count = 0;
    for fields in lines(SCALE_FILE) {
        let parse = |index: usize| fields[index].parse::<u64>().expect("a count");
        let step = u16::try_from(parse(3)).expect("a step");
        let exact = Exact::parse(fields[4]);
        let scale = laplace::scale_from_counts(parse(0), parse(1), parse(2), step);
        let bound = Dyadic::from_i128(16)
            .mul(&Dyadic::from_f64(U))
            .mul(&Dyadic::from_f64(exact.high));
        assert!(exact.within(scale, &bound), "{fields:?}: {scale}");
        count += 1;
    }
    assert_eq!(count, 10);
}

#[test]
fn the_shrinkage_is_within_sixteen_units_of_its_exact_value() {
    // The series below 1: a Horner sum of eleven terms, whose magnitudes sum to at most 0.09,
    // within gamma(22) of that. The closed form from 1 on: terms of at most 1, with exp and
    // expm1 within a few units in the last place. Within 16 units of 2^-53 everywhere.
    let bound = Dyadic::from_i128(16).mul(&Dyadic::from_f64(U));
    let mut previous = 0.0;
    let mut count = 0;
    for fields in lines(SHRINKAGE_FILE) {
        let rho: f64 = fields[0].parse().expect("a double");
        let value = laplace::shrinkage(rho);
        assert!(Exact::parse(fields[1]).within(value, &bound), "{rho}: {value}");
        if count < 3001 {
            assert!(value >= previous, "{rho}");
            previous = value;
        }
        count += 1;
    }
    assert_eq!(count, 3006);
}

#[test]
fn the_centres_are_the_means_of_their_bins() {
    // The centre is within (2|q| + 17) U Q of the mean of its bin: the shrinkage within 16U,
    // then a subtraction from |q| and a product with Q, each rounding within U of its result.
    let mut count = 0;
    for fields in lines(CENTRES_FILE) {
        let level: i16 = fields[0].parse().expect("a level");
        let step: u16 = fields[1].parse().expect("a step");
        let scale: f64 = fields[2].parse().expect("a scale");
        let mut levels = vec![0i16; BLOCK_SIZE];
        levels[1] = level;
        let centre = laplace::centres(&levels, &[step; BLOCK_SIZE], &[scale; BLOCK_SIZE])[1];
        let magnitude = i128::from(level.unsigned_abs());
        let bound = Dyadic::from_i128(2 * magnitude + 17)
            .mul(&Dyadic::from_f64(U))
            .mul(&Dyadic::from_i128(i128::from(step)));
        assert!(Exact::parse(fields[3]).within(centre, &bound), "{fields:?}: {centre}");
        let (size, reach) = (centre.abs(), f64::from(step));
        let whole = f64::from(level.unsigned_abs());
        assert!((whole - 0.5) * reach <= size && size <= whole * reach);
        assert_eq!(centre < 0.0, level < 0);
        count += 1;
    }
    assert_eq!(count, 35);
}
