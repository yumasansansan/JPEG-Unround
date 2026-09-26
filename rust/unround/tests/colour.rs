// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! The conversion of JFIF (docs/math.md, 8): its round trip, within a running bound
//! of its rounding.
//!
//! The exact conversions are inverses of one another (the Python tests checked the
//! rationals exactly), so what the computed ones leave of their input is within the
//! rounding they add up: each operation adds `U / (1 - U)` times its computed result,
//! a product carries its factors' bounds, and a constant brings its own distance from
//! its rational, at most `U / (1 - U)` of it. The operations are the crate's own, fused
//! where it fuses them, so the tracked values are the computed ones to the last bit.

mod support;

use jpeg_unround::colour::{self, BLUE_FROM_CB, CB_FROM_BLUE, CR_FROM_RED, GREEN_FROM_CB, GREEN_FROM_CR, RED_FROM_CR};
use jpeg_unround::colour::{Y_FROM_BLUE, Y_FROM_GREEN, Y_FROM_RED};
use jpeg_unround::dct::multiply_add;
use support::exact::Dyadic;
use support::rounding::U;
use support::synthetic::Numbers;

/// A computed value, and a bound of its distance from the exact value.
#[derive(Debug, Clone)]
struct Tracked {
    value: f64,
    bound: Dyadic,
}

/// `U / (1 - U)`, bounded above by the dyadic `U (1 + 2U)`.
fn unit() -> Dyadic {
    let u = Dyadic::from_f64(U);
    u.mul(&Dyadic::from_i128(1).add(&u.scaled(1)))
}

impl Tracked {
    fn exact(value: f64) -> Self {
        Self {
            value,
            bound: Dyadic::zero(),
        }
    }

    /// A constant, the double nearest to its rational: within `U / (1 - U)` of it.
    fn constant(value: f64) -> Self {
        Self {
            value,
            bound: unit().mul(&Dyadic::from_f64(value).abs()),
        }
    }

    fn rounded(value: f64, carried: &Dyadic) -> Self {
        Self {
            value,
            bound: carried.add(&unit().mul(&Dyadic::from_f64(value).abs())),
        }
    }

    fn sub(&self, other: &Self) -> Self {
        Self::rounded(self.value - other.value, &self.bound.add(&other.bound))
    }

    fn product_carried(&self, other: &Self) -> Dyadic {
        let (a, b) = (Dyadic::from_f64(self.value).abs(), Dyadic::from_f64(other.value).abs());
        a.mul(&other.bound)
            .add(&b.mul(&self.bound))
            .add(&self.bound.mul(&other.bound))
    }

    fn mul(&self, other: &Self) -> Self {
        Self::rounded(self.value * other.value, &self.product_carried(other))
    }

    /// `self * other + addend` as the crate computes it: fused where it fuses, one
    /// rounding of the whole, and two where it does not.
    fn multiply_add(&self, other: &Self, addend: &Self) -> Self {
        let value = multiply_add(self.value, other.value, addend.value);
        let fused = cfg!(any(target_arch = "aarch64", target_feature = "fma"));
        let mut carried = self.product_carried(other).add(&addend.bound);
        if !fused {
            carried = carried.add(&unit().mul(&Dyadic::from_f64(self.value * other.value).abs()));
        }
        Self::rounded(value, &carried)
    }

    fn negated(&self) -> Self {
        Self {
            value: -self.value,
            bound: self.bound.clone(),
        }
    }
}

fn to_rgb(luma: &Tracked, cb: &Tracked, cr: &Tracked) -> [Tracked; 3] {
    let centre = Tracked::exact(128.0);
    let (cb, cr) = (cb.sub(&centre), cr.sub(&centre));
    let red = Tracked::constant(RED_FROM_CR).multiply_add(&cr, luma);
    let blue = Tracked::constant(BLUE_FROM_CB).multiply_add(&cb, luma);
    let partial = Tracked::constant(GREEN_FROM_CB).negated().multiply_add(&cb, luma);
    let green = Tracked::constant(GREEN_FROM_CR).negated().multiply_add(&cr, &partial);
    [red, green, blue]
}

fn to_ycbcr(red: &Tracked, green: &Tracked, blue: &Tracked) -> [Tracked; 3] {
    let centre = Tracked::exact(128.0);
    let partial = Tracked::constant(Y_FROM_RED).multiply_add(red, &Tracked::constant(Y_FROM_GREEN).mul(green));
    let luma = Tracked::constant(Y_FROM_BLUE).multiply_add(blue, &partial);
    let cb = Tracked::constant(CB_FROM_BLUE).multiply_add(&blue.sub(&luma), &centre);
    let cr = Tracked::constant(CR_FROM_RED).multiply_add(&red.sub(&luma), &centre);
    [luma, cb, cr]
}

fn within(tracked: &Tracked, computed: f64, original: f64) {
    assert_eq!(computed.to_bits(), tracked.value.to_bits());
    let off = Dyadic::from_f64(computed).sub(&Dyadic::from_f64(original)).abs();
    assert!(off.at_most(&tracked.bound));
    // And the bound is of the order of the rounding: a few hundred units of 2^-53 of 256.
    let size = Dyadic::from_i128(1024 * 256).mul(&Dyadic::from_f64(U));
    assert!(tracked.bound.at_most(&size));
}

#[test]
fn the_round_trip_is_within_its_rounding() {
    let mut numbers = Numbers::new(62);
    let planes: Vec<f64> = (0..3 * 40).map(|_| numbers.uniform(-20.0, 280.0)).collect();
    let rgb = colour::to_rgb(&planes, 1, 40).expect("planes");
    let back = colour::to_ycbcr(&rgb, 1, 40).expect("a picture");
    for sample in 0..40 {
        let [luma, cb, cr] = [0, 1, 2].map(|channel| Tracked::exact(planes[channel * 40 + sample]));
        let [red, green, blue] = to_rgb(&luma, &cb, &cr);
        for (channel, tracked) in [&red, &green, &blue].into_iter().enumerate() {
            assert_eq!(rgb[3 * sample + channel].to_bits(), tracked.value.to_bits());
        }
        let tracked = to_ycbcr(&red, &green, &blue);
        for (channel, tracked) in tracked.iter().enumerate() {
            within(tracked, back[channel * 40 + sample], planes[channel * 40 + sample]);
        }
    }
    let picture: Vec<f64> = (0..3 * 40).map(|_| numbers.uniform(-20.0, 280.0)).collect();
    let again = colour::to_rgb(&colour::to_ycbcr(&picture, 1, 40).expect("a picture"), 1, 40).expect("planes");
    for sample in 0..40 {
        let [red, green, blue] = [0, 1, 2].map(|channel| Tracked::exact(picture[3 * sample + channel]));
        let [luma, cb, cr] = to_ycbcr(&red, &green, &blue);
        let tracked = to_rgb(&luma, &cb, &cr);
        for (channel, tracked) in tracked.iter().enumerate() {
            within(tracked, again[3 * sample + channel], picture[3 * sample + channel]);
        }
    }
}

#[test]
fn the_shapes_are_checked() {
    let error = colour::to_rgb(&[0.0; 10], 1, 4).expect_err("refused");
    assert!(error.to_string().contains("planes"), "{error}");
    let error = colour::to_ycbcr(&[0.0; 10], 1, 4).expect_err("refused");
    assert!(error.to_string().contains("RGB picture"), "{error}");
}
