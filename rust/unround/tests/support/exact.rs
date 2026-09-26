// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! Exact arithmetic for the tests: integers of any size, and the dyadic rationals
//! `m 2^e` made of them, which hold every double exactly. Sums, differences and
//! products of doubles are computed in them without rounding, so that what holds
//! exactly is checked exactly, and bounds of rounding error are compared with the
//! exact error.

use std::cmp::Ordering;

/// An integer of any size: a sign and a magnitude in limbs of 64 bits, the least
/// significant first, with no zero limb at the top (0 has none).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Big {
    negative: bool,
    limbs: Vec<u64>,
}

fn trim(limbs: &mut Vec<u64>) {
    while limbs.last() == Some(&0) {
        limbs.pop();
    }
}

fn compare_magnitudes(a: &[u64], b: &[u64]) -> Ordering {
    a.len().cmp(&b.len()).then_with(|| a.iter().rev().cmp(b.iter().rev()))
}

fn add_magnitudes(a: &[u64], b: &[u64]) -> Vec<u64> {
    let mut result = Vec::with_capacity(a.len().max(b.len()) + 1);
    let mut carry = 0u128;
    for index in 0..a.len().max(b.len()) {
        let sum =
            u128::from(a.get(index).copied().unwrap_or(0)) + u128::from(b.get(index).copied().unwrap_or(0)) + carry;
        result.push(low(sum));
        carry = sum >> 64;
    }
    if carry > 0 {
        result.push(low(carry));
    }
    trim(&mut result);
    result
}

/// `a - b` for magnitudes with `a >= b`.
fn subtract_magnitudes(a: &[u64], b: &[u64]) -> Vec<u64> {
    let mut result = Vec::with_capacity(a.len());
    let mut borrow = false;
    for (index, &limb) in a.iter().enumerate() {
        let other = b.get(index).copied().unwrap_or(0);
        let (first, over) = limb.overflowing_sub(other);
        let (second, under) = first.overflowing_sub(u64::from(borrow));
        result.push(second);
        borrow = over || under;
    }
    trim(&mut result);
    result
}

fn low(value: u128) -> u64 {
    u64::try_from(value & u128::from(u64::MAX)).expect("the low 64 bits fit")
}

impl Big {
    /// 0.
    #[must_use]
    pub fn zero() -> Self {
        Self {
            negative: false,
            limbs: Vec::new(),
        }
    }

    /// The integer `value`.
    #[must_use]
    pub fn from_i128(value: i128) -> Self {
        let magnitude = value.unsigned_abs();
        let mut limbs = vec![low(magnitude), low(magnitude >> 64)];
        trim(&mut limbs);
        Self {
            negative: value < 0 && !limbs.is_empty(),
            limbs,
        }
    }

    /// Whether it is 0.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.limbs.is_empty()
    }

    fn signed(negative: bool, limbs: Vec<u64>) -> Self {
        Self {
            negative: negative && !limbs.is_empty(),
            limbs,
        }
    }

    /// `self 2^bits`.
    #[must_use]
    pub fn shifted(&self, bits: u32) -> Self {
        if self.is_zero() {
            return Self::zero();
        }
        let whole = usize::try_from(bits / 64).expect("a shift fits");
        let part = bits % 64;
        let mut limbs = vec![0u64; whole];
        let mut carry = 0u64;
        for &limb in &self.limbs {
            if part == 0 {
                limbs.push(limb);
            } else {
                limbs.push((limb << part) | carry);
                carry = limb >> (64 - part);
            }
        }
        if carry > 0 {
            limbs.push(carry);
        }
        Self::signed(self.negative, limbs)
    }

    /// `self + other`.
    #[must_use]
    pub fn add(&self, other: &Self) -> Self {
        if self.negative == other.negative {
            return Self::signed(self.negative, add_magnitudes(&self.limbs, &other.limbs));
        }
        match compare_magnitudes(&self.limbs, &other.limbs) {
            Ordering::Equal => Self::zero(),
            Ordering::Greater => Self::signed(self.negative, subtract_magnitudes(&self.limbs, &other.limbs)),
            Ordering::Less => Self::signed(other.negative, subtract_magnitudes(&other.limbs, &self.limbs)),
        }
    }

    /// `-self`.
    #[must_use]
    pub fn negated(&self) -> Self {
        Self::signed(!self.negative, self.limbs.clone())
    }

    /// `self * other`.
    #[must_use]
    pub fn multiply(&self, other: &Self) -> Self {
        if self.is_zero() || other.is_zero() {
            return Self::zero();
        }
        let mut limbs = vec![0u64; self.limbs.len() + other.limbs.len()];
        for (i, &a) in self.limbs.iter().enumerate() {
            let mut carry = 0u128;
            for (j, &b) in other.limbs.iter().enumerate() {
                let product = u128::from(a) * u128::from(b) + u128::from(limbs[i + j]) + carry;
                limbs[i + j] = low(product);
                carry = product >> 64;
            }
            let mut index = i + other.limbs.len();
            while carry > 0 {
                let sum = u128::from(limbs[index]) + carry;
                limbs[index] = low(sum);
                carry = sum >> 64;
                index += 1;
            }
        }
        trim(&mut limbs);
        Self::signed(self.negative != other.negative, limbs)
    }

    /// The sign: -1, 0 or 1.
    #[must_use]
    pub fn signum(&self) -> i32 {
        if self.is_zero() {
            0
        } else if self.negative {
            -1
        } else {
            1
        }
    }
}

/// A dyadic rational `mantissa 2^exponent`, exact.
#[derive(Debug, Clone)]
pub struct Dyadic {
    mantissa: Big,
    exponent: i32,
}

const MANTISSA_BITS: u32 = 52;

impl Dyadic {
    /// 0.
    #[must_use]
    pub fn zero() -> Self {
        Self {
            mantissa: Big::zero(),
            exponent: 0,
        }
    }

    /// A double, exactly.
    ///
    /// # Panics
    ///
    /// For a double that is not finite.
    #[must_use]
    pub fn from_f64(value: f64) -> Self {
        assert!(value.is_finite(), "a finite double, not {value}");
        let bits = value.to_bits();
        let negative = bits >> 63 == 1;
        let biased = i32::try_from((bits >> MANTISSA_BITS) & 0x7ff).expect("an exponent fits");
        let fraction = bits & ((1u64 << MANTISSA_BITS) - 1);
        let (mantissa, exponent) = if biased == 0 {
            (fraction, -1074)
        } else {
            (fraction | (1u64 << MANTISSA_BITS), biased - 1075)
        };
        let magnitude = Big::from_i128(i128::from(mantissa));
        Self {
            mantissa: if negative { magnitude.negated() } else { magnitude },
            exponent,
        }
    }

    /// An integer, exactly.
    #[must_use]
    pub fn from_i128(value: i128) -> Self {
        Self {
            mantissa: Big::from_i128(value),
            exponent: 0,
        }
    }

    /// Both mantissas at the smaller of the two exponents.
    fn aligned(&self, other: &Self) -> (Big, Big, i32) {
        let exponent = self.exponent.min(other.exponent);
        let shift = |value: &Self| {
            value
                .mantissa
                .shifted(u32::try_from(value.exponent - exponent).expect("a shift fits"))
        };
        (shift(self), shift(other), exponent)
    }

    /// `self + other`.
    #[must_use]
    pub fn add(&self, other: &Self) -> Self {
        let (a, b, exponent) = self.aligned(other);
        Self {
            mantissa: a.add(&b),
            exponent,
        }
    }

    /// `self - other`.
    #[must_use]
    pub fn sub(&self, other: &Self) -> Self {
        self.add(&other.negated())
    }

    /// `-self`.
    #[must_use]
    pub fn negated(&self) -> Self {
        Self {
            mantissa: self.mantissa.negated(),
            exponent: self.exponent,
        }
    }

    /// `|self|`.
    #[must_use]
    pub fn abs(&self) -> Self {
        if self.mantissa.signum() < 0 {
            self.negated()
        } else {
            self.clone()
        }
    }

    /// `self * other`.
    #[must_use]
    pub fn mul(&self, other: &Self) -> Self {
        Self {
            mantissa: self.mantissa.multiply(&other.mantissa),
            exponent: self.exponent + other.exponent,
        }
    }

    /// `self 2^power`, exactly.
    #[must_use]
    pub fn scaled(&self, power: i32) -> Self {
        Self {
            mantissa: self.mantissa.clone(),
            exponent: self.exponent + power,
        }
    }

    /// How it compares with `other`.
    #[must_use]
    pub fn compare(&self, other: &Self) -> Ordering {
        self.sub(other).mantissa.signum().cmp(&0)
    }

    /// Whether it is at most `other`.
    #[must_use]
    pub fn at_most(&self, other: &Self) -> bool {
        self.compare(other) != Ordering::Greater
    }

    /// Whether it equals `other`.
    #[must_use]
    pub fn equals(&self, other: &Self) -> bool {
        self.compare(other) == Ordering::Equal
    }
}

/// The exact sum of doubles.
#[must_use]
pub fn sum(values: impl IntoIterator<Item = f64>) -> Dyadic {
    values
        .into_iter()
        .fold(Dyadic::zero(), |total, value| total.add(&Dyadic::from_f64(value)))
}

/// The exact inner product of two arrays of doubles.
#[must_use]
pub fn inner(a: &[f64], b: &[f64]) -> Dyadic {
    a.iter().zip(b).fold(Dyadic::zero(), |total, (&x, &y)| {
        total.add(&Dyadic::from_f64(x).mul(&Dyadic::from_f64(y)))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dyadic_arithmetic_is_exact() {
        // 0.1 + 0.2 in binary64 rounds; exactly, the doubles add to more than 0.3's double.
        let sum = Dyadic::from_f64(0.1).add(&Dyadic::from_f64(0.2));
        assert!(!sum.equals(&Dyadic::from_f64(0.3)));
        assert!(sum.equals(&Dyadic::from_f64(0.1).add(&Dyadic::from_f64(0.2))));
        let product = Dyadic::from_f64(1.5).mul(&Dyadic::from_f64(-2.25));
        assert!(product.equals(&Dyadic::from_f64(-3.375)));
        let big = Dyadic::from_f64(1e300).mul(&Dyadic::from_f64(1e300));
        assert!(Dyadic::from_f64(f64::MAX).at_most(&big));
        let tiny = Dyadic::from_f64(f64::from_bits(1));
        assert!(Dyadic::zero().at_most(&tiny));
        assert!(tiny.scaled(-1).add(&tiny.scaled(-1)).equals(&tiny));
    }
}

/// An exact value, as the pair of doubles `hi:lo` that holds it: `hi` the double nearest to
/// it and `lo` the double nearest to what remains, so that `hi + lo` is within about
/// 2^-106 of it (conformance/references.py and conformance/generate.py write them).
#[derive(Debug, Clone, Copy)]
pub struct Exact {
    /// The double nearest to the value.
    pub high: f64,
    /// The double nearest to what remains.
    pub low: f64,
}

impl Exact {
    /// The pair written `hi:lo`.
    #[must_use]
    pub fn parse(text: &str) -> Self {
        let (high, low) = text.split_once(':').expect("a pair hi:lo");
        Self {
            high: high.parse().expect("a double"),
            low: low.parse().expect("a double"),
        }
    }

    /// Whether `value` is within `bound` of it, in exact arithmetic, with 2^-100 of it for
    /// the pair's own rounding.
    #[must_use]
    pub fn within(self, value: f64, bound: &Dyadic) -> bool {
        let exact = Dyadic::from_f64(self.high).add(&Dyadic::from_f64(self.low));
        let slack = Dyadic::from_f64(self.high).abs().scaled(-100);
        Dyadic::from_f64(value).sub(&exact).abs().at_most(&bound.add(&slack))
    }
}
