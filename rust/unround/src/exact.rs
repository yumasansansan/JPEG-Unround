// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! What is rational, computed exactly, and how it enters floating point: once,
//! rounded to the nearest double (docs/math.md, Arithmetic). And the sums of
//! floating-point terms, in an order of their own.

use std::cmp::Ordering;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// The double nearest to an integer, ties to even.
#[must_use]
#[expect(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    reason = "an integer converts to the nearest double, ties to even, which is the rounding asked for"
)]
pub fn nearest_from_i128(value: i128) -> f64 {
    value as f64
}

/// The double nearest to an integer, ties to even.
#[must_use]
pub fn nearest_from_u64(value: u64) -> f64 {
    nearest_from_i128(i128::from(value))
}

/// The double nearest to a count, ties to even; exact below 2^53.
#[must_use]
pub fn nearest_from_usize(value: usize) -> f64 {
    // usize has 64 bits on the systems JPEG-Unround supports, and so fits in u64.
    nearest_from_u64(u64::try_from(value).unwrap_or(u64::MAX))
}

const MANTISSA_BITS: u32 = 52;
const EXPONENT_BIAS: i32 = 1023;

/// The double nearest to `numerator / denominator`, ties to even, for quotients
/// whose magnitude is 0 or a normal double.
///
/// The quotient's bits are those of long division, one at a time, so that nothing
/// rounds before the last step: the leading bit, the 52 that follow, the next as
/// the rounding bit, and whether any remainder is left.
///
/// # Panics
///
/// When the denominator is 0, or the quotient is not 0 and lies outside the range
/// of normal doubles.
#[must_use]
pub fn quotient(numerator: i128, denominator: i128) -> f64 {
    assert!(denominator != 0, "a quotient by 0");
    if numerator == 0 {
        return 0.0;
    }
    let negative = (numerator < 0) != (denominator < 0);
    let dividend = numerator.unsigned_abs();
    let divisor = denominator.unsigned_abs();
    // The integer part, then the bits after the point: remainder < divisor <= 2^127, and
    // doubling it stays below 2^128.
    let whole = dividend / divisor;
    let mut remainder = dividend % divisor;
    let mut bits: u128 = 0; // the significant bits so far, the leading 1 first
    let mut count: u32 = 0; // how many significant bits there are
    let mut exponent: i32; // the power of 2 of the leading bit
    if whole > 0 {
        let length = 128 - whole.leading_zeros();
        exponent = i32::try_from(length - 1).unwrap_or(i32::MAX);
        if length > MANTISSA_BITS + 2 {
            // More bits than a double holds, and a rounding bit: those below are sticky.
            let dropped = length - (MANTISSA_BITS + 2);
            bits = whole >> dropped;
            count = MANTISSA_BITS + 2;
            if whole & ((1u128 << dropped) - 1) != 0 {
                remainder |= 1; // anything left makes the result inexact
            }
        } else {
            bits = whole;
            count = length;
        }
    } else {
        exponent = 0;
    }
    while count < MANTISSA_BITS + 2 {
        remainder <<= 1;
        let bit = remainder >= divisor;
        if bit {
            remainder -= divisor;
        }
        if count == 0 {
            exponent -= 1;
            if bit {
                bits = 1;
                count = 1;
            }
        } else {
            bits = (bits << 1) | u128::from(bit);
            count += 1;
        }
    }
    // bits has 54 significant bits: the 53 of the double and the rounding bit.
    let sticky = remainder != 0;
    let rounding = bits & 1 == 1;
    let mut mantissa = bits >> 1;
    if rounding && (sticky || mantissa & 1 == 1) {
        mantissa += 1;
        if mantissa >> (MANTISSA_BITS + 1) != 0 {
            mantissa >>= 1;
            exponent += 1;
        }
    }
    let biased = exponent + EXPONENT_BIAS;
    assert!(
        (1..2 * EXPONENT_BIAS + 1).contains(&biased),
        "the quotient {numerator} / {denominator} is not a normal double"
    );
    let fraction = u64::try_from(mantissa & ((1u128 << MANTISSA_BITS) - 1)).unwrap_or(0);
    let biased = u64::try_from(biased).unwrap_or(0);
    let magnitude = f64::from_bits((biased << MANTISSA_BITS) | fraction);
    if negative { -magnitude } else { magnitude }
}

fn gcd(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// A rational number in lowest terms, with a positive denominator: exact
/// arithmetic for the few rationals that need it (the Bernoulli numbers of the
/// series of docs/math.md, 2.2). Its operations panic where a result does not fit
/// in 128 bits, rather than round.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Ratio {
    numerator: i128,
    denominator: i128,
}

impl Ratio {
    /// The rational `numerator / denominator`, in lowest terms.
    ///
    /// # Panics
    ///
    /// When the denominator is 0.
    #[must_use]
    pub fn new(numerator: i128, denominator: i128) -> Self {
        assert!(denominator != 0, "a rational with the denominator 0");
        let divisor = gcd(numerator.unsigned_abs(), denominator.unsigned_abs());
        let divisor = i128::try_from(divisor).expect("a divisor of an i128 fits in one");
        let sign = if denominator < 0 { -1 } else { 1 };
        Self {
            numerator: sign * (numerator / divisor),
            denominator: sign * (denominator / divisor),
        }
    }

    /// The integer `value`.
    #[must_use]
    pub fn integer(value: i128) -> Self {
        Self {
            numerator: value,
            denominator: 1,
        }
    }

    /// The numerator, in lowest terms.
    #[must_use]
    pub fn numerator(self) -> i128 {
        self.numerator
    }

    /// The denominator, positive, in lowest terms.
    #[must_use]
    pub fn denominator(self) -> i128 {
        self.denominator
    }

    /// The double nearest to it, ties to even ([`quotient`]).
    #[must_use]
    pub fn nearest(self) -> f64 {
        quotient(self.numerator, self.denominator)
    }
}

fn checked(value: Option<i128>) -> i128 {
    value.expect("a rational of JPEG-Unround fits in 128 bits")
}

impl Add for Ratio {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let divisor = i128::try_from(gcd(self.denominator.unsigned_abs(), other.denominator.unsigned_abs()))
            .expect("a divisor of an i128 fits in one");
        let left = checked(self.numerator.checked_mul(other.denominator / divisor));
        let right = checked(other.numerator.checked_mul(self.denominator / divisor));
        Self::new(
            checked(left.checked_add(right)),
            checked((self.denominator / divisor).checked_mul(other.denominator)),
        )
    }
}

impl Neg for Ratio {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            numerator: checked(self.numerator.checked_neg()),
            denominator: self.denominator,
        }
    }
}

impl Sub for Ratio {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        self + -other
    }
}

impl Mul for Ratio {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        // Cancelled across first, so that the products stay as small as they can.
        let first = Self::new(self.numerator, other.denominator);
        let second = Self::new(other.numerator, self.denominator);
        Self::new(
            checked(first.numerator.checked_mul(second.numerator)),
            checked(first.denominator.checked_mul(second.denominator)),
        )
    }
}

impl Div for Ratio {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        assert!(other.numerator != 0, "a rational divided by 0");
        self * Self::new(other.denominator, other.numerator)
    }
}

impl PartialOrd for Ratio {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Ratio {
    fn cmp(&self, other: &Self) -> Ordering {
        (*self - *other).numerator.cmp(&0)
    }
}

/// The binomial coefficient `C(n, k)`, exactly.
///
/// # Panics
///
/// When it does not fit in 128 bits.
#[must_use]
pub fn binomial(n: u32, k: u32) -> i128 {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut value: i128 = 1;
    for index in 0..k {
        // value * (n - index) is divisible by index + 1: C(n, index + 1) is an integer.
        value = checked(value.checked_mul(i128::from(n - index))) / i128::from(index + 1);
    }
    value
}

/// `n!`, exactly.
///
/// # Panics
///
/// When it does not fit in 128 bits (from 34! on).
#[must_use]
pub fn factorial(n: u32) -> i128 {
    (1..=n).fold(1i128, |product, factor| {
        checked(product.checked_mul(i128::from(factor)))
    })
}

/// The Bernoulli numbers `B_0`, ..., `B_(count - 1)`, exactly, from
/// `sum over j <= m of C(m + 1, j) B_j = 0` (`B_1 = -1/2`).
///
/// # Panics
///
/// When one does not fit in 128 bits; up to `B_40` they fit.
#[must_use]
pub fn bernoulli(count: usize) -> Vec<Ratio> {
    let mut numbers = Vec::with_capacity(count);
    if count == 0 {
        return numbers;
    }
    numbers.push(Ratio::integer(1));
    for m in 1..count {
        let order = u32::try_from(m).expect("a count of Bernoulli numbers fits in u32");
        let sum = numbers.iter().enumerate().fold(Ratio::integer(0), |sum, (j, &number)| {
            let index = u32::try_from(j).expect("an index below a u32 fits in one");
            sum + Ratio::integer(binomial(order + 1, index)) * number
        });
        numbers.push(-sum / Ratio::integer(i128::from(order + 1)));
    }
    numbers
}

/// The terms that a block of [`Sum`] adds one after another, before it is added
/// to the others in a tree.
const SUM_BLOCK: usize = 128;

/// The sum of floating-point terms, in an order that depends only on their order:
/// in blocks of 128, one after another, and the blocks' sums in a binary tree. A
/// term goes through at most 127 additions in its block, one for each level of the
/// tree, and one more for each level where the partial sums are added up at the
/// end, so the rounding of the sum of `n` terms is at most
/// `gamma(128 + 2 ceil(log2(n / 128)))` times the sum of their magnitudes, where
/// one after another it would be `gamma(n - 1)`.
#[derive(Debug, Clone, Default)]
pub struct Sum {
    block: f64,
    count: usize,
    // partials[level] holds the sum of 2^level blocks, where there is one.
    partials: Vec<Option<f64>>,
}

impl Sum {
    /// An empty sum.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a term.
    pub fn add(&mut self, term: f64) {
        self.block += term;
        self.count += 1;
        if self.count == SUM_BLOCK {
            let mut carried = self.block;
            self.block = 0.0;
            self.count = 0;
            for level in &mut self.partials {
                if let Some(partial) = level.take() {
                    carried += partial;
                } else {
                    *level = Some(carried);
                    return;
                }
            }
            self.partials.push(Some(carried));
        }
    }

    /// The sum of the terms added: the last block and the partial sums, from the
    /// smallest level up.
    #[must_use]
    pub fn total(&self) -> f64 {
        self.partials
            .iter()
            .flatten()
            .fold(self.block, |sum, &partial| sum + partial)
    }
}

impl Extend<f64> for Sum {
    fn extend<T: IntoIterator<Item = f64>>(&mut self, terms: T) {
        for term in terms {
            self.add(term);
        }
    }
}

/// The [`Sum`] of the terms.
#[must_use]
pub fn sum(terms: impl IntoIterator<Item = f64>) -> f64 {
    let mut total = Sum::new();
    total.extend(terms);
    total.total()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[expect(
        clippy::float_cmp,
        reason = "a correctly rounded quotient is compared to the last bit"
    )]
    fn quotients_are_those_of_a_correctly_rounded_division() {
        // Below 2^53 both integers are exact, and one division rounds correctly.
        let cases: [(i128, i128); 9] = [
            (1, 3),
            (2, 3),
            (-7, 10),
            (701, 500),
            (25_251, 73_375),
            (209_599, -293_500),
            (1, 1 << 60),
            (123_456_789_012_345, 97),
            ((1 << 52) - 1, 1 << 52),
        ];
        for (numerator, denominator) in cases {
            let expected = nearest_from_i128(numerator) / nearest_from_i128(denominator);
            assert_eq!(
                quotient(numerator, denominator),
                expected,
                "{numerator} / {denominator}"
            );
        }
        // Ties go to even: 2^53 + 1 lies halfway between 2^53 and 2^53 + 2, and 2^53 + 3
        // between 2^53 + 2 and 2^53 + 4.
        assert_eq!(quotient((1 << 53) + 1, 1), 9_007_199_254_740_992.0);
        assert_eq!(quotient((1 << 53) + 3, 1), 9_007_199_254_740_996.0);
        // Just above a tie rounds up; the remainder is sticky.
        assert_eq!(quotient((1 << 54) + 3, 2), 9_007_199_254_740_994.0);
        // Large numerators and denominators, beyond binary64's integers.
        assert_eq!(quotient(1 << 100, 1 << 98), 4.0);
        assert_eq!(quotient(3 << 100, 1 << 101), 1.5);
        assert_eq!(quotient(0, -5), 0.0);
    }

    #[test]
    fn rationals_are_kept_in_lowest_terms() {
        let half = Ratio::new(3, 6);
        assert_eq!((half.numerator(), half.denominator()), (1, 2));
        let negative = Ratio::new(4, -6);
        assert_eq!((negative.numerator(), negative.denominator()), (-2, 3));
        assert_eq!(half + negative, Ratio::new(-1, 6));
        assert_eq!(half * negative, Ratio::new(-1, 3));
        assert_eq!(half / negative, Ratio::new(-3, 4));
        assert!(negative < half);
    }

    #[test]
    fn the_bernoulli_numbers() {
        let numbers = bernoulli(23);
        let expected = [
            (1, 1),
            (-1, 2),
            (1, 6),
            (0, 1),
            (-1, 30),
            (0, 1),
            (1, 42),
            (0, 1),
            (-1, 30),
            (0, 1),
            (5, 66),
            (0, 1),
            (-691, 2730),
            (0, 1),
            (7, 6),
            (0, 1),
            (-3617, 510),
            (0, 1),
            (43_867, 798),
            (0, 1),
            (-174_611, 330),
            (0, 1),
            (854_513, 138),
        ];
        for (index, (number, (numerator, denominator))) in numbers.iter().zip(expected).enumerate() {
            assert_eq!(*number, Ratio::new(numerator, denominator), "B_{index}");
        }
        assert_eq!(binomial(23, 11), 1_352_078);
        assert_eq!(factorial(22), 1_124_000_727_777_607_680_000);
    }

    #[test]
    #[expect(
        clippy::float_cmp,
        reason = "sums of small integers are exact, and compared to the last bit"
    )]
    fn sums_add_every_term_once() {
        for count in [0usize, 1, 127, 128, 129, 1000, 4096, 12_345] {
            let terms = (0..count).map(|index| nearest_from_usize(index % 7));
            let expected: usize = (0..count).map(|index| index % 7).sum();
            assert_eq!(sum(terms), nearest_from_usize(expected), "{count}");
        }
    }
}
