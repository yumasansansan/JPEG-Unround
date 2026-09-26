// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! Finite differences: the adjoints, exactly (docs/math.md, 3).
//!
//! The fields are dyadic rationals `m / 2^k` with `|m| <= 1000` and `k <= 6`. The
//! operators add or subtract at most four of them, and halve, so every value they
//! compute is exact in binary64; the inner products are then taken in exact
//! arithmetic, and the adjoint identities hold to the last bit.

mod support;

use jpeg_unround::operators::{self, Shape};
use support::exact::{self, Dyadic};
use support::synthetic::Numbers;

const SHAPES: [(usize, usize); 6] = [(1, 1), (1, 6), (5, 1), (3, 4), (8, 8), (7, 5)];

fn dyadics(size: usize, numbers: &mut Numbers) -> Vec<f64> {
    (0..size)
        .map(|_| {
            let numerator = numbers.integer(-1000, 1000);
            let power = numbers.integer(0, 6);
            f64::from(i32::try_from(numerator).expect("fits")) / f64::from(1u32 << power)
        })
        .collect()
}

fn equal(left: &Dyadic, right: &Dyadic) -> bool {
    left.equals(right)
}

#[test]
fn div_is_exactly_minus_the_adjoint_of_grad() {
    let mut numbers = Numbers::new(1);
    for (height, width) in SHAPES {
        let shape = Shape { height, width };
        let size = shape.size();
        let x = dyadics(size, &mut numbers);
        let (p_x, p_y) = (dyadics(size, &mut numbers), dyadics(size, &mut numbers));
        let (mut g_x, mut g_y, mut divergence) = (vec![0.0; size], vec![0.0; size], vec![0.0; size]);
        operators::grad(&x, shape, &mut g_x, &mut g_y);
        operators::div(&p_x, &p_y, shape, &mut divergence);
        let left = exact::inner(&g_x, &p_x).add(&exact::inner(&g_y, &p_y));
        let right = exact::inner(&x, &divergence).negated();
        assert!(equal(&left, &right), "{height} x {width}");
    }
}

#[test]
fn backward_differences_are_exactly_minus_the_adjoints_of_forward_ones() {
    let mut numbers = Numbers::new(2);
    for (height, width) in SHAPES {
        let shape = Shape { height, width };
        let size = shape.size();
        let (f, g) = (dyadics(size, &mut numbers), dyadics(size, &mut numbers));
        let (mut forward, mut backward) = (vec![0.0; size], vec![0.0; size]);
        operators::forward_x(&f, shape, &mut forward);
        operators::backward_x(&g, shape, &mut backward);
        assert!(equal(
            &exact::inner(&forward, &g),
            &exact::inner(&f, &backward).negated()
        ));
        operators::forward_y(&f, shape, &mut forward);
        operators::backward_y(&g, shape, &mut backward);
        assert!(equal(
            &exact::inner(&forward, &g),
            &exact::inner(&f, &backward).negated()
        ));
    }
}

#[test]
fn div2_is_exactly_minus_the_adjoint_of_sym_grad() {
    // In the inner product of symmetric tensors, which counts the off-diagonal entry twice.
    let mut numbers = Numbers::new(3);
    for (height, width) in SHAPES {
        let shape = Shape { height, width };
        let size = shape.size();
        let (w_x, w_y) = (dyadics(size, &mut numbers), dyadics(size, &mut numbers));
        let (r11, r22, r12) = (
            dyadics(size, &mut numbers),
            dyadics(size, &mut numbers),
            dyadics(size, &mut numbers),
        );
        let (mut e11, mut e22, mut e12) = (vec![0.0; size], vec![0.0; size], vec![0.0; size]);
        operators::sym_grad(&w_x, &w_y, shape, &mut e11, &mut e22, &mut e12);
        let (mut d_x, mut d_y) = (vec![0.0; size], vec![0.0; size]);
        operators::div2(&r11, &r22, &r12, shape, &mut d_x, &mut d_y);
        let left = exact::inner(&e11, &r11)
            .add(&exact::inner(&e22, &r22))
            .add(&exact::inner(&e12, &r12).scaled(1));
        let right = exact::inner(&w_x, &d_x).add(&exact::inner(&w_y, &d_y)).negated();
        assert!(equal(&left, &right), "{height} x {width}");
    }
}
