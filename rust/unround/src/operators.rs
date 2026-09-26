// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! Finite differences on a plane, their adjoints, and pointwise norms
//! (docs/math.md, 3).
//!
//! A plane is `height x width` values, row by row: `i` the row from the top, `j`
//! the column from the left. A vector field is two planes, its entries across and
//! down; a symmetric tensor field three, `r11`, `r22` and `r12`, whose inner
//! product, norm and ball count `r12` twice, as the full 2x2 matrix does.
//!
//! [`grad`] and [`div`] are negative adjoints, and so are [`sym_grad`] and [`div2`]:
//! `<grad x, p> = -<x, div p>` and `<sym_grad w, r> = -<w, div2 r>`. Their
//! coefficients are integers, save the halving of `sym_grad`, so on values that are
//! small multiples of 1/2 they compute exactly, which the tests use to check the
//! adjoints.

use crate::dct::multiply_add;

/// The shape of a plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Shape {
    /// Rows.
    pub height: usize,
    /// Columns.
    pub width: usize,
}

impl Shape {
    /// The values of a plane of this shape.
    #[must_use]
    pub fn size(self) -> usize {
        self.height * self.width
    }
}

/// The forward difference across, 0 in the last column (Neumann).
pub fn forward_x(f: &[f64], shape: Shape, out: &mut [f64]) {
    for (source, target) in f.chunks_exact(shape.width).zip(out.chunks_exact_mut(shape.width)) {
        for (entry, pair) in target.iter_mut().zip(source.windows(2)) {
            *entry = pair[1] - pair[0];
        }
        target[shape.width - 1] = 0.0;
    }
}

/// The forward difference down, 0 in the last row (Neumann).
pub fn forward_y(f: &[f64], shape: Shape, out: &mut [f64]) {
    let width = shape.width;
    let inner = (shape.height - 1) * width;
    for ((entry, &here), &below) in out[..inner].iter_mut().zip(&f[..inner]).zip(&f[width..]) {
        *entry = below - here;
    }
    out[inner..].fill(0.0);
}

/// The negative adjoint of [`forward_x`]: `f[i][j]` for `j < W - 1`, less
/// `f[i][j - 1]` for `j > 0`.
pub fn backward_x(f: &[f64], shape: Shape, out: &mut [f64]) {
    let width = shape.width;
    for (source, target) in f.chunks_exact(width).zip(out.chunks_exact_mut(width)) {
        if width == 1 {
            // Neither the sample itself nor one before it: 0 - 0.
            target[0] = 0.0;
            continue;
        }
        // The first sample less none, the last none less the one before it, and
        // between them each sample less the one before it.
        target[0] = source[0] - 0.0;
        for (entry, pair) in target[1..width - 1].iter_mut().zip(source.windows(2)) {
            *entry = pair[1] - pair[0];
        }
        target[width - 1] = 0.0 - source[width - 2];
    }
}

/// The negative adjoint of [`forward_y`].
pub fn backward_y(f: &[f64], shape: Shape, out: &mut [f64]) {
    let width = shape.width;
    for (i, target) in out.chunks_exact_mut(width).enumerate() {
        for (j, entry) in target.iter_mut().enumerate() {
            let own = if i + 1 < shape.height { f[i * width + j] } else { 0.0 };
            let before = if i > 0 { f[(i - 1) * width + j] } else { 0.0 };
            *entry = own - before;
        }
    }
}

/// The gradient of a plane, by forward differences: across into `out_x`, down into
/// `out_y`.
pub fn grad(x: &[f64], shape: Shape, out_x: &mut [f64], out_y: &mut [f64]) {
    forward_x(x, shape, out_x);
    forward_y(x, shape, out_y);
}

/// The divergence of a vector field, `-grad` transposed.
pub fn div(p_x: &[f64], p_y: &[f64], shape: Shape, out: &mut [f64]) {
    let mut down = vec![0.0; shape.size()];
    backward_x(p_x, shape, out);
    backward_y(p_y, shape, &mut down);
    for (entry, &value) in out.iter_mut().zip(&down) {
        *entry += value;
    }
}

/// The symmetrized gradient of a vector field, by backward differences:
/// `(r11, r22, r12)`.
pub fn sym_grad(w_x: &[f64], w_y: &[f64], shape: Shape, r11: &mut [f64], r22: &mut [f64], r12: &mut [f64]) {
    backward_x(w_x, shape, r11);
    backward_y(w_y, shape, r22);
    let mut across = vec![0.0; shape.size()];
    backward_y(w_x, shape, r12);
    backward_x(w_y, shape, &mut across);
    // Halved: the midpoint of two doubles of these sizes is their sum over 2, and
    // dividing by 2 is exact.
    for (entry, &value) in r12.iter_mut().zip(&across) {
        *entry = f64::midpoint(*entry, value);
    }
}

/// The divergence of a symmetric tensor field, `-sym_grad` transposed, by forward
/// differences.
pub fn div2(r11: &[f64], r22: &[f64], r12: &[f64], shape: Shape, out_x: &mut [f64], out_y: &mut [f64]) {
    let mut second = vec![0.0; shape.size()];
    forward_x(r11, shape, out_x);
    forward_y(r12, shape, &mut second);
    for (entry, &value) in out_x.iter_mut().zip(&second) {
        *entry += value;
    }
    forward_x(r12, shape, out_y);
    forward_y(r22, shape, &mut second);
    for (entry, &value) in out_y.iter_mut().zip(&second) {
        *entry += value;
    }
}

/// `x^2 + y^2`, the square of a vector's norm.
#[inline]
#[must_use]
pub fn vector_square(x: f64, y: f64) -> f64 {
    multiply_add(x, x, y * y)
}

/// `r11^2 + r22^2 + 2 r12^2`, the square of a tensor's norm, the off-diagonal entry
/// counted twice.
#[inline]
#[must_use]
pub fn tensor_square(r11: f64, r22: f64, r12: f64) -> f64 {
    multiply_add(2.0 * r12, r12, multiply_add(r11, r11, r22 * r22))
}

/// The factor that a vector or tensor of the squared norm `square` is divided by to
/// project it onto the ball of the radius: `max(1, |v| / radius)`.
#[inline]
#[must_use]
pub fn ball_factor(square: f64, radius: f64) -> f64 {
    (square.sqrt() / radius).max(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[expect(clippy::float_cmp, reason = "differences of small integers are exact")]
    fn the_gradient_has_neumann_boundaries() {
        let shape = Shape { height: 3, width: 4 };
        let x: Vec<f64> = (0..12u32).map(|index| f64::from(index * index)).collect();
        let (mut across, mut down) = (vec![0.0; 12], vec![0.0; 12]);
        grad(&x, shape, &mut across, &mut down);
        for i in 0..3 {
            assert_eq!(across[i * 4 + 3], 0.0);
            for j in 0..3 {
                assert_eq!(across[i * 4 + j], x[i * 4 + j + 1] - x[i * 4 + j]);
            }
        }
        for j in 0..4 {
            assert_eq!(down[2 * 4 + j], 0.0);
            for i in 0..2 {
                assert_eq!(down[i * 4 + j], x[(i + 1) * 4 + j] - x[i * 4 + j]);
            }
        }
    }

    #[test]
    fn the_symmetrized_gradient_of_a_ramp_is_zero_inside() {
        let shape = Shape { height: 10, width: 12 };
        let ramp: Vec<f64> = (0..120u32)
            .map(|index| 3.0 * f64::from(index % 12) - 2.0 * f64::from(index / 12))
            .collect();
        let (mut across, mut down) = (vec![0.0; 120], vec![0.0; 120]);
        grad(&ramp, shape, &mut across, &mut down);
        let (mut r11, mut r22, mut r12) = (vec![0.0; 120], vec![0.0; 120], vec![0.0; 120]);
        sym_grad(&across, &down, shape, &mut r11, &mut r22, &mut r12);
        for i in 1..8 {
            for j in 1..10 {
                let index = i * 12 + j;
                assert_eq!((r11[index], r22[index], r12[index]), (0.0, 0.0, 0.0), "{i}, {j}");
            }
        }
    }

    #[test]
    #[expect(
        clippy::float_cmp,
        reason = "the norm of an exact square root is compared to the last bit"
    )]
    fn the_tensor_norm_counts_the_off_diagonal_twice() {
        assert_eq!(tensor_square(3.0, 4.0, 1.0), 27.0);
        assert_eq!(vector_square(3.0, 4.0).sqrt(), 5.0);
        assert_eq!(ball_factor(25.0, 10.0), 1.0);
        assert_eq!(ball_factor(400.0, 10.0), 2.0);
    }
}
