// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! Bounds of rounding error, from which the tests take their tolerances.
//!
//! `U` is the unit roundoff of binary64, 2^-53. A sum of `n` terms, or a dot product
//! of `n` pairs, computed in floating point in any order, and with any of its
//! products fused with the sums that take them, differs from the exact one by at
//! most `gamma(n)` times the sum of the magnitudes of its terms (N. J. Higham,
//! Accuracy and Stability of Numerical Algorithms, 2nd ed., sections 3.1 and 4.2).
//!
//! Every entry of the DCT's basis is the double nearest to the exact one, within `U`
//! of its magnitude. A product `B X` of the basis and a block is then within
//! `e1 |B| |X|` of the exact one, `e1 = gamma(8) (1 + U) + U`, about 9U; and the
//! block DCT, `(B X) B^T`, within `gamma(8) (1 + e1) (1 + U) + U (1 + e1) + e1`,
//! about 18U, times `|B| |X| |B|^T`. The bounds below take 19U, which covers the
//! terms of second order and the rounding of the bound's own computation.

use jpeg_unround::dct::{BASIS, BLOCK, BLOCK_SIZE, Block};

/// The unit roundoff of binary64.
pub const U: f64 = 1.0 / 9_007_199_254_740_992.0;

/// Higham's `gamma_n = n U / (1 - n U)`.
#[must_use]
pub fn gamma(n: usize) -> f64 {
    let count = f64::from(u32::try_from(n).expect("a count of terms fits in u32"));
    count * U / (1.0 - count * U)
}

fn absolute(block: &Block) -> Block {
    block.map(|row| row.map(f64::abs))
}

fn product(left: &Block, right: &Block) -> Block {
    let mut result = [[0.0; BLOCK]; BLOCK];
    for (i, row) in result.iter_mut().enumerate() {
        for (j, entry) in row.iter_mut().enumerate() {
            *entry = (0..BLOCK).map(|k| left[i][k] * right[k][j]).sum();
        }
    }
    result
}

fn transposed(block: &Block) -> Block {
    let mut result = [[0.0; BLOCK]; BLOCK];
    for (i, row) in block.iter().enumerate() {
        for (j, &value) in row.iter().enumerate() {
            result[j][i] = value;
        }
    }
    result
}

/// `|B| |X| |B|^T` of a block.
#[must_use]
pub fn forward_magnitude(block: &Block) -> Block {
    let basis = absolute(&BASIS);
    product(&product(&basis, &absolute(block)), &transposed(&basis))
}

/// `|B|^T |C| |B|` of a block of coefficients.
#[must_use]
pub fn inverse_magnitude(block: &Block) -> Block {
    let basis = absolute(&BASIS);
    product(&product(&transposed(&basis), &absolute(block)), &basis)
}

/// A bound, coefficient by coefficient, of the rounding of the DCT of a block.
#[must_use]
pub fn forward_error(block: &Block) -> Block {
    forward_magnitude(block).map(|row| row.map(|value| 19.0 * U * value))
}

/// A bound, sample by sample, of the rounding of the inverse DCT of a block.
#[must_use]
pub fn inverse_error(block: &Block) -> Block {
    inverse_magnitude(block).map(|row| row.map(|value| 19.0 * U * value))
}

/// A bound, coefficient by coefficient, of `|forward(inverse(c)) - c|` of a block:
/// the inverse rounds within `inverse_error`, the canvas it gives is within
/// `|B^T| |c| |B|` plus that, the forward rounds within `forward_error` of it, and
/// carries the inverse's rounding through `|B| e |B|^T`.
#[must_use]
pub fn roundtrip_error(block: &Block) -> Block {
    let first = inverse_error(block);
    let magnitude = inverse_magnitude(block);
    let mut canvas = [[0.0; BLOCK]; BLOCK];
    for i in 0..BLOCK {
        for j in 0..BLOCK {
            canvas[i][j] = magnitude[i][j] + first[i][j];
        }
    }
    let rounded = forward_error(&canvas);
    let carried = forward_magnitude(&first);
    let mut result = [[0.0; BLOCK]; BLOCK];
    for i in 0..BLOCK {
        for j in 0..BLOCK {
            result[i][j] = rounded[i][j] + carried[i][j];
        }
    }
    result
}

/// The 64 coefficients of block `index` of an array, as a block.
#[must_use]
pub fn block_of(values: &[f64], index: usize) -> Block {
    jpeg_unround::dct::gather(&values[index * BLOCK_SIZE..(index + 1) * BLOCK_SIZE])
}
