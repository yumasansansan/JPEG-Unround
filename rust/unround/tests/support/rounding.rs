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
//! of its magnitude. The 8-point transforms go by their even and odd halves
//! (docs/math.md, 1.1): a sum or difference of two values, rounded once, and then a
//! sum of four products; or two sums of four products, and their sum or difference,
//! rounded once. Each output is then within `e1 |C| |x|` of the exact one, `C` the
//! exact basis, `e1 = (1 + U)^2 (1 + gamma(4)) - 1 <= gamma(6)`; and the block DCT,
//! `(B X) B^T`, within `e1 (2 + e1)`, about 12U, times `|C| |X| |C|^T`, as is the
//! inverse. The bounds below take `gamma(6) (2 + gamma(6))` times the magnitudes
//! computed with `|B|`, and a factor `1 + gamma(50)` for how far those may lie below
//! the exact ones (`|C|` within `|B| / (1 - U)`, and two sums of eight nonnegative
//! products, each within `gamma(8)`) and for the rounding of the bound's own product.

use jpeg_unround::dct::{BASIS, BLOCK, BLOCK_SIZE, Block};

/// The unit roundoff of binary64.
pub const U: f64 = 1.0 / 9_007_199_254_740_992.0;

/// Higham's `gamma_n = n U / (1 - n U)`.
#[must_use]
pub fn gamma(n: usize) -> f64 {
    let count = f64::from(u32::try_from(n).expect("a count of terms fits in u32"));
    count * U / (1.0 - count * U)
}

/// The rounding of the block DCT relative to its magnitudes, with the factor that
/// makes the magnitudes computed in floating point bound it.
fn relative() -> f64 {
    gamma(6) * (2.0 + gamma(6)) * (1.0 + gamma(50))
}

/// A bound of the roundings that a term of a record's value goes through in either
/// layout (docs/math.md, Arithmetic), on a canvas of `height` x `width`: a row of at most
/// `8 W` terms in lanes (a block row of the model's sums, 64 terms a block),
/// `W + 7` additions; the rows' or the pixels' [`jpeg_unround::exact::Sum`], of at most
/// `H W` terms, `128 + 2 ceil(log2(H W / 128))`; and at most 16 more, of the channels,
/// the components and the parts of the value.
#[must_use]
pub fn record_roundings(height: usize, width: usize) -> usize {
    let blocks = (height * width).div_ceil(128).max(1);
    let levels = usize::try_from(blocks.next_power_of_two().trailing_zeros()).expect("a count of levels");
    width + 7 + 128 + 2 * levels + 16
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
    let relative = relative();
    forward_magnitude(block).map(|row| row.map(|value| relative * value))
}

/// A bound, sample by sample, of the rounding of the inverse DCT of a block.
#[must_use]
pub fn inverse_error(block: &Block) -> Block {
    let relative = relative();
    inverse_magnitude(block).map(|row| row.map(|value| relative * value))
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
