// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! The 8x8 block DCT of JPEG: the orthonormal DCT-II of every block
//! (docs/math.md, 1.1).
//!
//! A canvas is `height x width` samples, row by row, where both are multiples of 8.
//! Its coefficients are its blocks, block rows from the top and blocks from the
//! left, each 64 in natural order: entry `64 b + 8 v + u` is the coefficient of
//! vertical frequency `v` and horizontal frequency `u` of block `b`. That is the
//! layout of the C layer's coefficients.

use std::f64::consts::FRAC_1_SQRT_2;

/// The side of a block.
pub const BLOCK: usize = 8;

/// The coefficients of a block.
pub const BLOCK_SIZE: usize = BLOCK * BLOCK;

/// `cos(pi j / 16)` for `j = 0, ..., 8`, each the double nearest to it.
///
/// The literals have 25 digits, as the other implementations write them, and each
/// is read as the double nearest to it, which is the double nearest to the cosine;
/// `cos(pi / 4)` is `FRAC_1_SQRT_2`, which is. A library cosine of `pi j / 16` would
/// not give that: the angle is rounded before the cosine is taken, and near
/// `pi / 2` the cosine magnifies that rounding several times.
#[expect(
    clippy::excessive_precision,
    reason = "the constants are written to 25 digits, as the other implementations write them"
)]
pub const COSINES: [f64; 9] = [
    1.0,
    0.980_785_280_403_230_449_126_182_2,
    0.923_879_532_511_286_756_128_183_2,
    0.831_469_612_302_545_237_078_788_4,
    FRAC_1_SQRT_2,
    0.555_570_233_019_602_224_742_830_8,
    0.382_683_432_365_089_771_728_460_0,
    0.195_090_322_016_128_267_848_284_9,
    0.0,
];

/// `sqrt(1/8)`, the double nearest to it: half of the nearest to `1 / sqrt(2)`,
/// halving being exact.
pub const SQRT_EIGHTH: f64 = FRAC_1_SQRT_2 / 2.0;

const fn entry(k: usize, n: usize) -> f64 {
    if k == 0 {
        return SQRT_EIGHTH;
    }
    // cos(pi m / 16) for m = (2n + 1) k, from m reduced in integers: over a period of
    // 32, [0, 8] is as it is, (8, 16] and (16, 24] are negated, (24, 32) mirrored.
    let m = ((2 * n + 1) * k) % 32;
    let (j, negated) = if m <= 8 {
        (m, false)
    } else if m <= 16 {
        (16 - m, true)
    } else if m <= 24 {
        (m - 16, true)
    } else {
        (32 - m, false)
    };
    // Halving is exact: every entry is the double nearest to the entry of the exact basis.
    let half = COSINES[j] / 2.0;
    if negated { -half } else { half }
}

const fn basis(transposed: bool) -> [[f64; BLOCK]; BLOCK] {
    let mut matrix = [[0.0; BLOCK]; BLOCK];
    let mut k = 0;
    while k < BLOCK {
        let mut n = 0;
        while n < BLOCK {
            if transposed {
                matrix[n][k] = entry(k, n);
            } else {
                matrix[k][n] = entry(k, n);
            }
            n += 1;
        }
        k += 1;
    }
    matrix
}

/// The basis: row `k` is the basis vector of frequency `k`, each entry the double
/// nearest to its exact value.
pub const BASIS: [[f64; BLOCK]; BLOCK] = basis(false);

/// The basis transposed: row `n` holds the entries of every frequency at position `n`.
pub const BASIS_TRANSPOSED: [[f64; BLOCK]; BLOCK] = basis(true);

/// A block of samples or of coefficients, row by row.
pub type Block = [[f64; BLOCK]; BLOCK];

/// `a * b + c`, fused where the processor has a fused multiply-add
/// (docs/math.md, Arithmetic): one rounding there, two elsewhere.
#[inline]
#[must_use]
pub fn multiply_add(a: f64, b: f64, c: f64) -> f64 {
    #[cfg(any(target_arch = "aarch64", target_feature = "fma"))]
    {
        a.mul_add(b, c)
    }
    #[cfg(not(any(target_arch = "aarch64", target_feature = "fma")))]
    {
        a * b + c
    }
}

/// The product `left right` of two 8x8 matrices, row by row: each row of the result
/// is the sum over `k` of `left[i][k]` times row `k` of `right`, `k` in order.
#[inline]
fn product(left: &Block, right: &Block) -> Block {
    let mut result = [[0.0; BLOCK]; BLOCK];
    for (row, factors) in result.iter_mut().zip(left) {
        for (&factor, source) in factors.iter().zip(right) {
            for (entry, &value) in row.iter_mut().zip(source) {
                *entry = multiply_add(factor, value, *entry);
            }
        }
    }
    result
}

/// The coefficients of a block of samples, `(B X) B^T`.
#[inline]
#[must_use]
pub fn forward_block(samples: &Block) -> Block {
    product(&product(&BASIS, samples), &BASIS_TRANSPOSED)
}

/// The samples of a block of coefficients, `(B^T C) B`.
#[inline]
#[must_use]
pub fn inverse_block(coefficients: &Block) -> Block {
    product(&product(&BASIS_TRANSPOSED, coefficients), &BASIS)
}

/// The block of a canvas `width` samples wide whose top-left sample is at `row`,
/// `column`.
#[inline]
#[must_use]
pub fn load(canvas: &[f64], width: usize, row: usize, column: usize) -> Block {
    let mut block = [[0.0; BLOCK]; BLOCK];
    for (index, line) in block.iter_mut().enumerate() {
        let start = (row + index) * width + column;
        line.copy_from_slice(&canvas[start..start + BLOCK]);
    }
    block
}

/// Writes a block into a canvas `width` samples wide, its top-left sample at
/// `row`, `column`.
#[inline]
pub fn store(block: &Block, canvas: &mut [f64], width: usize, row: usize, column: usize) {
    for (index, line) in block.iter().enumerate() {
        let start = (row + index) * width + column;
        canvas[start..start + BLOCK].copy_from_slice(line);
    }
}

/// A block of coefficients as the 64 of the layout, in natural order.
#[inline]
pub fn flatten(block: &Block, coefficients: &mut [f64]) {
    for (line, target) in block.iter().zip(coefficients.as_chunks_mut::<BLOCK>().0) {
        *target = *line;
    }
}

/// The 64 coefficients of the layout, in natural order, as a block.
#[inline]
#[must_use]
pub fn gather(coefficients: &[f64]) -> Block {
    let mut block = [[0.0; BLOCK]; BLOCK];
    for (line, source) in block.iter_mut().zip(coefficients.as_chunks::<BLOCK>().0) {
        *line = *source;
    }
    block
}

/// `D`: the coefficients of every block of a canvas of `height x width` samples.
///
/// # Panics
///
/// When the canvas is not whole blocks, or not `height x width`.
#[must_use]
pub fn forward(canvas: &[f64], height: usize, width: usize) -> Vec<f64> {
    assert!(
        height.is_multiple_of(BLOCK) && width.is_multiple_of(BLOCK),
        "a canvas is whole blocks, and {height} x {width} is not"
    );
    assert_eq!(canvas.len(), height * width, "a canvas of {height} x {width}");
    let mut coefficients = vec![0.0; height * width];
    let columns = width / BLOCK;
    for (index, target) in coefficients.as_chunks_mut::<BLOCK_SIZE>().0.iter_mut().enumerate() {
        let (row, column) = (index / columns * BLOCK, index % columns * BLOCK);
        flatten(&forward_block(&load(canvas, width, row, column)), target);
    }
    coefficients
}

/// `D^T`, the inverse DCT: the canvas of `rows x columns` blocks with these
/// coefficients.
///
/// # Panics
///
/// When there are not `rows x columns` blocks of coefficients.
#[must_use]
pub fn inverse(coefficients: &[f64], rows: usize, columns: usize) -> Vec<f64> {
    assert_eq!(
        coefficients.len(),
        rows * columns * BLOCK_SIZE,
        "coefficients of {rows} x {columns} blocks"
    );
    let width = columns * BLOCK;
    let mut canvas = vec![0.0; coefficients.len()];
    for (index, block) in coefficients.as_chunks::<BLOCK_SIZE>().0.iter().enumerate() {
        let (row, column) = (index / columns * BLOCK, index % columns * BLOCK);
        store(&inverse_block(&gather(block)), &mut canvas, width, row, column);
    }
    canvas
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[expect(clippy::float_cmp, reason = "the basis is compared to the last bit")]
    fn the_basis_is_built_from_the_cosines() {
        assert_eq!(SQRT_EIGHTH, 0.125f64.sqrt());
        for &entry in &BASIS[0] {
            assert_eq!(entry, SQRT_EIGHTH);
        }
        // cos(pi (2n + 1) k / 16) / 2 for the first row of each frequency and a few more.
        assert_eq!(BASIS[1][0], COSINES[1] / 2.0);
        assert_eq!(BASIS[1][7], -COSINES[1] / 2.0);
        assert_eq!(BASIS[2][1], COSINES[6] / 2.0);
        assert_eq!(BASIS[4][1], -COSINES[4] / 2.0);
        // cos(49 pi / 16) = cos(17 pi / 16) = -cos(pi / 16).
        assert_eq!(BASIS[7][3], -COSINES[1] / 2.0);
        for k in 0..BLOCK {
            for n in 0..BLOCK {
                assert_eq!(BASIS_TRANSPOSED[n][k], BASIS[k][n]);
            }
        }
    }

    #[test]
    #[expect(
        clippy::float_cmp,
        reason = "blocks are moved, not computed, and compared to the last bit"
    )]
    fn blocks_are_in_raster_order() {
        let canvas: Vec<f64> = (0..16 * 24)
            .map(|index| f64::from(u32::try_from(index).expect("fits")))
            .collect();
        let block = load(&canvas, 24, 8, 16);
        assert_eq!(block[0][0], canvas[8 * 24 + 16]);
        assert_eq!(block[7][7], canvas[15 * 24 + 23]);
        let mut copy = vec![0.0; canvas.len()];
        for row in (0..16).step_by(8) {
            for column in (0..24).step_by(8) {
                store(&load(&canvas, 24, row, column), &mut copy, 24, row, column);
            }
        }
        assert_eq!(copy, canvas);
    }
}
