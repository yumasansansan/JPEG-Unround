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
//!
//! The transforms go block row by block row with the 8-point transforms of the
//! kernels ([`crate::kernels::forward8`]), which take eight streams at a time: down
//! the columns of the block row at once, its eight rows the streams; and then along
//! the rows of its blocks, each row of frequencies moved to eight streams, the same
//! column of every block one stream, as the solvers lay them out ([`crate::planar`]).

use std::f64::consts::FRAC_1_SQRT_2;

use crate::kernels;

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
    forward_into(canvas, width, height / BLOCK, width / BLOCK, &mut coefficients);
    coefficients
}

/// The coefficients of the `rows x columns` blocks at the top left of a canvas
/// `stride` samples wide, into `coefficients`: down the columns of every block of a
/// block row, and then along the rows, the arithmetic of the even and odd halves
/// (docs/math.md, 1.1).
///
/// # Panics
///
/// When the canvas does not hold the blocks, or `coefficients` is not their size.
pub fn forward_into(canvas: &[f64], stride: usize, rows: usize, columns: usize, coefficients: &mut [f64]) {
    let width = columns * BLOCK;
    assert!(
        width <= stride && canvas.len() >= rows * BLOCK * stride,
        "a canvas holds the blocks"
    );
    assert_eq!(
        coefficients.len(),
        rows * columns * BLOCK_SIZE,
        "room for the coefficients"
    );
    if width == 0 {
        return;
    }
    let row = BLOCK * width;
    let (mut down, mut streams, mut along) = (vec![0.0; row], vec![0.0; width], vec![0.0; width]);
    for (source, target) in canvas.chunks(BLOCK * stride).zip(coefficients.chunks_exact_mut(row)) {
        kernels::forward8(source, stride, &mut down, width, width);
        for v in 0..BLOCK {
            // Row v of each block's vertical frequencies, column u of every block in
            // stream u; its horizontal frequencies go to row v of each block.
            to_streams(&down[v * width..(v + 1) * width], BLOCK, &mut streams, columns);
            kernels::forward8(&streams, columns, &mut along, columns, columns);
            from_streams(&along, &mut target[v * BLOCK..], BLOCK_SIZE, columns);
        }
    }
}

/// Eight streams of `count` values, one after another in `streams`, from `count`
/// groups of eight, `stride` apart in `groups`: value `u` of group `m` to value `m` of
/// stream `u`.
fn to_streams(groups: &[f64], stride: usize, streams: &mut [f64], count: usize) {
    let (s0, rest) = streams.split_at_mut(count);
    let (s1, rest) = rest.split_at_mut(count);
    let (s2, rest) = rest.split_at_mut(count);
    let (s3, rest) = rest.split_at_mut(count);
    let (s4, rest) = rest.split_at_mut(count);
    let (s5, rest) = rest.split_at_mut(count);
    let (s6, s7) = rest.split_at_mut(count);
    let s7 = &mut s7[..count];
    for (m, group) in groups.chunks(stride).take(count).enumerate() {
        let group = &group[..BLOCK];
        (s0[m], s1[m], s2[m], s3[m]) = (group[0], group[1], group[2], group[3]);
        (s4[m], s5[m], s6[m], s7[m]) = (group[4], group[5], group[6], group[7]);
    }
}

/// Eight streams of `count` values to `count` groups of eight, `stride` apart: value
/// `m` of stream `u` to value `u` of group `m`.
fn from_streams(streams: &[f64], groups: &mut [f64], stride: usize, count: usize) {
    let (s0, rest) = streams.split_at(count);
    let (s1, rest) = rest.split_at(count);
    let (s2, rest) = rest.split_at(count);
    let (s3, rest) = rest.split_at(count);
    let (s4, rest) = rest.split_at(count);
    let (s5, rest) = rest.split_at(count);
    let (s6, s7) = rest.split_at(count);
    let s7 = &s7[..count];
    for (m, group) in groups.chunks_mut(stride).take(count).enumerate() {
        let group = &mut group[..BLOCK];
        (group[0], group[1], group[2], group[3]) = (s0[m], s1[m], s2[m], s3[m]);
        (group[4], group[5], group[6], group[7]) = (s4[m], s5[m], s6[m], s7[m]);
    }
}

/// `D^T`, the inverse DCT: the canvas of `rows x columns` blocks with these
/// coefficients.
///
/// # Panics
///
/// When there are not `rows x columns` blocks of coefficients.
#[must_use]
pub fn inverse(coefficients: &[f64], rows: usize, columns: usize) -> Vec<f64> {
    let mut canvas = vec![0.0; rows * columns * BLOCK_SIZE];
    inverse_into(coefficients, rows, columns, &mut canvas, columns * BLOCK);
    canvas
}

/// The samples of `rows x columns` blocks of coefficients into the top left of a
/// canvas `stride` samples wide: along the rows of every block of a block row, and
/// then down the columns, as [`forward_into`] computes the other way.
///
/// # Panics
///
/// When there are not `rows x columns` blocks of coefficients, or the canvas does not
/// hold them.
pub fn inverse_into(coefficients: &[f64], rows: usize, columns: usize, canvas: &mut [f64], stride: usize) {
    let width = columns * BLOCK;
    assert_eq!(
        coefficients.len(),
        rows * columns * BLOCK_SIZE,
        "coefficients of {rows} x {columns} blocks"
    );
    assert!(
        width <= stride && canvas.len() >= rows * BLOCK * stride,
        "a canvas holds the blocks"
    );
    if width == 0 {
        return;
    }
    let row = BLOCK * width;
    let (mut streams, mut along, mut down) = (vec![0.0; width], vec![0.0; width], vec![0.0; row]);
    for (source, target) in coefficients.chunks_exact(row).zip(canvas.chunks_mut(BLOCK * stride)) {
        for v in 0..BLOCK {
            // Row v of each block's coefficients, frequency u of every block in stream
            // u; the samples of its horizontal inverse go to row v of the block row.
            to_streams(&source[v * BLOCK..], BLOCK_SIZE, &mut streams, columns);
            kernels::inverse8(&streams, columns, &mut along, columns, columns);
            from_streams(&along, &mut down[v * width..(v + 1) * width], BLOCK, columns);
        }
        kernels::inverse8(&down, width, target, stride, width);
    }
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
