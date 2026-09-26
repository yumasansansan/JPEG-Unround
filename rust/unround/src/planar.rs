// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! The layout of the solvers: each row of a canvas as planes, one for each column of
//! an MCU, a plane holding that column of every MCU across (docs/math.md, 5.1).
//!
//! An MCU is `R x P` samples of the canvas, `P = 8 a` and `R = 8 b` for the largest
//! ratios of cells `a` across and `b` down, so that every component's blocks tile it.
//! The canvas's `W = P M` columns are `M` MCUs across; column `c` of a row is at
//! `(c mod P) M + (c div P)` in the row, and the rows follow one another. The same
//! column of every MCU is then one stream of `M` values, and so is the same
//! coefficient of every block of a component that lies at the same place in its MCU:
//! the 8-point transforms, the proximal maps and the differences are loops over the
//! MCUs across, which the kernels ([`crate::kernels`]) take.
//!
//! A component's coefficients are laid out alike, block row by block row: coefficient
//! `(v, u)` of block `h m + t` of a block row, `t < h` its place in the MCU, is at
//! `(v P_c + 8 t + u) M + m`, where `P_c = P / a_c` are the component's columns in an
//! MCU. Blocks that the component does not have -- beyond its last block of a row, in
//! the last MCU -- have places that nothing reads.

use std::sync::Arc;

use crate::dct::{BLOCK, BLOCK_SIZE};
use crate::error::Error;
use crate::frames::Frame;

/// The layout of one component in the planes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Part {
    /// The canvas's samples per sample of the component, down and across.
    pub ratio: (usize, usize),
    /// The component's columns in an MCU, `P / a`.
    pub planes: usize,
    /// Its blocks across an MCU, `planes / 8`.
    pub sets: usize,
    /// Its block rows in a band of MCUs, `R / (8 b)`.
    pub band_rows: usize,
    /// Its block rows.
    pub rows: usize,
    /// Its blocks across.
    pub columns: usize,
    /// For each place `t` of a block across an MCU, the MCUs across whose block there
    /// the component has: `m < valid[t]`.
    pub valid: Vec<usize>,
    /// The component's levels in the planes, 0 at the places of blocks that it does
    /// not have; made once, and shared by what takes the part.
    pub levels: Arc<[i16]>,
}

impl Part {
    /// The values of one block row of the component's coefficients in the planes.
    #[must_use]
    pub fn block_row(&self, across: usize) -> usize {
        BLOCK * self.planes * across
    }
}

/// The layout of a frame in the planes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    /// Rows of the canvas.
    pub height: usize,
    /// Columns of the canvas, `P M`.
    pub width: usize,
    /// `P`: the columns of an MCU.
    pub planes: usize,
    /// `M`: the MCUs across.
    pub across: usize,
    /// `R`: the rows of an MCU, the height of a band.
    pub band: usize,
    /// Each component's layout.
    pub parts: Vec<Part>,
}

impl Layout {
    /// The layout of a frame.
    ///
    /// # Errors
    ///
    /// [`Error::Unsupported`] where the MCUs do not tile the canvas.
    pub fn new(frame: &Frame) -> Result<Self, Error> {
        let channels = frame.channels();
        let most_across = channels.iter().map(|channel| channel.ratio().1).max().unwrap_or(1);
        let most_down = channels.iter().map(|channel| channel.ratio().0).max().unwrap_or(1);
        let (planes, band) = (BLOCK * most_across, BLOCK * most_down);
        let (height, width) = (frame.height(), frame.width());
        if !width.is_multiple_of(planes) || !height.is_multiple_of(band) {
            return Err(Error::Unsupported(format!(
                "MCUs of {band} x {planes} do not tile a canvas of {height} x {width}"
            )));
        }
        let across = width / planes;
        let mut parts = Vec::with_capacity(channels.len());
        for channel in channels {
            let (down, across_ratio) = channel.ratio();
            if !planes.is_multiple_of(BLOCK * across_ratio) || !band.is_multiple_of(BLOCK * down) {
                return Err(Error::Unsupported(format!(
                    "blocks of {} x {} do not tile MCUs of {band} x {planes}",
                    BLOCK * down,
                    BLOCK * across_ratio
                )));
            }
            let own_planes = planes / across_ratio;
            let sets = own_planes / BLOCK;
            let problem = channel.problem();
            let columns = problem.columns();
            let valid = (0..sets)
                .map(|set| {
                    if columns > set {
                        (columns - set).div_ceil(sets).min(across)
                    } else {
                        0
                    }
                })
                .collect();
            let mut part = Part {
                ratio: (down, across_ratio),
                planes: own_planes,
                sets,
                band_rows: band / (BLOCK * down),
                rows: problem.rows(),
                columns,
                valid,
                levels: Arc::from([]),
            };
            part.levels = Arc::from(values_to_planes(&part, across, problem.levels()));
            parts.push(part);
        }
        Ok(Self {
            height,
            width,
            planes,
            across,
            band,
            parts,
        })
    }

    /// The samples of one channel's canvas, `H W`.
    #[must_use]
    pub fn plane(&self) -> usize {
        self.height * self.width
    }

    /// The bands of MCU rows, `H / R`.
    #[must_use]
    pub fn bands(&self) -> usize {
        self.height / self.band
    }

    /// Where column `column` of a row is within the row.
    #[inline]
    #[must_use]
    pub fn place(&self, column: usize) -> usize {
        (column % self.planes) * self.across + column / self.planes
    }

    /// A canvas of every channel, `C x H x W`, from the natural layout (row by row,
    /// each from the left) to the planes.
    #[must_use]
    pub fn to_planes(&self, natural: &[f64]) -> Vec<f64> {
        let mut result = vec![0.0; natural.len()];
        for (source, target) in natural
            .chunks_exact(self.width)
            .zip(result.chunks_exact_mut(self.width))
        {
            // Plane p is column p of every MCU.
            for (p, plane) in target.chunks_exact_mut(self.across).enumerate() {
                for (value, &own) in plane.iter_mut().zip(source[p..].iter().step_by(self.planes)) {
                    *value = own;
                }
            }
        }
        result
    }

    /// A canvas of every channel from the planes to the natural layout.
    #[must_use]
    pub fn to_natural(&self, planar: &[f64]) -> Vec<f64> {
        let mut result = vec![0.0; planar.len()];
        self.write_natural(planar, &mut result);
        result
    }

    /// A canvas of every channel from the planes into one in the natural layout.
    pub fn write_natural(&self, planar: &[f64], natural: &mut [f64]) {
        for (source, target) in planar
            .chunks_exact(self.width)
            .zip(natural.chunks_exact_mut(self.width))
        {
            for (p, plane) in source.chunks_exact(self.across).enumerate() {
                for (value, &own) in target[p..].iter_mut().step_by(self.planes).zip(plane) {
                    *value = own;
                }
            }
        }
    }

    /// Where coefficient `index` of component `part` in natural order (64 to a block,
    /// block rows from the top) is in the planes.
    #[inline]
    #[must_use]
    pub fn coefficient_place(&self, part: &Part, index: usize) -> usize {
        let block = index / BLOCK_SIZE;
        let within = index % BLOCK_SIZE;
        let (row, column) = (block / part.columns, block % part.columns);
        let (v, u) = (within / BLOCK, within % BLOCK);
        let (mcu, set) = (column / part.sets, column % part.sets);
        row * part.block_row(self.across) + (v * part.planes + BLOCK * set + u) * self.across + mcu
    }

    /// A component's coefficients, or levels, from natural order to the planes; the
    /// places of blocks that it does not have are `T::default()`.
    #[must_use]
    pub fn values_to_planes<T: Copy + Default>(&self, part: &Part, natural: &[T]) -> Vec<T> {
        values_to_planes(part, self.across, natural)
    }

    /// A component's coefficients from the planes into natural order.
    pub fn write_values_natural(&self, part: &Part, planar: &[f64], natural: &mut [f64]) {
        let block_row = part.block_row(self.across);
        for (target, source) in natural
            .chunks_exact_mut(part.columns * BLOCK_SIZE)
            .zip(planar.chunks_exact(block_row))
        {
            for (column, block) in target.as_chunks_mut::<BLOCK_SIZE>().0.iter_mut().enumerate() {
                let (mcu, set) = (column / part.sets, column % part.sets);
                for (v, values) in block.as_chunks_mut::<BLOCK>().0.iter_mut().enumerate() {
                    let first = (v * part.planes + BLOCK * set) * self.across + mcu;
                    for (value, &own) in values.iter_mut().zip(source[first..].iter().step_by(self.across)) {
                        *value = own;
                    }
                }
            }
        }
    }
}

/// A component's values in natural order in the planes of `across` MCUs, block by
/// block: coefficient `(v, u)` of the block in column `c` of a block row is at
/// `(v P_c + 8 (c mod h) + u) M + c div h`.
fn values_to_planes<T: Copy + Default>(part: &Part, across: usize, natural: &[T]) -> Vec<T> {
    let block_row = part.block_row(across);
    let mut result = vec![T::default(); part.rows * block_row];
    for (source, target) in natural
        .chunks_exact(part.columns * BLOCK_SIZE)
        .zip(result.chunks_exact_mut(block_row))
    {
        for (column, block) in source.as_chunks::<BLOCK_SIZE>().0.iter().enumerate() {
            let (mcu, set) = (column / part.sets, column % part.sets);
            for (v, values) in block.as_chunks::<BLOCK>().0.iter().enumerate() {
                let first = (v * part.planes + BLOCK * set) * across + mcu;
                for (value, &own) in target[first..].iter_mut().step_by(across).zip(values) {
                    *value = own;
                }
            }
        }
    }
    result
}
