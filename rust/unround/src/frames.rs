// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! The components of a file on one canvas (docs/math.md, 1.3, 4.4 and 6.6).
//!
//! A [`Frame`] holds the problem of every component, with the ratio of its cells to
//! the canvas, and the canvas's shape. Its unknowns are one canvas per component,
//! stacked: `C x H x W` values, channel by channel, each row by row. The
//! coefficients of a component are the DCT of the means of its cells, over its
//! blocks; what no coefficient constrains -- the deviations within the cells and
//! the samples beyond the blocks -- is free. One component whose canvas is its
//! blocks has nothing free.
//!
//! Vector fields ([`Vector`]) and symmetric tensor fields ([`Tensor`]) hold each of
//! their entries as such a stack. The regularizers take the channels coupled, pixel
//! by pixel, or each on its own, with a weight for each channel's differences.

use crate::dct::{self, BLOCK, BLOCK_SIZE};
use crate::error::Error;
use crate::exact::{Sum, nearest_from_usize};
use crate::kernels;
use crate::model::{Problem, Step, Tgv, Tv};
use crate::operators::{self, Shape, tensor_square, vector_square};

/// Where the samples beyond the blocks start, and the centre of their box in 6.6.
pub const FREE_CENTRE: f64 = 128.0;

/// `R` of 6.6 by default: the box of a solution whose samples lie within [0, 255].
pub const FREE_RADIUS: f64 = 255.0;

/// A component on the canvas: its problem, for its own samples, and the ratio of
/// its cells, `(r_v, r_h)`: the canvas's samples per sample of the component down
/// and across.
#[derive(Debug, Clone)]
pub struct Channel {
    problem: Problem,
    ratio: (usize, usize),
}

impl Channel {
    /// A component with cells of `ratio`, `(down, across)`.
    #[must_use]
    pub fn new(problem: Problem, ratio: (usize, usize)) -> Self {
        Self { problem, ratio }
    }

    /// The component's problem.
    #[must_use]
    pub fn problem(&self) -> &Problem {
        &self.problem
    }

    /// The ratio of its cells, `(down, across)`.
    #[must_use]
    pub fn ratio(&self) -> (usize, usize) {
        self.ratio
    }

    /// `n`: the samples of the canvas in a cell.
    #[must_use]
    pub fn cells(&self) -> usize {
        self.ratio.0 * self.ratio.1
    }

    /// The rows and columns of the canvas that the component's blocks cover.
    #[must_use]
    pub fn extent(&self) -> (usize, usize) {
        (
            self.problem.height() * self.ratio.0,
            self.problem.width() * self.ratio.1,
        )
    }
}

/// The components of a file, on one canvas of `height x width` samples.
#[derive(Debug, Clone)]
pub struct Frame {
    channels: Vec<Channel>,
    height: usize,
    width: usize,
}

/// A primal point of a frame: the coefficients of every component, each within its
/// intervals; the canvas, `C x H x W`; and TGV's field `w`.
#[derive(Debug, Clone, PartialEq)]
pub struct Primal {
    /// The coefficients of each component.
    pub coefficients: Vec<Vec<f64>>,
    /// The canvas of every channel.
    pub canvas: Vec<f64>,
    /// TGV's vector field.
    pub w: Option<Vector>,
}

/// A vector field of every channel: its entries across and down, each `C x H x W`.
#[derive(Debug, Clone, PartialEq)]
pub struct Vector {
    /// The entries across.
    pub x: Vec<f64>,
    /// The entries down.
    pub y: Vec<f64>,
}

impl Vector {
    /// A field of zeros of `size` entries each.
    #[must_use]
    pub fn zeros(size: usize) -> Self {
        Self {
            x: vec![0.0; size],
            y: vec![0.0; size],
        }
    }
}

/// A symmetric tensor field of every channel: `r11`, `r22` and `r12`, each
/// `C x H x W`.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    /// The entries `r11`.
    pub xx: Vec<f64>,
    /// The entries `r22`.
    pub yy: Vec<f64>,
    /// The entries `r12`, counted twice by the inner product.
    pub xy: Vec<f64>,
}

impl Tensor {
    /// A field of zeros of `size` entries each.
    #[must_use]
    pub fn zeros(size: usize) -> Self {
        Self {
            xx: vec![0.0; size],
            yy: vec![0.0; size],
            xy: vec![0.0; size],
        }
    }
}

/// A dual point: the vector field `p`, and TGV's tensor field `r`.
#[derive(Debug, Clone, PartialEq)]
pub struct Dual {
    /// The dual of the first-order term.
    pub p: Vector,
    /// The dual of TGV's second-order term.
    pub r: Option<Tensor>,
}

/// `ceil(size / (8 ratio))`: a component's blocks, as libjpeg counts them.
fn blocks_of(size: usize, ratio: usize) -> usize {
    size.div_ceil(BLOCK * ratio)
}

impl Frame {
    /// A frame of these channels on a canvas of `height x width`.
    ///
    /// # Errors
    ///
    /// [`Error::Unsupported`] where there is no channel, cells do not tile the
    /// canvas, or blocks go beyond it.
    pub fn new(channels: Vec<Channel>, height: usize, width: usize) -> Result<Self, Error> {
        if channels.is_empty() {
            return Err(Error::Unsupported("a frame has a channel at least".into()));
        }
        for channel in &channels {
            let (down, across) = channel.ratio;
            let (rows, columns) = channel.extent();
            if down < 1 || across < 1 || !height.is_multiple_of(down) || !width.is_multiple_of(across) {
                return Err(Error::Unsupported(format!(
                    "cells of {down} x {across} do not tile a canvas of {height} x {width}"
                )));
            }
            if rows > height || columns > width {
                return Err(Error::Unsupported(format!(
                    "blocks over {rows} x {columns} samples go beyond a canvas of {height} x {width}"
                )));
            }
        }
        Ok(Self {
            channels,
            height,
            width,
        })
    }

    /// The frame of one component, whose canvas is its blocks: nothing is free.
    #[must_use]
    pub fn one(problem: Problem) -> Self {
        let (height, width) = (problem.height(), problem.width());
        Self {
            channels: vec![Channel::new(problem, (1, 1))],
            height,
            width,
        }
    }

    /// The frame of a file's components (docs/math.md, 1.3).
    ///
    /// `problems` are the components', `factors` their sampling factors `(h, v)`,
    /// across and down, and `size` the picture's rows and columns. One component's
    /// canvas is its blocks. Several share the picture rounded up to whole MCUs, and
    /// each needs the largest factors to be whole multiples of its own.
    ///
    /// # Errors
    ///
    /// [`Error::Unsupported`] for factors that are not whole multiples, or problems
    /// that do not have the blocks of the picture.
    pub fn of_file(problems: Vec<Problem>, factors: &[(usize, usize)], size: (usize, usize)) -> Result<Self, Error> {
        if problems.is_empty() || problems.len() != factors.len() {
            return Err(Error::Unsupported(format!(
                "a sampling factor for each of the components, not {} for {}",
                factors.len(),
                problems.len()
            )));
        }
        let mut problems = problems;
        if problems.len() == 1
            && let Some(problem) = problems.pop()
        {
            return Ok(Self::one(problem));
        }
        let (rows, columns) = size;
        let most_across = factors.iter().map(|&(across, _)| across).max().unwrap_or(1);
        let most_down = factors.iter().map(|&(_, down)| down).max().unwrap_or(1);
        let mut channels = Vec::with_capacity(problems.len());
        for (problem, &(across, down)) in problems.into_iter().zip(factors) {
            if across < 1 || down < 1 || !most_across.is_multiple_of(across) || !most_down.is_multiple_of(down) {
                return Err(Error::Unsupported(format!(
                    "the sampling factors {across} x {down} do not divide {most_across} x {most_down}: \
                     only whole cells are taken"
                )));
            }
            let ratio = (most_down / down, most_across / across);
            let blocks = (blocks_of(rows, ratio.0), blocks_of(columns, ratio.1));
            if (problem.rows(), problem.columns()) != blocks {
                return Err(Error::Unsupported(format!(
                    "a component of {} x {} blocks, where a picture of {rows} x {columns} has {} x {}",
                    problem.rows(),
                    problem.columns(),
                    blocks.0,
                    blocks.1
                )));
            }
            channels.push(Channel::new(problem, ratio));
        }
        let height = BLOCK * most_down * blocks_of(rows, most_down);
        let width = BLOCK * most_across * blocks_of(columns, most_across);
        Self::new(channels, height, width)
    }

    /// The channels.
    #[must_use]
    pub fn channels(&self) -> &[Channel] {
        &self.channels
    }

    /// Rows of the canvas.
    #[must_use]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Columns of the canvas.
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    /// The shape of one channel's canvas.
    #[must_use]
    pub fn shape(&self) -> Shape {
        Shape {
            height: self.height,
            width: self.width,
        }
    }

    /// The samples of one channel's canvas, `H W`.
    #[must_use]
    pub fn plane(&self) -> usize {
        self.height * self.width
    }

    /// The samples of all the channels' canvases, `C H W`.
    #[must_use]
    pub fn samples(&self) -> usize {
        self.channels.len() * self.plane()
    }

    /// Whether some of the channel's samples are free: cells of several samples, or
    /// samples beyond its blocks.
    #[must_use]
    pub fn free(&self, channel: &Channel) -> bool {
        channel.cells() > 1 || channel.extent() != (self.height, self.width)
    }

    /// The gamma of every channel: 1 for each where `None`.
    ///
    /// # Errors
    ///
    /// [`Error::Options`] unless there is a positive, finite weight for each channel.
    pub fn channel_weights(&self, weights: Option<&[f64]>) -> Result<Vec<f64>, Error> {
        let count = self.channels.len();
        let values = weights.map_or_else(|| vec![1.0; count], <[f64]>::to_vec);
        if values.len() != count || !values.iter().all(|&value| value > 0.0 && value.is_finite()) {
            return Err(Error::Options(format!(
                "a positive, finite weight for each of the {count} channels, not {values:?}"
            )));
        }
        Ok(values)
    }
}

/// The sums of the cells of a channel's canvas over the component's blocks, the
/// component's samples times `n`: `nu^-1 E S x` (docs/math.md, 4.4). Each sum adds
/// its cell row by row.
#[must_use]
pub fn sums(channel: &Channel, plane: &[f64], width: usize) -> Vec<f64> {
    let (down, across) = channel.ratio;
    let (rows, columns) = (channel.problem.height(), channel.problem.width());
    let mut result = vec![0.0; rows * columns];
    for (i, line) in result.chunks_exact_mut(columns).enumerate() {
        for (j, entry) in line.iter_mut().enumerate() {
            let mut total = 0.0;
            for a in 0..down {
                let start = (i * down + a) * width + j * across;
                for &value in &plane[start..start + across] {
                    total += value;
                }
            }
            *entry = total;
        }
    }
    result
}

/// `E S x`: the means of the cells of a channel's canvas over the component's blocks,
/// the component's samples.
#[must_use]
pub fn means(channel: &Channel, plane: &[f64], width: usize) -> Vec<f64> {
    let mut result = sums(channel, plane, width);
    let cells = channel.cells();
    if cells > 1 {
        let count = nearest_from_usize(cells);
        for value in &mut result {
            *value /= count;
        }
    }
    result
}

/// The component's samples on a channel's canvas: each repeated over its cell, and 0
/// beyond the blocks. This is `nu^-1 S^T E^T`: with the inverse DCT before it,
/// `nu^-1 A^T`.
#[must_use]
pub fn spread(frame: &Frame, channel: &Channel, samples: &[f64]) -> Vec<f64> {
    let (down, across) = channel.ratio;
    let columns = channel.problem.width();
    let (rows_covered, columns_covered) = channel.extent();
    let mut result = vec![0.0; frame.plane()];
    for (i, line) in result.chunks_exact_mut(frame.width).take(rows_covered).enumerate() {
        let source = &samples[(i / down) * columns..(i / down + 1) * columns];
        for (j, entry) in line[..columns_covered].iter_mut().enumerate() {
            *entry = source[j / across];
        }
    }
    result
}

/// The factors of the proximal map of every channel with the step `tau`: each
/// component's coefficients take the step `tau / n` (docs/math.md, 4.4).
#[must_use]
pub fn steps(frame: &Frame, tau: f64) -> Vec<Step> {
    frame
        .channels
        .iter()
        .map(|channel| channel.problem.step(tau / nearest_from_usize(channel.cells())))
        .collect()
}

/// `prox_{tau G}(v)` of a canvas `C x H x W`, with the factors of [`steps`]: every
/// component's coefficients into `coefficients`, and the canvas into `out`
/// (docs/math.md, 4.4).
///
/// The coefficients take the proximal map of the model with the step `tau / n`,
/// and the canvas moves by their change, repeated over the cells; its free samples
/// stay. Where nothing is free, the canvas is the inverse DCT of the coefficients.
pub fn prox(frame: &Frame, steps: &[Step], canvas: &[f64], coefficients: &mut [Vec<f64>], out: &mut [f64]) {
    let plane = frame.plane();
    for (index, (channel, step)) in frame.channels.iter().zip(steps).enumerate() {
        let source = &canvas[index * plane..(index + 1) * plane];
        let target = &mut out[index * plane..(index + 1) * plane];
        put(
            frame,
            channel,
            source,
            target,
            &mut coefficients[index],
            |block, values| {
                channel.problem.prox_block(step, block, values);
            },
        );
    }
}

/// The projection of a canvas `C x H x W` onto the quantization constraint set
/// (docs/math.md, 4.4): the proximal map with `G` the constraint alone.
#[must_use]
pub fn project(frame: &Frame, canvas: &[f64]) -> Vec<f64> {
    let plane = frame.plane();
    let mut out = vec![0.0; canvas.len()];
    for (index, channel) in frame.channels.iter().enumerate() {
        let source = &canvas[index * plane..(index + 1) * plane];
        let target = &mut out[index * plane..(index + 1) * plane];
        let mut coefficients = vec![0.0; channel.problem.samples()];
        put(frame, channel, source, target, &mut coefficients, |block, values| {
            let start = block * BLOCK_SIZE;
            for (offset, value) in values.iter_mut().enumerate() {
                *value = channel.problem.clip_one(start + offset, *value);
            }
        });
    }
    out
}

/// `v + nu^-1 A^T (zeta - A v)` for one channel, where `zeta` is what `map` makes of
/// the coefficients `A v` of each block: `map` gets the block's index and its
/// coefficients, and changes them in place; they are kept in `coefficients`.
fn put(
    frame: &Frame,
    channel: &Channel,
    source: &[f64],
    target: &mut [f64],
    coefficients: &mut [f64],
    mut map: impl FnMut(usize, &mut [f64]),
) {
    let width = frame.width;
    let (rows_covered, columns_covered) = channel.extent();
    let (block_rows, blocks_across) = (channel.problem.rows(), channel.problem.columns());
    if channel.cells() == 1 {
        dct::forward_into(source, width, block_rows, blocks_across, coefficients);
        for (block, values) in coefficients.as_chunks_mut::<BLOCK_SIZE>().0.iter_mut().enumerate() {
            map(block, values);
        }
        dct::inverse_into(coefficients, block_rows, blocks_across, target, width);
    } else {
        let own = means(channel, source, width);
        let columns = channel.problem.width();
        let mut change = vec![0.0; own.len()];
        dct::forward_into(&own, columns, block_rows, blocks_across, coefficients);
        for (block, values) in coefficients.as_chunks_mut::<BLOCK_SIZE>().0.iter_mut().enumerate() {
            map(block, values);
        }
        dct::inverse_into(coefficients, block_rows, blocks_across, &mut change, columns);
        for (value, &mean) in change.iter_mut().zip(&own) {
            *value -= mean;
        }
        let (down, across) = channel.ratio;
        for i in 0..rows_covered {
            let line = &change[(i / down) * columns..(i / down + 1) * columns];
            let start = i * width;
            for (j, (entry, &value)) in target[start..start + columns_covered]
                .iter_mut()
                .zip(&source[start..start + columns_covered])
                .enumerate()
            {
                *entry = value + line[j / across];
            }
        }
    }
    // The free samples beyond the blocks stay as they are.
    if (rows_covered, columns_covered) != (frame.height, width) {
        for i in 0..frame.height {
            let start = i * width;
            let from = if i < rows_covered { columns_covered } else { 0 };
            target[start + from..start + width].copy_from_slice(&source[start + from..start + width]);
        }
    }
}

/// How far each component's coefficients of a canvas lie outside their intervals,
/// in steps (0 inside).
#[must_use]
pub fn excess(frame: &Frame, canvas: &[f64]) -> Vec<Vec<f64>> {
    let plane = frame.plane();
    frame
        .channels
        .iter()
        .enumerate()
        .map(|(index, channel)| {
            let own = means(channel, &canvas[index * plane..(index + 1) * plane], frame.width);
            let problem = &channel.problem;
            problem.excess(&dct::forward(&own, problem.height(), problem.width()))
        })
        .collect()
}

/// The starting point: these coefficients, or else the data term's centres, clipped
/// to their intervals (docs/math.md, 5).
///
/// Each component's canvas is the inverse DCT of its coefficients, repeated over the
/// cells, and [`FREE_CENTRE`] beyond its blocks.
///
/// # Errors
///
/// [`Error::Options`] for coefficients given that are not finite, or not of every
/// component's size.
pub fn start(frame: &Frame, coefficients: Option<&[Vec<f64>]>) -> Result<Primal, Error> {
    if let Some(given) = coefficients
        && given.len() != frame.channels.len()
    {
        return Err(Error::Options(format!(
            "coefficients to start from for each of the {} components",
            frame.channels.len()
        )));
    }
    let plane = frame.plane();
    let mut chosen = Vec::with_capacity(frame.channels.len());
    let mut canvas = vec![0.0; frame.samples()];
    for (index, channel) in frame.channels.iter().enumerate() {
        let problem = &channel.problem;
        let mut values = match coefficients {
            None => problem.centres().to_vec(),
            Some(given) => {
                let own = &given[index];
                if own.len() != problem.samples() || !own.iter().all(|value| value.is_finite()) {
                    return Err(Error::Options(format!(
                        "the coefficients to start from are finite, {} of them",
                        problem.samples()
                    )));
                }
                own.clone()
            }
        };
        problem.clip(&mut values);
        let samples = dct::inverse(&values, problem.rows(), problem.columns());
        let target = &mut canvas[index * plane..(index + 1) * plane];
        if frame.free(channel) {
            let spread = spread(frame, channel, &samples);
            let (rows_covered, columns_covered) = channel.extent();
            for (i, (line, source)) in target
                .chunks_exact_mut(frame.width)
                .zip(spread.chunks_exact(frame.width))
                .enumerate()
            {
                for (j, (entry, &value)) in line.iter_mut().zip(source).enumerate() {
                    *entry = if i < rows_covered && j < columns_covered {
                        value
                    } else {
                        FREE_CENTRE
                    };
                }
            }
        } else {
            target.copy_from_slice(&samples);
        }
        chosen.push(values);
    }
    Ok(Primal {
        coefficients: chosen,
        canvas,
        w: None,
    })
}

/// `G` at coefficients within their intervals: the sum of the components' data terms.
#[must_use]
pub fn data_term(frame: &Frame, coefficients: &[Vec<f64>]) -> f64 {
    frame
        .channels
        .iter()
        .zip(coefficients)
        .fold(0.0, |value, (channel, own)| value + channel.problem.data_term(own))
}

/// `G*(xi)` of a canvas `xi`, `C x H x W`, bounded over the box of the radius where
/// samples are free (docs/math.md, 6.6).
///
/// For each component, `g*(nu^-1 A xi)`, and where it has free samples, the most that
/// `<zeta, x - Pi x>` reaches over the box: `<zeta, m> + radius ||zeta||_1`,
/// `zeta = xi - Pi xi`. Where nothing is free this is `G*` itself.
#[must_use]
pub fn conjugate(frame: &Frame, xi: &[f64], radius: f64) -> f64 {
    let plane = frame.plane();
    let mut value = 0.0;
    for (index, channel) in frame.channels.iter().enumerate() {
        let own = &xi[index * plane..(index + 1) * plane];
        let problem = &channel.problem;
        let summed = sums(channel, own, frame.width);
        value += problem.conjugate(&dct::forward(&summed, problem.height(), problem.width()));
        if frame.free(channel) {
            let averaged = spread(frame, channel, &means(channel, own, frame.width));
            let (rows_covered, columns_covered) = channel.extent();
            let (mut beyond, mut magnitude) = (Sum::new(), Sum::new());
            for (i, (line, mean)) in own
                .chunks_exact(frame.width)
                .zip(averaged.chunks_exact(frame.width))
                .enumerate()
            {
                for (j, (&sample, &average)) in line.iter().zip(mean).enumerate() {
                    let zeta = sample - average;
                    if i >= rows_covered || j >= columns_covered {
                        beyond.add(zeta);
                    }
                    magnitude.add(zeta.abs());
                }
            }
            value += FREE_CENTRE * beyond.total() + radius * magnitude.total();
        }
    }
    value
}

/// The forward differences of a plane at a sample, across and down (0 at the last
/// column and row).
#[inline]
fn differences(plane: &[f64], shape: Shape, i: usize, j: usize) -> (f64, f64) {
    let index = i * shape.width + j;
    let across = if j + 1 < shape.width {
        plane[index + 1] - plane[index]
    } else {
        0.0
    };
    let down = if i + 1 < shape.height {
        plane[index + shape.width] - plane[index]
    } else {
        0.0
    };
    (across, down)
}

/// The sum over the pixels of the norm of a field of every channel, computed pixel
/// by pixel by `square`, which gives the square of a channel's entry at a sample:
/// coupled, the norm of all the channels' entries at a pixel; apart, each channel's,
/// summed channel by channel, and the channels' sums added in their order, so that
/// the channels apart are the sum of their own totals.
fn total_norm(frame: &Frame, coupled: bool, mut square: impl FnMut(usize, usize, usize) -> f64) -> f64 {
    let count = frame.channels.len();
    if coupled {
        let mut total = Sum::new();
        for i in 0..frame.height {
            for j in 0..frame.width {
                let added = (0..count).fold(0.0, |added, channel| added + square(channel, i, j));
                total.add(added.sqrt());
            }
        }
        return total.total();
    }
    let mut sums = (0..count).map(|channel| {
        let mut total = Sum::new();
        for i in 0..frame.height {
            for j in 0..frame.width {
                total.add(square(channel, i, j).sqrt());
            }
        }
        total.total()
    });
    let first = sums.next().unwrap_or(0.0);
    sums.fold(first, |total, channel| total + channel)
}

/// The total variation of a canvas of several channels, `||gamma grad x||` in the norm
/// of the channels, without its weight (docs/math.md, 4.4).
///
/// # Errors
///
/// [`Error::Options`] for channel weights that the frame does not take.
pub fn variation(frame: &Frame, weights: &Tv, canvas: &[f64]) -> Result<f64, Error> {
    let gammas = frame.channel_weights(weights.channel_weights.as_deref())?;
    let shape = frame.shape();
    let plane = frame.plane();
    Ok(total_norm(frame, weights.coupled, |channel, i, j| {
        let (across, down) = differences(&canvas[channel * plane..(channel + 1) * plane], shape, i, j);
        vector_square(gammas[channel] * across, gammas[channel] * down)
    }))
}

/// `P(x)` of the TV model of several channels (docs/math.md, 4.4), at a point of
/// these coefficients and this canvas.
///
/// # Errors
///
/// [`Error::Options`] for channel weights that the frame does not take.
pub fn tv_objective(frame: &Frame, weights: &Tv, coefficients: &[Vec<f64>], canvas: &[f64]) -> Result<f64, Error> {
    Ok(weights.alpha * variation(frame, weights, canvas)? + data_term(frame, coefficients))
}

/// The divergence of a vector field of every channel, each channel times its weight.
fn weighted_divergence(frame: &Frame, gammas: &[f64], field: &Vector) -> Vec<f64> {
    let plane = frame.plane();
    let shape = frame.shape();
    let mut result = vec![0.0; frame.samples()];
    for (channel, &gamma) in gammas.iter().enumerate() {
        let range = channel * plane..(channel + 1) * plane;
        let target = &mut result[range.clone()];
        operators::div(&field.x[range.clone()], &field.y[range], shape, target);
        for value in target {
            *value *= gamma;
        }
    }
    result
}

/// The primal value of the TV model, and the dual value for a `p` within the dual
/// balls (docs/math.md, 6.1 and 6.6). Their difference is the gap, or where samples
/// are free, the partial gap of the radius.
///
/// # Errors
///
/// [`Error::Options`] for channel weights that the frame does not take.
pub fn tv_values(
    frame: &Frame,
    weights: &Tv,
    coefficients: &[Vec<f64>],
    canvas: &[f64],
    p: &Vector,
    radius: f64,
) -> Result<(f64, f64), Error> {
    let gammas = frame.channel_weights(weights.channel_weights.as_deref())?;
    let primal = tv_objective(frame, weights, coefficients, canvas)?;
    let xi = weighted_divergence(frame, &gammas, p);
    Ok((primal, -conjugate(frame, &xi, radius)))
}

/// The symmetrized gradient of a vector field, channel by channel.
#[must_use]
pub fn symmetrized(frame: &Frame, field: &Vector) -> Tensor {
    let plane = frame.plane();
    let shape = frame.shape();
    let mut result = Tensor::zeros(frame.samples());
    for channel in 0..frame.channels.len() {
        let range = channel * plane..(channel + 1) * plane;
        operators::sym_grad(
            &field.x[range.clone()],
            &field.y[range.clone()],
            shape,
            &mut result.xx[range.clone()],
            &mut result.yy[range.clone()],
            &mut result.xy[range],
        );
    }
    result
}

/// The gradient of a canvas, channel by channel.
#[must_use]
pub fn gradient(frame: &Frame, canvas: &[f64]) -> Vector {
    let plane = frame.plane();
    let shape = frame.shape();
    let mut result = Vector::zeros(frame.samples());
    for channel in 0..frame.channels.len() {
        let range = channel * plane..(channel + 1) * plane;
        operators::grad(
            &canvas[range.clone()],
            shape,
            &mut result.x[range.clone()],
            &mut result.y[range],
        );
    }
    result
}

/// The divergence of a tensor field, channel by channel.
#[must_use]
pub fn tensor_divergence(frame: &Frame, field: &Tensor) -> Vector {
    let plane = frame.plane();
    let shape = frame.shape();
    let mut result = Vector::zeros(frame.samples());
    for channel in 0..frame.channels.len() {
        let range = channel * plane..(channel + 1) * plane;
        operators::div2(
            &field.xx[range.clone()],
            &field.yy[range.clone()],
            &field.xy[range.clone()],
            shape,
            &mut result.x[range.clone()],
            &mut result.y[range],
        );
    }
    result
}

/// `P(x, w)` of the TGV model of several channels (docs/math.md, 4.4), at a point
/// of these coefficients, this canvas and this field `w`.
///
/// # Errors
///
/// [`Error::Options`] for channel weights that the frame does not take.
pub fn tgv_objective(
    frame: &Frame,
    weights: &Tgv,
    coefficients: &[Vec<f64>],
    canvas: &[f64],
    w: &Vector,
) -> Result<f64, Error> {
    let gammas = frame.channel_weights(weights.channel_weights.as_deref())?;
    let shape = frame.shape();
    let plane = frame.plane();
    let first = total_norm(frame, weights.coupled, |channel, i, j| {
        let (across, down) = differences(&canvas[channel * plane..(channel + 1) * plane], shape, i, j);
        let index = channel * plane + i * frame.width + j;
        let gamma = gammas[channel];
        vector_square(gamma * (across - w.x[index]), gamma * (down - w.y[index]))
    });
    let e = symmetrized(frame, w);
    let second = total_norm(frame, weights.coupled, |channel, i, j| {
        let index = channel * plane + i * frame.width + j;
        let gamma = gammas[channel];
        tensor_square(gamma * e.xx[index], gamma * e.yy[index], gamma * e.xy[index])
    });
    Ok(weights.alpha1 * first + weights.alpha0 * second + data_term(frame, coefficients))
}

/// The primal value, a dual value, and the scaling that made the dual feasible
/// (docs/math.md, 6.2 and 6.6), at a point of these coefficients, canvas and `w`,
/// with the dual's `r`.
///
/// `r` is scaled by `theta`, the most that keeps `div2 r` within `alpha1` in the norm
/// of the channels, and the dual is taken at `p = -div2 (theta r)`.
///
/// # Errors
///
/// [`Error::Options`] for channel weights that the frame does not take.
pub fn tgv_values(
    frame: &Frame,
    weights: &Tgv,
    coefficients: &[Vec<f64>],
    canvas: &[f64],
    w: &Vector,
    r: &Tensor,
    radius: f64,
) -> Result<(f64, f64, f64), Error> {
    let gammas = frame.channel_weights(weights.channel_weights.as_deref())?;
    let mut divergence = tensor_divergence(frame, r);
    let plane = frame.plane();
    let count = frame.channels.len();
    let mut largest: f64 = 0.0;
    if weights.coupled {
        for pixel in 0..plane {
            let added = (0..count).fold(0.0, |added, channel| {
                let index = channel * plane + pixel;
                added + vector_square(divergence.x[index], divergence.y[index])
            });
            largest = largest.max(added.sqrt());
        }
    } else {
        for (&x, &y) in divergence.x.iter().zip(&divergence.y) {
            largest = largest.max(vector_square(x, y).sqrt());
        }
    }
    let theta = if largest <= weights.alpha1 {
        1.0
    } else {
        weights.alpha1 / largest
    };
    for values in [&mut divergence.x, &mut divergence.y] {
        for value in values.iter_mut() {
            *value *= -theta;
        }
    }
    let xi = weighted_divergence(frame, &gammas, &divergence);
    let primal = tgv_objective(frame, weights, coefficients, canvas, w)?;
    Ok((primal, -conjugate(frame, &xi, radius), theta))
}

/// The sum over the channels of `gamma_c ||p_c + div2 r_c||_{2,1}`: what the partial
/// gap of 6.3 adds, over its radius.
///
/// # Errors
///
/// [`Error::Options`] for channel weights that the frame does not take.
pub fn tgv_residual(frame: &Frame, weights: &Tgv, p: &Vector, r: &Tensor) -> Result<f64, Error> {
    let gammas = frame.channel_weights(weights.channel_weights.as_deref())?;
    let divergence = tensor_divergence(frame, r);
    let plane = frame.plane();
    Ok(total_norm(frame, false, |channel, i, j| {
        let index = channel * plane + i * frame.width + j;
        let gamma = gammas[channel];
        vector_square(
            gamma * (p.x[index] + divergence.x[index]),
            gamma * (p.y[index] + divergence.y[index]),
        )
    }))
}

/// `G*(gamma div p)`, over the box of free samples of the radius (6.6): with the
/// primal value and the radius of 6.3 times [`tgv_residual`], TGV's partial gap.
///
/// # Errors
///
/// [`Error::Options`] for channel weights that the frame does not take.
pub fn dual_bound(frame: &Frame, channel_weights: Option<&[f64]>, p: &Vector, radius: f64) -> Result<f64, Error> {
    let gammas = frame.channel_weights(channel_weights)?;
    Ok(conjugate(frame, &weighted_divergence(frame, &gammas, p), radius))
}

/// Each vector of a field projected onto the ball of the radius, over all the
/// channels at a pixel where coupled.
pub fn project_vectors(frame: &Frame, field: &mut Vector, radius: f64, coupled: bool) {
    let plane = frame.plane();
    let count = frame.channels.len();
    if coupled {
        for pixel in 0..plane {
            let added = (0..count).fold(0.0, |added, channel| {
                let index = channel * plane + pixel;
                added + vector_square(field.x[index], field.y[index])
            });
            let scale = kernels::ball_scale(added, radius);
            for channel in 0..count {
                let index = channel * plane + pixel;
                field.x[index] *= scale;
                field.y[index] *= scale;
            }
        }
    } else {
        for (x, y) in field.x.iter_mut().zip(field.y.iter_mut()) {
            let scale = kernels::ball_scale(vector_square(*x, *y), radius);
            *x *= scale;
            *y *= scale;
        }
    }
}

/// Each tensor of a field projected onto the Frobenius ball of the radius, over all
/// the channels at a pixel where coupled.
pub fn project_tensors(frame: &Frame, field: &mut Tensor, radius: f64, coupled: bool) {
    let plane = frame.plane();
    let count = frame.channels.len();
    if coupled {
        for pixel in 0..plane {
            let added = (0..count).fold(0.0, |added, channel| {
                let index = channel * plane + pixel;
                added + tensor_square(field.xx[index], field.yy[index], field.xy[index])
            });
            let scale = kernels::ball_scale(added, radius);
            for channel in 0..count {
                let index = channel * plane + pixel;
                field.xx[index] *= scale;
                field.yy[index] *= scale;
                field.xy[index] *= scale;
            }
        }
    } else {
        for ((xx, yy), xy) in field.xx.iter_mut().zip(field.yy.iter_mut()).zip(field.xy.iter_mut()) {
            let scale = kernels::ball_scale(tensor_square(*xx, *yy, *xy), radius);
            *xx *= scale;
            *yy *= scale;
            *xy *= scale;
        }
    }
}
