// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! Pictures of rounded samples -- 8-bit PNG files, say -- reconstructed within the
//! intervals of their samples by TV or TGV (docs/math.md, 11).
//!
//! A sample of level `v` of a file of `b` bits stands for every value within half a
//! level of `v s`, `s = 255 / (2^b - 1)`, within 0 to 255. The unknown is the picture
//! itself, `C x H x W` values, channel by channel, each row by row; nothing is free.
//! The methods are those of [`crate::pdhg`], with its options and defaults but those
//! of when they stop and TGV's ratio of the steps, computed on the picture in its
//! natural layout, a channel's plane at a time.

use crate::dct::multiply_add;
use crate::error::Error;
use crate::exact::{Ratio, Sum, added, lanes, nearest_from_usize};
use crate::frames::{Tensor, Vector};
use crate::kernels::{self, at_least, at_most};
use crate::model::{Tgv, Tv};
use crate::operators::{Shape, backward_x, backward_y, forward_x, forward_y, tensor_square, vector_square};
use crate::pdhg::{self, Observer, Options, Plan, Record, Weights};
use crate::results::{History, Recorder, Stop};

/// The gap per sample at which TV and TGV stop on a picture of rounded samples, times
/// `alpha` of TV or `alpha1` of TGV (docs/math.md, 11): a fifth of TV's on JPEG
/// files and a fiftieth of TGV's, for the bands of a dim gradient, tens of samples
/// wide, which the methods smooth slowly.
pub const TOLERANCE: f64 = 1e-4;
/// The most iterations of TV and TGV on a picture of rounded samples.
pub const ITERATIONS: u64 = 10_000;
/// `tau / sigma` for TV on a picture of rounded samples, over `alpha` squared: that
/// of JPEG files.
pub const TV_RATIO: f64 = pdhg::TV_RATIO;
/// `tau / sigma` for TGV on a picture of rounded samples, over `alpha1` squared: a
/// tenth of that of JPEG files (docs/math.md, 11). Within the narrow intervals of
/// the samples the primal steps move little, and the dual ones carry the smoothing
/// across a band.
pub const TGV_RATIO: f64 = 0.3;

/// The options with the defaults of a picture of rounded samples in place of
/// `None`: [`TOLERANCE`] times the first-order weight `alpha`, [`ITERATIONS`], and
/// the ratio of the steps over `alpha` squared; the tolerance and the ratio as they
/// are where the defaults do not follow the weight.
fn defaults(options: &Options, alpha: f64, ratio: f64) -> Options {
    let scale = if options.scale_with_weight { alpha } else { 1.0 };
    Options {
        iterations: options.iterations.or(Some(ITERATIONS)),
        tolerance: options.tolerance.or(Some(TOLERANCE * scale)),
        step_ratio: options.step_ratio.or(Some(ratio / (scale * scale))),
        ..*options
    }
}

/// The data term of a picture of rounded samples: its weight, and the slack of the
/// intervals in levels (docs/math.md, 11).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Data {
    /// `mu`, the weight of the data term: 0 by default, the least TV or TGV within the
    /// intervals.
    pub mu: f64,
    /// The slack, in levels, that widens every interval on each side: 0 by default.
    pub slack: f64,
}

/// A picture of rounded samples: the intervals and centres of its samples, `C x H x
/// W`, and the weight of its data term.
#[derive(Debug, Clone)]
pub struct Model {
    channels: usize,
    shape: Shape,
    low: Vec<f64>,
    high: Vec<f64>,
    centre: Vec<f64>,
    mu: f64,
}

impl Model {
    /// The model of a picture of `channels x height x width` levels of `bits` bits
    /// (1, 2, 4, 8 or 16), channel by channel, each row by row.
    ///
    /// # Errors
    ///
    /// [`Error::Options`] for a picture that is empty or not of that size, bits or
    /// levels out of their range, and a weight or a slack that is negative or not
    /// finite.
    pub fn new(levels: &[u16], size: (usize, usize, usize), bits: u32, data: &Data) -> Result<Self, Error> {
        let (channels, height, width) = size;
        let count = channels.checked_mul(height).and_then(|plane| plane.checked_mul(width));
        if count != Some(levels.len()) || levels.is_empty() {
            return Err(Error::Options(format!(
                "a picture of {channels} x {height} x {width} levels, not empty, and not of {}",
                levels.len()
            )));
        }
        if ![1, 2, 4, 8, 16].contains(&bits) {
            return Err(Error::Options(format!("samples of 1, 2, 4, 8 or 16 bits, not {bits}")));
        }
        if !(data.mu >= 0.0 && data.mu.is_finite() && data.slack >= 0.0 && data.slack.is_finite()) {
            return Err(Error::Options(format!(
                "the weight of the data term and the slack are at least 0 and finite, not {} and {}",
                data.mu, data.slack
            )));
        }
        let top = (1u32 << bits) - 1;
        if let Some(&level) = levels.iter().find(|&&level| u32::from(level) > top) {
            return Err(Error::Options(format!(
                "a level of {bits} bits is at most {top}, not {level}"
            )));
        }
        let tables = Tables::new(top, data.slack);
        let lookup = |table: &[f64]| -> Vec<f64> { levels.iter().map(|&level| table[usize::from(level)]).collect() };
        Ok(Self {
            channels,
            shape: Shape { height, width },
            low: lookup(&tables.low),
            high: lookup(&tables.high),
            centre: lookup(&tables.centre),
            mu: data.mu,
        })
    }

    /// The channels.
    #[must_use]
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Rows.
    #[must_use]
    pub fn height(&self) -> usize {
        self.shape.height
    }

    /// Columns.
    #[must_use]
    pub fn width(&self) -> usize {
        self.shape.width
    }

    /// The lower ends of the intervals, `C x H x W`.
    #[must_use]
    pub fn low(&self) -> &[f64] {
        &self.low
    }

    /// The upper ends of the intervals.
    #[must_use]
    pub fn high(&self) -> &[f64] {
        &self.high
    }

    /// The centres: the levels on the scale of 0 to 255.
    #[must_use]
    pub fn centres(&self) -> &[f64] {
        &self.centre
    }

    /// `mu`.
    #[must_use]
    pub fn mu(&self) -> f64 {
        self.mu
    }

    fn plane(&self) -> usize {
        self.shape.size()
    }

    /// `prox_{tau G}(v)` of samples from `offset` on into `out`: `clip((v + t c) r, a,
    /// b)`, `t = tau mu` and `r = 1 / (1 + t)` (docs/math.md, 11).
    pub fn prox(&self, values: &[f64], tau: f64, offset: usize, out: &mut [f64]) {
        let t = tau * self.mu;
        let r = 1.0 / (1.0 + t);
        let span = offset..offset + values.len();
        let (centres, lows, highs) = (&self.centre[span.clone()], &self.low[span.clone()], &self.high[span]);
        for ((((target, &value), &centre), &low), &high) in out.iter_mut().zip(values).zip(centres).zip(lows).zip(highs)
        {
            *target = clip(multiply_add(t, centre, value) * r, low, high);
        }
    }

    /// The projection onto the intervals: the proximal map of the constraint alone.
    #[must_use]
    pub fn project(&self, values: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; values.len()];
        for (((target, &value), &low), &high) in out.iter_mut().zip(values).zip(&self.low).zip(&self.high) {
            *target = clip(value, low, high);
        }
        out
    }

    /// How far each sample lies outside its interval (0 inside).
    #[must_use]
    pub fn excess(&self, values: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; values.len()];
        for (((target, &value), &low), &high) in out.iter_mut().zip(values).zip(&self.low).zip(&self.high) {
            *target = at_least(at_least(low - value, value - high), 0.0);
        }
        out
    }

    /// `G(x)` at samples within their intervals: `mu / 2` times the sum of the squares
    /// of their distances from the centres, each row in lanes.
    #[must_use]
    pub fn value(&self, x: &[f64]) -> f64 {
        if self.mu == 0.0 {
            return 0.0;
        }
        let width = self.shape.width;
        let mut total = Sum::new();
        let mut terms = vec![0.0; width];
        for (row, centre) in x.chunks_exact(width).zip(self.centre.chunks_exact(width)) {
            for ((term, &value), &centre) in terms.iter_mut().zip(row).zip(centre) {
                let away = value - centre;
                *term = away * away;
            }
            total.add(lanes(&terms));
        }
        0.5 * self.mu * total.total()
    }

    /// `G*(xi)`, the sum of each sample's `g*` (docs/math.md, 11), each row in lanes.
    #[must_use]
    pub fn conjugate(&self, xi: &[f64]) -> f64 {
        let width = self.shape.width;
        let mu = self.mu;
        let mut total = Sum::new();
        let mut terms = vec![0.0; width];
        for (((row, centre), low), high) in xi
            .chunks_exact(width)
            .zip(self.centre.chunks_exact(width))
            .zip(self.low.chunks_exact(width))
            .zip(self.high.chunks_exact(width))
        {
            let (row, centre, low, high) = (&row[..width], &centre[..width], &low[..width], &high[..width]);
            for (k, term) in terms[..width].iter_mut().enumerate() {
                *term = conjugate_term(row[k], mu, centre[k], low[k], high[k]);
            }
            total.add(lanes(&terms));
        }
        total.total()
    }
}

/// The ends and the centre of every level of a file.
struct Tables {
    low: Vec<f64>,
    high: Vec<f64>,
    centre: Vec<f64>,
}

impl Tables {
    /// For levels 0 to `top`: `max(0, (v - 1/2 - slack) s)`, `min(255, (v + 1/2 + slack)
    /// s)` and `v s`, `s = 255 / top`. Without slack, each is a rational, rounded once to
    /// the nearest double; with it, the ends are computed in floating point.
    fn new(top: u32, slack: f64) -> Self {
        let wide = i128::from(top);
        let step = Ratio::new(255, wide).nearest();
        let (mut low, mut high, mut centre) = (Vec::new(), Vec::new(), Vec::new());
        for level in 0..=top {
            let value = i128::from(level);
            let (below, above) = if slack == 0.0 {
                (
                    Ratio::new((2 * value - 1) * 255, 2 * wide).nearest(),
                    Ratio::new((2 * value + 1) * 255, 2 * wide).nearest(),
                )
            } else {
                let own = f64::from(level);
                (((own - 0.5) - slack) * step, ((own + 0.5) + slack) * step)
            };
            low.push(at_least(below, 0.0));
            high.push(at_most(above, 255.0));
            centre.push(Ratio::new(value * 255, wide).nearest());
        }
        Self { low, high, centre }
    }
}

/// `value` within `low..=high`, by selections, which vector code makes of a comparison
/// and a blend.
#[inline]
fn clip(value: f64, low: f64, high: f64) -> f64 {
    at_most(at_least(value, low), high)
}

/// `g*(s)` of a sample (docs/math.md, 11): that of a coefficient of 4.1, with the
/// weight `mu`, its interval and its centre ([`kernels::conjugate_term`]).
#[inline]
fn conjugate_term(s: f64, mu: f64, centre: f64, low: f64, high: f64) -> f64 {
    kernels::conjugate_term::<false>(s, mu, (low, high), centre, 0.0, (low, high))
}

/// What the solvers return: the last point -- the picture, within its intervals, and
/// TGV's field -- the last dual point, how many iterations they took, why they
/// stopped, and what they recorded.
#[derive(Debug, Clone)]
pub struct Solution {
    /// The picture, `C x H x W`: the output of the last proximal step.
    pub picture: Vec<f64>,
    /// TGV's vector field.
    pub w: Option<Vector>,
    /// The dual of the first-order term.
    pub p: Vector,
    /// TGV's dual of the second-order term.
    pub r: Option<Tensor>,
    /// The iterations taken.
    pub iterations: u64,
    /// Why the solver stopped.
    pub stop: Stop,
    /// The records.
    pub history: History,
}

/// The differences of one channel's plane, with room of their own.
#[derive(Debug, Clone)]
struct Room {
    shape: Shape,
    first: Vec<f64>,
    second: Vec<f64>,
}

impl Room {
    fn new(shape: Shape) -> Self {
        Self {
            shape,
            first: vec![0.0; shape.size()],
            second: vec![0.0; shape.size()],
        }
    }

    /// `div p`, `-grad` transposed, into `out`.
    fn divergence(&mut self, px: &[f64], py: &[f64], out: &mut [f64]) {
        backward_x(px, self.shape, out);
        backward_y(py, self.shape, &mut self.first);
        add_into(out, &self.first);
    }

    /// `E w` by backward differences, into `(r11, r22, r12)`.
    fn symmetrized(&mut self, wx: &[f64], wy: &[f64], r11: &mut [f64], r22: &mut [f64], r12: &mut [f64]) {
        backward_x(wx, self.shape, r11);
        backward_y(wy, self.shape, r22);
        backward_y(wx, self.shape, r12);
        backward_x(wy, self.shape, &mut self.first);
        for (entry, &value) in r12.iter_mut().zip(&self.first) {
            *entry = f64::midpoint(*entry, value);
        }
    }

    /// `div2 r`, `-E` transposed, by forward differences, into `(out_x, out_y)`.
    fn tensor_divergence(&mut self, r11: &[f64], r22: &[f64], r12: &[f64], out_x: &mut [f64], out_y: &mut [f64]) {
        forward_x(r11, self.shape, out_x);
        forward_y(r12, self.shape, &mut self.first);
        add_into(out_x, &self.first);
        forward_x(r12, self.shape, out_y);
        forward_y(r22, self.shape, &mut self.second);
        add_into(out_y, &self.second);
    }
}

fn add_into(out: &mut [f64], other: &[f64]) {
    for (value, &own) in out.iter_mut().zip(other) {
        *value += own;
    }
}

/// `rho new + (1 - rho) old` into `old`.
fn relax(old: &mut [f64], new: &[f64], rho: f64) {
    let rest = 1.0 - rho;
    for (value, &fresh) in old.iter_mut().zip(new) {
        *value = multiply_add(rho, fresh, rest * *value);
    }
}

/// `2 new - old` into `out`.
fn extrapolate(out: &mut [f64], new: &[f64], old: &[f64]) {
    for ((value, &fresh), &stale) in out.iter_mut().zip(new).zip(old) {
        *value = 2.0 * fresh - stale;
    }
}

/// `p + step g` into `out`.
fn ascend(out: &mut [f64], p: &[f64], g: &[f64], step: f64) {
    for ((value, &own), &difference) in out.iter_mut().zip(p).zip(g) {
        *value = multiply_add(step, difference, own);
    }
}

/// `p + step (g - bar)` into `out`.
fn ascend_less(out: &mut [f64], p: &[f64], g: &[f64], bar: &[f64], step: f64) {
    for (((value, &own), &difference), &less) in out.iter_mut().zip(p).zip(g).zip(bar) {
        *value = multiply_add(step, difference - less, own);
    }
}

/// `squares + weight (x^2 + y^2)`.
fn add_vector_squares(squares: &mut [f64], x: &[f64], y: &[f64], weight: f64) {
    for ((square, &a), &b) in squares.iter_mut().zip(x).zip(y) {
        *square = multiply_add(weight, vector_square(a, b), *square);
    }
}

/// `squares + weight (a^2 + b^2 + 2 c^2)`.
fn add_tensor_squares(squares: &mut [f64], a: &[f64], b: &[f64], c: &[f64], weight: f64) {
    for (((square, &first), &second), &third) in squares.iter_mut().zip(a).zip(b).zip(c) {
        *square = multiply_add(weight, tensor_square(first, second, third), *square);
    }
}

/// `max(1, sqrt(square) / radius)` in place: what a vector of that squared norm is
/// divided by to project it onto the ball.
fn factors(squares: &mut [f64], radius: f64) {
    for square in squares.iter_mut() {
        *square = at_least(square.sqrt() / radius, 1.0);
    }
}

/// Each value divided by its factor.
fn divide(values: &mut [f64], factors: &[f64]) {
    for (value, &factor) in values.iter_mut().zip(factors) {
        *value /= factor;
    }
}

/// The fields of every channel, each `C x H x W`: two of a vector field or three of a
/// tensor field.
type Parts<'a> = &'a mut [&'a mut Vec<f64>];

/// Projects a vector field (two parts) or a tensor field (three) of every channel onto
/// the balls of `radius`, pixel by pixel: over all the channels where coupled, each on
/// its own otherwise. `squares` is room of a plane.
fn project(parts: Parts<'_>, plane: usize, radius: f64, coupled: bool, squares: &mut [f64]) {
    let channels = parts[0].len() / plane;
    let add = |squares: &mut [f64], parts: &[&mut Vec<f64>], channel: usize| {
        let span = channel * plane..(channel + 1) * plane;
        if parts.len() == 2 {
            add_vector_squares(squares, &parts[0][span.clone()], &parts[1][span], 1.0);
        } else {
            add_tensor_squares(
                squares,
                &parts[0][span.clone()],
                &parts[1][span.clone()],
                &parts[2][span],
                1.0,
            );
        }
    };
    if coupled {
        squares.fill(0.0);
        for channel in 0..channels {
            add(squares, parts, channel);
        }
        factors(squares, radius);
        for part in parts.iter_mut() {
            for chunk in part.chunks_exact_mut(plane) {
                divide(chunk, squares);
            }
        }
    } else {
        for channel in 0..channels {
            squares.fill(0.0);
            add(squares, parts, channel);
            factors(squares, radius);
            for part in parts.iter_mut() {
                divide(&mut part[channel * plane..(channel + 1) * plane], squares);
            }
        }
    }
}

/// The sum over the pixels of the norm of the channels' entries, each times its
/// channel's gamma: over all the channels at a pixel where coupled, and each channel's
/// sum, added in order, otherwise. `parts` are two (a vector field) or three (a tensor
/// field); each row's norms are added in lanes, and the rows in a [`Sum`].
fn norm_sum(parts: &[&[f64]], gammas: &[f64], shape: Shape, coupled: bool) -> f64 {
    let (plane, width) = (shape.size(), shape.width);
    let mut terms = vec![0.0; width];
    let squared: Vec<f64> = gammas.iter().map(|&gamma| gamma * gamma).collect();
    let add = |terms: &mut [f64], channel: usize, row: usize, weight: f64| {
        let start = channel * plane + row * width;
        let span = start..start + width;
        if parts.len() == 2 {
            add_vector_squares(terms, &parts[0][span.clone()], &parts[1][span], weight);
        } else {
            add_tensor_squares(
                terms,
                &parts[0][span.clone()],
                &parts[1][span.clone()],
                &parts[2][span],
                weight,
            );
        }
    };
    let roots = |terms: &mut [f64]| {
        for term in terms.iter_mut() {
            *term = term.sqrt();
        }
    };
    if coupled {
        let mut total = Sum::new();
        for row in 0..shape.height {
            terms.fill(0.0);
            for (channel, &weight) in squared.iter().enumerate() {
                add(&mut terms, channel, row, weight);
            }
            roots(&mut terms);
            total.add(lanes(&terms));
        }
        total.total()
    } else {
        let mut totals: Vec<Sum> = gammas.iter().map(|_| Sum::new()).collect();
        for (channel, &weight) in squared.iter().enumerate() {
            for row in 0..shape.height {
                terms.fill(0.0);
                add(&mut terms, channel, row, weight);
                roots(&mut terms);
                totals[channel].add(lanes(&terms));
            }
        }
        added(&totals)
    }
}

/// The largest norm over the pixels of a vector field of every channel, in the norm of
/// the channels: over all of them at a pixel where coupled, each channel's otherwise.
fn largest_norm(x: &[f64], y: &[f64], plane: usize, coupled: bool, squares: &mut [f64]) -> f64 {
    let channels = x.len() / plane;
    let mut largest: f64 = 0.0;
    let mut most = |squares: &[f64]| {
        for &square in squares {
            largest = largest.max(square);
        }
    };
    if coupled {
        squares.fill(0.0);
        for channel in 0..channels {
            let span = channel * plane..(channel + 1) * plane;
            add_vector_squares(squares, &x[span.clone()], &y[span], 1.0);
        }
        most(squares);
    } else {
        for channel in 0..channels {
            let span = channel * plane..(channel + 1) * plane;
            squares.fill(0.0);
            add_vector_squares(squares, &x[span.clone()], &y[span], 1.0);
            most(squares);
        }
    }
    largest.sqrt()
}

/// The gamma of each channel, 1 where `None`.
fn channel_weights(model: &Model, weights: Option<&[f64]>) -> Result<Vec<f64>, Error> {
    let values = weights.map_or_else(|| vec![1.0; model.channels], <[f64]>::to_vec);
    if values.len() != model.channels || !values.iter().all(|&value| value > 0.0 && value.is_finite()) {
        return Err(Error::Options(format!(
            "a positive, finite weight for each of the {} channels, not {values:?}",
            model.channels
        )));
    }
    Ok(values)
}

fn due(iteration: u64, settled: &Plan) -> bool {
    iteration == settled.iterations || (settled.record_every > 0 && iteration.is_multiple_of(settled.record_every))
}

/// The gap per sample of a record, and whether a tolerance stops the solver there.
fn stops(settled: &Plan, samples: f64, primal: f64, dual: f64, partial: f64) -> (f64, bool) {
    let gap = primal - dual;
    let per_sample = gap / samples;
    let met = (settled.tolerance > 0.0 && per_sample <= settled.tolerance)
        || (settled.relative_tolerance > 0.0 && gap <= settled.relative_tolerance * primal.abs())
        || (settled.partial_tolerance > 0.0 && partial / samples <= settled.partial_tolerance);
    (per_sample, met)
}

/// `gamma div (fx, fy)` of every channel into `xi`.
fn weighted_divergence(room: &mut Room, fx: &[f64], fy: &[f64], gammas: &[f64], xi: &mut [f64]) {
    let plane = room.shape.size();
    for (channel, &gamma) in gammas.iter().enumerate() {
        let span = channel * plane..(channel + 1) * plane;
        room.divergence(&fx[span.clone()], &fy[span.clone()], &mut xi[span.clone()]);
        for value in &mut xi[span] {
            *value *= gamma;
        }
    }
}

/// The gradient of every channel's plane into `(gx, gy)`.
fn gradient(shape: Shape, x: &[f64], gx: &mut [f64], gy: &mut [f64]) {
    let plane = shape.size();
    for ((own, out_x), out_y) in x
        .chunks_exact(plane)
        .zip(gx.chunks_exact_mut(plane))
        .zip(gy.chunks_exact_mut(plane))
    {
        forward_x(own, shape, out_x);
        forward_y(own, shape, out_y);
    }
}

/// The fields of TV's iterations, and room for its records.
struct TvState {
    x: Vec<f64>,
    px: Vec<f64>,
    py: Vec<f64>,
    new_x: Vec<f64>,
    new_px: Vec<f64>,
    new_py: Vec<f64>,
    gx: Vec<f64>,
    gy: Vec<f64>,
    xi: Vec<f64>,
}

/// Minimizes TV within the intervals of a picture of rounded samples (docs/math.md,
/// 11): the method of [`pdhg::solve_tv`], with its options and defaults but
/// [`TOLERANCE`], [`ITERATIONS`] and [`TV_RATIO`], from the centres and `p = 0`.
/// `observer` is called with each record after the first, and can stop the solver;
/// its records have no coefficients.
///
/// # Errors
///
/// [`Error::Options`] for options or weights that the method cannot take.
pub fn solve_tv(
    model: &Model,
    weights: &Tv,
    options: &Options,
    mut observer: Option<&mut Observer<'_>>,
) -> Result<Solution, Error> {
    let settled = pdhg::plan(&defaults(options, weights.alpha, TV_RATIO), Weights::Tv(weights))?;
    let (tau, sigma) = pdhg::steps(settled.norm_squared, settled.step_ratio, settled.step_product)?;
    let gammas = channel_weights(model, weights.channel_weights.as_deref())?;
    let (shape, plane) = (model.shape, model.plane());
    let size = model.low.len();
    let start = model.project(&model.centre);
    let mut state = TvState {
        x: start.clone(),
        px: vec![0.0; size],
        py: vec![0.0; size],
        new_x: start,
        new_px: vec![0.0; size],
        new_py: vec![0.0; size],
        gx: vec![0.0; size],
        gy: vec![0.0; size],
        xi: vec![0.0; size],
    };
    let mut room = Room::new(shape);
    let (mut work, mut bar, mut squares) = (vec![0.0; plane], vec![0.0; plane], vec![0.0; plane]);
    let (mut gx, mut gy) = (vec![0.0; plane], vec![0.0; plane]);
    let samples = nearest_from_usize(size);
    let values = |state: &mut TvState, room: &mut Room| -> (f64, f64) {
        gradient(shape, &state.new_x, &mut state.gx, &mut state.gy);
        let primal = weights.alpha * norm_sum(&[&state.gx, &state.gy], &gammas, shape, weights.coupled)
            + model.value(&state.new_x);
        weighted_divergence(room, &state.new_px, &state.new_py, &gammas, &mut state.xi);
        (primal, -model.conjugate(&state.xi))
    };
    let mut recorder = Recorder::new();
    let (primal, dual) = values(&mut state, &mut room);
    recorder.record(0, primal, dual, f64::NAN, f64::NAN);
    let mut stop = Stop::Iterations;
    let mut iteration = 0;
    while iteration < settled.iterations {
        recorder.resume();
        iteration += 1;
        for (channel, &gamma) in gammas.iter().enumerate() {
            let span = channel * plane..(channel + 1) * plane;
            // x~ = prox(x + tau gamma div p).
            room.divergence(&state.px[span.clone()], &state.py[span.clone()], &mut work);
            ascend(&mut bar, &state.x[span.clone()], &work, tau * gamma);
            model.prox(&bar, tau, span.start, &mut state.new_x[span.clone()]);
            // p + sigma gamma grad (2 x~ - x).
            extrapolate(&mut bar, &state.new_x[span.clone()], &state.x[span.clone()]);
            forward_x(&bar, shape, &mut gx);
            forward_y(&bar, shape, &mut gy);
            ascend(
                &mut state.new_px[span.clone()],
                &state.px[span.clone()],
                &gx,
                sigma * gamma,
            );
            ascend(&mut state.new_py[span.clone()], &state.py[span], &gy, sigma * gamma);
        }
        let mut fields = [&mut state.new_px, &mut state.new_py];
        project(&mut fields, plane, weights.alpha, weights.coupled, &mut squares);
        let rho = settled.relaxation;
        relax(&mut state.x, &state.new_x, rho);
        relax(&mut state.px, &state.new_px, rho);
        relax(&mut state.py, &state.new_py, rho);
        recorder.pause();
        if due(iteration, &settled) {
            let (primal, dual) = values(&mut state, &mut room);
            recorder.record(iteration, primal, dual, f64::NAN, f64::NAN);
            let (gap, met) = stops(&settled, samples, primal, dual, f64::NAN);
            if met {
                stop = Stop::Converged;
            }
            if let Some(observe) = observer.as_deref_mut() {
                let record = Record {
                    height: shape.height,
                    width: shape.width,
                    iteration,
                    gap,
                    primal,
                    dual,
                    coefficients: &[],
                    canvas: &state.new_x,
                    w: None,
                };
                if observe(&record).is_break() && !met {
                    stop = Stop::Observer;
                    break;
                }
            }
            if met {
                break;
            }
        }
    }
    Ok(Solution {
        picture: state.new_x,
        w: None,
        p: Vector {
            x: state.new_px,
            y: state.new_py,
        },
        r: None,
        iterations: iteration,
        stop,
        history: recorder.finish(),
    })
}

/// The fields of TGV's iterations, and room for its records.
struct TgvState {
    x: Vec<f64>,
    w: [Vec<f64>; 2],
    p: [Vec<f64>; 2],
    r: [Vec<f64>; 3],
    new_x: Vec<f64>,
    new_w: [Vec<f64>; 2],
    new_p: [Vec<f64>; 2],
    new_r: [Vec<f64>; 3],
    first: [Vec<f64>; 2],
    second: [Vec<f64>; 3],
    xi: Vec<f64>,
}

/// TGV's primal value, its feasible dual value, the scaling of `r` that made it
/// feasible, and the partial gap where its radius is given (docs/math.md, 6.2, 6.3
/// and 11), at the outputs of the proximal steps.
fn tgv_values(
    model: &Model,
    weights: &Tgv,
    gammas: &[f64],
    partial_radius: Option<f64>,
    state: &mut TgvState,
    room: &mut Room,
    squares: &mut [f64],
) -> (f64, f64, f64, f64) {
    let (shape, plane) = (model.shape, model.plane());
    let [ax, ay] = &mut state.first;
    gradient(shape, &state.new_x, ax, ay);
    for (value, &own) in ax.iter_mut().zip(&state.new_w[0]) {
        *value -= own;
    }
    for (value, &own) in ay.iter_mut().zip(&state.new_w[1]) {
        *value -= own;
    }
    let [e11, e22, e12] = &mut state.second;
    for channel in 0..model.channels {
        let span = channel * plane..(channel + 1) * plane;
        room.symmetrized(
            &state.new_w[0][span.clone()],
            &state.new_w[1][span.clone()],
            &mut e11[span.clone()],
            &mut e22[span.clone()],
            &mut e12[span],
        );
    }
    let primal = weights.alpha1 * norm_sum(&[ax, ay], gammas, shape, weights.coupled)
        + weights.alpha0 * norm_sum(&[e11, e22, e12], gammas, shape, weights.coupled)
        + model.value(&state.new_x);
    // div2 r of every channel into the first fields, then theta.
    let [dx, dy] = &mut state.first;
    for channel in 0..model.channels {
        let span = channel * plane..(channel + 1) * plane;
        let [r11, r22, r12] = &state.new_r;
        room.tensor_divergence(
            &r11[span.clone()],
            &r22[span.clone()],
            &r12[span.clone()],
            &mut dx[span.clone()],
            &mut dy[span],
        );
    }
    let largest = largest_norm(dx, dy, plane, weights.coupled, squares);
    let theta = if largest <= weights.alpha1 {
        1.0
    } else {
        weights.alpha1 / largest
    };
    let partial = match partial_radius {
        Some(radius) => {
            weighted_divergence(room, &state.new_p[0], &state.new_p[1], gammas, &mut state.xi);
            let bound = model.conjugate(&state.xi);
            let [e1, e2, _] = &mut state.second;
            for ((value, &p), &d) in e1.iter_mut().zip(&state.new_p[0]).zip(dx.iter()) {
                *value = p + d;
            }
            for ((value, &p), &d) in e2.iter_mut().zip(&state.new_p[1]).zip(dy.iter()) {
                *value = p + d;
            }
            primal + bound + radius * norm_sum(&[e1, e2], gammas, shape, false)
        }
        None => f64::NAN,
    };
    for value in dx.iter_mut() {
        *value *= -theta;
    }
    for value in dy.iter_mut() {
        *value *= -theta;
    }
    weighted_divergence(room, dx, dy, gammas, &mut state.xi);
    (primal, -model.conjugate(&state.xi), theta, partial)
}

/// Minimizes TGV within the intervals of a picture of rounded samples (docs/math.md,
/// 11): the method of [`pdhg::solve_tgv`], with its options and defaults but
/// [`TOLERANCE`], [`ITERATIONS`] and [`TGV_RATIO`], from the centres, `w` their
/// gradient, and `p` and `r` 0. The gap is that of the feasible dual (6.2), and the
/// partial gap that of 6.3 where its radius is given.
///
/// # Errors
///
/// [`Error::Options`] for options or weights that the method cannot take.
#[expect(
    clippy::too_many_lines,
    reason = "one iteration of TGV, step by step as docs/math.md, 5 writes it"
)]
pub fn solve_tgv(
    model: &Model,
    weights: &Tgv,
    options: &Options,
    mut observer: Option<&mut Observer<'_>>,
) -> Result<Solution, Error> {
    let settled = pdhg::plan(&defaults(options, weights.alpha1, TGV_RATIO), Weights::Tgv(weights))?;
    let (tau, sigma) = pdhg::steps(settled.norm_squared, settled.step_ratio, settled.step_product)?;
    let gammas = channel_weights(model, weights.channel_weights.as_deref())?;
    let (shape, plane) = (model.shape, model.plane());
    let size = model.low.len();
    let zeros = || vec![0.0; size];
    let start = model.project(&model.centre);
    let mut w = [zeros(), zeros()];
    {
        let [wx, wy] = &mut w;
        gradient(shape, &start, wx, wy);
    }
    let mut state = TgvState {
        x: start.clone(),
        new_w: w.clone(),
        w,
        p: [zeros(), zeros()],
        r: [zeros(), zeros(), zeros()],
        new_x: start,
        new_p: [zeros(), zeros()],
        new_r: [zeros(), zeros(), zeros()],
        first: [zeros(), zeros()],
        second: [zeros(), zeros(), zeros()],
        xi: zeros(),
    };
    let mut room = Room::new(shape);
    let room_plane = || vec![0.0; plane];
    let (mut work, mut bar, mut squares) = (room_plane(), room_plane(), room_plane());
    let (mut first, mut second) = (room_plane(), room_plane());
    let (mut bar_across, mut bar_down) = (room_plane(), room_plane());
    let (mut e11, mut e22, mut e12) = (room_plane(), room_plane(), room_plane());
    let samples = nearest_from_usize(size);
    let mut recorder = Recorder::new();
    let (primal, dual, theta, partial) = tgv_values(
        model,
        weights,
        &gammas,
        settled.partial_radius,
        &mut state,
        &mut room,
        &mut squares,
    );
    recorder.record(0, primal, dual, theta, partial);
    let mut stop = Stop::Iterations;
    let mut iteration = 0;
    while iteration < settled.iterations {
        recorder.resume();
        iteration += 1;
        for (channel, &gamma) in gammas.iter().enumerate() {
            let span = channel * plane..(channel + 1) * plane;
            let (descent, ascent) = (tau * gamma, sigma * gamma);
            // x~ = prox(x + tau gamma div p).
            room.divergence(&state.p[0][span.clone()], &state.p[1][span.clone()], &mut work);
            ascend(&mut bar, &state.x[span.clone()], &work, descent);
            model.prox(&bar, tau, span.start, &mut state.new_x[span.clone()]);
            // w~ = w + tau gamma (p + div2 r).
            room.tensor_divergence(
                &state.r[0][span.clone()],
                &state.r[1][span.clone()],
                &state.r[2][span.clone()],
                &mut first,
                &mut second,
            );
            for (part, divergence) in [(0, &first), (1, &second)] {
                let target = &mut state.new_w[part][span.clone()];
                let (old, p) = (&state.w[part][span.clone()], &state.p[part][span.clone()]);
                for (((value, &d), &w_old), &own) in target.iter_mut().zip(divergence.iter()).zip(old).zip(p) {
                    *value = multiply_add(descent, own + d, w_old);
                }
            }
            // The extrapolations, and p + sigma gamma (grad x- - w-) and r + sigma gamma E w-.
            extrapolate(&mut bar, &state.new_x[span.clone()], &state.x[span.clone()]);
            extrapolate(
                &mut bar_across,
                &state.new_w[0][span.clone()],
                &state.w[0][span.clone()],
            );
            extrapolate(&mut bar_down, &state.new_w[1][span.clone()], &state.w[1][span.clone()]);
            forward_x(&bar, shape, &mut first);
            forward_y(&bar, shape, &mut second);
            ascend_less(
                &mut state.new_p[0][span.clone()],
                &state.p[0][span.clone()],
                &first,
                &bar_across,
                ascent,
            );
            ascend_less(
                &mut state.new_p[1][span.clone()],
                &state.p[1][span.clone()],
                &second,
                &bar_down,
                ascent,
            );
            room.symmetrized(&bar_across, &bar_down, &mut e11, &mut e22, &mut e12);
            for (part, e) in [(0, &e11), (1, &e22), (2, &e12)] {
                ascend(
                    &mut state.new_r[part][span.clone()],
                    &state.r[part][span.clone()],
                    e,
                    ascent,
                );
            }
        }
        {
            let [p0, p1] = &mut state.new_p;
            let mut fields = [p0, p1];
            project(&mut fields, plane, weights.alpha1, weights.coupled, &mut squares);
        }
        {
            let [r0, r1, r2] = &mut state.new_r;
            let mut fields = [r0, r1, r2];
            project(&mut fields, plane, weights.alpha0, weights.coupled, &mut squares);
        }
        let rho = settled.relaxation;
        relax(&mut state.x, &state.new_x, rho);
        for part in 0..2 {
            relax(&mut state.w[part], &state.new_w[part], rho);
            relax(&mut state.p[part], &state.new_p[part], rho);
        }
        for part in 0..3 {
            relax(&mut state.r[part], &state.new_r[part], rho);
        }
        recorder.pause();
        if due(iteration, &settled) {
            let (primal, dual, theta, partial) = tgv_values(
                model,
                weights,
                &gammas,
                settled.partial_radius,
                &mut state,
                &mut room,
                &mut squares,
            );
            recorder.record(iteration, primal, dual, theta, partial);
            let (gap, met) = stops(&settled, samples, primal, dual, partial);
            if met {
                stop = Stop::Converged;
            }
            if let Some(observe) = observer.as_deref_mut() {
                let field = Vector {
                    x: state.new_w[0].clone(),
                    y: state.new_w[1].clone(),
                };
                let record = Record {
                    height: shape.height,
                    width: shape.width,
                    iteration,
                    gap,
                    primal,
                    dual,
                    coefficients: &[],
                    canvas: &state.new_x,
                    w: Some(&field),
                };
                if observe(&record).is_break() && !met {
                    stop = Stop::Observer;
                    break;
                }
            }
            if met {
                break;
            }
        }
    }
    let [wx, wy] = state.new_w;
    let [px, py] = state.new_p;
    let [rxx, ryy, rxy] = state.new_r;
    Ok(Solution {
        picture: state.new_x,
        w: Some(Vector { x: wx, y: wy }),
        p: Vector { x: px, y: py },
        r: Some(Tensor {
            xx: rxx,
            yy: ryy,
            xy: rxy,
        }),
        iterations: iteration,
        stop,
        history: recorder.finish(),
    })
}

/// The method of a picture of rounded samples, with its weights.
#[derive(Debug, Clone, Copy)]
pub enum Method<'a> {
    /// Total variation.
    Tv(&'a Tv),
    /// Second-order total generalized variation.
    Tgv(&'a Tgv),
}

/// Reconstructs a picture of rounded samples by the method (docs/math.md, 11).
///
/// # Errors
///
/// [`Error::Options`] for options or weights that the method cannot take.
pub fn solve(
    model: &Model,
    method: Method<'_>,
    options: &Options,
    observer: Option<&mut Observer<'_>>,
) -> Result<Solution, Error> {
    match method {
        Method::Tv(weights) => solve_tv(model, weights, options, observer),
        Method::Tgv(weights) => solve_tgv(model, weights, options, observer),
    }
}
