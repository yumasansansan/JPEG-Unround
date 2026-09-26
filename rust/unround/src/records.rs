// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! The values of a record, in the layout of the planes: the primal and dual values of
//! TV and TGV, and TGV's partial gap (docs/math.md, 6.1 to 6.3 and 6.6), from the
//! outputs of a sweep ([`crate::sweep`]).
//!
//! They are those of [`crate::frames`], with the same formulas ([`crate::kernels`]),
//! computed row by row: the terms of a row are added in lanes
//! ([`crate::exact::lanes`]), and the rows' sums in a [`Sum`]. The order differs from
//! that of the natural layout, and so do the last bits.

use crate::dct::BLOCK;
use crate::exact::{Sum, added, lanes, nearest_from_usize};
use crate::frames::{FREE_CENTRE, Frame};
use crate::kernels::{
    self, at_least, backward_across, backward_down, beyond, centre, conjugate_term, forward_across, forward_down,
    interval_end, squared_term,
};
use crate::model::{LEVEL_SHIFT_DC, Tgv, Tv};
use crate::operators::{tensor_square, vector_square};
use crate::planar::{Layout, Part};
use crate::sweep::{TgvOutputs, TvOutputs};

/// What the data term of a frequency takes.
#[derive(Debug, Clone, Copy, Default)]
struct Frequency {
    step: f64,
    shrunk: f64,
    shift: f64,
    weight: f64,
    charge: f64,
}

/// What the values of one component take: its layout and levels, the constants of
/// its frequencies, its slack, and whether it has free samples.
#[derive(Debug, Clone)]
struct Component {
    part: Part,
    frequencies: [[Frequency; BLOCK]; BLOCK],
    slack: f64,
    charged: bool,
    free: bool,
}

impl Component {
    /// The ends of a coefficient's interval, its centre, and the file's own interval.
    #[inline]
    fn model(&self, level: f64, frequency: &Frequency) -> ((f64, f64), f64, (f64, f64)) {
        let Frequency {
            step, shrunk, shift, ..
        } = *frequency;
        let ends = (
            interval_end(level, -0.5, -self.slack, step, shift),
            interval_end(level, 0.5, self.slack, step, shift),
        );
        let inner = (
            interval_end(level, -0.5, 0.0, step, shift),
            interval_end(level, 0.5, 0.0, step, shift),
        );
        (ends, centre(level, step, shrunk, shift), inner)
    }
}

/// The values of the records of a frame.
#[derive(Debug, Clone)]
pub struct Values {
    layout: Layout,
    components: Vec<Component>,
    gammas: Vec<f64>,
    terms: Vec<f64>,
    squares: Vec<f64>,
    first: Vec<f64>,
    second: Vec<f64>,
    third: Vec<f64>,
    fourth: Vec<f64>,
    xi: Vec<f64>,
    field: (Vec<f64>, Vec<f64>),
    grid: Vec<f64>,
    y: Vec<f64>,
    coefficients: Vec<f64>,
}

/// Row `row` of channel `channel` of a field in the planes.
fn row_of<'a>(field: &'a [f64], layout: &Layout, channel: usize, row: usize) -> &'a [f64] {
    let start = channel * layout.plane() + row * layout.width;
    &field[start..start + layout.width]
}

impl Values {
    /// The values of a frame's records, with the channel weights `gammas`.
    #[must_use]
    pub fn new(frame: &Frame, layout: &Layout, gammas: &[f64]) -> Self {
        let components = frame
            .channels()
            .iter()
            .zip(&layout.parts)
            .map(|(channel, part)| {
                let problem = channel.problem();
                let mut frequencies = [[Frequency::default(); BLOCK]; BLOCK];
                let costs = problem.cost().map(|cost| *cost.costs());
                for (index, frequency) in frequencies.iter_mut().flatten().enumerate() {
                    *frequency = Frequency {
                        step: problem.steps()[index],
                        shrunk: problem.shrunk()[index],
                        shift: if index == 0 { LEVEL_SHIFT_DC } else { 0.0 },
                        weight: problem.weights()[index],
                        charge: costs.map_or(0.0, |costs| costs[index]),
                    };
                }
                Component {
                    part: part.clone(),
                    frequencies,
                    slack: problem.slack(),
                    charged: problem.cost().is_some(),
                    free: frame.free(channel),
                }
            })
            .collect();
        let (size, width) = (layout.plane() * gammas.len(), layout.width);
        let grid = layout
            .parts
            .iter()
            .map(|part| BLOCK * part.planes * layout.across)
            .max()
            .unwrap_or(0);
        Self {
            layout: layout.clone(),
            components,
            gammas: gammas.to_vec(),
            terms: vec![0.0; (2 * width).max(grid)],
            squares: vec![0.0; width],
            first: vec![0.0; width],
            second: vec![0.0; width],
            third: vec![0.0; width],
            fourth: vec![0.0; width],
            xi: vec![0.0; size],
            field: (vec![0.0; size], vec![0.0; size]),
            grid: vec![0.0; grid],
            y: vec![0.0; grid],
            coefficients: vec![0.0; grid],
        }
    }

    /// TV's primal value and dual value at the outputs of a sweep: `alpha ||gamma grad
    /// x||` and `G`, and `-G*(gamma div p)` (docs/math.md, 6.1 and 6.6).
    pub fn tv(&mut self, weights: &Tv, outputs: &TvOutputs, radius: f64) -> (f64, f64) {
        let primal = weights.alpha * self.gradient_norm(&outputs.canvas, None, weights.coupled)
            + self.data_term(&outputs.coefficients);
        self.divergence(&outputs.px, &outputs.py);
        (primal, -self.conjugate(radius))
    }

    /// TGV's primal value, a dual value, and the scaling that made the dual feasible
    /// (docs/math.md, 6.2 and 6.6): `r` is scaled by `theta`, the most that keeps
    /// `div2 r` within `alpha1` in the norm of the channels, and the dual is taken at
    /// `p = -div2 (theta r)`.
    pub fn tgv(&mut self, weights: &Tgv, outputs: &TgvOutputs, radius: f64) -> (f64, f64, f64) {
        let first = self.gradient_norm(&outputs.first.canvas, Some((&outputs.wx, &outputs.wy)), weights.coupled);
        let second = self.symmetrized_norm(&outputs.wx, &outputs.wy, weights.coupled);
        let primal = weights.alpha1 * first + weights.alpha0 * second + self.data_term(&outputs.first.coefficients);
        self.tensor_divergence(&outputs.rxx, &outputs.ryy, &outputs.rxy);
        let largest = self.largest_norm(weights.coupled);
        let theta = if largest <= weights.alpha1 {
            1.0
        } else {
            weights.alpha1 / largest
        };
        let mut field = std::mem::take(&mut self.field);
        for values in [&mut field.0, &mut field.1] {
            for value in values.iter_mut() {
                *value *= -theta;
            }
        }
        self.divergence(&field.0, &field.1);
        self.field = field;
        (primal, -self.conjugate(radius), theta)
    }

    /// What TGV's partial gap adds to the primal value (docs/math.md, 6.3): `G*(gamma
    /// div p)`, and the sum over the channels of `gamma ||p + div2 r||`, which the radius
    /// multiplies.
    pub fn tgv_partial(&mut self, outputs: &TgvOutputs, radius: f64) -> (f64, f64) {
        self.tensor_divergence(&outputs.rxx, &outputs.ryy, &outputs.rxy);
        let residual = self.residual_norm(&outputs.first.px, &outputs.first.py);
        self.divergence(&outputs.first.px, &outputs.first.py);
        (self.conjugate(radius), residual)
    }

    /// `||gamma (grad x - w)||` summed over the pixels, in the norm of the channels: over
    /// all the channels at a pixel where coupled, and each channel's sum, added in order,
    /// otherwise.
    fn gradient_norm(&mut self, canvas: &[f64], w: Option<(&[f64], &[f64])>, coupled: bool) -> f64 {
        let layout = &self.layout;
        let (planes, across, height) = (layout.planes, layout.across, layout.height);
        let count = self.gammas.len();
        let mut totals: Vec<Sum> = (0..if coupled { 1 } else { count }).map(|_| Sum::new()).collect();
        for r in 0..height {
            self.squares.fill(0.0);
            for (channel, &gamma) in self.gammas.iter().enumerate() {
                let row = row_of(canvas, layout, channel, r);
                forward_across(row, &mut self.first, planes, across);
                let below = (r + 1 < height).then(|| row_of(canvas, layout, channel, r + 1));
                forward_down(row, below, &mut self.second);
                if let Some((wx, wy)) = w {
                    let (wx, wy) = (row_of(wx, layout, channel, r), row_of(wy, layout, channel, r));
                    subtract_rows(&mut self.first, wx);
                    subtract_rows(&mut self.second, wy);
                }
                if coupled {
                    add_vector_squares(&mut self.squares, &self.first, &self.second, gamma);
                } else {
                    self.terms
                        .iter_mut()
                        .zip(&self.first)
                        .zip(&self.second)
                        .for_each(|((term, &x), &y)| {
                            *term = vector_square(gamma * x, gamma * y).sqrt();
                        });
                    totals[channel].add(lanes(&self.terms[..layout.width]));
                }
            }
            if coupled {
                roots(&self.squares, &mut self.terms);
                totals[0].add(lanes(&self.terms[..layout.width]));
            }
        }
        added(&totals)
    }

    /// `||gamma E w||` summed over the pixels, in the norm of the channels, with the
    /// off-diagonal entry counted twice.
    fn symmetrized_norm(&mut self, wx: &[f64], wy: &[f64], coupled: bool) -> f64 {
        let layout = &self.layout;
        let (planes, across, height, width) = (layout.planes, layout.across, layout.height, layout.width);
        let count = self.gammas.len();
        let mut totals: Vec<Sum> = (0..if coupled { 1 } else { count }).map(|_| Sum::new()).collect();
        for r in 0..height {
            self.squares.fill(0.0);
            for (channel, &gamma) in self.gammas.iter().enumerate() {
                let (own_x, own_y) = (row_of(wx, layout, channel, r), row_of(wy, layout, channel, r));
                // The backward differences down: none of the row's own at the last row,
                // and none of the row above at the first.
                let (last, first_row) = (r + 1 == height, r == 0);
                let (above_x, above_y) = if first_row {
                    (None, None)
                } else {
                    (
                        Some(row_of(wx, layout, channel, r - 1)),
                        Some(row_of(wy, layout, channel, r - 1)),
                    )
                };
                backward_across(own_x, &mut self.first, planes, across);
                backward_down((!last).then_some(own_y), above_y, &mut self.second);
                backward_down((!last).then_some(own_x), above_x, &mut self.third);
                backward_across(own_y, &mut self.fourth, planes, across);
                let entries = (&self.first[..width], &self.second[..width], &self.third[..width]);
                let square = |i: usize| {
                    let xy = f64::midpoint(entries.2[i], self.fourth[i]);
                    tensor_square(gamma * entries.0[i], gamma * entries.1[i], gamma * xy)
                };
                if coupled {
                    for (i, total) in self.squares[..width].iter_mut().enumerate() {
                        *total += square(i);
                    }
                } else {
                    for (i, term) in self.terms[..width].iter_mut().enumerate() {
                        *term = square(i).sqrt();
                    }
                    totals[channel].add(lanes(&self.terms[..width]));
                }
            }
            if coupled {
                roots(&self.squares, &mut self.terms);
                totals[0].add(lanes(&self.terms[..width]));
            }
        }
        added(&totals)
    }

    /// The sum over the channels of `gamma ||p + div2 r||`, each channel's norms summed
    /// on its own, with `div2 r` in the field.
    fn residual_norm(&mut self, px: &[f64], py: &[f64]) -> f64 {
        let layout = &self.layout;
        let (height, width) = (layout.height, layout.width);
        let mut totals: Vec<Sum> = self.gammas.iter().map(|_| Sum::new()).collect();
        for (channel, &gamma) in self.gammas.iter().enumerate() {
            for r in 0..height {
                let (own_x, own_y) = (row_of(px, layout, channel, r), row_of(py, layout, channel, r));
                let (div_x, div_y) = (
                    row_of(&self.field.0, layout, channel, r),
                    row_of(&self.field.1, layout, channel, r),
                );
                for i in 0..width {
                    self.terms[i] = vector_square(gamma * (own_x[i] + div_x[i]), gamma * (own_y[i] + div_y[i])).sqrt();
                }
                totals[channel].add(lanes(&self.terms[..width]));
            }
        }
        added(&totals)
    }

    /// `div2 r` of every channel into the field: `(d/dx r11 + d/dy r12, d/dx r12 + d/dy
    /// r22)` by forward differences.
    #[expect(
        clippy::similar_names,
        reason = "the names are those of the formulas: the entries xx, yy and xy of the tensors, and their differences"
    )]
    fn tensor_divergence(&mut self, rxx: &[f64], ryy: &[f64], rxy: &[f64]) {
        let layout = &self.layout;
        let (planes, across, height, width) = (layout.planes, layout.across, layout.height, layout.width);
        for channel in 0..self.gammas.len() {
            for r in 0..height {
                let (below_xy, below_yy) = if r + 1 < height {
                    (
                        Some(row_of(rxy, layout, channel, r + 1)),
                        Some(row_of(ryy, layout, channel, r + 1)),
                    )
                } else {
                    (None, None)
                };
                forward_across(row_of(rxx, layout, channel, r), &mut self.first, planes, across);
                forward_down(row_of(rxy, layout, channel, r), below_xy, &mut self.second);
                forward_across(row_of(rxy, layout, channel, r), &mut self.third, planes, across);
                forward_down(row_of(ryy, layout, channel, r), below_yy, &mut self.fourth);
                let start = channel * layout.plane() + r * width;
                add_rows(&mut self.field.0[start..start + width], &self.first, &self.second);
                add_rows(&mut self.field.1[start..start + width], &self.third, &self.fourth);
            }
        }
    }

    /// The largest norm of the field over the pixels, in the norm of the channels.
    fn largest_norm(&mut self, coupled: bool) -> f64 {
        let layout = &self.layout;
        let (height, width, plane) = (layout.height, layout.width, layout.plane());
        let mut largest: f64 = 0.0;
        for r in 0..height {
            self.squares.fill(0.0);
            for channel in 0..self.gammas.len() {
                let start = channel * plane + r * width;
                let (x, y) = (&self.field.0[start..start + width], &self.field.1[start..start + width]);
                if coupled {
                    add_vector_squares(&mut self.squares, x, y, 1.0);
                } else {
                    self.terms
                        .iter_mut()
                        .zip(x)
                        .zip(y)
                        .for_each(|((term, &x), &y)| *term = vector_square(x, y).sqrt());
                    largest = at_least(largest, most(&self.terms[..width]));
                }
            }
            if coupled {
                roots(&self.squares, &mut self.terms);
                largest = at_least(largest, most(&self.terms[..width]));
            }
        }
        largest
    }

    /// `xi = gamma div (field)` of every channel, into `xi`: the backward differences of
    /// the entries across along the rows and of those down down the columns.
    fn divergence(&mut self, fx: &[f64], fy: &[f64]) {
        let layout = &self.layout;
        let (planes, across, height, width) = (layout.planes, layout.across, layout.height, layout.width);
        for (channel, &gamma) in self.gammas.iter().enumerate() {
            for r in 0..height {
                backward_across(row_of(fx, layout, channel, r), &mut self.first, planes, across);
                let own = (r + 1 < height).then(|| row_of(fy, layout, channel, r));
                let above = (r > 0).then(|| row_of(fy, layout, channel, r - 1));
                backward_down(own, above, &mut self.second);
                let start = channel * layout.plane() + r * width;
                for ((value, &a), &d) in self.xi[start..start + width]
                    .iter_mut()
                    .zip(&self.first)
                    .zip(&self.second)
                {
                    *value = (a + d) * gamma;
                }
            }
        }
    }

    /// `G*(xi)`, bounded over the box of the radius where samples are free (docs/math.md,
    /// 6.6): for each component, `g*` of the DCT of the sums of `xi` over its cells, and
    /// where it has free samples, `<zeta, m> + radius ||zeta||_1` of `zeta = xi - Pi xi`.
    fn conjugate(&mut self, radius: f64) -> f64 {
        let mut value = 0.0;
        for index in 0..self.components.len() {
            let (terms, beyond_sum, magnitude) = self.component_conjugate(index);
            value += terms;
            if self.components[index].free {
                value += FREE_CENTRE * beyond_sum + radius * magnitude;
            }
        }
        value
    }

    /// Of one component: the sum of `g*`'s terms, and the sums of `zeta` beyond its
    /// blocks and of `|zeta|` over its canvas.
    fn component_conjugate(&mut self, index: usize) -> (f64, f64, f64) {
        let layout = &self.layout;
        let component = &self.components[index];
        let part = &component.part;
        let (down, across_ratio) = part.ratio;
        let (width, across) = (layout.width, layout.across);
        let row = part.planes * across;
        let block_row = part.block_row(across);
        let cells = nearest_from_usize(down * across_ratio);
        let base = index * layout.plane();
        let (mut terms_sum, mut beyond_sum, mut magnitude) = (Sum::new(), Sum::new(), Sum::new());
        let block_rows = layout.height / (BLOCK * down);
        for by in 0..block_rows {
            // The sums of the cells of the block row's rows: the component's samples times n.
            let grid = &mut self.grid[..BLOCK * row];
            for (r, line) in grid.chunks_exact_mut(row).enumerate() {
                for (p, values) in line.chunks_exact_mut(across).enumerate() {
                    values.fill(0.0);
                    for a in 0..down {
                        let source = base + ((BLOCK * by + r) * down + a) * width;
                        for k in 0..across_ratio {
                            let start = source + (across_ratio * p + k) * across;
                            for (value, &sample) in values.iter_mut().zip(&self.xi[start..start + across]) {
                                *value += sample;
                            }
                        }
                    }
                }
            }
            let covered = by < part.rows;
            if covered {
                // The DCT of the block row, and g*'s terms of the coefficients there are.
                kernels::forward8(grid, row, &mut self.y, row, row);
                for v in 0..BLOCK {
                    for set in 0..part.sets {
                        let place = v * row + BLOCK * set * across;
                        kernels::forward8(
                            &self.y[place..],
                            across,
                            &mut self.coefficients[place..],
                            across,
                            across,
                        );
                    }
                }
                let levels = &component.part.levels[by * block_row..(by + 1) * block_row];
                for v in 0..BLOCK {
                    for (set, &count) in part.valid.iter().enumerate() {
                        for u in 0..BLOCK {
                            let start = (v * part.planes + BLOCK * set + u) * across;
                            let (values, own) =
                                (&self.coefficients[start..start + count], &levels[start..start + count]);
                            let frequency = &component.frequencies[v][u];
                            let terms = &mut self.terms[..count];
                            if component.charged {
                                conjugate_row::<true>(terms, values, own, component, frequency);
                            } else {
                                conjugate_row::<false>(terms, values, own, component, frequency);
                            }
                            terms_sum.add(lanes(terms));
                        }
                    }
                }
            }
            if component.free {
                // zeta: xi less the mean of its cell where the component covers the
                // sample, xi itself beyond.
                for r in 0..BLOCK * down {
                    let own = r / down;
                    let start = base + (BLOCK * by * down + r) * width;
                    let samples = &self.xi[start..start + width];
                    // zeta of the row, and in `beyond` zeta beyond the blocks, 0 within them.
                    let (terms, beyond) = self.terms.split_at_mut(width);
                    let beyond = &mut beyond[..width];
                    for p in 0..layout.planes {
                        let own_plane = p / across_ratio;
                        let valid = if covered { part.valid[own_plane / BLOCK] } else { 0 };
                        let mean_start = own * row + own_plane * across;
                        let means = &self.grid[mean_start..mean_start + across];
                        let values = &samples[p * across..(p + 1) * across];
                        let span = p * across..(p + 1) * across;
                        let (terms, beyond) = (&mut terms[span.clone()], &mut beyond[span]);
                        for m in 0..valid {
                            terms[m] = values[m] - means[m] / cells;
                        }
                        terms[valid..].copy_from_slice(&values[valid..]);
                        beyond[..valid].fill(0.0);
                        beyond[valid..].copy_from_slice(&values[valid..]);
                    }
                    beyond_sum.add(lanes(beyond));
                    for term in terms.iter_mut() {
                        *term = term.abs();
                    }
                    magnitude.add(lanes(terms));
                }
            }
        }
        (terms_sum.total(), beyond_sum.total(), magnitude.total())
    }

    /// `G` at the outputs' coefficients: half the sum of `w (c - centre)^2`, and the
    /// slack's cost, of every component, added in order.
    fn data_term(&mut self, coefficients: &[Vec<f64>]) -> f64 {
        let across = self.layout.across;
        let mut value = 0.0;
        for (component, values) in self.components.iter().zip(coefficients) {
            let part = &component.part;
            let block_row = part.block_row(across);
            let (mut squares, mut charged) = (Sum::new(), Sum::new());
            for by in 0..part.rows {
                let levels = &component.part.levels[by * block_row..(by + 1) * block_row];
                let own = &values[by * block_row..(by + 1) * block_row];
                for v in 0..BLOCK {
                    for (set, &count) in part.valid.iter().enumerate() {
                        for u in 0..BLOCK {
                            let start = (v * part.planes + BLOCK * set + u) * across;
                            let frequency = &component.frequencies[v][u];
                            let (coefficients, levels) = (&own[start..start + count], &levels[start..start + count]);
                            let terms = &mut self.terms[..count];
                            data_row(terms, coefficients, levels, component, frequency, false);
                            squares.add(lanes(terms));
                            if component.charged {
                                data_row(terms, coefficients, levels, component, frequency, true);
                                charged.add(lanes(terms));
                            }
                        }
                    }
                }
            }
            let mut own = 0.5 * squares.total();
            if component.charged {
                own += charged.total();
            }
            value += own;
        }
        value
    }
}

/// `G*`'s terms of a row of coefficients of one frequency.
#[inline]
fn conjugate_row<const CHARGED: bool>(
    terms: &mut [f64],
    values: &[f64],
    levels: &[i16],
    component: &Component,
    frequency: &Frequency,
) {
    let count = terms.len();
    let (values, levels) = (&values[..count], &levels[..count]);
    for i in 0..count {
        let (ends, centre, inner) = component.model(f64::from(levels[i]), frequency);
        terms[i] = conjugate_term::<CHARGED>(values[i], frequency.weight, ends, centre, frequency.charge, inner);
    }
}

/// `G`'s terms of a row of coefficients of one frequency: `w (c - centre)^2`, or where
/// `charged`, the cost of lying beyond the file's own interval.
#[inline]
fn data_row(
    terms: &mut [f64],
    values: &[f64],
    levels: &[i16],
    component: &Component,
    frequency: &Frequency,
    charged: bool,
) {
    let count = terms.len();
    let (values, levels) = (&values[..count], &levels[..count]);
    if charged {
        for i in 0..count {
            let (_, _, (lower, upper)) = component.model(f64::from(levels[i]), frequency);
            terms[i] = frequency.charge * beyond(values[i], lower, upper);
        }
    } else {
        for i in 0..count {
            let (_, centre, _) = component.model(f64::from(levels[i]), frequency);
            terms[i] = squared_term(values[i], frequency.weight, centre);
        }
    }
}

/// `row <- row - other`.
#[inline]
fn subtract_rows(row: &mut [f64], other: &[f64]) {
    for (value, &take) in row.iter_mut().zip(other) {
        *value -= take;
    }
}

/// `out <- first + second`.
#[inline]
fn add_rows(out: &mut [f64], first: &[f64], second: &[f64]) {
    for ((value, &a), &b) in out.iter_mut().zip(first).zip(second) {
        *value = a + b;
    }
}

/// `squares <- squares + |gamma (x, y)|^2`.
#[inline]
fn add_vector_squares(squares: &mut [f64], x: &[f64], y: &[f64], gamma: f64) {
    for ((square, &x), &y) in squares.iter_mut().zip(x).zip(y) {
        *square += vector_square(gamma * x, gamma * y);
    }
}

/// The square roots of a row.
#[inline]
fn roots(squares: &[f64], out: &mut [f64]) {
    for (value, &square) in out.iter_mut().zip(squares) {
        *value = square.sqrt();
    }
}

/// The largest value of a row, in eight lanes.
#[inline]
fn most(values: &[f64]) -> f64 {
    let mut lanes = [0.0f64; 8];
    let (chunks, rest) = values.as_chunks::<8>();
    for chunk in chunks {
        for (lane, &value) in lanes.iter_mut().zip(chunk) {
            *lane = at_least(*lane, value);
        }
    }
    let mut largest = lanes[0];
    for &lane in &lanes[1..] {
        largest = at_least(largest, lane);
    }
    for &value in rest {
        largest = at_least(largest, value);
    }
    largest
}
