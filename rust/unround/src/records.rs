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

use std::ops::Range;

use crate::dct::{BLOCK, multiply_add};
use crate::exact::{Sum, added, lanes, nearest_from_usize};
use crate::frames::{FREE_CENTRE, Frame};
use crate::kernels::{
    self, at_least, backward_across, backward_down, beyond, centre, conjugate_term, forward_across, forward_down,
    interval_end, most, squared_term,
};
use crate::model::{LEVEL_SHIFT_DC, Tgv, Tv};
use crate::operators::{tensor_square, vector_square};
use crate::planar::{Layout, Part};
use crate::sweep::{TgvOutputs, TvOutputs};

/// What the data term of a frequency takes: its weight in a block that is not flat,
/// and in one that is.
#[derive(Debug, Clone, Copy, Default)]
struct Frequency {
    step: f64,
    shrunk: f64,
    shift: f64,
    weight: f64,
    flat_weight: f64,
    charge: f64,
}

impl Frequency {
    /// The weight in a block, flat where `mark` is 1: `flat mark + own (1 - mark)`,
    /// exact for a mark of 1 or 0. A choice between the two fields would be compiled as
    /// a choice between their places and a read from it, in vectors a read from a place
    /// of each value's own, which NEON has no instruction for: the rows that take the
    /// weight would be scalar there.
    #[inline]
    fn weight_of(&self, mark: f64) -> f64 {
        multiply_add(self.flat_weight, mark, self.weight * (1.0 - mark))
    }
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

/// What a thread takes its rows of a record in: the terms of a row, the squares of its
/// pixels, the rows of the differences, and the grids of a block row's cells and of
/// their DCT.
#[derive(Debug, Clone)]
struct Room {
    terms: Vec<f64>,
    squares: Vec<f64>,
    first: Vec<f64>,
    second: Vec<f64>,
    third: Vec<f64>,
    fourth: Vec<f64>,
    grid: Vec<f64>,
    y: Vec<f64>,
    coefficients: Vec<f64>,
}

impl Room {
    /// The room of rows `width` wide and of block rows of grids of `grid` values.
    fn new(width: usize, grid: usize) -> Self {
        Self {
            terms: vec![0.0; (2 * width).max(grid)],
            squares: vec![0.0; width],
            first: vec![0.0; width],
            second: vec![0.0; width],
            third: vec![0.0; width],
            fourth: vec![0.0; width],
            grid: vec![0.0; grid],
            y: vec![0.0; grid],
            coefficients: vec![0.0; grid],
        }
    }
}

/// The values of the records of a frame. The rows of the canvas are taken in parts of
/// whole bands, on threads at once, each part with a room of its own: each row's terms
/// are those of one thread, and the rows' sums are added in the order of the rows, so
/// that the values are the same, to the last bit, whatever the parts.
#[derive(Debug, Clone)]
pub struct Values {
    layout: Layout,
    components: Vec<Component>,
    gammas: Vec<f64>,
    parts: Vec<Range<usize>>,
    rooms: Vec<Room>,
    xi: Vec<f64>,
    field: (Vec<f64>, Vec<f64>),
}

/// Row `row` of channel `channel` of a field in the planes.
fn row_of<'a>(field: &'a [f64], layout: &Layout, channel: usize, row: usize) -> &'a [f64] {
    let start = channel * layout.plane() + row * layout.width;
    &field[start..start + layout.width]
}

/// The rows of each part of every channel's plane of a field: for each part, its rows of
/// each channel.
fn rows_of_parts<'a>(field: &'a mut [f64], layout: &Layout, parts: &[Range<usize>]) -> Vec<Vec<&'a mut [f64]>> {
    let width = layout.width;
    let mut cut: Vec<Vec<&mut [f64]>> = parts.iter().map(|_| Vec::new()).collect();
    for plane in field.chunks_exact_mut(layout.plane()) {
        let mut rest = plane;
        for (own, range) in cut.iter_mut().zip(parts) {
            let (rows, after) = rest.split_at_mut(range.len() * width);
            own.push(rows);
            rest = after;
        }
    }
    cut
}

/// `work` on each part with its room and its input at once, the first part on this
/// thread and the others each on a thread of its own: their results, in the order of
/// the parts.
fn in_parts<T: Send, R: Send>(
    rooms: &mut [Room],
    parts: &[Range<usize>],
    inputs: Vec<T>,
    work: &(dyn Fn(&mut Room, Range<usize>, T) -> R + Sync),
) -> Vec<R> {
    std::thread::scope(|scope| {
        let mut taken = rooms.iter_mut().zip(parts).zip(inputs);
        let Some(((room, range), input)) = taken.next() else {
            return Vec::new();
        };
        let jobs: Vec<_> = taken
            .map(|((room, range), input)| scope.spawn(move || work(room, range.clone(), input)))
            .collect();
        let mut results = vec![work(room, range.clone(), input)];
        results.extend(
            jobs.into_iter()
                .map(|job| job.join().unwrap_or_else(|panic| std::panic::resume_unwind(panic))),
        );
        results
    })
}

/// The parts of `bands` bands of `band` rows that `threads` threads take: as many as the
/// threads, but no more than the bands, each of whole bands and as even as they can be.
fn parts_of(bands: usize, band: usize, threads: usize) -> Vec<Range<usize>> {
    let count = threads.clamp(1, bands.max(1));
    (0..count)
        .map(|k| band * (bands * k / count)..band * (bands * (k + 1) / count))
        .collect()
}

impl Values {
    /// The values of a frame's records, with the channel weights `gammas`, on one thread.
    #[must_use]
    pub fn new(frame: &Frame, layout: &Layout, gammas: &[f64]) -> Self {
        let components: Vec<Component> = frame
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
                        flat_weight: problem.flat_weights()[index],
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
        let size = layout.plane() * gammas.len();
        let values = Self {
            layout: layout.clone(),
            components,
            gammas: gammas.to_vec(),
            parts: Vec::new(),
            rooms: Vec::new(),
            xi: vec![0.0; size],
            field: (vec![0.0; size], vec![0.0; size]),
        };
        values.on_threads(1)
    }

    /// The same, its rows taken on `threads` threads at once: as many parts of whole
    /// bands as the threads, but no more than the bands.
    #[must_use]
    pub fn on_threads(mut self, threads: usize) -> Self {
        let layout = &self.layout;
        let grid = layout
            .parts
            .iter()
            .map(|part| BLOCK * part.planes * layout.across)
            .max()
            .unwrap_or(0);
        self.parts = parts_of(layout.bands(), layout.band, threads);
        self.rooms = self.parts.iter().map(|_| Room::new(layout.width, grid)).collect();
        self
    }

    /// TV's primal value and dual value at the outputs of a sweep: `alpha ||gamma grad
    /// x||` and `G`, and `-G*(gamma div p)` (docs/math.md, 6.1 and 6.6).
    pub fn tv(&mut self, weights: &Tv, outputs: &TvOutputs, radius: f64) -> (f64, f64) {
        let Self {
            layout,
            components,
            gammas,
            parts,
            rooms,
            xi,
            ..
        } = self;
        let (layout, components, gammas) = (&*layout, &components[..], &gammas[..]);
        let coupled = weights.coupled;
        // Each part's terms of the gradient's norm, of G, and of G* at gamma div p.
        let runs = in_parts(
            rooms,
            parts,
            rows_of_parts(xi, layout, parts),
            &|room, range, mut own_xi| {
                let gradient = gradient_terms(room, (layout, gammas), &outputs.canvas, None, coupled, range.clone());
                let data = data_terms(room, layout.across, components, &outputs.coefficients, range.clone());
                divergence_rows(
                    room,
                    (layout, gammas),
                    (&outputs.px, &outputs.py),
                    range.clone(),
                    &mut own_xi,
                );
                let conjugate = conjugate_terms(room, layout, components, &own_xi, range);
                (gradient, data, conjugate)
            },
        );
        let gradient = rows_added(runs.iter().map(|run| &run.0), if coupled { 1 } else { gammas.len() });
        let data: Vec<&[Terms2]> = runs.iter().map(|run| &run.1[..]).collect();
        let primal = weights.alpha * gradient + data_value(components, &data);
        let conjugates: Vec<&[Terms3]> = runs.iter().map(|run| &run.2[..]).collect();
        let conjugate = conjugate_value(components, &conjugates, radius);
        (primal, -conjugate)
    }

    /// TGV's primal value, a dual value, and the scaling that made the dual feasible
    /// (docs/math.md, 6.2 and 6.6): `r` is scaled by `theta`, the most that keeps
    /// `div2 r` within `alpha1` in the norm of the channels, and the dual is taken at
    /// `p = -div2 (theta r)`.
    pub fn tgv(&mut self, weights: &Tgv, outputs: &TgvOutputs, radius: f64) -> (f64, f64, f64) {
        let Self {
            layout,
            components,
            gammas,
            parts,
            rooms,
            xi,
            field,
        } = self;
        let (layout, components, gammas) = (&*layout, &components[..], &gammas[..]);
        let coupled = weights.coupled;
        let sums = if coupled { 1 } else { gammas.len() };
        let r = (&outputs.rxx[..], &outputs.ryy[..], &outputs.rxy[..]);
        // Each part's terms of the norms of the first and second order and of G, and its
        // rows of div2 r and their largest norms.
        let runs = in_parts(
            rooms,
            parts,
            field_parts(field, layout, parts),
            &|room, range, (mut xs, mut ys)| {
                let w = (&outputs.wx[..], &outputs.wy[..]);
                let first = gradient_terms(
                    room,
                    (layout, gammas),
                    &outputs.first.canvas,
                    Some(w),
                    coupled,
                    range.clone(),
                );
                let second = symmetrized_terms(room, (layout, gammas), w, coupled, range.clone());
                let data = data_terms(
                    room,
                    layout.across,
                    components,
                    &outputs.first.coefficients,
                    range.clone(),
                );
                tensor_divergence_rows(room, layout, r, range.clone(), (&mut xs, &mut ys));
                let largest = largest_terms(room, layout, (&xs, &ys), coupled, range);
                (first, second, data, largest)
            },
        );
        let first = rows_added(runs.iter().map(|run| &run.0), sums);
        let second = rows_added(runs.iter().map(|run| &run.1), sums);
        let data: Vec<&[Terms2]> = runs.iter().map(|run| &run.2[..]).collect();
        let primal = weights.alpha1 * first + weights.alpha0 * second + data_value(components, &data);
        // The largest of the parts' largest of their rows' largest: of norms, which are at
        // least 0 or NaN, which every largest leaves, the same whatever the order.
        let largest = runs.iter().fold(0.0, |largest, run| at_least(largest, most(&run.3)));
        let theta = if largest <= weights.alpha1 {
            1.0
        } else {
            weights.alpha1 / largest
        };
        in_parts(rooms, parts, field_parts(field, layout, parts), &|_, _, (xs, ys)| {
            scale_rows(xs, ys, -theta);
        });
        // xi = gamma div (field), and each part's terms of G* there.
        let field = (&field.0[..], &field.1[..]);
        let runs = in_parts(
            rooms,
            parts,
            rows_of_parts(xi, layout, parts),
            &|room, range, mut own_xi| {
                divergence_rows(room, (layout, gammas), field, range.clone(), &mut own_xi);
                conjugate_terms(room, layout, components, &own_xi, range)
            },
        );
        let conjugates: Vec<&[Terms3]> = runs.iter().map(|run| &run[..]).collect();
        (primal, -conjugate_value(components, &conjugates, radius), theta)
    }

    /// What TGV's partial gap adds to the primal value (docs/math.md, 6.3): `G*(gamma
    /// div p)`, and the sum over the channels of `gamma ||p + div2 r||`, which the radius
    /// multiplies.
    pub fn tgv_partial(&mut self, outputs: &TgvOutputs, radius: f64) -> (f64, f64) {
        let Self {
            layout,
            components,
            gammas,
            parts,
            rooms,
            xi,
            field,
        } = self;
        let (layout, components, gammas) = (&*layout, &components[..], &gammas[..]);
        let r = (&outputs.rxx[..], &outputs.ryy[..], &outputs.rxy[..]);
        let p = (&outputs.first.px[..], &outputs.first.py[..]);
        let inputs: Vec<_> = field_parts(field, layout, parts)
            .into_iter()
            .zip(rows_of_parts(xi, layout, parts))
            .collect();
        // Each part's rows of div2 r, its terms of the residual's norms, its rows of xi =
        // gamma div p, and its terms of G* there.
        let runs = in_parts(rooms, parts, inputs, &|room, range, ((mut xs, mut ys), mut own_xi)| {
            tensor_divergence_rows(room, layout, r, range.clone(), (&mut xs, &mut ys));
            let residual = residual_terms(room, (layout, gammas), p, (&xs, &ys), range.clone());
            divergence_rows(room, (layout, gammas), p, range.clone(), &mut own_xi);
            (residual, conjugate_terms(room, layout, components, &own_xi, range))
        });
        let conjugates: Vec<&[Terms3]> = runs.iter().map(|run| &run.1[..]).collect();
        let conjugate = conjugate_value(components, &conjugates, radius);
        (conjugate, rows_added(runs.iter().map(|run| &run.0), gammas.len()))
    }
}

/// A part's rows of every channel's plane of a field.
type PartRows<'a> = Vec<&'a mut [f64]>;

/// The rows of each part of the field's entries across and down.
fn field_parts<'a>(
    field: &'a mut (Vec<f64>, Vec<f64>),
    layout: &Layout,
    parts: &[Range<usize>],
) -> Vec<(PartRows<'a>, PartRows<'a>)> {
    let (x, y) = field;
    rows_of_parts(x, layout, parts)
        .into_iter()
        .zip(rows_of_parts(y, layout, parts))
        .collect()
}

/// The terms of `||gamma (grad x - w)||` of the rows `range`, `w` where given: of each
/// row, the sum of the norms over all the channels at a pixel where coupled, and each
/// channel's otherwise.
fn gradient_terms(
    room: &mut Room,
    (layout, gammas): (&Layout, &[f64]),
    canvas: &[f64],
    w: Option<(&[f64], &[f64])>,
    coupled: bool,
    range: Range<usize>,
) -> Vec<f64> {
    let (planes, across, height) = (layout.planes, layout.across, layout.height);
    let mut terms = Vec::with_capacity(range.len() * if coupled { 1 } else { gammas.len() });
    for r in range {
        room.squares.fill(0.0);
        for (channel, &gamma) in gammas.iter().enumerate() {
            let row = row_of(canvas, layout, channel, r);
            forward_across(row, &mut room.first, planes, across);
            let below = (r + 1 < height).then(|| row_of(canvas, layout, channel, r + 1));
            forward_down(row, below, &mut room.second);
            if let Some((wx, wy)) = w {
                let (wx, wy) = (row_of(wx, layout, channel, r), row_of(wy, layout, channel, r));
                subtract_rows(&mut room.first, wx);
                subtract_rows(&mut room.second, wy);
            }
            if coupled {
                add_vector_squares(&mut room.squares, &room.first, &room.second, gamma);
            } else {
                room.terms
                    .iter_mut()
                    .zip(&room.first)
                    .zip(&room.second)
                    .for_each(|((term, &x), &y)| {
                        *term = vector_square(gamma * x, gamma * y).sqrt();
                    });
                terms.push(lanes(&room.terms[..layout.width]));
            }
        }
        if coupled {
            roots(&room.squares, &mut room.terms);
            terms.push(lanes(&room.terms[..layout.width]));
        }
    }
    terms
}

/// The terms of `||gamma E w||` of the rows `range`, the off-diagonal entry counted
/// twice: of each row, over all the channels at a pixel where coupled, and each
/// channel's otherwise.
fn symmetrized_terms(
    room: &mut Room,
    (layout, gammas): (&Layout, &[f64]),
    (wx, wy): (&[f64], &[f64]),
    coupled: bool,
    range: Range<usize>,
) -> Vec<f64> {
    let (planes, across, height, width) = (layout.planes, layout.across, layout.height, layout.width);
    let mut terms = Vec::with_capacity(range.len() * if coupled { 1 } else { gammas.len() });
    for r in range {
        room.squares.fill(0.0);
        for (channel, &gamma) in gammas.iter().enumerate() {
            let (own_x, own_y) = (row_of(wx, layout, channel, r), row_of(wy, layout, channel, r));
            // The backward differences down: none of the row's own at the last row, and
            // none of the row above at the first.
            let (last, first_row) = (r + 1 == height, r == 0);
            let (above_x, above_y) = if first_row {
                (None, None)
            } else {
                (
                    Some(row_of(wx, layout, channel, r - 1)),
                    Some(row_of(wy, layout, channel, r - 1)),
                )
            };
            backward_across(own_x, &mut room.first, planes, across);
            backward_down((!last).then_some(own_y), above_y, &mut room.second);
            backward_down((!last).then_some(own_x), above_x, &mut room.third);
            backward_across(own_y, &mut room.fourth, planes, across);
            let entries = (&room.first[..width], &room.second[..width], &room.third[..width]);
            let fourth = &room.fourth[..width];
            let square = |i: usize| {
                let xy = f64::midpoint(entries.2[i], fourth[i]);
                tensor_square(gamma * entries.0[i], gamma * entries.1[i], gamma * xy)
            };
            if coupled {
                for (i, total) in room.squares[..width].iter_mut().enumerate() {
                    *total += square(i);
                }
            } else {
                for (i, term) in room.terms[..width].iter_mut().enumerate() {
                    *term = square(i).sqrt();
                }
                terms.push(lanes(&room.terms[..width]));
            }
        }
        if coupled {
            roots(&room.squares, &mut room.terms);
            terms.push(lanes(&room.terms[..width]));
        }
    }
    terms
}

/// The terms of `gamma ||p + div2 r||` of the rows `range`, each channel's of each row,
/// with the part's rows of `div2 r` in `(xs, ys)`.
fn residual_terms(
    room: &mut Room,
    (layout, gammas): (&Layout, &[f64]),
    (px, py): (&[f64], &[f64]),
    (xs, ys): (&[&mut [f64]], &[&mut [f64]]),
    range: Range<usize>,
) -> Vec<f64> {
    let width = layout.width;
    let mut terms = Vec::with_capacity(range.len() * gammas.len());
    for r in range.clone() {
        let start = (r - range.start) * width;
        for (channel, &gamma) in gammas.iter().enumerate() {
            let (own_x, own_y) = (row_of(px, layout, channel, r), row_of(py, layout, channel, r));
            let (div_x, div_y) = (&xs[channel][start..start + width], &ys[channel][start..start + width]);
            for i in 0..width {
                room.terms[i] = vector_square(gamma * (own_x[i] + div_x[i]), gamma * (own_y[i] + div_y[i])).sqrt();
            }
            terms.push(lanes(&room.terms[..width]));
        }
    }
    terms
}

/// `div2 r` of every channel of the rows `range` into the part's rows `(xs, ys)`:
/// `(d/dx r11 + d/dy r12, d/dx r12 + d/dy r22)` by forward differences.
#[expect(
    clippy::similar_names,
    reason = "the names are those of the formulas: the entries xx, yy and xy of the tensors, and their differences"
)]
fn tensor_divergence_rows(
    room: &mut Room,
    layout: &Layout,
    (rxx, ryy, rxy): (&[f64], &[f64], &[f64]),
    range: Range<usize>,
    (xs, ys): (&mut [&mut [f64]], &mut [&mut [f64]]),
) {
    let (planes, across, height, width) = (layout.planes, layout.across, layout.height, layout.width);
    for (channel, (xs, ys)) in xs.iter_mut().zip(ys.iter_mut()).enumerate() {
        for r in range.clone() {
            let (below_xy, below_yy) = if r + 1 < height {
                (
                    Some(row_of(rxy, layout, channel, r + 1)),
                    Some(row_of(ryy, layout, channel, r + 1)),
                )
            } else {
                (None, None)
            };
            forward_across(row_of(rxx, layout, channel, r), &mut room.first, planes, across);
            forward_down(row_of(rxy, layout, channel, r), below_xy, &mut room.second);
            forward_across(row_of(rxy, layout, channel, r), &mut room.third, planes, across);
            forward_down(row_of(ryy, layout, channel, r), below_yy, &mut room.fourth);
            let start = (r - range.start) * width;
            add_rows(&mut xs[start..start + width], &room.first, &room.second);
            add_rows(&mut ys[start..start + width], &room.third, &room.fourth);
        }
    }
}

/// The largest norms of the part's rows `(xs, ys)` of the field, in the norm of the
/// channels: of each row, over all the channels at a pixel where coupled, and each
/// channel's otherwise.
fn largest_terms(
    room: &mut Room,
    layout: &Layout,
    (xs, ys): (&[&mut [f64]], &[&mut [f64]]),
    coupled: bool,
    range: Range<usize>,
) -> Vec<f64> {
    let width = layout.width;
    let mut largest = Vec::with_capacity(range.len() * xs.len());
    for r in range.clone() {
        let start = (r - range.start) * width;
        room.squares.fill(0.0);
        for (x, y) in xs.iter().zip(ys) {
            let (x, y) = (&x[start..start + width], &y[start..start + width]);
            if coupled {
                add_vector_squares(&mut room.squares, x, y, 1.0);
            } else {
                room.terms
                    .iter_mut()
                    .zip(x)
                    .zip(y)
                    .for_each(|((term, &x), &y)| *term = vector_square(x, y).sqrt());
                largest.push(most(&room.terms[..width]));
            }
        }
        if coupled {
            roots(&room.squares, &mut room.terms);
            largest.push(most(&room.terms[..width]));
        }
    }
    largest
}

/// The part's rows of the field times `factor`.
fn scale_rows(xs: Vec<&mut [f64]>, ys: Vec<&mut [f64]>, factor: f64) {
    for values in xs.into_iter().chain(ys) {
        for value in values.iter_mut() {
            *value *= factor;
        }
    }
}

/// `xi = gamma div (fx, fy)` of every channel of the rows `range`, into the part's rows
/// `own_xi`: the backward differences of the entries across along the rows and of those
/// down down the columns.
fn divergence_rows(
    room: &mut Room,
    (layout, gammas): (&Layout, &[f64]),
    (fx, fy): (&[f64], &[f64]),
    range: Range<usize>,
    own_xi: &mut [&mut [f64]],
) {
    let (planes, across, height, width) = (layout.planes, layout.across, layout.height, layout.width);
    for ((channel, &gamma), xi) in gammas.iter().enumerate().zip(own_xi.iter_mut()) {
        for r in range.clone() {
            backward_across(row_of(fx, layout, channel, r), &mut room.first, planes, across);
            let own = (r + 1 < height).then(|| row_of(fy, layout, channel, r));
            let above = (r > 0).then(|| row_of(fy, layout, channel, r - 1));
            backward_down(own, above, &mut room.second);
            let start = (r - range.start) * width;
            for ((value, &a), &d) in xi[start..start + width].iter_mut().zip(&room.first).zip(&room.second) {
                *value = (a + d) * gamma;
            }
        }
    }
}

/// The terms of `G*(xi)` of each component's block rows in the rows `range`, whose rows
/// of `xi` are the part's `own_xi`: those of `g*`, of `zeta` beyond its blocks, and of
/// `|zeta|`, each in the order of their sums.
fn conjugate_terms(
    room: &mut Room,
    layout: &Layout,
    components: &[Component],
    own_xi: &[&mut [f64]],
    range: Range<usize>,
) -> Vec<Terms3> {
    components
        .iter()
        .zip(own_xi)
        .map(|(component, xi)| {
            let block = BLOCK * component.part.ratio.0;
            let mut sums = (Vec::new(), Vec::new(), Vec::new());
            for by in range.start / block..range.end / block {
                conjugate_block_row(room, layout, component, (xi, range.start), by, &mut sums);
            }
            sums
        })
        .collect()
}

/// The terms of each component's `G` at its coefficients of the block rows in the rows
/// `range`: those of `w (c - centre)^2` and of the slack's cost.
fn data_terms(
    room: &mut Room,
    across: usize,
    components: &[Component],
    coefficients: &[Vec<f64>],
    range: Range<usize>,
) -> Vec<Terms2> {
    components
        .iter()
        .zip(coefficients)
        .map(|(component, values)| {
            let block = BLOCK * component.part.ratio.0;
            let mut sums = (Vec::new(), Vec::new());
            for by in range.start / block..(range.end / block).min(component.part.rows) {
                data_block_row(room, across, component, values, by, &mut sums);
            }
            sums
        })
        .collect()
}

/// The terms of two sums of a component.
type Terms2 = (Vec<f64>, Vec<f64>);

/// The terms of three sums of a component.
type Terms3 = (Vec<f64>, Vec<f64>, Vec<f64>);

/// `G*(xi)`, bounded over the box of the radius where samples are free (docs/math.md,
/// 6.6): for each component, `g*` of the DCT of the sums of `xi` over its cells, and
/// where it has free samples, `<zeta, m> + radius ||zeta||_1` of `zeta = xi - Pi xi`;
/// from the parts' terms, added in the order of the parts.
fn conjugate_value(components: &[Component], runs: &[&[Terms3]], radius: f64) -> f64 {
    let mut value = 0.0;
    for (index, component) in components.iter().enumerate() {
        let (mut terms_sum, mut beyond_sum, mut magnitude) = (Sum::new(), Sum::new(), Sum::new());
        for run in runs {
            let (terms, beyond, magnitudes) = &run[index];
            terms_sum.extend(terms.iter().copied());
            beyond_sum.extend(beyond.iter().copied());
            magnitude.extend(magnitudes.iter().copied());
        }
        value += terms_sum.total();
        if component.free {
            value += FREE_CENTRE * beyond_sum.total() + radius * magnitude.total();
        }
    }
    value
}

/// `G` at the outputs' coefficients: half the sum of `w (c - centre)^2`, and the
/// slack's cost, of every component, added in order; from the parts' terms, added in
/// the order of the parts.
fn data_value(components: &[Component], runs: &[&[Terms2]]) -> f64 {
    let mut value = 0.0;
    for (index, component) in components.iter().enumerate() {
        let (mut squares, mut charged) = (Sum::new(), Sum::new());
        for run in runs {
            let (halves, costs) = &run[index];
            squares.extend(halves.iter().copied());
            charged.extend(costs.iter().copied());
        }
        let mut own = 0.5 * squares.total();
        if component.charged {
            own += charged.total();
        }
        value += own;
    }
    value
}

/// The sums of the rows' terms, `sums` to a row, each added row after row, and then
/// the sums in their order.
fn rows_added<'a>(rows: impl Iterator<Item = &'a Vec<f64>>, sums: usize) -> f64 {
    let mut totals: Vec<Sum> = (0..sums).map(|_| Sum::new()).collect();
    for terms in rows {
        for row in terms.chunks_exact(sums) {
            for (total, &term) in totals.iter_mut().zip(row) {
                total.add(term);
            }
        }
    }
    added(&totals)
}

/// The terms of block row `by` of a component of `G*(xi)`: those of `g*`, where the
/// component covers the block row, and where it has free samples those of `zeta`
/// beyond its blocks and of `|zeta|`, each in the order of their sums.
fn conjugate_block_row(
    room: &mut Room,
    layout: &Layout,
    component: &Component,
    (xi, first): (&[f64], usize),
    by: usize,
    (terms_sum, beyond_sum, magnitude): &mut Terms3,
) {
    let part = &component.part;
    let (down, across_ratio) = part.ratio;
    let (width, across) = (layout.width, layout.across);
    let row = part.planes * across;
    let block_row = part.block_row(across);
    let cells = nearest_from_usize(down * across_ratio);
    // xi holds the rows of the component's channel from row `first` on.
    // The sums of the cells of the block row's rows: the component's samples times n.
    let grid = &mut room.grid[..BLOCK * row];
    for (r, line) in grid.chunks_exact_mut(row).enumerate() {
        for (p, values) in line.chunks_exact_mut(across).enumerate() {
            values.fill(0.0);
            for a in 0..down {
                let source = ((BLOCK * by + r) * down + a - first) * width;
                for k in 0..across_ratio {
                    let start = source + (across_ratio * p + k) * across;
                    for (value, &sample) in values.iter_mut().zip(&xi[start..start + across]) {
                        *value += sample;
                    }
                }
            }
        }
    }
    let covered = by < part.rows;
    if covered {
        // The DCT of the block row, and g*'s terms of the coefficients there are.
        kernels::forward8(grid, row, &mut room.y, row, row);
        for v in 0..BLOCK {
            for set in 0..part.sets {
                let place = v * row + BLOCK * set * across;
                kernels::forward8(
                    &room.y[place..],
                    across,
                    &mut room.coefficients[place..],
                    across,
                    across,
                );
            }
        }
        let levels = &component.part.levels[by * block_row..(by + 1) * block_row];
        let flat = component.part.flat_row(by, across);
        for v in 0..BLOCK {
            for (set, &count) in part.valid.iter().enumerate() {
                let marks = &flat[set * across..set * across + count];
                for u in 0..BLOCK {
                    let start = (v * part.planes + BLOCK * set + u) * across;
                    let (values, own) = (&room.coefficients[start..start + count], &levels[start..start + count]);
                    let frequency = &component.frequencies[v][u];
                    let terms = &mut room.terms[..count];
                    if component.charged {
                        conjugate_row::<true>(terms, values, own, marks, component, frequency);
                    } else {
                        conjugate_row::<false>(terms, values, own, marks, component, frequency);
                    }
                    terms_sum.push(lanes(terms));
                }
            }
        }
    }
    if component.free {
        // zeta: xi less the mean of its cell where the component covers the
        // sample, xi itself beyond.
        for r in 0..BLOCK * down {
            let own = r / down;
            let start = (BLOCK * by * down + r - first) * width;
            let samples = &xi[start..start + width];
            // zeta of the row, and in `beyond` zeta beyond the blocks, 0 within them.
            let (terms, beyond) = room.terms.split_at_mut(width);
            let beyond = &mut beyond[..width];
            for p in 0..layout.planes {
                let own_plane = p / across_ratio;
                let valid = if covered { part.valid[own_plane / BLOCK] } else { 0 };
                let mean_start = own * row + own_plane * across;
                let means = &room.grid[mean_start..mean_start + across];
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
            beyond_sum.push(lanes(beyond));
            for term in terms.iter_mut() {
                *term = term.abs();
            }
            magnitude.push(lanes(terms));
        }
    }
}

/// The terms of block row `by` of a component's `G` at its coefficients: those of
/// `w (c - centre)^2`, and where it is charged those of the slack's cost, each in the
/// order of their sums.
fn data_block_row(
    room: &mut Room,
    across: usize,
    component: &Component,
    values: &[f64],
    by: usize,
    (squares, charged): &mut Terms2,
) {
    let part = &component.part;
    let block_row = part.block_row(across);
    let levels = &component.part.levels[by * block_row..(by + 1) * block_row];
    let own = &values[by * block_row..(by + 1) * block_row];
    let flat = component.part.flat_row(by, across);
    for v in 0..BLOCK {
        for (set, &count) in part.valid.iter().enumerate() {
            let marks = &flat[set * across..set * across + count];
            for u in 0..BLOCK {
                let start = (v * part.planes + BLOCK * set + u) * across;
                let frequency = &component.frequencies[v][u];
                let (coefficients, levels) = (&own[start..start + count], &levels[start..start + count]);
                let terms = &mut room.terms[..count];
                data_row(terms, coefficients, levels, marks, component, frequency, false);
                squares.push(lanes(terms));
                if component.charged {
                    data_row(terms, coefficients, levels, marks, component, frequency, true);
                    charged.push(lanes(terms));
                }
            }
        }
    }
}

/// `G*`'s terms of a row of coefficients of one frequency, each with the weight of
/// its block, flat where its mark is 1.
#[inline]
fn conjugate_row<const CHARGED: bool>(
    terms: &mut [f64],
    values: &[f64],
    levels: &[i16],
    marks: &[f64],
    component: &Component,
    frequency: &Frequency,
) {
    let count = terms.len();
    let (values, levels, marks) = (&values[..count], &levels[..count], &marks[..count]);
    for i in 0..count {
        let (ends, centre, inner) = component.model(f64::from(levels[i]), frequency);
        let weight = frequency.weight_of(marks[i]);
        terms[i] = conjugate_term::<CHARGED>(values[i], weight, ends, centre, frequency.charge, inner);
    }
}

/// `G`'s terms of a row of coefficients of one frequency: `w (c - centre)^2`, `w` the
/// weight of the coefficient's block, flat where its mark is 1, or where `charged`,
/// the cost of lying beyond the file's own interval.
#[inline]
fn data_row(
    terms: &mut [f64],
    values: &[f64],
    levels: &[i16],
    marks: &[f64],
    component: &Component,
    frequency: &Frequency,
    charged: bool,
) {
    let count = terms.len();
    let (values, levels, marks) = (&values[..count], &levels[..count], &marks[..count]);
    if charged {
        for i in 0..count {
            let (_, _, (lower, upper)) = component.model(f64::from(levels[i]), frequency);
            terms[i] = frequency.charge * beyond(values[i], lower, upper);
        }
    } else {
        for i in 0..count {
            let (_, centre, _) = component.model(f64::from(levels[i]), frequency);
            terms[i] = squared_term(values[i], frequency.weight_of(marks[i]), centre);
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
