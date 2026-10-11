// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! The kernels of the solvers: loops over streams of values, every value taking the
//! same operations as its neighbours, which the compiler turns into vector
//! instructions of the processor's width (docs/math.md, Arithmetic).
//!
//! A kernel takes its streams as slices of one length, cut before the loop, so that
//! the loop has no bounds checks and no branches but selects, and computes one value
//! of every output stream in each step. In the layout of the solvers (`planar`) the
//! streams are the planes of rows: the same column of every MCU across.
//!
//! The formulas at the top are the arithmetic of one value. The code of the natural
//! layout uses them too, so that the solvers and their definitions in the natural
//! layout agree to the last bit.

use std::ops::Range;

use crate::dct::{BASIS, multiply_add};
use crate::operators::{tensor_square, vector_square};

/// `x + step (across + down)`: the primal step at a sample, `step` being `tau gamma`
/// and `across + down` the divergence of the dual there.
#[inline]
#[must_use]
pub fn descent(x: f64, step: f64, across: f64, down: f64) -> f64 {
    multiply_add(step, across + down, x)
}

/// `2 new - old`: the extrapolated point.
#[inline]
#[must_use]
pub fn extrapolated(new: f64, old: f64) -> f64 {
    multiply_add(2.0, new, -old)
}

/// `p + step g`: the dual step at a sample, `step` being `sigma gamma`.
#[inline]
#[must_use]
pub fn ascent(p: f64, step: f64, g: f64) -> f64 {
    multiply_add(step, g, p)
}

/// `rho new + rest old`, `rest` being `1 - rho`: the relaxed step.
#[inline]
#[must_use]
pub fn relaxed(rho: f64, rest: f64, new: f64, old: f64) -> f64 {
    multiply_add(rho, new, rest * old)
}

/// The larger of a value and a bound: the bound where the value is below it.
#[inline]
#[must_use]
pub fn at_least(value: f64, bound: f64) -> f64 {
    if value < bound { bound } else { value }
}

/// The smaller of a value and a bound: the bound where the value is above it.
#[inline]
#[must_use]
pub fn at_most(value: f64, bound: f64) -> f64 {
    if value > bound { bound } else { value }
}

/// The largest of 0 and the values, in eight lanes: vector code, where a largest
/// taken one value after another would not be, its order kept for NaNs.
#[inline]
#[must_use]
pub fn most(values: &[f64]) -> f64 {
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

/// 1 for a value above 0, and 0 for one that is not, to count them: the comparison
/// and the count on one line, of the vector loop and of its remainder alike, which
/// the check of vector code pairs by their lines.
#[inline]
#[must_use]
#[expect(
    clippy::bool_to_int_with_if,
    reason = "usize::from is a line of the core library's, which the vector loop takes and its remainder not"
)]
pub fn above_zero(value: f64) -> usize {
    if value > 0.0 { 1 } else { 0 }
}

/// What a vector or tensor of the squared norm `square` is multiplied by to project
/// it onto the ball of the radius: `radius / |v|` beyond the ball, and 1 exactly within
/// it; one square root and one division.
#[inline]
#[must_use]
pub fn ball_scale(square: f64, radius: f64) -> f64 {
    let norm = square.sqrt();
    if norm > radius { radius / norm } else { 1.0 }
}

/// The end of a coefficient's interval from its level: `((q - 1/2) - slack) Q` below
/// and `((q + 1/2) + slack) Q` above, `half` being `-1/2` or `1/2` and `widen`
/// `-slack` or `slack`, and `shift` the level shift of DC (0 elsewhere); the
/// arithmetic of the model (docs/math.md, 1.2).
#[inline]
#[must_use]
pub fn interval_end(level: f64, half: f64, widen: f64, step: f64, shift: f64) -> f64 {
    ((level + half) + widen) * step + shift
}

/// The centre of the data term from a level: 0 for the level 0, and otherwise
/// `sign(q) (|q| - shrunk) Q`, `shrunk` being the MMSE shrinkage of the frequency (0
/// for the middles and for DC); `shift` is added (docs/math.md, 2.2).
#[inline]
#[must_use]
pub fn centre(level: f64, step: f64, shrunk: f64, shift: f64) -> f64 {
    let magnitude = (level.abs() - shrunk) * step;
    let signed = if level < 0.0 { -magnitude } else { magnitude };
    (if level == 0.0 { 0.0 } else { signed }) + shift
}

/// The proximal step of the data term at a coefficient, before its clip:
/// `(value + scaled centre) inverse`, `scaled` being `tau w` and `inverse`
/// `1 / (1 + tau w)` (docs/math.md, 4.1).
#[inline]
#[must_use]
pub fn proximal(value: f64, scaled: f64, inverse: f64, centre: f64) -> f64 {
    multiply_add(scaled, centre, value) * inverse
}

/// The proximal step with the cost of leaving the file's own interval: beyond that
/// interval, back towards it by `shrink`, but not past its end.
#[inline]
#[must_use]
pub fn charged(moved: f64, shrink: f64, inner_lower: f64, inner_upper: f64) -> f64 {
    if moved > inner_upper {
        at_least(moved - shrink, inner_upper)
    } else if moved < inner_lower {
        at_most(moved + shrink, inner_lower)
    } else {
        moved
    }
}

/// What the proximal map of a component takes of one frequency.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Frequency {
    /// `Q`.
    pub step: f64,
    /// The MMSE shrinkage of the centres, 0 for the middles.
    pub shrunk: f64,
    /// `tau w`.
    pub scaled: f64,
    /// `1 / (1 + tau w)`.
    pub inverse: f64,
    /// `tau` times the cost over `1 + tau w`.
    pub shrink: f64,
    /// The level shift of DC, 0 elsewhere.
    pub shift: f64,
}

/// The coefficient of the proximal map from its value before, and its level; where
/// `CHARGED`, with the cost of leaving the file's own interval.
#[inline]
fn prox_one<const CHARGED: bool>(value: f64, level: f64, frequency: &Frequency, slack: f64) -> f64 {
    let Frequency {
        step,
        shrunk,
        scaled,
        inverse,
        shrink,
        shift,
    } = *frequency;
    let moved = proximal(value, scaled, inverse, centre(level, step, shrunk, shift));
    let lower = interval_end(level, -0.5, -slack, step, shift);
    let upper = interval_end(level, 0.5, slack, step, shift);
    let kept = if CHARGED {
        let inner_lower = interval_end(level, -0.5, 0.0, step, shift);
        let inner_upper = interval_end(level, 0.5, 0.0, step, shift);
        charged(moved, shrink, inner_lower, inner_upper)
    } else {
        moved
    };
    at_most(at_least(kept, lower), upper)
}

/// `B[k][n]` of the basis, for the formulas below.
const fn b(k: usize, n: usize) -> f64 {
    BASIS[k][n]
}

/// The even frequency `k` of the 8-point DCT from the sums `s[n] = x[n] + x[7 - n]`,
/// and the odd from the differences, each by four products in this order
/// (docs/math.md, 1.1).
#[inline]
fn forward_half(k: usize, s: &[f64; 4]) -> f64 {
    multiply_add(
        b(k, 3),
        s[3],
        multiply_add(b(k, 2), s[2], multiply_add(b(k, 1), s[1], b(k, 0) * s[0])),
    )
}

/// The even part at position `n < 4` of the inverse 8-point DCT, from the even
/// frequencies `c[0], c[2], c[4], c[6]`, or the odd part from the odd ones, each by
/// four products in this order (docs/math.md, 1.1).
#[inline]
fn inverse_half(n: usize, first: usize, c: &[f64; 8]) -> f64 {
    multiply_add(
        b(first + 6, n),
        c[first + 6],
        multiply_add(
            b(first + 4, n),
            c[first + 4],
            multiply_add(b(first + 2, n), c[first + 2], b(first, n) * c[first]),
        ),
    )
}

/// The 8-point DCT of eight values.
#[inline]
fn forward8_values(x: &[f64; 8]) -> [f64; 8] {
    let s = [x[0] + x[7], x[1] + x[6], x[2] + x[5], x[3] + x[4]];
    let d = [x[0] - x[7], x[1] - x[6], x[2] - x[5], x[3] - x[4]];
    [
        forward_half(0, &s),
        forward_half(1, &d),
        forward_half(2, &s),
        forward_half(3, &d),
        forward_half(4, &s),
        forward_half(5, &d),
        forward_half(6, &s),
        forward_half(7, &d),
    ]
}

/// The inverse 8-point DCT of eight coefficients.
#[inline]
fn inverse8_values(c: &[f64; 8]) -> [f64; 8] {
    let even = [
        inverse_half(0, 0, c),
        inverse_half(1, 0, c),
        inverse_half(2, 0, c),
        inverse_half(3, 0, c),
    ];
    let odd = [
        inverse_half(0, 1, c),
        inverse_half(1, 1, c),
        inverse_half(2, 1, c),
        inverse_half(3, 1, c),
    ];
    [
        even[0] + odd[0],
        even[1] + odd[1],
        even[2] + odd[2],
        even[3] + odd[3],
        even[3] - odd[3],
        even[2] - odd[2],
        even[1] - odd[1],
        even[0] - odd[0],
    ]
}

/// The 8-point DCT across eight streams of a buffer, value by value, into eight
/// streams of another: `output[k][m]` is the sum of `B[k][n] input[n][m]`, stream `k`
/// of each from `k` times its stride on, `count` values each. Each stream is split off
/// the buffer after the one before, and none is taken from an array of streams.
#[inline]
pub fn forward8(input: &[f64], input_stride: usize, output: &mut [f64], output_stride: usize, count: usize) {
    let (i0, rest) = input.split_at(input_stride);
    let (i1, rest) = rest.split_at(input_stride);
    let (i2, rest) = rest.split_at(input_stride);
    let (i3, rest) = rest.split_at(input_stride);
    let (i4, rest) = rest.split_at(input_stride);
    let (i5, rest) = rest.split_at(input_stride);
    let (i6, i7) = rest.split_at(input_stride);
    let (o0, rest) = output.split_at_mut(output_stride);
    let (o1, rest) = rest.split_at_mut(output_stride);
    let (o2, rest) = rest.split_at_mut(output_stride);
    let (o3, rest) = rest.split_at_mut(output_stride);
    let (o4, rest) = rest.split_at_mut(output_stride);
    let (o5, rest) = rest.split_at_mut(output_stride);
    let (o6, o7) = rest.split_at_mut(output_stride);
    let (i0, i1, i2, i3) = (&i0[..count], &i1[..count], &i2[..count], &i3[..count]);
    let (i4, i5, i6, i7) = (&i4[..count], &i5[..count], &i6[..count], &i7[..count]);
    let (o0, o1, o2, o3) = (&mut o0[..count], &mut o1[..count], &mut o2[..count], &mut o3[..count]);
    let (o4, o5, o6, o7) = (&mut o4[..count], &mut o5[..count], &mut o6[..count], &mut o7[..count]);
    for m in 0..count {
        let c = forward8_values(&[i0[m], i1[m], i2[m], i3[m], i4[m], i5[m], i6[m], i7[m]]);
        o0[m] = c[0];
        o1[m] = c[1];
        o2[m] = c[2];
        o3[m] = c[3];
        o4[m] = c[4];
        o5[m] = c[5];
        o6[m] = c[6];
        o7[m] = c[7];
    }
}

/// The inverse 8-point DCT across eight streams of a buffer into eight of another, as
/// [`forward8`] takes them.
#[inline]
pub fn inverse8(input: &[f64], input_stride: usize, output: &mut [f64], output_stride: usize, count: usize) {
    let (i0, rest) = input.split_at(input_stride);
    let (i1, rest) = rest.split_at(input_stride);
    let (i2, rest) = rest.split_at(input_stride);
    let (i3, rest) = rest.split_at(input_stride);
    let (i4, rest) = rest.split_at(input_stride);
    let (i5, rest) = rest.split_at(input_stride);
    let (i6, i7) = rest.split_at(input_stride);
    let (o0, rest) = output.split_at_mut(output_stride);
    let (o1, rest) = rest.split_at_mut(output_stride);
    let (o2, rest) = rest.split_at_mut(output_stride);
    let (o3, rest) = rest.split_at_mut(output_stride);
    let (o4, rest) = rest.split_at_mut(output_stride);
    let (o5, rest) = rest.split_at_mut(output_stride);
    let (o6, o7) = rest.split_at_mut(output_stride);
    let (i0, i1, i2, i3) = (&i0[..count], &i1[..count], &i2[..count], &i3[..count]);
    let (i4, i5, i6, i7) = (&i4[..count], &i5[..count], &i6[..count], &i7[..count]);
    let (o0, o1, o2, o3) = (&mut o0[..count], &mut o1[..count], &mut o2[..count], &mut o3[..count]);
    let (o4, o5, o6, o7) = (&mut o4[..count], &mut o5[..count], &mut o6[..count], &mut o7[..count]);
    for m in 0..count {
        let x = inverse8_values(&[i0[m], i1[m], i2[m], i3[m], i4[m], i5[m], i6[m], i7[m]]);
        o0[m] = x[0];
        o1[m] = x[1];
        o2[m] = x[2];
        o3[m] = x[3];
        o4[m] = x[4];
        o5[m] = x[5];
        o6[m] = x[6];
        o7[m] = x[7];
    }
}

/// The proximal maps of `G` of the eight streams of a row of coefficients, in place:
/// stream `u` of `values` from `u stride` on and of `levels` from `u level_stride` on,
/// with the frequency `frequencies[u]`, `count` values each; where `CHARGED`, the slack
/// has a cost.
#[inline]
pub fn prox_row<const CHARGED: bool>(
    values: &mut [f64],
    stride: usize,
    levels: &[i16],
    level_stride: usize,
    frequencies: &[Frequency; 8],
    slack: f64,
    count: usize,
) {
    for (u, frequency) in frequencies.iter().enumerate() {
        let values = &mut values[u * stride..u * stride + count];
        let levels = &levels[u * level_stride..u * level_stride + count];
        for (value, &level) in values.iter_mut().zip(levels) {
            *value = prox_one::<CHARGED>(*value, f64::from(level), frequency, slack);
        }
    }
}

/// [`prox_row`] with flat blocks: the value at place `m` of each stream takes the
/// factors of `flat[u]` where `marks[m]` is 1 (its block is flat) and those of
/// `frequencies[u]` where it is 0. Each factor is `flat mark + own (1 - mark)`, which is
/// one of the two to the last bit -- a product by 1 or 0 and a sum with 0 are exact --
/// so that each value is that of its block's own map; and sums of products, where a
/// choice between the two would be made in the scalar remainder by a blend of 128 bits,
/// keep the arithmetic of the full width.
#[inline]
#[expect(
    clippy::too_many_arguments,
    reason = "the row takes the streams, the levels, both kinds of factors and the marks"
)]
pub fn prox_row_flat<const CHARGED: bool>(
    values: &mut [f64],
    stride: usize,
    levels: &[i16],
    level_stride: usize,
    frequencies: &[Frequency; 8],
    flat: &[Frequency; 8],
    marks: &[f64],
    slack: f64,
    count: usize,
) {
    let marks = &marks[..count];
    // Both kinds of factors are taken as values before the row, so that nothing in the
    // loop reads from a place that depends on the mark, which NEON has no vector load for.
    for (u, (&own, &flat)) in frequencies.iter().zip(flat).enumerate() {
        let values = &mut values[u * stride..u * stride + count];
        let levels = &levels[u * level_stride..u * level_stride + count];
        for ((value, &level), &mark) in values.iter_mut().zip(levels).zip(marks) {
            let other = 1.0 - mark;
            let taken = Frequency {
                scaled: multiply_add(flat.scaled, mark, own.scaled * other),
                inverse: multiply_add(flat.inverse, mark, own.inverse * other),
                shrink: multiply_add(flat.shrink, mark, own.shrink * other),
                ..own
            };
            *value = prox_one::<CHARGED>(*value, f64::from(level), &taken, slack);
        }
    }
}

/// The descent of one row of a channel's canvas: `x + step (across + down)`, `across`
/// the backward difference of `px` along the row, `down` that of the entries down from
/// the row's own (`own`, zeros at the last row) and the row above (`above`, zeros at
/// the first).
#[inline]
#[expect(
    clippy::too_many_arguments,
    reason = "the row takes the point, the dual rows, the step and the layout"
)]
pub fn descend_row(
    out: &mut [f64],
    x: &[f64],
    px: &[f64],
    own: &[f64],
    above: &[f64],
    step: f64,
    planes: usize,
    across: usize,
) {
    let width = planes * across;
    // The entry before is in the plane before, `across` places back ...
    let count = width - across;
    let (target, x_rest, here, before) = (&mut out[across..], &x[across..], &px[across..], &px[..count]);
    let (own_rest, above_rest) = (&own[across..], &above[across..]);
    for i in 0..count {
        target[i] = descent(x_rest[i], step, here[i] - before[i], own_rest[i] - above_rest[i]);
    }
    // ... but in plane 0 it is the last plane's, of the MCU before.
    let (target, x_rest, here, before) = (
        &mut out[1..across],
        &x[1..across],
        &px[1..across],
        &px[count..width - 1],
    );
    let (own_rest, above_rest) = (&own[1..across], &above[1..across]);
    for i in 0..across - 1 {
        target[i] = descent(x_rest[i], step, here[i] - before[i], own_rest[i] - above_rest[i]);
    }
    // The first sample of the row has no entry before it, and the last one no own entry.
    out[0] = descent(x[0], step, px[0], own[0] - above[0]);
    let (last, second) = (width - 1, count - 1);
    out[last] = descent(x[last], step, 0.0 - px[second], own[last] - above[last]);
}

/// The dual step of one row of a channel: `p + step grad bar`, the forward differences
/// of the extrapolated row `bar` along it and down to the row `below` (none at the
/// last row, where they are 0), and the scales of its projection onto the ball
/// ([`Scales`]).
#[inline(always)]
#[expect(
    clippy::inline_always,
    reason = "the kernel's variants are chosen where it is called, its loops vector code there"
)]
#[expect(
    clippy::too_many_arguments,
    reason = "the row takes the dual rows in and out, the extrapolated rows, the step, the layout and the scales"
)]
pub fn ascend_row(
    tx: &mut [f64],
    ty: &mut [f64],
    px: &[f64],
    py: &[f64],
    bar: &[f64],
    below: Option<&[f64]>,
    step: f64,
    planes: usize,
    across: usize,
    scales: Scales<'_>,
) {
    let (t, p, layout) = ((tx, ty), (px, py), (planes, across));
    // No row below, whose entries are not read: the row's own stand in for them.
    let rows = (bar, below.unwrap_or(bar));
    let Scales {
        row,
        radius,
        add,
        finish,
    } = scales;
    let scale = (row, radius);
    match (below.is_some(), add, finish) {
        (true, false, false) => ascend_rows::<true, false, false>(t, p, rows, step, layout, scale),
        (true, false, true) => ascend_rows::<true, false, true>(t, p, rows, step, layout, scale),
        (true, true, false) => ascend_rows::<true, true, false>(t, p, rows, step, layout, scale),
        (true, true, true) => ascend_rows::<true, true, true>(t, p, rows, step, layout, scale),
        (false, false, false) => ascend_rows::<false, false, false>(t, p, rows, step, layout, scale),
        (false, false, true) => ascend_rows::<false, false, true>(t, p, rows, step, layout, scale),
        (false, true, false) => ascend_rows::<false, true, false>(t, p, rows, step, layout, scale),
        (false, true, true) => ascend_rows::<false, true, true>(t, p, rows, step, layout, scale),
    }
}

/// Where a dual step takes the scales of its projection onto the ball of `radius`:
/// into `row`, a channel's own, or coupled channels' at each pixel in turn -- the
/// squares of the first channel written, those of each channel after it added in the
/// order of the channels, and the sums of the last made the scales (the projection of
/// docs/math.md, 5, a square root and a division where the ball is left).
#[derive(Debug)]
pub struct Scales<'a> {
    /// The row of the squares, and at last of the scales.
    pub row: &'a mut [f64],
    /// The radius of the ball.
    pub radius: f64,
    /// Whether the squares are added to the row's, those of the channels before.
    pub add: bool,
    /// Whether the sums are made the scales: a channel's own, or the last channel's.
    pub finish: bool,
}

impl<'a> Scales<'a> {
    /// A channel's own scales.
    #[must_use]
    pub fn own(row: &'a mut [f64], radius: f64) -> Self {
        Self {
            row,
            radius,
            add: false,
            finish: true,
        }
    }

    /// The part of channel `channel` of the scales of `count` coupled channels.
    #[must_use]
    pub fn coupled(row: &'a mut [f64], radius: f64, channel: usize, count: usize) -> Self {
        Self {
            row,
            radius,
            add: channel > 0,
            finish: channel + 1 == count,
        }
    }
}

/// A sample's part of [`Scales`], from the row's value there: its square, added to the
/// row's where `ADD`, and the scale of the sum where `FINISH`.
#[inline]
#[must_use]
pub fn scaled<const ADD: bool, const FINISH: bool>(row: f64, square: f64, radius: f64) -> f64 {
    let sum = if ADD { row + square } else { square };
    if FINISH { ball_scale(sum, radius) } else { sum }
}

/// [`ascend_row`] with the row below where `BELOW`, and differences down of 0
/// otherwise; its squares added where `ADD`, and made the scales where `FINISH`.
#[inline(always)]
#[expect(
    clippy::inline_always,
    reason = "a variant's loops belong in its kernel, where they are vector code with the kernel's other loops"
)]
fn ascend_rows<const BELOW: bool, const ADD: bool, const FINISH: bool>(
    (tx, ty): (&mut [f64], &mut [f64]),
    p: Pair<'_>,
    (bar, below): Pair<'_>,
    step: f64,
    (planes, across): (usize, usize),
    (scale, radius): (&mut [f64], f64),
) {
    let width = planes * across;
    let count = width - across;
    // The entry after is in the plane after, `across` places on ...
    ascend_stretch::<BELOW, ADD, FINISH>(
        (&mut tx[..count], &mut ty[..count]),
        p,
        (bar, &bar[across..], below),
        step,
        (&mut scale[..count], radius),
    );
    // ... but in the last plane it is plane 0's, of the MCU after.
    ascend_stretch::<BELOW, ADD, FINISH>(
        (&mut tx[count..width - 1], &mut ty[count..width - 1]),
        pair_from(p, count),
        (&bar[count..], &bar[1..], &below[count..]),
        step,
        (&mut scale[count..width - 1], radius),
    );
    // The last sample has no entry after it.
    let last = width - 1;
    let down = if BELOW { below[last] - bar[last] } else { 0.0 };
    let (x, y) = (ascent(p.0[last], step, 0.0), ascent(p.1[last], step, down));
    tx[last] = x;
    ty[last] = y;
    scale[last] = scaled::<ADD, FINISH>(scale[last], vector_square(x, y), radius);
}

/// A stretch of [`ascend_rows`]: the extrapolated canvas, the entries after its own
/// across and those below them, every stream as long as the outputs or longer.
#[inline(always)]
#[expect(
    clippy::inline_always,
    reason = "a stretch's loop belongs in its kernel, where it is vector code with the kernel's other loops"
)]
fn ascend_stretch<const BELOW: bool, const ADD: bool, const FINISH: bool>(
    (tx, ty): (&mut [f64], &mut [f64]),
    p: Pair<'_>,
    (bar, after, below): (&[f64], &[f64], &[f64]),
    step: f64,
    (scale, radius): (&mut [f64], f64),
) {
    let length = tx.len();
    let (ty, scale) = (&mut ty[..length], &mut scale[..length]);
    let (px, py) = (&p.0[..length], &p.1[..length]);
    let (here, after, below) = (&bar[..length], &after[..length], &below[..length]);
    for i in 0..length {
        let down = if BELOW { below[i] - here[i] } else { 0.0 };
        let (x, y) = (ascent(px[i], step, after[i] - here[i]), ascent(py[i], step, down));
        tx[i] = x;
        ty[i] = y;
        scale[i] = scaled::<ADD, FINISH>(scale[i], vector_square(x, y), radius);
    }
}

/// The projection of a channel's dual step `(tx, ty)` by `scale`, and the relaxation of
/// `(px, py)` towards the projected step, the output of the proximal step; where
/// `KEEP`, that output is written to `(out_px, out_py)`.
#[inline]
#[expect(
    clippy::too_many_arguments,
    reason = "the row takes the dual step, its scales, the dual and the outputs"
)]
#[expect(
    clippy::similar_names,
    reason = "the names are those of the formulas: p_x and p_y and the outputs of each"
)]
pub fn settle_row<const KEEP: bool>(
    tx: &[f64],
    ty: &[f64],
    scale: &[f64],
    px: &mut [f64],
    py: &mut [f64],
    out_px: &mut [f64],
    out_py: &mut [f64],
    rho: f64,
    rest: f64,
) {
    let width = tx.len();
    let (ty, scale) = (&ty[..width], &scale[..width]);
    let (px, py) = (&mut px[..width], &mut py[..width]);
    let kept = if KEEP { width } else { 0 };
    let (out_px, out_py) = (&mut out_px[..kept], &mut out_py[..kept]);
    for i in 0..width {
        let (dual_x, dual_y) = (tx[i] * scale[i], ty[i] * scale[i]);
        if KEEP {
            out_px[i] = dual_x;
            out_py[i] = dual_y;
        }
        px[i] = relaxed(rho, rest, dual_x, px[i]);
        py[i] = relaxed(rho, rest, dual_y, py[i]);
    }
}

/// `2 new - old` along a row: the extrapolated point.
#[inline]
pub fn extrapolate_row(bar: &mut [f64], new: &[f64], old: &[f64]) {
    for ((value, &fresh), &previous) in bar.iter_mut().zip(new).zip(old) {
        *value = extrapolated(fresh, previous);
    }
}

/// The extrapolation `2 new - x` of a row of the canvas into `bar`, and the relaxation
/// of `x` towards `new`, the output of the proximal step, which is written to `kept`
/// where `KEEP`: `x` is read last here in an iteration.
#[inline]
pub fn extrapolate_relax_row<const KEEP: bool>(
    bar: &mut [f64],
    new: &[f64],
    x: &mut [f64],
    kept: &mut [f64],
    rho: f64,
    rest: f64,
) {
    let width = bar.len();
    let (new, x) = (&new[..width], &mut x[..width]);
    let kept = &mut kept[..if KEEP { width } else { 0 }];
    for i in 0..width {
        let (fresh, old) = (new[i], x[i]);
        bar[i] = extrapolated(fresh, old);
        x[i] = relaxed(rho, rest, fresh, old);
        if KEEP {
            kept[i] = fresh;
        }
    }
}

/// `w + step (p + (across + down))`: TGV's step of `w` at a sample, `step` being
/// `tau gamma` and `across + down` an entry of the divergence of `r`.
#[inline]
#[must_use]
pub fn advanced(w: f64, step: f64, p: f64, across: f64, down: f64) -> f64 {
    multiply_add(step, p + (across + down), w)
}

/// The forward difference along a row of the planes: the entry after less the entry,
/// and 0 at the last sample.
#[inline]
pub fn forward_across(row: &[f64], out: &mut [f64], planes: usize, across: usize) {
    let width = planes * across;
    let count = width - across;
    // The entry after is in the plane after, `across` places on ...
    let (target, here, after) = (&mut out[..count], &row[..count], &row[across..width]);
    for i in 0..count {
        target[i] = after[i] - here[i];
    }
    // ... but in the last plane it is plane 0's, of the MCU after.
    let (target, here, after) = (&mut out[count..width - 1], &row[count..width - 1], &row[1..across]);
    for i in 0..across - 1 {
        target[i] = after[i] - here[i];
    }
    out[width - 1] = 0.0;
}

/// The backward difference along a row of the planes, the negative adjoint of
/// [`forward_across`]: the entry less the entry before, the first sample's own entry,
/// and at the last sample 0 less the entry before.
#[inline]
pub fn backward_across(row: &[f64], out: &mut [f64], planes: usize, across: usize) {
    let width = planes * across;
    let count = width - across;
    // The entry before is in the plane before, `across` places back ...
    let (target, here, before) = (&mut out[across..width], &row[across..width], &row[..count]);
    for i in 0..count {
        target[i] = here[i] - before[i];
    }
    // ... but in plane 0 it is the last plane's, of the MCU before.
    let (target, here, before) = (&mut out[1..across], &row[1..across], &row[count..width - 1]);
    for i in 0..across - 1 {
        target[i] = here[i] - before[i];
    }
    out[0] = row[0];
    out[width - 1] = 0.0 - row[count - 1];
}

/// The forward difference down at a row: the row below less the row, and 0 at the
/// last row, where there is none below.
#[inline]
pub fn forward_down(row: &[f64], below: Option<&[f64]>, out: &mut [f64]) {
    match below {
        Some(below) => {
            for ((entry, &here), &next) in out.iter_mut().zip(row).zip(below) {
                *entry = next - here;
            }
        }
        None => out.fill(0.0),
    }
}

/// The backward difference down at a row, the negative adjoint of [`forward_down`]:
/// the row's own entries (none at the last row) less those of the row above (none at
/// the first).
#[inline]
pub fn backward_down(own: Option<&[f64]>, above: Option<&[f64]>, out: &mut [f64]) {
    match (own, above) {
        (Some(own), Some(above)) => {
            for ((entry, &here), &before) in out.iter_mut().zip(own).zip(above) {
                *entry = here - before;
            }
        }
        (Some(own), None) => {
            let width = out.len();
            out.copy_from_slice(&own[..width]);
        }
        (None, Some(above)) => {
            for (entry, &before) in out.iter_mut().zip(above) {
                *entry = 0.0 - before;
            }
        }
        (None, None) => out.fill(0.0),
    }
}

/// TGV's step of `w` along a row, `new = w + step (p + div2 r)`: its extrapolation
/// `2 new - w` into `bar`, `w` relaxed towards it (`rho new + rest w`, the relaxation
/// `(rho, rest)`), and `new` written to `kept` where given; from the row of `r`, whose
/// entries are `r11`, `r22` and `r12`, and the rows of `r22` and `r12` below it (none
/// at the last row). The entries across take the forward differences of `r11` across
/// and of `r12` down, those down the forward differences of `r12` across and of `r22`
/// down: the entry after or below less the entry, and 0 where there is none.
#[inline(always)]
#[expect(
    clippy::inline_always,
    reason = "the kernel's variants are chosen where it is called, its loops vector code there"
)]
#[expect(
    clippy::too_many_arguments,
    reason = "the row takes the field in and out, the dual, the tensor and its rows below, the step and the layout"
)]
pub fn advance_row(
    bar: PairMut<'_>,
    w: PairMut<'_>,
    kept: Option<PairMut<'_>>,
    p: Pair<'_>,
    r: (&[f64], &[f64], &[f64]),
    below: Option<Pair<'_>>,
    (step, rho, rest): (f64, f64, f64),
    planes: usize,
    across: usize,
) {
    let (layout, steps) = ((planes, across), (step, rho, rest));
    // No row below, whose entries are not read: the row's own stand in for them.
    match (below, kept) {
        (Some(below), Some(kept)) => advance_rows::<true, true>((bar, w, kept), p, r, below, steps, layout),
        (Some(below), None) => advance_rows::<true, false>((bar, w, (&mut [], &mut [])), p, r, below, steps, layout),
        (None, Some(kept)) => advance_rows::<false, true>((bar, w, kept), p, r, (r.1, r.2), steps, layout),
        (None, None) => advance_rows::<false, false>((bar, w, (&mut [], &mut [])), p, r, (r.1, r.2), steps, layout),
    }
}

/// [`advance_row`] with the rows below where `BELOW`, and differences down of 0
/// otherwise; `new` kept where `KEEP`.
#[inline(always)]
#[expect(
    clippy::inline_always,
    reason = "a variant's loops belong in its kernel, where they are vector code with the kernel's other loops"
)]
#[expect(
    clippy::similar_names,
    reason = "the names are those of the formulas: the differences of the entries xy and yy"
)]
fn advance_rows<const BELOW: bool, const KEEP: bool>(
    ((bar_x, bar_y), (wx, wy), (kept_x, kept_y)): (PairMut<'_>, PairMut<'_>, PairMut<'_>),
    p: (&[f64], &[f64]),
    r: (&[f64], &[f64], &[f64]),
    below: (&[f64], &[f64]),
    (step, rho, rest): (f64, f64, f64),
    (planes, across): (usize, usize),
) {
    let width = planes * across;
    let count = width - across;
    let kept = if KEEP { width } else { 0 };
    let (kept_x, kept_y) = (&mut kept_x[..kept], &mut kept_y[..kept]);
    let part = |end: usize| if KEEP { end } else { 0 };
    // The entry after is in the plane after, `across` places on ...
    advance_stretch::<BELOW, KEEP>(
        (
            (&mut bar_x[..count], &mut bar_y[..count]),
            (&mut wx[..count], &mut wy[..count]),
            (&mut kept_x[..part(count)], &mut kept_y[..part(count)]),
        ),
        (p, r),
        (&r.0[across..], &r.2[across..]),
        below,
        (step, rho, rest),
    );
    // ... but in the last plane it is plane 0's, of the MCU after.
    advance_stretch::<BELOW, KEEP>(
        (
            (&mut bar_x[count..width - 1], &mut bar_y[count..width - 1]),
            (&mut wx[count..width - 1], &mut wy[count..width - 1]),
            (
                &mut kept_x[part(count)..part(width - 1)],
                &mut kept_y[part(count)..part(width - 1)],
            ),
        ),
        (
            (&p.0[count..], &p.1[count..]),
            (&r.0[count..], &r.1[count..], &r.2[count..]),
        ),
        (&r.0[1..], &r.2[1..]),
        (&below.0[count..], &below.1[count..]),
        (step, rho, rest),
    );
    // The last sample has no entry after it.
    let last = width - 1;
    let (down_xy, down_yy) = if BELOW {
        (below.1[last] - r.2[last], below.0[last] - r.1[last])
    } else {
        (0.0, 0.0)
    };
    let (old_x, old_y) = (wx[last], wy[last]);
    let x = advanced(old_x, step, p.0[last], 0.0, down_xy);
    let y = advanced(old_y, step, p.1[last], 0.0, down_yy);
    bar_x[last] = extrapolated(x, old_x);
    bar_y[last] = extrapolated(y, old_y);
    wx[last] = relaxed(rho, rest, x, old_x);
    wy[last] = relaxed(rho, rest, y, old_y);
    if KEEP {
        kept_x[last] = x;
        kept_y[last] = y;
    }
}

/// A stretch of [`advance_rows`], whose entries after those of `r11` and `r12` across
/// are `after`: every stream as long as the outputs or longer.
#[inline(always)]
#[expect(
    clippy::inline_always,
    reason = "a stretch's loop belongs in its kernel, where it is vector code with the kernel's other loops"
)]
#[expect(
    clippy::similar_names,
    reason = "the names are those of the formulas: the entries xx, yy and xy of the tensors"
)]
fn advance_stretch<const BELOW: bool, const KEEP: bool>(
    ((bar_x, bar_y), (wx, wy), (kept_x, kept_y)): (PairMut<'_>, PairMut<'_>, PairMut<'_>),
    (p, r): (Pair<'_>, Triple<'_>),
    after: (&[f64], &[f64]),
    below: (&[f64], &[f64]),
    (step, rho, rest): (f64, f64, f64),
) {
    let length = bar_x.len();
    let (bar_y, wx, wy) = (&mut bar_y[..length], &mut wx[..length], &mut wy[..length]);
    let kept = if KEEP { length } else { 0 };
    let (kept_x, kept_y) = (&mut kept_x[..kept], &mut kept_y[..kept]);
    let (px, py) = (&p.0[..length], &p.1[..length]);
    let (rxx, ryy, rxy) = (&r.0[..length], &r.1[..length], &r.2[..length]);
    let (xx_after, xy_after) = (&after.0[..length], &after.1[..length]);
    let (yy_below, xy_below) = (&below.0[..length], &below.1[..length]);
    for i in 0..length {
        let (down_xy, down_yy) = if BELOW {
            (xy_below[i] - rxy[i], yy_below[i] - ryy[i])
        } else {
            (0.0, 0.0)
        };
        let (old_x, old_y) = (wx[i], wy[i]);
        let x = advanced(old_x, step, px[i], xx_after[i] - rxx[i], down_xy);
        let y = advanced(old_y, step, py[i], xy_after[i] - rxy[i], down_yy);
        bar_x[i] = extrapolated(x, old_x);
        bar_y[i] = extrapolated(y, old_y);
        wx[i] = relaxed(rho, rest, x, old_x);
        wy[i] = relaxed(rho, rest, y, old_y);
        if KEEP {
            kept_x[i] = x;
            kept_y[i] = y;
        }
    }
}

/// TGV's step of `p` along a row, before its projection: `p + step (grad bar - bar_w)`,
/// from the extrapolated canvas `bar` and its row `below` (none at the last row), whose
/// forward differences across and down are the gradient (the entry after or below less
/// the entry, and 0 where there is none), and the extrapolated field `field`; and the
/// scales of its projection onto the ball ([`Scales`]).
#[inline(always)]
#[expect(
    clippy::inline_always,
    reason = "the kernel's variants are chosen where it is called, its loops vector code there"
)]
#[expect(
    clippy::too_many_arguments,
    reason = "the row takes the dual in and out, the extrapolated rows, the step, the layout and the scales"
)]
pub fn ascend_p_row(
    t: (&mut [f64], &mut [f64]),
    p: (&[f64], &[f64]),
    bar: &[f64],
    below: Option<&[f64]>,
    field: (&[f64], &[f64]),
    step: f64,
    planes: usize,
    across: usize,
    scales: Scales<'_>,
) {
    let layout = (planes, across);
    // No row below, whose entries are not read: the row's own stand in for them.
    let rows = (bar, below.unwrap_or(bar));
    let Scales {
        row,
        radius,
        add,
        finish,
    } = scales;
    let scale = (row, radius);
    match (below.is_some(), add, finish) {
        (true, false, false) => ascend_p_rows::<true, false, false>(t, p, rows, field, step, layout, scale),
        (true, false, true) => ascend_p_rows::<true, false, true>(t, p, rows, field, step, layout, scale),
        (true, true, false) => ascend_p_rows::<true, true, false>(t, p, rows, field, step, layout, scale),
        (true, true, true) => ascend_p_rows::<true, true, true>(t, p, rows, field, step, layout, scale),
        (false, false, false) => ascend_p_rows::<false, false, false>(t, p, rows, field, step, layout, scale),
        (false, false, true) => ascend_p_rows::<false, false, true>(t, p, rows, field, step, layout, scale),
        (false, true, false) => ascend_p_rows::<false, true, false>(t, p, rows, field, step, layout, scale),
        (false, true, true) => ascend_p_rows::<false, true, true>(t, p, rows, field, step, layout, scale),
    }
}

/// [`ascend_p_row`] with the row below where `BELOW`, and differences down of 0
/// otherwise; its squares added where `ADD`, and made the scales where `FINISH`.
#[inline(always)]
#[expect(
    clippy::inline_always,
    reason = "a variant's loops belong in its kernel, where they are vector code with the kernel's other loops"
)]
fn ascend_p_rows<const BELOW: bool, const ADD: bool, const FINISH: bool>(
    t: (&mut [f64], &mut [f64]),
    p: (&[f64], &[f64]),
    (bar, below): (&[f64], &[f64]),
    field: (&[f64], &[f64]),
    step: f64,
    (planes, across): (usize, usize),
    (scale, radius): (&mut [f64], f64),
) {
    let width = planes * across;
    let count = width - across;
    let (tx, ty) = t;
    // The entry after is in the plane after, `across` places on ...
    ascend_p_stretch::<BELOW, ADD, FINISH>(
        (&mut tx[..count], &mut ty[..count]),
        p,
        (bar, &bar[across..], below),
        field,
        step,
        (&mut scale[..count], radius),
    );
    // ... but in the last plane it is plane 0's, of the MCU after.
    ascend_p_stretch::<BELOW, ADD, FINISH>(
        (&mut tx[count..width - 1], &mut ty[count..width - 1]),
        (&p.0[count..], &p.1[count..]),
        (&bar[count..], &bar[1..], &below[count..]),
        (&field.0[count..], &field.1[count..]),
        step,
        (&mut scale[count..width - 1], radius),
    );
    // The last sample has no entry after it.
    let last = width - 1;
    let down = if BELOW { below[last] - bar[last] } else { 0.0 };
    let x = ascent(p.0[last], step, 0.0 - field.0[last]);
    let y = ascent(p.1[last], step, down - field.1[last]);
    tx[last] = x;
    ty[last] = y;
    scale[last] = scaled::<ADD, FINISH>(scale[last], vector_square(x, y), radius);
}

/// A stretch of [`ascend_p_rows`]: the extrapolated canvas, the entries after its own
/// across and those below them, every stream as long as the outputs or longer.
#[inline(always)]
#[expect(
    clippy::inline_always,
    reason = "a stretch's loop belongs in its kernel, where it is vector code with the kernel's other loops"
)]
fn ascend_p_stretch<const BELOW: bool, const ADD: bool, const FINISH: bool>(
    t: (&mut [f64], &mut [f64]),
    p: (&[f64], &[f64]),
    (bar, after, below): (&[f64], &[f64], &[f64]),
    field: (&[f64], &[f64]),
    step: f64,
    (scale, radius): (&mut [f64], f64),
) {
    let (tx, ty) = t;
    let length = tx.len();
    let ty = &mut ty[..length];
    let (px, py) = (&p.0[..length], &p.1[..length]);
    let (here, after, below) = (&bar[..length], &after[..length], &below[..length]);
    let (field_x, field_y) = (&field.0[..length], &field.1[..length]);
    let scale = &mut scale[..length];
    for i in 0..length {
        let down = if BELOW { below[i] - here[i] } else { 0.0 };
        let x = ascent(px[i], step, (after[i] - here[i]) - field_x[i]);
        let y = ascent(py[i], step, down - field_y[i]);
        tx[i] = x;
        ty[i] = y;
        scale[i] = scaled::<ADD, FINISH>(scale[i], vector_square(x, y), radius);
    }
}

/// TGV's step of `r` along a row, before its projection: `r + step E bar_w`, from the
/// extrapolated field `field` of the row, whose backward differences across are the
/// entry less the entry before (the first sample's own entry, and 0 less the entry
/// before at the last sample), and whose backward differences down are the row's own
/// entries `own` less those of the row `above`: `own` the field's, but zeros at the
/// last row, and `above` zeros at the first row; and the scales of its projection onto
/// the Frobenius ball ([`Scales`]).
#[inline(always)]
#[expect(
    clippy::inline_always,
    reason = "the kernel's variants are chosen where it is called, its loops vector code there"
)]
#[expect(
    clippy::too_many_arguments,
    reason = "the row takes the tensor in and out, the extrapolated rows, the step, the layout and the scales"
)]
pub fn ascend_r_row(
    t: (&mut [f64], &mut [f64], &mut [f64]),
    r: (&[f64], &[f64], &[f64]),
    field: (&[f64], &[f64]),
    own: (&[f64], &[f64]),
    above: (&[f64], &[f64]),
    step: f64,
    planes: usize,
    across: usize,
    scales: Scales<'_>,
) {
    let layout = (planes, across);
    let Scales {
        row,
        radius,
        add,
        finish,
    } = scales;
    let (rows, scale) = ((own, above), (row, radius));
    match (add, finish) {
        (false, false) => ascend_r_rows::<false, false>(t, r, field, rows, step, layout, scale),
        (false, true) => ascend_r_rows::<false, true>(t, r, field, rows, step, layout, scale),
        (true, false) => ascend_r_rows::<true, false>(t, r, field, rows, step, layout, scale),
        (true, true) => ascend_r_rows::<true, true>(t, r, field, rows, step, layout, scale),
    }
}

/// [`ascend_r_row`], its squares added where `ADD`, and made the scales where `FINISH`.
#[inline(always)]
#[expect(
    clippy::inline_always,
    reason = "a variant's loops belong in its kernel, where they are vector code with the kernel's other loops"
)]
fn ascend_r_rows<const ADD: bool, const FINISH: bool>(
    t: (&mut [f64], &mut [f64], &mut [f64]),
    r: (&[f64], &[f64], &[f64]),
    field: Pair<'_>,
    (own, above): (Pair<'_>, Pair<'_>),
    step: f64,
    (planes, across): (usize, usize),
    (scale, radius): (&mut [f64], f64),
) {
    let width = planes * across;
    let count = width - across;
    let (txx, tyy, txy) = t;
    // The entry before is in the plane before, `across` places back ...
    ascend_r_stretch::<ADD, FINISH>(
        (
            &mut txx[across..width - 1],
            &mut tyy[across..width - 1],
            &mut txy[across..width - 1],
        ),
        (&r.0[across..], &r.1[across..], &r.2[across..]),
        (pair_from(field, across), field),
        (pair_from(own, across), pair_from(above, across)),
        step,
        (&mut scale[across..width - 1], radius),
    );
    // ... but in plane 0 it is the last plane's, of the MCU before.
    ascend_r_stretch::<ADD, FINISH>(
        (&mut txx[1..across], &mut tyy[1..across], &mut txy[1..across]),
        (&r.0[1..], &r.1[1..], &r.2[1..]),
        (pair_from(field, 1), pair_from(field, count)),
        (pair_from(own, 1), pair_from(above, 1)),
        step,
        (&mut scale[1..across], radius),
    );
    // The first sample has no entry before it, and the last one no own entry across.
    let last = width - 1;
    for (i, across_x, across_y) in [
        (0, field.0[0], field.1[0]),
        (last, 0.0 - field.0[count - 1], 0.0 - field.1[count - 1]),
    ] {
        let (down_x, down_y) = (own.0[i] - above.0[i], own.1[i] - above.1[i]);
        let xx = ascent(r.0[i], step, across_x);
        let yy = ascent(r.1[i], step, down_y);
        let xy = ascent(r.2[i], step, f64::midpoint(down_x, across_y));
        txx[i] = xx;
        tyy[i] = yy;
        txy[i] = xy;
        scale[i] = scaled::<ADD, FINISH>(scale[i], tensor_square(xx, yy, xy), radius);
    }
}

/// Two rows: the entries across and down of a field, or of its step.
type Pair<'a> = (&'a [f64], &'a [f64]);

/// Two rows written: the entries across and down of a field, or of its step.
type PairMut<'a> = (&'a mut [f64], &'a mut [f64]);

/// Three rows: the entries `r11`, `r22` and `r12` of a tensor field.
type Triple<'a> = (&'a [f64], &'a [f64], &'a [f64]);

/// Three rows written: the entries `r11`, `r22` and `r12` of a tensor field.
type TripleMut<'a> = (&'a mut [f64], &'a mut [f64], &'a mut [f64]);

/// The rows of a pair from `start` on.
#[inline]
fn pair_from(pair: Pair<'_>, start: usize) -> Pair<'_> {
    (&pair.0[start..], &pair.1[start..])
}

/// A stretch of [`ascend_r_row`]: the extrapolated field and its entries before
/// across, and its own entries and those above for the differences down, every stream
/// as long as the outputs or longer.
#[inline(always)]
#[expect(
    clippy::inline_always,
    reason = "a stretch's loop belongs in its kernel, where it is vector code with the kernel's other loops"
)]
fn ascend_r_stretch<const ADD: bool, const FINISH: bool>(
    t: (&mut [f64], &mut [f64], &mut [f64]),
    r: (&[f64], &[f64], &[f64]),
    (field, before): (Pair<'_>, Pair<'_>),
    (own, above): (Pair<'_>, Pair<'_>),
    step: f64,
    (scale, radius): (&mut [f64], f64),
) {
    let (txx, tyy, txy) = t;
    let length = txx.len();
    let (tyy, txy) = (&mut tyy[..length], &mut txy[..length]);
    let (rxx, ryy, rxy) = (&r.0[..length], &r.1[..length], &r.2[..length]);
    let (field_x, field_y) = (&field.0[..length], &field.1[..length]);
    let (before_x, before_y) = (&before.0[..length], &before.1[..length]);
    let (own_x, own_y) = (&own.0[..length], &own.1[..length]);
    let (above_x, above_y) = (&above.0[..length], &above.1[..length]);
    let scale = &mut scale[..length];
    for i in 0..length {
        let (across_x, across_y) = (field_x[i] - before_x[i], field_y[i] - before_y[i]);
        let (down_x, down_y) = (own_x[i] - above_x[i], own_y[i] - above_y[i]);
        let xx = ascent(rxx[i], step, across_x);
        let yy = ascent(ryy[i], step, down_y);
        let xy = ascent(rxy[i], step, f64::midpoint(down_x, across_y));
        txx[i] = xx;
        tyy[i] = yy;
        txy[i] = xy;
        scale[i] = scaled::<ADD, FINISH>(scale[i], tensor_square(xx, yy, xy), radius);
    }
}

/// The dual step of a row of a channel whose projection is its own, projected and
/// relaxed in the same loop: `p + step (grad bar - field)`, TV's without a field
/// ([`ascend_row`]) and TGV's with the extrapolated field ([`ascend_p_row`]); its
/// projection onto the ball of `radius` ([`Scales::own`], [`settle_row`]); `p` relaxed
/// towards it in place (`rho`, `rest`); and the projected step written to `kept` where
/// given. Every value is computed as by those kernels one after another.
#[inline(always)]
#[expect(
    clippy::inline_always,
    reason = "the kernel's variants are chosen where it is called, its loops vector code there"
)]
#[expect(
    clippy::too_many_arguments,
    reason = "the row takes the dual in and out, the outputs, the extrapolated rows, the step, the projection and the layout"
)]
pub fn dual_p_row(
    p: PairMut<'_>,
    kept: Option<PairMut<'_>>,
    bar: &[f64],
    below: Option<&[f64]>,
    field: Option<Pair<'_>>,
    step: f64,
    (radius, rho, rest): (f64, f64, f64),
    planes: usize,
    across: usize,
) {
    let (layout, model) = ((planes, across), (step, radius, rho, rest));
    // No row below, whose entries are not read: the row's own stand in for them; and
    // no field, whose entries are not read: the canvas stands in for it.
    let rows = (bar, below.unwrap_or(bar));
    let field_rows = field.unwrap_or((bar, bar));
    let kept_rows = (kept.is_some(), kept.unwrap_or((&mut [], &mut [])));
    match (below.is_some(), field.is_some()) {
        (true, true) => dual_p_rows::<true, true>(p, kept_rows, rows, field_rows, model, layout),
        (true, false) => dual_p_rows::<true, false>(p, kept_rows, rows, field_rows, model, layout),
        (false, true) => dual_p_rows::<false, true>(p, kept_rows, rows, field_rows, model, layout),
        (false, false) => dual_p_rows::<false, false>(p, kept_rows, rows, field_rows, model, layout),
    }
}

/// [`dual_p_row`] with the row below where `BELOW` and the field where `FIELD`; the
/// outputs kept where the first of `kept` says so.
#[inline(always)]
#[expect(
    clippy::inline_always,
    reason = "a variant's loops belong in its kernel, where they are vector code with the kernel's other loops"
)]
fn dual_p_rows<const BELOW: bool, const FIELD: bool>(
    p: PairMut<'_>,
    (keep, kept): (bool, PairMut<'_>),
    rows: Pair<'_>,
    field: Pair<'_>,
    model: (f64, f64, f64, f64),
    layout: (usize, usize),
) {
    if keep {
        dual_p_kept::<BELOW, FIELD, true>(p, kept, rows, field, model, layout);
    } else {
        dual_p_kept::<BELOW, FIELD, false>(p, kept, rows, field, model, layout);
    }
}

/// [`dual_p_rows`], the outputs kept where `KEEP`.
#[inline(always)]
#[expect(
    clippy::inline_always,
    reason = "a variant's loops belong in its kernel, where they are vector code with the kernel's other loops"
)]
fn dual_p_kept<const BELOW: bool, const FIELD: bool, const KEEP: bool>(
    (px, py): PairMut<'_>,
    (kept_x, kept_y): PairMut<'_>,
    (bar, below): Pair<'_>,
    field: Pair<'_>,
    model: (f64, f64, f64, f64),
    (planes, across): (usize, usize),
) {
    let width = planes * across;
    let count = width - across;
    let part = |end: usize| if KEEP { end } else { 0 };
    // The entry after is in the plane after, `across` places on ...
    dual_p_stretch::<BELOW, FIELD, KEEP>(
        (&mut px[..count], &mut py[..count]),
        (&mut kept_x[..part(count)], &mut kept_y[..part(count)]),
        (bar, &bar[across..], below),
        field,
        model,
    );
    // ... but in the last plane it is plane 0's, of the MCU after.
    dual_p_stretch::<BELOW, FIELD, KEEP>(
        (&mut px[count..width - 1], &mut py[count..width - 1]),
        (
            &mut kept_x[part(count)..part(width - 1)],
            &mut kept_y[part(count)..part(width - 1)],
        ),
        (&bar[count..], &bar[1..], &below[count..]),
        pair_from(field, count),
        model,
    );
    // The last sample has no entry after it.
    let last = width - 1;
    let (step, radius, rho, rest) = model;
    let down = if BELOW { below[last] - bar[last] } else { 0.0 };
    let (across_x, down_y) = if FIELD {
        (0.0 - field.0[last], down - field.1[last])
    } else {
        (0.0, down)
    };
    let (old_x, old_y) = (px[last], py[last]);
    let (x, y) = (ascent(old_x, step, across_x), ascent(old_y, step, down_y));
    let scale = ball_scale(vector_square(x, y), radius);
    let (dual_x, dual_y) = (x * scale, y * scale);
    px[last] = relaxed(rho, rest, dual_x, old_x);
    py[last] = relaxed(rho, rest, dual_y, old_y);
    if KEEP {
        kept_x[last] = dual_x;
        kept_y[last] = dual_y;
    }
}

/// A stretch of [`dual_p_kept`]: the dual and its outputs, the extrapolated canvas,
/// the entries after its own across and those below them, and the field, every stream
/// as long as the dual or longer.
#[inline(always)]
#[expect(
    clippy::inline_always,
    reason = "a stretch's loop belongs in its kernel, where it is vector code with the kernel's other loops"
)]
fn dual_p_stretch<const BELOW: bool, const FIELD: bool, const KEEP: bool>(
    (px, py): PairMut<'_>,
    (kept_x, kept_y): PairMut<'_>,
    (bar, after, below): Triple<'_>,
    field: Pair<'_>,
    (step, radius, rho, rest): (f64, f64, f64, f64),
) {
    let length = px.len();
    let py = &mut py[..length];
    let kept = if KEEP { length } else { 0 };
    let (kept_x, kept_y) = (&mut kept_x[..kept], &mut kept_y[..kept]);
    let (here, after, below) = (&bar[..length], &after[..length], &below[..length]);
    let (field_x, field_y) = (&field.0[..length], &field.1[..length]);
    for i in 0..length {
        let down = if BELOW { below[i] - here[i] } else { 0.0 };
        let (across_x, down_y) = if FIELD {
            ((after[i] - here[i]) - field_x[i], down - field_y[i])
        } else {
            (after[i] - here[i], down)
        };
        let (old_x, old_y) = (px[i], py[i]);
        let (x, y) = (ascent(old_x, step, across_x), ascent(old_y, step, down_y));
        let scale = ball_scale(vector_square(x, y), radius);
        let (dual_x, dual_y) = (x * scale, y * scale);
        px[i] = relaxed(rho, rest, dual_x, old_x);
        py[i] = relaxed(rho, rest, dual_y, old_y);
        if KEEP {
            kept_x[i] = dual_x;
            kept_y[i] = dual_y;
        }
    }
}

/// TGV's step of `r` of a row of a channel whose projection is its own, projected and
/// relaxed in the same loop: the step of [`ascend_r_row`], its projection onto the
/// Frobenius ball of `radius` ([`Scales::own`], [`settle_tensor_row`]), `r` relaxed
/// towards it in place (`rho`, `rest`), and the projected step written to `kept`
/// where given. Every value is computed as by those kernels one after another.
#[inline(always)]
#[expect(
    clippy::inline_always,
    reason = "the kernel's variants are chosen where it is called, its loops vector code there"
)]
#[expect(
    clippy::too_many_arguments,
    reason = "the row takes the tensor in and out, the outputs, the extrapolated rows, the step, the projection and the layout"
)]
pub fn dual_r_row(
    r: TripleMut<'_>,
    kept: Option<TripleMut<'_>>,
    field: Pair<'_>,
    own: Pair<'_>,
    above: Pair<'_>,
    step: f64,
    (radius, rho, rest): (f64, f64, f64),
    planes: usize,
    across: usize,
) {
    let (rows, model, layout) = ((field, own, above), (step, radius, rho, rest), (planes, across));
    match kept {
        Some(kept) => dual_r_rows::<true>(r, kept, rows, model, layout),
        None => dual_r_rows::<false>(r, (&mut [], &mut [], &mut []), rows, model, layout),
    }
}

/// [`dual_r_row`], the outputs kept where `KEEP`.
#[inline(always)]
#[expect(
    clippy::inline_always,
    reason = "a variant's loops belong in its kernel, where they are vector code with the kernel's other loops"
)]
#[expect(
    clippy::similar_names,
    reason = "the names are those of the formulas: the entries xx, yy and xy of the tensors"
)]
fn dual_r_rows<const KEEP: bool>(
    (rxx, ryy, rxy): TripleMut<'_>,
    (kept_xx, kept_yy, kept_xy): TripleMut<'_>,
    (field, own, above): (Pair<'_>, Pair<'_>, Pair<'_>),
    model: (f64, f64, f64, f64),
    (planes, across): (usize, usize),
) {
    let width = planes * across;
    let count = width - across;
    let part = |range: Range<usize>| if KEEP { range } else { 0..0 };
    // The entry before is in the plane before, `across` places back ...
    let main = part(across..width - 1);
    dual_r_stretch::<KEEP>(
        (
            &mut rxx[across..width - 1],
            &mut ryy[across..width - 1],
            &mut rxy[across..width - 1],
        ),
        (
            &mut kept_xx[main.clone()],
            &mut kept_yy[main.clone()],
            &mut kept_xy[main],
        ),
        (pair_from(field, across), field),
        (pair_from(own, across), pair_from(above, across)),
        model,
    );
    // ... but in plane 0 it is the last plane's, of the MCU before.
    let first = part(1..across);
    dual_r_stretch::<KEEP>(
        (&mut rxx[1..across], &mut ryy[1..across], &mut rxy[1..across]),
        (
            &mut kept_xx[first.clone()],
            &mut kept_yy[first.clone()],
            &mut kept_xy[first],
        ),
        (pair_from(field, 1), pair_from(field, count)),
        (pair_from(own, 1), pair_from(above, 1)),
        model,
    );
    // The first sample has no entry before it, and the last one no own entry across.
    let (step, radius, rho, rest) = model;
    let last = width - 1;
    for (i, across_x, across_y) in [
        (0, field.0[0], field.1[0]),
        (last, 0.0 - field.0[count - 1], 0.0 - field.1[count - 1]),
    ] {
        let (down_x, down_y) = (own.0[i] - above.0[i], own.1[i] - above.1[i]);
        let (old_xx, old_yy, old_xy) = (rxx[i], ryy[i], rxy[i]);
        let xx = ascent(old_xx, step, across_x);
        let yy = ascent(old_yy, step, down_y);
        let xy = ascent(old_xy, step, f64::midpoint(down_x, across_y));
        let scale = ball_scale(tensor_square(xx, yy, xy), radius);
        let (dual_xx, dual_yy, dual_xy) = (xx * scale, yy * scale, xy * scale);
        rxx[i] = relaxed(rho, rest, dual_xx, old_xx);
        ryy[i] = relaxed(rho, rest, dual_yy, old_yy);
        rxy[i] = relaxed(rho, rest, dual_xy, old_xy);
        if KEEP {
            kept_xx[i] = dual_xx;
            kept_yy[i] = dual_yy;
            kept_xy[i] = dual_xy;
        }
    }
}

/// A stretch of [`dual_r_rows`]: the tensor and its outputs, the extrapolated field and
/// its entries before across, and its own entries and those above for the differences
/// down, every stream as long as the tensor or longer.
#[inline(always)]
#[expect(
    clippy::inline_always,
    reason = "a stretch's loop belongs in its kernel, where it is vector code with the kernel's other loops"
)]
#[expect(
    clippy::similar_names,
    reason = "the names are those of the formulas: the entries xx, yy and xy of the tensors"
)]
fn dual_r_stretch<const KEEP: bool>(
    (rxx, ryy, rxy): TripleMut<'_>,
    (kept_xx, kept_yy, kept_xy): TripleMut<'_>,
    (field, before): (Pair<'_>, Pair<'_>),
    (own, above): (Pair<'_>, Pair<'_>),
    (step, radius, rho, rest): (f64, f64, f64, f64),
) {
    let length = rxx.len();
    let (ryy, rxy) = (&mut ryy[..length], &mut rxy[..length]);
    let kept = if KEEP { length } else { 0 };
    let (kept_xx, kept_yy, kept_xy) = (&mut kept_xx[..kept], &mut kept_yy[..kept], &mut kept_xy[..kept]);
    let (field_x, field_y) = (&field.0[..length], &field.1[..length]);
    let (before_x, before_y) = (&before.0[..length], &before.1[..length]);
    let (own_x, own_y) = (&own.0[..length], &own.1[..length]);
    let (above_x, above_y) = (&above.0[..length], &above.1[..length]);
    for i in 0..length {
        let (across_x, across_y) = (field_x[i] - before_x[i], field_y[i] - before_y[i]);
        let (down_x, down_y) = (own_x[i] - above_x[i], own_y[i] - above_y[i]);
        let (old_xx, old_yy, old_xy) = (rxx[i], ryy[i], rxy[i]);
        let xx = ascent(old_xx, step, across_x);
        let yy = ascent(old_yy, step, down_y);
        let xy = ascent(old_xy, step, f64::midpoint(down_x, across_y));
        let scale = ball_scale(tensor_square(xx, yy, xy), radius);
        let (dual_xx, dual_yy, dual_xy) = (xx * scale, yy * scale, xy * scale);
        rxx[i] = relaxed(rho, rest, dual_xx, old_xx);
        ryy[i] = relaxed(rho, rest, dual_yy, old_yy);
        rxy[i] = relaxed(rho, rest, dual_xy, old_xy);
        if KEEP {
            kept_xx[i] = dual_xx;
            kept_yy[i] = dual_yy;
            kept_xy[i] = dual_xy;
        }
    }
}

/// The projection of a channel's tensor step `(txx, tyy, txy)` by `scale`, and the
/// relaxation of `r`; where `KEEP`, the projected step is written to `kept`.
#[inline]
#[expect(
    clippy::similar_names,
    reason = "the names are those of the formulas: the entries xx, yy and xy of the tensors, and their differences"
)]
pub fn settle_tensor_row<const KEEP: bool>(
    t: (&[f64], &[f64], &[f64]),
    scale: &[f64],
    r: (&mut [f64], &mut [f64], &mut [f64]),
    kept: (&mut [f64], &mut [f64], &mut [f64]),
    rho: f64,
    rest: f64,
) {
    let (txx, tyy, txy) = t;
    let width = txx.len();
    let (tyy, txy, scale) = (&tyy[..width], &txy[..width], &scale[..width]);
    let ((rxx, ryy, rxy), (out_xx, out_yy, out_xy)) = (r, kept);
    let (rxx, ryy, rxy) = (&mut rxx[..width], &mut ryy[..width], &mut rxy[..width]);
    let length = if KEEP { width } else { 0 };
    let (out_xx, out_yy, out_xy) = (&mut out_xx[..length], &mut out_yy[..length], &mut out_xy[..length]);
    for i in 0..width {
        let (xx, yy, xy) = (txx[i] * scale[i], tyy[i] * scale[i], txy[i] * scale[i]);
        if KEEP {
            out_xx[i] = xx;
            out_yy[i] = yy;
            out_xy[i] = xy;
        }
        rxx[i] = relaxed(rho, rest, xx, rxx[i]);
        ryy[i] = relaxed(rho, rest, yy, ryy[i]);
        rxy[i] = relaxed(rho, rest, xy, rxy[i]);
    }
}

/// The term of `G*` of one coefficient, given its coefficient `s` of `D xi`
/// (docs/math.md, 4.1): the most that `s c - (w/2) (c - centre)^2` reaches over the
/// interval `ends`, and where `CHARGED`, less `charge` times how far `c` lies beyond
/// the file's own interval `inner`.
#[inline]
#[must_use]
pub fn conjugate_term<const CHARGED: bool>(
    s: f64,
    weight: f64,
    ends: (f64, f64),
    centre: f64,
    charge: f64,
    inner: (f64, f64),
) -> f64 {
    let (lower, upper) = ends;
    if !CHARGED {
        if weight > 0.0 {
            let best = at_most(at_least(centre + s / weight, lower), upper);
            let difference = best - centre;
            return s * best - 0.5 * weight * (difference * difference);
        }
        // With no weight, the largest of s c over the interval is at one of its ends.
        return at_least(s * lower, s * upper);
    }
    let (inner_lower, inner_upper) = inner;
    // The largest of s c - (w/2) (c - centre)^2 - cost dist(c, inner) over the widened
    // interval: at the quadratic's own top where that is within the inner interval;
    // beyond it, at the top of the piece with the cost, but not before the inner end;
    // with no weight, at the end of the piece whose slope keeps its sign.
    let best = if weight > 0.0 {
        let top = centre + s / weight;
        if top > inner_upper {
            at_most(at_least(centre + (s - charge) / weight, inner_upper), upper)
        } else if top < inner_lower {
            at_least(at_most(centre + (s + charge) / weight, inner_lower), lower)
        } else {
            top
        }
    } else if s > charge {
        upper
    } else if s > 0.0 {
        inner_upper
    } else if s < -charge {
        lower
    } else {
        inner_lower
    };
    let difference = best - centre;
    let beyond = at_least(inner_lower - best, 0.0) + at_least(best - inner_upper, 0.0);
    s * best - 0.5 * weight * (difference * difference) - charge * beyond
}

/// The term of the data term `G` of one coefficient within its interval: `w (c -
/// centre)^2`, which `G` halves.
#[inline]
#[must_use]
pub fn squared_term(value: f64, weight: f64, centre: f64) -> f64 {
    let difference = value - centre;
    weight * difference * difference
}

/// How far a coefficient lies beyond an interval: 0 within it.
#[inline]
#[must_use]
pub fn beyond(value: f64, lower: f64, upper: f64) -> f64 {
    at_least(lower - value, 0.0) + at_least(value - upper, 0.0)
}
