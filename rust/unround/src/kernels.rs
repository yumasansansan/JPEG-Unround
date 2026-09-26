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
/// last row, where they are 0).
#[inline]
#[expect(
    clippy::too_many_arguments,
    reason = "the row takes the dual rows in and out, the extrapolated rows, the step and the layout"
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
) {
    let width = planes * across;
    // The entry after is in the plane after, `across` places on ...
    let count = width - across;
    let (target, p, here, after) = (&mut tx[..count], &px[..count], &bar[..count], &bar[across..]);
    for i in 0..count {
        target[i] = ascent(p[i], step, after[i] - here[i]);
    }
    // ... but in the last plane it is plane 0's, of the MCU after.
    let (target, p, here, after) = (
        &mut tx[count..width - 1],
        &px[count..width - 1],
        &bar[count..width - 1],
        &bar[1..across],
    );
    for i in 0..across - 1 {
        target[i] = ascent(p[i], step, after[i] - here[i]);
    }
    tx[width - 1] = ascent(px[width - 1], step, 0.0);
    match below {
        Some(below) => {
            for (((t, &p), &next), &here) in ty.iter_mut().zip(&py[..width]).zip(&below[..width]).zip(&bar[..width]) {
                *t = ascent(p, step, next - here);
            }
        }
        None => {
            for (t, &p) in ty.iter_mut().zip(&py[..width]) {
                *t = ascent(p, step, 0.0);
            }
        }
    }
}

/// What projects the dual rows of every channel onto the balls of the radius, into
/// `scale`: over all the channels at a pixel where coupled (one row for all), each
/// channel's own otherwise (a row for each).
#[inline]
pub fn scale_rows(tx: &[f64], ty: &[f64], scale: &mut [f64], width: usize, count: usize, radius: f64, coupled: bool) {
    if !coupled || count == 1 {
        for ((scale, &x), &y) in scale.iter_mut().zip(tx).zip(ty) {
            *scale = ball_scale(vector_square(x, y), radius);
        }
        return;
    }
    match count {
        2 => scale_coupled::<2>(tx, ty, &mut scale[..width], width, radius),
        3 => scale_coupled::<3>(tx, ty, &mut scale[..width], width, radius),
        4 => scale_coupled::<4>(tx, ty, &mut scale[..width], width, radius),
        _ => {
            // The squares added channel after channel, and then their scales.
            let scale = &mut scale[..width];
            scale.fill(0.0);
            for (xs, ys) in tx.chunks_exact(width).zip(ty.chunks_exact(width)).take(count) {
                for ((square, &x), &y) in scale.iter_mut().zip(xs).zip(ys) {
                    *square += vector_square(x, y);
                }
            }
            for value in scale.iter_mut() {
                *value = ball_scale(*value, radius);
            }
        }
    }
}

/// The scales of the coupled projection of `C` channels' dual rows.
#[inline]
pub fn scale_coupled<const C: usize>(tx: &[f64], ty: &[f64], scale: &mut [f64], width: usize, radius: f64) {
    let xs: [&[f64]; C] = std::array::from_fn(|c| &tx[c * width..(c + 1) * width]);
    let ys: [&[f64]; C] = std::array::from_fn(|c| &ty[c * width..(c + 1) * width]);
    for (m, scale) in scale[..width].iter_mut().enumerate() {
        let mut square = 0.0;
        for c in 0..C {
            square += vector_square(xs[c][m], ys[c][m]);
        }
        *scale = ball_scale(square, radius);
    }
}

/// The projection of a channel's dual step `(tx, ty)` by `scale`, and the relaxation of
/// its point `(x, px, py)` towards the outputs of the proximal steps, `new` and the
/// projected step; where `KEEP`, those outputs are written to `(out_x, out_px, out_py)`.
#[inline]
#[expect(
    clippy::too_many_arguments,
    reason = "the row takes the dual step, its scales, the new canvas, the point and the outputs"
)]
#[expect(
    clippy::similar_names,
    reason = "the names are those of the formulas: x, p_x and p_y and the outputs of each"
)]
pub fn settle_row<const KEEP: bool>(
    tx: &[f64],
    ty: &[f64],
    scale: &[f64],
    new: &[f64],
    x: &mut [f64],
    px: &mut [f64],
    py: &mut [f64],
    out_x: &mut [f64],
    out_px: &mut [f64],
    out_py: &mut [f64],
    rho: f64,
    rest: f64,
) {
    let width = tx.len();
    let (ty, scale, new) = (&ty[..width], &scale[..width], &new[..width]);
    let (x, px, py) = (&mut x[..width], &mut px[..width], &mut py[..width]);
    let kept = if KEEP { width } else { 0 };
    let (out_x, out_px, out_py) = (&mut out_x[..kept], &mut out_px[..kept], &mut out_py[..kept]);
    for i in 0..width {
        let (dual_x, dual_y) = (tx[i] * scale[i], ty[i] * scale[i]);
        if KEEP {
            out_x[i] = new[i];
            out_px[i] = dual_x;
            out_py[i] = dual_y;
        }
        x[i] = relaxed(rho, rest, new[i], x[i]);
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

/// TGV's step of `w` along a row: `w + step (p + div2 r)`, from the differences of `r`:
/// `across_xx` and `down_xy` for the entries across, `across_xy` and `down_yy` for
/// those down.
#[inline]
#[expect(
    clippy::too_many_arguments,
    reason = "the row takes the field in and out, the dual and the differences of r"
)]
#[expect(
    clippy::similar_names,
    reason = "the names are those of the formulas: the entries xx, yy and xy of the tensors, and their differences"
)]
pub fn advance_row(
    out_x: &mut [f64],
    out_y: &mut [f64],
    w: (&[f64], &[f64]),
    p: (&[f64], &[f64]),
    across_xx: &[f64],
    down_xy: &[f64],
    across_xy: &[f64],
    down_yy: &[f64],
    step: f64,
) {
    let width = out_x.len();
    let (wx, wy, px, py) = (&w.0[..width], &w.1[..width], &p.0[..width], &p.1[..width]);
    let (across_xx, down_xy) = (&across_xx[..width], &down_xy[..width]);
    let (across_xy, down_yy, out_y) = (&across_xy[..width], &down_yy[..width], &mut out_y[..width]);
    for i in 0..width {
        out_x[i] = advanced(wx[i], step, px[i], across_xx[i], down_xy[i]);
        out_y[i] = advanced(wy[i], step, py[i], across_xy[i], down_yy[i]);
    }
}

/// TGV's step of `p` along a row, before its projection: `p + step (grad bar - bar_w)`,
/// from the forward differences of the extrapolated canvas and the extrapolated field.
#[inline]
pub fn ascend_p_row(
    tx: &mut [f64],
    ty: &mut [f64],
    p: (&[f64], &[f64]),
    gradient: (&[f64], &[f64]),
    field: (&[f64], &[f64]),
    step: f64,
) {
    let width = tx.len();
    let (px, py) = (&p.0[..width], &p.1[..width]);
    let (across, down) = (&gradient.0[..width], &gradient.1[..width]);
    let (field_x, field_y, ty) = (&field.0[..width], &field.1[..width], &mut ty[..width]);
    for i in 0..width {
        tx[i] = ascent(px[i], step, across[i] - field_x[i]);
        ty[i] = ascent(py[i], step, down[i] - field_y[i]);
    }
}

/// TGV's step of `r` along a row, before its projection: `r + step E bar_w`, from the
/// backward differences of the extrapolated field: `across_x` and `down_x` of its
/// entries across, `across_y` and `down_y` of those down.
#[inline]
pub fn ascend_r_row(
    out: (&mut [f64], &mut [f64], &mut [f64]),
    r: (&[f64], &[f64], &[f64]),
    across_x: &[f64],
    down_x: &[f64],
    across_y: &[f64],
    down_y: &[f64],
    step: f64,
) {
    let (txx, tyy, txy) = out;
    let width = txx.len();
    let (tyy, txy) = (&mut tyy[..width], &mut txy[..width]);
    let (rxx, ryy, rxy) = (&r.0[..width], &r.1[..width], &r.2[..width]);
    let (across_x, down_x) = (&across_x[..width], &down_x[..width]);
    let (across_y, down_y) = (&across_y[..width], &down_y[..width]);
    for i in 0..width {
        txx[i] = ascent(rxx[i], step, across_x[i]);
        tyy[i] = ascent(ryy[i], step, down_y[i]);
        txy[i] = ascent(rxy[i], step, f64::midpoint(down_x[i], across_y[i]));
    }
}

/// What projects the tensor rows of every channel onto the Frobenius balls of the
/// radius, into `scale`: over all the channels at a pixel where coupled (one row for
/// all), each channel's own otherwise.
#[inline]
#[expect(
    clippy::too_many_arguments,
    reason = "the row takes the three entries, the scales and the model of the ball"
)]
pub fn scale_tensor_rows(
    txx: &[f64],
    tyy: &[f64],
    txy: &[f64],
    scale: &mut [f64],
    width: usize,
    count: usize,
    radius: f64,
    coupled: bool,
) {
    if !coupled || count == 1 {
        for (((scale, &xx), &yy), &xy) in scale.iter_mut().zip(txx).zip(tyy).zip(txy) {
            *scale = ball_scale(tensor_square(xx, yy, xy), radius);
        }
        return;
    }
    match count {
        2 => scale_tensor_coupled::<2>(txx, tyy, txy, &mut scale[..width], width, radius),
        3 => scale_tensor_coupled::<3>(txx, tyy, txy, &mut scale[..width], width, radius),
        4 => scale_tensor_coupled::<4>(txx, tyy, txy, &mut scale[..width], width, radius),
        _ => {
            // The squares added channel after channel, and then their scales.
            let scale = &mut scale[..width];
            scale.fill(0.0);
            for ((xxs, yys), xys) in txx
                .chunks_exact(width)
                .zip(tyy.chunks_exact(width))
                .zip(txy.chunks_exact(width))
                .take(count)
            {
                for (((square, &xx), &yy), &xy) in scale.iter_mut().zip(xxs).zip(yys).zip(xys) {
                    *square += tensor_square(xx, yy, xy);
                }
            }
            for value in scale.iter_mut() {
                *value = ball_scale(*value, radius);
            }
        }
    }
}

/// The scales of the coupled projection of `C` channels' tensor rows.
#[inline]
pub fn scale_tensor_coupled<const C: usize>(
    txx: &[f64],
    tyy: &[f64],
    txy: &[f64],
    scale: &mut [f64],
    width: usize,
    radius: f64,
) {
    let xxs: [&[f64]; C] = std::array::from_fn(|c| &txx[c * width..(c + 1) * width]);
    let yys: [&[f64]; C] = std::array::from_fn(|c| &tyy[c * width..(c + 1) * width]);
    let xys: [&[f64]; C] = std::array::from_fn(|c| &txy[c * width..(c + 1) * width]);
    for (m, scale) in scale[..width].iter_mut().enumerate() {
        let mut square = 0.0;
        for c in 0..C {
            square += tensor_square(xxs[c][m], yys[c][m], xys[c][m]);
        }
        *scale = ball_scale(square, radius);
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

/// The relaxation of a vector field towards `new`; where `KEEP`, `new` is written to
/// `kept`.
#[inline]
pub fn relax_pair_row<const KEEP: bool>(
    new: (&[f64], &[f64]),
    field: (&mut [f64], &mut [f64]),
    kept: (&mut [f64], &mut [f64]),
    rho: f64,
    rest: f64,
) {
    let width = new.0.len();
    let (new_x, new_y) = (new.0, &new.1[..width]);
    let ((x, y), (out_x, out_y)) = (field, kept);
    let (x, y) = (&mut x[..width], &mut y[..width]);
    let length = if KEEP { width } else { 0 };
    let (out_x, out_y) = (&mut out_x[..length], &mut out_y[..length]);
    for i in 0..width {
        if KEEP {
            out_x[i] = new_x[i];
            out_y[i] = new_y[i];
        }
        x[i] = relaxed(rho, rest, new_x[i], x[i]);
        y[i] = relaxed(rho, rest, new_y[i], y[i]);
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
