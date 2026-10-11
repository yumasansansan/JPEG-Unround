// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! The kernels of the solvers, each in a function of its own that is not inlined, so
//! that its loops can be read in the assembly (ci/vectorization.py): in the library
//! they are inlined where they are used. Run, it calls each once on a few values.

use std::hint::black_box;

use jpeg_unround::kernels::{self, Frequency};

/// The 8-point DCT across eight streams.
#[inline(never)]
pub fn inspect_forward8(input: &[f64], input_stride: usize, output: &mut [f64], output_stride: usize, count: usize) {
    kernels::forward8(input, input_stride, output, output_stride, count);
}

/// The inverse 8-point DCT across eight streams.
#[inline(never)]
pub fn inspect_inverse8(input: &[f64], input_stride: usize, output: &mut [f64], output_stride: usize, count: usize) {
    kernels::inverse8(input, input_stride, output, output_stride, count);
}

/// The proximal maps of a row of coefficients.
#[inline(never)]
pub fn inspect_prox_row(values: &mut [f64], levels: &[i16], frequencies: &[Frequency; 8], slack: f64, count: usize) {
    kernels::prox_row::<false>(values, count, levels, count, frequencies, slack, count);
}

/// The proximal maps of a row of coefficients, with the cost of the slack.
#[inline(never)]
pub fn inspect_prox_row_charged(
    values: &mut [f64],
    levels: &[i16],
    frequencies: &[Frequency; 8],
    slack: f64,
    count: usize,
) {
    kernels::prox_row::<true>(values, count, levels, count, frequencies, slack, count);
}

/// The proximal maps of a row of coefficients, flat blocks with their own factors.
#[inline(never)]
pub fn inspect_prox_row_flat(
    values: &mut [f64],
    levels: &[i16],
    frequencies: [&[Frequency; 8]; 2],
    marks: &[f64],
    slack: f64,
    count: usize,
) {
    let [frequencies, flat] = frequencies;
    kernels::prox_row_flat::<false>(values, count, levels, count, frequencies, flat, marks, slack, count);
}

/// The same with the cost of the slack.
#[inline(never)]
pub fn inspect_prox_row_flat_charged(
    values: &mut [f64],
    levels: &[i16],
    frequencies: [&[Frequency; 8]; 2],
    marks: &[f64],
    slack: f64,
    count: usize,
) {
    let [frequencies, flat] = frequencies;
    kernels::prox_row_flat::<true>(values, count, levels, count, frequencies, flat, marks, slack, count);
}

/// The primal descent of a row.
#[inline(never)]
pub fn inspect_descend_row(out: &mut [f64], rows: [&[f64]; 4], step: f64, planes: usize, across: usize) {
    let [x, px, own, above] = rows;
    kernels::descend_row(out, x, px, own, above, step, planes, across);
}

/// The projection of a dual row and the relaxation of the dual.
#[inline(never)]
pub fn inspect_settle_row(rows: [&[f64]; 3], dual: [&mut [f64]; 2], rho: f64, rest: f64) {
    let [tx, ty, scale] = rows;
    let [px, py] = dual;
    kernels::settle_row::<false>(tx, ty, scale, px, py, &mut [], &mut [], rho, rest);
}

/// The extrapolation of a row.
#[inline(never)]
pub fn inspect_extrapolate_row(bar: &mut [f64], new: &[f64], old: &[f64]) {
    kernels::extrapolate_row(bar, new, old);
}

/// The extrapolation of a row of the canvas and the relaxation of the canvas.
#[inline(never)]
pub fn inspect_extrapolate_relax_row(bar: &mut [f64], new: &[f64], x: &mut [f64], rho: f64, rest: f64) {
    kernels::extrapolate_relax_row::<false>(bar, new, x, &mut [], rho, rest);
}

/// The forward differences across a row.
#[inline(never)]
pub fn inspect_forward_across(row: &[f64], out: &mut [f64], planes: usize, across: usize) {
    kernels::forward_across(row, out, planes, across);
}

/// The backward differences across a row.
#[inline(never)]
pub fn inspect_backward_across(row: &[f64], out: &mut [f64], planes: usize, across: usize) {
    kernels::backward_across(row, out, planes, across);
}

/// The forward differences down, from a row and the row below.
#[inline(never)]
pub fn inspect_forward_down(row: &[f64], below: &[f64], out: &mut [f64]) {
    kernels::forward_down(row, Some(below), out);
}

/// The backward differences down, from a row and the row above.
#[inline(never)]
pub fn inspect_backward_down(own: &[f64], above: &[f64], out: &mut [f64]) {
    kernels::backward_down(Some(own), Some(above), out);
}

/// The scales of an inspection: the first of two coupled channels' (squares written),
/// a channel's own, and the last of two coupled channels' (squares added, and scales).
fn scales_of(row: &mut [f64], mode: usize) -> kernels::Scales<'_> {
    match mode {
        0 => kernels::Scales::coupled(row, 1.0, 0, 2),
        1 => kernels::Scales::own(row, 1.0),
        _ => kernels::Scales::coupled(row, 1.0, 1, 2),
    }
}

/// The dual ascent of a row, the squares of the first of coupled channels written.
#[inline(never)]
pub fn inspect_ascend_row(t: [&mut [f64]; 3], rows: [&[f64]; 4], step: f64, planes: usize, across: usize) {
    let [tx, ty, scale] = t;
    let [px, py, bar, below] = rows;
    let scales = scales_of(scale, 0);
    kernels::ascend_row(tx, ty, px, py, bar, Some(below), step, planes, across, scales);
}

/// The dual ascent of a row with the scales of its own projection.
#[inline(never)]
pub fn inspect_ascend_row_scaled(t: [&mut [f64]; 3], rows: [&[f64]; 4], step: f64, planes: usize, across: usize) {
    let [tx, ty, scale] = t;
    let [px, py, bar, below] = rows;
    let scales = scales_of(scale, 1);
    kernels::ascend_row(tx, ty, px, py, bar, Some(below), step, planes, across, scales);
}

/// The dual ascent of a row with the scales of the last of coupled channels.
#[inline(never)]
pub fn inspect_ascend_row_summed(t: [&mut [f64]; 3], rows: [&[f64]; 4], step: f64, planes: usize, across: usize) {
    let [tx, ty, scale] = t;
    let [px, py, bar, below] = rows;
    let scales = scales_of(scale, 2);
    kernels::ascend_row(tx, ty, px, py, bar, Some(below), step, planes, across, scales);
}

/// TGV's step of `p` along a row, the squares of the first of coupled channels written.
#[inline(never)]
pub fn inspect_ascend_p_row(t: [&mut [f64]; 3], rows: [&[f64]; 6], planes: usize, across: usize) {
    let [tx, ty, scale] = t;
    let [px, py, bar, under, field_x, field_y] = rows;
    let scales = scales_of(scale, 0);
    kernels::ascend_p_row(
        (tx, ty),
        (px, py),
        bar,
        Some(under),
        (field_x, field_y),
        0.1,
        planes,
        across,
        scales,
    );
}

/// TGV's step of `p` along a row with the scales of its own projection.
#[inline(never)]
pub fn inspect_ascend_p_row_scaled(t: [&mut [f64]; 3], rows: [&[f64]; 6], planes: usize, across: usize) {
    let [tx, ty, scale] = t;
    let [px, py, bar, under, field_x, field_y] = rows;
    let scales = scales_of(scale, 1);
    kernels::ascend_p_row(
        (tx, ty),
        (px, py),
        bar,
        Some(under),
        (field_x, field_y),
        0.1,
        planes,
        across,
        scales,
    );
}

/// TGV's step of `p` along a row with the scales of the last of coupled channels.
#[inline(never)]
pub fn inspect_ascend_p_row_summed(t: [&mut [f64]; 3], rows: [&[f64]; 6], planes: usize, across: usize) {
    let [tx, ty, scale] = t;
    let [px, py, bar, under, field_x, field_y] = rows;
    let scales = scales_of(scale, 2);
    kernels::ascend_p_row(
        (tx, ty),
        (px, py),
        bar,
        Some(under),
        (field_x, field_y),
        0.1,
        planes,
        across,
        scales,
    );
}

/// TGV's step of `p` along the last row with the scales of its own projection.
#[inline(never)]
pub fn inspect_ascend_p_last_row(t: [&mut [f64]; 3], rows: [&[f64]; 6], planes: usize, across: usize) {
    let [tx, ty, scale] = t;
    let [px, py, bar, _, field_x, field_y] = rows;
    let scales = scales_of(scale, 1);
    kernels::ascend_p_row(
        (tx, ty),
        (px, py),
        bar,
        None,
        (field_x, field_y),
        0.1,
        planes,
        across,
        scales,
    );
}

/// TGV's step of `r` along a row, the squares of the first of coupled channels written.
#[inline(never)]
pub fn inspect_ascend_r_row(t: [&mut [f64]; 4], rows: [&[f64]; 9], planes: usize, across: usize) {
    let [txx, tyy, txy, scale] = t;
    let [rxx, ryy, rxy, field_x, field_y, own_x, own_y, above_x, above_y] = rows;
    kernels::ascend_r_row(
        (txx, tyy, txy),
        (rxx, ryy, rxy),
        (field_x, field_y),
        (own_x, own_y),
        (above_x, above_y),
        0.1,
        planes,
        across,
        scales_of(scale, 0),
    );
}

/// TGV's step of `r` along a row with the scales of its own projection.
#[inline(never)]
pub fn inspect_ascend_r_row_scaled(t: [&mut [f64]; 4], rows: [&[f64]; 9], planes: usize, across: usize) {
    let [txx, tyy, txy, scale] = t;
    let [rxx, ryy, rxy, field_x, field_y, own_x, own_y, above_x, above_y] = rows;
    kernels::ascend_r_row(
        (txx, tyy, txy),
        (rxx, ryy, rxy),
        (field_x, field_y),
        (own_x, own_y),
        (above_x, above_y),
        0.1,
        planes,
        across,
        scales_of(scale, 1),
    );
}

/// TGV's step of `r` along a row with the scales of the last of coupled channels.
#[inline(never)]
pub fn inspect_ascend_r_row_summed(t: [&mut [f64]; 4], rows: [&[f64]; 9], planes: usize, across: usize) {
    let [txx, tyy, txy, scale] = t;
    let [rxx, ryy, rxy, field_x, field_y, own_x, own_y, above_x, above_y] = rows;
    kernels::ascend_r_row(
        (txx, tyy, txy),
        (rxx, ryy, rxy),
        (field_x, field_y),
        (own_x, own_y),
        (above_x, above_y),
        0.1,
        planes,
        across,
        scales_of(scale, 2),
    );
}

/// TV's dual step of a row of a channel's own projection, projected and relaxed.
#[inline(never)]
pub fn inspect_dual_row(p: [&mut [f64]; 2], rows: [&[f64]; 2], planes: usize, across: usize) {
    let [px, py] = p;
    let [bar, below] = rows;
    kernels::dual_p_row(
        (px, py),
        None,
        bar,
        Some(below),
        None,
        0.1,
        (1.0, 1.9, -0.9),
        planes,
        across,
    );
}

/// TGV's step of `p` of a channel's own projection, projected and relaxed.
#[inline(never)]
pub fn inspect_dual_p_row(p: [&mut [f64]; 2], rows: [&[f64]; 4], planes: usize, across: usize) {
    let [px, py] = p;
    let [bar, below, field_x, field_y] = rows;
    let field = Some((field_x, field_y));
    kernels::dual_p_row(
        (px, py),
        None,
        bar,
        Some(below),
        field,
        0.1,
        (1.0, 1.9, -0.9),
        planes,
        across,
    );
}

/// The same at the last row, its outputs kept.
#[inline(never)]
pub fn inspect_dual_p_last_row_kept(p: [&mut [f64]; 4], rows: [&[f64]; 3], planes: usize, across: usize) {
    let [px, py, kept_x, kept_y] = p;
    let [bar, field_x, field_y] = rows;
    let (kept, field) = (Some((kept_x, kept_y)), Some((field_x, field_y)));
    kernels::dual_p_row((px, py), kept, bar, None, field, 0.1, (1.0, 1.9, -0.9), planes, across);
}

/// TGV's step of `r` of a channel's own projection, projected and relaxed.
#[inline(never)]
pub fn inspect_dual_r_row(r: [&mut [f64]; 3], rows: [&[f64]; 6], planes: usize, across: usize) {
    let [rxx, ryy, rxy] = r;
    let [field_x, field_y, own_x, own_y, above_x, above_y] = rows;
    kernels::dual_r_row(
        (rxx, ryy, rxy),
        None,
        (field_x, field_y),
        (own_x, own_y),
        (above_x, above_y),
        0.1,
        (1.0, 1.9, -0.9),
        planes,
        across,
    );
}

/// TGV's step of `w` along a row, its extrapolation and the relaxation of `w`.
#[inline(never)]
pub fn inspect_advance_row(out: [&mut [f64]; 4], rows: [&[f64]; 7], step: f64, planes: usize, across: usize) {
    let [bar_x, bar_y, wx, wy] = out;
    let [px, py, rxx, ryy, rxy, yy_below, xy_below] = rows;
    kernels::advance_row(
        (bar_x, bar_y),
        (wx, wy),
        None,
        (px, py),
        (rxx, ryy, rxy),
        Some((yy_below, xy_below)),
        (step, 1.9, -0.9),
        planes,
        across,
    );
}

/// The same with the output of the step kept.
#[inline(never)]
pub fn inspect_advance_row_kept(out: [&mut [f64]; 6], rows: [&[f64]; 7], step: f64, planes: usize, across: usize) {
    let [bar_x, bar_y, wx, wy, kept_x, kept_y] = out;
    let [px, py, rxx, ryy, rxy, yy_below, xy_below] = rows;
    kernels::advance_row(
        (bar_x, bar_y),
        (wx, wy),
        Some((kept_x, kept_y)),
        (px, py),
        (rxx, ryy, rxy),
        Some((yy_below, xy_below)),
        (step, 1.9, -0.9),
        planes,
        across,
    );
}

/// TGV's step of `w` along the last row, its extrapolation and the relaxation of `w`.
#[inline(never)]
pub fn inspect_advance_last_row(out: [&mut [f64]; 4], rows: [&[f64]; 5], step: f64, planes: usize, across: usize) {
    let [bar_x, bar_y, wx, wy] = out;
    let [px, py, rxx, ryy, rxy] = rows;
    kernels::advance_row(
        (bar_x, bar_y),
        (wx, wy),
        None,
        (px, py),
        (rxx, ryy, rxy),
        None,
        (step, 1.9, -0.9),
        planes,
        across,
    );
}

/// The projection of a tensor row and the relaxation of `r`.
#[inline(never)]
pub fn inspect_settle_tensor_row(rows: [&[f64]; 4], r: [&mut [f64]; 3], rho: f64, rest: f64) {
    let [txx, tyy, txy, scale] = rows;
    let [rxx, ryy, rxy] = r;
    let none: (&mut [f64], &mut [f64], &mut [f64]) = (&mut [], &mut [], &mut []);
    kernels::settle_tensor_row::<false>((txx, tyy, txy), scale, (rxx, ryy, rxy), none, rho, rest);
}

/// The proximal maps of a row, of every kind.
fn prox_kernels(coefficients: &mut [f64], levels: &[i16], across: usize) {
    let frequency = Frequency {
        step: 10.0,
        shrunk: 0.25,
        scaled: 0.5,
        inverse: 1.0 / 1.5,
        shrink: 0.1,
        shift: 0.0,
    };
    let frequencies = [frequency; 8];
    inspect_prox_row(coefficients, levels, &frequencies, 0.0, across);
    inspect_prox_row_charged(coefficients, levels, &frequencies, 0.5, across);
    let marks: Vec<f64> = (0..across).map(|m| if m % 3 == 0 { 1.0 } else { 0.0 }).collect();
    let flat = frequencies.map(|frequency| Frequency {
        scaled: 0.5 * frequency.scaled,
        inverse: 1.0 / (1.0 + 0.5 * frequency.scaled),
        ..frequency
    });
    inspect_prox_row_flat(coefficients, levels, [&frequencies, &flat], &marks, 0.0, across);
    inspect_prox_row_flat_charged(coefficients, levels, [&frequencies, &flat], &marks, 0.5, across);
}

/// TGV's steps of `w` and `p` at a row with a row below it and at the last row, and
/// the steps of TV and TGV with the scales of their projections: the sum of what they
/// write.
fn tgv_steps(row: &[f64], below: &[f64], planes: usize, across: usize) -> f64 {
    let width = planes * across;
    let [mut wx, mut wy, mut lx, mut ly, mut tx, mut ty, mut kx, mut ky]: [Vec<f64>; 8] =
        std::array::from_fn(|_| vec![0.0; width]);
    let (mut sx, mut sy) = (row.to_vec(), below.to_vec());
    let rows = [row, row, row, below, row, below, row];
    inspect_advance_row([&mut wx, &mut wy, &mut sx, &mut sy], rows, 0.1, planes, across);
    inspect_advance_row_kept(
        [&mut tx, &mut ty, &mut sx, &mut sy, &mut lx, &mut ly],
        rows,
        0.1,
        planes,
        across,
    );
    let rows = [row, row, below, row, below];
    inspect_advance_last_row([&mut kx, &mut ky, &mut sx, &mut sy], rows, 0.1, planes, across);
    let [mut scale, mut sxx, mut syy, mut sxy, mut scale_r]: [Vec<f64>; 5] = std::array::from_fn(|_| vec![0.0; width]);
    let rows = [row, row, below, row, &wx, &wy];
    inspect_ascend_p_row([&mut tx, &mut ty, &mut scale], rows, planes, across);
    inspect_ascend_p_row_summed([&mut tx, &mut ty, &mut scale], rows, planes, across);
    inspect_ascend_p_row_scaled([&mut sx, &mut sy, &mut scale], rows, planes, across);
    let rows = [row, row, below, row, &lx, &ly];
    inspect_ascend_p_last_row([&mut kx, &mut ky, &mut scale], rows, planes, across);
    let rows = [row, row, below, row];
    inspect_ascend_row_scaled([&mut sx, &mut sy, &mut scale], rows, 0.1, planes, across);
    inspect_ascend_row_summed([&mut sx, &mut sy, &mut scale], rows, 0.1, planes, across);
    let rows = [row, row, row, &wx, &wy, &wx, &wy, below, row];
    inspect_ascend_r_row_scaled([&mut sxx, &mut syy, &mut sxy, &mut scale_r], rows, planes, across);
    inspect_ascend_r_row_summed([&mut sxx, &mut syy, &mut sxy, &mut scale_r], rows, planes, across);
    inspect_dual_row([&mut sx, &mut sy], [row, below], planes, across);
    inspect_dual_p_row([&mut tx, &mut ty], [row, below, &wx, &wy], planes, across);
    inspect_dual_p_last_row_kept([&mut tx, &mut ty, &mut kx, &mut ky], [row, &wx, &wy], planes, across);
    inspect_dual_r_row(
        [&mut sxx, &mut syy, &mut sxy],
        [&wx, &wy, &wx, &wy, below, row],
        planes,
        across,
    );
    [wx, wy, lx, ly, tx, ty, kx, ky, sx, sy, scale, sxx, syy, sxy, scale_r]
        .iter()
        .flatten()
        .sum()
}

fn main() {
    // The sizes are opaque, so that no kernel is specialized for them.
    let (planes, across) = (black_box(8), black_box(12));
    let width = planes * across;
    let values: Vec<f64> = (0..8 * width)
        .map(|index| f64::from(u32::try_from(index % 97).unwrap_or(0)))
        .collect();
    let levels: Vec<i16> = (0..8 * width)
        .map(|index| i16::try_from(index % 5).unwrap_or(0) - 2)
        .collect();
    let mut out = vec![0.0; 8 * width];
    inspect_forward8(black_box(&values), width, &mut out, width, width);
    let mut back = vec![0.0; 8 * width];
    inspect_inverse8(black_box(&out), width, &mut back, width, width);
    let mut coefficients = values[..8 * across].to_vec();
    prox_kernels(&mut coefficients, &levels, across);
    let row = &values[..width];
    let mut target = vec![0.0; width];
    inspect_descend_row(&mut target, [row, row, row, row], 0.1, planes, across);
    let (mut tx, mut ty) = (vec![0.0; 3 * width], vec![0.0; 3 * width]);
    let mut scale = vec![0.0; 3 * width];
    inspect_ascend_row(
        [&mut tx[..width], &mut ty[..width], &mut scale[..width]],
        [row, row, row, row],
        0.1,
        planes,
        across,
    );
    let (mut x, mut px, mut py) = (row.to_vec(), row.to_vec(), row.to_vec());
    inspect_settle_row(
        [&tx[..width], &ty[..width], &scale[..width]],
        [&mut px, &mut py],
        1.9,
        -0.9,
    );
    inspect_extrapolate_row(&mut target, row, &x);
    inspect_extrapolate_relax_row(&mut target, row, &mut x, 1.9, -0.9);
    // TGV's kernels.
    let (mut first, mut second) = (vec![0.0; width], vec![0.0; width]);
    let (mut third, mut fourth) = (vec![0.0; width], vec![0.0; width]);
    inspect_forward_across(row, &mut first, planes, across);
    inspect_backward_across(row, &mut second, planes, across);
    inspect_forward_down(row, &x, &mut third);
    inspect_backward_down(row, &x, &mut fourth);
    let steps = tgv_steps(row, &x, planes, across);
    let (wx, wy) = (row.to_vec(), x.clone());
    let (mut txx, mut tyy, mut txy) = (vec![0.0; 3 * width], vec![0.0; 3 * width], vec![0.0; 3 * width]);
    let mut scale_r = vec![0.0; width];
    inspect_ascend_r_row(
        [&mut txx[..width], &mut tyy[..width], &mut txy[..width], &mut scale_r],
        [row, row, row, &first, &second, &third, &fourth, &wx, &wy],
        planes,
        across,
    );
    let (mut rxx, mut ryy, mut rxy) = (row.to_vec(), row.to_vec(), row.to_vec());
    inspect_settle_tensor_row(
        [&txx[..width], &tyy[..width], &txy[..width], &scale[..width]],
        [&mut rxx, &mut ryy, &mut rxy],
        1.9,
        -0.9,
    );
    let total: f64 = back
        .iter()
        .chain(&coefficients)
        .chain(&target)
        .chain(&px)
        .chain(&rxx)
        .chain(&wx)
        .sum();
    println!("{}", black_box(total + steps).is_finite());
}
