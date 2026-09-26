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

/// The primal descent of a row.
#[inline(never)]
pub fn inspect_descend_row(out: &mut [f64], rows: [&[f64]; 4], step: f64, planes: usize, across: usize) {
    let [x, px, own, above] = rows;
    kernels::descend_row(out, x, px, own, above, step, planes, across);
}

/// The dual ascent of a row.
#[inline(never)]
pub fn inspect_ascend_row(tx: &mut [f64], ty: &mut [f64], rows: [&[f64]; 4], step: f64, planes: usize, across: usize) {
    let [px, py, bar, below] = rows;
    kernels::ascend_row(tx, ty, px, py, bar, Some(below), step, planes, across);
}

/// The scales of the projection of one channel's dual row.
#[inline(never)]
pub fn inspect_scale_rows(tx: &[f64], ty: &[f64], scale: &mut [f64], width: usize, radius: f64) {
    kernels::scale_rows(tx, ty, scale, width, 1, radius, true);
}

/// The scales of the coupled projection of three channels' dual rows.
#[inline(never)]
pub fn inspect_scale_coupled(tx: &[f64], ty: &[f64], scale: &mut [f64], width: usize, radius: f64) {
    kernels::scale_coupled::<3>(tx, ty, scale, width, radius);
}

/// The projection of a dual row and the relaxation of the point.
#[inline(never)]
pub fn inspect_settle_row(rows: [&[f64]; 4], point: [&mut [f64]; 3], rho: f64, rest: f64) {
    let [tx, ty, scale, new] = rows;
    let [x, px, py] = point;
    let (a, b, c): (&mut [f64], &mut [f64], &mut [f64]) = (&mut [], &mut [], &mut []);
    kernels::settle_row::<false>(tx, ty, scale, new, x, px, py, a, b, c, rho, rest);
}

/// The extrapolation of a row.
#[inline(never)]
pub fn inspect_extrapolate_row(bar: &mut [f64], new: &[f64], old: &[f64]) {
    kernels::extrapolate_row(bar, new, old);
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

/// TGV's step of `w` along a row.
#[inline(never)]
#[expect(
    clippy::similar_names,
    reason = "the names are those of the formulas: the differences of the tensor's entries xx, yy and xy"
)]
pub fn inspect_advance_row(out: [&mut [f64]; 2], rows: [&[f64]; 8], step: f64) {
    let [out_x, out_y] = out;
    let [wx, wy, px, py, across_xx, down_xy, across_xy, down_yy] = rows;
    kernels::advance_row(
        out_x,
        out_y,
        (wx, wy),
        (px, py),
        across_xx,
        down_xy,
        across_xy,
        down_yy,
        step,
    );
}

/// TGV's step of `p` along a row.
#[inline(never)]
pub fn inspect_ascend_p_row(tx: &mut [f64], ty: &mut [f64], rows: [&[f64]; 6], step: f64) {
    let [px, py, across, down, field_x, field_y] = rows;
    kernels::ascend_p_row(tx, ty, (px, py), (across, down), (field_x, field_y), step);
}

/// TGV's step of `r` along a row.
#[inline(never)]
pub fn inspect_ascend_r_row(out: [&mut [f64]; 3], rows: [&[f64]; 7], step: f64) {
    let [txx, tyy, txy] = out;
    let [rxx, ryy, rxy, across_x, down_x, across_y, down_y] = rows;
    kernels::ascend_r_row(
        (txx, tyy, txy),
        (rxx, ryy, rxy),
        across_x,
        down_x,
        across_y,
        down_y,
        step,
    );
}

/// The scales of the projection of one channel's tensor row.
#[inline(never)]
pub fn inspect_scale_tensor_rows(rows: [&[f64]; 3], scale: &mut [f64], width: usize, radius: f64) {
    let [txx, tyy, txy] = rows;
    kernels::scale_tensor_rows(txx, tyy, txy, scale, width, 1, radius, true);
}

/// The scales of the coupled projection of three channels' tensor rows.
#[inline(never)]
pub fn inspect_scale_tensor_coupled(rows: [&[f64]; 3], scale: &mut [f64], width: usize, radius: f64) {
    let [txx, tyy, txy] = rows;
    kernels::scale_tensor_coupled::<3>(txx, tyy, txy, scale, width, radius);
}

/// The projection of a tensor row and the relaxation of `r`.
#[inline(never)]
pub fn inspect_settle_tensor_row(rows: [&[f64]; 4], r: [&mut [f64]; 3], rho: f64, rest: f64) {
    let [txx, tyy, txy, scale] = rows;
    let [rxx, ryy, rxy] = r;
    let none: (&mut [f64], &mut [f64], &mut [f64]) = (&mut [], &mut [], &mut []);
    kernels::settle_tensor_row::<false>((txx, tyy, txy), scale, (rxx, ryy, rxy), none, rho, rest);
}

/// The relaxation of a pair of rows of a field.
#[inline(never)]
pub fn inspect_relax_pair_row(new: [&[f64]; 2], field: [&mut [f64]; 2], rho: f64, rest: f64) {
    let [new_x, new_y] = new;
    let [x, y] = field;
    let none: (&mut [f64], &mut [f64]) = (&mut [], &mut []);
    kernels::relax_pair_row::<false>((new_x, new_y), (x, y), none, rho, rest);
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
    let frequency = Frequency {
        step: 10.0,
        shrunk: 0.25,
        scaled: 0.5,
        inverse: 1.0 / 1.5,
        shrink: 0.1,
        shift: 0.0,
    };
    let frequencies = [frequency; 8];
    let mut out = vec![0.0; 8 * width];
    inspect_forward8(black_box(&values), width, &mut out, width, width);
    let mut back = vec![0.0; 8 * width];
    inspect_inverse8(black_box(&out), width, &mut back, width, width);
    let mut coefficients = values[..8 * across].to_vec();
    inspect_prox_row(&mut coefficients, &levels, &frequencies, 0.0, across);
    inspect_prox_row_charged(&mut coefficients, &levels, &frequencies, 0.5, across);
    let row = &values[..width];
    let mut target = vec![0.0; width];
    inspect_descend_row(&mut target, [row, row, row, row], 0.1, planes, across);
    let (mut tx, mut ty) = (vec![0.0; 3 * width], vec![0.0; 3 * width]);
    inspect_ascend_row(
        &mut tx[..width],
        &mut ty[..width],
        [row, row, row, row],
        0.1,
        planes,
        across,
    );
    let mut scale = vec![0.0; 3 * width];
    inspect_scale_rows(&tx[..width], &ty[..width], &mut scale[..width], width, 1.0);
    inspect_scale_coupled(&tx, &ty, &mut scale[..width], width, 1.0);
    let (mut x, mut px, mut py) = (row.to_vec(), row.to_vec(), row.to_vec());
    inspect_settle_row(
        [&tx[..width], &ty[..width], &scale[..width], row],
        [&mut x, &mut px, &mut py],
        1.9,
        -0.9,
    );
    inspect_extrapolate_row(&mut target, row, &x);
    // TGV's kernels.
    let (mut first, mut second) = (vec![0.0; width], vec![0.0; width]);
    let (mut third, mut fourth) = (vec![0.0; width], vec![0.0; width]);
    inspect_forward_across(row, &mut first, planes, across);
    inspect_backward_across(row, &mut second, planes, across);
    inspect_forward_down(row, &x, &mut third);
    inspect_backward_down(row, &x, &mut fourth);
    let (mut wx, mut wy) = (vec![0.0; width], vec![0.0; width]);
    inspect_advance_row(
        [&mut wx, &mut wy],
        [row, row, row, row, &first, &second, &third, &fourth],
        0.1,
    );
    inspect_ascend_p_row(
        &mut tx[..width],
        &mut ty[..width],
        [row, row, &first, &second, &wx, &wy],
        0.1,
    );
    let (mut txx, mut tyy, mut txy) = (vec![0.0; 3 * width], vec![0.0; 3 * width], vec![0.0; 3 * width]);
    inspect_ascend_r_row(
        [&mut txx[..width], &mut tyy[..width], &mut txy[..width]],
        [row, row, row, &first, &second, &third, &fourth],
        0.1,
    );
    inspect_scale_tensor_rows(
        [&txx[..width], &tyy[..width], &txy[..width]],
        &mut scale[..width],
        width,
        1.0,
    );
    inspect_scale_tensor_coupled([&txx, &tyy, &txy], &mut scale[..width], width, 1.0);
    let (mut rxx, mut ryy, mut rxy) = (row.to_vec(), row.to_vec(), row.to_vec());
    inspect_settle_tensor_row(
        [&txx[..width], &tyy[..width], &txy[..width], &scale[..width]],
        [&mut rxx, &mut ryy, &mut rxy],
        1.9,
        -0.9,
    );
    inspect_relax_pair_row([&first, &second], [&mut wx, &mut wy], 1.9, -0.9);
    let total: f64 = back
        .iter()
        .chain(&coefficients)
        .chain(&target)
        .chain(&px)
        .chain(&rxx)
        .chain(&wx)
        .sum();
    println!("{}", black_box(total).is_finite());
}
