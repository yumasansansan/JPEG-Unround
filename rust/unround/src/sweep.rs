// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! One iteration of the primal-dual method as one sweep over the canvas, band of MCU
//! rows by band, in the layout of the planes (docs/math.md, 5.1).
//!
//! The iteration of TV takes the primal step of a band -- the descent along the
//! divergence of the dual, and the proximal map of `G` block by block -- and then
//! the dual step and the relaxation of the band before it, whose forward differences
//! need the new band's first row. Only two bands of the new canvas are kept, and the
//! current point is read and written once; the outputs of the proximal steps are
//! written out where a record asks for them.

use crate::dct::BLOCK;
use crate::error::Error;
use crate::frames::{FREE_CENTRE, Frame};
use crate::kernels::{
    self, Frequency, advance_row, ascend_p_row, ascend_r_row, ascend_row, backward_across, backward_down, descend_row,
    extrapolate_row, forward_across, forward_down, relax_pair_row, scale_rows, scale_tensor_rows, settle_row,
    settle_tensor_row,
};
use crate::model::{LEVEL_SHIFT_DC, Step};
use crate::planar::{Layout, Part};

/// What the proximal map of one component takes: its layout and levels, the
/// constants of each frequency, its slack, and the samples of its cells.
#[derive(Debug, Clone)]
pub struct Component {
    part: Part,
    frequencies: [[Frequency; BLOCK]; BLOCK],
    slack: f64,
    charged: bool,
    cells: f64,
}

impl Component {
    /// The component of a channel with the factors of its proximal map.
    fn new(part: &Part, problem: &crate::model::Problem, step: &Step) -> Self {
        let mut frequencies = [[Frequency::default(); BLOCK]; BLOCK];
        let (scaled, inverse, shrink) = step.factors();
        for (index, frequency) in frequencies.iter_mut().flatten().enumerate() {
            *frequency = Frequency {
                step: problem.steps()[index],
                shrunk: problem.shrunk()[index],
                scaled: scaled[index],
                inverse: inverse[index],
                shrink: shrink[index],
                shift: if index == 0 { LEVEL_SHIFT_DC } else { 0.0 },
            };
        }
        Self {
            part: part.clone(),
            frequencies,
            slack: problem.slack(),
            charged: problem.cost().is_some(),
            cells: crate::exact::nearest_from_usize(part.ratio.0 * part.ratio.1),
        }
    }
}

/// The current point of TV in the planes: the canvas and the dual field.
#[derive(Debug, Clone)]
pub struct TvPoint {
    /// The canvas, `C x H x W`.
    pub x: Vec<f64>,
    /// The dual's entries across.
    pub px: Vec<f64>,
    /// The dual's entries down.
    pub py: Vec<f64>,
}

/// The outputs of the proximal steps, in the planes: the canvas, each component's
/// coefficients, and the dual field.
#[derive(Debug, Clone)]
pub struct TvOutputs {
    /// The canvas `x~`.
    pub canvas: Vec<f64>,
    /// Each component's coefficients of it.
    pub coefficients: Vec<Vec<f64>>,
    /// The dual `p~` across.
    pub px: Vec<f64>,
    /// The dual `p~` down.
    pub py: Vec<f64>,
}

impl TvOutputs {
    /// Room for the outputs of a frame.
    #[must_use]
    pub fn new(layout: &Layout) -> Self {
        let size = layout.plane() * layout.parts.len();
        Self {
            canvas: vec![0.0; size],
            coefficients: layout
                .parts
                .iter()
                .map(|part| vec![0.0; part.rows * part.block_row(layout.across)])
                .collect(),
            px: vec![0.0; size],
            py: vec![0.0; size],
        }
    }
}

/// The room a band of blocks needs: the rows of the vertical transforms and the grid
/// of a component's cells.
#[derive(Debug, Clone)]
struct Room {
    y: Vec<f64>,
    c: Vec<f64>,
    grid: Vec<f64>,
    fresh: Vec<f64>,
}

/// The 8x8 blocks of one block row of a component's grid, `rows` its rows (`8 x P_c x
/// M`, the planes one after another), into `fresh`: the DCT down the columns of every
/// plane at once, the proximal map along the rows between the DCT and its inverse,
/// set of blocks by set, and the inverse down the columns. The coefficients go to
/// `kept` where `KEEP`.
#[expect(
    clippy::too_many_arguments,
    reason = "the blocks take the rows in and out, the room, the component and the coefficients kept"
)]
fn prox_blocks<const KEEP: bool>(
    rows: &[f64],
    fresh: &mut [f64],
    y: &mut [f64],
    c: &mut [f64],
    component: &Component,
    levels: &[i16],
    kept: &mut [f64],
    across: usize,
) {
    let part = &component.part;
    let row = part.planes * across;
    // Down the columns of every plane: y[v] is vertical frequency v of the whole row.
    kernels::forward8(rows, row, y, row, row);
    for (set, &count) in part.valid.iter().enumerate() {
        if count == 0 {
            continue;
        }
        // Along the rows: coefficient row v of every block of the set, into the
        // coefficients c[u], their proximal maps, and back into y[v].
        for v in 0..BLOCK {
            let place = v * row + BLOCK * set * across;
            kernels::forward8(&y[place..], across, c, across, count);
            let start = |u: usize| (v * part.planes + BLOCK * set + u) * across;
            let (frequencies, own) = (&component.frequencies[v], &levels[start(0)..]);
            if component.charged {
                kernels::prox_row::<true>(c, across, own, across, frequencies, component.slack, count);
            } else {
                kernels::prox_row::<false>(c, across, own, across, frequencies, component.slack, count);
            }
            if KEEP {
                for u in 0..BLOCK {
                    kept[start(u)..start(u) + count].copy_from_slice(&c[u * across..u * across + count]);
                }
            }
            kernels::inverse8(c, across, &mut y[place..], across, count);
        }
    }
    // Up the columns of every plane, into fresh.
    kernels::inverse8(y, row, fresh, row, row);
    // What the component does not have stays.
    for (set, &count) in part.valid.iter().enumerate() {
        if count < across {
            for i in 0..BLOCK {
                for u in 0..BLOCK {
                    let start = i * row + (BLOCK * set + u) * across;
                    fresh[start + count..start + across].copy_from_slice(&rows[start + count..start + across]);
                }
            }
        }
    }
}

/// The primal step of both models, band by band: the descent along the divergence of
/// `p`, and the proximal map of `G` of every component.
#[derive(Debug, Clone)]
struct Primal {
    layout: Layout,
    components: Vec<Component>,
    descents: Vec<f64>,
    work: Vec<f64>,
    room: Room,
    zeros: Vec<f64>,
}

impl Primal {
    fn new(frame: &Frame, layout: &Layout, steps: &[Step], gammas: &[f64], tau: f64) -> Self {
        let components: Vec<Component> = frame
            .channels()
            .iter()
            .zip(&layout.parts)
            .zip(steps)
            .map(|((channel, part), step)| Component::new(part, channel.problem(), step))
            .collect();
        let grid = layout
            .parts
            .iter()
            .map(|part| (layout.band / part.ratio.0) * part.planes * layout.across)
            .max()
            .unwrap_or(0);
        Self {
            layout: layout.clone(),
            components,
            descents: gammas.iter().map(|gamma| tau * gamma).collect(),
            work: vec![0.0; layout.band * layout.width],
            room: Room {
                y: vec![0.0; BLOCK * layout.planes * layout.across],
                c: vec![0.0; BLOCK * layout.across],
                grid: vec![0.0; grid],
                fresh: vec![0.0; grid],
            },
            zeros: vec![0.0; layout.width],
        }
    }

    /// The primal step of a band, `x + tau gamma div p` and the proximal map of `G`,
    /// into `target` (`C x R x W`); the coefficients to `kept` where `KEEP`.
    fn band<const KEEP: bool>(
        &mut self,
        band: usize,
        point: (&[f64], &[f64], &[f64]),
        target: &mut [f64],
        kept: &mut [Vec<f64>],
    ) {
        let (x, px, py) = point;
        let layout = &self.layout;
        let (width, rows) = (layout.width, layout.band);
        let plane = layout.plane();
        for (channel, component) in self.components.iter().enumerate() {
            let base = channel * plane;
            let first = band * rows;
            for i in 0..rows {
                let r = first + i;
                let down_above = if r > 0 {
                    row_of(py, base, r - 1, width)
                } else {
                    &self.zeros[..]
                };
                let down_own = if r + 1 < layout.height {
                    row_of(py, base, r, width)
                } else {
                    &self.zeros[..]
                };
                descend_row(
                    &mut self.work[i * width..(i + 1) * width],
                    row_of(x, base, r, width),
                    row_of(px, base, r, width),
                    down_own,
                    down_above,
                    self.descents[channel],
                    layout.planes,
                    layout.across,
                );
            }
            let own = &mut target[channel * rows * width..(channel + 1) * rows * width];
            let kept_part: &mut [f64] = if KEEP { &mut kept[channel] } else { &mut [] };
            prox_band::<KEEP>(layout, component, band, &self.work, own, &mut self.room, kept_part);
        }
    }
}

/// One iteration of TV as a sweep over the bands.
#[derive(Debug, Clone)]
pub struct TvSweep {
    primal: Primal,
    ascents: Vec<f64>,
    radius: f64,
    coupled: bool,
    rho: f64,
    rest: f64,
    ring: [Vec<f64>; 2],
    bar: [Vec<f64>; 2],
    tx: Vec<f64>,
    ty: Vec<f64>,
    scale: Vec<f64>,
}

impl TvSweep {
    /// The sweep of a frame's TV model with the steps `tau` and `sigma`, the channel
    /// weights `gammas`, the radius `alpha` of the dual balls, coupled or not, and the
    /// relaxation `rho`; `steps` are the factors of each component's proximal map.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "the sweep takes the model, both steps and the relaxation"
    )]
    pub fn new(
        frame: &Frame,
        layout: &Layout,
        steps: &[Step],
        gammas: &[f64],
        tau: f64,
        sigma: f64,
        alpha: f64,
        coupled: bool,
        rho: f64,
    ) -> Self {
        let count = gammas.len();
        let band = layout.band * layout.width;
        Self {
            primal: Primal::new(frame, layout, steps, gammas, tau),
            ascents: gammas.iter().map(|gamma| sigma * gamma).collect(),
            radius: alpha,
            coupled,
            rho,
            rest: 1.0 - rho,
            ring: [vec![0.0; count * band], vec![0.0; count * band]],
            bar: [vec![0.0; count * layout.width], vec![0.0; count * layout.width]],
            tx: vec![0.0; count * layout.width],
            ty: vec![0.0; count * layout.width],
            scale: vec![0.0; count * layout.width],
        }
    }

    /// One iteration from the point, which moves to the relaxed point; the outputs of
    /// the proximal steps are written to `outputs` where given.
    pub fn iterate(&mut self, point: &mut TvPoint, mut outputs: Option<&mut TvOutputs>) {
        let bands = self.primal.layout.bands();
        for band in 0..=bands {
            if band < bands {
                let fields = (&point.x[..], &point.px[..], &point.py[..]);
                let target = &mut self.ring[band % 2];
                if let Some(out) = outputs.as_deref_mut() {
                    self.primal.band::<true>(band, fields, target, &mut out.coefficients);
                } else {
                    self.primal.band::<false>(band, fields, target, &mut []);
                }
            }
            if band > 0 {
                self.dual(band - 1, point, outputs.as_deref_mut());
            }
        }
    }

    /// The dual step and the relaxation of a band, whose next band's primal step is
    /// in the ring.
    #[expect(
        clippy::similar_names,
        reason = "the names are those of the formulas: x, p_x and p_y and the outputs of each"
    )]
    fn dual(&mut self, band: usize, point: &mut TvPoint, mut outputs: Option<&mut TvOutputs>) {
        let layout = &self.primal.layout;
        let (width, rows, height) = (layout.width, layout.band, layout.height);
        let plane = layout.plane();
        let count = self.ascents.len();
        for i in 0..rows {
            let r = band * rows + i;
            for channel in 0..count {
                let row = channel * plane + r * width;
                let span = channel * width..(channel + 1) * width;
                if r == 0 {
                    extrapolate_row(
                        &mut self.bar[0][span.clone()],
                        ring_row(&self.ring, channel, rows, width, 0),
                        &point.x[row..row + width],
                    );
                }
                let (current, next) = self.bar.split_at_mut(1);
                let (current, next) = (&current[0][span.clone()], &mut next[0][span.clone()]);
                let below = if r + 1 < height {
                    let next_row = row + width;
                    extrapolate_row(
                        next,
                        ring_row(&self.ring, channel, rows, width, r + 1),
                        &point.x[next_row..next_row + width],
                    );
                    Some(&*next)
                } else {
                    None
                };
                ascend_row(
                    &mut self.tx[span.clone()],
                    &mut self.ty[span.clone()],
                    &point.px[row..row + width],
                    &point.py[row..row + width],
                    current,
                    below,
                    self.ascents[channel],
                    layout.planes,
                    layout.across,
                );
            }
            scale_rows(
                &self.tx,
                &self.ty,
                &mut self.scale,
                width,
                count,
                self.radius,
                self.coupled,
            );
            for channel in 0..count {
                let row = channel * plane + r * width;
                let span = channel * width..(channel + 1) * width;
                let scales = if self.coupled { 0..width } else { span.clone() };
                let start = (channel * rows + i) * width;
                let new = &self.ring[band % 2][start..start + width];
                let (tx, ty, scale) = (&self.tx[span.clone()], &self.ty[span], &self.scale[scales]);
                let (x, px, py) = (
                    &mut point.x[row..row + width],
                    &mut point.px[row..row + width],
                    &mut point.py[row..row + width],
                );
                if let Some(out) = outputs.as_deref_mut() {
                    let (kept_x, kept_px, kept_py) = (
                        &mut out.canvas[row..row + width],
                        &mut out.px[row..row + width],
                        &mut out.py[row..row + width],
                    );
                    settle_row::<true>(
                        tx, ty, scale, new, x, px, py, kept_x, kept_px, kept_py, self.rho, self.rest,
                    );
                } else {
                    let (none_x, none_px, none_py): (&mut [f64], &mut [f64], &mut [f64]) = (&mut [], &mut [], &mut []);
                    settle_row::<false>(
                        tx, ty, scale, new, x, px, py, none_x, none_px, none_py, self.rho, self.rest,
                    );
                }
            }
            self.bar.swap(0, 1);
        }
    }
}

/// The current point of TGV in the planes: the canvas, the field `w`, and the duals
/// `p` and `r`.
#[derive(Debug, Clone)]
pub struct TgvPoint {
    /// The canvas, `C x H x W`.
    pub x: Vec<f64>,
    /// `w` across.
    pub wx: Vec<f64>,
    /// `w` down.
    pub wy: Vec<f64>,
    /// `p` across.
    pub px: Vec<f64>,
    /// `p` down.
    pub py: Vec<f64>,
    /// `r11`.
    pub rxx: Vec<f64>,
    /// `r22`.
    pub ryy: Vec<f64>,
    /// `r12`.
    pub rxy: Vec<f64>,
}

/// The outputs of TGV's proximal steps, in the planes.
#[derive(Debug, Clone)]
pub struct TgvOutputs {
    /// `x~`, its coefficients, and `p~`, as TV's.
    pub first: TvOutputs,
    /// `w~` across.
    pub wx: Vec<f64>,
    /// `w~` down.
    pub wy: Vec<f64>,
    /// `r~11`.
    pub rxx: Vec<f64>,
    /// `r~22`.
    pub ryy: Vec<f64>,
    /// `r~12`.
    pub rxy: Vec<f64>,
}

impl TgvOutputs {
    /// Room for the outputs of a frame.
    #[must_use]
    pub fn new(layout: &Layout) -> Self {
        let size = layout.plane() * layout.parts.len();
        Self {
            first: TvOutputs::new(layout),
            wx: vec![0.0; size],
            wy: vec![0.0; size],
            rxx: vec![0.0; size],
            ryy: vec![0.0; size],
            rxy: vec![0.0; size],
        }
    }
}

/// Rows of the differences that a row of TGV's steps takes, one of each channel.
#[derive(Debug, Clone)]
struct Differences {
    first: Vec<f64>,
    second: Vec<f64>,
    third: Vec<f64>,
    fourth: Vec<f64>,
}

/// One iteration of TGV as a sweep over the bands: the primal step of a band and the
/// step of `w` on it, and then the dual steps and the relaxation of the band before.
#[derive(Debug, Clone)]
pub struct TgvSweep {
    primal: Primal,
    ascents: Vec<f64>,
    radii: (f64, f64),
    coupled: bool,
    rho: f64,
    rest: f64,
    ring: [Vec<f64>; 2],
    ring_w: [(Vec<f64>, Vec<f64>); 2],
    bar: [Vec<f64>; 2],
    bar_w: [(Vec<f64>, Vec<f64>); 2],
    differences: Differences,
    tx: Vec<f64>,
    ty: Vec<f64>,
    txx: Vec<f64>,
    tyy: Vec<f64>,
    txy: Vec<f64>,
    scale: Vec<f64>,
    scale_r: Vec<f64>,
}

impl TgvSweep {
    /// The sweep of a frame's TGV model with the steps `tau` and `sigma`, the channel
    /// weights `gammas`, the radii `(alpha1, alpha0)` of the balls of `p` and `r`,
    /// coupled or not, and the relaxation `rho`; `steps` are the factors of each
    /// component's proximal map.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "the sweep takes the model, both steps and the relaxation"
    )]
    pub fn new(
        frame: &Frame,
        layout: &Layout,
        steps: &[Step],
        gammas: &[f64],
        tau: f64,
        sigma: f64,
        radii: (f64, f64),
        coupled: bool,
        rho: f64,
    ) -> Self {
        let count = gammas.len();
        let (band, row) = (count * layout.band * layout.width, count * layout.width);
        let pair = || (vec![0.0; row], vec![0.0; row]);
        Self {
            primal: Primal::new(frame, layout, steps, gammas, tau),
            ascents: gammas.iter().map(|gamma| sigma * gamma).collect(),
            radii,
            coupled,
            rho,
            rest: 1.0 - rho,
            ring: [vec![0.0; band], vec![0.0; band]],
            ring_w: [(vec![0.0; band], vec![0.0; band]), (vec![0.0; band], vec![0.0; band])],
            bar: [vec![0.0; row], vec![0.0; row]],
            bar_w: [pair(), pair()],
            differences: Differences {
                first: vec![0.0; layout.width],
                second: vec![0.0; layout.width],
                third: vec![0.0; layout.width],
                fourth: vec![0.0; layout.width],
            },
            tx: vec![0.0; row],
            ty: vec![0.0; row],
            txx: vec![0.0; row],
            tyy: vec![0.0; row],
            txy: vec![0.0; row],
            scale: vec![0.0; row],
            scale_r: vec![0.0; row],
        }
    }

    /// One iteration from the point, which moves to the relaxed point; the outputs of
    /// the proximal steps are written to `outputs` where given.
    pub fn iterate(&mut self, point: &mut TgvPoint, mut outputs: Option<&mut TgvOutputs>) {
        let bands = self.primal.layout.bands();
        for band in 0..=bands {
            if band < bands {
                let fields = (&point.x[..], &point.px[..], &point.py[..]);
                let target = &mut self.ring[band % 2];
                if let Some(out) = outputs.as_deref_mut() {
                    self.primal
                        .band::<true>(band, fields, target, &mut out.first.coefficients);
                } else {
                    self.primal.band::<false>(band, fields, target, &mut []);
                }
                self.advance(band, point);
            }
            if band > 0 {
                self.dual(band - 1, point, outputs.as_deref_mut());
            }
        }
    }

    /// TGV's step of `w` on a band, `w + tau gamma (p + div2 r)`, into the ring.
    fn advance(&mut self, band: usize, point: &TgvPoint) {
        let layout = &self.primal.layout;
        let (width, rows, height) = (layout.width, layout.band, layout.height);
        let plane = layout.plane();
        let Differences {
            first,
            second,
            third,
            fourth,
        } = &mut self.differences;
        let (ring_x, ring_y) = &mut self.ring_w[band % 2];
        for (channel, &step) in self.primal.descents.iter().enumerate() {
            let base = channel * plane;
            for i in 0..rows {
                let r = band * rows + i;
                let below = (r + 1 < height).then_some(r + 1);
                forward_across(row_of(&point.rxx, base, r, width), first, layout.planes, layout.across);
                forward_down(
                    row_of(&point.rxy, base, r, width),
                    below.map(|next| row_of(&point.rxy, base, next, width)),
                    second,
                );
                forward_across(row_of(&point.rxy, base, r, width), third, layout.planes, layout.across);
                forward_down(
                    row_of(&point.ryy, base, r, width),
                    below.map(|next| row_of(&point.ryy, base, next, width)),
                    fourth,
                );
                let start = (channel * rows + i) * width;
                advance_row(
                    &mut ring_x[start..start + width],
                    &mut ring_y[start..start + width],
                    (row_of(&point.wx, base, r, width), row_of(&point.wy, base, r, width)),
                    (row_of(&point.px, base, r, width), row_of(&point.py, base, r, width)),
                    first,
                    second,
                    third,
                    fourth,
                    step,
                );
            }
        }
    }

    /// The dual steps and the relaxation of a band, whose next band's primal step is in
    /// the ring.
    fn dual(&mut self, band: usize, point: &mut TgvPoint, mut outputs: Option<&mut TgvOutputs>) {
        let layout = &self.primal.layout;
        let (width, rows) = (layout.width, layout.band);
        let count = self.ascents.len();
        let (alpha1, alpha0) = self.radii;
        for i in 0..rows {
            for channel in 0..count {
                self.ascend(band, i, channel, point);
            }
            scale_rows(&self.tx, &self.ty, &mut self.scale, width, count, alpha1, self.coupled);
            scale_tensor_rows(
                &self.txx,
                &self.tyy,
                &self.txy,
                &mut self.scale_r,
                width,
                count,
                alpha0,
                self.coupled,
            );
            for channel in 0..count {
                self.settle(band, i, channel, point, outputs.as_deref_mut());
            }
            self.bar.swap(0, 1);
        }
    }

    /// The steps of `p` and `r` of row `i` of a band of one channel, before their
    /// projections: from the extrapolated canvas, whose next row it extrapolates, and
    /// the extrapolated field of the row, beside that of the row above.
    #[expect(
        clippy::similar_names,
        reason = "the names are those of the formulas: the field w and its extrapolation"
    )]
    fn ascend(&mut self, band: usize, i: usize, channel: usize, point: &TgvPoint) {
        let layout = &self.primal.layout;
        let (width, rows, height) = (layout.width, layout.band, layout.height);
        let (planes, across) = (layout.planes, layout.across);
        let r = band * rows + i;
        let row = channel * layout.plane() + r * width;
        let span = channel * width..(channel + 1) * width;
        let start = (channel * rows + i) * width;
        if r == 0 {
            extrapolate_row(
                &mut self.bar[0][span.clone()],
                ring_row(&self.ring, channel, rows, width, 0),
                &point.x[row..row + width],
            );
        }
        let (current, next) = self.bar.split_at_mut(1);
        let (current, next) = (&current[0][span.clone()], &mut next[0][span.clone()]);
        let below = if r + 1 < height {
            let next_row = row + width;
            extrapolate_row(
                next,
                ring_row(&self.ring, channel, rows, width, r + 1),
                &point.x[next_row..next_row + width],
            );
            Some(&*next)
        } else {
            None
        };
        let (own_w, above_w) = if r.is_multiple_of(2) {
            let (own, above) = self.bar_w.split_at_mut(1);
            (&mut own[0], &above[0])
        } else {
            let (above, own) = self.bar_w.split_at_mut(1);
            (&mut own[0], &above[0])
        };
        let (new_x, new_y) = &self.ring_w[band % 2];
        extrapolate_row(
            &mut own_w.0[span.clone()],
            &new_x[start..start + width],
            &point.wx[row..row + width],
        );
        extrapolate_row(
            &mut own_w.1[span.clone()],
            &new_y[start..start + width],
            &point.wy[row..row + width],
        );
        let (bar_wx, bar_wy) = (&own_w.0[span.clone()], &own_w.1[span.clone()]);
        let Differences {
            first,
            second,
            third,
            fourth,
        } = &mut self.differences;
        forward_across(current, first, planes, across);
        forward_down(current, below, second);
        ascend_p_row(
            &mut self.tx[span.clone()],
            &mut self.ty[span.clone()],
            (&point.px[row..row + width], &point.py[row..row + width]),
            (first, second),
            (bar_wx, bar_wy),
            self.ascents[channel],
        );
        // The backward differences down: none of the row's own at the last row, and
        // none of the row above at the first.
        let (last, first_row) = (r + 1 == height, r == 0);
        backward_across(bar_wx, first, planes, across);
        backward_down(
            (!last).then_some(bar_wx),
            (!first_row).then_some(&above_w.0[span.clone()]),
            second,
        );
        backward_across(bar_wy, third, planes, across);
        backward_down(
            (!last).then_some(bar_wy),
            (!first_row).then_some(&above_w.1[span.clone()]),
            fourth,
        );
        ascend_r_row(
            (
                &mut self.txx[span.clone()],
                &mut self.tyy[span.clone()],
                &mut self.txy[span],
            ),
            (
                &point.rxx[row..row + width],
                &point.ryy[row..row + width],
                &point.rxy[row..row + width],
            ),
            first,
            second,
            third,
            fourth,
            self.ascents[channel],
        );
    }

    /// The projections of the steps of row `i` of a band of one channel, and the
    /// relaxation of its point; the outputs of the proximal steps to `outputs` where
    /// given.
    #[expect(
        clippy::similar_names,
        reason = "the names are those of the formulas: x, p_x and p_y and the outputs of each"
    )]
    fn settle(&self, band: usize, i: usize, channel: usize, point: &mut TgvPoint, outputs: Option<&mut TgvOutputs>) {
        let layout = &self.primal.layout;
        let (width, rows) = (layout.width, layout.band);
        let row = channel * layout.plane() + (band * rows + i) * width;
        let span = channel * width..(channel + 1) * width;
        let scales = if self.coupled { 0..width } else { span.clone() };
        let start = (channel * rows + i) * width;
        let new = &self.ring[band % 2][start..start + width];
        let (new_x, new_y) = (
            &self.ring_w[band % 2].0[start..start + width],
            &self.ring_w[band % 2].1[start..start + width],
        );
        let (tx, ty, scale) = (
            &self.tx[span.clone()],
            &self.ty[span.clone()],
            &self.scale[scales.clone()],
        );
        let tensor = (&self.txx[span.clone()], &self.tyy[span.clone()], &self.txy[span]);
        let scale_r = &self.scale_r[scales];
        let (x, px, py) = (
            &mut point.x[row..row + width],
            &mut point.px[row..row + width],
            &mut point.py[row..row + width],
        );
        let field = (&mut point.wx[row..row + width], &mut point.wy[row..row + width]);
        let r_rows = (
            &mut point.rxx[row..row + width],
            &mut point.ryy[row..row + width],
            &mut point.rxy[row..row + width],
        );
        if let Some(out) = outputs {
            let (kept_x, kept_px, kept_py) = (
                &mut out.first.canvas[row..row + width],
                &mut out.first.px[row..row + width],
                &mut out.first.py[row..row + width],
            );
            settle_row::<true>(
                tx, ty, scale, new, x, px, py, kept_x, kept_px, kept_py, self.rho, self.rest,
            );
            let kept_w = (&mut out.wx[row..row + width], &mut out.wy[row..row + width]);
            relax_pair_row::<true>((new_x, new_y), field, kept_w, self.rho, self.rest);
            let kept_r = (
                &mut out.rxx[row..row + width],
                &mut out.ryy[row..row + width],
                &mut out.rxy[row..row + width],
            );
            settle_tensor_row::<true>(tensor, scale_r, r_rows, kept_r, self.rho, self.rest);
        } else {
            let (none_x, none_px, none_py): (&mut [f64], &mut [f64], &mut [f64]) = (&mut [], &mut [], &mut []);
            settle_row::<false>(
                tx, ty, scale, new, x, px, py, none_x, none_px, none_py, self.rho, self.rest,
            );
            let none_w: (&mut [f64], &mut [f64]) = (&mut [], &mut []);
            relax_pair_row::<false>((new_x, new_y), field, none_w, self.rho, self.rest);
            let none_r: (&mut [f64], &mut [f64], &mut [f64]) = (&mut [], &mut [], &mut []);
            settle_tensor_row::<false>(tensor, scale_r, r_rows, none_r, self.rho, self.rest);
        }
    }
}

/// The proximal map of `G` of one component's band: its blocks, from the rows `work`
/// of the band into `target`; the coefficients to `kept` where `KEEP`.
fn prox_band<const KEEP: bool>(
    layout: &Layout,
    component: &Component,
    band: usize,
    work: &[f64],
    target: &mut [f64],
    room: &mut Room,
    kept: &mut [f64],
) {
    let part = &component.part;
    let (width, across) = (layout.width, layout.across);
    let (down, across_ratio) = part.ratio;
    let block_row = part.block_row(across);
    if (down, across_ratio) == (1, 1) {
        for bv in 0..part.band_rows {
            let by = band * part.band_rows + bv;
            let span = BLOCK * bv * width..BLOCK * (bv + 1) * width;
            if by >= part.rows {
                target[span.clone()].copy_from_slice(&work[span]);
                continue;
            }
            let levels = &component.part.levels[by * block_row..(by + 1) * block_row];
            let kept_row: &mut [f64] = if KEEP {
                &mut kept[by * block_row..(by + 1) * block_row]
            } else {
                &mut []
            };
            prox_blocks::<KEEP>(
                &work[span.clone()],
                &mut target[span],
                &mut room.y,
                &mut room.c,
                component,
                levels,
                kept_row,
                across,
            );
        }
        return;
    }
    // Cells of several samples: the blocks of the means of the cells, and the canvas
    // moved by the change of its cell's mean.
    let own_rows = layout.band / down;
    let row = part.planes * across;
    let grid = &mut room.grid[..own_rows * row];
    let count = component.cells;
    for (r, line) in grid.chunks_exact_mut(row).enumerate() {
        for (p, values) in line.chunks_exact_mut(across).enumerate() {
            values.fill(0.0);
            for a in 0..down {
                let source = (down * r + a) * width;
                for k in 0..across_ratio {
                    let start = source + (across_ratio * p + k) * across;
                    for (value, &sample) in values.iter_mut().zip(&work[start..start + across]) {
                        *value += sample;
                    }
                }
            }
            for value in values.iter_mut() {
                *value /= count;
            }
        }
    }
    let fresh = &mut room.fresh[..own_rows * row];
    fresh.copy_from_slice(grid);
    for bv in 0..part.band_rows {
        let by = band * part.band_rows + bv;
        if by >= part.rows {
            continue;
        }
        let span = BLOCK * bv * row..BLOCK * (bv + 1) * row;
        let levels = &component.part.levels[by * block_row..(by + 1) * block_row];
        let kept_row: &mut [f64] = if KEEP {
            &mut kept[by * block_row..(by + 1) * block_row]
        } else {
            &mut []
        };
        prox_blocks::<KEEP>(
            &grid[span.clone()],
            &mut fresh[span],
            &mut room.y,
            &mut room.c,
            component,
            levels,
            kept_row,
            across,
        );
    }
    // The change of each mean, spread over its cell; what the component does not have
    // stays.
    for r in 0..layout.band {
        let own = r / down;
        let by = band * part.band_rows + own / BLOCK;
        for p in 0..layout.planes {
            let own_plane = p / across_ratio;
            let valid = if by < part.rows {
                part.valid[own_plane / BLOCK]
            } else {
                0
            };
            let start = r * width + p * across;
            let own_start = own * row + own_plane * across;
            let (source, out) = (&work[start..start + across], &mut target[start..start + across]);
            let (new, old) = (
                &fresh[own_start..own_start + across],
                &grid[own_start..own_start + across],
            );
            for m in 0..valid {
                out[m] = source[m] + (new[m] - old[m]);
            }
            out[valid..].copy_from_slice(&source[valid..]);
        }
    }
}

/// Row `row` of the canvas of `channel` in the ring of two bands of `rows` rows.
fn ring_row(ring: &[Vec<f64>; 2], channel: usize, rows: usize, width: usize, row: usize) -> &[f64] {
    let start = (channel * rows + row % rows) * width;
    &ring[(row / rows) % 2][start..start + width]
}

/// Row `row` of a channel's canvas that starts at `base`, `width` values.
fn row_of(field: &[f64], base: usize, row: usize, width: usize) -> &[f64] {
    &field[base + row * width..base + (row + 1) * width]
}

/// The starting point in the planes: each component's coefficients, the data term's
/// centres or those given (in natural order), clipped to their intervals; and the
/// canvas, their inverse DCT repeated over the cells, and [`FREE_CENTRE`] beyond the
/// blocks (`frames::start` in the natural layout).
///
/// # Errors
///
/// [`Error::Options`] for coefficients given that are not finite, or not of every
/// component's size.
pub fn start(frame: &Frame, layout: &Layout, given: Option<&[Vec<f64>]>) -> Result<(Vec<f64>, Vec<Vec<f64>>), Error> {
    if let Some(given) = given
        && given.len() != frame.channels().len()
    {
        return Err(Error::Options(format!(
            "coefficients to start from for each of the {} components",
            frame.channels().len()
        )));
    }
    let (width, across) = (layout.width, layout.across);
    let mut canvas = vec![FREE_CENTRE; layout.plane() * layout.parts.len()];
    let mut chosen = Vec::with_capacity(layout.parts.len());
    for (index, (channel, part)) in frame.channels().iter().zip(&layout.parts).enumerate() {
        let problem = channel.problem();
        let row = part.planes * across;
        let block_row = part.block_row(across);
        let levels = &part.levels;
        let mut coefficients = match given {
            Some(given) => {
                let own = &given[index];
                if own.len() != problem.samples() || !own.iter().all(|value| value.is_finite()) {
                    return Err(Error::Options(format!(
                        "the coefficients to start from are finite, {} of them",
                        problem.samples()
                    )));
                }
                layout.values_to_planes(part, own)
            }
            None => vec![0.0; part.rows * block_row],
        };
        let (steps, shrunk, slack) = (problem.steps(), problem.shrunk(), problem.slack());
        for by in 0..part.rows {
            for v in 0..BLOCK {
                for (set, &count) in part.valid.iter().enumerate() {
                    for u in 0..BLOCK {
                        let frequency = BLOCK * v + u;
                        let shift = if frequency == 0 { LEVEL_SHIFT_DC } else { 0.0 };
                        let step = steps[frequency];
                        let start = by * block_row + (v * part.planes + BLOCK * set + u) * across;
                        let (values, own) = (&mut coefficients[start..start + count], &levels[start..start + count]);
                        for (value, &level) in values.iter_mut().zip(own) {
                            let level = f64::from(level);
                            if given.is_none() {
                                *value = kernels::centre(level, step, shrunk[frequency], shift);
                            }
                            let lower = kernels::interval_end(level, -0.5, -slack, step, shift);
                            let upper = kernels::interval_end(level, 0.5, slack, step, shift);
                            *value = kernels::at_most(kernels::at_least(*value, lower), upper);
                        }
                    }
                }
            }
        }
        // The inverse DCT, block row by block row: along the rows, then up the columns;
        // the samples over their cells where the component has the block.
        let (down, across_ratio) = part.ratio;
        let mut y = vec![0.0; BLOCK * row];
        let mut grid = vec![0.0; BLOCK * row];
        for by in 0..part.rows {
            let own = &coefficients[by * block_row..(by + 1) * block_row];
            for v in 0..BLOCK {
                for set in 0..part.sets {
                    let place = v * row + BLOCK * set * across;
                    kernels::inverse8(&own[place..], across, &mut y[place..], across, across);
                }
            }
            kernels::inverse8(&y, row, &mut grid, row, row);
            for r in 0..BLOCK * down {
                let target_row = index * layout.plane() + ((by * BLOCK) * down + r) * width;
                let own_row = &grid[(r / down) * row..(r / down + 1) * row];
                for p in 0..layout.planes {
                    let own_plane = p / across_ratio;
                    let valid = part.valid[own_plane / BLOCK];
                    let start = target_row + p * across;
                    let source = &own_row[own_plane * across..own_plane * across + valid];
                    canvas[start..start + valid].copy_from_slice(source);
                }
            }
        }
        chosen.push(coefficients);
    }
    Ok((canvas, chosen))
}

/// The gradient of a canvas of every channel in the planes, by forward differences: the
/// entries across and down (`frames::gradient` in the natural layout).
#[must_use]
pub fn gradient(layout: &Layout, canvas: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let (width, height) = (layout.width, layout.height);
    let mut across = vec![0.0; canvas.len()];
    let mut down = vec![0.0; canvas.len()];
    for ((((own, lines), target_x), target_y), _) in canvas
        .chunks_exact(layout.plane())
        .zip(canvas.chunks_exact(layout.plane()))
        .zip(across.chunks_exact_mut(layout.plane()))
        .zip(down.chunks_exact_mut(layout.plane()))
        .zip(0..)
    {
        for r in 0..height {
            let row = &own[r * width..(r + 1) * width];
            let below = (r + 1 < height).then(|| &lines[(r + 1) * width..(r + 2) * width]);
            forward_across(
                row,
                &mut target_x[r * width..(r + 1) * width],
                layout.planes,
                layout.across,
            );
            forward_down(row, below, &mut target_y[r * width..(r + 1) * width]);
        }
    }
    (across, down)
}
