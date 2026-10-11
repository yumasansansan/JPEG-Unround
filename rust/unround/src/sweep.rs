// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! One iteration of the primal-dual method as one sweep over the canvas, band of MCU
//! rows by band, in the layout of the planes (docs/math.md, 5.1).
//!
//! The iteration of TV takes the primal step of a band -- the descent along the
//! divergence of the dual, and the proximal map of `G` block by block -- and then
//! the dual step and the relaxation of the band before it, whose forward differences
//! need the new band's first row; TGV's takes the step of `w` of each row of that
//! band just before its dual steps. Only two bands of the new canvas are kept, and
//! the current point is read and written once; the outputs of the proximal steps are
//! written out where a record asks for them.

use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard};

use crate::dct::BLOCK;
use crate::error::Error;
use crate::frames::{FREE_CENTRE, Frame};
use crate::kernels::{
    self, Frequency, Scales, advance_row, ascend_p_row, ascend_r_row, ascend_row, descend_row, dual_p_row, dual_r_row,
    extrapolate_relax_row, forward_across, forward_down, settle_row, settle_tensor_row,
};
use crate::model::{LEVEL_SHIFT_DC, Step};
use crate::planar::{Layout, Part};

/// What the proximal map of one component takes: its layout and levels, the
/// constants of each frequency in a block that is not flat and in one that is,
/// whether they differ in a block there is, its slack, and the samples of its cells.
#[derive(Debug, Clone)]
pub struct Component {
    part: Part,
    frequencies: [[Frequency; BLOCK]; BLOCK],
    flat_frequencies: [[Frequency; BLOCK]; BLOCK],
    flat: bool,
    slack: f64,
    charged: bool,
    cells: f64,
}

impl Component {
    /// The component of a channel with the factors of its proximal map.
    fn new(part: &Part, problem: &crate::model::Problem, step: &Step) -> Self {
        let constants = |(scaled, inverse, shrink): (&[f64; 64], &[f64; 64], &[f64; 64])| {
            let mut frequencies = [[Frequency::default(); BLOCK]; BLOCK];
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
            frequencies
        };
        let (frequencies, flat_frequencies) = (constants(step.factors()), constants(step.flat_factors()));
        // The flat blocks' own map, where they have one; otherwise every block's.
        let flat = problem.flat().contains(&true) && flat_frequencies != frequencies;
        Self {
            part: part.clone(),
            frequencies,
            flat_frequencies,
            flat,
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

/// The 8x8 blocks of block row `by` of a component's grid, `rows` its rows (`8 x P_c x
/// M`, the planes one after another), into `fresh`: the DCT down the columns of every
/// plane at once, the proximal map along the rows between the DCT and its inverse,
/// set of blocks by set, each block with its own map by the marks of the block row's
/// flat blocks, and the inverse down the columns. The coefficients go to `kept` where
/// `KEEP`.
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
    by: usize,
    kept: &mut [f64],
    across: usize,
) {
    let part = &component.part;
    let row = part.planes * across;
    let block_row = part.block_row(across);
    let (levels, marks) = (
        &part.levels[by * block_row..(by + 1) * block_row],
        part.flat_row(by, across),
    );
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
            let (flat, marks) = (&component.flat_frequencies[v], &marks[set * across..]);
            match (component.flat, component.charged) {
                (false, true) => kernels::prox_row::<true>(c, across, own, across, frequencies, component.slack, count),
                (false, false) => {
                    kernels::prox_row::<false>(c, across, own, across, frequencies, component.slack, count);
                }
                (true, true) => {
                    kernels::prox_row_flat::<true>(
                        c,
                        across,
                        own,
                        across,
                        frequencies,
                        flat,
                        marks,
                        component.slack,
                        count,
                    );
                }
                (true, false) => {
                    kernels::prox_row_flat::<false>(
                        c,
                        across,
                        own,
                        across,
                        frequencies,
                        flat,
                        marks,
                        component.slack,
                        count,
                    );
                }
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

/// The rows of a strip of a field in the planes: rows `first..` of every channel's
/// plane, a slice for each channel.
#[derive(Debug)]
struct Rows<'a> {
    channels: Vec<&'a mut [f64]>,
    first: usize,
    width: usize,
}

impl Rows<'_> {
    /// Row `r` of a channel, one of the strip's rows.
    fn row(&self, channel: usize, r: usize) -> &[f64] {
        let start = (r - self.first) * self.width;
        &self.channels[channel][start..start + self.width]
    }

    /// Row `r` of a channel, to write.
    fn row_mut(&mut self, channel: usize, r: usize) -> &mut [f64] {
        let start = (r - self.first) * self.width;
        &mut self.channels[channel][start..start + self.width]
    }
}

/// A field of the planes cut into the strips of rows `ranges`, which follow one another
/// from the first row: each strip's rows of every channel.
fn cut<'a>(field: &'a mut [f64], layout: &Layout, ranges: &[Range<usize>]) -> Vec<Rows<'a>> {
    let width = layout.width;
    let mut strips: Vec<Rows<'a>> = ranges
        .iter()
        .map(|range| Rows {
            channels: Vec::new(),
            first: range.start,
            width,
        })
        .collect();
    for channel in field.chunks_exact_mut(layout.plane()) {
        let mut rest = channel;
        for (strip, range) in strips.iter_mut().zip(ranges) {
            let (own, tail) = std::mem::take(&mut rest).split_at_mut(range.len() * width);
            strip.channels.push(own);
            rest = tail;
        }
    }
    strips
}

/// The block rows of a strip of a component's coefficients in the planes: block rows
/// `first..`, each of `size` values.
#[derive(Debug)]
struct BlockRows<'a> {
    values: &'a mut [f64],
    first: usize,
    size: usize,
}

impl BlockRows<'_> {
    /// Block row `by`, to write.
    fn row_mut(&mut self, by: usize) -> &mut [f64] {
        let start = (by - self.first) * self.size;
        &mut self.values[start..start + self.size]
    }
}

/// Each component's coefficients cut into the strips of bands `bands`: for each strip,
/// each component's block rows of the strip's bands that the component has.
fn cut_blocks<'a>(
    coefficients: &'a mut [Vec<f64>],
    layout: &Layout,
    bands: &[Range<usize>],
) -> Vec<Vec<BlockRows<'a>>> {
    let mut strips: Vec<Vec<BlockRows<'a>>> = bands.iter().map(|_| Vec::new()).collect();
    for (values, part) in coefficients.iter_mut().zip(&layout.parts) {
        let size = part.block_row(layout.across);
        let mut rest: &'a mut [f64] = values;
        for (strip, range) in strips.iter_mut().zip(bands) {
            let first = (range.start * part.band_rows).min(part.rows);
            let end = (range.end * part.band_rows).min(part.rows);
            let (own, tail) = std::mem::take(&mut rest).split_at_mut((end - first) * size);
            strip.push(BlockRows {
                values: own,
                first,
                size,
            });
            rest = tail;
        }
    }
    strips
}

/// The bands of each strip that `threads` threads sweep at once: as many strips as
/// threads, or as bands where there are fewer, the first strips taking one band more
/// where the bands do not divide evenly. One thread sweeps the canvas as one strip.
#[must_use]
pub fn strips(bands: usize, threads: usize) -> Vec<Range<usize>> {
    let count = threads.clamp(1, bands.max(1));
    let (each, extra) = (bands / count, bands % count);
    let mut first = 0;
    (0..count)
        .map(|strip| {
            let end = first + each + usize::from(strip < extra);
            let range = first..end;
            first = end;
            range
        })
        .collect()
}

/// The strips to a thread where there are several threads: a thread that is quicker
/// than the others sweeps more of them.
const STRIPS_PER_THREAD: usize = 2;

/// The strips that `threads` threads sweep a canvas of `bands` bands in: one for one
/// thread, and for more [`STRIPS_PER_THREAD`] to a thread, but none of less than a band.
#[must_use]
pub fn strip_count(bands: usize, threads: usize) -> usize {
    if threads <= 1 {
        1
    } else {
        threads.saturating_mul(STRIPS_PER_THREAD).clamp(1, bands.max(1))
    }
}

/// Waits until a strip has given, or done, what `at_least` counts, unless a strip has
/// failed.
fn wait_until(given: &AtomicU64, at_least: u64, failed: &AtomicBool) {
    let mut spins = 0_u32;
    while given.load(Ordering::Acquire) < at_least {
        assert!(!failed.load(Ordering::Relaxed), "a strip beside this one failed");
        if spins < 4096 {
            spins += 1;
            std::hint::spin_loop();
        } else {
            std::thread::yield_now();
        }
    }
}

/// What a strip holds while a thread sweeps it: its own room, and its rows of the point
/// and of the outputs.
#[derive(Debug)]
struct Work<'s, State, Point, Kept> {
    state: &'s mut State,
    point: Point,
    kept: Option<Kept>,
}

/// Runs `iterations` iterations of every strip, `step` an iteration of one, on `threads`
/// threads, this one among them. Each takes the next iteration of a strip -- the
/// strips of an iteration in order, and the iterations in order -- once that strip's
/// iteration before is done, so that a thread that is quicker sweeps more. A strip
/// waits only for strips taken before it, or for the next one, which a thread takes
/// once its own is done: every wait ends.
fn drive<W: Send>(
    works: Vec<W>,
    iterations: u64,
    threads: usize,
    exchange: &Exchange,
    step: &(dyn Fn(&mut W, usize, u64) + Sync),
) {
    let count = works.len();
    let works: Vec<Mutex<W>> = works.into_iter().map(Mutex::new).collect();
    let done: Vec<AtomicU64> = (0..count).map(|_| AtomicU64::new(0)).collect();
    let next = AtomicU64::new(0);
    let strips = u64::try_from(count).unwrap_or(u64::MAX);
    let total = iterations.saturating_mul(strips);
    let worker = || {
        let _watch = Watch(&exchange.failed);
        loop {
            let item = next.fetch_add(1, Ordering::Relaxed);
            if item >= total {
                break;
            }
            let (iteration, strip) = (item / strips, usize::try_from(item % strips).unwrap_or(0));
            wait_until(&done[strip], iteration, &exchange.failed);
            let mut work = works[strip].lock().expect("a strip beside this one failed");
            step(&mut work, strip, iteration);
            done[strip].store(iteration + 1, Ordering::Release);
        }
    };
    std::thread::scope(|scope| {
        for _ in 1..threads.min(count) {
            scope.spawn(worker);
        }
        worker();
    });
}

/// A buffer of each strip for the iterations of either parity.
type Buffers = Vec<[Mutex<Vec<f64>>; 2]>;

/// What the strips of a sweep give one another in an iteration, by its parity: each
/// strip's last row and first row of the fields that the strips beside it read, as they
/// were when the iteration began, and the extrapolated canvas of its first row once its
/// first band has it. `shown` and `published` are the iterations, counted from 1,
/// whose rows and whose canvas each strip has given; a strip waits for what it takes.
/// What one iteration gives, the next leaves alone: a strip gives again only once the
/// strips beside it have begun the iteration after the one that took it.
#[derive(Debug)]
struct Exchange {
    last: Buffers,
    first: Buffers,
    bar: Buffers,
    shown: Vec<AtomicU64>,
    published: Vec<AtomicU64>,
    failed: AtomicBool,
}

impl Exchange {
    /// Room for `count` strips: `last` and `first` values of the rows given, and `bar`
    /// of the canvas.
    fn new(count: usize, last: usize, first: usize, bar: usize) -> Self {
        let room = |size: usize| -> Buffers {
            (0..count)
                .map(|_| [Mutex::new(vec![0.0; size]), Mutex::new(vec![0.0; size])])
                .collect()
        };
        Self {
            last: room(last),
            first: room(first),
            bar: room(bar),
            shown: (0..count).map(|_| AtomicU64::new(0)).collect(),
            published: (0..count).map(|_| AtomicU64::new(0)).collect(),
            failed: AtomicBool::new(false),
        }
    }
}

/// Raises the exchange's `failed` where a strip's thread unwinds, so that the strips
/// waiting for it stop waiting.
struct Watch<'e>(&'e AtomicBool);

impl Drop for Watch<'_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.0.store(true, Ordering::Relaxed);
        }
    }
}

/// A buffer of the exchange, to read or write.
fn locked(buffer: &Mutex<Vec<f64>>) -> MutexGuard<'_, Vec<f64>> {
    buffer.lock().expect("a strip beside this one failed")
}

/// Where a strip lies among the strips of an iteration, and the exchange between them:
/// none where it is the only one.
#[derive(Debug, Clone, Copy)]
struct Place<'e> {
    strip: usize,
    count: usize,
    exchange: Option<&'e Exchange>,
    iteration: u64,
    parity: usize,
}

impl Place<'_> {
    /// The place of the only strip.
    fn alone() -> Self {
        Self {
            strip: 0,
            count: 1,
            exchange: None,
            iteration: 1,
            parity: 0,
        }
    }

    /// Waits until a strip has given what the iteration takes.
    fn wait(&self, given: &AtomicU64) {
        if let Some(exchange) = self.exchange {
            wait_until(given, self.iteration, &exchange.failed);
        }
    }

    /// Gives the strips beside this one its last row of the fields `last`, where there
    /// is a strip below, and its first row of the fields `first`, where there is one
    /// above, as they are.
    fn show(&self, last: &[&Rows<'_>], last_row: usize, first: &[&Rows<'_>], first_row: usize) {
        let Some(exchange) = self.exchange else {
            return;
        };
        let rows = |fields: &[&Rows<'_>], row: usize, buffer: &mut [f64]| {
            let mut place = 0;
            for field in fields {
                for channel in 0..field.channels.len() {
                    let values = field.row(channel, row);
                    buffer[place..place + values.len()].copy_from_slice(values);
                    place += values.len();
                }
            }
        };
        if self.strip + 1 < self.count {
            rows(last, last_row, &mut locked(&exchange.last[self.strip][self.parity]));
        }
        if self.strip > 0 {
            rows(first, first_row, &mut locked(&exchange.first[self.strip][self.parity]));
        }
        exchange.shown[self.strip].store(self.iteration, Ordering::Release);
    }

    /// The strip above's last row of the fields that it gives, into `into`.
    fn take_last(&self, into: &mut [f64]) {
        if let Some(exchange) = self.exchange {
            self.wait(&exchange.shown[self.strip - 1]);
            into.copy_from_slice(&locked(&exchange.last[self.strip - 1][self.parity]));
        }
    }

    /// The strip below's first row of the fields that it gives, into `into`.
    fn take_first(&self, into: &mut [f64]) {
        if let Some(exchange) = self.exchange {
            self.wait(&exchange.shown[self.strip + 1]);
            into.copy_from_slice(&locked(&exchange.first[self.strip + 1][self.parity]));
        }
    }

    /// Gives the strip above the extrapolated canvas of this strip's first row.
    fn publish(&self, bar: &[f64]) {
        if let Some(exchange) = self.exchange
            && self.strip > 0
        {
            locked(&exchange.bar[self.strip][self.parity]).copy_from_slice(bar);
            exchange.published[self.strip].store(self.iteration, Ordering::Release);
        }
    }

    /// The extrapolated canvas of the strip below's first row, into `into`.
    fn take_bar(&self, into: &mut [f64]) {
        if let Some(exchange) = self.exchange {
            self.wait(&exchange.published[self.strip + 1]);
            into.copy_from_slice(&locked(&exchange.bar[self.strip + 1][self.parity]));
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

    /// The primal step of a band of a strip, `x + tau gamma div p` and the proximal map
    /// of `G`, into `target` (`C x R x W`); the coefficients to `kept` where `KEEP`.
    /// `above` is the row of `p` down above the strip, each channel's, where the strip
    /// is not the first.
    fn band<const KEEP: bool>(
        &mut self,
        band: usize,
        point: (&Rows<'_>, &Rows<'_>, &Rows<'_>),
        above: &[f64],
        target: &mut [f64],
        kept: &mut [BlockRows<'_>],
    ) {
        let (x, px, py) = point;
        let layout = &self.layout;
        let (width, rows) = (layout.width, layout.band);
        for (channel, component) in self.components.iter().enumerate() {
            let first = band * rows;
            for i in 0..rows {
                let r = first + i;
                let down_above = if r == 0 {
                    &self.zeros[..]
                } else if r == py.first {
                    &above[channel * width..(channel + 1) * width]
                } else {
                    py.row(channel, r - 1)
                };
                let down_own = if r + 1 < layout.height {
                    py.row(channel, r)
                } else {
                    &self.zeros[..]
                };
                descend_row(
                    &mut self.work[i * width..(i + 1) * width],
                    x.row(channel, r),
                    px.row(channel, r),
                    down_own,
                    down_above,
                    self.descents[channel],
                    layout.planes,
                    layout.across,
                );
            }
            let own = &mut target[channel * rows * width..(channel + 1) * rows * width];
            let kept_part = if KEEP { kept.get_mut(channel) } else { None };
            prox_band::<KEEP>(layout, component, band, &self.work, own, &mut self.room, kept_part);
        }
    }
}

/// TV's point, cut to a strip.
#[derive(Debug)]
struct TvRows<'a> {
    x: Rows<'a>,
    px: Rows<'a>,
    py: Rows<'a>,
}

/// TV's outputs, cut to a strip.
#[derive(Debug)]
struct TvKept<'a> {
    canvas: Rows<'a>,
    px: Rows<'a>,
    py: Rows<'a>,
    coefficients: Vec<BlockRows<'a>>,
}

/// The model of TV's dual steps and relaxation, which every strip takes.
#[derive(Debug, Clone)]
struct TvModel {
    ascents: Vec<f64>,
    radius: f64,
    coupled: bool,
    rho: f64,
    rest: f64,
}

/// A strip's own room in TV's sweep: the primal step's, the ring of two bands of the
/// new canvas, the extrapolated canvas of two rows, the dual steps of a row, and the
/// row of `p` down above the strip.
#[derive(Debug, Clone)]
struct TvState {
    primal: Primal,
    ring: [Vec<f64>; 2],
    bar: [Vec<f64>; 2],
    tx: Vec<f64>,
    ty: Vec<f64>,
    scale: Vec<f64>,
    above: Vec<f64>,
}

/// One iteration of TV as a sweep over the bands, in strips of bands at once.
#[derive(Debug, Clone)]
pub struct TvSweep {
    model: TvModel,
    states: Vec<TvState>,
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
        let (band, row) = (count * layout.band * layout.width, count * layout.width);
        Self {
            model: TvModel {
                ascents: gammas.iter().map(|gamma| sigma * gamma).collect(),
                radius: alpha,
                coupled,
                rho,
                rest: 1.0 - rho,
            },
            states: vec![TvState {
                primal: Primal::new(frame, layout, steps, gammas, tau),
                ring: [vec![0.0; band], vec![0.0; band]],
                bar: [vec![0.0; row], vec![0.0; row]],
                tx: vec![0.0; row],
                ty: vec![0.0; row],
                scale: vec![0.0; row],
                above: vec![0.0; row],
            }],
        }
    }

    /// One iteration from the point, which moves to the relaxed point; the outputs of
    /// the proximal steps are written to `outputs` where given.
    pub fn iterate(&mut self, point: &mut TvPoint, outputs: Option<&mut TvOutputs>) {
        self.run(point, outputs, 1, 1);
    }

    /// `iterations` iterations from the point, the canvas swept by `threads` threads at
    /// once in strips of bands ([`strip_count`], [`strips`]): to the last bit as by one.
    /// The outputs of the last iteration's proximal steps are written to `outputs` where
    /// given.
    pub fn run(&mut self, point: &mut TvPoint, outputs: Option<&mut TvOutputs>, iterations: u64, threads: usize) {
        if iterations == 0 {
            return;
        }
        let layout = self.states[0].primal.layout.clone();
        let bands = strips(layout.bands(), strip_count(layout.bands(), threads));
        let ranges: Vec<Range<usize>> = bands
            .iter()
            .map(|range| range.start * layout.band..range.end * layout.band)
            .collect();
        while self.states.len() < bands.len() {
            let copy = self.states[0].clone();
            self.states.push(copy);
        }
        let row = self.model.ascents.len() * layout.width;
        let exchange = (bands.len() > 1).then(|| Exchange::new(bands.len(), row, 0, row));
        let mut fields = cut(&mut point.x, &layout, &ranges)
            .into_iter()
            .zip(cut(&mut point.px, &layout, &ranges))
            .zip(cut(&mut point.py, &layout, &ranges))
            .map(|((x, px), py)| TvRows { x, px, py });
        let mut kept: Vec<Option<TvKept<'_>>> = match outputs {
            Some(out) => cut(&mut out.canvas, &layout, &ranges)
                .into_iter()
                .zip(cut(&mut out.px, &layout, &ranges))
                .zip(cut(&mut out.py, &layout, &ranges))
                .zip(cut_blocks(&mut out.coefficients, &layout, &bands))
                .map(|(((canvas, px), py), coefficients)| {
                    Some(TvKept {
                        canvas,
                        px,
                        py,
                        coefficients,
                    })
                })
                .collect(),
            None => bands.iter().map(|_| None).collect(),
        };
        let (model, count) = (&self.model, bands.len());
        let (Some(exchange), true) = (exchange, count > 1) else {
            if let (Some(mut point), Some(mut kept)) = (fields.next(), kept.pop()) {
                for done in 0..iterations {
                    let keep = if done + 1 == iterations { kept.as_mut() } else { None };
                    self.states[0].iteration(model, &Place::alone(), &bands[0], &mut point, keep);
                }
            }
            return;
        };
        let works: Vec<Work<'_, TvState, TvRows<'_>, TvKept<'_>>> = self
            .states
            .iter_mut()
            .zip(fields)
            .zip(kept)
            .map(|((state, point), kept)| Work { state, point, kept })
            .collect();
        let step = |work: &mut Work<'_, TvState, TvRows<'_>, TvKept<'_>>, strip: usize, done: u64| {
            let place = Place {
                strip,
                count,
                exchange: Some(&exchange),
                iteration: done + 1,
                parity: usize::from(done % 2 == 1),
            };
            let keep = if done + 1 == iterations {
                work.kept.as_mut()
            } else {
                None
            };
            work.state
                .iteration(model, &place, &bands[strip], &mut work.point, keep);
        };
        drive(works, iterations, threads, &exchange, &step);
    }
}

impl TvState {
    /// One iteration of a strip of bands, among the strips at `place`: before anything
    /// moves, the strip's last row of p down given to the strip below, and the strip
    /// above's taken for this strip's first row; and then the sweep.
    fn iteration(
        &mut self,
        model: &TvModel,
        place: &Place<'_>,
        bands: &Range<usize>,
        point: &mut TvRows<'_>,
        kept: Option<&mut TvKept<'_>>,
    ) {
        let rows = self.primal.layout.band;
        let (first_row, last_row) = (bands.start * rows, bands.end * rows - 1);
        place.show(&[&point.py], last_row, &[], first_row);
        if place.strip > 0 {
            place.take_last(&mut self.above);
        }
        self.sweep(model, bands, point, kept, place);
    }

    /// One iteration of a strip: the primal step of a band and then the dual step and
    /// the relaxation of the band before it, the extrapolated canvas of the strip's
    /// first row given to the strip above, and that of the row below the strip taken
    /// from the strip below.
    fn sweep(
        &mut self,
        model: &TvModel,
        bands: &Range<usize>,
        point: &mut TvRows<'_>,
        mut kept: Option<&mut TvKept<'_>>,
        place: &Place<'_>,
    ) {
        let (width, rows) = (self.primal.layout.width, self.primal.layout.band);
        let first_row = bands.start * rows;
        for band in bands.start..=bands.end {
            if band < bands.end {
                let fields = (&point.x, &point.px, &point.py);
                let target = &mut self.ring[band % 2];
                if let Some(out) = kept.as_deref_mut() {
                    self.primal
                        .band::<true>(band, fields, &self.above, target, &mut out.coefficients);
                } else {
                    self.primal.band::<false>(band, fields, &self.above, target, &mut []);
                }
                if band == bands.start {
                    for channel in 0..model.ascents.len() {
                        extrapolate_relax(
                            &mut self.bar[0][channel * width..(channel + 1) * width],
                            ring_row(&self.ring, channel, rows, width, first_row),
                            point.x.row_mut(channel, first_row),
                            kept_canvas(&mut kept, channel, first_row),
                            (model.rho, model.rest),
                        );
                    }
                    place.publish(&self.bar[0]);
                }
            }
            if band > bands.start {
                self.dual(model, band - 1, bands, point, kept.as_deref_mut(), place);
            }
        }
    }

    /// The dual step and the relaxation of a band, whose next band's primal step is in
    /// the ring, or at the strip's last band, whose next row's extrapolated canvas is
    /// the strip below's; the canvas of each row is relaxed where it is extrapolated.
    fn dual(
        &mut self,
        model: &TvModel,
        band: usize,
        bands: &Range<usize>,
        point: &mut TvRows<'_>,
        mut kept: Option<&mut TvKept<'_>>,
        place: &Place<'_>,
    ) {
        let layout = &self.primal.layout;
        let (width, rows, height) = (layout.width, layout.band, layout.height);
        let count = model.ascents.len();
        let end_row = bands.end * rows;
        // Each channel's own projection, or the channels' together.
        let own = !model.coupled || count == 1;
        for i in 0..rows {
            let r = band * rows + i;
            if r + 1 == end_row && end_row < height {
                place.take_bar(&mut self.bar[1]);
            }
            for channel in 0..count {
                let span = channel * width..(channel + 1) * width;
                let (current, next) = self.bar.split_at_mut(1);
                let (current, next) = (&current[0][span.clone()], &mut next[0][span.clone()]);
                let below = if r + 1 < height {
                    if r + 1 < end_row {
                        extrapolate_relax(
                            next,
                            ring_row(&self.ring, channel, rows, width, r + 1),
                            point.x.row_mut(channel, r + 1),
                            kept_canvas(&mut kept, channel, r + 1),
                            (model.rho, model.rest),
                        );
                    }
                    Some(&*next)
                } else {
                    None
                };
                if own {
                    // The step projected and relaxed in its own loop.
                    let kept_p = kept
                        .as_deref_mut()
                        .map(|out| (out.px.row_mut(channel, r), out.py.row_mut(channel, r)));
                    dual_p_row(
                        (point.px.row_mut(channel, r), point.py.row_mut(channel, r)),
                        kept_p,
                        current,
                        below,
                        None,
                        model.ascents[channel],
                        (model.radius, model.rho, model.rest),
                        layout.planes,
                        layout.across,
                    );
                    continue;
                }
                ascend_row(
                    &mut self.tx[span.clone()],
                    &mut self.ty[span],
                    point.px.row(channel, r),
                    point.py.row(channel, r),
                    current,
                    below,
                    model.ascents[channel],
                    layout.planes,
                    layout.across,
                    Scales::coupled(&mut self.scale[..width], model.radius, channel, count),
                );
            }
            // The channels' projections together, once all their steps are taken.
            for channel in (0..count).filter(|_| !own) {
                let span = channel * width..(channel + 1) * width;
                let (tx, ty, scale) = (&self.tx[span.clone()], &self.ty[span], &self.scale[..width]);
                let (px, py) = (point.px.row_mut(channel, r), point.py.row_mut(channel, r));
                if let Some(out) = kept.as_deref_mut() {
                    let (p_x, p_y) = (out.px.row_mut(channel, r), out.py.row_mut(channel, r));
                    settle_row::<true>(tx, ty, scale, px, py, p_x, p_y, model.rho, model.rest);
                } else {
                    settle_row::<false>(tx, ty, scale, px, py, &mut [], &mut [], model.rho, model.rest);
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

/// TGV's point, cut to a strip.
#[derive(Debug)]
struct TgvRows<'a> {
    x: Rows<'a>,
    wx: Rows<'a>,
    wy: Rows<'a>,
    px: Rows<'a>,
    py: Rows<'a>,
    rxx: Rows<'a>,
    ryy: Rows<'a>,
    rxy: Rows<'a>,
}

/// TGV's outputs, cut to a strip.
#[derive(Debug)]
struct TgvKept<'a> {
    first: TvKept<'a>,
    wx: Rows<'a>,
    wy: Rows<'a>,
    rxx: Rows<'a>,
    ryy: Rows<'a>,
    rxy: Rows<'a>,
}

/// The fields of the last row of a strip that the strip below takes, in the order of
/// the exchange: `p` down (the primal step's row above), and `p` across, `w` and `r`
/// (the step of `w` and its extrapolation at the row above).
const LAST_FIELDS: usize = 7;

/// The fields of the first row of a strip that the strip above takes: `r12` and `r22`,
/// the rows below its last row in the step of `w`.
const FIRST_FIELDS: usize = 2;

/// The model of TGV's dual steps and relaxation, which every strip takes.
#[derive(Debug, Clone)]
struct TgvModel {
    ascents: Vec<f64>,
    radii: (f64, f64),
    coupled: bool,
    rho: f64,
    rest: f64,
}

/// A strip's own room in TGV's sweep: TV's, the extrapolated field `w` at two rows,
/// the steps of `p` and `r` of a row and their projections,
/// and a row of zeros for the differences down where there is no row; the rows that
/// the strips beside it give, and the new field of the row above the strip.
#[derive(Debug, Clone)]
struct TgvState {
    primal: Primal,
    ring: [Vec<f64>; 2],
    bar: [Vec<f64>; 2],
    bar_w: [(Vec<f64>, Vec<f64>); 2],
    tx: Vec<f64>,
    ty: Vec<f64>,
    txx: Vec<f64>,
    tyy: Vec<f64>,
    txy: Vec<f64>,
    scale: Vec<f64>,
    scale_r: Vec<f64>,
    zeros: Vec<f64>,
    above: Vec<f64>,
    below: Vec<f64>,
    fresh_w: (Vec<f64>, Vec<f64>),
}

/// One iteration of TGV as a sweep over the bands, in strips of bands at once: the
/// primal step of a band, and then the steps of `w`, the dual steps and the relaxation
/// of the band before, row by row.
#[derive(Debug, Clone)]
pub struct TgvSweep {
    model: TgvModel,
    states: Vec<TgvState>,
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
            model: TgvModel {
                ascents: gammas.iter().map(|gamma| sigma * gamma).collect(),
                radii,
                coupled,
                rho,
                rest: 1.0 - rho,
            },
            states: vec![TgvState {
                primal: Primal::new(frame, layout, steps, gammas, tau),
                ring: [vec![0.0; band], vec![0.0; band]],
                bar: [vec![0.0; row], vec![0.0; row]],
                bar_w: [pair(), pair()],
                tx: vec![0.0; row],
                ty: vec![0.0; row],
                txx: vec![0.0; row],
                tyy: vec![0.0; row],
                txy: vec![0.0; row],
                scale: vec![0.0; row],
                scale_r: vec![0.0; row],
                zeros: vec![0.0; layout.width],
                above: vec![0.0; LAST_FIELDS * row],
                below: vec![0.0; FIRST_FIELDS * row],
                fresh_w: (vec![0.0; layout.width], vec![0.0; layout.width]),
            }],
        }
    }

    /// One iteration from the point, which moves to the relaxed point; the outputs of
    /// the proximal steps are written to `outputs` where given.
    pub fn iterate(&mut self, point: &mut TgvPoint, outputs: Option<&mut TgvOutputs>) {
        self.run(point, outputs, 1, 1);
    }

    /// `iterations` iterations from the point, the canvas swept by `threads` threads at
    /// once in strips of bands ([`strip_count`], [`strips`]): to the last bit as by one.
    /// The outputs of the last iteration's proximal steps are written to `outputs` where
    /// given.
    pub fn run(&mut self, point: &mut TgvPoint, outputs: Option<&mut TgvOutputs>, iterations: u64, threads: usize) {
        if iterations == 0 {
            return;
        }
        let layout = self.states[0].primal.layout.clone();
        let bands = strips(layout.bands(), strip_count(layout.bands(), threads));
        let ranges: Vec<Range<usize>> = bands
            .iter()
            .map(|range| range.start * layout.band..range.end * layout.band)
            .collect();
        while self.states.len() < bands.len() {
            let copy = self.states[0].clone();
            self.states.push(copy);
        }
        let row = self.model.ascents.len() * layout.width;
        let exchange =
            (bands.len() > 1).then(|| Exchange::new(bands.len(), LAST_FIELDS * row, FIRST_FIELDS * row, row));
        let mut fields = tgv_strips(point, &layout, &ranges).into_iter();
        let mut kept: Vec<Option<TgvKept<'_>>> = match outputs {
            Some(out) => tgv_kept_strips(out, &layout, &ranges, &bands)
                .into_iter()
                .map(Some)
                .collect(),
            None => bands.iter().map(|_| None).collect(),
        };
        let (model, count) = (&self.model, bands.len());
        let (Some(exchange), true) = (exchange, count > 1) else {
            if let (Some(mut point), Some(mut kept)) = (fields.next(), kept.pop()) {
                for done in 0..iterations {
                    let keep = if done + 1 == iterations { kept.as_mut() } else { None };
                    self.states[0].iteration(model, &Place::alone(), &bands[0], &mut point, keep);
                }
            }
            return;
        };
        let works: Vec<Work<'_, TgvState, TgvRows<'_>, TgvKept<'_>>> = self
            .states
            .iter_mut()
            .zip(fields)
            .zip(kept)
            .map(|((state, point), kept)| Work { state, point, kept })
            .collect();
        let step = |work: &mut Work<'_, TgvState, TgvRows<'_>, TgvKept<'_>>, strip: usize, done: u64| {
            let place = Place {
                strip,
                count,
                exchange: Some(&exchange),
                iteration: done + 1,
                parity: usize::from(done % 2 == 1),
            };
            let keep = if done + 1 == iterations {
                work.kept.as_mut()
            } else {
                None
            };
            work.state
                .iteration(model, &place, &bands[strip], &mut work.point, keep);
        };
        drive(works, iterations, threads, &exchange, &step);
    }
}

/// TGV's point cut into the strips of rows `ranges`.
fn tgv_strips<'a>(point: &'a mut TgvPoint, layout: &Layout, ranges: &[Range<usize>]) -> Vec<TgvRows<'a>> {
    let TgvPoint {
        x,
        wx,
        wy,
        px,
        py,
        rxx,
        ryy,
        rxy,
    } = point;
    cut(x, layout, ranges)
        .into_iter()
        .zip(cut(wx, layout, ranges))
        .zip(cut(wy, layout, ranges))
        .zip(cut(px, layout, ranges))
        .zip(cut(py, layout, ranges))
        .zip(cut(rxx, layout, ranges))
        .zip(cut(ryy, layout, ranges))
        .zip(cut(rxy, layout, ranges))
        .map(|(((((((x, wx), wy), px), py), rxx), ryy), rxy)| TgvRows {
            x,
            wx,
            wy,
            px,
            py,
            rxx,
            ryy,
            rxy,
        })
        .collect()
}

/// TGV's outputs cut into the strips of rows `ranges`, of bands `bands`.
fn tgv_kept_strips<'a>(
    outputs: &'a mut TgvOutputs,
    layout: &Layout,
    ranges: &[Range<usize>],
    bands: &[Range<usize>],
) -> Vec<TgvKept<'a>> {
    let TgvOutputs {
        first,
        wx,
        wy,
        rxx,
        ryy,
        rxy,
    } = outputs;
    cut(&mut first.canvas, layout, ranges)
        .into_iter()
        .zip(cut(&mut first.px, layout, ranges))
        .zip(cut(&mut first.py, layout, ranges))
        .zip(cut_blocks(&mut first.coefficients, layout, bands))
        .zip(cut(wx, layout, ranges))
        .zip(cut(wy, layout, ranges))
        .zip(cut(rxx, layout, ranges))
        .zip(cut(ryy, layout, ranges))
        .zip(cut(rxy, layout, ranges))
        .map(
            |((((((((canvas, px), py), coefficients), wx), wy), rxx), ryy), rxy)| TgvKept {
                first: TvKept {
                    canvas,
                    px,
                    py,
                    coefficients,
                },
                wx,
                wy,
                rxx,
                ryy,
                rxy,
            },
        )
        .collect()
}

impl TgvState {
    /// One iteration of a strip of bands, among the strips at `place`: before anything
    /// moves, the rows that the strips beside this one take given to them, and the strip
    /// above's last row taken, from which this strip makes the new field of the row
    /// above it; and then the sweep.
    fn iteration(
        &mut self,
        model: &TgvModel,
        place: &Place<'_>,
        bands: &Range<usize>,
        point: &mut TgvRows<'_>,
        kept: Option<&mut TgvKept<'_>>,
    ) {
        let rows = self.primal.layout.band;
        let (first_row, last_row) = (bands.start * rows, bands.end * rows - 1);
        let last = [
            &point.py, &point.px, &point.wx, &point.wy, &point.rxx, &point.rxy, &point.ryy,
        ];
        place.show(&last, last_row, &[&point.rxy, &point.ryy], first_row);
        if place.strip > 0 {
            place.take_last(&mut self.above);
            self.field_above(model, first_row, point);
        }
        self.sweep(model, bands, point, kept, place);
    }

    /// The extrapolated field of the row above the strip, as the strip above makes it in
    /// the iteration (the step of `w` and its extrapolation), from that strip's last row
    /// as it was and this strip's first row of `r`: into the row of `bar_w` that the
    /// strip's first row takes as the row above.
    fn field_above(&mut self, model: &TgvModel, first_row: usize, point: &TgvRows<'_>) {
        let layout = &self.primal.layout;
        let (width, planes, across) = (layout.width, layout.planes, layout.across);
        let count = model.ascents.len();
        let slot = (first_row - 1) % 2;
        let (fresh_x, fresh_y) = &mut self.fresh_w;
        for (channel, &step) in self.primal.descents.iter().enumerate() {
            let field = |k: usize| &self.above[(k * count + channel) * width..(k * count + channel + 1) * width];
            let (py, px, wx, wy, rxx, rxy, ryy) =
                (field(0), field(1), field(2), field(3), field(4), field(5), field(6));
            let below = (point.ryy.row(channel, first_row), point.rxy.row(channel, first_row));
            let span = channel * width..(channel + 1) * width;
            let (bar_x, bar_y) = &mut self.bar_w[slot];
            // The step relaxes a copy of the row's field, which the strip above relaxes.
            fresh_x.copy_from_slice(wx);
            fresh_y.copy_from_slice(wy);
            advance_row(
                (&mut bar_x[span.clone()], &mut bar_y[span]),
                (fresh_x, fresh_y),
                None,
                (px, py),
                (rxx, ryy, rxy),
                Some(below),
                (step, model.rho, model.rest),
                planes,
                across,
            );
        }
    }

    /// One iteration of a strip: the primal step of a band, and then the steps of `w`,
    /// the dual steps and the relaxation of the band before it; the extrapolated canvas
    /// of the strip's first row given to the strip above, and that of the row below the
    /// strip taken from the strip below.
    fn sweep(
        &mut self,
        model: &TgvModel,
        bands: &Range<usize>,
        point: &mut TgvRows<'_>,
        mut kept: Option<&mut TgvKept<'_>>,
        place: &Place<'_>,
    ) {
        let layout = &self.primal.layout;
        let (width, rows) = (layout.width, layout.band);
        let first_row = bands.start * rows;
        for band in bands.start..=bands.end {
            if band < bands.end {
                let fields = (&point.x, &point.px, &point.py);
                let target = &mut self.ring[band % 2];
                let above = &self.above[..model.ascents.len() * width];
                if let Some(out) = kept.as_deref_mut() {
                    self.primal
                        .band::<true>(band, fields, above, target, &mut out.first.coefficients);
                } else {
                    self.primal.band::<false>(band, fields, above, target, &mut []);
                }
                if band == bands.start {
                    for channel in 0..model.ascents.len() {
                        extrapolate_relax(
                            &mut self.bar[0][channel * width..(channel + 1) * width],
                            ring_row(&self.ring, channel, rows, width, first_row),
                            point.x.row_mut(channel, first_row),
                            kept.as_deref_mut()
                                .map(|out| out.first.canvas.row_mut(channel, first_row)),
                            (model.rho, model.rest),
                        );
                    }
                    place.publish(&self.bar[0]);
                }
            }
            if band > bands.start {
                self.dual(model, band - 1, bands, point, kept.as_deref_mut(), place);
            }
        }
    }

    /// The steps of `w`, the dual steps and the relaxation of a band, whose next band's
    /// primal step is in the ring, or at the strip's last band, whose next row's
    /// extrapolated canvas and rows of `r` are the strip below's.
    fn dual(
        &mut self,
        model: &TgvModel,
        band: usize,
        bands: &Range<usize>,
        point: &mut TgvRows<'_>,
        mut kept: Option<&mut TgvKept<'_>>,
        place: &Place<'_>,
    ) {
        let layout = &self.primal.layout;
        let (rows, height) = (layout.band, layout.height);
        let count = model.ascents.len();
        let end_row = bands.end * rows;
        for i in 0..rows {
            let r = band * rows + i;
            if r + 1 == end_row && end_row < height {
                place.take_bar(&mut self.bar[1]);
                place.take_first(&mut self.below);
            }
            for channel in 0..count {
                self.ascend(model, (band, i, channel, end_row), point, kept.as_deref_mut());
            }
            // The channels' projections together, once all their steps are taken; a
            // channel's own were taken in the loops of its steps.
            for channel in (0..count).filter(|_| model.coupled && count > 1) {
                self.settle(model, (band, i, channel), point, kept.as_deref_mut());
            }
            self.bar.swap(0, 1);
        }
    }

    /// The step of `w` of row `i` of a band of one channel, `w + tau gamma (p + div2 r)`,
    /// its extrapolation and the relaxation of `w`, and the steps of `p` and `r` of the
    /// row, before their projections: from the extrapolated canvas, whose next row it
    /// extrapolates and relaxes (but at the strip's last row), and the extrapolated field
    /// of the row, beside that of the row above. At the strip's last row, the rows of `r`
    /// below are the strip below's, as they were. The outputs of the proximal steps of
    /// the canvas and of `w` go to `kept` where given.
    #[expect(
        clippy::similar_names,
        reason = "the names are those of the formulas: the field w and its extrapolation"
    )]
    fn ascend(
        &mut self,
        model: &TgvModel,
        (band, i, channel, end_row): (usize, usize, usize, usize),
        point: &mut TgvRows<'_>,
        mut kept: Option<&mut TgvKept<'_>>,
    ) {
        let layout = &self.primal.layout;
        let (width, rows, height) = (layout.width, layout.band, layout.height);
        let (planes, across) = (layout.planes, layout.across);
        let r = band * rows + i;
        let span = channel * width..(channel + 1) * width;
        let (current, next) = self.bar.split_at_mut(1);
        let (current, next) = (&current[0][span.clone()], &mut next[0][span.clone()]);
        let below = if r + 1 < height {
            if r + 1 < end_row {
                extrapolate_relax(
                    next,
                    ring_row(&self.ring, channel, rows, width, r + 1),
                    point.x.row_mut(channel, r + 1),
                    kept.as_deref_mut().map(|out| out.first.canvas.row_mut(channel, r + 1)),
                    (model.rho, model.rest),
                );
            }
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
        // The rows of r22 and r12 below.
        let count = model.ascents.len();
        let below_r = if r + 1 >= height {
            None
        } else if r + 1 == end_row {
            let rows_of = |k: usize| &self.below[(k * count + channel) * width..(k * count + channel + 1) * width];
            Some((rows_of(1), rows_of(0)))
        } else {
            Some((point.ryy.row(channel, r + 1), point.rxy.row(channel, r + 1)))
        };
        let kept_w = kept
            .as_deref_mut()
            .map(|out| (out.wx.row_mut(channel, r), out.wy.row_mut(channel, r)));
        advance_row(
            (&mut own_w.0[span.clone()], &mut own_w.1[span.clone()]),
            (point.wx.row_mut(channel, r), point.wy.row_mut(channel, r)),
            kept_w,
            (point.px.row(channel, r), point.py.row(channel, r)),
            (
                point.rxx.row(channel, r),
                point.ryy.row(channel, r),
                point.rxy.row(channel, r),
            ),
            below_r,
            (self.primal.descents[channel], model.rho, model.rest),
            planes,
            across,
        );
        let (bar_wx, bar_wy) = (&own_w.0[span.clone()], &own_w.1[span.clone()]);
        // The backward differences down: none of the row's own at the last row, and
        // none of the row above at the first, zeros in their place.
        let zeros = (&self.zeros[..], &self.zeros[..]);
        let own = if r + 1 == height { zeros } else { (bar_wx, bar_wy) };
        let above = if r == 0 {
            zeros
        } else {
            (&above_w.0[span.clone()], &above_w.1[span.clone()])
        };
        if !model.coupled || count == 1 {
            // Each channel's own projections, taken in the loops of its steps.
            let rows = ((current, below), ((bar_wx, bar_wy), own, above));
            tgv_own_steps(model, (r, channel), point, kept, rows, (planes, across));
            return;
        }
        let (scales, scales_r) = tgv_scales(model, (&mut self.scale, &mut self.scale_r), channel, width);
        ascend_p_row(
            (&mut self.tx[span.clone()], &mut self.ty[span.clone()]),
            (point.px.row(channel, r), point.py.row(channel, r)),
            current,
            below,
            (bar_wx, bar_wy),
            model.ascents[channel],
            planes,
            across,
            scales,
        );
        ascend_r_row(
            (
                &mut self.txx[span.clone()],
                &mut self.tyy[span.clone()],
                &mut self.txy[span.clone()],
            ),
            (
                point.rxx.row(channel, r),
                point.ryy.row(channel, r),
                point.rxy.row(channel, r),
            ),
            (bar_wx, bar_wy),
            own,
            above,
            model.ascents[channel],
            planes,
            across,
            scales_r,
        );
    }

    /// The projections of the dual steps of row `i` of a band of one channel, and the
    /// relaxation of `p` and `r`; the outputs of the proximal steps to `kept` where
    /// given.
    fn settle(
        &self,
        model: &TgvModel,
        (band, i, channel): (usize, usize, usize),
        point: &mut TgvRows<'_>,
        kept: Option<&mut TgvKept<'_>>,
    ) {
        let layout = &self.primal.layout;
        let (width, rows) = (layout.width, layout.band);
        let r = band * rows + i;
        let span = channel * width..(channel + 1) * width;
        let scales = if model.coupled { 0..width } else { span.clone() };
        let (tx, ty, scale) = (
            &self.tx[span.clone()],
            &self.ty[span.clone()],
            &self.scale[scales.clone()],
        );
        let tensor = (&self.txx[span.clone()], &self.tyy[span.clone()], &self.txy[span]);
        let scale_r = &self.scale_r[scales];
        let TgvRows {
            px, py, rxx, ryy, rxy, ..
        } = point;
        let (px, py) = (px.row_mut(channel, r), py.row_mut(channel, r));
        let r_rows = (
            rxx.row_mut(channel, r),
            ryy.row_mut(channel, r),
            rxy.row_mut(channel, r),
        );
        let (rho, rest) = (model.rho, model.rest);
        if let Some(out) = kept {
            let (p_x, p_y) = (out.first.px.row_mut(channel, r), out.first.py.row_mut(channel, r));
            settle_row::<true>(tx, ty, scale, px, py, p_x, p_y, rho, rest);
            let kept_r = (
                out.rxx.row_mut(channel, r),
                out.ryy.row_mut(channel, r),
                out.rxy.row_mut(channel, r),
            );
            settle_tensor_row::<true>(tensor, scale_r, r_rows, kept_r, rho, rest);
        } else {
            settle_row::<false>(tx, ty, scale, px, py, &mut [], &mut [], rho, rest);
            let none_r: (&mut [f64], &mut [f64], &mut [f64]) = (&mut [], &mut [], &mut []);
            settle_tensor_row::<false>(tensor, scale_r, r_rows, none_r, rho, rest);
        }
    }
}

/// The scales of the projections of TGV's dual steps of a channel among coupled
/// channels: its squares added to those of the channels before it.
fn tgv_scales<'s>(
    model: &TgvModel,
    (scale, scale_r): (&'s mut [f64], &'s mut [f64]),
    channel: usize,
    width: usize,
) -> (Scales<'s>, Scales<'s>) {
    let (alpha1, alpha0) = model.radii;
    let count = model.ascents.len();
    (
        Scales::coupled(&mut scale[..width], alpha1, channel, count),
        Scales::coupled(&mut scale_r[..width], alpha0, channel, count),
    )
}

/// The rows of the canvas that TGV's step of `p` takes, the row and the row below it,
/// and the rows of the field that its steps take: its own extrapolation, its own
/// entries down and those of the row above.
type StepRows<'a> = (
    (&'a [f64], Option<&'a [f64]>),
    ((&'a [f64], &'a [f64]), (&'a [f64], &'a [f64]), (&'a [f64], &'a [f64])),
);

/// TGV's dual steps of row `r` of a channel whose projections are its own, projected
/// and relaxed in their own loops; the outputs of the proximal steps to `kept` where
/// given.
fn tgv_own_steps(
    model: &TgvModel,
    (r, channel): (usize, usize),
    point: &mut TgvRows<'_>,
    kept: Option<&mut TgvKept<'_>>,
    ((current, below), (field, own, above)): StepRows<'_>,
    (planes, across): (usize, usize),
) {
    let (alpha1, alpha0) = model.radii;
    let step = model.ascents[channel];
    let (kept_p, kept_r) = match kept {
        Some(out) => (
            Some((out.first.px.row_mut(channel, r), out.first.py.row_mut(channel, r))),
            Some((
                out.rxx.row_mut(channel, r),
                out.ryy.row_mut(channel, r),
                out.rxy.row_mut(channel, r),
            )),
        ),
        None => (None, None),
    };
    dual_p_row(
        (point.px.row_mut(channel, r), point.py.row_mut(channel, r)),
        kept_p,
        current,
        below,
        Some(field),
        step,
        (alpha1, model.rho, model.rest),
        planes,
        across,
    );
    dual_r_row(
        (
            point.rxx.row_mut(channel, r),
            point.ryy.row_mut(channel, r),
            point.rxy.row_mut(channel, r),
        ),
        kept_r,
        field,
        own,
        above,
        step,
        (alpha0, model.rho, model.rest),
        planes,
        across,
    );
}

/// The extrapolation of a row of the canvas into `bar` and the relaxation of the row
/// `x` towards `new` (`rho` and `rest`), `new` written to `kept` where given.
fn extrapolate_relax(bar: &mut [f64], new: &[f64], x: &mut [f64], kept: Option<&mut [f64]>, (rho, rest): (f64, f64)) {
    match kept {
        Some(kept) => extrapolate_relax_row::<true>(bar, new, x, kept, rho, rest),
        None => extrapolate_relax_row::<false>(bar, new, x, &mut [], rho, rest),
    }
}

/// TV's row `r` of the canvas's output of channel `channel`, where the outputs are kept.
fn kept_canvas<'k>(kept: &'k mut Option<&mut TvKept<'_>>, channel: usize, r: usize) -> Option<&'k mut [f64]> {
    kept.as_deref_mut().map(|out| out.canvas.row_mut(channel, r))
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
    mut kept: Option<&mut BlockRows<'_>>,
) {
    let part = &component.part;
    let (width, across) = (layout.width, layout.across);
    let (down, across_ratio) = part.ratio;
    if (down, across_ratio) == (1, 1) {
        for bv in 0..part.band_rows {
            let by = band * part.band_rows + bv;
            let span = BLOCK * bv * width..BLOCK * (bv + 1) * width;
            if by >= part.rows {
                target[span.clone()].copy_from_slice(&work[span]);
                continue;
            }
            let kept_row: &mut [f64] = match kept.as_deref_mut() {
                Some(rows) if KEEP => rows.row_mut(by),
                _ => &mut [],
            };
            prox_blocks::<KEEP>(
                &work[span.clone()],
                &mut target[span],
                &mut room.y,
                &mut room.c,
                component,
                by,
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
        let kept_row: &mut [f64] = match kept.as_deref_mut() {
            Some(rows) if KEEP => rows.row_mut(by),
            _ => &mut [],
        };
        prox_blocks::<KEEP>(
            &grid[span.clone()],
            &mut fresh[span],
            &mut room.y,
            &mut room.c,
            component,
            by,
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
