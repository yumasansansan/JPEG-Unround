// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! The primal-dual hybrid gradient method of Chambolle and Pock for TV and TGV, in
//! the form of Condat with a relaxation (docs/math.md, 5 and 6).
//!
//! Each iterate's canvas is the output of the proximal map of `G`, and so lies in
//! the quantization constraint set: its coefficients are kept, and are the result.
//! The solvers stop at the first record where a tolerance of the options is met
//! (the duality gap per sample, the gap relative to the primal value, or TGV's
//! partial gap per sample), or after the most iterations the options allow, or
//! where an observer asks them to. Every value that shapes the iterations is an
//! option; the models' defaults stand where none is given.

use std::ops::ControlFlow;

use crate::dct::multiply_add;
use crate::error::Error;
use crate::exact::nearest_from_usize;
use crate::frames::{self, Dual, Frame, Primal, Tensor, Vector};
use crate::model::Step;
use crate::model::{Tgv, Tv};
use crate::results::{FrameResult, Recorder, Stop};

/// A bound of `||grad||^2` (docs/math.md, 3.3): TV's `L^2` by default, times the
/// largest channel weight squared.
pub const TV_NORM_SQUARED: f64 = 8.0;

/// A bound of `||K||^2` for TGV's `K(x, w) = (grad x - w, E w)` (docs/math.md, 3.3),
/// `(17 + sqrt(33)) / 2`: TGV's `L^2` by default, alike.
#[must_use]
pub fn tgv_norm_squared() -> f64 {
    f64::midpoint(17.0, 33.0f64.sqrt())
}

/// `sigma tau L^2` by default, below 1 as the method's convergence asks.
pub const STEP_PRODUCT: f64 = 0.99;

/// `tau / sigma` for TV, over `alpha` squared (docs/math.md, 6.5).
pub const TV_RATIO: f64 = 30.0;
/// `tau / sigma` for TGV, over `alpha1` squared.
pub const TGV_RATIO: f64 = 10.0;
/// The gap per sample at which TV stops, times `alpha`.
pub const TV_TOLERANCE: f64 = 2e-4;
/// The gap per sample at which TGV stops, times `alpha1`.
pub const TGV_TOLERANCE: f64 = 1e-2;
/// The most iterations of TV.
pub const TV_ITERATIONS: u64 = 20_000;
/// The most iterations of TGV.
pub const TGV_ITERATIONS: u64 = 10_000;
/// The relaxation of TV's iterations.
pub const TV_RELAXATION: f64 = 1.9;
/// The relaxation of TGV's iterations.
pub const TGV_RELAXATION: f64 = 1.9;

/// When a solver stops, its steps, and how often it records (docs/math.md, 5, 6.4
/// and 6.6).
///
/// `iterations` is the most iterations. The solver stops earlier, at the first
/// record where one of these holds, each off at 0: the duality gap per sample is at
/// most `tolerance`; the gap is at most `relative_tolerance` times the primal value;
/// or, for TGV, the partial gap of the radius `partial_radius` (6.3) is at most
/// `partial_tolerance` per sample. `step_ratio` is `tau / sigma`, `relaxation` the
/// `rho` of the relaxed steps, in (0, 2), `step_product` `sigma tau L^2`, in (0, 1),
/// and `norm_squared` the `L^2` of the steps, a bound of `||K||^2`: below `||K||^2`
/// the method need not converge. Where they are `None`, they are the model's
/// defaults ([`plan`]), which follow the weight of the first-order term unless
/// `scale_with_weight` is false. Every `record_every` iterations (0: none but the
/// last), and after the last, the solver records the primal and dual values, and
/// checks the tolerances. Where a frame has free samples, every gap is the partial
/// gap of `free_radius`, `R` of 6.6.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Options {
    /// The most iterations.
    pub iterations: Option<u64>,
    /// The gap per sample at which the solver stops.
    pub tolerance: Option<f64>,
    /// The gap, relative to the primal value, at which the solver stops.
    pub relative_tolerance: f64,
    /// TGV's partial gap per sample at which the solver stops.
    pub partial_tolerance: f64,
    /// `tau / sigma`.
    pub step_ratio: Option<f64>,
    /// `rho`, in (0, 2).
    pub relaxation: Option<f64>,
    /// `sigma tau L^2`, in (0, 1).
    pub step_product: f64,
    /// `L^2`, a bound of `||K||^2`.
    pub norm_squared: Option<f64>,
    /// Whether the defaults of the ratio and of the tolerance follow the weight.
    pub scale_with_weight: bool,
    /// How many iterations lie between records.
    pub record_every: u64,
    /// The radius of TGV's partial gap, where it is taken.
    pub partial_radius: Option<f64>,
    /// `R` of the box of free samples.
    pub free_radius: f64,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            iterations: None,
            tolerance: None,
            relative_tolerance: 0.0,
            partial_tolerance: 0.0,
            step_ratio: None,
            relaxation: None,
            step_product: STEP_PRODUCT,
            norm_squared: None,
            scale_with_weight: true,
            record_every: 10,
            partial_radius: None,
            free_radius: frames::FREE_RADIUS,
        }
    }
}

/// What a solver does: its [`Options`], with the model's defaults in place of `None`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Plan {
    /// The most iterations.
    pub iterations: u64,
    /// The gap per sample at which the solver stops.
    pub tolerance: f64,
    /// The gap, relative to the primal value, at which the solver stops.
    pub relative_tolerance: f64,
    /// TGV's partial gap per sample at which the solver stops.
    pub partial_tolerance: f64,
    /// `tau / sigma`.
    pub step_ratio: f64,
    /// `rho`.
    pub relaxation: f64,
    /// `sigma tau L^2`.
    pub step_product: f64,
    /// `L^2`.
    pub norm_squared: f64,
    /// How many iterations lie between records.
    pub record_every: u64,
    /// The radius of TGV's partial gap, where it is taken.
    pub partial_radius: Option<f64>,
    /// `R` of the box of free samples.
    pub free_radius: f64,
}

/// The weights of the model a solver minimizes.
#[derive(Debug, Clone, Copy)]
pub enum Weights<'a> {
    /// Total variation.
    Tv(&'a Tv),
    /// Second-order total generalized variation.
    Tgv(&'a Tgv),
}

/// The options with the model's defaults in place of `None`, checked (docs/math.md,
/// 5 and 6.5).
///
/// The defaults were chosen with the weight `alpha` of TV, or `alpha1` of TGV, of 1.
/// The dual variables are in its units and the objective scales with it, so the
/// ratio of the steps is the default over its square and the tolerance the default
/// times it, unless `scale_with_weight` is false. `L^2` is by default the bound of
/// the model times the largest channel weight squared.
///
/// # Errors
///
/// [`Error::Options`] for what the method cannot run with: every value out of its
/// range, in one message.
pub fn plan(options: &Options, weights: Weights<'_>) -> Result<Plan, Error> {
    let (iterations, tolerance, ratio, alpha, relaxation, mut norm_squared, channel_weights) = match weights {
        Weights::Tv(tv) => {
            if options.partial_radius.is_some() || options.partial_tolerance != 0.0 {
                return Err(Error::Options("TV has no partial gap".into()));
            }
            (
                TV_ITERATIONS,
                TV_TOLERANCE,
                TV_RATIO,
                tv.alpha,
                TV_RELAXATION,
                TV_NORM_SQUARED,
                tv.channel_weights.as_deref(),
            )
        }
        Weights::Tgv(tgv) => {
            if tgv.alpha0.is_nan() || tgv.alpha0 <= 0.0 {
                return Err(Error::Options(format!(
                    "the weight of the second-order term is positive, not {}",
                    tgv.alpha0
                )));
            }
            (
                TGV_ITERATIONS,
                TGV_TOLERANCE,
                TGV_RATIO,
                tgv.alpha1,
                TGV_RELAXATION,
                tgv_norm_squared(),
                tgv.channel_weights.as_deref(),
            )
        }
    };
    if alpha.is_nan() || alpha <= 0.0 {
        return Err(Error::Options(format!(
            "the weight of the first-order term is positive, not {alpha}"
        )));
    }
    if let Some(gammas) = channel_weights {
        if gammas.is_empty() || !gammas.iter().all(|&gamma| gamma > 0.0 && gamma.is_finite()) {
            return Err(Error::Options(format!(
                "the channel weights are positive and finite, not {gammas:?}"
            )));
        }
        let largest = gammas.iter().copied().fold(0.0, f64::max);
        norm_squared *= largest * largest;
    }
    let scale = if options.scale_with_weight { alpha } else { 1.0 };
    let settled = Plan {
        iterations: options.iterations.unwrap_or(iterations),
        tolerance: options.tolerance.unwrap_or(tolerance * scale),
        relative_tolerance: options.relative_tolerance,
        partial_tolerance: options.partial_tolerance,
        step_ratio: options.step_ratio.unwrap_or(ratio / (scale * scale)),
        relaxation: options.relaxation.unwrap_or(relaxation),
        step_product: options.step_product,
        norm_squared: options.norm_squared.unwrap_or(norm_squared),
        record_every: options.record_every,
        partial_radius: options.partial_radius,
        free_radius: options.free_radius,
    };
    check(&settled)?;
    Ok(settled)
}

/// Refuses each value out of its range, all of them in one error.
fn check(settled: &Plan) -> Result<(), Error> {
    let radius = settled.partial_radius;
    let ranges = [
        (
            settled.tolerance >= 0.0,
            format!("the tolerance is at least 0, not {}", settled.tolerance),
        ),
        (
            settled.relative_tolerance >= 0.0,
            format!(
                "the relative tolerance is at least 0, not {}",
                settled.relative_tolerance
            ),
        ),
        (
            settled.partial_tolerance >= 0.0,
            format!("the partial tolerance is at least 0, not {}", settled.partial_tolerance),
        ),
        (
            settled.step_ratio > 0.0 && settled.step_ratio.is_finite(),
            format!(
                "the ratio of the steps is positive and finite, not {}",
                settled.step_ratio
            ),
        ),
        (
            settled.relaxation > 0.0 && settled.relaxation < 2.0,
            format!("the relaxation is in (0, 2), not {}", settled.relaxation),
        ),
        (
            settled.step_product > 0.0 && settled.step_product < 1.0,
            format!(
                "the product of the steps and L^2 is in (0, 1), not {}",
                settled.step_product
            ),
        ),
        (
            settled.norm_squared > 0.0 && settled.norm_squared.is_finite(),
            format!("L^2 is positive and finite, not {}", settled.norm_squared),
        ),
        (
            radius.is_none_or(|radius| radius >= 0.0),
            format!("the radius of the partial gap is at least 0, not {radius:?}"),
        ),
        (
            settled.partial_tolerance == 0.0 || radius.is_some(),
            "the partial tolerance needs the partial gap's radius".to_owned(),
        ),
        (
            (0.0..f64::INFINITY).contains(&settled.free_radius),
            format!(
                "the radius of the free samples is at least 0 and finite, not {}",
                settled.free_radius
            ),
        ),
    ];
    let wrong: Vec<String> = ranges
        .into_iter()
        .filter(|(holds, _)| !holds)
        .map(|(_, message)| message)
        .collect();
    if wrong.is_empty() {
        Ok(())
    } else {
        Err(Error::Options(wrong.join("; ")))
    }
}

/// `tau` and `sigma`, with `tau / sigma = ratio` and `sigma tau norm_squared = product`.
///
/// # Errors
///
/// [`Error::Options`] unless the ratio and `L^2` are positive and finite, and the
/// product in (0, 1).
pub fn steps(norm_squared: f64, ratio: f64, product: f64) -> Result<(f64, f64), Error> {
    let valid = ratio > 0.0
        && ratio.is_finite()
        && norm_squared > 0.0
        && norm_squared.is_finite()
        && product > 0.0
        && product < 1.0;
    if !valid {
        return Err(Error::Options(format!(
            "a ratio and an L^2 that are positive and finite, and a product in (0, 1), \
             not {ratio}, {norm_squared} and {product}"
        )));
    }
    Ok((
        (ratio * product / norm_squared).sqrt(),
        (product / (ratio * norm_squared)).sqrt(),
    ))
}

/// Where a solver starts, where not at its default (docs/math.md, 5).
///
/// `coefficients` are clipped to their intervals, and are the data term's centres
/// unless given. `w` is TGV's field, and is the gradient of the start's canvas
/// unless given. `p` and TGV's `r` are where the dual starts, projected onto their
/// balls, and are 0 unless given. TV has no `w` and no `r`. The solvers copy them.
#[derive(Debug, Clone, Default)]
pub struct Initial {
    /// The coefficients of every component.
    pub coefficients: Option<Vec<Vec<f64>>>,
    /// TGV's vector field.
    pub w: Option<Vector>,
    /// The dual of the first-order term.
    pub p: Option<Vector>,
    /// The dual of TGV's second-order term.
    pub r: Option<Tensor>,
}

/// A record, as an observer sees it: the iteration, its gap per sample (what the
/// tolerance is compared with; where samples are free, the partial gap of 6.6), the
/// primal and dual values, and the point.
#[derive(Debug)]
pub struct Record<'a> {
    /// The iteration.
    pub iteration: u64,
    /// The gap per sample.
    pub gap: f64,
    /// The primal value.
    pub primal: f64,
    /// The dual value.
    pub dual: f64,
    /// The coefficients of every component.
    pub coefficients: &'a [Vec<f64>],
    /// The canvas of every channel.
    pub canvas: &'a [f64],
    /// TGV's vector field.
    pub w: Option<&'a Vector>,
}

/// What a solver calls with each record after the first; it stops where this breaks.
pub type Observer<'o> = dyn FnMut(&Record<'_>) -> ControlFlow<()> + 'o;

fn checked_field(values: &[f64], size: usize, name: &str) -> Result<(), Error> {
    if values.len() != size {
        return Err(Error::Options(format!(
            "{name} is a field of {size} entries, not {}",
            values.len()
        )));
    }
    if !values.iter().all(|value| value.is_finite()) {
        return Err(Error::Options(format!("the field {name} to start from is finite")));
    }
    Ok(())
}

fn vector_start(given: Option<&Vector>, size: usize, name: &str) -> Result<Option<Vector>, Error> {
    let Some(field) = given else {
        return Ok(None);
    };
    checked_field(&field.x, size, name)?;
    checked_field(&field.y, size, name)?;
    Ok(Some(field.clone()))
}

fn tensor_start(given: Option<&Tensor>, size: usize) -> Result<Option<Tensor>, Error> {
    let Some(field) = given else {
        return Ok(None);
    };
    checked_field(&field.xx, size, "r")?;
    checked_field(&field.yy, size, "r")?;
    checked_field(&field.xy, size, "r")?;
    Ok(Some(field.clone()))
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

/// `f[j]` for `j < W - 1`, less `f[j - 1]` for `j > 0`: the backward difference of a
/// row, the negative adjoint of the forward one.
#[inline]
fn backward_across(row: &[f64], out: &mut [f64]) {
    let width = row.len();
    if width == 1 {
        out[0] = 0.0;
        return;
    }
    out[0] = row[0];
    for ((entry, &here), &before) in out[1..width - 1].iter_mut().zip(&row[1..]).zip(row) {
        *entry = here - before;
    }
    out[width - 1] = 0.0 - row[width - 2];
}

/// The backward difference down at a row, from the row itself (`None` at the last
/// row) and the one above (`None` at the first).
#[inline]
fn backward_down(own: Option<&[f64]>, before: Option<&[f64]>, out: &mut [f64]) {
    match (own, before) {
        (Some(own), Some(before)) => {
            for ((entry, &here), &above) in out.iter_mut().zip(own).zip(before) {
                *entry = here - above;
            }
        }
        (Some(own), None) => out.copy_from_slice(own),
        (None, Some(before)) => {
            for (entry, &above) in out.iter_mut().zip(before) {
                *entry = 0.0 - above;
            }
        }
        (None, None) => out.fill(0.0),
    }
}

/// The forward difference across a row, 0 at its end.
#[inline]
fn forward_across(row: &[f64], out: &mut [f64]) {
    let width = row.len();
    for ((entry, &here), &next) in out.iter_mut().zip(row).zip(&row[1..]) {
        *entry = next - here;
    }
    out[width - 1] = 0.0;
}

/// The forward difference down at a row, from the row and the one below (`None` at
/// the last row).
#[inline]
fn forward_down(own: &[f64], next: Option<&[f64]>, out: &mut [f64]) {
    match next {
        Some(next) => {
            for ((entry, &here), &below) in out.iter_mut().zip(own).zip(next) {
                *entry = below - here;
            }
        }
        None => out.fill(0.0),
    }
}

/// The rows of a plane: row `i` of `height`, and the rows above and below it.
struct Rows<'a> {
    own: &'a [f64],
    above: Option<&'a [f64]>,
    below: Option<&'a [f64]>,
    last: bool,
}

fn rows(plane: &[f64], width: usize, height: usize, i: usize) -> Rows<'_> {
    let start = i * width;
    Rows {
        own: &plane[start..start + width],
        above: (i > 0).then(|| &plane[start - width..start]),
        below: (i + 1 < height).then(|| &plane[start + width..start + 2 * width]),
        last: i + 1 == height,
    }
}

/// Scratch rows for the differences of one row.
struct Scratch {
    first: Vec<f64>,
    second: Vec<f64>,
    third: Vec<f64>,
}

impl Scratch {
    fn new(width: usize) -> Self {
        Self {
            first: vec![0.0; width],
            second: vec![0.0; width],
            third: vec![0.0; width],
        }
    }
}

/// `x + tau (gamma div p)`, channel by channel, into `out`.
fn descend(frame: &Frame, gammas: &[f64], tau: f64, x: &[f64], p: &Vector, out: &mut [f64], scratch: &mut Scratch) {
    let (height, width, plane) = (frame.height(), frame.width(), frame.plane());
    for (channel, &gamma) in gammas.iter().enumerate() {
        let range = channel * plane..(channel + 1) * plane;
        let (p_x, p_y) = (&p.x[range.clone()], &p.y[range.clone()]);
        let (source, target) = (&x[range.clone()], &mut out[range]);
        for i in 0..height {
            let down = rows(p_y, width, height, i);
            backward_across(&p_x[i * width..(i + 1) * width], &mut scratch.first);
            backward_down((!down.last).then_some(down.own), down.above, &mut scratch.second);
            let start = i * width;
            for (((entry, &value), &across), &downward) in target[start..start + width]
                .iter_mut()
                .zip(&source[start..start + width])
                .zip(&scratch.first)
                .zip(&scratch.second)
            {
                *entry = multiply_add(tau, gamma * (across + downward), value);
            }
        }
    }
}

/// `2 a - b`, entry by entry, into `out`: the extrapolated point.
fn extrapolate(a: &[f64], b: &[f64], out: &mut [f64]) {
    for ((entry, &new), &old) in out.iter_mut().zip(a).zip(b) {
        *entry = multiply_add(2.0, new, -old);
    }
}

/// `current <- rho tilde + (1 - rho) current`.
fn relax(rho: f64, tilde: &[f64], current: &mut [f64]) {
    let rest = 1.0 - rho;
    for (value, &new) in current.iter_mut().zip(tilde) {
        *value = multiply_add(rho, new, rest * *value);
    }
}

/// `p + sigma (gamma grad bar)`, channel by channel, into `out` (TV's ascent).
fn ascend_tv(
    frame: &Frame,
    gammas: &[f64],
    sigma: f64,
    bar: &[f64],
    p: &Vector,
    out: &mut Vector,
    scratch: &mut Scratch,
) {
    let (height, width, plane) = (frame.height(), frame.width(), frame.plane());
    for (channel, &gamma) in gammas.iter().enumerate() {
        let range = channel * plane..(channel + 1) * plane;
        let own = &bar[range.clone()];
        for i in 0..height {
            let lines = rows(own, width, height, i);
            forward_across(lines.own, &mut scratch.first);
            forward_down(lines.own, lines.below, &mut scratch.second);
            let start = channel * plane + i * width;
            let span = start..start + width;
            for ((entry, &value), &across) in out.x[span.clone()]
                .iter_mut()
                .zip(&p.x[span.clone()])
                .zip(&scratch.first)
            {
                *entry = multiply_add(sigma, gamma * across, value);
            }
            for ((entry, &value), &down) in out.y[span.clone()].iter_mut().zip(&p.y[span]).zip(&scratch.second) {
                *entry = multiply_add(sigma, gamma * down, value);
            }
        }
    }
}

/// Minimizes the TV model of a frame (docs/math.md, 4.4 and 5).
///
/// `initial` is where to start, the data term's centres and `p = 0` unless given.
/// `observer` is called with each record after the first; it can stop the solver.
///
/// Each iteration takes the proximal steps from the current point `(x, p)` to
/// `(x~, p~)` and moves the current point to `rho (x~, p~) + (1 - rho) (x, p)`. What
/// is recorded, observed and returned are `(x~, p~)`: `x~` is an output of the
/// proximal map of `G`, in the constraint set, and `p~` is in the dual ball, even
/// where the current point is not.
///
/// # Errors
///
/// [`Error::Options`] for options, weights or a start that the method cannot take.
pub fn solve_tv(
    frame: &Frame,
    weights: &Tv,
    options: &Options,
    initial: Option<&Initial>,
    mut observer: Option<&mut Observer<'_>>,
) -> Result<FrameResult, Error> {
    let settled = plan(options, Weights::Tv(weights))?;
    let (tau, sigma) = steps(settled.norm_squared, settled.step_ratio, settled.step_product)?;
    let rho = settled.relaxation;
    #[expect(clippy::float_cmp, reason = "a relaxation of exactly 1 takes the new point as it is")]
    let unrelaxed = rho == 1.0;
    let gammas = frame.channel_weights(weights.channel_weights.as_deref())?;
    if initial.is_some_and(|first| first.w.is_some() || first.r.is_some()) {
        return Err(Error::Options("TV has no field w or r to start from".into()));
    }
    let size = frame.samples();
    let start = frames::start(frame, initial.and_then(|first| first.coefficients.as_deref()))?;
    let (mut coefficients, mut canvas) = (start.coefficients, start.canvas);
    let mut x = canvas.clone();
    let mut p =
        vector_start(initial.and_then(|first| first.p.as_ref()), size, "p")?.unwrap_or_else(|| Vector::zeros(size));
    frames::project_vectors(frame, &mut p, weights.alpha, weights.coupled);
    let mut p_out = p.clone();
    let radius = settled.free_radius;
    let samples = nearest_from_usize(size);
    let steps = frames::steps(frame, tau);
    let mut work = vec![0.0; size];
    let mut scratch = Scratch::new(frame.width());
    let mut recorder = Recorder::new();
    let (primal, dual) = frames::tv_values(frame, weights, &coefficients, &canvas, &p_out, radius)?;
    recorder.record(0, primal, dual, f64::NAN, f64::NAN);
    let mut stop = Stop::Iterations;
    let mut iteration = 0;
    while iteration < settled.iterations {
        recorder.resume();
        iteration += 1;
        descend(frame, &gammas, tau, &x, &p, &mut work, &mut scratch);
        frames::prox(frame, &steps, &work, &mut coefficients, &mut canvas);
        extrapolate(&canvas, &x, &mut work);
        ascend_tv(frame, &gammas, sigma, &work, &p, &mut p_out, &mut scratch);
        frames::project_vectors(frame, &mut p_out, weights.alpha, weights.coupled);
        if unrelaxed {
            x.copy_from_slice(&canvas);
            p.clone_from(&p_out);
        } else {
            relax(rho, &canvas, &mut x);
            relax(rho, &p_out.x, &mut p.x);
            relax(rho, &p_out.y, &mut p.y);
        }
        recorder.pause();
        if due(iteration, &settled) {
            let (primal, dual) = frames::tv_values(frame, weights, &coefficients, &canvas, &p_out, radius)?;
            recorder.record(iteration, primal, dual, f64::NAN, f64::NAN);
            let (gap, met) = stops(&settled, samples, primal, dual, f64::NAN);
            if met {
                stop = Stop::Converged;
            }
            if let Some(observe) = observer.as_deref_mut() {
                let record = Record {
                    iteration,
                    gap,
                    primal,
                    dual,
                    coefficients: &coefficients,
                    canvas: &canvas,
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
    Ok(FrameResult {
        primal: Primal {
            coefficients,
            canvas,
            w: None,
        },
        dual: Some(Dual { p: p_out, r: None }),
        iterations: iteration,
        stop,
        history: recorder.finish(),
    })
}

/// `w + tau (gamma (p + div2 r))`, channel by channel, into `out` (TGV's step of `w`).
#[expect(
    clippy::too_many_arguments,
    reason = "the step reads the field, both duals and the scratch rows"
)]
fn advance_w(
    frame: &Frame,
    gammas: &[f64],
    tau: f64,
    w: &Vector,
    p: &Vector,
    r: &Tensor,
    out: &mut Vector,
    scratch: &mut Scratch,
) {
    let (height, width, plane) = (frame.height(), frame.width(), frame.plane());
    for (channel, &gamma) in gammas.iter().enumerate() {
        let range = channel * plane..(channel + 1) * plane;
        let (xx, yy, xy) = (&r.xx[range.clone()], &r.yy[range.clone()], &r.xy[range]);
        for i in 0..height {
            let start = i * width;
            // div2 r = (d/dx r11 + d/dy r12, d/dx r12 + d/dy r22), by forward differences.
            forward_across(&xx[start..start + width], &mut scratch.first);
            let lines = rows(xy, width, height, i);
            forward_down(lines.own, lines.below, &mut scratch.second);
            let span = channel * plane + start..channel * plane + start + width;
            for (((entry, &value), &dual), (&across, &down)) in out.x[span.clone()]
                .iter_mut()
                .zip(&w.x[span.clone()])
                .zip(&p.x[span.clone()])
                .zip(scratch.first.iter().zip(&scratch.second))
            {
                *entry = multiply_add(tau, gamma * (dual + (across + down)), value);
            }
            forward_across(lines.own, &mut scratch.first);
            let lines = rows(yy, width, height, i);
            forward_down(lines.own, lines.below, &mut scratch.second);
            for (((entry, &value), &dual), (&across, &down)) in out.y[span.clone()]
                .iter_mut()
                .zip(&w.y[span.clone()])
                .zip(&p.y[span])
                .zip(scratch.first.iter().zip(&scratch.second))
            {
                *entry = multiply_add(tau, gamma * (dual + (across + down)), value);
            }
        }
    }
}

/// `p + sigma (gamma (grad bar - bar_w))`, channel by channel, into `out` (TGV's
/// ascent of `p`).
#[expect(
    clippy::too_many_arguments,
    reason = "the step reads the extrapolated point, the dual and the scratch rows"
)]
fn ascend_p(
    frame: &Frame,
    gammas: &[f64],
    sigma: f64,
    bar: &[f64],
    bar_w: &Vector,
    p: &Vector,
    out: &mut Vector,
    scratch: &mut Scratch,
) {
    let (height, width, plane) = (frame.height(), frame.width(), frame.plane());
    for (channel, &gamma) in gammas.iter().enumerate() {
        let range = channel * plane..(channel + 1) * plane;
        let own = &bar[range];
        for i in 0..height {
            let lines = rows(own, width, height, i);
            forward_across(lines.own, &mut scratch.first);
            forward_down(lines.own, lines.below, &mut scratch.second);
            let start = channel * plane + i * width;
            let span = start..start + width;
            for (((entry, &value), &across), &field) in out.x[span.clone()]
                .iter_mut()
                .zip(&p.x[span.clone()])
                .zip(&scratch.first)
                .zip(&bar_w.x[span.clone()])
            {
                *entry = multiply_add(sigma, gamma * (across - field), value);
            }
            for (((entry, &value), &down), &field) in out.y[span.clone()]
                .iter_mut()
                .zip(&p.y[span.clone()])
                .zip(&scratch.second)
                .zip(&bar_w.y[span])
            {
                *entry = multiply_add(sigma, gamma * (down - field), value);
            }
        }
    }
}

/// `r + sigma (gamma E bar_w)`, channel by channel, into `out` (TGV's ascent of `r`).
fn ascend_r(
    frame: &Frame,
    gammas: &[f64],
    sigma: f64,
    bar_w: &Vector,
    r: &Tensor,
    out: &mut Tensor,
    scratch: &mut Scratch,
) {
    let (height, width, plane) = (frame.height(), frame.width(), frame.plane());
    for (channel, &gamma) in gammas.iter().enumerate() {
        let range = channel * plane..(channel + 1) * plane;
        let (w_x, w_y) = (&bar_w.x[range.clone()], &bar_w.y[range]);
        for i in 0..height {
            let start = i * width;
            let span = channel * plane + start..channel * plane + start + width;
            // E w = (d_x w1, d_y w2, (d_y w1 + d_x w2) / 2), by backward differences.
            backward_across(&w_x[start..start + width], &mut scratch.first);
            for ((entry, &value), &across) in out.xx[span.clone()]
                .iter_mut()
                .zip(&r.xx[span.clone()])
                .zip(&scratch.first)
            {
                *entry = multiply_add(sigma, gamma * across, value);
            }
            let lines = rows(w_y, width, height, i);
            backward_down((!lines.last).then_some(lines.own), lines.above, &mut scratch.second);
            for ((entry, &value), &down) in out.yy[span.clone()]
                .iter_mut()
                .zip(&r.yy[span.clone()])
                .zip(&scratch.second)
            {
                *entry = multiply_add(sigma, gamma * down, value);
            }
            let lines = rows(w_x, width, height, i);
            backward_down((!lines.last).then_some(lines.own), lines.above, &mut scratch.second);
            backward_across(&w_y[start..start + width], &mut scratch.third);
            for (((entry, &value), &down), &across) in out.xy[span.clone()]
                .iter_mut()
                .zip(&r.xy[span])
                .zip(&scratch.second)
                .zip(&scratch.third)
            {
                *entry = multiply_add(sigma, gamma * f64::midpoint(down, across), value);
            }
        }
    }
}

/// The values of a record of TGV: the primal and dual values, the scaling of the
/// feasible dual, and the partial gap where its radius is given.
fn tgv_record(
    frame: &Frame,
    weights: &Tgv,
    settled: &Plan,
    point: (&[Vec<f64>], &[f64], &Vector),
    dual: (&Vector, &Tensor),
) -> Result<(f64, f64, f64, f64), Error> {
    let (coefficients, canvas, w) = point;
    let (p, r) = dual;
    let radius = settled.free_radius;
    let (primal, dual_value, theta) = frames::tgv_values(frame, weights, coefficients, canvas, w, r, radius)?;
    let partial = match settled.partial_radius {
        Some(partial_radius) => {
            let residual = frames::tgv_residual(frame, weights, p, r)?;
            let bound = frames::dual_bound(frame, weights.channel_weights.as_deref(), p, radius)?;
            primal + bound + partial_radius * residual
        }
        None => f64::NAN,
    };
    Ok((primal, dual_value, theta, partial))
}

/// The iterates of TGV: the current point and dual, the outputs of their proximal
/// steps, and room for the extrapolated field and point.
struct TgvIterates {
    x: Vec<f64>,
    canvas: Vec<f64>,
    coefficients: Vec<Vec<f64>>,
    w: Vector,
    w_out: Vector,
    bar_w: Vector,
    p: Vector,
    p_out: Vector,
    r: Tensor,
    r_out: Tensor,
    work: Vec<f64>,
}

impl TgvIterates {
    /// The start: the data term's centres, `w` the gradient of their canvas, and `p`
    /// and `r` 0, where `initial` does not give them; `p` and `r` projected onto their
    /// balls.
    fn new(frame: &Frame, weights: &Tgv, initial: Option<&Initial>) -> Result<Self, Error> {
        let size = frame.samples();
        let start = frames::start(frame, initial.and_then(|first| first.coefficients.as_deref()))?;
        let w = match vector_start(initial.and_then(|first| first.w.as_ref()), size, "w")? {
            Some(w) => w,
            None => frames::gradient(frame, &start.canvas),
        };
        let mut p =
            vector_start(initial.and_then(|first| first.p.as_ref()), size, "p")?.unwrap_or_else(|| Vector::zeros(size));
        frames::project_vectors(frame, &mut p, weights.alpha1, weights.coupled);
        let mut r =
            tensor_start(initial.and_then(|first| first.r.as_ref()), size)?.unwrap_or_else(|| Tensor::zeros(size));
        frames::project_tensors(frame, &mut r, weights.alpha0, weights.coupled);
        Ok(Self {
            x: start.canvas.clone(),
            canvas: start.canvas,
            coefficients: start.coefficients,
            w_out: w.clone(),
            w,
            bar_w: Vector::zeros(size),
            p_out: p.clone(),
            p,
            r_out: r.clone(),
            r,
            work: vec![0.0; size],
        })
    }

    /// The proximal steps from the current point: `x~`, `w~`, `p~` and `r~`.
    #[expect(
        clippy::too_many_arguments,
        reason = "a step takes the model, both steps and the scratch rows"
    )]
    fn step(
        &mut self,
        frame: &Frame,
        weights: &Tgv,
        gammas: &[f64],
        proximal: &[Step],
        tau: f64,
        sigma: f64,
        scratch: &mut Scratch,
    ) {
        descend(frame, gammas, tau, &self.x, &self.p, &mut self.work, scratch);
        frames::prox(frame, proximal, &self.work, &mut self.coefficients, &mut self.canvas);
        advance_w(frame, gammas, tau, &self.w, &self.p, &self.r, &mut self.w_out, scratch);
        extrapolate(&self.canvas, &self.x, &mut self.work);
        extrapolate(&self.w_out.x, &self.w.x, &mut self.bar_w.x);
        extrapolate(&self.w_out.y, &self.w.y, &mut self.bar_w.y);
        ascend_p(
            frame,
            gammas,
            sigma,
            &self.work,
            &self.bar_w,
            &self.p,
            &mut self.p_out,
            scratch,
        );
        frames::project_vectors(frame, &mut self.p_out, weights.alpha1, weights.coupled);
        ascend_r(frame, gammas, sigma, &self.bar_w, &self.r, &mut self.r_out, scratch);
        frames::project_tensors(frame, &mut self.r_out, weights.alpha0, weights.coupled);
    }

    /// The current point moved to `rho` times the outputs plus `1 - rho` times itself.
    fn relax(&mut self, rho: f64, unrelaxed: bool) {
        if unrelaxed {
            self.x.copy_from_slice(&self.canvas);
            self.w.clone_from(&self.w_out);
            self.p.clone_from(&self.p_out);
            self.r.clone_from(&self.r_out);
        } else {
            relax(rho, &self.canvas, &mut self.x);
            relax(rho, &self.w_out.x, &mut self.w.x);
            relax(rho, &self.w_out.y, &mut self.w.y);
            relax(rho, &self.p_out.x, &mut self.p.x);
            relax(rho, &self.p_out.y, &mut self.p.y);
            relax(rho, &self.r_out.xx, &mut self.r.xx);
            relax(rho, &self.r_out.yy, &mut self.r.yy);
            relax(rho, &self.r_out.xy, &mut self.r.xy);
        }
    }

    /// The values of a record at the outputs of the proximal steps.
    fn record(&self, frame: &Frame, weights: &Tgv, settled: &Plan) -> Result<(f64, f64, f64, f64), Error> {
        tgv_record(
            frame,
            weights,
            settled,
            (&self.coefficients, &self.canvas, &self.w_out),
            (&self.p_out, &self.r_out),
        )
    }
}

/// Minimizes the TGV model of a frame (docs/math.md, 4.4 and 5).
///
/// `initial` is where to start: the data term's centres, `w` the gradient of their
/// canvas, and `p` and `r` 0, unless given. The gap is that of the feasible dual
/// (6.2), partial where samples are free (6.6). The steps are relaxed as
/// [`solve_tv`]'s are, and what is recorded, observed and returned are the outputs
/// of the proximal steps.
///
/// # Errors
///
/// [`Error::Options`] for options, weights or a start that the method cannot take.
pub fn solve_tgv(
    frame: &Frame,
    weights: &Tgv,
    options: &Options,
    initial: Option<&Initial>,
    mut observer: Option<&mut Observer<'_>>,
) -> Result<FrameResult, Error> {
    let settled = plan(options, Weights::Tgv(weights))?;
    let (tau, sigma) = steps(settled.norm_squared, settled.step_ratio, settled.step_product)?;
    let rho = settled.relaxation;
    #[expect(clippy::float_cmp, reason = "a relaxation of exactly 1 takes the new point as it is")]
    let unrelaxed = rho == 1.0;
    let gammas = frame.channel_weights(weights.channel_weights.as_deref())?;
    let mut iterates = TgvIterates::new(frame, weights, initial)?;
    let samples = nearest_from_usize(frame.samples());
    let proximal = frames::steps(frame, tau);
    let mut scratch = Scratch::new(frame.width());
    let mut recorder = Recorder::new();
    let (primal, dual, theta, partial) = iterates.record(frame, weights, &settled)?;
    recorder.record(0, primal, dual, theta, partial);
    let mut stop = Stop::Iterations;
    let mut iteration = 0;
    while iteration < settled.iterations {
        recorder.resume();
        iteration += 1;
        iterates.step(frame, weights, &gammas, &proximal, tau, sigma, &mut scratch);
        iterates.relax(rho, unrelaxed);
        recorder.pause();
        if due(iteration, &settled) {
            let (primal, dual, theta, partial) = iterates.record(frame, weights, &settled)?;
            recorder.record(iteration, primal, dual, theta, partial);
            let (gap, met) = stops(&settled, samples, primal, dual, partial);
            if met {
                stop = Stop::Converged;
            }
            if let Some(observe) = observer.as_deref_mut() {
                let record = Record {
                    iteration,
                    gap,
                    primal,
                    dual,
                    coefficients: &iterates.coefficients,
                    canvas: &iterates.canvas,
                    w: Some(&iterates.w_out),
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
    Ok(FrameResult {
        primal: Primal {
            coefficients: iterates.coefficients,
            canvas: iterates.canvas,
            w: Some(iterates.w_out),
        },
        dual: Some(Dual {
            p: iterates.p_out,
            r: Some(iterates.r_out),
        }),
        iterations: iteration,
        stop,
        history: recorder.finish(),
    })
}
