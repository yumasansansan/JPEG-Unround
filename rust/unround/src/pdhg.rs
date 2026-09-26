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

use crate::error::Error;
use crate::exact::nearest_from_usize;
use crate::frames::{self, Dual, Frame, Primal, Tensor, Vector};
use crate::model::{Tgv, Tv};
use crate::planar::Layout;
use crate::records::Values;
use crate::results::{FrameResult, Recorder, Stop};
use crate::sweep::{self, TgvOutputs, TgvPoint, TgvSweep, TvOutputs, TvPoint, TvSweep};

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
/// primal and dual values, and the point, on a canvas of `height x width` samples
/// for each channel.
#[derive(Debug)]
pub struct Record<'a> {
    /// Rows of the canvas.
    pub height: usize,
    /// Columns of the canvas.
    pub width: usize,
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

/// TV's point and outputs at the start, in the planes: the start's canvas and
/// coefficients, and `p` projected where it is given.
fn tv_start(
    frame: &Frame,
    layout: &Layout,
    weights: &Tv,
    initial: Option<&Initial>,
) -> Result<(TvPoint, TvOutputs), Error> {
    let (canvas, coefficients) = sweep::start(frame, layout, initial.and_then(|first| first.coefficients.as_deref()))?;
    let given = initial.and_then(|first| first.p.as_ref());
    let (px, py, p_given) = dual_start(frame, layout, given, weights.alpha, weights.coupled)?;
    let point = TvPoint {
        x: canvas.clone(),
        px: copy(&px, p_given),
        py: copy(&py, p_given),
    };
    // The outputs are the start until a record writes over them.
    Ok((
        point,
        TvOutputs {
            canvas,
            coefficients,
            px,
            py,
        },
    ))
}

/// TGV's point and outputs at the start, in the planes: the start's canvas and
/// coefficients, `w` given or the gradient of the canvas, and `p` and `r` projected
/// where they are given.
fn tgv_start(
    frame: &Frame,
    layout: &Layout,
    weights: &Tgv,
    initial: Option<&Initial>,
) -> Result<(TgvPoint, TgvOutputs), Error> {
    let size = frame.samples();
    let (canvas, coefficients) = sweep::start(frame, layout, initial.and_then(|first| first.coefficients.as_deref()))?;
    let (wx, wy) = match vector_start(initial.and_then(|first| first.w.as_ref()), size, "w")? {
        Some(given) => (layout.to_planes(&given.x), layout.to_planes(&given.y)),
        None => sweep::gradient(layout, &canvas),
    };
    let given = initial.and_then(|first| first.p.as_ref());
    let (px, py, p_given) = dual_start(frame, layout, given, weights.alpha1, weights.coupled)?;
    let (rxx, ryy, rxy, r_given) = match tensor_start(initial.and_then(|first| first.r.as_ref()), size)? {
        Some(mut given) => {
            frames::project_tensors(frame, &mut given, weights.alpha0, weights.coupled);
            (
                layout.to_planes(&given.xx),
                layout.to_planes(&given.yy),
                layout.to_planes(&given.xy),
                true,
            )
        }
        None => (vec![0.0; size], vec![0.0; size], vec![0.0; size], false),
    };
    let point = TgvPoint {
        x: canvas.clone(),
        wx: wx.clone(),
        wy: wy.clone(),
        px: copy(&px, p_given),
        py: copy(&py, p_given),
        rxx: copy(&rxx, r_given),
        ryy: copy(&ryy, r_given),
        rxy: copy(&rxy, r_given),
    };
    // The outputs are the start until a record writes over them.
    let outputs = TgvOutputs {
        first: TvOutputs {
            canvas,
            coefficients,
            px,
            py,
        },
        wx,
        wy,
        rxx,
        ryy,
        rxy,
    };
    Ok((point, outputs))
}

/// The dual field `p` to start from in the planes: the one given, projected onto the
/// balls of the radius, or 0; and whether it was given.
fn dual_start(
    frame: &Frame,
    layout: &Layout,
    given: Option<&Vector>,
    radius: f64,
    coupled: bool,
) -> Result<(Vec<f64>, Vec<f64>, bool), Error> {
    let size = frame.samples();
    Ok(match vector_start(given, size, "p")? {
        Some(mut given) => {
            frames::project_vectors(frame, &mut given, radius, coupled);
            (layout.to_planes(&given.x), layout.to_planes(&given.y), true)
        }
        None => (vec![0.0; size], vec![0.0; size], false),
    })
}

/// Each component's coefficients from the planes into natural order.
fn write_coefficients(layout: &Layout, planar: &[Vec<f64>], natural: &mut [Vec<f64>]) {
    for ((part, own), target) in layout.parts.iter().zip(planar).zip(natural) {
        layout.write_values_natural(part, own, target);
    }
}

/// TV's outputs in the natural layout: the canvas and `p` in the memory of the
/// point, which is no longer needed, and the coefficients into `coefficients`.
fn tv_natural(
    layout: &Layout,
    point: TvPoint,
    outputs: &TvOutputs,
    coefficients: &mut [Vec<f64>],
) -> (Vec<f64>, Vector) {
    let TvPoint { x: mut canvas, px, py } = point;
    let mut p = Vector { x: px, y: py };
    layout.write_natural(&outputs.canvas, &mut canvas);
    layout.write_natural(&outputs.px, &mut p.x);
    layout.write_natural(&outputs.py, &mut p.y);
    write_coefficients(layout, &outputs.coefficients, coefficients);
    (canvas, p)
}

/// TGV's outputs in the natural layout: the canvas, `w`, `p` and `r` in the memory of
/// the point, which is no longer needed, and the coefficients into `coefficients`.
fn tgv_natural(
    layout: &Layout,
    point: TgvPoint,
    outputs: &TgvOutputs,
    coefficients: &mut [Vec<f64>],
) -> (Vec<f64>, Vector, Vector, Tensor) {
    let TgvPoint {
        x: mut canvas,
        wx,
        wy,
        px,
        py,
        rxx,
        ryy,
        rxy,
    } = point;
    let mut w = Vector { x: wx, y: wy };
    let mut p = Vector { x: px, y: py };
    let mut r = Tensor {
        xx: rxx,
        yy: ryy,
        xy: rxy,
    };
    layout.write_natural(&outputs.first.canvas, &mut canvas);
    write_coefficients(layout, &outputs.first.coefficients, coefficients);
    for (planar, natural) in [
        (&outputs.wx, &mut w.x),
        (&outputs.wy, &mut w.y),
        (&outputs.first.px, &mut p.x),
        (&outputs.first.py, &mut p.y),
        (&outputs.rxx, &mut r.xx),
        (&outputs.ryy, &mut r.yy),
        (&outputs.rxy, &mut r.xy),
    ] {
        layout.write_natural(planar, natural);
    }
    (canvas, w, p, r)
}

/// A copy of a field of the start where it was given; one made anew where it is 0,
/// whose memory is written first by the iterations.
fn copy(field: &[f64], given: bool) -> Vec<f64> {
    if given { field.to_vec() } else { vec![0.0; field.len()] }
}

/// Room for each component's coefficients in natural order.
fn natural_room(layout: &Layout) -> Vec<Vec<f64>> {
    layout
        .parts
        .iter()
        .map(|part| vec![0.0; part.rows * part.columns * crate::dct::BLOCK_SIZE])
        .collect()
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
    let gammas = frame.channel_weights(weights.channel_weights.as_deref())?;
    if initial.is_some_and(|first| first.w.is_some() || first.r.is_some()) {
        return Err(Error::Options("TV has no field w or r to start from".into()));
    }
    let size = frame.samples();
    let layout = Layout::new(frame)?;
    let (mut point, mut outputs) = tv_start(frame, &layout, weights, initial)?;
    let proximal = frames::steps(frame, tau);
    let mut sweep = TvSweep::new(
        frame,
        &layout,
        &proximal,
        &gammas,
        tau,
        sigma,
        weights.alpha,
        weights.coupled,
        settled.relaxation,
    );
    let mut values = Values::new(frame, &layout, &gammas);
    let radius = settled.free_radius;
    let samples = nearest_from_usize(size);
    let mut recorder = Recorder::new();
    let (primal, dual) = values.tv(weights, &outputs, radius);
    recorder.record(0, primal, dual, f64::NAN, f64::NAN);
    let mut stop = Stop::Iterations;
    let mut iteration = 0;
    let mut coefficients = natural_room(&layout);
    let mut seen = if observer.is_some() {
        vec![0.0; size]
    } else {
        Vec::new()
    };
    while iteration < settled.iterations {
        recorder.resume();
        iteration += 1;
        let keep = due(iteration, &settled);
        sweep.iterate(&mut point, keep.then_some(&mut outputs));
        recorder.pause();
        if keep {
            let (primal, dual) = values.tv(weights, &outputs, radius);
            recorder.record(iteration, primal, dual, f64::NAN, f64::NAN);
            let (gap, met) = stops(&settled, samples, primal, dual, f64::NAN);
            if met {
                stop = Stop::Converged;
            }
            if let Some(observe) = observer.as_deref_mut() {
                layout.write_natural(&outputs.canvas, &mut seen);
                write_coefficients(&layout, &outputs.coefficients, &mut coefficients);
                let record = Record {
                    height: frame.height(),
                    width: frame.width(),
                    iteration,
                    gap,
                    primal,
                    dual,
                    coefficients: &coefficients,
                    canvas: &seen,
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
    let (canvas, p_out) = tv_natural(&layout, point, &outputs, &mut coefficients);
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
    let gammas = frame.channel_weights(weights.channel_weights.as_deref())?;
    let size = frame.samples();
    let layout = Layout::new(frame)?;
    let (mut point, mut outputs) = tgv_start(frame, &layout, weights, initial)?;
    let proximal = frames::steps(frame, tau);
    let mut sweep = TgvSweep::new(
        frame,
        &layout,
        &proximal,
        &gammas,
        tau,
        sigma,
        (weights.alpha1, weights.alpha0),
        weights.coupled,
        settled.relaxation,
    );
    let mut values = Values::new(frame, &layout, &gammas);
    let samples = nearest_from_usize(size);
    let mut recorder = Recorder::new();
    let radius = settled.free_radius;
    let record = |values: &mut Values, outputs: &TgvOutputs| {
        let (primal, dual, theta) = values.tgv(weights, outputs, radius);
        let partial = match settled.partial_radius {
            Some(partial_radius) => {
                let (bound, residual) = values.tgv_partial(outputs, radius);
                primal + bound + partial_radius * residual
            }
            None => f64::NAN,
        };
        (primal, dual, theta, partial)
    };
    let (primal, dual, theta, partial) = record(&mut values, &outputs);
    recorder.record(0, primal, dual, theta, partial);
    let mut stop = Stop::Iterations;
    let mut iteration = 0;
    let mut coefficients = natural_room(&layout);
    let (mut seen, mut seen_w) = if observer.is_some() {
        (vec![0.0; size], Vector::zeros(size))
    } else {
        (Vec::new(), Vector::zeros(0))
    };
    while iteration < settled.iterations {
        recorder.resume();
        iteration += 1;
        let keep = due(iteration, &settled);
        sweep.iterate(&mut point, keep.then_some(&mut outputs));
        recorder.pause();
        if keep {
            let (primal, dual, theta, partial) = record(&mut values, &outputs);
            recorder.record(iteration, primal, dual, theta, partial);
            let (gap, met) = stops(&settled, samples, primal, dual, partial);
            if met {
                stop = Stop::Converged;
            }
            if let Some(observe) = observer.as_deref_mut() {
                layout.write_natural(&outputs.first.canvas, &mut seen);
                write_coefficients(&layout, &outputs.first.coefficients, &mut coefficients);
                layout.write_natural(&outputs.wx, &mut seen_w.x);
                layout.write_natural(&outputs.wy, &mut seen_w.y);
                let record = Record {
                    height: frame.height(),
                    width: frame.width(),
                    iteration,
                    gap,
                    primal,
                    dual,
                    coefficients: &coefficients,
                    canvas: &seen,
                    w: Some(&seen_w),
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
    let (canvas, w_out, p_out, r_out) = tgv_natural(&layout, point, &outputs, &mut coefficients);
    Ok(FrameResult {
        primal: Primal {
            coefficients,
            canvas,
            w: Some(w_out),
        },
        dual: Some(Dual {
            p: p_out,
            r: Some(r_out),
        }),
        iterations: iteration,
        stop,
        history: recorder.finish(),
    })
}
