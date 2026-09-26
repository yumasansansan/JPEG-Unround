// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! What the solvers return: the last point, and a history of the objective as
//! they went.

use std::time::{Duration, Instant};

use crate::frames::{Dual, Primal};

/// One entry per record: the iteration, and the values at the iterate after it.
///
/// `seconds` is the solver's own time up to the record, not counting the records.
/// `dual` is the dual value, `-inf` where the solver has none; `scaling` is the
/// `theta` that made TGV's dual feasible (docs/math.md, 6.2), and `partial_gap` the
/// gap of 6.3, NaN where not taken.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct History {
    /// The iteration of each record.
    pub iterations: Vec<u64>,
    /// The solver's own time up to each record.
    pub seconds: Vec<f64>,
    /// The primal value of each record.
    pub primal: Vec<f64>,
    /// The dual value of each record.
    pub dual: Vec<f64>,
    /// TGV's scaling of each record.
    pub scaling: Vec<f64>,
    /// TGV's partial gap of each record.
    pub partial_gap: Vec<f64>,
}

impl History {
    /// The duality gap of each record: an upper bound of the primal value's distance
    /// to its least.
    #[must_use]
    pub fn gap(&self) -> Vec<f64> {
        self.primal
            .iter()
            .zip(&self.dual)
            .map(|(&primal, &dual)| primal - dual)
            .collect()
    }
}

/// How a solver came to stop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Stop {
    /// A tolerance of the options was met.
    Converged,
    /// It took the most iterations allowed.
    Iterations,
    /// The observer asked it to.
    Observer,
    /// The subgradient method's direction vanished.
    Stationary,
}

/// The last point of a solver of a frame, how many iterations it took, why it
/// stopped, and what it recorded.
#[derive(Debug, Clone)]
pub struct FrameResult {
    /// The last point: its coefficients, within their intervals, and its canvas.
    pub primal: Primal,
    /// The last dual point, where the solver has one.
    pub dual: Option<Dual>,
    /// The iterations taken.
    pub iterations: u64,
    /// Why the solver stopped.
    pub stop: Stop,
    /// The records.
    pub history: History,
}

impl FrameResult {
    /// Whether a tolerance stopped the solver.
    #[must_use]
    pub fn converged(&self) -> bool {
        self.stop == Stop::Converged
    }
}

/// Collects a [`History`], and the solver's own time between records.
#[derive(Debug)]
pub(crate) struct Recorder {
    history: History,
    elapsed: Duration,
    started: Instant,
}

impl Recorder {
    pub(crate) fn new() -> Self {
        Self {
            history: History::default(),
            elapsed: Duration::ZERO,
            started: Instant::now(),
        }
    }

    /// Starts the solver's clock.
    pub(crate) fn resume(&mut self) {
        self.started = Instant::now();
    }

    /// Stops the solver's clock, before a record.
    pub(crate) fn pause(&mut self) {
        self.elapsed += self.started.elapsed();
    }

    /// Keeps the values of an iteration.
    pub(crate) fn record(&mut self, iteration: u64, primal: f64, dual: f64, scaling: f64, partial_gap: f64) {
        self.history.iterations.push(iteration);
        self.history.seconds.push(self.elapsed.as_secs_f64());
        self.history.primal.push(primal);
        self.history.dual.push(dual);
        self.history.scaling.push(scaling);
        self.history.partial_gap.push(partial_gap);
    }

    pub(crate) fn finish(self) -> History {
        self.history
    }
}
