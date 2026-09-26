// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! The model of one component: the quantization constraint set and the data term
//! (docs/math.md, 1.2 and 4.1), and the weights of the regularizers (4.2 to 4.4).
//!
//! A [`Problem`] holds what the file says about the component: the interval of every
//! coefficient, and the centres and the weights of the data term `G`. `G` is
//! separable in the coefficients, and so are its proximal map and its conjugate,
//! which are computed coefficient by coefficient.
//!
//! Coefficients are those of the canvas of samples that are not level-shifted:
//! JPEG's level shift, `D(x - 128)`, moves only the DC coefficient of a block, by
//! `8 x 128 = 1024` exactly, and that is added to the DC intervals and centres, so
//! that nothing is added to or taken from the samples (docs/math.md, 1.1).

use crate::dct::{BLOCK, BLOCK_SIZE};
use crate::error::Error;
use crate::exact::Sum;
use crate::kernels;
use crate::laplace;

/// What the level shift of 128 adds to the DC coefficient of a block: 128 times 8,
/// exactly.
pub const LEVEL_SHIFT_DC: f64 = 1024.0;

/// The centres of the data term.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Centres {
    /// The MMSE centres of the Laplace model (docs/math.md, 2.2).
    #[default]
    Mmse,
    /// The middles of the intervals, `q Q`, which are exact.
    Midpoint,
}

/// The options of `G`: the weights and the centres of the data term, and the
/// intervals' slack (docs/math.md, 1.2 and 4.1).
///
/// `mu` weights the data term; where it is `None`, it is `mu_scale` times the mean of
/// the component's 64 steps to the power `mu_power`, which follows the
/// quantization. `slack` widens every interval by that many steps on each side;
/// `slack_cost` is what leaving the file's own interval costs, within the slack,
/// per step (0: nothing). The weight of an AC coefficient is `mu / Q^power`, and
/// that of DC `dc_weight` times it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DataTerm {
    /// The weight of the data term, or `None` for the rule of the quantization.
    pub mu: Option<f64>,
    /// The scale of the rule of `mu`.
    pub mu_scale: f64,
    /// The power of the mean step in the rule of `mu`.
    pub mu_power: f64,
    /// The slack of the intervals, in steps on each side.
    pub slack: f64,
    /// The weight of DC, a factor of that of AC.
    pub dc_weight: f64,
    /// The centres of the data term.
    pub centres: Centres,
    /// The power of the steps in the weights.
    pub power: f64,
    /// What leaving the file's own interval costs, per step.
    pub slack_cost: f64,
}

impl Default for DataTerm {
    fn default() -> Self {
        Self {
            mu: Some(1e-3),
            mu_scale: 1.0,
            mu_power: 1.0,
            slack: 0.0,
            dc_weight: 0.0,
            centres: Centres::Mmse,
            power: 2.0,
            slack_cost: 0.0,
        }
    }
}

/// The weight of total variation (docs/math.md, 4.2), and how it takes several
/// channels (4.4).
///
/// `channel_weights` are the `gamma` of the channels' differences, 1 for each where
/// `None`; `coupled` takes the channels together, pixel by pixel, or each on its
/// own. Neither matters to one channel.
#[derive(Debug, Clone, PartialEq)]
pub struct Tv {
    /// The weight of the total variation.
    pub alpha: f64,
    /// The weight of each channel's differences.
    pub channel_weights: Option<Vec<f64>>,
    /// Whether the channels are taken together.
    pub coupled: bool,
}

impl Default for Tv {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            channel_weights: None,
            coupled: true,
        }
    }
}

/// The weights of second-order total generalized variation (docs/math.md, 4.3 and
/// 4.4): `alpha1` weights `||grad x - w||`, and `alpha0` `||E w||`. The channels are
/// taken as [`Tv`] takes them.
#[derive(Debug, Clone, PartialEq)]
pub struct Tgv {
    /// The weight of the first-order part.
    pub alpha1: f64,
    /// The weight of the second-order part.
    pub alpha0: f64,
    /// The weight of each channel's differences.
    pub channel_weights: Option<Vec<f64>>,
    /// Whether the channels are taken together.
    pub coupled: bool,
}

impl Default for Tgv {
    fn default() -> Self {
        Self {
            alpha1: 1.0,
            alpha0: 2.0,
            channel_weights: None,
            coupled: true,
        }
    }
}

/// The ends of the file's own intervals, and what the data term charges for each
/// unit a coefficient lies beyond them, where the slack has a cost.
#[derive(Debug, Clone)]
pub struct Cost {
    inner_lower: Vec<f64>,
    inner_upper: Vec<f64>,
    costs: [f64; BLOCK_SIZE],
}

impl Cost {
    /// The lower ends of the file's own intervals.
    #[must_use]
    pub fn inner_lower(&self) -> &[f64] {
        &self.inner_lower
    }

    /// The upper ends of the file's own intervals.
    #[must_use]
    pub fn inner_upper(&self) -> &[f64] {
        &self.inner_upper
    }

    /// The cost of each frequency, per unit of a coefficient: `beta / Q`.
    #[must_use]
    pub fn costs(&self) -> &[f64; BLOCK_SIZE] {
        &self.costs
    }
}

/// A component to reconstruct: `rows x columns` blocks of coefficients, 64 to a
/// block in natural order, each with its interval, the centre of the data term,
/// and the weight and the step of its frequency.
#[derive(Debug, Clone)]
pub struct Problem {
    rows: usize,
    columns: usize,
    levels: Vec<i16>,
    shrunk: [f64; BLOCK_SIZE],
    slack: f64,
    lower: Vec<f64>,
    upper: Vec<f64>,
    centres: Vec<f64>,
    steps: [f64; BLOCK_SIZE],
    weights: [f64; BLOCK_SIZE],
    cost: Option<Cost>,
}

/// What a step of `tau` does to each frequency in the proximal map: `tau w`,
/// `1 / (1 + tau w)`, and `tau` times the cost over `1 + tau w`.
#[derive(Debug, Clone)]
pub struct Step {
    scaled: [f64; BLOCK_SIZE],
    inverse: [f64; BLOCK_SIZE],
    shrink: [f64; BLOCK_SIZE],
}

impl Step {
    /// `tau w`, `1 / (1 + tau w)` and `tau` times the cost over `1 + tau w`, for each
    /// frequency.
    #[must_use]
    pub fn factors(&self) -> (&[f64; BLOCK_SIZE], &[f64; BLOCK_SIZE], &[f64; BLOCK_SIZE]) {
        (&self.scaled, &self.inverse, &self.shrink)
    }
}

/// `mu` of the data term: `data.mu`, or where it is `None`, `mu_scale` times the
/// mean step to the power `mu_power` (docs/math.md, 4.1).
///
/// The mean of the 64 steps is exact: their sum is an integer below 2^22, and
/// dividing it by 64 is exact. The power is 1 by default, which rounds nothing
/// more.
#[must_use]
pub fn rule_mu(data: &DataTerm, table: &[u16; BLOCK_SIZE]) -> f64 {
    if let Some(mu) = data.mu {
        return mu;
    }
    let sum: u32 = table.iter().map(|&step| u32::from(step)).sum();
    let mean = f64::from(sum) / 64.0;
    #[expect(clippy::float_cmp, reason = "the power 1 is taken exactly as it is given")]
    let plain = data.mu_power == 1.0;
    data.mu_scale * if plain { mean } else { mean.powf(data.mu_power) }
}

fn finite_at_least_zero(value: f64) -> bool {
    (0.0..f64::INFINITY).contains(&value)
}

/// `Q^power`: exact for the powers 1 and 2 of steps of 16 bits.
fn powers(step: f64, power: f64) -> f64 {
    #[expect(clippy::float_cmp, reason = "the powers 1 and 2 are taken exactly as they are given")]
    let (square, plain) = (power == 2.0, power == 1.0);
    if square {
        step * step
    } else if plain {
        step
    } else {
        step.powf(power)
    }
}

impl Problem {
    /// The problem of a component with these quantized levels and quantization table.
    ///
    /// `levels` are `rows x columns` blocks of 64, and `table` the steps. `data` are
    /// the options of `G`. `scale` is the Laplace scale of each frequency for the
    /// MMSE centres, which [`laplace::scales`] estimates unless it is given. The ends
    /// are `((q - 1/2) - slack) Q` and `((q + 1/2) + slack) Q`, in that order, and
    /// those of DC and its centre have [`LEVEL_SHIFT_DC`] added; without slack every
    /// one of them is exact, and so is every middle.
    ///
    /// # Errors
    ///
    /// [`Error::Options`] for options out of their ranges, levels that are not
    /// `rows x columns` blocks, or a step below 1.
    pub fn new(
        levels: &[i16],
        rows: usize,
        columns: usize,
        table: &[u16; BLOCK_SIZE],
        data: &DataTerm,
        scale: Option<&[f64; BLOCK_SIZE]>,
    ) -> Result<Self, Error> {
        check(levels, rows, columns, table, data, scale)?;
        let mu = rule_mu(data, table);
        let mut steps = [0.0; BLOCK_SIZE];
        let mut weights = [0.0; BLOCK_SIZE];
        for ((step, weight), &entry) in steps.iter_mut().zip(&mut weights).zip(table) {
            *step = f64::from(entry);
            *weight = mu / powers(*step, data.power);
        }
        weights[0] *= data.dc_weight;
        let slack = data.slack;
        // The shrinkage of each frequency: the MMSE centres', or none for the middles
        // q Q, which are exact: |q| <= 2^15 and Q < 2^16.
        let shrunk = match data.centres {
            Centres::Mmse => {
                let estimated;
                let scale = if let Some(scale) = scale {
                    scale
                } else {
                    estimated = laplace::scales(levels, table);
                    &estimated
                };
                laplace::shrinkages(table, scale)
            }
            Centres::Midpoint => [0.0; BLOCK_SIZE],
        };
        let mut shifts = [0.0; BLOCK_SIZE];
        shifts[0] = LEVEL_SHIFT_DC;
        // A value of each coefficient from its level and its frequency's constants, block
        // by block.
        let each = |value: &dyn Fn(f64, usize) -> f64| -> Vec<f64> {
            let mut values = vec![0.0; levels.len()];
            for (block, own) in values
                .as_chunks_mut::<BLOCK_SIZE>()
                .0
                .iter_mut()
                .zip(levels.as_chunks::<BLOCK_SIZE>().0)
            {
                for (frequency, (value_of, &level)) in block.iter_mut().zip(own).enumerate() {
                    *value_of = value(f64::from(level), frequency);
                }
            }
            values
        };
        let ends = |half: f64, widen: f64| {
            each(&|level, frequency| kernels::interval_end(level, half, widen, steps[frequency], shifts[frequency]))
        };
        let mut problem = Self {
            rows,
            columns,
            levels: levels.to_vec(),
            shrunk,
            slack,
            lower: ends(-0.5, -slack),
            upper: ends(0.5, slack),
            centres: each(&|level, frequency| {
                kernels::centre(level, steps[frequency], shrunk[frequency], shifts[frequency])
            }),
            steps,
            weights,
            cost: None,
        };
        if slack > 0.0 && data.slack_cost > 0.0 {
            let inner_lower = ends(-0.5, 0.0);
            let inner_upper = ends(0.5, 0.0);
            let mut costs = [0.0; BLOCK_SIZE];
            for (cost, &step) in costs.iter_mut().zip(&steps) {
                *cost = data.slack_cost / step;
            }
            problem.cost = Some(Cost {
                inner_lower,
                inner_upper,
                costs,
            });
        }
        Ok(problem)
    }

    /// Block rows.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The quantized levels, 64 to a block in natural order.
    #[must_use]
    pub fn levels(&self) -> &[i16] {
        &self.levels
    }

    /// How far towards 0 the centre of each frequency lies, in steps: the MMSE
    /// shrinkage of its scale (docs/math.md, 2.2), 0 for DC and for the middles.
    #[must_use]
    pub fn shrunk(&self) -> &[f64; BLOCK_SIZE] {
        &self.shrunk
    }

    /// The slack of the intervals, in steps on each side.
    #[must_use]
    pub fn slack(&self) -> f64 {
        self.slack
    }

    /// Blocks across.
    #[must_use]
    pub fn columns(&self) -> usize {
        self.columns
    }

    /// The rows of samples of the component's canvas.
    #[must_use]
    pub fn height(&self) -> usize {
        self.rows * BLOCK
    }

    /// The samples across the component's canvas.
    #[must_use]
    pub fn width(&self) -> usize {
        self.columns * BLOCK
    }

    /// The samples of the component's canvas.
    #[must_use]
    pub fn samples(&self) -> usize {
        self.lower.len()
    }

    /// The lower ends of the intervals, widened by the slack.
    #[must_use]
    pub fn lower(&self) -> &[f64] {
        &self.lower
    }

    /// The upper ends of the intervals, widened by the slack.
    #[must_use]
    pub fn upper(&self) -> &[f64] {
        &self.upper
    }

    /// The centres of the data term.
    #[must_use]
    pub fn centres(&self) -> &[f64] {
        &self.centres
    }

    /// The quantization step of each frequency.
    #[must_use]
    pub fn steps(&self) -> &[f64; BLOCK_SIZE] {
        &self.steps
    }

    /// The weight of each frequency in the data term, `mu` included.
    #[must_use]
    pub fn weights(&self) -> &[f64; BLOCK_SIZE] {
        &self.weights
    }

    /// The file's own intervals and the costs of leaving them, where the slack has
    /// a cost.
    #[must_use]
    pub fn cost(&self) -> Option<&Cost> {
        self.cost.as_ref()
    }

    /// Coefficient `index`, clipped to its interval.
    #[inline]
    #[must_use]
    pub fn clip_one(&self, index: usize, value: f64) -> f64 {
        kernels::at_most(kernels::at_least(value, self.lower[index]), self.upper[index])
    }

    /// The coefficients, clipped to their intervals.
    pub fn clip(&self, coefficients: &mut [f64]) {
        for (index, value) in coefficients.iter_mut().enumerate() {
            *value = self.clip_one(index, *value);
        }
    }

    /// How far each coefficient lies outside its interval, in steps (0 inside).
    #[must_use]
    pub fn excess(&self, coefficients: &[f64]) -> Vec<f64> {
        coefficients
            .iter()
            .enumerate()
            .map(|(index, &value)| {
                let below = (self.lower[index] - value).max(0.0);
                let above = (value - self.upper[index]).max(0.0);
                (below + above) / self.steps[index % BLOCK_SIZE]
            })
            .collect()
    }

    /// How far coefficient `index` lies beyond the file's own interval (0 within it),
    /// in coefficient units; without a cost, beyond the interval.
    #[inline]
    #[must_use]
    pub fn beyond_one(&self, index: usize, value: f64) -> f64 {
        let (lower, upper) = match &self.cost {
            Some(cost) => (cost.inner_lower[index], cost.inner_upper[index]),
            None => (self.lower[index], self.upper[index]),
        };
        kernels::beyond(value, lower, upper)
    }

    /// The factors of the proximal map with the step `tau` (docs/math.md, 4.1).
    #[must_use]
    pub fn step(&self, tau: f64) -> Step {
        let mut step = Step {
            scaled: [0.0; BLOCK_SIZE],
            inverse: [0.0; BLOCK_SIZE],
            shrink: [0.0; BLOCK_SIZE],
        };
        for (frequency, &weight) in self.weights.iter().enumerate() {
            let scaled = tau * weight;
            let inverse = 1.0 / (1.0 + scaled);
            step.scaled[frequency] = scaled;
            step.inverse[frequency] = inverse;
            if let Some(cost) = &self.cost {
                step.shrink[frequency] = tau * cost.costs[frequency] * inverse;
            }
        }
        step
    }

    /// The coefficients of `prox_{tau G}(v)` of a block, from those of `v`, in place
    /// (docs/math.md, 4.1): `(e + tau w centre) / (1 + tau w)`, clipped to the
    /// interval. Where the slack has a cost, the minimizer of the quadratic that lies
    /// beyond the file's own interval moves back towards it by `tau` times the cost
    /// over `1 + tau w`, but not past its end, before it is clipped.
    #[inline]
    pub fn prox_block(&self, step: &Step, block: usize, coefficients: &mut [f64]) {
        let start = block * BLOCK_SIZE;
        let range = start..start + BLOCK_SIZE;
        let centres = &self.centres[range.clone()];
        let lower = &self.lower[range.clone()];
        let upper = &self.upper[range.clone()];
        match &self.cost {
            None => {
                for (frequency, value) in coefficients.iter_mut().enumerate() {
                    let moved = kernels::proximal(
                        *value,
                        step.scaled[frequency],
                        step.inverse[frequency],
                        centres[frequency],
                    );
                    *value = kernels::at_most(kernels::at_least(moved, lower[frequency]), upper[frequency]);
                }
            }
            Some(cost) => {
                let inner_lower = &cost.inner_lower[range.clone()];
                let inner_upper = &cost.inner_upper[range];
                for (frequency, value) in coefficients.iter_mut().enumerate() {
                    let moved = kernels::proximal(
                        *value,
                        step.scaled[frequency],
                        step.inverse[frequency],
                        centres[frequency],
                    );
                    let kept = kernels::charged(
                        moved,
                        step.shrink[frequency],
                        inner_lower[frequency],
                        inner_upper[frequency],
                    );
                    *value = kernels::at_most(kernels::at_least(kept, lower[frequency]), upper[frequency]);
                }
            }
        }
    }

    /// The coefficients of `prox_{tau G}(v)`, from those of `v`, in place.
    pub fn prox(&self, tau: f64, coefficients: &mut [f64]) {
        let step = self.step(tau);
        for (block, values) in coefficients.as_chunks_mut::<BLOCK_SIZE>().0.iter_mut().enumerate() {
            self.prox_block(&step, block, values);
        }
    }

    /// `G` at coefficients within their intervals: half the sum of `w (c - centre)^2`,
    /// and the slack's cost. The terms of a block row are added in lanes
    /// ([`crate::exact::lanes`]), and the block rows' sums in a [`Sum`].
    #[must_use]
    pub fn data_term(&self, coefficients: &[f64]) -> f64 {
        let row = self.columns * BLOCK_SIZE;
        let mut terms = vec![0.0; row];
        let mut squares = Sum::new();
        for (values, centres) in coefficients.chunks_exact(row).zip(self.centres.chunks_exact(row)) {
            // Block by block, each frequency with its weight.
            for ((terms, values), centres) in terms
                .as_chunks_mut::<BLOCK_SIZE>()
                .0
                .iter_mut()
                .zip(values.as_chunks::<BLOCK_SIZE>().0)
                .zip(centres.as_chunks::<BLOCK_SIZE>().0)
            {
                for (((term, &value), &centre), &weight) in terms.iter_mut().zip(values).zip(centres).zip(&self.weights)
                {
                    *term = kernels::squared_term(value, weight, centre);
                }
            }
            squares.add(crate::exact::lanes(&terms));
        }
        let mut value = 0.5 * squares.total();
        if let Some(cost) = &self.cost {
            let mut charged = Sum::new();
            for ((values, lower), upper) in coefficients
                .chunks_exact(row)
                .zip(cost.inner_lower.chunks_exact(row))
                .zip(cost.inner_upper.chunks_exact(row))
            {
                for (((terms, values), lower), upper) in terms
                    .as_chunks_mut::<BLOCK_SIZE>()
                    .0
                    .iter_mut()
                    .zip(values.as_chunks::<BLOCK_SIZE>().0)
                    .zip(lower.as_chunks::<BLOCK_SIZE>().0)
                    .zip(upper.as_chunks::<BLOCK_SIZE>().0)
                {
                    for ((term, ((&coefficient, &low), &high)), &charge) in terms
                        .iter_mut()
                        .zip(values.iter().zip(lower).zip(upper))
                        .zip(&cost.costs)
                    {
                        *term = charge * kernels::beyond(coefficient, low, high);
                    }
                }
                charged.add(crate::exact::lanes(&terms));
            }
            value += charged.total();
        }
        value
    }

    /// The term of coefficient `index` of `G*`, given its coefficient `s` of `D xi`
    /// (docs/math.md, 4.1).
    #[inline]
    #[must_use]
    pub fn conjugate_one(&self, index: usize, s: f64) -> f64 {
        let frequency = index % BLOCK_SIZE;
        let weight = self.weights[frequency];
        let ends = (self.lower[index], self.upper[index]);
        let centre = self.centres[index];
        match &self.cost {
            None => kernels::conjugate_term::<false>(s, weight, ends, centre, 0.0, ends),
            Some(cost) => {
                let inner = (cost.inner_lower[index], cost.inner_upper[index]);
                kernels::conjugate_term::<true>(s, weight, ends, centre, cost.costs[frequency], inner)
            }
        }
    }

    /// `G*(xi)`, given the coefficients `s = D xi` (docs/math.md, 4.1). The terms of a
    /// block row are added in lanes ([`crate::exact::lanes`]), and the block rows' sums
    /// in a [`Sum`].
    #[must_use]
    pub fn conjugate(&self, coefficients: &[f64]) -> f64 {
        let row = self.columns * BLOCK_SIZE;
        let mut terms = vec![0.0; row];
        let mut total = Sum::new();
        for (block_row, values) in coefficients.chunks_exact(row).enumerate() {
            let span = block_row * row..(block_row + 1) * row;
            let (lower, upper, centres) = (
                &self.lower[span.clone()],
                &self.upper[span.clone()],
                &self.centres[span.clone()],
            );
            // Block by block, each frequency with its constants.
            let own = terms
                .as_chunks_mut::<BLOCK_SIZE>()
                .0
                .iter_mut()
                .zip(values.as_chunks::<BLOCK_SIZE>().0)
                .zip(
                    lower
                        .as_chunks::<BLOCK_SIZE>()
                        .0
                        .iter()
                        .zip(upper.as_chunks::<BLOCK_SIZE>().0),
                )
                .zip(centres.as_chunks::<BLOCK_SIZE>().0);
            match &self.cost {
                None => {
                    for (((terms, values), (lower, upper)), centres) in own {
                        for (k, term) in terms.iter_mut().enumerate() {
                            let ends = (lower[k], upper[k]);
                            *term = kernels::conjugate_term::<false>(
                                values[k],
                                self.weights[k],
                                ends,
                                centres[k],
                                0.0,
                                ends,
                            );
                        }
                    }
                }
                Some(cost) => {
                    let (inner_lower, inner_upper) = (&cost.inner_lower[span.clone()], &cost.inner_upper[span]);
                    let inner = inner_lower
                        .as_chunks::<BLOCK_SIZE>()
                        .0
                        .iter()
                        .zip(inner_upper.as_chunks::<BLOCK_SIZE>().0);
                    for ((((terms, values), (lower, upper)), centres), (inner_lower, inner_upper)) in own.zip(inner) {
                        for (k, term) in terms.iter_mut().enumerate() {
                            *term = kernels::conjugate_term::<true>(
                                values[k],
                                self.weights[k],
                                (lower[k], upper[k]),
                                centres[k],
                                cost.costs[k],
                                (inner_lower[k], inner_upper[k]),
                            );
                        }
                    }
                }
            }
            total.add(crate::exact::lanes(&terms));
        }
        total.total()
    }
}

fn check(
    levels: &[i16],
    rows: usize,
    columns: usize,
    table: &[u16; BLOCK_SIZE],
    data: &DataTerm,
    scale: Option<&[f64; BLOCK_SIZE]>,
) -> Result<(), Error> {
    let refuse = |message: String| Err(Error::Options(message));
    if !(finite_at_least_zero(data.mu_scale) && data.mu_power.is_finite()) {
        return refuse(format!(
            "the scale of mu is at least 0 and finite, and its power finite, not {} and {}",
            data.mu_scale, data.mu_power
        ));
    }
    let mu = rule_mu(data, table);
    if !(finite_at_least_zero(mu) && finite_at_least_zero(data.slack) && finite_at_least_zero(data.dc_weight)) {
        return refuse(format!(
            "mu, slack and the weight of DC are at least 0 and finite, not {mu}, {} and {}",
            data.slack, data.dc_weight
        ));
    }
    if !finite_at_least_zero(data.power) {
        return refuse(format!(
            "the power of the steps in the weights is at least 0 and finite, not {}",
            data.power
        ));
    }
    if !finite_at_least_zero(data.slack_cost) {
        return refuse(format!(
            "the cost of the slack is at least 0 and finite, not {}",
            data.slack_cost
        ));
    }
    if data.centres == Centres::Midpoint && scale.is_some() {
        return refuse("a Laplace scale is for the MMSE centres, not the middles".into());
    }
    if levels.len() != rows * columns * BLOCK_SIZE {
        return refuse(format!(
            "levels of {rows} x {columns} blocks of 64, not {} levels",
            levels.len()
        ));
    }
    if table.contains(&0) {
        return refuse("a quantization step is at least 1".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn levels_with(entries: &[(usize, i16)]) -> Vec<i16> {
        let mut levels = vec![0i16; BLOCK_SIZE];
        for &(index, level) in entries {
            levels[index] = level;
        }
        levels
    }

    #[test]
    #[expect(clippy::float_cmp, reason = "exact values are compared to the last bit")]
    fn the_intervals_and_the_weights() {
        let levels = levels_with(&[(0, 3), (2 * 8 + 1, -2)]);
        let data = DataTerm {
            mu: Some(0.5),
            slack: 0.25,
            ..DataTerm::default()
        };
        let problem = Problem::new(&levels, 1, 1, &[10; BLOCK_SIZE], &data, None).expect("a problem");
        // Every one of these is exact in binary floating point. The DC interval carries the
        // level shift, 8 x 128.
        assert_eq!(problem.lower()[0], 22.5 + 1024.0);
        assert_eq!(problem.upper()[0], 37.5 + 1024.0);
        assert_eq!(problem.centres()[0], 30.0 + 1024.0);
        assert_eq!(problem.lower()[17], -27.5);
        assert_eq!(problem.upper()[17], -12.5);
        assert_eq!(problem.weights()[0], 0.0);
        assert_eq!(problem.weights()[17], 0.5 / 100.0);
        assert_eq!((problem.height(), problem.width(), problem.samples()), (8, 8, 64));
    }

    #[test]
    #[expect(clippy::float_cmp, reason = "exact values are compared to the last bit")]
    fn without_slack_the_intervals_are_exact() {
        // (q - 1/2) Q is exact for integers of the sizes a file holds, and so are the ends.
        let mut levels = Vec::new();
        for level in [-2048i16, -1, 0, 1, 2047] {
            levels.extend(std::iter::repeat_n(level, BLOCK_SIZE));
        }
        let data = DataTerm {
            mu: Some(1.0),
            ..DataTerm::default()
        };
        let problem = Problem::new(&levels, 1, 5, &[255; BLOCK_SIZE], &data, None).expect("a problem");
        for (index, &level) in levels.iter().enumerate() {
            let doubled = (2 * i64::from(level) - 1) * 255 + if index % BLOCK_SIZE == 0 { 2048 } else { 0 };
            let exact = f64::from(i32::try_from(doubled).expect("fits")) / 2.0;
            assert_eq!(problem.lower()[index], exact, "{index}");
        }
    }

    #[test]
    fn the_options_of_g_out_of_their_ranges_are_refused() {
        let levels = vec![0i16; BLOCK_SIZE];
        let cases = [
            (
                DataTerm {
                    mu: Some(-1.0),
                    ..DataTerm::default()
                },
                "at least 0",
            ),
            (
                DataTerm {
                    mu: Some(f64::INFINITY),
                    ..DataTerm::default()
                },
                "finite",
            ),
            (
                DataTerm {
                    slack: f64::NAN,
                    ..DataTerm::default()
                },
                "at least 0",
            ),
            (
                DataTerm {
                    dc_weight: -1.0,
                    ..DataTerm::default()
                },
                "at least 0",
            ),
            (
                DataTerm {
                    power: -1.0,
                    ..DataTerm::default()
                },
                "power",
            ),
            (
                DataTerm {
                    slack_cost: f64::NAN,
                    ..DataTerm::default()
                },
                "cost of the slack",
            ),
            (
                DataTerm {
                    mu: None,
                    mu_scale: -1.0,
                    ..DataTerm::default()
                },
                "scale of mu",
            ),
            (
                DataTerm {
                    mu: None,
                    mu_power: f64::INFINITY,
                    ..DataTerm::default()
                },
                "power",
            ),
        ];
        for (data, fragment) in cases {
            let error = Problem::new(&levels, 1, 1, &[1; BLOCK_SIZE], &data, None).expect_err("refused");
            assert!(error.to_string().contains(fragment), "{error}");
        }
        let midpoint = DataTerm {
            centres: Centres::Midpoint,
            ..DataTerm::default()
        };
        let error =
            Problem::new(&levels, 1, 1, &[1; BLOCK_SIZE], &midpoint, Some(&[1.0; BLOCK_SIZE])).expect_err("refused");
        assert!(error.to_string().contains("Laplace scale"), "{error}");
        let error = Problem::new(&levels, 1, 2, &[1; BLOCK_SIZE], &DataTerm::default(), None).expect_err("refused");
        assert!(error.to_string().contains("blocks"), "{error}");
        let mut table = [1; BLOCK_SIZE];
        table[5] = 0;
        let error = Problem::new(&levels, 1, 1, &table, &DataTerm::default(), None).expect_err("refused");
        assert!(error.to_string().contains("at least 1"), "{error}");
    }

    #[test]
    #[expect(clippy::float_cmp, reason = "exact values are compared to the last bit")]
    fn the_weight_of_dc_the_power_and_the_middles() {
        let levels = levels_with(&[(0, 3), (17, -2)]);
        let table = [10; BLOCK_SIZE];
        let data = DataTerm {
            mu: Some(0.5),
            dc_weight: 2.0,
            centres: Centres::Midpoint,
            ..DataTerm::default()
        };
        let problem = Problem::new(&levels, 1, 1, &table, &data, None).expect("a problem");
        // mu / Q^2 rounds once, and doubling it is exact; the middles q Q are exact, and
        // DC's has the level shift.
        assert_eq!(problem.weights()[0], 2.0 * (0.5 / 100.0));
        assert_eq!(problem.weights()[17], 0.5 / 100.0);
        assert_eq!(problem.centres()[0], 30.0 + 1024.0);
        assert_eq!(problem.centres()[17], -20.0);
        assert_eq!(problem.centres()[9], 0.0);
        for (power, expected) in [(1.0, 0.5 / 10.0), (3.0, 0.5 / 1000.0)] {
            let data = DataTerm {
                mu: Some(0.5),
                power,
                ..DataTerm::default()
            };
            let problem = Problem::new(&levels, 1, 1, &table, &data, None).expect("a problem");
            assert_eq!(problem.weights()[17], expected);
        }
        // The MMSE centre of a level that is not 0 lies nearer 0 than the middle.
        let data = DataTerm {
            mu: Some(0.5),
            ..DataTerm::default()
        };
        let mmse = Problem::new(&levels, 1, 1, &table, &data, None).expect("a problem");
        assert!(-20.0 < mmse.centres()[17] && mmse.centres()[17] < -15.0);
        assert_eq!(mmse.weights()[0], 0.0);
    }

    #[test]
    #[expect(clippy::float_cmp, reason = "exact values are compared to the last bit")]
    fn mu_follows_the_steps_where_it_is_not_given() {
        let mut table = [10; BLOCK_SIZE];
        table[0] = 74;
        let rule = |mu: Option<f64>, mu_power: f64| DataTerm {
            mu,
            mu_scale: 5.0,
            mu_power,
            ..DataTerm::default()
        };
        assert_eq!(rule_mu(&rule(None, 1.0), &table), 5.0 * 11.0);
        assert_eq!(rule_mu(&rule(None, 2.0), &table), 5.0 * 121.0);
        assert_eq!(rule_mu(&rule(Some(2.5), 1.0), &table), 2.5);
    }

    #[test]
    #[expect(clippy::float_cmp, reason = "exact values are compared to the last bit")]
    fn a_slack_costs_only_where_it_is_given() {
        let levels = levels_with(&[(0, 3), (17, -2)]);
        let table = [10; BLOCK_SIZE];
        for (slack, slack_cost) in [(0.5, 0.0), (0.0, 2.0), (0.0, 0.0)] {
            let data = DataTerm {
                slack,
                slack_cost,
                ..DataTerm::default()
            };
            let problem = Problem::new(&levels, 1, 1, &table, &data, None).expect("a problem");
            assert!(problem.cost().is_none());
        }
        let data = DataTerm {
            slack: 0.5,
            slack_cost: 2.0,
            ..DataTerm::default()
        };
        let problem = Problem::new(&levels, 1, 1, &table, &data, None).expect("a problem");
        let cost = problem.cost().expect("a cost");
        // The file's own intervals, exactly, and the cost per unit of a coefficient, 2 / Q.
        assert_eq!(cost.inner_lower()[0], 25.0 + 1024.0);
        assert_eq!(cost.inner_upper()[17], -15.0);
        assert_eq!(problem.lower()[17], -30.0);
        assert_eq!(cost.costs()[17], 2.0 / 10.0);
    }
}
