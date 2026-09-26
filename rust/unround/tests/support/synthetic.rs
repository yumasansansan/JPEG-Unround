// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! Small problems for the tests: pictures quantized by the DCT here, with fixed
//! seeds, and random numbers from a generator of our own (`SplitMix64`).

use jpeg_unround::colour;
use jpeg_unround::dct::{self, BLOCK, BLOCK_SIZE};
use jpeg_unround::frames::{Channel, Frame};
use jpeg_unround::model::{DataTerm, Problem};

/// The luminance table of the JPEG standard (Annex K), at quality 50, in natural order.
pub const ANNEX_K: [u16; BLOCK_SIZE] = [
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51,
    87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92, 49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
];

/// The chrominance table of the JPEG standard (Annex K), at quality 50, in natural order.
pub const ANNEX_K_CHROMA: [u16; BLOCK_SIZE] = [
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99, 47, 66, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99,
];

/// A deterministic stream of numbers, the `SplitMix64` generator.
#[derive(Debug, Clone)]
pub struct Numbers(u64);

impl Numbers {
    /// A stream from a seed.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// The next 64 bits.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A number in [0, 1), a multiple of 2^-53.
    pub fn unit(&mut self) -> f64 {
        let bits = u32::try_from(self.next_u64() >> 43).expect("21 bits fit");
        let low = u32::try_from((self.next_u64() >> 32) & 0xFFFF_FFFF).expect("32 bits fit");
        (f64::from(bits) * 4_294_967_296.0 + f64::from(low)) / 9_007_199_254_740_992.0
    }

    /// A number in [low, high).
    pub fn uniform(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.unit()
    }

    /// A draw from the normal distribution of this mean and deviation (Box-Muller).
    pub fn normal(&mut self, mean: f64, deviation: f64) -> f64 {
        let first = 1.0 - self.unit(); // in (0, 1]
        let second = self.unit();
        mean + deviation * (-2.0 * first.ln()).sqrt() * (std::f64::consts::TAU * second).cos()
    }

    /// An integer from `low` to `high`, both included.
    pub fn integer(&mut self, low: i64, high: i64) -> i64 {
        let span = u64::try_from(high - low + 1).expect("low <= high");
        low + i64::try_from(self.next_u64() % span).expect("the offset fits")
    }
}

/// A whole double as an integer.
///
/// # Panics
///
/// Where it is not whole, or not within the range of `i64`.
#[must_use]
#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "a whole double within the range of i64 converts exactly"
)]
pub fn whole(value: f64) -> i64 {
    assert!(
        value.fract() == 0.0 && value.abs() < 9.0e18,
        "a whole number within the range of i64, not {value}"
    );
    value as i64
}

fn index_value(value: usize) -> f64 {
    f64::from(u32::try_from(value).expect("an index fits in u32"))
}

/// A canvas of samples about 128: a ramp, a bright disc with a sharp edge, and a
/// little noise.
#[must_use]
pub fn picture(rows: usize, columns: usize, seed: u64) -> Vec<f64> {
    let mut numbers = Numbers::new(seed);
    let centre_y = numbers.uniform(0.3, 0.7) * index_value(rows);
    let centre_x = numbers.uniform(0.3, 0.7) * index_value(columns);
    let radius = 0.25 * index_value(rows.min(columns));
    let mut canvas = Vec::with_capacity(rows * columns);
    for i in 0..rows {
        for j in 0..columns {
            let (y, x) = (index_value(i), index_value(j));
            let ramp = 90.0 * (x / index_value(columns.max(2) - 1)) - 60.0 * (y / index_value(rows.max(2) - 1));
            let inside = (y - centre_y).powi(2) + (x - centre_x).powi(2) < radius * radius;
            let disc = if inside { 70.0 } else { 0.0 };
            canvas.push(ramp + disc + numbers.normal(0.0, 2.0) + 108.0);
        }
    }
    canvas
}

/// An RGB picture, `rows x columns x 3` interleaved, about 128: ramps of their own in
/// each channel, a disc of another colour with a sharp edge, and a little noise.
#[must_use]
pub fn colour_picture(rows: usize, columns: usize, seed: u64) -> Vec<f64> {
    let mut numbers = Numbers::new(seed);
    let centre_y = numbers.uniform(0.3, 0.7) * index_value(rows);
    let centre_x = numbers.uniform(0.3, 0.7) * index_value(columns);
    let radius = 0.25 * index_value(rows.min(columns));
    let mut picture = Vec::with_capacity(3 * rows * columns);
    for i in 0..rows {
        for j in 0..columns {
            let across = index_value(j) / index_value(columns.max(2) - 1);
            let down = index_value(i) / index_value(rows.max(2) - 1);
            let ramps = [
                90.0 * across - 40.0 * down,
                60.0 * down - 30.0 * across,
                50.0 * (across + down),
            ];
            let inside = (index_value(i) - centre_y).powi(2) + (index_value(j) - centre_x).powi(2) < radius * radius;
            let disc = if inside { [70.0, -50.0, 20.0] } else { [0.0; 3] };
            for channel in 0..3 {
                picture.push(ramps[channel] + disc[channel] + numbers.normal(0.0, 2.0) + 100.0);
            }
        }
    }
    picture
}

/// A table times a factor, rounded to the nearest and at least 1.
#[must_use]
pub fn scaled(table: &[u16; BLOCK_SIZE], factor: f64) -> [u16; BLOCK_SIZE] {
    table.map(|step| {
        let value = (f64::from(step) * factor).round_ties_even().max(1.0);
        u16::try_from(whole(value)).expect("a step fits in u16")
    })
}

/// The quantized levels of a canvas of samples, level-shifted and rounded to the
/// nearest (ties to even) as an encoder would.
#[must_use]
pub fn levels(canvas: &[f64], height: usize, width: usize, table: &[u16; BLOCK_SIZE]) -> Vec<i16> {
    let shifted: Vec<f64> = canvas.iter().map(|&sample| sample - 128.0).collect();
    dct::forward(&shifted, height, width)
        .iter()
        .enumerate()
        .map(|(index, &coefficient)| {
            let level = (coefficient / f64::from(table[index % BLOCK_SIZE])).round_ties_even();
            i16::try_from(whole(level)).expect("a level fits in i16")
        })
        .collect()
}

/// A problem of `rows x columns` samples, and the canvas whose quantization it is,
/// with the Annex K table times `scale`.
#[must_use]
pub fn problem(rows: usize, columns: usize, seed: u64, scale: f64, data: &DataTerm) -> (Problem, Vec<f64>) {
    let canvas = picture(rows, columns, seed);
    let table = scaled(&ANNEX_K, scale);
    let levels = levels(&canvas, rows, columns, &table);
    let problem = Problem::new(&levels, rows / BLOCK, columns / BLOCK, &table, data, None).expect("a problem");
    (problem, canvas)
}

/// A component of a file: its levels, its table and its blocks, (down, across).
pub type Component = (Vec<i16>, [u16; BLOCK_SIZE], (usize, usize));

/// A colour file's components as an encoder would quantize them: for each of Y, Cb
/// and Cr, its levels, its table and its blocks (down, across); and the YCbCr canvas
/// (`3 x H x W`) they came from.
///
/// The picture of `rows x columns` is converted to YCbCr and filled out to the canvas
/// by repeating its last row and column; the chroma, with cells of `ratio`
/// `(down, across)`, are the means of their cells. Y takes the luminance table of
/// Annex K times 2, and the chroma the chrominance table times 2. Every level is the
/// nearest to its coefficient, so the canvas itself is in the constraint set.
#[must_use]
pub fn colour_levels(rows: usize, columns: usize, seed: u64, ratio: (usize, usize)) -> (Vec<Component>, Vec<f64>) {
    let (down, across) = ratio;
    let height = BLOCK * down * rows.div_ceil(BLOCK * down);
    let width = BLOCK * across * columns.div_ceil(BLOCK * across);
    let planes = colour::to_ycbcr(&colour_picture(rows, columns, seed), rows, columns).expect("a picture");
    let mut canvas = vec![0.0; 3 * height * width];
    for channel in 0..3 {
        for i in 0..height {
            for j in 0..width {
                let source = channel * rows * columns + i.min(rows - 1) * columns + j.min(columns - 1);
                canvas[channel * height * width + i * width + j] = planes[source];
            }
        }
    }
    let mut components = Vec::new();
    for (index, cells) in [(1, 1), ratio, ratio].into_iter().enumerate() {
        let table = scaled(if index == 0 { &ANNEX_K } else { &ANNEX_K_CHROMA }, 2.0);
        let blocks = (rows.div_ceil(BLOCK * cells.0), columns.div_ceil(BLOCK * cells.1));
        let (own_height, own_width) = (BLOCK * blocks.0, BLOCK * blocks.1);
        let plane = &canvas[index * height * width..(index + 1) * height * width];
        let count = f64::from(u32::try_from(cells.0 * cells.1).expect("fits"));
        let mut shrunk = vec![0.0; own_height * own_width];
        for i in 0..own_height {
            for j in 0..own_width {
                let mut total = 0.0;
                for a in 0..cells.0 {
                    for b in 0..cells.1 {
                        total += plane[(i * cells.0 + a) * width + j * cells.1 + b];
                    }
                }
                shrunk[i * own_width + j] = total / count;
            }
        }
        components.push((levels(&shrunk, own_height, own_width, &table), table, blocks));
    }
    (components, canvas)
}

/// A frame of Y, Cb and Cr of [`colour_levels`], and the YCbCr canvas (`3 x H x W`) it
/// came from.
#[must_use]
pub fn colour_frame(
    rows: usize,
    columns: usize,
    seed: u64,
    ratio: (usize, usize),
    data: &DataTerm,
) -> (Frame, Vec<f64>) {
    let (components, canvas) = colour_levels(rows, columns, seed, ratio);
    let (down, across) = ratio;
    let height = BLOCK * down * rows.div_ceil(BLOCK * down);
    let width = BLOCK * across * columns.div_ceil(BLOCK * across);
    let channels = components
        .into_iter()
        .zip([(1, 1), ratio, ratio])
        .map(|((levels, table, blocks), cells)| {
            let problem = Problem::new(&levels, blocks.0, blocks.1, &table, data, None).expect("a problem");
            Channel::new(problem, cells)
        })
        .collect();
    (Frame::new(channels, height, width).expect("a frame"), canvas)
}
