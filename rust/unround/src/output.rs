// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! Pictures of 8 or 16 bits, made from the binary64 result by an ordered dither of a
//! blue-noise matrix or by rounding (docs/math.md, 10), and the PNG and PNM files that
//! hold them.
//!
//! A PNG file is written by the C layer, with libpng: gray or RGB, with alpha or
//! without, not interlaced, with the file's ICC profile where the profile goes with
//! the picture. PNM holds gray or RGB alone.

use jpegio_sys::{ErrorKind, Png, Samples};

use crate::blue_noise::RANKS;
use crate::error::Error;

/// The bytes of a file, and the first warning of its writing, or an empty string.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Written {
    /// The file.
    pub bytes: Vec<u8>,
    /// Such as an ICC profile that is not written, and why.
    pub warning: String,
}

/// How the samples become levels of 8 or 16 bits (docs/math.md, 10).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Dither {
    /// The ordered dither of the blue-noise matrix, the channels of a pixel by the
    /// matrix shifted apart.
    #[default]
    BlueNoise,
    /// The ordered dither of the blue-noise matrix, every channel of a pixel by the
    /// same threshold.
    BlueNoiseTogether,
    /// Each sample rounded to the nearest level, halves away from 0.
    None,
}

/// The samples of a file of integers: 8 or 16 bits, and how they are made.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Depth {
    /// 16 bits, 257 levels to a unit, rather than 8.
    pub sixteen: bool,
    /// How the levels are made.
    pub dither: Dither,
}

/// The side of the blue-noise matrix.
const SIDE: usize = 64;

/// The shifts of the matrix, in rows and columns, of channels 0 to 3 dithered apart.
const SHIFTS: [(usize, usize); 4] = [(0, 0), (0, 32), (32, 0), (32, 32)];

/// The thresholds of the ranks, `(2R + 1) / 8192`, row by row: 13 bits each, exact.
#[expect(clippy::as_conversions, reason = "a rank below 4096 converts exactly")]
static THRESHOLDS: [f64; SIDE * SIDE] = {
    let mut thresholds = [0.0; SIDE * SIDE];
    let mut index = 0;
    while index < SIDE * SIDE {
        thresholds[index] = (2.0 * RANKS[index] as f64 + 1.0) / 8192.0;
        index += 1;
    }
    thresholds
};

/// The level of `y + error` by the threshold: the level below, or the one above where
/// the fraction reaches `1 - threshold`, clamped to `0..=top`. `y` is the rounding of the
/// exact `y + error` (docs/math.md, 10), on the same side of `floor(y) + 1 - threshold`,
/// a double, unless it is that double itself, where the sign of the error tells. Both
/// differences are exact where it matters: `y - floor(y)` for `y >= 0`, and `1 - t` for
/// a threshold of 13 bits.
#[expect(
    clippy::float_cmp,
    reason = "the fraction is compared with the threshold exactly, as docs/math.md, 10, says"
)]
fn dithered(y: f64, error: f64, threshold: f64, top: f64) -> f64 {
    let whole = y.floor();
    let fraction = y - whole;
    let limit = 1.0 - threshold;
    let above = if fraction > limit || (fraction == limit && error >= 0.0) {
        1.0
    } else {
        0.0
    };
    (whole + above).clamp(0.0, top)
}

/// The nearest level to `y + error`, halves away from 0, clamped to `0..=top`: that of
/// `y`, which is on the same side of every half as `y + error` unless it is the half
/// itself, where the sign of the error tells (docs/math.md, 10).
///
/// The rounding is exact: `f64::round` compares the fraction with 1/2 as it is,
/// where `floor(y + 1/2)` would round `0.49999999999999994` up.
#[expect(
    clippy::float_cmp,
    reason = "the fraction is compared with 1/2 exactly, as docs/math.md, 10, says"
)]
fn rounded(y: f64, error: f64, top: f64) -> f64 {
    let whole = y.floor();
    let level = if y - whole == 0.5 && error < 0.0 {
        whole
    } else {
        y.round()
    };
    level.clamp(0.0, top)
}

#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a whole number from 0 to 255 converts exactly"
)]
fn byte(level: f64) -> u8 {
    level as u8
}

#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a whole number from 0 to 65535 converts exactly"
)]
fn word(level: f64) -> u16 {
    level as u16
}

/// The levels of a picture of rows of `width` pixels of `channels` interleaved
/// samples (1 to 4): each sample times `scale`, exactly, as its rounding and the error
/// of that rounding, made as the dither says, within `0..=top`, and stored by `store`.
/// Sample `c` of pixel `(i, j)` takes the threshold of the matrix at
/// `((i + a) mod 64, (j + b) mod 64)`, `(a, b)` its channel's shift.
fn levels<T: Copy + Default>(
    samples: &[f64],
    width: usize,
    channels: usize,
    scale: f64,
    top: f64,
    dither: Dither,
    store: impl Fn(f64) -> T + Copy,
) -> Vec<T> {
    let mut levels = vec![T::default(); samples.len()];
    if dither == Dither::None {
        for (level, &sample) in levels.iter_mut().zip(samples) {
            let y = sample * scale;
            *level = store(rounded(y, sample.mul_add(scale, -y), top));
        }
        return levels;
    }
    let row_length = width * channels;
    // The thresholds of 64 pixels of a row, interleaved as its samples are: the
    // samples of the row take them 64 pixels at a time.
    let period = SIDE * channels;
    let mut pattern = vec![0.0; period];
    for (row, (line, out)) in samples
        .chunks(row_length)
        .zip(levels.chunks_mut(row_length))
        .enumerate()
    {
        for (index, threshold) in pattern.iter_mut().enumerate() {
            let (column, channel) = (index / channels, index % channels);
            let (down, across) = if dither == Dither::BlueNoise {
                SHIFTS[channel]
            } else {
                (0, 0)
            };
            *threshold = THRESHOLDS[((row + down) % SIDE) * SIDE + (column + across) % SIDE];
        }
        for (piece, out) in line.chunks(period).zip(out.chunks_mut(period)) {
            for ((level, &sample), &threshold) in out.iter_mut().zip(piece).zip(&pattern) {
                let y = sample * scale;
                *level = store(dithered(y, sample.mul_add(scale, -y), threshold, top));
            }
        }
    }
    levels
}

/// 8-bit samples of a picture of `height x width` pixels of `channels` interleaved
/// samples: each sample is its level, made as the dither says and clamped to 0-255.
///
/// # Errors
///
/// [`Error::Options`] for a picture that is empty, of other than 1 or 3 channels, not
/// of that size, or of samples that are not finite.
pub fn eight_bits(
    samples: &[f64],
    height: usize,
    width: usize,
    channels: usize,
    dither: Dither,
) -> Result<Vec<u8>, Error> {
    shape(samples, height, width, channels)?;
    finite(samples)?;
    Ok(levels(samples, width, channels, 1.0, 255.0, dither, byte))
}

/// 16-bit samples of a picture of `height x width` pixels of `channels` interleaved
/// samples: each sample times 257, exactly, so that 255 is 65535, is the level, made as
/// the dither says and clamped to 0-65535.
///
/// # Errors
///
/// As [`eight_bits`].
pub fn sixteen_bits(
    samples: &[f64],
    height: usize,
    width: usize,
    channels: usize,
    dither: Dither,
) -> Result<Vec<u16>, Error> {
    shape(samples, height, width, channels)?;
    finite(samples)?;
    Ok(levels(samples, width, channels, 257.0, 65535.0, dither, word))
}

/// [`Error::Options`] for the first sample that is not finite, if any: whether all
/// are is taken over every sample, without stopping, as one vectorized loop.
fn finite(samples: &[f64]) -> Result<(), Error> {
    if samples.iter().fold(true, |all, sample| all & sample.is_finite()) {
        return Ok(());
    }
    let sample = samples
        .iter()
        .copied()
        .find(|sample| !sample.is_finite())
        .unwrap_or(f64::NAN);
    Err(Error::Options(format!("a sample to quantize is not finite: {sample}")))
}

/// [`Error::Options`] for a picture that is empty, of other than 1 to 4 channels
/// (gray, gray and alpha, RGB, RGB and alpha), or not of `height x width` pixels.
fn shape(samples: &[f64], height: usize, width: usize, channels: usize) -> Result<(), Error> {
    let count = height
        .checked_mul(width)
        .and_then(|pixels| pixels.checked_mul(channels));
    if !(1..=4).contains(&channels) || count != Some(samples.len()) || height == 0 || width == 0 {
        return Err(Error::Options(format!(
            "a picture of {height} x {width} x 1 to 4 samples, not empty, and not of {} samples in {channels} channels",
            samples.len()
        )));
    }
    Ok(())
}

/// The bytes of a PNM file of `height x width` pixels of 1 (`P5`) or 3 (`P6`)
/// interleaved samples, of 8 bits, or of 16 bits, most significant byte first, as
/// [`eight_bits`] and [`sixteen_bits`] make them.
///
/// # Errors
///
/// [`Error::Options`] for a picture that is not of that size, of other than 1 or 3
/// channels, or of samples that are not finite.
pub fn pnm(samples: &[f64], height: usize, width: usize, channels: usize, depth: Depth) -> Result<Vec<u8>, Error> {
    shape(samples, height, width, channels)?;
    if channels == 2 || channels == 4 {
        return Err(Error::Options("PNM holds no alpha: write PNG or TIFF".into()));
    }
    let kind = if channels == 1 { "P5" } else { "P6" };
    let top = if depth.sixteen { 65535 } else { 255 };
    let mut file = format!("{kind}\n{width} {height}\n{top}\n").into_bytes();
    if depth.sixteen {
        for value in sixteen_bits(samples, height, width, channels, depth.dither)? {
            file.extend_from_slice(&value.to_be_bytes());
        }
    } else {
        file.extend(eight_bits(samples, height, width, channels, depth.dither)?);
    }
    Ok(file)
}

/// The bytes of a PNG file of `height x width` pixels of 1 (gray), 2 (gray and
/// alpha), 3 (RGB) or 4 (RGB and alpha) interleaved samples, of 8 bits as [`eight_bits`] makes them or of 16 as
/// [`sixteen_bits`] does; compressed at zlib's level `compression`, 0 to 9 (`None`:
/// zlib's default), which is the ICC profile's too. The ICC profile is written where
/// it goes with the picture ([`jpegio_sys::check_icc`]); otherwise it is left out,
/// and the warning says why.
///
/// # Errors
///
/// [`Error::Options`] for a picture that is not of that size, of other than 1 or 3
/// channels, larger than PNG allows, or of samples that are not finite, and a level
/// above 9; [`Error::Write`] where the C layer could not write the file.
pub fn png(
    samples: &[f64],
    height: usize,
    width: usize,
    channels: usize,
    depth: Depth,
    compression: Option<u8>,
    icc_profile: Option<&[u8]>,
) -> Result<Written, Error> {
    shape(samples, height, width, channels)?;
    // PNG's limit of a side, 2^31 - 1.
    let side = |value: usize| {
        u32::try_from(value)
            .ok()
            .filter(|&side| side <= 0x7FFF_FFFF)
            .ok_or_else(|| Error::Options(format!("a side of {value} pixels is larger than PNG allows")))
    };
    let (rows, columns) = (side(height)?, side(width)?);
    if let Some(level) = compression.filter(|&level| level > 9) {
        return Err(Error::Options(format!("zlib's level is 0 to 9, not {level}")));
    }
    let eight;
    let sixteen_samples;
    let samples = if depth.sixteen {
        sixteen_samples = sixteen_bits(samples, height, width, channels, depth.dither)?;
        Samples::Sixteen(&sixteen_samples)
    } else {
        eight = eight_bits(samples, height, width, channels, depth.dither)?;
        Samples::Eight(&eight)
    };
    let png = Png {
        width: columns,
        height: rows,
        channels: u32::try_from(channels).unwrap_or(0),
        samples,
        compression,
        icc_profile,
    };
    match jpegio_sys::write_png(&png) {
        Ok(written) => Ok(Written {
            bytes: written.bytes,
            warning: written.warning,
        }),
        Err(error) if error.kind() == ErrorKind::Argument => Err(Error::Options(error.message().to_owned())),
        Err(error) => Err(Error::Write(error)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `y`, at least 2^-20 and below 2^20 or 0, as `m / 2^k` exactly.
    #[expect(
        clippy::as_conversions,
        clippy::cast_sign_loss,
        reason = "the fields of a double's bits"
    )]
    fn dyadic(y: f64) -> (i128, u32) {
        let bits = y.to_bits();
        let exponent = ((bits >> 52) & 0x7FF) as i32;
        let fraction = i128::from(bits & ((1 << 52) - 1));
        if exponent == 0 {
            assert_eq!(fraction, 0, "no subnormal numbers here");
            return (0, 0);
        }
        let power = exponent - 1075;
        let mantissa = fraction | (1 << 52);
        if power >= 0 {
            (mantissa << power, 0)
        } else {
            (mantissa, (-power) as u32)
        }
    }

    /// The level of docs/math.md, 10, in integers: `floor(y)`, plus 1 where the
    /// fraction is at least `1 - (2R + 1) / 8192`, clamped to `0..=top`.
    fn exact_level(y: f64, rank: u16, top: u16) -> u16 {
        exact_level_of(y, 1, rank, top)
    }

    /// The same of `y = scale x`, the product taken exactly.
    fn exact_level_of(x: f64, scale: i128, rank: u16, top: u16) -> u16 {
        if x < 0.0 {
            return 0;
        }
        let (numerator, power) = dyadic(x);
        let product = numerator * scale;
        let whole = product >> power;
        let fraction = product - (whole << power);
        let threshold = 2 * i128::from(rank) + 1;
        let above = fraction * 8192 >= (8192 - threshold) << power;
        u16::try_from((whole + i128::from(above)).clamp(0, i128::from(top))).expect("within 0..=top")
    }

    /// The nearest level to `scale x`, the product taken exactly, halves up (away from
    /// 0, for `x >= 0`), clamped to `0..=top`.
    fn exact_rounded(x: f64, scale: i128, top: u16) -> u16 {
        if x < 0.0 {
            return 0;
        }
        let (numerator, power) = dyadic(x);
        let product = numerator * scale;
        let whole = product >> power;
        let fraction = product - (whole << power);
        let above = 2 * fraction >= 1 << power;
        u16::try_from((whole + i128::from(above)).clamp(0, i128::from(top))).expect("within 0..=top")
    }

    /// Samples whose product with 257 rounds to `target`, a double, and the doubles
    /// beside them: the double nearest to `target / 257` and its neighbours.
    fn near_257ths(target: f64) -> [f64; 3] {
        let near = target / 257.0;
        [near.next_down(), near, near.next_up()]
    }

    #[test]
    fn rounding_takes_halves_away_from_zero_and_clamps() {
        let samples = [0.499_999_999_999_999_94, 0.5, 1.5, 2.5, -0.4, -7.0, 254.5, 255.4, 1e300];
        let rounded = eight_bits(&samples, 1, 9, 1, Dither::None).expect("finite samples");
        assert_eq!(rounded, [0, 1, 2, 3, 0, 0, 255, 255, 255]);
        let sixteen = sixteen_bits(&[255.0, 1.0, 0.0], 1, 1, 3, Dither::None).expect("finite samples");
        assert_eq!(sixteen, [65535, 257, 0]);
        assert!(eight_bits(&[f64::NAN], 1, 1, 1, Dither::None).is_err());
        assert!(eight_bits(&[f64::INFINITY], 1, 1, 1, Dither::BlueNoise).is_err());
    }

    #[test]
    fn the_matrix_holds_every_rank_once_and_its_thresholds_exactly() {
        let mut seen = [false; SIDE * SIDE];
        for (&rank, &threshold) in RANKS.iter().zip(&THRESHOLDS) {
            let index = usize::from(rank);
            assert!(!seen[index], "rank {rank} twice");
            seen[index] = true;
            // (2R + 1) / 8192 and 1 - t, each exact: the products with 8192 are whole.
            assert_eq!(threshold * 8192.0, f64::from(2 * rank + 1));
            assert_eq!((1.0 - threshold) * 8192.0, f64::from(8191 - 2 * rank));
        }
    }

    #[test]
    fn a_dithered_level_is_the_exact_one() {
        // Samples of every kind of fraction: whole, halves, the thresholds' own ends and
        // their neighbours, and others at random.
        let mut values = vec![0.0, 1.0, 0.5, 254.999_999_999_999_97, 255.0, 300.0, 65534.6];
        for rank in [0u16, 1, 2047, 2048, 4094, 4095] {
            let end = 1.0 - (2.0 * f64::from(rank) + 1.0) / 8192.0;
            for whole in [0.0, 7.0, 254.0] {
                let at = whole + end;
                values.extend([at, at.next_down(), at.next_up()]);
            }
        }
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        for _ in 0..2000 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            values.push(f64::from(u32::try_from(state >> 46).expect("18 bits")) / 1024.0 + 1e-6);
        }
        for &value in &values {
            for (index, &rank) in RANKS.iter().enumerate().step_by(97) {
                let threshold = THRESHOLDS[index];
                let eight = f64::from(exact_level(value, rank, 255));
                assert_eq!(dithered(value, 0.0, threshold, 255.0), eight, "{value}, {rank}");
                let sixteen = f64::from(exact_level(value, rank, 65535));
                assert_eq!(dithered(value, 0.0, threshold, 65535.0), sixteen, "{value}, {rank}");
            }
        }
        // Below 0 either level clamps to 0.
        for value in [-1e-300, -0.3, -1.0, -1e9] {
            for threshold in [THRESHOLDS[0], 1.0 - 1.0 / 8192.0] {
                assert_eq!(dithered(value, 0.0, threshold, 255.0), 0.0);
            }
        }
    }

    #[test]
    fn over_the_matrix_a_level_keeps_its_mean() {
        // A constant fraction f goes up at floor(4096 f + 1/2) of the 4096 thresholds.
        for (whole, eighths) in [
            (3.0, 0u32),
            (3.0, 1),
            (100.0, 8191),
            (100.0, 4096),
            (0.0, 5),
            (200.0, 6000),
        ] {
            // f = eighths / 8192, so that floor(4096 f + 1/2) = (eighths + 1) / 2.
            let y = whole + f64::from(eighths) / 8192.0;
            let up = THRESHOLDS
                .iter()
                .filter(|&&threshold| dithered(y, 0.0, threshold, 255.0) > whole)
                .count();
            assert_eq!(up, usize::try_from(eighths.div_ceil(2)).expect("small"), "{y}");
        }
    }

    #[test]
    fn the_channels_take_the_matrix_shifted_apart_or_alike() {
        // A flat picture of 70 x 70 pixels of 3 channels, each sample a quarter above 9:
        // the levels of channel c at (i, j) are channel 0's at the place its shift moves to.
        let (height, width) = (70, 70);
        let samples = vec![9.25; height * width * 3];
        let apart = eight_bits(&samples, height, width, 3, Dither::BlueNoise).expect("a picture");
        let together = eight_bits(&samples, height, width, 3, Dither::BlueNoiseTogether).expect("a picture");
        let at =
            |levels: &[u8], row: usize, column: usize, channel: usize| levels[(row * width + column) * 3 + channel];
        for row in 0..height {
            for column in 0..width {
                let level = 9 + u8::from(1.0 - THRESHOLDS[(row % SIDE) * SIDE + column % SIDE] <= 0.25);
                assert_eq!(at(&apart, row, column, 0), level);
                for channel in 0..3 {
                    assert_eq!(at(&together, row, column, channel), level);
                }
                let (down, across) = SHIFTS[1];
                let shifted = THRESHOLDS[((row + down) % SIDE) * SIDE + (column + across) % SIDE];
                assert_eq!(at(&apart, row, column, 1), 9 + u8::from(1.0 - shifted <= 0.25));
                let (down, across) = SHIFTS[2];
                let shifted = THRESHOLDS[((row + down) % SIDE) * SIDE + (column + across) % SIDE];
                assert_eq!(at(&apart, row, column, 2), 9 + u8::from(1.0 - shifted <= 0.25));
            }
        }
    }

    #[test]
    #[expect(
        clippy::float_cmp,
        reason = "products are compared exactly with the doubles they round to"
    )]
    fn sixteen_bits_dither_257_times_the_sample() {
        let mut samples = vec![0.0, 255.0, 1.0 / 257.0, 128.5, -3.0, 256.0];
        // Samples whose product rounds onto the place where the dither of its column goes
        // up, from below it or from above, and their neighbours.
        for column in 6..64 {
            let place = f64::from(u32::try_from(column).expect("small") * 997) + (1.0 - THRESHOLDS[column]);
            let near = near_257ths(place);
            let from_below = near.iter().copied().find(|&sample| {
                let y = sample * 257.0;
                y == place && sample.mul_add(257.0, -y) < 0.0
            });
            samples.push(from_below.unwrap_or(near[column % 3]));
        }
        let width = samples.len();
        let made = sixteen_bits(&samples, 1, width, 1, Dither::BlueNoise).expect("finite samples");
        let mut onto = 0;
        for (column, (&sample, &level)) in samples.iter().zip(&made).enumerate() {
            let expected = exact_level_of(sample, 257, RANKS[column], 65535);
            assert_eq!(level, expected, "{sample}");
            // The product rounded alone takes the other level where it lands on the place
            // but the exact one lies below it.
            let y = sample * 257.0;
            if y - y.floor() == 1.0 - THRESHOLDS[column] && sample.mul_add(257.0, -y) < 0.0 {
                onto += 1;
                assert_eq!(level, exact_level(y.floor(), RANKS[column], 65535));
            }
        }
        assert!(onto > 0, "no product landed on a place from below");
    }

    #[test]
    #[expect(
        clippy::float_cmp,
        reason = "products are compared exactly with the doubles they round to"
    )]
    fn sixteen_bits_round_257_times_the_sample() {
        // Every half k + 1/2 near which a 16-bit level of a sample of 0 to 255 lies, the
        // samples whose products round onto it and their neighbours.
        let mut samples = Vec::new();
        let mut onto = 0;
        for whole in (0..65535).step_by(7) {
            for sample in near_257ths(f64::from(whole) + 0.5) {
                let y = sample * 257.0;
                if y == f64::from(whole) + 0.5 && sample.mul_add(257.0, -y) < 0.0 {
                    onto += 1;
                }
                samples.push(sample);
            }
        }
        assert!(onto > 1000, "{onto} products landed on a half from below");
        let made = sixteen_bits(&samples, 1, samples.len(), 1, Dither::None).expect("finite samples");
        for (&sample, &level) in samples.iter().zip(&made) {
            assert_eq!(level, exact_rounded(sample, 257, 65535), "{sample}");
        }
        // 8 bits: the sample is its own level, rounded.
        for sample in [0.5, 0.5_f64.next_down(), 254.5, 254.5_f64.next_down()] {
            let made = eight_bits(&[sample], 1, 1, 1, Dither::None).expect("a finite sample");
            assert_eq!(u16::from(made[0]), exact_rounded(sample, 1, 255), "{sample}");
        }
    }

    #[test]
    fn a_pnm_file_has_its_header_and_samples() {
        let rounding = Depth {
            sixteen: false,
            dither: Dither::None,
        };
        let file = pnm(&[0.0, 255.0, 128.4], 1, 3, 1, rounding).expect("a picture");
        assert_eq!(file, b"P5\n3 1\n255\n\x00\xff\x80");
        let sixteen = Depth {
            sixteen: true,
            dither: Dither::None,
        };
        let file = pnm(&[1.0, 2.0, 3.0], 1, 1, 3, sixteen).expect("a picture");
        assert_eq!(file, b"P6\n1 1\n65535\n\x01\x01\x02\x02\x03\x03");
        // Whole samples are their own levels under the dither too.
        let file = pnm(&[0.0, 255.0, 128.0], 1, 3, 1, Depth::default()).expect("a picture");
        assert_eq!(file, b"P5\n3 1\n255\n\x00\xff\x80");
    }
}
