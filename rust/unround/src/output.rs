// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! Pictures of 8 or 16 bits, rounded from the binary64 result, and the PNM files
//! that hold them.

use crate::error::Error;

/// The integer nearest to a sample, halves away from 0, clamped to `0..=top`.
///
/// The rounding is exact: `f64::round` compares the fraction with 1/2 as it is,
/// where `floor(x + 1/2)` would round `0.49999999999999994` up.
#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a whole number from 0 to 65535 converts exactly"
)]
fn quantized(sample: f64, top: f64) -> u16 {
    sample.round().clamp(0.0, top) as u16
}

/// 8-bit samples of a picture: each rounded to the nearest integer, halves away from
/// 0, and clamped to 0-255.
///
/// # Errors
///
/// [`Error::Options`] for a sample that is not finite.
pub fn eight_bits(samples: &[f64]) -> Result<Vec<u8>, Error> {
    finite(samples)?;
    Ok(samples
        .iter()
        .map(|&sample| u8::try_from(quantized(sample, 255.0)).unwrap_or(u8::MAX))
        .collect())
}

/// 16-bit samples of a picture: each times 257, so that 255 is 65535, rounded to the
/// nearest integer, halves away from 0, and clamped to 0-65535.
///
/// # Errors
///
/// [`Error::Options`] for a sample that is not finite.
pub fn sixteen_bits(samples: &[f64]) -> Result<Vec<u16>, Error> {
    finite(samples)?;
    Ok(samples
        .iter()
        .map(|&sample| quantized(sample * 257.0, 65535.0))
        .collect())
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

/// The bytes of a PNM file of `height x width` pixels of 1 (`P5`) or 3 (`P6`)
/// interleaved samples, of 8 bits, or of 16 bits, most significant byte first.
///
/// # Errors
///
/// [`Error::Options`] for a picture that is not of that size, of other than 1 or 3
/// channels, or of samples that are not finite.
pub fn pnm(samples: &[f64], height: usize, width: usize, channels: usize, sixteen: bool) -> Result<Vec<u8>, Error> {
    if !(channels == 1 || channels == 3) || samples.len() != height * width * channels || height == 0 || width == 0 {
        return Err(Error::Options(format!(
            "a picture of {height} x {width} x 1 or 3 samples, not empty, and not of {} samples in {channels} channels",
            samples.len()
        )));
    }
    let kind = if channels == 1 { "P5" } else { "P6" };
    let top = if sixteen { 65535 } else { 255 };
    let mut file = format!("{kind}\n{width} {height}\n{top}\n").into_bytes();
    if sixteen {
        for value in sixteen_bits(samples)? {
            file.extend_from_slice(&value.to_be_bytes());
        }
    } else {
        file.extend(eight_bits(samples)?);
    }
    Ok(file)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn samples_round_halves_away_from_zero_and_clamp() {
        let samples = [0.499_999_999_999_999_94, 0.5, 1.5, 2.5, -0.4, -7.0, 254.5, 255.4, 1e300];
        let rounded = eight_bits(&samples).expect("finite samples");
        assert_eq!(rounded, [0, 1, 2, 3, 0, 0, 255, 255, 255]);
        assert_eq!(
            sixteen_bits(&[255.0, 1.0, 0.0]).expect("finite samples"),
            [65535, 257, 0]
        );
        assert!(eight_bits(&[f64::NAN]).is_err());
    }

    #[test]
    fn a_pnm_file_has_its_header_and_samples() {
        let file = pnm(&[0.0, 255.0, 128.4], 1, 3, 1, false).expect("a picture");
        assert_eq!(file, b"P5\n3 1\n255\n\x00\xff\x80");
        let file = pnm(&[1.0, 2.0, 3.0], 1, 1, 3, true).expect("a picture");
        assert_eq!(file, b"P6\n1 1\n65535\n\x01\x01\x02\x02\x03\x03");
    }
}
