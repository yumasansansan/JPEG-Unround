// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! A picture of a file's size made elsewhere -- by another decoder, say -- measured
//! against the file's intervals, and projected onto them (docs/math.md, 12).
//!
//! Its canvas is made as an encoder makes the file's: of a file in YCbCr, the Y, Cb
//! and Cr of JFIF's conversion of its RGB; of a greyscale or RGB file, its own
//! channels; each channel extended over the canvas's whole MCUs by repeating its last
//! column and then its last row. The picture is taken as the file stores it, before
//! its EXIF orientation.

use jpegio_sys::ColorSpace;

use crate::colour;
use crate::decode::{self, Decoded, Input, Settings};
use crate::error::Error;
use crate::frames::{self, Frame};
use crate::kernels;

/// A picture projected onto the intervals of a file.
#[derive(Debug, Clone)]
pub struct Projection {
    /// The projection, as a reconstruction of the file without a solver: its picture,
    /// planes and canvas, and its coefficients, each within its interval.
    pub decoded: Decoded,
    /// How far each coefficient of the picture given lay outside its interval, in
    /// steps (0 within it): component by component, 64 to a block in natural order.
    pub excess: Vec<Vec<f64>>,
}

/// How much of a component lies outside its intervals.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Summary {
    /// The component's coefficients.
    pub coefficients: usize,
    /// How many of them lie outside their intervals.
    pub outside: usize,
    /// The largest excess, in steps: 0 where none lies outside.
    pub largest: f64,
}

/// The summary of a component's excess, whose values are at least 0.
#[must_use]
pub fn summary(excess: &[f64]) -> Summary {
    let mut outside = 0;
    for &value in excess {
        outside += kernels::above_zero(value);
    }
    Summary {
        coefficients: excess.len(),
        outside,
        largest: kernels::most(excess),
    }
}

/// The samples to a pixel of a picture of an input: 1 of a greyscale file, 3 (RGB) of
/// a file in colour.
#[must_use]
pub fn channels_of(input: &Input) -> usize {
    if input.color_space == ColorSpace::Grayscale {
        1
    } else {
        3
    }
}

/// The canvas of a picture of the input's size, `height x width x channels` samples
/// interleaved, on the input's frame: `C x H x W`, channel by channel.
///
/// # Errors
///
/// [`Error::Options`] for a picture that is not of the input's size, not of its
/// file's channels, or of samples that are not all finite.
pub fn canvas_of(input: &Input, frame: &Frame, picture: &[f64], channels: usize) -> Result<Vec<f64>, Error> {
    let (height, width) = (input.height, input.width);
    let expected = channels_of(input);
    if channels != expected || picture.len() != height * width * channels || height == 0 || width == 0 {
        return Err(Error::Options(format!(
            "a picture of the file is {height} x {width} x {expected} samples, not {} in {channels} channels",
            picture.len()
        )));
    }
    if !picture.iter().all(|sample| sample.is_finite()) {
        return Err(Error::Options("a picture's samples are finite".into()));
    }
    let size = height * width;
    let planes = if input.color_space == ColorSpace::YCbCr {
        colour::to_ycbcr(picture, height, width)?
    } else {
        let mut planes = vec![0.0; channels * size];
        for (pixel, samples) in picture.chunks_exact(channels).enumerate() {
            for (channel, &sample) in samples.iter().enumerate() {
                planes[channel * size + pixel] = sample;
            }
        }
        planes
    };
    let (rows, columns) = (frame.height(), frame.width());
    let mut canvas = Vec::with_capacity(channels * rows * columns);
    for plane in planes.chunks_exact(size) {
        for i in 0..rows {
            let row = &plane[i.min(height - 1) * width..][..width];
            canvas.extend_from_slice(row);
            canvas.extend(std::iter::repeat_n(row[width - 1], columns - width));
        }
    }
    Ok(canvas)
}

/// How far each coefficient of a picture of the input's size lies outside its
/// interval, with the settings' slack, in steps (0 within it): component by
/// component, 64 to a block in natural order.
///
/// # Errors
///
/// The errors of [`decode::frame_of`] and of [`canvas_of`].
pub fn excess(input: &Input, settings: &Settings, picture: &[f64], channels: usize) -> Result<Vec<Vec<f64>>, Error> {
    let frame = decode::frame_of(input, settings)?;
    let canvas = canvas_of(input, &frame, picture, channels)?;
    Ok(frames::excess(&frame, &canvas))
}

/// The projection of a picture of the input's size onto its intervals, with the
/// settings' slack, and the picture's excess.
///
/// # Errors
///
/// The errors of [`decode::frame_of`] and of [`canvas_of`].
pub fn project(input: &Input, settings: &Settings, picture: &[f64], channels: usize) -> Result<Projection, Error> {
    let frame = decode::frame_of(input, settings)?;
    let canvas = canvas_of(input, &frame, picture, channels)?;
    let excess = frames::excess(&frame, &canvas);
    let (projected, coefficients) = frames::project_with_coefficients(&frame, &canvas);
    let decoded = decode::decoded_of(input, frame, projected, coefficients, None)?;
    Ok(Projection { decoded, excess })
}
