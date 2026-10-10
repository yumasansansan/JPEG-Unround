// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! A PNG file reconstructed within the intervals of its rounded samples
//! (docs/math.md, 11): its colours as one picture, its alpha, where it has one, as
//! another, each by the method of the settings.

use crate::decode::{Method, Settings};
use crate::error::Error;
use crate::model::{DataTerm, Tgv, Tv};
use crate::pdhg::Observer;
use crate::samples::{self, Data, Model, Solution};

/// A PNG file reconstructed: its picture, interleaved as the file's samples are (gray
/// or RGB, then alpha), and what the solvers returned.
#[derive(Debug, Clone)]
pub struct Reconstructed {
    /// The picture, `height x width x channels`, the samples of a pixel together.
    pub picture: Vec<f64>,
    /// Rows.
    pub height: usize,
    /// Columns.
    pub width: usize,
    /// Samples to a pixel: 1 to 4, the last of 2 and 4 alpha.
    pub channels: usize,
    /// The bits of the file's samples.
    pub bits: u32,
    /// The file's ICC profile.
    pub icc_profile: Option<Vec<u8>>,
    /// The colours' solution.
    pub colours: Solution,
    /// The alpha's solution, where the file has alpha.
    pub alpha: Option<Solution>,
    /// The first warning of the reading, or an empty string.
    pub warning: String,
}

/// Whether data is a PNG file, by its signature.
#[must_use]
pub fn is_png(data: &[u8]) -> bool {
    data.starts_with(&[0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1A, b'\n'])
}

/// The data term of a PNG file from the settings: `--mu`, a number (0 where it
/// follows the rule, which is JPEG's), and `--slack`, in levels. The other options of
/// the data term are JPEG's, and are refused where they are not their defaults.
fn data_of(settings: &Settings) -> Result<Data, Error> {
    let [term] = settings.data.as_slice() else {
        return Err(Error::Options(
            "a PNG file takes one value of each option of the data term, not one for each component".into(),
        ));
    };
    let jpeg = DataTerm::default();
    let defaults = DataTerm {
        mu: term.mu,
        slack: term.slack,
        ..jpeg
    };
    if *term != defaults {
        return Err(Error::Options(
            "of the data term, a PNG file takes --mu and --slack alone: the others are for JPEG files".into(),
        ));
    }
    Ok(Data {
        mu: term.mu.unwrap_or(0.0),
        slack: term.slack,
    })
}

/// Reconstructs a PNG file by the settings: TV or TGV (the default), the colours with
/// the settings' weights of the channels, the alpha alone.
///
/// # Errors
///
/// [`Error::Read`] where the C layer cannot read the file, [`Error::Unsupported`] for a
/// palette file, and [`Error::Options`] for settings that a PNG file does not take.
pub fn reconstruct(
    data: &[u8],
    settings: &Settings,
    mut observer: Option<&mut Observer<'_>>,
) -> Result<Reconstructed, Error> {
    let terms = data_of(settings)?;
    let method = match settings.method {
        Method::Tv => samples::Method::Tv(&settings.tv),
        Method::Tgv => samples::Method::Tgv(&settings.tgv),
        Method::Mmse | Method::Subgradient => {
            return Err(Error::Options("a PNG file is reconstructed by TV or TGV".into()));
        }
    };
    let picture = jpegio_sys::read_png(data, settings.read.max_pixels).map_err(|error| match error.kind() {
        jpegio_sys::ErrorKind::Unsupported => Error::Unsupported(error.message().to_owned()),
        jpegio_sys::ErrorKind::Limit => Error::Options(error.message().to_owned()),
        _ => Error::Read(error),
    })?;
    let (height, width) = (
        usize::try_from(picture.height).unwrap_or(0),
        usize::try_from(picture.width).unwrap_or(0),
    );
    let channels = usize::try_from(picture.channels).unwrap_or(0);
    let colours = if channels == 2 || channels == 4 {
        channels - 1
    } else {
        channels
    };
    let plane = height * width;
    // The file's levels, channel by channel.
    let planes = |channel: usize| -> Vec<u16> {
        picture
            .samples
            .iter()
            .skip(channel)
            .step_by(channels)
            .copied()
            .collect()
    };
    let mut levels = Vec::with_capacity(colours * plane);
    for channel in 0..colours {
        levels.extend(planes(channel));
    }
    let model = Model::new(&levels, (colours, height, width), picture.bits, &terms)?;
    let colour_solution = samples::solve(&model, method, &settings.pdhg, observer.as_deref_mut())?;
    // The alpha alone: one channel, without the colours' weights of channels.
    let variation = Tv {
        channel_weights: None,
        ..settings.tv.clone()
    };
    let generalized = Tgv {
        channel_weights: None,
        ..settings.tgv.clone()
    };
    let alone = match method {
        samples::Method::Tv(_) => samples::Method::Tv(&variation),
        samples::Method::Tgv(_) => samples::Method::Tgv(&generalized),
    };
    let alpha = if colours < channels {
        let alpha_model = Model::new(&planes(channels - 1), (1, height, width), picture.bits, &terms)?;
        Some(samples::solve(&alpha_model, alone, &settings.pdhg, observer)?)
    } else {
        None
    };
    let mut interleaved = vec![0.0; channels * plane];
    for channel in 0..colours {
        for (target, &value) in interleaved
            .iter_mut()
            .skip(channel)
            .step_by(channels)
            .zip(&colour_solution.picture[channel * plane..(channel + 1) * plane])
        {
            *target = value;
        }
    }
    if let Some(solution) = &alpha {
        for (target, &value) in interleaved
            .iter_mut()
            .skip(channels - 1)
            .step_by(channels)
            .zip(&solution.picture)
        {
            *target = value;
        }
    }
    Ok(Reconstructed {
        picture: interleaved,
        height,
        width,
        channels,
        bits: picture.bits,
        icc_profile: picture.icc_profile,
        colours: colour_solution,
        alpha,
        warning: picture.warning,
    })
}
