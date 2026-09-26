// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! Decoding a JPEG file: its components, reconstructed within their intervals
//! (docs/math.md, 1.3 and 8).
//!
//! The components are reconstructed on one canvas ([`frames`]) by one of the
//! methods, and the picture is cut from it. The methods are the decoder of the data
//! term's centres ([`Method::Mmse`]: the MMSE decoder with the default centres,
//! docs/math.md, 2.3), TV and TGV by the primal-dual method (5), and, for greyscale
//! files, TV by the subgradient method of jpeg2png's kind (7), which is there to be
//! compared with. A file in YCbCr is solved in YCbCr, and its picture is the RGB that
//! JFIF's conversion gives (8); that of a file in RGB, or greyscale, is its canvas.

use jpegio_sys::{ColorSpace, Image};

use crate::colour;
use crate::error::Error;
use crate::frames::{self, Frame};
use crate::model::{DataTerm, Problem, Tgv, Tv};
use crate::pdhg::{self, Observer};
use crate::results::FrameResult;
use crate::subgradient;

/// How a file is reconstructed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Method {
    /// The decoder of the data term's centres.
    Mmse,
    /// TV by the primal-dual method.
    Tv,
    /// TGV by the primal-dual method.
    #[default]
    Tgv,
    /// TV by the subgradient method, for greyscale files.
    Subgradient,
}

/// The method, the model, and the solvers' options.
///
/// `data` are the options of `G`: one for every component, or one for each. `tv`
/// and `tgv` are the weights of the models, with how they take the channels; `pdhg`
/// and `subgradient` the options of the solvers; `read` the limits of the C layer.
#[derive(Debug, Clone, PartialEq)]
pub struct Settings {
    /// How the file is reconstructed.
    pub method: Method,
    /// The options of `G`: one, or one for each component.
    pub data: Vec<DataTerm>,
    /// The weights of TV.
    pub tv: Tv,
    /// The weights of TGV.
    pub tgv: Tgv,
    /// The options of the primal-dual method.
    pub pdhg: pdhg::Options,
    /// The options of the subgradient method.
    pub subgradient: subgradient::Options,
    /// The limits of the read.
    pub read: jpegio_sys::Options,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            method: Method::default(),
            data: vec![DataTerm::default()],
            tv: Tv::default(),
            tgv: Tgv::default(),
            pdhg: pdhg::Options::default(),
            subgradient: subgradient::Options::default(),
            read: jpegio_sys::Options::default(),
        }
    }
}

/// A reconstructed file.
///
/// `picture` is in binary64, `height x width x channels`, interleaved: of a
/// greyscale file or one in RGB, the solver's canvas itself, cut to the picture; of
/// one in YCbCr, the RGB that JFIF's conversion gives of that (docs/math.md, 8). It
/// is neither clamped nor rounded. `planes` are the canvas cut to the picture,
/// channel by channel, as the solver left it: the Y, Cb and Cr of a file in YCbCr.
/// `coefficients` are every component's, each within its intervals. `result` is the
/// solver's, `None` for the MMSE decoder.
#[derive(Debug, Clone)]
pub struct Decoded {
    /// The picture, interleaved.
    pub picture: Vec<f64>,
    /// Rows.
    pub height: usize,
    /// Columns.
    pub width: usize,
    /// Samples to a pixel: 1 or 3.
    pub channels: usize,
    /// The canvas cut to the picture, channel by channel.
    pub planes: Vec<f64>,
    /// The file's color space.
    pub color_space: ColorSpace,
    /// Every component's coefficients.
    pub coefficients: Vec<Vec<f64>>,
    /// The whole canvas of every channel, `C x H x W`, as the solver left it: the
    /// picture and what lies beyond it.
    pub canvas: Vec<f64>,
    /// The components on their canvas.
    pub frame: Frame,
    /// What the solver returned.
    pub result: Option<FrameResult>,
    /// The file's ICC profile.
    pub icc_profile: Option<Vec<u8>>,
    /// The file's EXIF orientation, 1 to 8, or 0 where it has none.
    pub exif_orientation: i32,
}

/// A component of a file: its quantized levels, `rows x columns` blocks of 64 in
/// natural order, its quantization table, and its sampling factors, across and down.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Component {
    /// The levels, block rows from the top, each block in natural order.
    pub levels: Vec<i16>,
    /// Block rows.
    pub rows: usize,
    /// Blocks across.
    pub columns: usize,
    /// The quantization table, in natural order.
    pub quant_table: [u16; 64],
    /// The horizontal sampling factor.
    pub h_samp_factor: usize,
    /// The vertical sampling factor.
    pub v_samp_factor: usize,
}

/// What a reconstruction starts from: the picture's size, its color space, and its
/// components, as a file holds them or as they are given; and the file's ICC profile
/// and EXIF orientation, which the reconstruction carries along.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Input {
    /// Rows of the picture.
    pub height: usize,
    /// Columns of the picture.
    pub width: usize,
    /// The color space of the components.
    pub color_space: ColorSpace,
    /// The components: one, or three.
    pub components: Vec<Component>,
    /// The ICC profile, where there is one.
    pub icc_profile: Option<Vec<u8>>,
    /// The EXIF orientation, 1 to 8, or 0 where there is none.
    pub exif_orientation: i32,
}

fn factor(value: i32) -> Result<usize, Error> {
    usize::try_from(value).map_err(|_| Error::Unsupported(format!("a sampling factor of {value}")))
}

fn blocks(value: u32) -> usize {
    // Lossless on the 64-bit systems JPEG-Unround supports.
    usize::try_from(value).unwrap_or(usize::MAX)
}

impl Input {
    /// The input of an image that the C layer read.
    ///
    /// # Errors
    ///
    /// [`Error::Unsupported`] for sampling factors that are not positive.
    pub fn of_image(image: &Image) -> Result<Self, Error> {
        let components = image
            .components()
            .map(|component| {
                Ok(Component {
                    levels: component.coefficients().to_vec(),
                    rows: blocks(component.height_in_blocks()),
                    columns: blocks(component.width_in_blocks()),
                    quant_table: *component.quant_table(),
                    h_samp_factor: factor(component.h_samp_factor())?,
                    v_samp_factor: factor(component.v_samp_factor())?,
                })
            })
            .collect::<Result<Vec<Component>, Error>>()?;
        Ok(Self {
            height: blocks(image.height()),
            width: blocks(image.width()),
            color_space: image.color_space(),
            components,
            icc_profile: image.icc_profile().map(<[u8]>::to_vec),
            exif_orientation: image.exif_orientation(),
        })
    }

    /// The input of a JPEG file in memory.
    ///
    /// # Errors
    ///
    /// [`Error::Read`] for a file that the C layer does not read, and the errors of
    /// [`Input::of_image`].
    pub fn read(data: &[u8], options: &jpegio_sys::Options) -> Result<Self, Error> {
        Self::of_image(&Image::read(data, options)?)
    }
}

/// The data term of component `index`: the one of every component, or its own.
fn data_of(settings: &Settings, index: usize, count: usize) -> Result<&DataTerm, Error> {
    match settings.data.len() {
        1 => Ok(&settings.data[0]),
        length if length == count => Ok(&settings.data[index]),
        length => Err(Error::Options(format!(
            "options of G for each of the {count} components, or one for all, not {length}"
        ))),
    }
}

/// The frame of an input's components (docs/math.md, 1.3), with the model of the
/// settings.
///
/// # Errors
///
/// [`Error::Options`] for options of `G` out of their ranges, or levels that are not
/// of the component's blocks; [`Error::Unsupported`] for a color space and a count of
/// components that do not go together, or sampling factors that are not whole
/// multiples.
pub fn frame_of(input: &Input, settings: &Settings) -> Result<Frame, Error> {
    let count = input.components.len();
    let expected = if input.color_space == ColorSpace::Grayscale {
        1
    } else {
        3
    };
    if count != expected {
        return Err(Error::Unsupported(format!(
            "a file of the color space {:?} has {expected} components, not {count}",
            input.color_space
        )));
    }
    let mut problems = Vec::with_capacity(count);
    let mut factors = Vec::with_capacity(count);
    for (index, component) in input.components.iter().enumerate() {
        let data = data_of(settings, index, count)?;
        problems.push(Problem::new(
            &component.levels,
            component.rows,
            component.columns,
            &component.quant_table,
            data,
            None,
        )?);
        factors.push((component.h_samp_factor, component.v_samp_factor));
    }
    Frame::of_file(problems, &factors, (input.height, input.width))
}

/// Reconstructs an input, greyscale or in colour. `observer` is called with the
/// solver's records, and can stop it.
///
/// # Errors
///
/// The errors of [`frame_of`] and of the solvers.
pub fn solve(input: &Input, settings: &Settings, observer: Option<&mut Observer<'_>>) -> Result<Decoded, Error> {
    let frame = frame_of(input, settings)?;
    let result = match settings.method {
        Method::Mmse => None,
        Method::Tv => Some(pdhg::solve_tv(&frame, &settings.tv, &settings.pdhg, None, observer)?),
        Method::Tgv => Some(pdhg::solve_tgv(&frame, &settings.tgv, &settings.pdhg, None, observer)?),
        Method::Subgradient => Some(subgradient::solve_tv(
            &frame,
            &settings.tv,
            &settings.subgradient,
            None,
            observer,
        )?),
    };
    let point = match &result {
        Some(solved) => solved.primal.clone(),
        None => frames::start(&frame, None)?,
    };
    let (height, width) = (input.height, input.width);
    let count = frame.channels().len();
    let mut planes = Vec::with_capacity(count * height * width);
    for channel in 0..count {
        let plane = &point.canvas[channel * frame.plane()..(channel + 1) * frame.plane()];
        for row in plane.chunks_exact(frame.width()).take(height) {
            planes.extend_from_slice(&row[..width]);
        }
    }
    let (picture, channels) = match (input.color_space, count) {
        (ColorSpace::YCbCr, _) => (colour::to_rgb(&planes, height, width)?, 3),
        (_, 1) => (planes.clone(), 1),
        _ => {
            let size = height * width;
            let mut interleaved = Vec::with_capacity(count * size);
            for pixel in 0..size {
                for channel in 0..count {
                    interleaved.push(planes[channel * size + pixel]);
                }
            }
            (interleaved, count)
        }
    };
    Ok(Decoded {
        picture,
        height,
        width,
        channels,
        planes,
        color_space: input.color_space,
        coefficients: point.coefficients,
        canvas: point.canvas,
        frame,
        result,
        icc_profile: input.icc_profile.clone(),
        exif_orientation: input.exif_orientation,
    })
}

/// Reconstructs a JPEG file, greyscale or in colour: [`solve`] of its [`Input`].
///
/// # Errors
///
/// [`Error::Read`] for a file that the C layer does not read, and the errors of
/// [`solve`].
pub fn decode(data: &[u8], settings: &Settings, observer: Option<&mut Observer<'_>>) -> Result<Decoded, Error> {
    solve(&Input::read(data, &settings.read)?, settings, observer)
}
