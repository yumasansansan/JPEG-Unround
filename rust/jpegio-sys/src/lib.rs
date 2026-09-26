// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! The C layer of JPEG-Unround (`unround/jpegio.h`) for Rust.
//!
//! The C layer reads what the reconstruction needs from a JPEG file held in
//! memory -- the quantized DCT coefficients of every component, the quantization
//! table each was quantized with, the sampling factors, the ICC profile and the
//! EXIF orientation -- and decodes the component planes as libjpeg's standard
//! decoder does, before upsampling and color conversion. [`ffi`] declares its
//! structures and functions as C has them. [`Image`] and [`Planes`] own what a
//! read returns, free it when they are dropped, and give its arrays as slices.
//!
//! [`testing`] writes JPEG files from coefficients, for tests, with libjpeg's own
//! encoder.
//!
//! All the unsafe code of the Rust implementation is in this crate: the other
//! crates forbid it.

pub mod ffi;
pub mod testing;

use std::ffi::{CStr, c_char};
use std::fmt;
use std::slice;

/// What went wrong, as the C layer's status says it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorKind {
    /// An argument the C layer cannot take.
    Argument,
    /// Data that libjpeg could not read.
    Decode,
    /// A JPEG outside the project's scope: 12-bit, lossless, CMYK and the like.
    Unsupported,
    /// A file larger than the options allow.
    Limit,
    /// An allocation that failed.
    Memory,
}

impl ErrorKind {
    fn from_status(status: ffi::Status) -> Option<Self> {
        match status {
            ffi::ERROR_ARGUMENT => Some(Self::Argument),
            ffi::ERROR_DECODE => Some(Self::Decode),
            ffi::ERROR_UNSUPPORTED => Some(Self::Unsupported),
            ffi::ERROR_LIMIT => Some(Self::Limit),
            ffi::ERROR_MEMORY => Some(Self::Memory),
            _ => None,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Argument => "argument",
            Self::Decode => "decode",
            Self::Unsupported => "unsupported",
            Self::Limit => "limit",
            Self::Memory => "memory",
        }
    }
}

/// A JPEG file that the C layer could not read, or would not.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    kind: ErrorKind,
    message: String,
}

impl Error {
    /// What went wrong.
    #[must_use]
    pub fn kind(&self) -> ErrorKind {
        self.kind
    }

    /// The C layer's description of it.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }

    fn from_status(status: ffi::Status, message: String) -> Self {
        match ErrorKind::from_status(status) {
            Some(kind) => Self { kind, message },
            None => Self {
                kind: ErrorKind::Decode,
                message: format!("the C layer returned status {status}: {message}"),
            },
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} ({})", self.message, self.kind.name())
    }
}

impl std::error::Error for Error {}

/// The limits a read accepts; zero means the C layer's defaults (1 << 28 pixels,
/// 500 scans).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Options {
    /// The largest width * height a file may declare.
    pub max_pixels: u64,
    /// The most scans a progressive file may have.
    pub max_scans: u32,
    /// Whether a corrupt-data warning fails the read.
    pub warnings_are_errors: bool,
}

impl Options {
    fn to_ffi(self) -> ffi::Options {
        ffi::Options {
            max_pixels: self.max_pixels,
            // The C layer takes an int32_t; more scans than that is no limit at all.
            max_scans: i32::try_from(self.max_scans).unwrap_or(i32::MAX),
            warnings_are_errors: i32::from(self.warnings_are_errors),
        }
    }
}

/// The color space of the components, as the file declares it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ColorSpace {
    /// One component.
    Grayscale,
    /// Three components: luma and two chroma differences.
    YCbCr,
    /// Three components stored without a color transform.
    Rgb,
}

const MESSAGE_SIZE: usize = 256;

fn message_text(message: &[c_char; MESSAGE_SIZE]) -> String {
    let bytes: Vec<u8> = message
        .iter()
        .take_while(|&&c| c != 0)
        .map(|&c| c.to_ne_bytes()[0])
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

fn to_usize(value: u32) -> usize {
    // Lossless on the 64-bit systems JPEG-Unround supports (ffi asserts them).
    usize::try_from(value).unwrap_or(usize::MAX)
}

fn coefficient_count(component: &ffi::Component) -> Option<usize> {
    to_usize(component.width_in_blocks)
        .checked_mul(to_usize(component.height_in_blocks))?
        .checked_mul(ffi::BLOCK_SIZE)
}

fn sample_count(plane: &ffi::Plane) -> Option<usize> {
    to_usize(plane.stride).checked_mul(to_usize(plane.height))
}

fn component_count(num_components: i32) -> Option<usize> {
    usize::try_from(num_components)
        .ok()
        .filter(|&count| count <= ffi::MAX_COMPONENTS)
}

/// The coefficients and the metadata of a JPEG file, as the C layer reads them.
#[derive(Debug)]
pub struct Image {
    raw: ffi::Image,
    color_space: ColorSpace,
    components: usize,
    warning: String,
}

// SAFETY: an Image owns the memory its pointers lead to, alone; the C layer
// keeps no reference to it and ties it to no thread, and nothing changes it
// through a shared reference.
unsafe impl Send for Image {}
// SAFETY: as above.
unsafe impl Sync for Image {}

impl Image {
    /// Reads the coefficients and the metadata of a JPEG file in memory.
    ///
    /// # Errors
    ///
    /// An [`Error`] when the data is not a JPEG file the C layer reads, or asks
    /// for more than `options` allow.
    pub fn read(data: &[u8], options: &Options) -> Result<Self, Error> {
        let c_options = options.to_ffi();
        let mut raw = ffi::Image::EMPTY;
        let mut message: [c_char; MESSAGE_SIZE] = [0; MESSAGE_SIZE];
        // SAFETY: data is valid for data.len() bytes, c_options and raw for the
        // accesses the function makes, and message for MESSAGE_SIZE bytes, all of
        // them for the length of the call.
        let status = unsafe {
            ffi::unround_jpegio_read(
                data.as_ptr(),
                data.len(),
                &raw const c_options,
                &raw mut raw,
                message.as_mut_ptr(),
                MESSAGE_SIZE,
            )
        };
        let text = message_text(&message);
        if status != ffi::OK {
            return Err(Error::from_status(status, text));
        }
        // From here on, raw owns memory, and the Image frees it when dropped: an
        // image that breaks a promise of the C layer is freed as well.
        let mut image = Self {
            raw,
            color_space: ColorSpace::YCbCr,
            components: 0,
            warning: text,
        };
        let components = component_count(image.raw.num_components);
        let color_space = match image.raw.color_space {
            ffi::GRAYSCALE => Some(ColorSpace::Grayscale),
            ffi::YCBCR => Some(ColorSpace::YCbCr),
            ffi::RGB => Some(ColorSpace::Rgb),
            _ => None,
        };
        let (Some(components), Some(color_space)) = (components, color_space) else {
            return Err(Error {
                kind: ErrorKind::Decode,
                message: "the C layer returned an image it cannot have".into(),
            });
        };
        let complete = image.raw.components[..components]
            .iter()
            .all(|component| coefficient_count(component).is_some() && !component.coefficients.is_null());
        if !complete {
            return Err(Error {
                kind: ErrorKind::Decode,
                message: "the C layer returned a component it cannot have".into(),
            });
        }
        image.components = components;
        image.color_space = color_space;
        Ok(image)
    }

    /// Samples across.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.raw.width
    }

    /// Rows.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.raw.height
    }

    /// The color space of the components.
    #[must_use]
    pub fn color_space(&self) -> ColorSpace {
        self.color_space
    }

    /// The largest horizontal sampling factor.
    #[must_use]
    pub fn max_h_samp_factor(&self) -> i32 {
        self.raw.max_h_samp_factor
    }

    /// The largest vertical sampling factor.
    #[must_use]
    pub fn max_v_samp_factor(&self) -> i32 {
        self.raw.max_v_samp_factor
    }

    /// Whether the file is progressive.
    #[must_use]
    pub fn progressive(&self) -> bool {
        self.raw.progressive != 0
    }

    /// Whether the file is arithmetic-coded.
    #[must_use]
    pub fn arithmetic(&self) -> bool {
        self.raw.arithmetic != 0
    }

    /// 1 to 8 from the EXIF APP1 marker, or 0 when there is none.
    #[must_use]
    pub fn exif_orientation(&self) -> i32 {
        self.raw.exif_orientation
    }

    /// Corrupt-data warnings libjpeg reported while reading.
    #[must_use]
    pub fn warnings(&self) -> i32 {
        self.raw.warnings
    }

    /// The first of those warnings, or an empty string.
    #[must_use]
    pub fn warning(&self) -> &str {
        &self.warning
    }

    /// The components, one for grayscale and three otherwise.
    #[must_use]
    pub fn components(&self) -> impl ExactSizeIterator<Item = Component<'_>> + '_ {
        self.raw.components[..self.components]
            .iter()
            .map(|raw| Component { raw })
    }

    /// One component, when there is one of that index.
    #[must_use]
    pub fn component(&self, index: usize) -> Option<Component<'_>> {
        self.raw.components[..self.components]
            .get(index)
            .map(|raw| Component { raw })
    }

    /// The ICC profile of the APP2 markers, when the file has one.
    #[must_use]
    pub fn icc_profile(&self) -> Option<&[u8]> {
        let size = usize::try_from(self.raw.icc_profile_size).ok()?;
        if self.raw.icc_profile.is_null() || size == 0 {
            return None;
        }
        // SAFETY: a successful read allocates icc_profile_size bytes at
        // icc_profile, which live until the image is dropped.
        Some(unsafe { slice::from_raw_parts(self.raw.icc_profile, size) })
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        // SAFETY: raw is what a successful read filled in, freed nowhere else.
        unsafe { ffi::unround_jpegio_image_free(&raw mut self.raw) };
    }
}

/// One component of an [`Image`].
#[derive(Debug, Clone, Copy)]
pub struct Component<'image> {
    raw: &'image ffi::Component,
}

impl<'image> Component<'image> {
    /// The component identifier of the SOF marker.
    #[must_use]
    pub fn id(&self) -> i32 {
        self.raw.id
    }

    /// 1 to 4.
    #[must_use]
    pub fn h_samp_factor(&self) -> i32 {
        self.raw.h_samp_factor
    }

    /// 1 to 4.
    #[must_use]
    pub fn v_samp_factor(&self) -> i32 {
        self.raw.v_samp_factor
    }

    /// 0 to 3: the DQT slot the SOF marker names for the component.
    #[must_use]
    pub fn quant_table_slot(&self) -> i32 {
        self.raw.quant_table_slot
    }

    /// Samples of this component that cover the image.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.raw.width
    }

    /// Rows of this component that cover the image.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.raw.height
    }

    /// ceil(width / 8): the blocks that carry image data.
    #[must_use]
    pub fn width_in_blocks(&self) -> u32 {
        self.raw.width_in_blocks
    }

    /// ceil(height / 8).
    #[must_use]
    pub fn height_in_blocks(&self) -> u32 {
        self.raw.height_in_blocks
    }

    /// The quantization table the component was quantized with, in natural
    /// order: entry `v * 8 + u` for vertical frequency v and horizontal frequency u.
    #[must_use]
    pub fn quant_table(&self) -> &'image [u16; ffi::BLOCK_SIZE] {
        &self.raw.quant_table
    }

    /// The coefficients: `height_in_blocks * width_in_blocks` blocks, block rows
    /// from the top and blocks from the left, each 64 in natural order. The dummy
    /// blocks that complete an MCU are not included.
    #[must_use]
    pub fn coefficients(&self) -> &'image [i16] {
        let count = coefficient_count(self.raw).unwrap_or(0);
        if self.raw.coefficients.is_null() || count == 0 {
            return &[];
        }
        // SAFETY: a successful read allocates this many coefficients at this
        // pointer (Image::read checked both), which live as long as the image.
        unsafe { slice::from_raw_parts(self.raw.coefficients, count) }
    }

    /// The 64 coefficients of the block in block column `x` and block row `y`.
    #[must_use]
    pub fn block(&self, x: u32, y: u32) -> Option<&'image [i16; ffi::BLOCK_SIZE]> {
        if x >= self.raw.width_in_blocks || y >= self.raw.height_in_blocks {
            return None;
        }
        let index = to_usize(y)
            .checked_mul(to_usize(self.raw.width_in_blocks))?
            .checked_add(to_usize(x))?;
        let start = index.checked_mul(ffi::BLOCK_SIZE)?;
        self.coefficients()
            .get(start..start.checked_add(ffi::BLOCK_SIZE)?)?
            .try_into()
            .ok()
    }
}

/// The component planes of a JPEG file, as the C layer decodes them.
#[derive(Debug)]
pub struct Planes {
    raw: ffi::Planes,
    count: usize,
    warning: String,
}

// SAFETY: as for Image.
unsafe impl Send for Planes {}
// SAFETY: as for Image.
unsafe impl Sync for Planes {}

impl Planes {
    /// Decodes the component planes of a JPEG file in memory.
    ///
    /// # Errors
    ///
    /// An [`Error`] when the data is not a JPEG file the C layer reads, or asks
    /// for more than `options` allow.
    pub fn decode(data: &[u8], options: &Options) -> Result<Self, Error> {
        let c_options = options.to_ffi();
        let mut raw = ffi::Planes::EMPTY;
        let mut message: [c_char; MESSAGE_SIZE] = [0; MESSAGE_SIZE];
        // SAFETY: as in Image::read.
        let status = unsafe {
            ffi::unround_jpegio_decode_planes(
                data.as_ptr(),
                data.len(),
                &raw const c_options,
                &raw mut raw,
                message.as_mut_ptr(),
                MESSAGE_SIZE,
            )
        };
        let text = message_text(&message);
        if status != ffi::OK {
            return Err(Error::from_status(status, text));
        }
        let mut planes = Self {
            raw,
            count: 0,
            warning: text,
        };
        let Some(count) = component_count(planes.raw.num_components) else {
            return Err(Error {
                kind: ErrorKind::Decode,
                message: "the C layer returned planes it cannot have".into(),
            });
        };
        let complete = planes.raw.planes[..count]
            .iter()
            .all(|plane| sample_count(plane).is_some() && !plane.samples.is_null() && plane.width <= plane.stride);
        if !complete {
            return Err(Error {
                kind: ErrorKind::Decode,
                message: "the C layer returned a plane it cannot have".into(),
            });
        }
        planes.count = count;
        Ok(planes)
    }

    /// The planes, one for each component.
    #[must_use]
    pub fn planes(&self) -> impl ExactSizeIterator<Item = Plane<'_>> + '_ {
        self.raw.planes[..self.count].iter().map(|raw| Plane { raw })
    }

    /// One plane, when there is one of that index.
    #[must_use]
    pub fn plane(&self, index: usize) -> Option<Plane<'_>> {
        self.raw.planes[..self.count].get(index).map(|raw| Plane { raw })
    }

    /// The first corrupt-data warning libjpeg reported while decoding, or an empty string.
    #[must_use]
    pub fn warning(&self) -> &str {
        &self.warning
    }
}

impl Drop for Planes {
    fn drop(&mut self) {
        // SAFETY: raw is what a successful decoding filled in, freed nowhere else.
        unsafe { ffi::unround_jpegio_planes_free(&raw mut self.raw) };
    }
}

/// One component of [`Planes`]: the inverse DCT of its coefficients with
/// libjpeg's accurate integer method, level-shifted and clamped to 0-255, padded
/// to whole blocks.
#[derive(Debug, Clone, Copy)]
pub struct Plane<'planes> {
    raw: &'planes ffi::Plane,
}

impl<'planes> Plane<'planes> {
    /// `width_in_blocks * 8`.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.raw.width
    }

    /// `height_in_blocks * 8`.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.raw.height
    }

    /// Samples from one row to the next.
    #[must_use]
    pub fn stride(&self) -> u32 {
        self.raw.stride
    }

    /// All the samples: `stride * height` of them.
    #[must_use]
    pub fn samples(&self) -> &'planes [u8] {
        let count = sample_count(self.raw).unwrap_or(0);
        if self.raw.samples.is_null() || count == 0 {
            return &[];
        }
        // SAFETY: a successful decoding allocates this many samples at this
        // pointer (Planes::decode checked both), which live as long as the planes.
        unsafe { slice::from_raw_parts(self.raw.samples, count) }
    }

    /// The `width` samples of row `y`.
    #[must_use]
    pub fn row(&self, y: u32) -> Option<&'planes [u8]> {
        if y >= self.raw.height {
            return None;
        }
        let start = to_usize(y).checked_mul(to_usize(self.raw.stride))?;
        self.samples().get(start..start.checked_add(to_usize(self.raw.width))?)
    }
}

/// `UNROUND_JPEGIO_ABI_VERSION` of the C layer that is linked.
#[must_use]
pub fn abi_version() -> i32 {
    ffi::unround_jpegio_abi_version()
}

/// The name and version of the libjpeg the C layer is built with.
#[must_use]
pub fn libjpeg_version() -> &'static str {
    let version = ffi::unround_jpegio_libjpeg_version();
    if version.is_null() {
        return "";
    }
    // SAFETY: the C layer returns a NUL-terminated string of static storage.
    let text = unsafe { CStr::from_ptr(version) };
    text.to_str().unwrap_or("")
}
