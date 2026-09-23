// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! The structures and functions of `unround/jpegio.h`, as C declares them.
//!
//! The layouts are asserted below with the sizes and offsets the C layer has on
//! the 64-bit systems JPEG-Unround supports; the Python implementation checks the
//! same numbers. A status and a color space are plain integers here rather than
//! Rust enums, since a value C might return outside an enum would make one
//! undefined.

use std::ffi::c_char;
use std::mem::{offset_of, size_of};
use std::ptr;

/// `UNROUND_JPEGIO_ABI_VERSION` of the header these declarations follow.
pub const ABI_VERSION: i32 = 1;
/// `UNROUND_JPEGIO_MAX_COMPONENTS`.
pub const MAX_COMPONENTS: usize = 4;
/// `UNROUND_JPEGIO_BLOCK_SIZE`: the coefficients of a block.
pub const BLOCK_SIZE: usize = 64;

/// `unround_jpegio_status`.
pub type Status = i32;
/// `UNROUND_JPEGIO_OK`.
pub const OK: Status = 0;
/// `UNROUND_JPEGIO_ERROR_ARGUMENT`: a null pointer, or a size the layer cannot pass on.
pub const ERROR_ARGUMENT: Status = 1;
/// `UNROUND_JPEGIO_ERROR_DECODE`: libjpeg could not read the data.
pub const ERROR_DECODE: Status = 2;
/// `UNROUND_JPEGIO_ERROR_UNSUPPORTED`: a JPEG outside the project's scope.
pub const ERROR_UNSUPPORTED: Status = 3;
/// `UNROUND_JPEGIO_ERROR_LIMIT`: larger than the options allow.
pub const ERROR_LIMIT: Status = 4;
/// `UNROUND_JPEGIO_ERROR_MEMORY`: an allocation failed.
pub const ERROR_MEMORY: Status = 5;

/// `UNROUND_JPEGIO_GRAYSCALE`.
pub const GRAYSCALE: i32 = 1;
/// `UNROUND_JPEGIO_YCBCR`.
pub const YCBCR: i32 = 2;
/// `UNROUND_JPEGIO_RGB`: three components stored without a color transform.
pub const RGB: i32 = 3;

/// `unround_jpegio_options`: the limits a read accepts.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Options {
    /// width * height; 0 means 1 << 28.
    pub max_pixels: u64,
    /// SOS markers of a progressive file; 0 means 500.
    pub max_scans: i32,
    /// Nonzero: a corrupt-data warning fails the read.
    pub warnings_are_errors: i32,
}

/// `unround_jpegio_component`: one component of a file.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Component {
    /// The component identifier of the SOF marker.
    pub id: i32,
    /// 1 to 4.
    pub h_samp_factor: i32,
    /// 1 to 4.
    pub v_samp_factor: i32,
    /// 0 to 3: the DQT slot the SOF marker names for the component.
    pub quant_table_slot: i32,
    /// Samples of this component that cover the image.
    pub width: u32,
    /// Rows of this component that cover the image.
    pub height: u32,
    /// ceil(width / 8): the blocks that carry image data.
    pub width_in_blocks: u32,
    /// ceil(height / 8).
    pub height_in_blocks: u32,
    /// The quantization table the component was quantized with, in natural order.
    pub quant_table: [u16; BLOCK_SIZE],
    /// `height_in_blocks * width_in_blocks` blocks of 64 coefficients in natural order.
    pub coefficients: *mut i16,
}

/// `unround_jpegio_image`: what a read returns.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Image {
    /// Samples across.
    pub width: u32,
    /// Rows.
    pub height: u32,
    /// `GRAYSCALE`, `YCBCR` or `RGB`.
    pub color_space: i32,
    /// 1 or 3.
    pub num_components: i32,
    /// The largest horizontal sampling factor.
    pub max_h_samp_factor: i32,
    /// The largest vertical sampling factor.
    pub max_v_samp_factor: i32,
    /// 1 when the file is progressive.
    pub progressive: i32,
    /// 1 when the file is arithmetic-coded.
    pub arithmetic: i32,
    /// 1 to 8 from the EXIF APP1 marker, or 0 when there is none.
    pub exif_orientation: i32,
    /// Corrupt-data warnings libjpeg reported while reading.
    pub warnings: i32,
    /// The components; the first `num_components` of them are filled in.
    pub components: [Component; MAX_COMPONENTS],
    /// The ICC profile of the APP2 markers, or a null pointer.
    pub icc_profile: *mut u8,
    /// The bytes of the ICC profile.
    pub icc_profile_size: u64,
}

/// `unround_jpegio_plane`: one component decoded to 8-bit samples.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Plane {
    /// `width_in_blocks * 8`.
    pub width: u32,
    /// `height_in_blocks * 8`.
    pub height: u32,
    /// Samples from one row to the next.
    pub stride: u32,
    /// Unused.
    pub reserved: u32,
    /// `stride * height` samples.
    pub samples: *mut u8,
}

/// `unround_jpegio_planes`: what a decoding of the planes returns.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Planes {
    /// 1 or 3.
    pub num_components: i32,
    /// Unused.
    pub reserved: i32,
    /// The planes; the first `num_components` of them are filled in.
    pub planes: [Plane; MAX_COMPONENTS],
}

impl Component {
    /// A component with nothing in it, as the C layer leaves one it does not fill in.
    pub const EMPTY: Self = Self {
        id: 0,
        h_samp_factor: 0,
        v_samp_factor: 0,
        quant_table_slot: 0,
        width: 0,
        height: 0,
        width_in_blocks: 0,
        height_in_blocks: 0,
        quant_table: [0; BLOCK_SIZE],
        coefficients: ptr::null_mut(),
    };
}

impl Image {
    /// An image with nothing in it, as the C layer leaves one after a failure or a free.
    pub const EMPTY: Self = Self {
        width: 0,
        height: 0,
        color_space: 0,
        num_components: 0,
        max_h_samp_factor: 0,
        max_v_samp_factor: 0,
        progressive: 0,
        arithmetic: 0,
        exif_orientation: 0,
        warnings: 0,
        components: [Component::EMPTY; MAX_COMPONENTS],
        icc_profile: ptr::null_mut(),
        icc_profile_size: 0,
    };
}

impl Plane {
    /// A plane with nothing in it.
    pub const EMPTY: Self = Self {
        width: 0,
        height: 0,
        stride: 0,
        reserved: 0,
        samples: ptr::null_mut(),
    };
}

impl Planes {
    /// Planes with nothing in them, as the C layer leaves them after a failure or a free.
    pub const EMPTY: Self = Self {
        num_components: 0,
        reserved: 0,
        planes: [Plane::EMPTY; MAX_COMPONENTS],
    };
}

const _: () = assert!(size_of::<usize>() == 8, "JPEG-Unround supports 64-bit systems");
const _: () = assert!(size_of::<Options>() == 16);
const _: () = assert!(size_of::<Component>() == 168);
const _: () = assert!(offset_of!(Component, quant_table) == 32);
const _: () = assert!(offset_of!(Component, coefficients) == 160);
const _: () = assert!(size_of::<Image>() == 728);
const _: () = assert!(offset_of!(Image, components) == 40);
const _: () = assert!(offset_of!(Image, icc_profile) == 712);
const _: () = assert!(size_of::<Plane>() == 24);
const _: () = assert!(size_of::<Planes>() == 104);

unsafe extern "C" {
    /// `UNROUND_JPEGIO_ABI_VERSION` of the library that is linked.
    pub safe fn unround_jpegio_abi_version() -> i32;

    /// The name and version of the libjpeg the C layer is built with, as a
    /// NUL-terminated string of static storage.
    pub safe fn unround_jpegio_libjpeg_version() -> *const c_char;

    /// Reads the coefficients and the metadata of a JPEG file in memory.
    ///
    /// # Safety
    ///
    /// `data` is valid for reads of `size` bytes (or null when `size` is 0),
    /// `options` is null or valid, `image` is valid for writes, and `message` is
    /// valid for writes of `message_size` bytes (or null when that is 0). On
    /// success, `image` owns memory that only `unround_jpegio_image_free` may free.
    pub fn unround_jpegio_read(
        data: *const u8,
        size: usize,
        options: *const Options,
        image: *mut Image,
        message: *mut c_char,
        message_size: usize,
    ) -> Status;

    /// Frees what a successful read allocated, and empties the image.
    ///
    /// # Safety
    ///
    /// `image` is null, or a valid image that a read filled in or that is empty.
    pub fn unround_jpegio_image_free(image: *mut Image);

    /// Decodes the component planes of a JPEG file in memory.
    ///
    /// # Safety
    ///
    /// As for `unround_jpegio_read`, with `planes` in place of `image`.
    pub fn unround_jpegio_decode_planes(
        data: *const u8,
        size: usize,
        options: *const Options,
        planes: *mut Planes,
        message: *mut c_char,
        message_size: usize,
    ) -> Status;

    /// Frees what a successful decoding allocated, and empties the planes.
    ///
    /// # Safety
    ///
    /// `planes` is null, or valid planes that a decoding filled in or that are empty.
    pub fn unround_jpegio_planes_free(planes: *mut Planes);
}
