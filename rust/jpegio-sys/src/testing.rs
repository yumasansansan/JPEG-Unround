// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! JPEG files for tests, written by libjpeg's own encoder from coefficients that a
//! test chooses (`test_jpeg` of the tests' support in C), and PNG files read back
//! by libpng (`test_png`). Nothing here is part of JPEG-Unround: the tests of this
//! crate and of the crates built on it use it, so that what a read returns can be
//! compared with what went in, and what is written with what comes back.

use std::ffi::{c_char, c_int, c_void};
use std::ptr;
use std::slice;

/// `test_jpeg` of the tests' support in C.
#[repr(C)]
struct Spec {
    width: u32,
    height: u32,
    color_space: i32,
    h_samp_factor: [i32; 4],
    v_samp_factor: [i32; 4],
    quant_table_slot: [i32; 4],
    quant_tables: [[u16; 64]; 4],
    coefficients: [*const i16; 4],
    data_precision: i32,
    progressive: i32,
    sequential_scans: i32,
    arithmetic: i32,
    optimize_coding: i32,
    restart_interval: i32,
    icc_profile: *const u8,
    icc_profile_size: u32,
    app1: *const u8,
    app1_size: u32,
}

unsafe extern "C" {
    fn test_jpeg_components(spec: *const Spec) -> i32;
    fn test_jpeg_blocks_wide(spec: *const Spec, component: i32) -> u32;
    fn test_jpeg_blocks_high(spec: *const Spec, component: i32) -> u32;
    fn test_jpeg_write(
        spec: *const Spec,
        data: *mut *mut u8,
        size: *mut usize,
        message: *mut c_char,
        message_size: usize,
    ) -> c_int;
    fn free(memory: *mut c_void);
    fn test_png_read(
        data: *const u8,
        size: usize,
        png: *mut TestPng,
        message: *mut c_char,
        message_size: usize,
    ) -> c_int;
    fn test_png_free(png: *mut TestPng);
}

/// `test_png` of the tests' support in C.
#[repr(C)]
struct TestPng {
    width: u32,
    height: u32,
    channels: i32,
    bits: i32,
    interlaced: i32,
    reserved: i32,
    samples: *mut c_void,
    icc_profile: *mut u8,
    icc_profile_size: u64,
    icc_name: [c_char; 80],
    warning: [c_char; 256],
}

const _: () = assert!(size_of::<TestPng>() == 384);

fn text_of(characters: &[c_char]) -> String {
    let bytes: Vec<u8> = characters
        .iter()
        .take_while(|&&c| c != 0)
        .map(|&c| c.to_ne_bytes()[0])
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// How the components of a test file are laid out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// One component.
    Grayscale,
    /// Y, Cb and Cr with the sampling factors (h, v) of each.
    YCbCr([(i32, i32); 3]),
    /// R, G and B, stored without a transform, each with the factors (1, 1).
    Rgb,
}

/// A test file to write: its size, its layout, the quantization table of each
/// component (natural order, 1 to 32767), and how it is coded.
#[derive(Debug, Clone)]
pub struct File {
    /// Samples across.
    pub width: u32,
    /// Rows.
    pub height: u32,
    /// The components and their sampling factors.
    pub layout: Layout,
    /// One table for each component, in natural order.
    pub quant_tables: Vec<[u16; 64]>,
    /// Whether the file is progressive.
    pub progressive: bool,
    /// Whether it is arithmetic-coded.
    pub arithmetic: bool,
    /// An ICC profile for its APP2 markers, or none where empty.
    pub icc_profile: Vec<u8>,
    /// The payload of an APP1 marker (EXIF), or none where empty.
    pub exif: Vec<u8>,
}

impl File {
    fn spec(&self) -> Result<Spec, String> {
        let components = match self.layout {
            Layout::Grayscale => 1,
            Layout::YCbCr(_) | Layout::Rgb => 3,
        };
        if self.quant_tables.len() != components {
            return Err(format!(
                "a quantization table for each of the {components} components, not {}",
                self.quant_tables.len()
            ));
        }
        let mut spec = Spec {
            width: self.width,
            height: self.height,
            color_space: match self.layout {
                Layout::Grayscale => crate::ffi::GRAYSCALE,
                Layout::YCbCr(_) => crate::ffi::YCBCR,
                Layout::Rgb => crate::ffi::RGB,
            },
            h_samp_factor: [0; 4],
            v_samp_factor: [0; 4],
            quant_table_slot: [0, 1, 2, 0],
            quant_tables: [[0; 64]; 4],
            coefficients: [ptr::null(); 4],
            data_precision: 0,
            progressive: i32::from(self.progressive),
            sequential_scans: 0,
            arithmetic: i32::from(self.arithmetic),
            optimize_coding: 0,
            restart_interval: 0,
            icc_profile: ptr::null(),
            icc_profile_size: 0,
            app1: ptr::null(),
            app1_size: 0,
        };
        if let Layout::YCbCr(factors) = self.layout {
            for (index, (h, v)) in factors.into_iter().enumerate() {
                spec.h_samp_factor[index] = h;
                spec.v_samp_factor[index] = v;
            }
        }
        for (slot, table) in spec.quant_tables.iter_mut().zip(&self.quant_tables) {
            *slot = *table;
        }
        Ok(spec)
    }

    /// The blocks of each component that carry image data, (across, down), as
    /// libjpeg counts them: the coefficients of component `c` are
    /// `across * down` blocks of 64.
    ///
    /// # Errors
    ///
    /// A description of what the file cannot be.
    pub fn blocks(&self) -> Result<Vec<(usize, usize)>, String> {
        let spec = self.spec()?;
        // SAFETY: spec is a valid test_jpeg for the length of the call.
        let count = unsafe { test_jpeg_components(&raw const spec) };
        (0..count)
            .map(|component| {
                // SAFETY: as above, and component is one of its components.
                let wide = unsafe { test_jpeg_blocks_wide(&raw const spec, component) };
                // SAFETY: as above.
                let high = unsafe { test_jpeg_blocks_high(&raw const spec, component) };
                let across = usize::try_from(wide).map_err(|error| error.to_string())?;
                let down = usize::try_from(high).map_err(|error| error.to_string())?;
                Ok((across, down))
            })
            .collect()
    }

    /// The file, with these coefficients: for each component, the blocks that
    /// [`File::blocks`] counts, block rows from the top, each 64 in natural order.
    ///
    /// # Errors
    ///
    /// A description of what the encoder could not write.
    pub fn write(&self, coefficients: &[Vec<i16>]) -> Result<Vec<u8>, String> {
        let blocks = self.blocks()?;
        if coefficients.len() != blocks.len() {
            return Err(format!(
                "coefficients for each of the {} components, not {}",
                blocks.len(),
                coefficients.len()
            ));
        }
        for (index, ((across, down), values)) in blocks.iter().zip(coefficients).enumerate() {
            if values.len() != across * down * 64 {
                return Err(format!(
                    "component {index} has {across} x {down} blocks of 64 coefficients, not {}",
                    values.len()
                ));
            }
        }
        let mut spec = self.spec()?;
        for (slot, values) in spec.coefficients.iter_mut().zip(coefficients) {
            *slot = values.as_ptr();
        }
        if !self.icc_profile.is_empty() {
            spec.icc_profile = self.icc_profile.as_ptr();
            spec.icc_profile_size = u32::try_from(self.icc_profile.len()).map_err(|error| error.to_string())?;
        }
        if !self.exif.is_empty() {
            spec.app1 = self.exif.as_ptr();
            spec.app1_size = u32::try_from(self.exif.len()).map_err(|error| error.to_string())?;
        }
        let mut data: *mut u8 = ptr::null_mut();
        let mut size = 0usize;
        let mut message: [c_char; 256] = [0; 256];
        // SAFETY: spec points to coefficients of the counts the encoder asks for, and
        // to the profile and the payload of their sizes, which live until the call
        // returns; data, size and message are valid for writes, message for 256 bytes.
        let result =
            unsafe { test_jpeg_write(&raw const spec, &raw mut data, &raw mut size, message.as_mut_ptr(), 256) };
        if result != 0 {
            return Err(text_of(&message));
        }
        // SAFETY: test_jpeg_write allocated size bytes at data with malloc().
        let bytes = unsafe { slice::from_raw_parts(data, size) }.to_vec();
        // SAFETY: data came from malloc(), and is freed once.
        unsafe { free(data.cast()) };
        Ok(bytes)
    }
}

/// A PNG file as libpng reads it, with its defaults but for pictures of any size
/// and chunks of any length.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadPng {
    /// Pixels across.
    pub width: u32,
    /// Rows.
    pub height: u32,
    /// Samples to a pixel: 1 (gray), 2 (gray and alpha), 3 (RGB) or 4 (RGBA).
    pub channels: u32,
    /// Bits of a sample.
    pub bits: u32,
    /// Whether the file is interlaced.
    pub interlaced: bool,
    /// The samples, rows from the top, the samples of a pixel together, each as the
    /// file holds it (0 to 255 for 8 bits).
    pub samples: Vec<u16>,
    /// The profile of the iCCP chunk.
    pub icc_profile: Option<Vec<u8>>,
    /// The name of that profile.
    pub icc_name: String,
    /// libpng's first warning in reading the file, or an empty string.
    pub warning: String,
}

/// Reads a PNG file with libpng; a palette file is refused.
///
/// # Errors
///
/// libpng's description of what it could not read.
pub fn read_png(data: &[u8]) -> Result<ReadPng, String> {
    let mut raw = TestPng {
        width: 0,
        height: 0,
        channels: 0,
        bits: 0,
        interlaced: 0,
        reserved: 0,
        samples: ptr::null_mut(),
        icc_profile: ptr::null_mut(),
        icc_profile_size: 0,
        icc_name: [0; 80],
        warning: [0; 256],
    };
    let mut message: [c_char; 256] = [0; 256];
    // SAFETY: data is valid for data.len() bytes, raw and message for writes,
    // message for 256 bytes, for the length of the call.
    let result = unsafe { test_png_read(data.as_ptr(), data.len(), &raw mut raw, message.as_mut_ptr(), 256) };
    if result != 0 {
        return Err(text_of(&message));
    }
    let count = usize::try_from(raw.width)
        .ok()
        .zip(usize::try_from(raw.height).ok())
        .zip(usize::try_from(raw.channels).ok())
        .and_then(|((width, height), channels)| width.checked_mul(height)?.checked_mul(channels));
    let samples = match (count, raw.bits) {
        (Some(count), 16) if !raw.samples.is_null() => {
            // SAFETY: test_png_read allocates this many 16-bit samples for 16 bits.
            unsafe { slice::from_raw_parts(raw.samples.cast::<u16>(), count) }.to_vec()
        }
        (Some(count), 1..=8) if !raw.samples.is_null() => {
            // SAFETY: and this many bytes, one to a sample, for 8 bits or fewer.
            let bytes = unsafe { slice::from_raw_parts(raw.samples.cast::<u8>(), count) };
            bytes.iter().map(|&byte| u16::from(byte)).collect()
        }
        _ => Vec::new(),
    };
    let icc_profile = match usize::try_from(raw.icc_profile_size) {
        Ok(size) if !raw.icc_profile.is_null() => {
            // SAFETY: test_png_read allocates the profile's bytes at icc_profile.
            Some(unsafe { slice::from_raw_parts(raw.icc_profile, size) }.to_vec())
        }
        _ => None,
    };
    let read = ReadPng {
        width: raw.width,
        height: raw.height,
        channels: u32::try_from(raw.channels).unwrap_or(0),
        bits: u32::try_from(raw.bits).unwrap_or(0),
        interlaced: raw.interlaced != 0,
        samples,
        icc_profile,
        icc_name: text_of(&raw.icc_name),
        warning: text_of(&raw.warning),
    };
    // SAFETY: raw is what the successful read filled in, freed once.
    unsafe { test_png_free(&raw mut raw) };
    Ok(read)
}
