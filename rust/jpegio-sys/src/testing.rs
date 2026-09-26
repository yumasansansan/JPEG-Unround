// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! JPEG files for tests, written by libjpeg's own encoder from coefficients that a
//! test chooses (`test_jpeg` of the tests' support in C). Nothing here is part of
//! JPEG-Unround: the tests of this crate and of the crates built on it use it, so
//! that what a read returns can be compared with what went in.

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
        let mut data: *mut u8 = ptr::null_mut();
        let mut size = 0usize;
        let mut message: [c_char; 256] = [0; 256];
        // SAFETY: spec points to coefficients of the counts the encoder asks for,
        // which live until the call returns; data, size and message are valid for
        // writes, message for 256 bytes.
        let result =
            unsafe { test_jpeg_write(&raw const spec, &raw mut data, &raw mut size, message.as_mut_ptr(), 256) };
        if result != 0 {
            let bytes: Vec<u8> = message
                .iter()
                .take_while(|&&c| c != 0)
                .map(|&c| c.to_ne_bytes()[0])
                .collect();
            return Err(String::from_utf8_lossy(&bytes).into_owned());
        }
        // SAFETY: test_jpeg_write allocated size bytes at data with malloc().
        let bytes = unsafe { slice::from_raw_parts(data, size) }.to_vec();
        // SAFETY: data came from malloc(), and is freed once.
        unsafe { free(data.cast()) };
        Ok(bytes)
    }
}
