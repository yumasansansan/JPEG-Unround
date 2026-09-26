// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! Pictures in binary64, written as TIFF with 64-bit IEEE floating-point samples.
//!
//! The samples go into the file as they are, bit for bit: no conversion, no scaling
//! (they are in the units of the samples, 0 to 255 for 8-bit JPEG, not normalized
//! to 0-1), and no compression that could change them. The file is baseline TIFF
//! 6.0 in little-endian byte order, one strip, with `SampleFormat` 3 (IEEE floating
//! point) and `BitsPerSample` 64.
//!
//! The three samples of a colour picture are R, G and B, or JFIF's Y, Cb and Cr,
//! which the file then declares: `PhotometricInterpretation` `YCbCr`, without
//! subsampling, with JFIF's coefficients and full range (docs/math.md, 8).
//!
//! A classic TIFF addresses at most 4 GiB, which is a little under 2^29 greyscale
//! samples in binary64.

use std::path::Path;

use crate::error::Error;

const SHORT: u16 = 3;
const LONG: u16 = 4;
const RATIONAL: u16 = 5;
const HEADER: usize = 8;
const ENTRY: usize = 12;
const IFD: usize = 2 + 4; // the count of entries and the next IFD's offset, besides the entries
const LIMIT: usize = 1 << 32;
const BITS: u16 = 64;
const IEEE: u16 = 3; // SampleFormat: IEEE floating point
const BLACK_IS_ZERO: u16 = 1;
const RGB: u16 = 2;
const YCBCR: u16 = 6;
const JFIF_COEFFICIENTS: [u32; 6] = [299, 1000, 587, 1000, 114, 1000]; // K_R, K_G and K_B, as rationals
const FULL_RANGE: [u32; 12] = [0, 1, 255, 1, 128, 1, 255, 1, 128, 1, 255, 1]; // black and white of Y, Cb and Cr

/// An IFD entry whose value, of at most four bytes, is held in the entry itself.
fn entry(tag: u16, kind: u16, count: u32, value: &[u8]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(ENTRY);
    bytes.extend_from_slice(&tag.to_le_bytes());
    bytes.extend_from_slice(&kind.to_le_bytes());
    bytes.extend_from_slice(&count.to_le_bytes());
    let mut held = [0u8; 4];
    held[..value.len()].copy_from_slice(value);
    bytes.extend_from_slice(&held);
    bytes
}

fn offset(value: usize) -> Result<[u8; 4], Error> {
    u32::try_from(value)
        .map(u32::to_le_bytes)
        .map_err(|_| Error::Options(format!("an offset of {value} does not fit a classic TIFF")))
}

fn shorts(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
}

fn longs(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
}

/// Where the values that do not fit in an entry lie, and what the entries of the
/// file are.
struct Layout {
    values: Vec<u8>,
    ifd_offset: usize,
    entries: Vec<Vec<u8>>,
}

fn layout(height: usize, width: usize, channels: usize, ycbcr: bool) -> Result<Layout, Error> {
    let size = |value: usize| u32::try_from(value).map_err(|_| Error::Options(format!("{value} does not fit a TIFF")));
    let data_length = height * width * channels * 8;
    // The layout: the header, the image data, the values that do not fit in an entry,
    // and the IFD. Entries are in ascending order of their tags, as TIFF requires.
    let several = channels > 1;
    let count = u16::try_from(channels).unwrap_or(u16::MAX);
    let bits = if several {
        shorts(&vec![BITS; channels])
    } else {
        Vec::new()
    };
    let formats = if several {
        shorts(&vec![IEEE; channels])
    } else {
        Vec::new()
    };
    let resolution = longs(&[1, 1]);
    let coefficients = if ycbcr { longs(&JFIF_COEFFICIENTS) } else { Vec::new() };
    let reference = if ycbcr { longs(&FULL_RANGE) } else { Vec::new() };
    let bits_offset = HEADER + data_length;
    let format_offset = bits_offset + bits.len();
    let resolution_offset = format_offset + formats.len();
    let coefficients_offset = resolution_offset + resolution.len();
    let reference_offset = coefficients_offset + coefficients.len();
    let values_end = reference_offset + reference.len();
    let ifd_offset = values_end + values_end % 2; // an IFD starts on a word boundary
    let photometric = if ycbcr {
        YCBCR
    } else if several {
        RGB
    } else {
        BLACK_IS_ZERO
    };
    let one = shorts(&[1]);
    let mut entries = vec![
        entry(256, LONG, 1, &size(width)?.to_le_bytes()),  // ImageWidth
        entry(257, LONG, 1, &size(height)?.to_le_bytes()), // ImageLength
        entry(
            258,
            SHORT,
            u32::from(count),
            &if several {
                offset(bits_offset)?.to_vec()
            } else {
                shorts(&[BITS])
            },
        ),
        entry(259, SHORT, 1, &one),                             // Compression: none
        entry(262, SHORT, 1, &shorts(&[photometric])),          // PhotometricInterpretation
        entry(273, LONG, 1, &offset(HEADER)?),                  // StripOffsets
        entry(277, SHORT, 1, &shorts(&[count])),                // SamplesPerPixel
        entry(278, LONG, 1, &size(height)?.to_le_bytes()),      // RowsPerStrip: one strip
        entry(279, LONG, 1, &size(data_length)?.to_le_bytes()), // StripByteCounts
        entry(282, RATIONAL, 1, &offset(resolution_offset)?),   // XResolution: 1/1
        entry(283, RATIONAL, 1, &offset(resolution_offset)?),   // YResolution: 1/1
        entry(284, SHORT, 1, &one),                             // PlanarConfiguration: chunky
        entry(296, SHORT, 1, &one),                             // ResolutionUnit: none
        entry(
            339,
            SHORT,
            u32::from(count),
            &if several {
                offset(format_offset)?.to_vec()
            } else {
                shorts(&[IEEE])
            },
        ),
    ];
    if ycbcr {
        entries.push(entry(529, RATIONAL, 3, &offset(coefficients_offset)?)); // YCbCrCoefficients: JFIF's
        entries.push(entry(530, SHORT, 2, &shorts(&[1, 1]))); // YCbCrSubSampling: none
        entries.push(entry(531, SHORT, 1, &one)); // YCbCrPositioning: centred
        entries.push(entry(532, RATIONAL, 6, &offset(reference_offset)?)); // ReferenceBlackWhite: full range
    }
    let mut values = Vec::new();
    for part in [bits, formats, resolution, coefficients, reference] {
        values.extend(part);
    }
    Ok(Layout {
        values,
        ifd_offset,
        entries,
    })
}

/// The bytes of a TIFF of binary64 samples: `height x width` pixels of `channels`
/// samples each (1, or 3 interleaved), exactly as they are. With `ycbcr`, the three
/// samples of a pixel are JFIF's Y, Cb and Cr, and the file says so.
///
/// # Errors
///
/// [`Error::Options`] for a picture that is empty, not of that size, not of 1 or 3
/// channels, of Y, Cb and Cr in other than 3 channels, or too large for a classic
/// TIFF.
pub fn float64(samples: &[f64], height: usize, width: usize, channels: usize, ycbcr: bool) -> Result<Vec<u8>, Error> {
    if height == 0 || width == 0 || !(channels == 1 || channels == 3) || samples.len() != height * width * channels {
        return Err(Error::Options(format!(
            "a picture is height x width x 1 or 3 samples, and not empty: {height} x {width} x {channels}, \
             {} samples",
            samples.len()
        )));
    }
    if ycbcr && channels != 3 {
        return Err(Error::Options(format!(
            "Y, Cb and Cr are three samples to a pixel, not {channels}"
        )));
    }
    let Layout {
        values,
        ifd_offset,
        entries,
    } = layout(height, width, channels, ycbcr)?;
    if ifd_offset + IFD + ENTRY * entries.len() > LIMIT {
        return Err(Error::Options(format!(
            "a picture of {height} x {width} x {channels} binary64 samples does not fit a classic TIFF"
        )));
    }
    let mut file = Vec::with_capacity(ifd_offset + IFD + ENTRY * entries.len());
    file.extend_from_slice(b"II");
    file.extend_from_slice(&42u16.to_le_bytes());
    file.extend_from_slice(&offset(ifd_offset)?);
    for &sample in samples {
        file.extend_from_slice(&sample.to_le_bytes());
    }
    file.extend_from_slice(&values);
    file.resize(ifd_offset, 0);
    file.extend_from_slice(&u16::try_from(entries.len()).unwrap_or(u16::MAX).to_le_bytes());
    for bytes in &entries {
        file.extend_from_slice(bytes);
    }
    file.extend_from_slice(&0u32.to_le_bytes());
    Ok(file)
}

/// Writes [`float64`] of a picture to a file.
///
/// # Errors
///
/// As [`float64`], and [`Error::Io`] where the file cannot be written.
pub fn write_float64(
    path: &Path,
    samples: &[f64],
    height: usize,
    width: usize,
    channels: usize,
    ycbcr: bool,
) -> Result<(), Error> {
    let bytes = float64(samples, height, width, channels, ycbcr)?;
    std::fs::write(path, bytes)?;
    Ok(())
}
