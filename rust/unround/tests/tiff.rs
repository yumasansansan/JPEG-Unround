// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! Binary64 output: TIFF files whose samples are the picture's to the last bit.
//!
//! A reader of baseline TIFF written here from the specification (TIFF 6.0: the
//! header, the IFD and its entries, the strip) reads the files back; the samples are
//! compared as the integers of their bits, so that -0.0 differs from 0.0 and nothing
//! rounds.

mod support;

use std::collections::BTreeMap;

use jpeg_unround::tiff;
use support::synthetic::Numbers;

/// The entries of the first IFD, by tag: their type, count, and the four bytes of
/// their value or offset.
struct Read {
    entries: BTreeMap<u16, (u16, u32, [u8; 4])>,
    bytes: Vec<u8>,
}

fn u16_at(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}

fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]])
}

fn index(value: u32) -> usize {
    usize::try_from(value).expect("an offset fits")
}

impl Read {
    fn new(bytes: Vec<u8>) -> Self {
        assert_eq!(&bytes[..4], b"II\x2a\x00", "little-endian TIFF");
        let ifd = index(u32_at(&bytes, 4));
        assert_eq!(ifd % 2, 0, "an IFD on a word boundary");
        let count = usize::from(u16_at(&bytes, ifd));
        let mut entries = BTreeMap::new();
        let mut previous = 0;
        for entry in 0..count {
            let at = ifd + 2 + 12 * entry;
            let tag = u16_at(&bytes, at);
            assert!(tag > previous, "the tags in ascending order");
            previous = tag;
            let value = [bytes[at + 8], bytes[at + 9], bytes[at + 10], bytes[at + 11]];
            entries.insert(tag, (u16_at(&bytes, at + 2), u32_at(&bytes, at + 4), value));
        }
        assert_eq!(u32_at(&bytes, ifd + 2 + 12 * count), 0, "one IFD");
        Self { entries, bytes }
    }

    /// The values of a SHORT or LONG entry.
    fn numbers(&self, tag: u16) -> Vec<u32> {
        let (kind, count, value) = self.entries[&tag];
        let count = index(count);
        let size = if kind == 3 { 2 } else { 4 };
        let source: Vec<u8> = if count * size <= 4 {
            value.to_vec()
        } else {
            let offset = index(u32::from_le_bytes(value));
            self.bytes[offset..offset + count * size].to_vec()
        };
        (0..count)
            .map(|item| {
                if kind == 3 {
                    u32::from(u16_at(&source, 2 * item))
                } else {
                    u32_at(&source, 4 * item)
                }
            })
            .collect()
    }

    /// The numerators and denominators of a RATIONAL entry.
    fn rationals(&self, tag: u16) -> Vec<u32> {
        let (kind, count, value) = self.entries[&tag];
        assert_eq!(kind, 5);
        let offset = index(u32::from_le_bytes(value));
        (0..2 * index(count))
            .map(|item| u32_at(&self.bytes, offset + 4 * item))
            .collect()
    }

    fn samples(&self) -> Vec<u64> {
        let offset = index(self.numbers(273)[0]);
        let length = index(self.numbers(279)[0]);
        self.bytes[offset..offset + length]
            .as_chunks::<8>()
            .0
            .iter()
            .map(|&chunk| u64::from_le_bytes(chunk))
            .collect()
    }
}

/// Random binary64 samples, with the values that a careless conversion would change.
fn awkward(count: usize) -> Vec<f64> {
    let mut numbers = Numbers::new(60);
    let mut samples: Vec<f64> = (0..count).map(|_| numbers.uniform(-10.0, 265.0)).collect();
    let special = [
        -0.0,
        0.0,
        f64::from_bits(1),
        -f64::from_bits(1),
        f64::MIN_POSITIVE,
        f64::MAX,
        1.0 / 3.0,
        255.0,
        128.0f64.next_up(),
    ];
    for (sample, &value) in samples.iter_mut().zip(&special) {
        *sample = value;
    }
    samples
}

#[test]
fn the_samples_are_written_bit_for_bit() {
    for (height, width, channels) in [(1, 1, 1), (7, 13, 1), (21, 30, 1), (5, 6, 3)] {
        let picture = awkward(height * width * channels);
        let file = Read::new(tiff::float64(&picture, height, width, channels, false).expect("a file"));
        let widths = u32::try_from(width).expect("fits");
        let heights = u32::try_from(height).expect("fits");
        let samples = u32::try_from(channels).expect("fits");
        assert_eq!(file.numbers(256), [widths]);
        assert_eq!(file.numbers(257), [heights]);
        assert_eq!(file.numbers(258), vec![64; channels]);
        assert_eq!(file.numbers(259), [1]);
        assert_eq!(file.numbers(262), [if channels == 3 { 2 } else { 1 }]);
        assert_eq!(file.numbers(277), [samples]);
        assert_eq!(file.numbers(278), [heights]);
        assert_eq!(file.numbers(284), [1]);
        assert_eq!(file.numbers(339), vec![3; channels]);
        let bits: Vec<u64> = picture.iter().map(|value| value.to_bits()).collect();
        assert_eq!(file.samples(), bits);
    }
}

#[test]
fn ycbcr_is_declared_as_it_is() {
    let picture = awkward(4 * 5 * 3);
    let file = Read::new(tiff::float64(&picture, 4, 5, 3, true).expect("a file"));
    assert_eq!(file.numbers(262), [6]);
    assert_eq!(file.numbers(530), [1, 1]);
    assert_eq!(file.numbers(531), [1]);
    assert_eq!(file.rationals(529), [299, 1000, 587, 1000, 114, 1000]);
    assert_eq!(file.rationals(532), [0, 1, 255, 1, 128, 1, 255, 1, 128, 1, 255, 1]);
    let bits: Vec<u64> = picture.iter().map(|value| value.to_bits()).collect();
    assert_eq!(file.samples(), bits);
    let error = tiff::float64(&picture[..20], 4, 5, 1, true).expect_err("refused");
    assert!(error.to_string().contains("three samples"), "{error}");
}

#[test]
fn other_shapes_are_refused() {
    for (samples, height, width, channels) in [(8, 2, 2, 2), (0, 0, 3, 1), (5, 2, 2, 1)] {
        let error = tiff::float64(&vec![0.0; samples], height, width, channels, false).expect_err("refused");
        assert!(error.to_string().contains("height x width"), "{error}");
    }
}
