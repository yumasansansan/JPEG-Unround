// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! PNG output: files that libpng reads back (`jpegio_sys::testing::read_png`) as
//! the samples that `eight_bits` and `sixteen_bits` make of the picture, dithered or
//! rounded, at every level of compression, with the ICC profile where it goes with
//! the picture and a warning where it does not.

mod support;

use jpeg_unround::Error;
use jpeg_unround::output::{self, Depth, Dither, Written};
use jpegio_sys::testing::read_png;
use support::metadata;
use support::synthetic::Numbers;

/// Samples of a picture, with halves and values beyond 0-255 to round and clamp.
fn picture(count: usize, seed: u64) -> Vec<f64> {
    let mut numbers = Numbers::new(seed);
    let mut samples: Vec<f64> = (0..count).map(|_| numbers.uniform(-20.0, 275.0)).collect();
    for (sample, value) in samples
        .iter_mut()
        .zip([0.5, 254.5, -0.0, 1e300, 127.499_999_999_999_99])
    {
        *sample = value;
    }
    samples
}

/// The file read back: checks its size and depth, and returns its samples and profile.
fn read_back(
    written: &Written,
    height: usize,
    width: usize,
    channels: usize,
    bits: u32,
) -> (Vec<u16>, Option<Vec<u8>>) {
    let back = read_png(&written.bytes).expect("the file reads back");
    let size = |value: usize| u32::try_from(value).expect("fits");
    assert_eq!(
        (back.height, back.width, back.channels),
        (size(height), size(width), size(channels))
    );
    assert_eq!(back.bits, bits);
    assert!(!back.interlaced);
    assert_eq!(back.warning, "");
    (back.samples, back.icc_profile)
}

/// 8 bits, or 16, dithered as asked.
fn depth(sixteen: bool, dither: Dither) -> Depth {
    Depth { sixteen, dither }
}

#[test]
fn the_samples_are_those_of_eight_and_sixteen_bits() {
    for (height, width, channels) in [(1, 1, 1), (9, 14, 1), (7, 5, 3), (70, 66, 3)] {
        let samples = picture(height * width * channels, 3);
        for dither in [Dither::BlueNoise, Dither::BlueNoiseTogether, Dither::None] {
            let eight: Vec<u16> = output::eight_bits(&samples, height, width, channels, dither)
                .expect("finite samples")
                .into_iter()
                .map(u16::from)
                .collect();
            let sixteen = output::sixteen_bits(&samples, height, width, channels, dither).expect("finite samples");
            for compression in [None, Some(0), Some(9)] {
                let written = output::png(
                    &samples,
                    height,
                    width,
                    channels,
                    depth(false, dither),
                    compression,
                    None,
                )
                .expect("a file");
                assert_eq!(written.warning, "");
                assert_eq!(read_back(&written, height, width, channels, 8), (eight.clone(), None));
                let written = output::png(
                    &samples,
                    height,
                    width,
                    channels,
                    depth(true, dither),
                    compression,
                    None,
                )
                .expect("a file");
                assert_eq!(
                    read_back(&written, height, width, channels, 16),
                    (sixteen.clone(), None)
                );
            }
        }
    }
}

#[test]
fn a_dithered_tile_keeps_its_mean() {
    // Tiles of 64 x 64 pixels side by side, each of one value: a tile takes every
    // threshold of the matrix once, so that its levels go up at floor(4096 f + 1/2) of
    // its 4096 pixels, f the value's fraction (docs/math.md, 10), and their mean is the
    // value to 1/8192. Rounded, every pixel of a tile has the nearest level.
    let eighths: [u32; 8] = [0, 1, 2047, 2048, 4096, 5000, 8190, 8191];
    let (height, width) = (64, 64 * eighths.len());
    let value = |tile: usize| 37.0 + f64::from(eighths[tile]) / 8192.0;
    let samples: Vec<f64> = (0..height * width).map(|index| value(index % width / 64)).collect();
    let dithered = output::eight_bits(&samples, height, width, 1, Dither::BlueNoise).expect("a picture");
    let rounded = output::eight_bits(&samples, height, width, 1, Dither::None).expect("a picture");
    for (tile, &numerator) in eighths.iter().enumerate() {
        let levels = (0..height).flat_map(|row| (64 * tile..64 * tile + 64).map(move |column| row * width + column));
        // 37 + f rounds to 38 from f = 1/2 on, halves away from 0.
        let nearest = if numerator >= 4096 { 38 } else { 37 };
        let mut up = 0;
        for index in levels {
            up += usize::from(dithered[index] == 38);
            assert!(dithered[index] == 37 || dithered[index] == 38);
            assert_eq!(rounded[index], nearest, "tile {tile}");
        }
        assert_eq!(
            up,
            usize::try_from(numerator.div_ceil(2)).expect("small"),
            "tile {tile}"
        );
    }
}

#[test]
fn the_icc_profile_goes_where_it_goes_with_the_picture() {
    let rgb = metadata::profile(700, *b"RGB ", 3);
    let gray = metadata::profile(333, *b"GRAY", 4);
    for (channels, profile) in [(3, &rgb), (1, &gray)] {
        let samples = picture(4 * 6 * channels, 5);
        let written = output::png(
            &samples,
            4,
            6,
            channels,
            depth(true, Dither::BlueNoise),
            None,
            Some(profile),
        )
        .expect("a file");
        assert_eq!(written.warning, "");
        assert_eq!(read_back(&written, 4, 6, channels, 16).1.as_ref(), Some(profile));
    }
    for (channels, profile) in [(1, &rgb), (3, &gray)] {
        let samples = picture(4 * 6 * channels, 5);
        let reason = jpegio_sys::check_icc(profile, u32::try_from(channels).expect("fits")).expect_err("refused");
        let written = output::png(&samples, 4, 6, channels, Depth::default(), Some(9), Some(profile)).expect("a file");
        assert_eq!(written.warning, format!("the ICC profile is not written: {reason}"));
        assert_eq!(read_back(&written, 4, 6, channels, 8).1, None);
    }
}

#[test]
fn wrong_pictures_are_refused() {
    let samples = picture(12, 6);
    let refused = [
        output::png(&samples, 2, 2, 3, Depth::default(), Some(10), None),
        output::png(&samples, 2, 3, 3, Depth::default(), None, None),
        output::png(&samples[..10], 2, 1, 5, Depth::default(), None, None),
        output::png(&samples[..0], 0, 4, 3, Depth::default(), None, None),
        output::png(&[f64::NAN; 3], 1, 1, 3, depth(true, Dither::None), None, None),
    ];
    for result in refused {
        assert!(matches!(result, Err(Error::Options(_))), "{result:?}");
    }
}
