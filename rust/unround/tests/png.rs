// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! PNG output: files that libpng reads back (`jpegio_sys::testing::read_png`) as
//! the samples that `eight_bits` and `sixteen_bits` round the picture to, at every
//! level of compression, with the ICC profile where it goes with the picture and a
//! warning where it does not.

mod support;

use jpeg_unround::Error;
use jpeg_unround::output::{self, Written};
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

#[test]
fn the_samples_are_those_of_eight_and_sixteen_bits() {
    for (height, width, channels) in [(1, 1, 1), (9, 14, 1), (7, 5, 3)] {
        let samples = picture(height * width * channels, 3);
        let eight: Vec<u16> = output::eight_bits(&samples)
            .expect("finite samples")
            .into_iter()
            .map(u16::from)
            .collect();
        let sixteen = output::sixteen_bits(&samples).expect("finite samples");
        for compression in [None, Some(0), Some(5), Some(9)] {
            let written = output::png(&samples, height, width, channels, false, compression, None).expect("a file");
            assert_eq!(written.warning, "");
            assert_eq!(read_back(&written, height, width, channels, 8), (eight.clone(), None));
            let written = output::png(&samples, height, width, channels, true, compression, None).expect("a file");
            assert_eq!(
                read_back(&written, height, width, channels, 16),
                (sixteen.clone(), None)
            );
        }
    }
}

#[test]
fn the_icc_profile_goes_where_it_goes_with_the_picture() {
    let rgb = metadata::profile(700, *b"RGB ", 3);
    let gray = metadata::profile(333, *b"GRAY", 4);
    for (channels, profile) in [(3, &rgb), (1, &gray)] {
        let samples = picture(4 * 6 * channels, 5);
        let written = output::png(&samples, 4, 6, channels, true, None, Some(profile)).expect("a file");
        assert_eq!(written.warning, "");
        assert_eq!(read_back(&written, 4, 6, channels, 16).1.as_ref(), Some(profile));
    }
    for (channels, profile) in [(1, &rgb), (3, &gray)] {
        let samples = picture(4 * 6 * channels, 5);
        let reason = jpegio_sys::check_icc(profile, u32::try_from(channels).expect("fits")).expect_err("refused");
        let written = output::png(&samples, 4, 6, channels, false, Some(9), Some(profile)).expect("a file");
        assert_eq!(written.warning, format!("the ICC profile is not written: {reason}"));
        assert_eq!(read_back(&written, 4, 6, channels, 8).1, None);
    }
}

#[test]
fn wrong_pictures_are_refused() {
    let samples = picture(12, 6);
    let refused = [
        output::png(&samples, 2, 2, 3, false, Some(10), None),
        output::png(&samples, 2, 3, 3, false, None, None),
        output::png(&samples, 6, 1, 2, false, None, None),
        output::png(&samples[..0], 0, 4, 3, false, None, None),
        output::png(&[f64::NAN; 3], 1, 1, 3, true, None, None),
    ];
    for result in refused {
        assert!(matches!(result, Err(Error::Options(_))), "{result:?}");
    }
}
