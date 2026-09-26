// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! Tests of the C layer's writing of PNG files as Rust calls it. Pictures of every
//! kind it writes are read back by libpng (`testing::read_png`), and what comes
//! back has to be what went in, exactly, with no warning. An ICC profile has to
//! come back where the C layer's check takes it, and be left out, with a warning,
//! where the check refuses it. Then the refusals of wrong pictures, and threads
//! that write at once.

use std::thread;

use jpegio_sys::testing::read_png;
use jpegio_sys::{ErrorKind, Png, Samples, check_icc, write_png};

/// A deterministic stream of numbers, the `SplitMix64` generator.
struct Numbers(u64);

impl Numbers {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn bytes(&mut self, count: usize) -> Vec<u8> {
        (0..count).map(|_| self.next().to_le_bytes()[0]).collect()
    }

    fn words(&mut self, count: usize) -> Vec<u16> {
        (0..count)
            .map(|_| {
                let [low, high, ..] = self.next().to_le_bytes();
                u16::from_le_bytes([low, high])
            })
            .collect()
    }
}

/// A well-formed ICC profile of version 2 and the size given, a monitor's, of the
/// data color space given: its header, one tag, and the tag's data, of random
/// bytes.
fn profile(size: usize, space: [u8; 4], numbers: &mut Numbers) -> Vec<u8> {
    let mut bytes = numbers.bytes(size);
    bytes[..132].fill(0);
    let put = |bytes: &mut Vec<u8>, at: usize, value: u32| bytes[at..at + 4].copy_from_slice(&value.to_be_bytes());
    put(&mut bytes, 0, u32::try_from(size).expect("a small profile"));
    bytes[8] = 2;
    bytes[12..16].copy_from_slice(b"mntr");
    bytes[16..20].copy_from_slice(&space);
    bytes[20..24].copy_from_slice(b"XYZ ");
    bytes[36..40].copy_from_slice(b"acsp");
    put(&mut bytes, 68, 0x0000_F6D6); // the illuminant D50
    put(&mut bytes, 72, 0x0001_0000);
    put(&mut bytes, 76, 0x0000_D32D);
    put(&mut bytes, 128, 1);
    bytes[132..136].copy_from_slice(b"desc");
    put(&mut bytes, 136, 144);
    put(&mut bytes, 140, u32::try_from(size - 144).expect("a small profile"));
    bytes
}

fn widened(samples: Samples<'_>) -> Vec<u16> {
    match samples {
        Samples::Eight(samples) => samples.iter().map(|&sample| u16::from(sample)).collect(),
        Samples::Sixteen(samples) => samples.to_vec(),
    }
}

/// Writes a picture, reads it back, and checks that it is what went in; returns the
/// warning of the writing and the profile read back.
fn round_trip(png: &Png<'_>) -> (String, Option<Vec<u8>>) {
    let written = write_png(png).expect("the picture is written");
    let back = read_png(&written.bytes).expect("the file reads back");
    assert_eq!(
        (back.width, back.height, back.channels),
        (png.width, png.height, png.channels)
    );
    let bits = match png.samples {
        Samples::Eight(_) => 8,
        Samples::Sixteen(_) => 16,
    };
    assert_eq!(back.bits, bits);
    assert!(!back.interlaced);
    assert_eq!(back.samples, widened(png.samples));
    assert_eq!(back.warning, "", "libpng warns of nothing in reading the file");
    if back.icc_profile.is_some() {
        assert_eq!(back.icc_name, "ICC profile");
    }
    (written.warning, back.icc_profile)
}

#[test]
fn pictures_read_back_as_written() {
    let mut numbers = Numbers(1);
    for channels in [1u32, 3] {
        for (width, height) in [(1u32, 1u32), (5, 3), (64, 17), (3, 70)] {
            for compression in [None, Some(0), Some(1), Some(9)] {
                let count = usize::try_from(width * height * channels).expect("fits");
                let eight = numbers.bytes(count);
                let sixteen = numbers.words(count);
                for samples in [Samples::Eight(&eight), Samples::Sixteen(&sixteen)] {
                    let png = Png {
                        width,
                        height,
                        channels,
                        samples,
                        compression,
                        icc_profile: None,
                    };
                    let (warning, profile) = round_trip(&png);
                    assert_eq!(warning, "");
                    assert_eq!(profile, None);
                }
            }
        }
    }
}

#[test]
fn the_icc_profile_goes_with_the_picture() {
    let mut numbers = Numbers(2);
    let rgb = profile(600, *b"RGB ", &mut numbers);
    let gray = profile(400, *b"GRAY", &mut numbers);
    assert_eq!(check_icc(&rgb, 3), Ok(()));
    assert_eq!(check_icc(&gray, 1), Ok(()));
    let eight = numbers.bytes(4 * 3 * 3);
    for (channels, icc) in [(3u32, &rgb), (1, &gray)] {
        let count = usize::try_from(4 * 3 * channels).expect("fits");
        let png = Png {
            width: 4,
            height: 3,
            channels,
            samples: Samples::Eight(&eight[..count]),
            compression: None,
            icc_profile: Some(icc),
        };
        let (warning, back) = round_trip(&png);
        assert_eq!(warning, "");
        assert_eq!(back.as_ref(), Some(icc));
    }

    // A profile of another color space than the picture's, and one too short, are
    // left out with the check's reason.
    let reason = check_icc(&rgb, 1).expect_err("an RGB profile does not go with gray");
    assert!(reason.contains("'RGB '"), "{reason}");
    let refused: [(&[u8], u32); 3] = [(&rgb, 1), (&gray, 3), (&rgb[..100], 3)];
    for (icc, channels) in refused {
        let count = usize::try_from(4 * 3 * channels).expect("fits");
        let png = Png {
            width: 4,
            height: 3,
            channels,
            samples: Samples::Eight(&eight[..count]),
            compression: Some(3),
            icc_profile: Some(icc),
        };
        let reason = check_icc(icc, channels).expect_err("the check refuses the profile");
        let (warning, back) = round_trip(&png);
        assert_eq!(warning, format!("the ICC profile is not written: {reason}"));
        assert_eq!(back, None);
    }
}

#[test]
fn wrong_pictures_are_refused() {
    let samples = [0u8; 12];
    let good = Png {
        width: 2,
        height: 2,
        channels: 3,
        samples: Samples::Eight(&samples),
        compression: None,
        icc_profile: None,
    };
    assert!(write_png(&good).is_ok());
    let wrong = [
        Png { width: 3, ..good },
        Png {
            samples: Samples::Eight(&samples[..11]),
            ..good
        },
        Png {
            samples: Samples::Sixteen(&[0; 11]),
            ..good
        },
        Png {
            width: 0,
            height: 5,
            ..good
        },
        Png {
            channels: 2,
            width: 3,
            ..good
        },
        Png {
            channels: 4,
            width: 3,
            height: 1,
            ..good
        },
        Png {
            compression: Some(10),
            ..good
        },
    ];
    for png in wrong {
        let error = write_png(&png).expect_err("the picture is refused");
        assert_eq!(error.kind(), ErrorKind::Argument, "{error}");
        assert!(!error.message().is_empty());
    }
}

#[test]
fn threads_write_at_once() {
    let mut numbers = Numbers(3);
    let pictures: Vec<Vec<u16>> = (0..4).map(|_| numbers.words(40 * 30 * 3)).collect();
    thread::scope(|scope| {
        for thread_index in 0..8 {
            let pictures = &pictures;
            scope.spawn(move || {
                for round in 0..10 {
                    let samples = &pictures[(thread_index + round) % pictures.len()];
                    let png = Png {
                        width: 40,
                        height: 30,
                        channels: 3,
                        samples: Samples::Sixteen(samples),
                        compression: Some(u8::try_from(round % 10).expect("a level")),
                        icc_profile: None,
                    };
                    round_trip(&png);
                }
            });
        }
    });
}

#[test]
fn the_linked_libpng_is_the_one_declared() {
    assert_eq!(jpegio_sys::libpng_version(), "libpng 1.6.58, zlib-ng 2.3.3");
}
