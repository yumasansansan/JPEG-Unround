// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! The EXIF orientation of a picture: how the picture as the file stores it is
//! turned and mirrored to show it upright (EXIF's tag `Orientation`, 274).
//!
//! Value `o` says where the stored picture's first row and first column are to be
//! seen: 1 top and left (as stored), 2 top and right, 3 bottom and right, 4 bottom
//! and left, 5 left and top, 6 right and top, 7 right and bottom, 8 left and
//! bottom. From 5 on, the rows become columns, and a picture of `h x w` is shown as
//! `w x h`. Pixel `(r, c)` of what is shown, of the stored picture `S` of `h` rows
//! and `w` columns, is:
//!
//! | `o` | shown | `o` | shown |
//! |---|---|---|---|
//! | 1 | `S[r][c]` | 5 | `S[c][r]` |
//! | 2 | `S[r][w-1-c]` | 6 | `S[h-1-c][r]` |
//! | 3 | `S[h-1-r][w-1-c]` | 7 | `S[h-1-c][w-1-r]` |
//! | 4 | `S[h-1-r][c]` | 8 | `S[c][w-1-r]` |
//!
//! Turning the picture moves its pixels and changes none: every sample of the
//! result is one of the picture's, bit for bit.

use crate::error::Error;

/// A picture turned upright: its samples, interleaved as the picture's are, and its
/// rows and columns.
#[derive(Debug, Clone, PartialEq)]
pub struct Oriented<T> {
    /// `height x width x channels` samples.
    pub samples: Vec<T>,
    /// Rows.
    pub height: usize,
    /// Columns.
    pub width: usize,
}

/// Whether an orientation turns the rows into columns (5 to 8).
#[must_use]
pub fn transposes(orientation: i32) -> bool {
    (5..=8).contains(&orientation)
}

/// The side of the square tiles that a picture whose rows become columns is turned
/// in, so that what is read and what is written stay in the caches.
const TILE: usize = 64;

/// The picture of `height x width` pixels of `channels` interleaved samples, turned
/// and mirrored as EXIF's `orientation` says to show it: 1 to 8, or 0 where the file
/// has none, which is 1.
///
/// # Errors
///
/// [`Error::Options`] for an orientation other than 0 to 8, or a picture that is not
/// of that size.
pub fn oriented<T: Copy>(
    samples: &[T],
    height: usize,
    width: usize,
    channels: usize,
    orientation: i32,
) -> Result<Oriented<T>, Error> {
    if !(0..=8).contains(&orientation) {
        return Err(Error::Options(format!(
            "an EXIF orientation is 1 to 8, or 0 for none, not {orientation}"
        )));
    }
    let count = height
        .checked_mul(width)
        .and_then(|pixels| pixels.checked_mul(channels));
    if channels == 0 || count != Some(samples.len()) {
        return Err(Error::Options(format!(
            "a picture of {height} x {width} pixels of {channels} samples, not of {} samples",
            samples.len()
        )));
    }
    if samples.is_empty() {
        let (height, width) = if transposes(orientation) {
            (width, height)
        } else {
            (height, width)
        };
        return Ok(Oriented {
            samples: Vec::new(),
            height,
            width,
        });
    }
    let row = width * channels;
    if !transposes(orientation) {
        // The stored rows, each forwards or backwards, from the top or the bottom.
        let (up, back) = match orientation {
            2 => (false, true),
            3 => (true, true),
            4 => (true, false),
            _ => (false, false),
        };
        let mut turned = Vec::with_capacity(samples.len());
        for r in 0..height {
            let source = if up { height - 1 - r } else { r };
            let pixels = &samples[source * row..(source + 1) * row];
            if back {
                for pixel in pixels.chunks_exact(channels).rev() {
                    turned.extend_from_slice(pixel);
                }
            } else {
                turned.extend_from_slice(pixels);
            }
        }
        return Ok(Oriented {
            samples: turned,
            height,
            width,
        });
    }
    // Row r of the result is column r or w-1-r of the stored picture, read from
    // its top or its bottom: pixel (r, c) is stored at (c or h-1-c, r or w-1-r).
    let (up, back) = match orientation {
        6 => (true, false),
        7 => (true, true),
        8 => (false, true),
        _ => (false, false),
    };
    let (rows, columns) = (width, height);
    let mut turned = vec![samples[0]; samples.len()];
    for r0 in (0..rows).step_by(TILE) {
        for c0 in (0..columns).step_by(TILE) {
            for r in r0..(r0 + TILE).min(rows) {
                let column = if back { width - 1 - r } else { r };
                for c in c0..(c0 + TILE).min(columns) {
                    let stored = if up { height - 1 - c } else { c };
                    let from = stored * row + column * channels;
                    let to = (r * columns + c) * channels;
                    turned[to..to + channels].copy_from_slice(&samples[from..from + channels]);
                }
            }
        }
    }
    Ok(Oriented {
        samples: turned,
        height: rows,
        width: columns,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pixel (r, c) of the stored picture as the table of the module has it.
    fn shown(orientation: i32, height: usize, width: usize, r: usize, c: usize) -> (usize, usize) {
        match orientation {
            2 => (r, width - 1 - c),
            3 => (height - 1 - r, width - 1 - c),
            4 => (height - 1 - r, c),
            5 => (c, r),
            6 => (height - 1 - c, r),
            7 => (height - 1 - c, width - 1 - r),
            8 => (c, width - 1 - r),
            _ => (r, c),
        }
    }

    #[test]
    fn every_orientation_is_its_table() {
        // Sizes on both sides of a tile, and pixels of two samples, each of which
        // says where it is stored.
        for (height, width) in [(1, 1), (1, 5), (4, 1), (3, 7), (70, 130), (129, 64)] {
            let channels = 2;
            let samples: Vec<u32> = (0..height * width * channels)
                .map(|index| u32::try_from(index).expect("fits"))
                .collect();
            for orientation in 0..=8 {
                let turned = oriented(&samples, height, width, channels, orientation).expect("a picture");
                let (rows, columns) = if transposes(orientation) {
                    (width, height)
                } else {
                    (height, width)
                };
                assert_eq!((turned.height, turned.width), (rows, columns));
                for r in 0..rows {
                    for c in 0..columns {
                        let (sr, sc) = shown(orientation, height, width, r, c);
                        for channel in 0..channels {
                            assert_eq!(
                                turned.samples[(r * columns + c) * channels + channel],
                                samples[(sr * width + sc) * channels + channel],
                                "orientation {orientation}, {height} x {width}, pixel ({r}, {c})"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn a_picture_of_two_by_three_is_turned_as_it_is_shown() {
        // 1 2 3
        // 4 5 6
        let samples = [1, 2, 3, 4, 5, 6];
        let turned = |orientation| oriented(&samples, 2, 3, 1, orientation).expect("a picture").samples;
        assert_eq!(turned(2), [3, 2, 1, 6, 5, 4]);
        assert_eq!(turned(3), [6, 5, 4, 3, 2, 1]);
        assert_eq!(turned(4), [4, 5, 6, 1, 2, 3]);
        assert_eq!(turned(5), [1, 4, 2, 5, 3, 6]);
        // The stored picture turned a quarter clockwise, and anticlockwise.
        assert_eq!(turned(6), [4, 1, 5, 2, 6, 3]);
        assert_eq!(turned(7), [6, 3, 5, 2, 4, 1]);
        assert_eq!(turned(8), [3, 6, 2, 5, 1, 4]);
    }

    #[test]
    fn samples_move_bit_for_bit() {
        let samples = [-0.0, f64::from_bits(1), f64::NAN, 0.1 + 0.2, f64::MAX, 255.0];
        for orientation in 1..=8 {
            let turned = oriented(&samples, 3, 2, 1, orientation).expect("a picture");
            let mut bits: Vec<u64> = turned.samples.iter().map(|sample| sample.to_bits()).collect();
            let mut expected: Vec<u64> = samples.iter().map(|sample| sample.to_bits()).collect();
            bits.sort_unstable();
            expected.sort_unstable();
            assert_eq!(bits, expected);
        }
    }

    #[test]
    fn wrong_orientations_and_pictures_are_refused() {
        assert!(oriented(&[0u8; 6], 2, 3, 1, 9).is_err());
        assert!(oriented(&[0u8; 6], 2, 3, 1, -1).is_err());
        assert!(oriented(&[0u8; 6], 2, 2, 1, 1).is_err());
        assert!(oriented(&[0u8; 6], 3, 2, 0, 1).is_err());
        let empty = oriented::<u8>(&[], 0, 4, 3, 6).expect("an empty picture");
        assert_eq!((empty.height, empty.width), (4, 0));
    }
}
