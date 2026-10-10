// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! A picture made elsewhere, measured against a file's intervals and projected onto
//! them (docs/math.md, 12): its canvas made as an encoder makes a file's, its excess
//! and its projection those of `frames` on that canvas to the last bit, a
//! reconstruction within the intervals, a spike outside them, and the projection of
//! the spike back within them to the rounding of the DCTs.

#![allow(
    clippy::float_cmp,
    reason = "the tests compare doubles to the last bit where the arithmetic is the same"
)]

mod support;

use jpeg_unround::colour;
use jpeg_unround::dct::{self, BLOCK_SIZE};
use jpeg_unround::decode::{self, Input, Method, Settings};
use jpeg_unround::frames;
use jpeg_unround::picture;
use jpegio_sys::testing::{File, Layout};
use support::{rounding, synthetic};

/// A greyscale file of `height x width` samples, of the quantization of a picture of
/// whole blocks.
fn grey_file(height: usize, width: usize, seed: u64) -> Vec<u8> {
    let table = synthetic::scaled(&synthetic::ANNEX_K, 2.0);
    let (rows, columns) = (height.div_ceil(8) * 8, width.div_ceil(8) * 8);
    let canvas = synthetic::picture(rows, columns, seed);
    let levels = synthetic::levels(&canvas, rows, columns, &table);
    let file = File {
        width: u32::try_from(width).expect("small"),
        height: u32::try_from(height).expect("small"),
        layout: Layout::Grayscale,
        quant_tables: vec![table],
        progressive: false,
        arithmetic: false,
        icc_profile: Vec::new(),
        exif: Vec::new(),
    };
    file.write(std::slice::from_ref(&levels)).expect("a file")
}

/// A colour file of 20 x 30 pixels, its chroma in cells of 2 x 2.
fn colour_file(seed: u64) -> Vec<u8> {
    let (components, _) = synthetic::colour_levels(20, 30, seed, (2, 2));
    let file = File {
        width: 30,
        height: 20,
        layout: Layout::YCbCr([(2, 2), (1, 1), (1, 1)]),
        quant_tables: components.iter().map(|(_, table, _)| *table).collect(),
        progressive: false,
        arithmetic: false,
        icc_profile: Vec::new(),
        exif: Vec::new(),
    };
    let levels: Vec<Vec<i16>> = components.into_iter().map(|(levels, _, _)| levels).collect();
    file.write(&levels).expect("a file")
}

fn input_of(data: &[u8]) -> Input {
    Input::read(data, &jpegio_sys::Options::default()).expect("the file is read")
}

fn mmse() -> Settings {
    Settings {
        method: Method::Mmse,
        ..Settings::default()
    }
}

#[test]
fn the_canvas_repeats_the_picture_s_last_column_and_row() {
    // Greyscale: 13 x 10 samples on a canvas of 16 x 16.
    let input = input_of(&grey_file(13, 10, 3));
    let frame = decode::frame_of(&input, &Settings::default()).expect("a frame");
    assert_eq!((frame.height(), frame.width()), (16, 16));
    let picture: Vec<f64> = (0..13 * 10)
        .map(|k| f64::from(u32::try_from(k).expect("small")) * 0.5)
        .collect();
    let canvas = picture::canvas_of(&input, &frame, &picture, 1).expect("a canvas");
    for i in 0..16 {
        for j in 0..16 {
            assert_eq!(canvas[i * 16 + j], picture[i.min(12) * 10 + j.min(9)], "({i}, {j})");
        }
    }
    // YCbCr: the Y, Cb and Cr of JFIF's conversion, each extended alike.
    let input = input_of(&colour_file(5));
    let frame = decode::frame_of(&input, &Settings::default()).expect("a frame");
    assert_eq!((frame.height(), frame.width()), (32, 32));
    let rgb: Vec<f64> = (0..20 * 30 * 3)
        .map(|k| f64::from(u32::try_from(k * 37 % 256).expect("small")) + 0.25)
        .collect();
    let canvas = picture::canvas_of(&input, &frame, &rgb, 3).expect("a canvas");
    let planes = colour::to_ycbcr(&rgb, 20, 30).expect("planes");
    for channel in 0..3 {
        for i in 0..32 {
            for j in 0..32 {
                let expected = planes[channel * 600 + i.min(19) * 30 + j.min(29)];
                assert_eq!(canvas[channel * 1024 + i * 32 + j].to_bits(), expected.to_bits());
            }
        }
    }
}

#[test]
fn the_excess_and_the_projection_are_those_of_the_canvas() {
    let data = colour_file(7);
    let input = input_of(&data);
    let settings = Settings::default();
    let decoded = decode::decode(&data, &mmse(), None).expect("the file decodes");
    // The MMSE decoder's picture, moved: a picture made elsewhere.
    let picture: Vec<f64> = decoded
        .picture
        .iter()
        .enumerate()
        .map(|(k, &sample)| sample + if k % 7 == 0 { 9.0 } else { -2.5 })
        .collect();
    let frame = decode::frame_of(&input, &settings).expect("a frame");
    let canvas = picture::canvas_of(&input, &frame, &picture, 3).expect("a canvas");
    let excess = picture::excess(&input, &settings, &picture, 3).expect("the excess");
    assert_eq!(excess, frames::excess(&frame, &canvas));
    assert!(excess.iter().flatten().any(|&value| value > 0.0));
    let projection = picture::project(&input, &settings, &picture, 3).expect("a projection");
    assert_eq!(projection.excess, excess);
    let (projected, coefficients) = frames::project_with_coefficients(&frame, &canvas);
    assert_eq!(projection.decoded.canvas, projected);
    assert_eq!(projection.decoded.coefficients, coefficients);
    assert_eq!(projected, frames::project(&frame, &canvas));
    // The coefficients are the canvas's, clipped.
    let plane = frame.plane();
    for (index, channel) in frame.channels().iter().enumerate() {
        let problem = channel.problem();
        let own = frames::means(channel, &canvas[index * plane..(index + 1) * plane], frame.width());
        let mut clipped = dct::forward(&own, problem.height(), problem.width());
        problem.clip(&mut clipped);
        assert_eq!(coefficients[index], clipped);
    }
    // The picture is the projected canvas cut, in JFIF's RGB.
    let mut planes = Vec::new();
    for channel in 0..3 {
        for i in 0..20 {
            planes.extend_from_slice(&projected[channel * plane + i * frame.width()..][..30]);
        }
    }
    assert_eq!(projection.decoded.planes, planes);
    assert_eq!(
        projection.decoded.picture,
        colour::to_rgb(&planes, 20, 30).expect("RGB")
    );
    assert!(projection.decoded.result.is_none());
}

#[test]
fn a_reconstruction_lies_within_and_a_spike_is_projected_back() {
    // Of whole blocks, so that the canvas is the picture itself.
    let data = grey_file(24, 32, 11);
    let input = input_of(&data);
    let settings = Settings::default();
    let decoded = decode::decode(&data, &mmse(), None).expect("the file decodes");
    // The MMSE centres lie inside their intervals, far beyond the rounding of a DCT.
    let excess = picture::excess(&input, &settings, &decoded.picture, 1).expect("the excess");
    assert!(excess[0].iter().all(|&value| value == 0.0));
    // A spike leaves the intervals in its block alone.
    let mut spiked = decoded.picture.clone();
    spiked[10 * 32 + 13] += 1500.0;
    let excess = picture::excess(&input, &settings, &spiked, 1).expect("the excess");
    let spike_block = 4 + 1;
    for (block, values) in excess[0].as_chunks::<BLOCK_SIZE>().0.iter().enumerate() {
        assert_eq!(
            values.iter().any(|&value| value > 0.0),
            block == spike_block,
            "block {block}"
        );
    }
    // Projected, it lies within them to the rounding of the inverse DCT and of the
    // forward DCT that measures it again.
    let projection = picture::project(&input, &settings, &spiked, 1).expect("a projection");
    let again = picture::excess(&input, &settings, &projection.decoded.picture, 1).expect("the excess");
    let frame = decode::frame_of(&input, &settings).expect("a frame");
    let problem = frame.channels()[0].problem();
    for block in 0..problem.rows() * problem.columns() {
        let reach = rounding::roundtrip_error(&rounding::block_of(&projection.decoded.coefficients[0], block));
        for (k, &value) in again[0][block * BLOCK_SIZE..(block + 1) * BLOCK_SIZE]
            .iter()
            .enumerate()
        {
            assert!(
                value * problem.steps()[k] <= reach[k / 8][k % 8],
                "block {block}, coefficient {k}: {value}"
            );
        }
    }
}

#[test]
fn a_picture_that_is_not_the_file_s_is_refused() {
    let data = colour_file(13);
    let input = input_of(&data);
    let settings = Settings::default();
    let picture = vec![100.0; 20 * 30 * 3];
    assert!(picture::excess(&input, &settings, &picture, 1).is_err());
    assert!(picture::excess(&input, &settings, &picture[1..], 3).is_err());
    let mut odd = picture.clone();
    odd[5] = f64::NAN;
    assert!(picture::project(&input, &settings, &odd, 3).is_err());
    odd[5] = f64::INFINITY;
    assert!(picture::project(&input, &settings, &odd, 3).is_err());
    assert!(picture::project(&input, &settings, &picture, 3).is_ok());
    assert_eq!(picture::channels_of(&input), 3);
}
