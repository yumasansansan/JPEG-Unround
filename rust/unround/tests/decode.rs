// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! Decoding files: every method keeps the intervals, and the settings reach the model
//! (docs/math.md, 1.3 and 8).
//!
//! The files are written by libjpeg's encoder from levels chosen here, the
//! quantization of synthetic pictures, so that what the decoder solves is known.

#![allow(
    clippy::float_cmp,
    reason = "the tests compare doubles to the last bit where the arithmetic is exact or the same"
)]

mod support;

use jpeg_unround::colour;
use jpeg_unround::decode::{self, Decoded, Method, Settings};
use jpeg_unround::frames;
use jpeg_unround::model::{Centres, DataTerm, Tv};
use jpeg_unround::pdhg;
use jpeg_unround::subgradient;
use jpegio_sys::ColorSpace;
use jpegio_sys::testing::{File, Layout};
use support::synthetic;

const INVARIANT: f64 = 1e-4; // in steps

/// A greyscale file of 20 x 30 samples: 3 x 4 blocks of the quantization of a picture.
fn grey_file(seed: u64) -> (Vec<u8>, Vec<i16>, [u16; 64]) {
    let table = synthetic::scaled(&synthetic::ANNEX_K, 2.0);
    let canvas = synthetic::picture(24, 32, seed);
    let levels = synthetic::levels(&canvas, 24, 32, &table);
    let file = File {
        width: 30,
        height: 20,
        layout: Layout::Grayscale,
        quant_tables: vec![table],
        progressive: false,
        arithmetic: false,
    };
    let data = file.write(std::slice::from_ref(&levels)).expect("a file");
    (data, levels, table)
}

/// A colour file of 20 x 30 pixels, its chroma in cells of `ratio` `(down, across)`.
fn colour_file(seed: u64, ratio: (usize, usize)) -> Vec<u8> {
    let (components, _) = synthetic::colour_levels(20, 30, seed, ratio);
    let factor = |value: usize| i32::try_from(value).expect("fits");
    let file = File {
        width: 30,
        height: 20,
        layout: Layout::YCbCr([(factor(ratio.1), factor(ratio.0)), (1, 1), (1, 1)]),
        quant_tables: components.iter().map(|(_, table, _)| *table).collect(),
        progressive: true,
        arithmetic: false,
    };
    let levels: Vec<Vec<i16>> = components.into_iter().map(|(levels, _, _)| levels).collect();
    file.write(&levels).expect("a file")
}

fn short(method: Method) -> Settings {
    Settings {
        method,
        pdhg: pdhg::Options {
            iterations: Some(60),
            ..pdhg::Options::default()
        },
        subgradient: subgradient::Options {
            iterations: 15,
            ..subgradient::Options::default()
        },
        ..Settings::default()
    }
}

/// The coefficients are within their intervals exactly, and the canvas's own within the
/// invariant.
fn keeps_the_intervals(decoded: &Decoded) {
    let frame = &decoded.frame;
    for (channel, own) in frame.channels().iter().zip(&decoded.coefficients) {
        assert!(channel.problem().excess(own).iter().all(|&excess| excess == 0.0));
    }
    let canvas = match &decoded.result {
        Some(result) => result.primal.canvas.clone(),
        None => frames::start(frame, None).expect("a start").canvas,
    };
    for excess in frames::excess(frame, &canvas) {
        assert!(excess.iter().all(|&value| value <= INVARIANT));
    }
}

#[test]
fn each_method_keeps_the_intervals() {
    let (data, levels, table) = grey_file(30);
    for method in [Method::Mmse, Method::Tv, Method::Tgv, Method::Subgradient] {
        let decoded = decode::decode(&data, &short(method), None).expect("the file decodes");
        assert_eq!((decoded.height, decoded.width, decoded.channels), (20, 30, 1));
        assert_eq!(decoded.color_space, ColorSpace::Grayscale);
        assert_eq!(decoded.picture, decoded.planes);
        assert_eq!(decoded.result.is_none(), method == Method::Mmse);
        keeps_the_intervals(&decoded);
        // The file's levels are the problem's: the middles of the intervals are q Q.
        let problem = decoded.frame.channels()[0].problem();
        for (index, &level) in levels.iter().enumerate() {
            let middle = f64::from(level) * f64::from(table[index % 64]) + if index % 64 == 0 { 1024.0 } else { 0.0 };
            assert!(problem.lower()[index] < middle && middle < problem.upper()[index]);
        }
    }
}

#[test]
fn the_mmse_decoder_is_the_centres() {
    let (data, _, _) = grey_file(31);
    let decoded = decode::decode(&data, &short(Method::Mmse), None).expect("the file decodes");
    let problem = decoded.frame.channels()[0].problem();
    assert_eq!(decoded.coefficients[0], problem.centres());
}

#[test]
fn the_settings_reach_the_model() {
    let (data, levels, table) = grey_file(32);
    let settings = Settings {
        data: vec![DataTerm {
            mu: Some(0.25),
            slack: 0.5,
            centres: Centres::Midpoint,
            ..DataTerm::default()
        }],
        ..short(Method::Mmse)
    };
    let decoded = decode::decode(&data, &settings, None).expect("the file decodes");
    let problem = decoded.frame.channels()[0].problem();
    assert_eq!(problem.weights()[9], 0.25 / f64::from(table[9]).powi(2));
    for (index, &level) in levels.iter().enumerate().skip(1).take(63) {
        let step = f64::from(table[index]);
        assert_eq!(problem.lower()[index], ((f64::from(level) - 0.5) - 0.5) * step);
        assert_eq!(problem.centres()[index], f64::from(level) * step);
    }
    let rule = Settings {
        data: vec![DataTerm {
            mu: None,
            mu_scale: 2.0,
            ..DataTerm::default()
        }],
        ..short(Method::Mmse)
    };
    let decoded = decode::decode(&data, &rule, None).expect("the file decodes");
    let mean = f64::from(table.iter().map(|&step| u32::from(step)).sum::<u32>()) / 64.0;
    let weight = decoded.frame.channels()[0].problem().weights()[9];
    assert_eq!(weight, (2.0 * mean) / f64::from(table[9]).powi(2));
}

#[test]
fn colour_files_keep_their_intervals() {
    for ratio in [(2, 2), (1, 2), (1, 1)] {
        let data = colour_file(33, ratio);
        for method in [Method::Mmse, Method::Tv, Method::Tgv] {
            let decoded = decode::decode(&data, &short(method), None).expect("the file decodes");
            assert_eq!(decoded.color_space, ColorSpace::YCbCr);
            assert_eq!((decoded.height, decoded.width, decoded.channels), (20, 30, 3));
            assert_eq!(decoded.planes.len(), 3 * 20 * 30);
            keeps_the_intervals(&decoded);
            // The picture is JFIF's RGB of the planes, to the last bit.
            let rgb = colour::to_rgb(&decoded.planes, 20, 30).expect("the planes");
            assert_eq!(decoded.picture, rgb);
            let ratios: Vec<(usize, usize)> = decoded.frame.channels().iter().map(frames::Channel::ratio).collect();
            assert_eq!(ratios, [(1, 1), ratio, ratio]);
        }
    }
}

#[test]
fn options_of_g_for_each_component() {
    let data = colour_file(34, (2, 2));
    let each = Settings {
        data: vec![
            DataTerm {
                mu: Some(1.0),
                ..DataTerm::default()
            },
            DataTerm {
                mu: Some(2.0),
                ..DataTerm::default()
            },
            DataTerm {
                mu: Some(4.0),
                ..DataTerm::default()
            },
        ],
        ..short(Method::Mmse)
    };
    let decoded = decode::decode(&data, &each, None).expect("the file decodes");
    let weights: Vec<f64> = decoded
        .frame
        .channels()
        .iter()
        .map(|channel| channel.problem().weights()[9] * channel.problem().steps()[9].powi(2))
        .collect();
    assert_eq!(weights, [1.0, 2.0, 4.0]);
    let two = Settings {
        data: vec![DataTerm::default(); 2],
        ..short(Method::Mmse)
    };
    let error = decode::decode(&data, &two, None).expect_err("refused");
    assert!(error.to_string().contains("for each of the 3 components"), "{error}");
}

#[test]
fn the_channels_are_coupled_or_not_as_the_weights_say() {
    let data = colour_file(35, (2, 2));
    let coupled = decode::decode(&data, &short(Method::Tv), None).expect("the file decodes");
    let apart = Settings {
        tv: Tv {
            coupled: false,
            ..Tv::default()
        },
        ..short(Method::Tv)
    };
    let apart = decode::decode(&data, &apart, None).expect("the file decodes");
    assert_ne!(coupled.picture, apart.picture);
}

#[test]
fn the_subgradient_method_is_for_greyscale_files() {
    let data = colour_file(36, (2, 2));
    let error = decode::decode(&data, &short(Method::Subgradient), None).expect_err("refused");
    assert!(error.to_string().contains("one component"), "{error}");
}
