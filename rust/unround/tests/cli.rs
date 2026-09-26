// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! The command line (docs/cli.md): the program that the crate builds, run on files
//! written here. What it writes has to be what the library makes of the file: the
//! picture turned upright as the file's EXIF orientation says (or as stored, where
//! asked), in TIFF, PNG or PNM, with the file's ICC profile where the format holds
//! one and the profile goes with the picture.

mod support;

use std::path::{Path, PathBuf};
use std::process::Command;

use jpeg_unround::decode::{self, Decoded, Method, Settings};
use jpeg_unround::{orientation, output, pdhg, tiff};
use jpegio_sys::testing::{File, Layout, read_png};
use support::{metadata, synthetic};

/// A directory of its own for a test, in the system's temporary directory.
fn directory(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("unround-cli-{}-{name}", std::process::id()));
    if path.exists() {
        std::fs::remove_dir_all(&path).expect("the old directory is removed");
    }
    std::fs::create_dir_all(&path).expect("a directory");
    path
}

fn grey_file() -> Vec<u8> {
    let table = synthetic::scaled(&synthetic::ANNEX_K, 2.0);
    let canvas = synthetic::picture(24, 32, 80);
    let levels = synthetic::levels(&canvas, 24, 32, &table);
    let file = File {
        width: 30,
        height: 20,
        layout: Layout::Grayscale,
        quant_tables: vec![table],
        progressive: false,
        arithmetic: false,
        icc_profile: Vec::new(),
        exif: Vec::new(),
    };
    file.write(&[levels]).expect("a file")
}

fn unround(arguments: &[&str]) -> (i32, String) {
    let output = Command::new(env!("CARGO_BIN_EXE_unround"))
        .args(arguments)
        .output()
        .expect("the program runs");
    (
        output.status.code().expect("an exit status"),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

#[test]
fn the_result_is_the_library_s_to_the_last_bit() {
    let place = directory("result");
    let input = place.join("in.jpg");
    std::fs::write(&input, grey_file()).expect("the file is written");
    let output = place.join("out.tif");
    let report = place.join("report.json");
    let (status, errors) = unround(&[
        "--method",
        "tv",
        "--iterations",
        "40",
        "--record-every",
        "20",
        "--quiet",
        "--report",
        report.to_str().expect("a path"),
        input.to_str().expect("a path"),
        output.to_str().expect("a path"),
    ]);
    assert_eq!(status, 0, "{errors}");
    let settings = Settings {
        method: Method::Tv,
        pdhg: pdhg::Options {
            iterations: Some(40),
            record_every: 20,
            ..pdhg::Options::default()
        },
        ..Settings::default()
    };
    let decoded = decode::decode(&grey_file(), &settings, None).expect("the file decodes");
    let expected = tiff::float64(&decoded.picture, 20, 30, 1, false, None).expect("a file");
    assert_eq!(std::fs::read(&output).expect("the result"), expected.bytes);
    let text = std::fs::read_to_string(&report).expect("the report");
    for fragment in [
        "\"method\": \"tv\"",
        "\"iterations\": 40",
        "\"height\": 20",
        "\"records\"",
    ] {
        assert!(text.contains(fragment), "{fragment} in {text}");
    }
    // A file that exists is not overwritten, unless asked.
    let (status, errors) = unround(&[input.to_str().expect("a path"), output.to_str().expect("a path")]);
    assert_eq!(status, 3, "{errors}");
    let (status, errors) = unround(&[
        "--method",
        "mmse",
        "--overwrite",
        input.to_str().expect("a path"),
        output.to_str().expect("a path"),
    ]);
    assert_eq!(status, 0, "{errors}");
    std::fs::remove_dir_all(&place).expect("the directory is removed");
}

#[test]
fn a_command_line_that_is_refused_says_why() {
    let (status, errors) = unround(&["--alpha", "0", "--relaxation", "3", "in.jpg"]);
    assert_eq!(status, 2);
    assert!(
        errors.contains("--alpha is positive") && errors.contains("--relaxation is in (0, 2)"),
        "{errors}"
    );
    let (status, errors) = unround(&["--version"]);
    assert_eq!(status, 0, "{errors}");
    let place = directory("missing");
    let (status, errors) = unround(&[place.join("none.jpg").to_str().expect("a path")]);
    assert_eq!(status, 1, "{errors}");
    std::fs::remove_dir_all(&place).expect("the directory is removed");
}

/// A colour file of 20 x 30 pixels with an ICC profile and an EXIF orientation.
fn colour_file(profile: &[u8], exif: Vec<u8>) -> Vec<u8> {
    let (components, _) = synthetic::colour_levels(20, 30, 90, (2, 2));
    let file = File {
        width: 30,
        height: 20,
        layout: Layout::YCbCr([(2, 2), (1, 1), (1, 1)]),
        quant_tables: components.iter().map(|(_, table, _)| *table).collect(),
        progressive: false,
        arithmetic: false,
        icc_profile: profile.to_vec(),
        exif,
    };
    let levels: Vec<Vec<i16>> = components.into_iter().map(|(levels, _, _)| levels).collect();
    file.write(&levels).expect("a file")
}

fn text(path: &Path) -> &str {
    path.to_str().expect("a path")
}

/// The picture of the decoded file turned as `orientation` says, and its size.
fn turned(samples: &[f64], decoded: &Decoded, channels: usize, orientation: i32) -> (Vec<f64>, usize, usize) {
    let turned =
        orientation::oriented(samples, decoded.height, decoded.width, channels, orientation).expect("a picture");
    (turned.samples, turned.height, turned.width)
}

#[test]
fn the_picture_is_turned_upright_with_its_icc_profile() {
    let place = directory("orientation");
    let profile = metadata::profile(560, *b"RGB ", 9);
    let data = colour_file(&profile, metadata::exif(6, true));
    let input = place.join("in.jpg");
    std::fs::write(&input, &data).expect("the file is written");
    let settings = Settings {
        method: Method::Mmse,
        ..Settings::default()
    };
    let decoded = decode::decode(&data, &settings, None).expect("the file decodes");
    assert_eq!(
        (decoded.exif_orientation, decoded.icc_profile.as_deref()),
        (6, Some(profile.as_slice()))
    );

    // PNG, upright: 30 rows of 20, with the profile.
    let png = place.join("out.png");
    let report = place.join("report.json");
    let (status, errors) = unround(&["--method", "mmse", "--report", text(&report), text(&input), text(&png)]);
    assert_eq!((status, errors.as_str()), (0, ""));
    let back = read_png(&std::fs::read(&png).expect("the result")).expect("a PNG file");
    let (upright, height, width) = turned(&decoded.picture, &decoded, 3, 6);
    assert_eq!((back.height, back.width, back.channels, back.bits), (30, 20, 3, 8));
    assert_eq!((height, width), (30, 20));
    let eight: Vec<u16> = output::eight_bits(&upright)
        .expect("finite")
        .into_iter()
        .map(u16::from)
        .collect();
    assert_eq!(back.samples, eight);
    assert_eq!(back.icc_profile.as_deref(), Some(profile.as_slice()));
    let report = std::fs::read_to_string(&report).expect("the report");
    for fragment in [
        "\"exif_orientation\": 6",
        "\"oriented\": true",
        "\"icc_profile\": true",
        "\"warnings\": []",
    ] {
        assert!(report.contains(fragment), "{fragment} in {report}");
    }

    // As stored, in 16 bits, not compressed, and without the profile.
    let kept = place.join("kept.png");
    let (status, errors) = unround(&[
        "--method",
        "mmse",
        "--orientation",
        "keep",
        "--bits",
        "16",
        "--compression",
        "0",
        "--no-icc",
        text(&input),
        text(&kept),
    ]);
    assert_eq!((status, errors.as_str()), (0, ""));
    let back = read_png(&std::fs::read(&kept).expect("the result")).expect("a PNG file");
    assert_eq!((back.height, back.width, back.bits), (20, 30, 16));
    assert_eq!(back.samples, output::sixteen_bits(&decoded.picture).expect("finite"));
    assert_eq!(back.icc_profile, None);

    // TIFF, upright, bit for bit, with the profile; and Y, Cb and Cr turned alike.
    let tif = place.join("out.tif");
    let (status, errors) = unround(&["--method", "mmse", text(&input), text(&tif)]);
    assert_eq!((status, errors.as_str()), (0, ""));
    let expected = tiff::float64(&upright, 30, 20, 3, false, Some(&profile)).expect("a file");
    assert_eq!(std::fs::read(&tif).expect("the result"), expected.bytes);
    let ycbcr = place.join("ycbcr.tif");
    let (status, errors) = unround(&["--method", "mmse", "--ycbcr", text(&input), text(&ycbcr)]);
    assert_eq!((status, errors.as_str()), (0, ""));
    let size = decoded.height * decoded.width;
    let interleaved: Vec<f64> = (0..3 * size)
        .map(|index| decoded.planes[(index % 3) * size + index / 3])
        .collect();
    let (planes, height, width) = turned(&interleaved, &decoded, 3, 6);
    let expected = tiff::float64(&planes, height, width, 3, true, Some(&profile)).expect("a file");
    assert_eq!(std::fs::read(&ycbcr).expect("the result"), expected.bytes);

    // PNM holds no profile, which is said, unless asked for none or for quiet.
    let ppm = place.join("out.ppm");
    let (status, errors) = unround(&["--method", "mmse", text(&input), text(&ppm)]);
    assert_eq!(status, 0, "{errors}");
    assert!(errors.contains("PNM holds no ICC profile"), "{errors}");
    let expected = output::pnm(&upright, 30, 20, 3, false).expect("a file");
    assert_eq!(std::fs::read(&ppm).expect("the result"), expected);
    for quiet in ["--no-icc", "--quiet"] {
        let (status, errors) = unround(&["--method", "mmse", "--overwrite", quiet, text(&input), text(&ppm)]);
        assert_eq!((status, errors.as_str()), (0, ""));
    }
    std::fs::remove_dir_all(&place).expect("the directory is removed");
}

#[test]
fn every_orientation_is_applied() {
    let place = directory("orientations");
    let settings = Settings {
        method: Method::Mmse,
        ..Settings::default()
    };
    for orientation in 1..=8u16 {
        let data = colour_file(&[], metadata::exif(orientation, orientation % 2 == 0));
        let input = place.join(format!("in{orientation}.jpg"));
        std::fs::write(&input, &data).expect("the file is written");
        let output = place.join(format!("out{orientation}.tif"));
        let (status, errors) = unround(&["--method", "mmse", text(&input), text(&output)]);
        assert_eq!((status, errors.as_str()), (0, ""));
        let decoded = decode::decode(&data, &settings, None).expect("the file decodes");
        assert_eq!(decoded.exif_orientation, i32::from(orientation));
        let (upright, height, width) = turned(&decoded.picture, &decoded, 3, i32::from(orientation));
        let expected = tiff::float64(&upright, height, width, 3, false, None).expect("a file");
        assert_eq!(
            std::fs::read(&output).expect("the result"),
            expected.bytes,
            "orientation {orientation}"
        );
    }
    std::fs::remove_dir_all(&place).expect("the directory is removed");
}

#[test]
fn a_profile_that_does_not_go_with_the_picture_is_left_out() {
    let place = directory("profile");
    // A gray profile in a colour file.
    let data = colour_file(&metadata::profile(300, *b"GRAY", 2), Vec::new());
    let input = place.join("in.jpg");
    std::fs::write(&input, &data).expect("the file is written");
    let report = place.join("report.json");
    for name in ["out.png", "out.tif"] {
        let output = place.join(name);
        let (status, errors) = unround(&[
            "--method",
            "mmse",
            "--overwrite",
            "--report",
            text(&report),
            text(&input),
            text(&output),
        ]);
        assert_eq!(status, 0, "{errors}");
        assert!(errors.contains("the ICC profile is not written: "), "{errors}");
        let report = std::fs::read_to_string(&report).expect("the report");
        for fragment in [
            "\"icc_profile\": false",
            "\"oriented\": false",
            "\"exif_orientation\": 0",
            "the ICC profile is not written",
        ] {
            assert!(report.contains(fragment), "{fragment} in {report}");
        }
    }
    let back = read_png(&std::fs::read(place.join("out.png")).expect("the result")).expect("a PNG file");
    assert_eq!(back.icc_profile, None);
    std::fs::remove_dir_all(&place).expect("the directory is removed");
}
