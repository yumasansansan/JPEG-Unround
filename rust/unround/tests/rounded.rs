// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! PNG files reconstructed within the intervals of their rounded samples
//! (docs/math.md, 11; docs/cli.md): the library's result read back by the C layer,
//! rounded to the file's own 8 or 16 bits it gives back the file's levels, and the
//! command line's files -- TIFF with alpha declared, PNG with alpha, PNM refused --
//! with the report and the refusals.

use std::path::PathBuf;
use std::process::Command;

use jpeg_unround::decode::{Method, Settings};
use jpeg_unround::{output, rounded};
use jpegio_sys::testing::read_png;
use jpegio_sys::{Png, Samples};

/// A directory of its own for a test, in the system's temporary directory.
fn directory(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("unround-rounded-{}-{name}", std::process::id()));
    if path.exists() {
        std::fs::remove_dir_all(&path).expect("the old directory is removed");
    }
    std::fs::create_dir_all(&path).expect("a directory");
    path
}

/// A smooth RGBA picture of 8 bits, as drawn (binary64) and as rounded into a PNG file.
fn rgba() -> (Vec<f64>, Vec<u8>, usize, usize) {
    let (height, width) = (20, 28);
    let mut drawn = Vec::with_capacity(height * width * 4);
    for i in 0..height {
        for j in 0..width {
            let (y, x) = (
                f64::from(u32::try_from(i).expect("small")),
                f64::from(u32::try_from(j).expect("small")),
            );
            drawn.extend([
                30.0 + 0.27 * x,
                90.0 + 0.19 * y,
                160.0 - 0.13 * x - 0.07 * y,
                200.0 + 0.4 * x,
            ]);
        }
    }
    // The nearest levels, halves away from 0, as the library rounds 8-bit output.
    let levels = output::eight_bits(&drawn, height, width, 4, output::Dither::None).expect("finite values");
    let png = Png {
        width: u32::try_from(width).expect("small"),
        height: u32::try_from(height).expect("small"),
        channels: 4,
        samples: Samples::Eight(&levels),
        compression: None,
        icc_profile: None,
    };
    let file = jpegio_sys::write_png(&png).expect("a file").bytes;
    (drawn, file, height, width)
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
fn a_png_file_is_reconstructed_within_its_intervals() {
    let (drawn, file, height, width) = rgba();
    assert!(rounded::is_png(&file));
    let reconstructed = rounded::reconstruct(&file, &Settings::default(), None).expect("reconstructed");
    assert_eq!(
        (reconstructed.height, reconstructed.width, reconstructed.channels),
        (height, width, 4)
    );
    assert_eq!(reconstructed.bits, 8);
    assert!(reconstructed.alpha.is_some());
    let levels = read_png(&file).expect("the file").samples;
    let mut error = 0.0;
    let mut rounded_error = 0.0;
    for ((&value, &level), &truth) in reconstructed.picture.iter().zip(&levels).zip(&drawn) {
        let level = f64::from(level);
        // Within half a level of the file's, and of 0 to 255.
        assert!(
            (value - level).abs() <= 0.5 && (0.0..=255.0).contains(&value),
            "{value} for {level}"
        );
        error += (value - truth) * (value - truth);
        rounded_error += (level - truth) * (level - truth);
    }
    // The ramps are found: far nearer than the levels (measured: a fifth of their
    // error or less; a regression check).
    assert!(error < rounded_error / 4.0, "{error} against {rounded_error}");
    // Rounded to 8 bits without a dither, by TGV and by TV, it gives back the levels.
    let tv = Settings {
        method: Method::Tv,
        ..Settings::default()
    };
    for settings in [Settings::default(), tv] {
        let reconstructed = rounded::reconstruct(&file, &settings, None).expect("reconstructed");
        let back =
            output::eight_bits(&reconstructed.picture, height, width, 4, output::Dither::None).expect("finite values");
        assert!(back.iter().map(|&level| u16::from(level)).eq(levels.iter().copied()));
    }
}

#[test]
fn a_file_of_sixteen_bits_gives_back_its_levels() {
    // A smooth grey picture rounded to 16 bits, where a level is 1/257 wide.
    let (height, width) = (18, 24);
    let drawn: Vec<f64> = (0..height * width)
        .map(|k| {
            let (i, j) = (
                f64::from(u32::try_from(k / width).expect("small")),
                f64::from(u32::try_from(k % width).expect("small")),
            );
            40.0 + 3.1 * j + 1.7 * i + 0.05 * i * j
        })
        .collect();
    let levels = output::sixteen_bits(&drawn, height, width, 1, output::Dither::None).expect("finite values");
    let png = Png {
        width: u32::try_from(width).expect("small"),
        height: u32::try_from(height).expect("small"),
        channels: 1,
        samples: Samples::Sixteen(&levels),
        compression: None,
        icc_profile: None,
    };
    let file = jpegio_sys::write_png(&png).expect("a file").bytes;
    let tv = Settings {
        method: Method::Tv,
        ..Settings::default()
    };
    for settings in [Settings::default(), tv] {
        let reconstructed = rounded::reconstruct(&file, &settings, None).expect("reconstructed");
        assert_eq!(reconstructed.bits, 16);
        let back = output::sixteen_bits(&reconstructed.picture, height, width, 1, output::Dither::None)
            .expect("finite values");
        assert_eq!(back, levels);
    }
}

#[test]
fn the_command_line_writes_alpha_where_the_format_holds_it() {
    let place = directory("formats");
    let (_, file, height, width) = rgba();
    let input = place.join("in.png");
    std::fs::write(&input, &file).expect("the file is written");
    let text = |path: &PathBuf| path.to_str().expect("a path").to_owned();
    // PNG of 16 bits, with alpha, and the report.
    let (png, report) = (place.join("out.png"), place.join("report.json"));
    let (status, errors) = unround(&[
        "--quiet",
        "--bits",
        "16",
        "--report",
        &text(&report),
        &text(&input),
        &text(&png),
    ]);
    assert_eq!((status, errors.as_str()), (0, ""));
    let back = read_png(&std::fs::read(&png).expect("the result")).expect("a PNG file");
    assert_eq!(
        (back.height, back.width, back.channels, back.bits),
        (
            u32::try_from(height).expect("small"),
            u32::try_from(width).expect("small"),
            4,
            16
        )
    );
    let report = std::fs::read_to_string(&report).expect("the report");
    for fragment in [
        "\"input_format\": \"png\"",
        "\"bits\": 8",
        "\"alpha\": true",
        "\"alpha_solver\"",
    ] {
        assert!(report.contains(fragment), "{fragment} in {report}");
    }
    // TIFF declares the alpha: four samples, ExtraSamples (338) of one value, 2.
    let tif = place.join("out.tif");
    let (status, errors) = unround(&["--quiet", &text(&input), &text(&tif)]);
    assert_eq!((status, errors.as_str()), (0, ""));
    let bytes = std::fs::read(&tif).expect("the result");
    let ifd = usize::try_from(u32::from_le_bytes(bytes[4..8].try_into().expect("4 bytes"))).expect("fits");
    let count = usize::from(u16::from_le_bytes([bytes[ifd], bytes[ifd + 1]]));
    let entry = |tag: u16| {
        (0..count).find_map(|index| {
            let at = ifd + 2 + 12 * index;
            (u16::from_le_bytes([bytes[at], bytes[at + 1]]) == tag).then(|| {
                (
                    u16::from_le_bytes([bytes[at + 8], bytes[at + 9]]),
                    u32::from_le_bytes(bytes[at + 4..at + 8].try_into().expect("4")),
                )
            })
        })
    };
    assert_eq!(entry(277), Some((4, 1)), "SamplesPerPixel");
    assert_eq!(entry(338), Some((2, 1)), "ExtraSamples: unassociated alpha");
    // PNM holds no alpha; the method is TV or TGV; --ycbcr is JPEG's.
    for (arguments, status) in [
        (vec!["--quiet", &text(&input), &text(&place.join("out.ppm"))], 3),
        (
            vec![
                "--quiet",
                "--method",
                "mmse",
                &text(&input),
                &text(&place.join("m.tif")),
            ],
            2,
        ),
        (
            vec!["--quiet", "--mu-scale", "3", &text(&input), &text(&place.join("s.tif"))],
            2,
        ),
        (
            vec!["--quiet", "--ycbcr", &text(&input), &text(&place.join("y.tif"))],
            2,
        ),
    ] {
        let (found, errors) = unround(&arguments);
        assert_eq!(found, status, "{arguments:?}: {errors}");
    }
    std::fs::remove_dir_all(&place).expect("the directory is removed");
}
