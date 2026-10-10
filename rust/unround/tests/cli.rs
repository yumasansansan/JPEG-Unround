// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! The command line (docs/cli.md): the program that the crate builds, run on files
//! written here. What it writes has to be what the library makes of the file: the
//! picture turned upright as the file's EXIF orientation says (or as stored, where
//! asked), in TIFF, PNG or PNM, with the file's ICC profile where the format holds
//! one and the profile goes with the picture; and with `--project-only`, the
//! library's projection of the picture given, taken as turned as the result is.

mod support;

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

use jpeg_unround::decode::{self, Decoded, Input, Method, Settings};
use jpeg_unround::model::DataTerm;
use jpeg_unround::{orientation, output, pdhg, picture, tiff};
use jpegio_sys::testing::{File, Layout, read_png};
use jpegio_sys::{Png, Samples};
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
    let eight: Vec<u16> = output::eight_bits(&upright, 30, 20, 3, output::Dither::BlueNoise)
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
        "\"dither\": \"blue-noise\"",
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
    let sixteen = output::sixteen_bits(&decoded.picture, 20, 30, 3, output::Dither::BlueNoise).expect("finite");
    assert_eq!(back.samples, sixteen);
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
    let expected = output::pnm(&upright, 30, 20, 3, output::Depth::default()).expect("a file");
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

#[test]
fn the_summary_names_the_stop_as_the_report_does() {
    let place = directory("summary");
    let input = place.join("in.jpg");
    std::fs::write(&input, grey_file()).expect("the file is written");
    let output = place.join("out.tif");
    let (status, errors) = unround(&[
        "--method",
        "tv",
        "--iterations",
        "5",
        "--tolerance",
        "0",
        text(&input),
        text(&output),
    ]);
    assert_eq!(status, 0, "{errors}");
    assert_eq!(errors, format!("{}: 5 iterations, iterations\n", output.display()));
    std::fs::remove_dir_all(&place).expect("the directory is removed");
}

#[test]
fn a_file_cut_short_is_written_with_the_warning_of_its_reading() {
    let place = directory("cut");
    let data = grey_file();
    let input = place.join("in.jpg");
    std::fs::write(&input, &data[..data.len() - 40]).expect("the file is written");
    let (output, report) = (place.join("out.tif"), place.join("report.json"));
    let (status, errors) = unround(&[
        "--method",
        "mmse",
        "--report",
        text(&report),
        text(&input),
        text(&output),
    ]);
    assert_eq!(status, 0, "{errors}");
    let said = format!("unround: {}: Premature end of JPEG file", output.display());
    assert!(errors.contains(&said), "{errors}");
    let report = std::fs::read_to_string(&report).expect("the report");
    assert!(report.contains("Premature end of JPEG file"), "{report}");
    // Not read where warnings are errors.
    let (status, errors) = unround(&[
        "--method",
        "mmse",
        "--warnings-are-errors",
        "--overwrite",
        text(&input),
        text(&output),
    ]);
    assert_eq!(status, 1, "{errors}");
    std::fs::remove_dir_all(&place).expect("the directory is removed");
}

/// A PNG file of 8 or 16 bits of `height x width` pixels.
fn png_file(samples: Samples<'_>, height: usize, width: usize, channels: usize) -> Vec<u8> {
    let png = Png {
        width: u32::try_from(width).expect("small"),
        height: u32::try_from(height).expect("small"),
        channels: u32::try_from(channels).expect("small"),
        samples,
        compression: None,
        icc_profile: None,
    };
    jpegio_sys::write_png(&png).expect("a PNG file").bytes
}

/// What standard error says of each component's excess, as the library measures it.
fn excess_lines(output: &Path, names: &[&str], excess: &[Vec<f64>]) -> String {
    let mut lines = String::new();
    for (name, values) in names.iter().zip(excess) {
        let summary = picture::summary(values);
        let prefix = if name.is_empty() {
            String::new()
        } else {
            format!("{name}: ")
        };
        let by = if summary.outside > 0 {
            format!(", by at most {:.3e} steps", summary.largest)
        } else {
            String::new()
        };
        let _ = writeln!(
            lines,
            "{}: {prefix}{} of {} coefficients of the picture lay outside their intervals{by}",
            output.display(),
            summary.outside,
            summary.coefficients
        );
    }
    lines
}

/// A colour file turned by its EXIF orientation 6, with an ICC profile, and a picture
/// of it made elsewhere: the MMSE decoder's in 8 bits as the file stores it, three
/// samples moved far.
struct Elsewhere {
    input: PathBuf,
    profile: Vec<u8>,
    levels: Vec<u8>,
    library: Input,
}

fn elsewhere(place: &Path) -> Elsewhere {
    let profile = metadata::profile(560, *b"RGB ", 9);
    let data = colour_file(&profile, metadata::exif(6, true));
    let input = place.join("in.jpg");
    std::fs::write(&input, &data).expect("the file is written");
    let mmse = Settings {
        method: Method::Mmse,
        ..Settings::default()
    };
    let decoded = decode::decode(&data, &mmse, None).expect("the file decodes");
    let mut levels = output::eight_bits(&decoded.picture, 20, 30, 3, output::Dither::None).expect("finite");
    levels[(5 * 30 + 7) * 3..(5 * 30 + 8) * 3].copy_from_slice(&[255, 0, 255]);
    levels[(14 * 30 + 22) * 3 + 1] = 0;
    let library = Input::read(&data, &jpegio_sys::Options::default()).expect("the file is read");
    Elsewhere {
        input,
        profile,
        levels,
        library,
    }
}

impl Elsewhere {
    /// The picture as the file stores it, of the levels' values.
    fn stored(&self) -> Vec<f64> {
        self.levels.iter().map(|&level| f64::from(level)).collect()
    }

    /// The picture as a PNG file, upright or as stored, in `place`.
    fn png(&self, place: &Path, upright: bool) -> PathBuf {
        let path = place.join(if upright { "upright.png" } else { "stored.png" });
        let file = if upright {
            let shown = orientation::oriented(&self.levels, 20, 30, 3, 6).expect("a picture");
            png_file(Samples::Eight(&shown.samples), 30, 20, 3)
        } else {
            png_file(Samples::Eight(&self.levels), 20, 30, 3)
        };
        std::fs::write(&path, file).expect("written");
        path
    }
}

#[test]
fn a_picture_made_elsewhere_is_projected_as_the_library_projects_it() {
    let place = directory("projection");
    let made = elsewhere(&place);
    let projection = picture::project(&made.library, &Settings::default(), &made.stored(), 3).expect("a projection");
    assert!(projection.excess.iter().flatten().any(|&value| value > 0.0));
    let (upright, height, width) = turned(&projection.decoded.picture, &projection.decoded, 3, 6);
    // Upright, as the result is: TIFF, bit for bit, with the file's profile.
    let picture_path = made.png(&place, true);
    let (output, report) = (place.join("out.tif"), place.join("report.json"));
    let (status, errors) = unround(&[
        "--project-only",
        text(&picture_path),
        "--report",
        text(&report),
        text(&made.input),
        text(&output),
    ]);
    assert_eq!(status, 0, "{errors}");
    assert_eq!(errors, excess_lines(&output, &["Y", "Cb", "Cr"], &projection.excess));
    let expected = tiff::float64(&upright, height, width, 3, false, Some(&made.profile)).expect("a file");
    assert_eq!(std::fs::read(&output).expect("the result"), expected.bytes);
    let report = std::fs::read_to_string(&report).expect("the report");
    for fragment in [
        "\"method\": \"projection\"".to_owned(),
        format!("\"picture\": \"{}\"", picture_path.display()),
        "\"oriented\": true".to_owned(),
        "\"icc_profile\": true".to_owned(),
        "\"settings\": [\"--slack=0.0\"]".to_owned(),
    ] {
        assert!(report.contains(&fragment), "{fragment} in {report}");
    }
    let summaries: Vec<String> = projection
        .excess
        .iter()
        .map(|values| {
            let summary = picture::summary(values);
            format!(
                "{{\"coefficients\": {}, \"outside\": {}, \"largest\": {:?}}}",
                summary.coefficients, summary.outside, summary.largest
            )
        })
        .collect();
    let excess = format!("\"excess\": [{}]", summaries.join(", "));
    assert!(report.contains(&excess), "{excess} in {report}");
    assert!(!report.contains("\"records\""), "{report}");

    // PNG of 16 bits, rounded, of the projection with a slack: the library's.
    let slack = Settings {
        data: vec![DataTerm {
            slack: 0.5,
            ..DataTerm::default()
        }],
        ..Settings::default()
    };
    let widened = picture::project(&made.library, &slack, &made.stored(), 3).expect("a projection");
    let (upright, height, width) = turned(&widened.decoded.picture, &widened.decoded, 3, 6);
    let png = place.join("slack.png");
    let (status, errors) = unround(&[
        "--project-only",
        text(&picture_path),
        "--slack",
        "0.5",
        "--bits",
        "16",
        "--dither",
        "none",
        "--quiet",
        text(&made.input),
        text(&png),
    ]);
    assert_eq!((status, errors.as_str()), (0, ""));
    let back = read_png(&std::fs::read(&png).expect("the result")).expect("a PNG file");
    let sixteen = output::sixteen_bits(&upright, height, width, 3, output::Dither::None).expect("finite");
    assert_eq!(back.samples, sixteen);
    std::fs::remove_dir_all(&place).expect("the directory is removed");
}

#[test]
fn a_picture_is_taken_as_turned_as_the_result_is() {
    let place = directory("projection-turned");
    let made = elsewhere(&place);
    let projection = picture::project(&made.library, &Settings::default(), &made.stored(), 3).expect("a projection");
    // As stored, with --orientation keep, and the Y, Cb and Cr of the projected canvas.
    let stored_path = made.png(&place, false);
    let kept = place.join("kept.tif");
    let (status, errors) = unround(&[
        "--project-only",
        text(&stored_path),
        "--orientation",
        "keep",
        "--ycbcr",
        "--quiet",
        text(&made.input),
        text(&kept),
    ]);
    assert_eq!((status, errors.as_str()), (0, ""));
    let size = 20 * 30;
    let interleaved: Vec<f64> = (0..3 * size)
        .map(|index| projection.decoded.planes[(index % 3) * size + index / 3])
        .collect();
    let expected = tiff::float64(&interleaved, 20, 30, 3, true, Some(&made.profile)).expect("a file");
    assert_eq!(std::fs::read(&kept).expect("the result"), expected.bytes);
    // The picture as stored is not the upright one, nor that one the stored, which the
    // messages say.
    let (status, errors) = unround(&["--project-only", text(&stored_path), text(&made.input), text(&kept)]);
    assert_eq!(status, 1, "{errors}");
    assert!(
        errors.contains("a picture of the file is 30 x 20 pixels, not 20 x 30, as the file stores it"),
        "{errors}"
    );
    let upright_path = made.png(&place, true);
    let (status, errors) = unround(&[
        "--project-only",
        text(&upright_path),
        "--orientation=keep",
        text(&made.input),
        text(&kept),
    ]);
    assert_eq!(status, 1, "{errors}");
    assert!(
        errors.contains("turned upright, which --orientation apply takes"),
        "{errors}"
    );
    // From standard input, to standard output, where no summary goes with it.
    let run = Command::new(env!("CARGO_BIN_EXE_unround"))
        .args(["--project-only", "-", "--format", "pnm", text(&made.input), "-"])
        .stdin(std::fs::File::open(&upright_path).expect("the picture"))
        .output()
        .expect("the program runs");
    assert_eq!(run.status.code(), Some(0));
    let (upright, height, width) = turned(&projection.decoded.picture, &projection.decoded, 3, 6);
    let expected = output::pnm(&upright, height, width, 3, output::Depth::default()).expect("a file");
    assert_eq!(run.stdout, expected);
    // The warning of the profile alone.
    assert_eq!(
        String::from_utf8_lossy(&run.stderr),
        "unround: -: PNM holds no ICC profile: the file's is not written\n"
    );
    std::fs::remove_dir_all(&place).expect("the directory is removed");
}

#[test]
fn a_greyscale_file_s_picture_is_grey_or_rgb_alike() {
    let place = directory("grey-projection");
    let data = grey_file();
    let input = place.join("in.jpg");
    std::fs::write(&input, &data).expect("the file is written");
    let library = Input::read(&data, &jpegio_sys::Options::default()).expect("the file is read");
    // 16 bits, level v standing for v / 257.
    let levels: Vec<u16> = (0..20 * 30_u32)
        .map(|index| u16::try_from(index.wrapping_mul(2_654_435_761) >> 16).expect("16 bits"))
        .collect();
    let values: Vec<f64> = levels.iter().map(|&level| f64::from(level) / 257.0).collect();
    let projection = picture::project(&library, &Settings::default(), &values, 1).expect("a projection");
    let expected = tiff::float64(&projection.decoded.picture, 20, 30, 1, false, None).expect("a file");
    let rgb: Vec<u16> = levels.iter().flat_map(|&level| [level; 3]).collect();
    for (name, file) in [
        ("grey.png", png_file(Samples::Sixteen(&levels), 20, 30, 1)),
        ("rgb.png", png_file(Samples::Sixteen(&rgb), 20, 30, 3)),
    ] {
        let picture_path = place.join(name);
        std::fs::write(&picture_path, file).expect("written");
        let output = place.join(format!("{name}.tif"));
        let (status, errors) = unround(&["--project-only", text(&picture_path), text(&input), text(&output)]);
        assert_eq!(status, 0, "{errors}");
        assert_eq!(errors, excess_lines(&output, &[""], &projection.excess));
        assert_eq!(std::fs::read(&output).expect("the result"), expected.bytes, "{name}");
    }
    // Pictures that are not taken.
    let mut unlike = rgb.clone();
    unlike[4] ^= 1;
    let grey_alpha: Vec<u8> = (0..20 * 30 * 2)
        .map(|index| u8::try_from(index % 251).expect("small"))
        .collect();
    let small = vec![0_u8; 19 * 30];
    for (name, file, fragment) in [
        (
            "unlike.png",
            png_file(Samples::Sixteen(&unlike), 20, 30, 3),
            "a greyscale file's picture is grey, or RGB of three channels alike",
        ),
        (
            "alpha.png",
            png_file(Samples::Eight(&grey_alpha), 20, 30, 2),
            "a picture with alpha is not a JPEG file's",
        ),
        (
            "small.png",
            png_file(Samples::Eight(&small), 19, 30, 1),
            "a picture of the file is 20 x 30 pixels, not 19 x 30",
        ),
        ("not.png", data.clone(), "the picture to project is a PNG file"),
    ] {
        let picture_path = place.join(name);
        std::fs::write(&picture_path, file).expect("written");
        let (status, errors) = unround(&[
            "--project-only",
            text(&picture_path),
            text(&input),
            text(&place.join("x.tif")),
        ]);
        assert_eq!(status, 1, "{name}: {errors}");
        assert!(errors.contains(fragment), "{fragment} in {errors}");
    }
    // A colour file's picture is RGB.
    let colour = place.join("colour.jpg");
    std::fs::write(&colour, colour_file(&[], Vec::new())).expect("written");
    let grey_path = place.join("grey.png");
    let (status, errors) = unround(&[
        "--project-only",
        text(&grey_path),
        text(&colour),
        text(&place.join("x.tif")),
    ]);
    assert_eq!(status, 1, "{errors}");
    assert!(errors.contains("a colour file's picture is RGB"), "{errors}");
    // A command line that a projection does not take.
    for (arguments, fragment) in [
        (
            vec!["--method", "tv", "--alpha", "2"],
            "--method is not taken with --project-only",
        ),
        (
            vec!["--iterations", "3"],
            "--iterations is not taken with --project-only",
        ),
    ] {
        let mut all = vec!["--project-only", text(&grey_path)];
        all.extend(arguments);
        all.extend([text(&input), "-"]);
        let (status, errors) = unround(&all);
        assert_eq!(status, 2, "{errors}");
        assert!(errors.contains(fragment), "{fragment} in {errors}");
    }
    let (status, errors) = unround(&["--project-only", "-", "-"]);
    assert_eq!(status, 2, "{errors}");
    assert!(errors.contains("not both the standard input"), "{errors}");
    // INPUT is a JPEG file.
    let (status, errors) = unround(&[
        "--project-only",
        text(&grey_path),
        text(&grey_path),
        text(&place.join("x.tif")),
    ]);
    assert_eq!(status, 2, "{errors}");
    assert!(errors.contains("not of a PNG file"), "{errors}");
    std::fs::remove_dir_all(&place).expect("the directory is removed");
}
