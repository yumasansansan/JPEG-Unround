// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! The command line (docs/cli.md): the program that the crate builds, run on files
//! written here.

mod support;

use std::path::PathBuf;
use std::process::Command;

use jpeg_unround::decode::{self, Method, Settings};
use jpeg_unround::{pdhg, tiff};
use jpegio_sys::testing::{File, Layout};
use support::synthetic;

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
    let expected = tiff::float64(&decoded.picture, 20, 30, 1, false).expect("a file");
    assert_eq!(std::fs::read(&output).expect("the result"), expected);
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
