// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! Builds the C layer, and libjpeg-turbo, libpng and zlib-ng under it, with the
//! CMake project at the root of the repository, through the preset that matches the
//! Cargo profile (release for a release build, debug otherwise), into `OUT_DIR`;
//! then links their static libraries. The configuration checks the toolchain as
//! every build of the project does: Clang, LLD and the LLVM tools of one version,
//! first on `PATH`.
//!
//! libjpeg's encoder and standard decoding (`unround_test_jpeg`) and libpng's
//! reading (`unround_test_png`), for the tests, are built and linked as well;
//! nothing of them ends up in a program that does not call them, since a static
//! library gives the linker only what is asked for.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn run(command: &mut Command) {
    let status = command
        .status()
        .unwrap_or_else(|error| panic!("{command:?} does not run: {error}"));
    assert!(status.success(), "{command:?} failed with {status}");
}

fn main() {
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("Cargo sets CARGO_MANIFEST_DIR"));
    let root = manifest
        .parent()
        .and_then(Path::parent)
        .expect("jpegio-sys lies two directories below the root of the repository")
        .to_path_buf();
    let build = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo sets OUT_DIR")).join("cmake");
    let preset = if env::var("PROFILE").as_deref() == Ok("release") {
        "release"
    } else {
        "debug"
    };

    run(Command::new("cmake").arg("-S").arg(&root).arg("-B").arg(&build).args([
        "--preset",
        preset,
        "-DUNROUND_WITH_CPP=OFF",
        "-DUNROUND_BUILD_TESTS=ON",
    ]));
    run(Command::new("cmake").arg("--build").arg(&build).args([
        "--target",
        "unround_jpegio",
        "unround_test_jpeg",
        "unround_test_png",
    ]));

    let windows = env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows");
    println!(
        "cargo::rustc-link-search=native={}",
        build.join("c").join("jpegio").display()
    );
    for library in ["libjpeg-turbo", "libpng", "zlib-ng"] {
        println!(
            "cargo::rustc-link-search=native={}",
            build.join(library).join("install").join("lib").display()
        );
    }
    println!("cargo::rustc-link-lib=static=unround_jpegio");
    println!("cargo::rustc-link-lib=static=unround_test_jpeg");
    println!("cargo::rustc-link-lib=static=unround_test_png");
    // The names the libraries give themselves (cmake/Libjpeg.cmake, cmake/Libpng.cmake).
    let names = if windows {
        ["jpeg-static", "libpng16_static", "z"]
    } else {
        ["jpeg", "png16", "z"]
    };
    for name in names {
        println!("cargo::rustc-link-lib=static={name}");
    }
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        println!("cargo::rustc-link-lib=m");
    }

    for path in [
        "CMakeLists.txt",
        "CMakePresets.json",
        "cmake",
        "c/jpegio",
        "extern/libjpeg-turbo",
        "extern/libpng",
        "extern/zlib-ng",
    ] {
        println!("cargo::rerun-if-changed={}", root.join(path).display());
    }
}
