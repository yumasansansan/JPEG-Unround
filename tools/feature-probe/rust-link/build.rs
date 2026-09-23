// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! Compiles csrc/shim.c with Clang (C23) into a static library, with the same
//! archiver as the rest of the project, and links it.

use std::{env, path::PathBuf, process::Command};

fn main() {
    let out = PathBuf::from(env::var("OUT_DIR").expect("cargo sets OUT_DIR"));
    let windows = env::var("CARGO_CFG_TARGET_OS").expect("cargo sets the target OS") == "windows";
    let object = out.join(if windows { "shim.obj" } else { "shim.o" });
    let library = out.join(if windows { "shim.lib" } else { "libshim.a" });

    let mut clang = Command::new("clang");
    clang.args(["-std=c23", "-O2", "-Wall", "-Wextra", "-Werror", "-c", "csrc/shim.c", "-o"]);
    clang.arg(&object);
    if windows {
        // The C runtime as a DLL (/MD), as Rust links it.
        clang.arg("-fms-runtime-lib=dll");
    }
    assert!(clang.status().expect("clang runs").success(), "clang failed");

    let status = Command::new("llvm-ar")
        .arg("rcs")
        .arg(&library)
        .arg(&object)
        .status()
        .expect("llvm-ar runs");
    assert!(status.success(), "llvm-ar failed");

    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=shim");
    println!("cargo:rerun-if-changed=csrc/shim.c");
}
