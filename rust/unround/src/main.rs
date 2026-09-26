// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! `unround`, the command line of JPEG-Unround (docs/cli.md).
#![forbid(unsafe_code)]

fn main() -> std::process::ExitCode {
    jpeg_unround::cli::main()
}
