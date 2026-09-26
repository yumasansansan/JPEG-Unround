// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! JPEG decoding that removes compression artifacts within the quantization
//! intervals: the reference implementation of JPEG-Unround.
//!
//! The quantized DCT coefficients of a file are taken as intervals, not values, and
//! of the pictures whose coefficients lie within them, the one that total variation
//! or total generalized variation finds most natural is found by the primal-dual
//! method. The mathematics is docs/math.md's.
//!
//! - [`dct`]: the 8x8 block DCT of JPEG.
//! - [`laplace`]: the Laplace model of AC coefficients and their MMSE centres.
//! - [`operators`]: finite differences, their adjoints, and pointwise norms.
//! - [`model`]: the quantization constraint set and the data term of a component.
//! - [`frames`]: the components of a file on one canvas, and the objectives and gaps.
//! - [`pdhg`]: the primal-dual method for TV and TGV.
//! - [`subgradient`]: a subgradient method of jpeg2png's kind, to compare with.
//! - [`results`]: what the solvers return.
//! - [`colour`]: the YCbCr of JFIF, and its inverse.
//! - [`decode`]: a file, reconstructed by one of the methods.
//! - [`tiff`]: the result in binary64, written bit for bit.
//! - [`exact`]: what is rational, computed exactly, and sums in a fixed order.
//! - [`output`]: pictures of 8 or 16 bits, and PNM files.
//! - [`cli`]: the command line, `unround` (docs/cli.md).
#![forbid(unsafe_code)]

pub mod cli;
pub mod colour;
pub mod dct;
pub mod decode;
pub mod error;
pub mod exact;
pub mod frames;
pub mod laplace;
pub mod model;
pub mod operators;
pub mod output;
pub mod pdhg;
pub mod results;
pub mod subgradient;
pub mod tiff;

pub use decode::{Decoded, Method, Settings, decode};
pub use error::Error;
