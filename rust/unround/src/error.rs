// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! What goes wrong.

use std::fmt;

/// Why a file was not reconstructed, or a value not taken.
#[derive(Debug)]
pub enum Error {
    /// The C layer could not read the file, or would not.
    Read(jpegio_sys::Error),
    /// A file whose layout JPEG-Unround does not take: sampling factors that are not
    /// whole multiples of one another, say.
    Unsupported(String),
    /// Options, or arrays given, out of their ranges: every one that is, in one
    /// message.
    Options(String),
    /// A file that could not be written or read.
    Io(std::io::Error),
    /// A PNG file that the C layer could not write.
    Write(jpegio_sys::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Read(error) => write!(formatter, "the file is not read: {error}"),
            Self::Unsupported(message) | Self::Options(message) => formatter.write_str(message),
            Self::Io(error) => write!(formatter, "{error}"),
            Self::Write(error) => write!(formatter, "the file is not written: {error}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Read(error) | Self::Write(error) => Some(error),
            Self::Io(error) => Some(error),
            Self::Unsupported(_) | Self::Options(_) => None,
        }
    }
}

impl From<jpegio_sys::Error> for Error {
    fn from(error: jpegio_sys::Error) -> Self {
        Self::Read(error)
    }
}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}
