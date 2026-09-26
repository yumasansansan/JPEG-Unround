// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! What the tests share: exact arithmetic, bounds of rounding error, small
//! problems made with fixed seeds, and what a JPEG file carries besides its
//! picture. Each test file uses a part of it.
#![allow(dead_code, reason = "each test file uses a part of the support")]

pub mod exact;
pub mod metadata;
pub mod rounding;
pub mod synthetic;
