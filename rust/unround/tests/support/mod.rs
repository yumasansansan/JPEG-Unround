// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

//! What the tests share: exact arithmetic, bounds of rounding error, and small
//! problems made with fixed seeds. Each test file uses a part of it.
#![allow(dead_code, reason = "each test file uses a part of the support")]

pub mod exact;
pub mod rounding;
pub mod synthetic;
