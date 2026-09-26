// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The C interface's header, compiled as C23 with every warning of the build: its
// structures have the sizes and offsets that it asserts, which are those that the
// Rust crate asserts of its own (rust/capi/src/lib.rs).
#include "unround.h"

#include <stddef.h>

static_assert(offsetof(unround_record, canvas) == 56, "the layout of unround_record");
static_assert(offsetof(unround_component, rows) == 8, "the layout of unround_component");
static_assert(offsetof(unround_component, h_samp_factor) == 16, "the layout of unround_component");
static_assert(offsetof(unround_input, color_space) == 8, "the layout of unround_input");
