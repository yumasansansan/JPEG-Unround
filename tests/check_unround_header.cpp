// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The C interface's header, compiled as C++ with every warning of the build.
#include "unround.h"

#include <cstddef>

static_assert(offsetof(unround_record, canvas) == 56, "the layout of unround_record");
