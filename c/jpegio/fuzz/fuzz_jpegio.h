// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The fuzz target of the C layer (fuzz_jpegio.c), which libFuzzer calls in a
// fuzzing build and smoke.c calls in every build.
#ifndef UNROUND_FUZZ_JPEGIO_H
#define UNROUND_FUZZ_JPEGIO_H

#include <stddef.h>
#include <stdint.h>

// Reads the bytes as a JPEG file and decodes its planes, and aborts when the C
// layer breaks a promise about what it returns. It always returns 0.
int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size);

#endif
