// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// PNG files read back by libpng, for the tests of the C layer and of what is built
// on it: the samples as the file holds them (16-bit ones in this machine's byte
// order, with no gamma or other transformation), the profile of an iCCP chunk, and
// libpng's first warning. libpng reads them with its defaults, but for pictures of
// any width and height, and chunks of any size.
//
// Nothing here is part of JPEG-Unround; the tests link it. Like the C layer, it
// keeps libpng's longjmp to itself.
#ifndef UNROUND_TEST_PNG_H
#define UNROUND_TEST_PNG_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct test_png {
  uint32_t width;
  uint32_t height;
  int32_t channels;  // 1 (gray), 2 (gray and alpha), 3 (RGB) or 4 (RGBA)
  int32_t bits;      // 1, 2, 4, 8 or 16; samples of fewer than 8 bits are one to a byte
  int32_t interlaced;
  int32_t reserved;
  // height * width * channels samples, rows from the top: uint8_t for 8 bits or
  // fewer, uint16_t for 16. Allocated by malloc().
  void* samples;
  uint8_t* icc_profile;  // the profile of the iCCP chunk, or a null pointer
  uint64_t icc_profile_size;
  char icc_name[80];  // the name of the iCCP profile
  char warning[256];  // libpng's first warning in reading the file, or an empty string
} test_png;

// Returns 0 on success, with png filled in, or nonzero with a NUL-terminated
// description in message. A palette file is refused.
int test_png_read(const uint8_t* data, size_t size, test_png* png, char* message, size_t message_size);

// Frees what a successful test_png_read() allocated, and empties png.
void test_png_free(test_png* png);

#ifdef __cplusplus
}
#endif

#endif
