// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// JPEG files for the tests of the C layer and of what is built on it, made by
// libjpeg's own encoder: from coefficients a test chooses, so that what a read
// returns can be compared with what went in, or, for the lossless process, from
// samples. And libjpeg's standard decoding, for comparing the planes with.
//
// Nothing here is part of JPEG-Unround; the tests link it. Like the C layer, it
// keeps libjpeg's longjmp to itself.
#ifndef UNROUND_TEST_JPEG_H
#define UNROUND_TEST_JPEG_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TEST_JPEG_MAX_COMPONENTS 4

typedef enum test_jpeg_color_space : int32_t {
  TEST_JPEG_GRAYSCALE = 1,
  TEST_JPEG_YCBCR = 2,
  TEST_JPEG_RGB = 3,
  TEST_JPEG_CMYK = 4,
} test_jpeg_color_space;

// What to encode. Fields left zero take the value that makes sense: sampling
// factors of 1, precision 8, no restart markers.
typedef struct test_jpeg {
  uint32_t width;
  uint32_t height;
  int32_t color_space;  // test_jpeg_color_space; also says how many components there are
  int32_t h_samp_factor[TEST_JPEG_MAX_COMPONENTS];
  int32_t v_samp_factor[TEST_JPEG_MAX_COMPONENTS];
  int32_t quant_table_slot[TEST_JPEG_MAX_COMPONENTS];
  uint16_t quant_tables[TEST_JPEG_MAX_COMPONENTS][64];  // by slot, natural order, 1 to 32767
  // test_jpeg_blocks_wide() * test_jpeg_blocks_high() blocks of 64 per
  // component, block rows from the top, natural order.
  const int16_t* coefficients[TEST_JPEG_MAX_COMPONENTS];
  int32_t data_precision;    // 8, or 12 for a file the C layer refuses
  int32_t progressive;       // libjpeg's simple progression script
  int32_t sequential_scans;  // nonzero: one scan per component, in order, rather than one interleaved scan
  int32_t arithmetic;
  int32_t optimize_coding;
  int32_t restart_interval;  // in MCUs
  const uint8_t* icc_profile;
  uint32_t icc_profile_size;
  const uint8_t* app1;  // the payload of an APP1 marker (EXIF), or a null pointer
  uint32_t app1_size;
} test_jpeg;

int32_t test_jpeg_components(const test_jpeg* spec);

// The blocks of a component that carry image data, as libjpeg counts them.
uint32_t test_jpeg_blocks_wide(const test_jpeg* spec, int32_t component);
uint32_t test_jpeg_blocks_high(const test_jpeg* spec, int32_t component);

// Each returns 0 on success, with *data allocated by malloc() and *size its
// length, or nonzero with a NUL-terminated description in message.
int test_jpeg_write(const test_jpeg* spec, uint8_t** data, size_t* size, char* message, size_t message_size);

// An 8-bit grayscale file of the lossless process, from width * height samples.
int test_jpeg_write_lossless(uint32_t width, uint32_t height, const uint8_t* samples, uint8_t** data, size_t* size,
                             char* message, size_t message_size);

// libjpeg's standard decoding with the inverse DCT the C layer uses and no
// block smoothing, fancy upsampling or color conversion: *samples holds width *
// height * components samples, interleaved, allocated by malloc().
int test_jpeg_decode(const uint8_t* data, size_t size, uint8_t** samples, uint32_t* width, uint32_t* height,
                     int32_t* components, char* message, size_t message_size);

#ifdef __cplusplus
}
#endif

#endif
