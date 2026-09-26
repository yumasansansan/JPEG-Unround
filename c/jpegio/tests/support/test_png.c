// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// libpng's error function longjmps back to the one setjmp of test_png_read(),
// which stands as the whole controlling expression of a switch; what must survive
// the jump lives in a heap object allocated before it.

#include "test_png.h"

#include <setjmp.h>
#include <stdckdint.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <png.h>

typedef struct reader {
  jmp_buf jump;
  png_structp png;
  png_infop info;
  const uint8_t* data;
  size_t size;
  size_t at;
  png_bytepp rows;
  char message[256];
  char warning[256];
} reader;

[[noreturn]] static void fail_with(reader* r, const char* text) {
  (void)snprintf(r->message, sizeof r->message, "%s", text);
  longjmp(r->jump, 1);
}

[[noreturn]] static void on_error(png_structp png, png_const_charp text) {
  fail_with((reader*)png_get_error_ptr(png), text);
}

static void on_warning(png_structp png, png_const_charp text) {
  reader* r = (reader*)png_get_error_ptr(png);
  if (r->warning[0] == '\0') (void)snprintf(r->warning, sizeof r->warning, "%s", text);
}

static void on_read(png_structp png, png_bytep bytes, size_t length) {
  reader* r = (reader*)png_get_io_ptr(png);
  if (length > r->size - r->at) fail_with(r, "the file ends early");
  memcpy(bytes, r->data + r->at, length);
  r->at += length;
}

static bool little_endian(void) {
  const uint16_t one = 1;
  uint8_t first = 0;
  memcpy(&first, &one, 1);
  return first == 1;
}

static void read_file(reader* r, test_png* out) {
  r->png = png_create_read_struct(PNG_LIBPNG_VER_STRING, r, on_error, on_warning);
  if (r->png == nullptr) fail_with(r, "libpng could not start");
  r->info = png_create_info_struct(r->png);
  if (r->info == nullptr) fail_with(r, "libpng could not start");
  png_set_read_fn(r->png, r, on_read);
  png_set_user_limits(r->png, PNG_UINT_31_MAX, PNG_UINT_31_MAX);
  png_set_chunk_malloc_max(r->png, 0);
  png_read_info(r->png, r->info);

  png_uint_32 width = 0;
  png_uint_32 height = 0;
  int bits = 0;
  int color_type = 0;
  int interlace = 0;
  (void)png_get_IHDR(r->png, r->info, &width, &height, &bits, &color_type, &interlace, nullptr, nullptr);
  if ((color_type & PNG_COLOR_MASK_PALETTE) != 0) fail_with(r, "a palette file is not read here");
  out->width = (uint32_t)width;
  out->height = (uint32_t)height;
  out->bits = (int32_t)bits;
  out->channels = (int32_t)png_get_channels(r->png, r->info);
  out->interlaced = interlace != PNG_INTERLACE_NONE ? 1 : 0;
  if (bits < 8) png_set_packing(r->png);
  if (bits == 16 && little_endian()) png_set_swap(r->png);
  if (interlace != PNG_INTERLACE_NONE) (void)png_set_interlace_handling(r->png);
  png_read_update_info(r->png, r->info);

  png_charp name = nullptr;
  int compression = 0;
  png_bytep profile = nullptr;
  png_uint_32 profile_size = 0;
  if (png_get_iCCP(r->png, r->info, &name, &compression, &profile, &profile_size) != 0) {
    out->icc_profile = (uint8_t*)malloc(profile_size > 0 ? (size_t)profile_size : (size_t)1);
    if (out->icc_profile == nullptr) fail_with(r, "out of memory");
    if (profile_size > 0) memcpy(out->icc_profile, profile, (size_t)profile_size);
    out->icc_profile_size = (uint64_t)profile_size;
    (void)snprintf(out->icc_name, sizeof out->icc_name, "%s", name);
  }

  const size_t row_bytes = png_get_rowbytes(r->png, r->info);
  size_t total = 0;
  if (ckd_mul(&total, row_bytes, (size_t)height)) fail_with(r, "the picture is too large");
  out->samples = malloc(total > 0 ? total : 1);
  r->rows = (png_bytepp)calloc(height > 0 ? (size_t)height : (size_t)1, sizeof(png_bytep));
  if (out->samples == nullptr || r->rows == nullptr) fail_with(r, "out of memory");
  for (size_t y = 0; y < (size_t)height; ++y) r->rows[y] = (png_bytep)out->samples + y * row_bytes;
  png_read_image(r->png, r->rows);
  png_read_end(r->png, nullptr);
  (void)snprintf(out->warning, sizeof out->warning, "%s", r->warning);
}

void test_png_free(test_png* png) {
  if (png == nullptr) return;
  free(png->samples);
  free(png->icc_profile);
  *png = (test_png){};
}

int test_png_read(const uint8_t* data, size_t size, test_png* png, char* message, size_t message_size) {
  if (png == nullptr || data == nullptr) return 1;
  *png = (test_png){};
  reader* r = (reader*)calloc(1, sizeof *r);
  if (r == nullptr) {
    if (message != nullptr && message_size > 0) (void)snprintf(message, message_size, "out of memory");
    return 1;
  }
  r->data = data;
  r->size = size;
  int status = 0;
  switch (setjmp(r->jump)) {
    case 0:
      read_file(r, png);
      status = 0;
      break;
    default:
      status = 1;
      break;
  }
  png_destroy_read_struct(&r->png, &r->info, nullptr);
  free((void*)r->rows);
  if (status != 0) {
    test_png_free(png);
    if (message != nullptr && message_size > 0) (void)snprintf(message, message_size, "%s", r->message);
  }
  free(r);
  return status;
}
