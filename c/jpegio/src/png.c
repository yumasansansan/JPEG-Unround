// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// PNG files, written by libpng into memory. libpng reports an error by calling
// the error function it is given, which must not return: here it longjmps back
// to the setjmp of write_once(), as the reading of JPEG files does (jpegio.c).
// Every libpng call runs below the setjmp, and what must survive the jump --
// libpng's structures, the bytes written so far, the message -- lives in a heap
// object allocated before it. setjmp stands only where C allows it, as the whole
// controlling expression of a switch.
//
// The ICC profile is checked before libpng sees it, as libpng checks the profile
// of an iCCP chunk when it reads one and drops what fails: a profile that it would
// drop, or that its writing would fail the whole file for, is left out, and the
// file is written without it. libpng's reading also drops an iCCP chunk shorter
// than it takes, which a small profile that compresses well gives: that is found
// when the chunk is written, before any row, and the file is written again with
// the profile stored rather than compressed.

#include "unround/jpegio.h"

#include <setjmp.h>
#include <stdckdint.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <png.h>
#include <zlib.h>

// The name of the profile in the iCCP chunk, 1 to 79 Latin-1 characters.
static const char icc_name[] = "ICC profile";
// The ICC profile's header: 128 bytes, and a tag count after it.
static constexpr size_t icc_header_size = 132;
// libpng's reading of an iCCP chunk reads its first 81 bytes (the longest name,
// its end and the compression method) and then wants at least the 11 of the
// shortest zlib stream; it drops a shorter chunk (pngrutil.c, png_handle_iCCP).
// A stored profile makes a longer chunk, as a profile has at least 132 bytes.
static constexpr uint32_t iccp_least = 81 + 11;
// libpng's readers leave out an ICC profile of more bytes than this, unless they
// allow more (png_set_chunk_malloc_max).
static constexpr uint64_t icc_read_limit = PNG_USER_CHUNK_MALLOC_MAX;

typedef struct writer {
  jmp_buf jump;
  png_structp png;
  png_infop info;
  unround_jpegio_status failure;
  bool store_icc;  // the profile is stored, not compressed
  bool again;      // the chunk of the compressed profile was too short
  uint8_t* data;
  size_t size;
  size_t capacity;
  char message[256];
  char warning[256];
} writer;

[[noreturn]] static void fail_with(writer* w, unround_jpegio_status status, const char* text) {
  w->failure = status;
  (void)snprintf(w->message, sizeof w->message, "%s", text);
  longjmp(w->jump, 1);
}

// Leaves this writing of the file, for another with the profile stored.
[[noreturn]] static void start_over(writer* w) {
  w->again = true;
  w->store_icc = true;
  longjmp(w->jump, 1);
}

[[noreturn]] static void on_error(png_structp png, png_const_charp text) {
  fail_with((writer*)png_get_error_ptr(png), UNROUND_JPEGIO_ERROR_ENCODE, text);
}

// The first warning is kept as the message.
static void on_warning(png_structp png, png_const_charp text) {
  writer* w = (writer*)png_get_error_ptr(png);
  if (w->warning[0] == '\0') (void)snprintf(w->warning, sizeof w->warning, "libpng: %s", text);
}

// Appends the bytes libpng writes, doubling the room as it runs out.
static void on_write(png_structp png, png_bytep bytes, size_t length) {
  writer* w = (writer*)png_get_io_ptr(png);
  size_t needed = 0;
  if (ckd_add(&needed, w->size, length)) fail_with(w, UNROUND_JPEGIO_ERROR_LIMIT, "the file is too large");
  if (needed > w->capacity) {
    size_t capacity = w->capacity > 0 ? w->capacity : (size_t)65536;
    while (capacity < needed) {
      if (ckd_mul(&capacity, capacity, (size_t)2)) {
        capacity = needed;
        break;
      }
    }
    uint8_t* grown = (uint8_t*)realloc(w->data, capacity);
    if (grown == nullptr) fail_with(w, UNROUND_JPEGIO_ERROR_MEMORY, "out of memory");
    w->data = grown;
    w->capacity = capacity;
  }
  if (length > 0) memcpy(w->data + w->size, bytes, length);
  w->size = needed;
}

// Nothing is buffered outside the bytes themselves.
static void on_flush(png_structp /* png */) {}

static void copy_message(char* message, size_t message_size, const char* text) {
  if (message == nullptr || message_size == 0) return;
  (void)snprintf(message, message_size, "%s", text);
}

static uint32_t big_endian_u32(const uint8_t* p) {
  return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | (uint32_t)p[3];
}

// A signature of four characters, as a big-endian number.
static uint32_t signature_of(char a, char b, char c, char d) {
  return ((uint32_t)(uint8_t)a << 24) | ((uint32_t)(uint8_t)b << 16) | ((uint32_t)(uint8_t)c << 8) |
         (uint32_t)(uint8_t)d;
}

// Four bytes of a profile as text, with what is not printable as '?'.
static void signature_text(const uint8_t* p, char text[5]) {
  for (size_t i = 0; i < 4; ++i) text[i] = (p[i] >= 0x20 && p[i] < 0x7F) ? (char)p[i] : '?';
  text[4] = '\0';
}

// Why an ICC profile does not go with a picture of the channels, into reason; or
// false when it does. The checks are libpng's, in its order (png.c,
// png_icc_check_length, png_icc_check_header and png_icc_check_tag_table).
static bool icc_refused(const uint8_t* profile, uint64_t size, int32_t channels, char* reason, size_t reason_size) {
  if (profile == nullptr) {
    (void)snprintf(reason, reason_size, "no profile is given");
    return true;
  }
  if (channels != 1 && channels != 3) {
    (void)snprintf(reason, reason_size, "a picture here has 1 or 3 channels, not %d", (int)channels);
    return true;
  }
  if (size < icc_header_size) {
    (void)snprintf(reason, reason_size, "it is %llu bytes, fewer than the %zu of its header", (unsigned long long)size,
                   icc_header_size);
    return true;
  }
  if (size > (uint64_t)PNG_UINT_31_MAX) {
    (void)snprintf(reason, reason_size, "it is %llu bytes, more than a PNG chunk holds", (unsigned long long)size);
    return true;
  }
  const uint32_t declared = big_endian_u32(profile);
  if ((uint64_t)declared != size) {
    (void)snprintf(reason, reason_size, "its header says %lu bytes, and it is %llu", (unsigned long)declared,
                   (unsigned long long)size);
    return true;
  }
  if (profile[8] > 3 && size % 4 != 0) {
    (void)snprintf(reason, reason_size, "it is of version %u and %llu bytes, not a multiple of 4", (unsigned)profile[8],
                   (unsigned long long)size);
    return true;
  }
  const uint64_t tags = (uint64_t)big_endian_u32(profile + 128);
  if (tags > (size - icc_header_size) / 12) {
    (void)snprintf(reason, reason_size, "its %llu tags do not fit in it", (unsigned long long)tags);
    return true;
  }
  const uint32_t intent = big_endian_u32(profile + 64);
  if (intent >= 0xFFFFU) {
    (void)snprintf(reason, reason_size, "its rendering intent is %lu", (unsigned long)intent);
    return true;
  }
  char text[5] = {};
  if (big_endian_u32(profile + 36) != signature_of('a', 'c', 's', 'p')) {
    signature_text(profile + 36, text);
    (void)snprintf(reason, reason_size, "its signature is '%s', not 'acsp'", text);
    return true;
  }
  const uint32_t space = big_endian_u32(profile + 16);
  const uint32_t expected = channels == 1 ? signature_of('G', 'R', 'A', 'Y') : signature_of('R', 'G', 'B', ' ');
  if (space != expected) {
    signature_text(profile + 16, text);
    (void)snprintf(reason, reason_size, "its data color space is '%s', not the picture's '%s'", text,
                   channels == 1 ? "GRAY" : "RGB ");
    return true;
  }
  const uint32_t profile_class = big_endian_u32(profile + 12);
  if (profile_class == signature_of('a', 'b', 's', 't') || profile_class == signature_of('l', 'i', 'n', 'k')) {
    signature_text(profile + 12, text);
    (void)snprintf(reason, reason_size, "it is of the class '%s', which does not go with a picture", text);
    return true;
  }
  const uint32_t connection = big_endian_u32(profile + 20);
  if (connection != signature_of('X', 'Y', 'Z', ' ') && connection != signature_of('L', 'a', 'b', ' ')) {
    signature_text(profile + 20, text);
    (void)snprintf(reason, reason_size, "its connection space is '%s', neither 'XYZ ' nor 'Lab '", text);
    return true;
  }
  for (uint64_t i = 0; i < tags; ++i) {
    const uint8_t* tag = profile + icc_header_size + 12 * (size_t)i;
    const uint64_t start = (uint64_t)big_endian_u32(tag + 4);
    const uint64_t length = (uint64_t)big_endian_u32(tag + 8);
    if (start > size || length > size - start) {
      signature_text(tag, text);
      (void)snprintf(reason, reason_size, "its tag '%s' lies outside it", text);
      return true;
    }
  }
  return false;
}

// Whether the iCCP chunk among the bytes written is shorter than libpng's reading
// takes.
static bool iccp_too_short(const writer* w) {
  size_t at = 8;  // after the PNG signature
  while (at <= w->size && w->size - at >= 8) {
    const uint32_t length = big_endian_u32(w->data + at);
    if (memcmp(w->data + at + 4, "iCCP", 4) == 0) return length < iccp_least;
    if (ckd_add(&at, at, (size_t)12) || ckd_add(&at, at, (size_t)length)) return false;
  }
  return false;
}

// Whether this machine keeps the most significant byte of a uint16_t second.
static bool little_endian(void) {
  const uint16_t one = 1;
  uint8_t first = 0;
  memcpy(&first, &one, 1);
  return first == 1;
}

// Why the picture cannot be written, or a null pointer; and the bytes of a row.
static const char* picture_refused(const unround_jpegio_png* png, size_t* row_bytes) {
  if (png->width == 0 || png->height == 0) return "the picture is empty";
  if (png->width > PNG_UINT_31_MAX || png->height > PNG_UINT_31_MAX) return "the picture is larger than PNG allows";
  if (png->channels != 1 && png->channels != 3) return "a PNG picture here has 1 or 3 channels";
  if (png->bits != 8 && png->bits != 16) return "a PNG picture here has 8 or 16 bits a sample";
  if (png->compression < -1 || png->compression > 9) return "the compression is -1 or 0 to 9";
  if (png->reserved != 0) return "the reserved field is 0";
  if (png->samples == nullptr) return "no samples are given";
  if (png->icc_profile == nullptr && png->icc_profile_size > 0) return "no ICC profile is given for its size";
  size_t bytes = 0;
  size_t total = 0;
  if (ckd_mul(&bytes, (size_t)png->width, (size_t)png->channels) || ckd_mul(&bytes, bytes, (size_t)(png->bits / 8)) ||
      ckd_mul(&total, bytes, (size_t)png->height)) {
    return "the picture is too large";
  }
  *row_bytes = bytes;
  return nullptr;
}

static void write_file(writer* w, const unround_jpegio_png* png, size_t row_bytes, bool with_icc) {
  w->png = png_create_write_struct(PNG_LIBPNG_VER_STRING, w, on_error, on_warning);
  if (w->png == nullptr) fail_with(w, UNROUND_JPEGIO_ERROR_MEMORY, "libpng could not start");
  w->info = png_create_info_struct(w->png);
  if (w->info == nullptr) fail_with(w, UNROUND_JPEGIO_ERROR_MEMORY, "libpng could not start");
  png_set_write_fn(w->png, w, on_write, on_flush);
  // libpng refuses by default what is wider or taller than a million samples.
  png_set_user_limits(w->png, PNG_UINT_31_MAX, PNG_UINT_31_MAX);
  // The level is that of the samples and that of the profile, which libpng
  // compresses as it does text.
  if (png->compression >= 0) {
    png_set_compression_level(w->png, png->compression);
    png_set_text_compression_level(w->png, png->compression);
  }
  if (w->store_icc) png_set_text_compression_level(w->png, 0);
  png_set_IHDR(w->png, w->info, (png_uint_32)png->width, (png_uint_32)png->height, png->bits,
               png->channels == 1 ? PNG_COLOR_TYPE_GRAY : PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
               PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
  if (with_icc) {
    png_set_iCCP(w->png, w->info, icc_name, PNG_COMPRESSION_TYPE_BASE, png->icc_profile,
                 (png_uint_32)png->icc_profile_size);
  }
  png_write_info(w->png, w->info);
  if (with_icc && !w->store_icc && iccp_too_short(w)) start_over(w);
  // PNG keeps the most significant byte of a 16-bit sample first.
  if (png->bits == 16 && little_endian()) png_set_swap(w->png);
  const uint8_t* rows = (const uint8_t*)png->samples;
  for (size_t y = 0; y < (size_t)png->height; ++y) png_write_row(w->png, rows + y * row_bytes);
  png_write_end(w->png, nullptr);
}

// One writing of the file: libpng's errors jump back to the setjmp here.
static unround_jpegio_status write_once(writer* w, const unround_jpegio_png* png, size_t row_bytes, bool with_icc) {
  unround_jpegio_status status = UNROUND_JPEGIO_OK;
  switch (setjmp(w->jump)) {
    case 0:
      write_file(w, png, row_bytes, with_icc);
      status = UNROUND_JPEGIO_OK;
      break;
    default:
      status = w->failure;
      break;
  }
  png_destroy_write_struct(&w->png, &w->info);
  return status;
}

int32_t unround_jpegio_check_icc(const uint8_t* profile, uint64_t size, int32_t channels, char* reason,
                                 size_t reason_size) {
  char why[160] = {};
  const bool refused = icc_refused(profile, size, channels, why, sizeof why);
  if (reason != nullptr && reason_size > 0) (void)snprintf(reason, reason_size, "%s", why);
  return refused ? 0 : 1;
}

const char* unround_jpegio_libpng_version(void) { return "libpng " PNG_LIBPNG_VER_STRING ", zlib-ng " ZLIBNG_VERSION; }

void unround_jpegio_bytes_free(unround_jpegio_bytes* bytes) {
  if (bytes == nullptr) return;
  free(bytes->data);
  *bytes = (unround_jpegio_bytes){};
}

unround_jpegio_status unround_jpegio_write_png(const unround_jpegio_png* png, unround_jpegio_bytes* file, char* message,
                                               size_t message_size) {
  if (file == nullptr || (message == nullptr && message_size > 0)) return UNROUND_JPEGIO_ERROR_ARGUMENT;
  *file = (unround_jpegio_bytes){};
  if (png == nullptr) {
    copy_message(message, message_size, "no picture is given");
    return UNROUND_JPEGIO_ERROR_ARGUMENT;
  }
  size_t row_bytes = 0;
  const char* refused = picture_refused(png, &row_bytes);
  if (refused != nullptr) {
    copy_message(message, message_size, refused);
    return UNROUND_JPEGIO_ERROR_ARGUMENT;
  }
  writer* w = (writer*)calloc(1, sizeof *w);
  if (w == nullptr) {
    copy_message(message, message_size, "out of memory");
    return UNROUND_JPEGIO_ERROR_MEMORY;
  }
  bool with_icc = false;
  if (png->icc_profile != nullptr) {
    char reason[160] = {};
    with_icc = (bool)!icc_refused(png->icc_profile, png->icc_profile_size, png->channels, reason, sizeof reason);
    if (!with_icc) {
      (void)snprintf(w->warning, sizeof w->warning, "the ICC profile is not written: %s", reason);
    } else if (png->icc_profile_size > icc_read_limit) {
      (void)snprintf(w->warning, sizeof w->warning,
                     "the ICC profile is written, but it is %llu bytes, and libpng's readers leave out one of more "
                     "than %llu unless they allow more",
                     (unsigned long long)png->icc_profile_size, (unsigned long long)icc_read_limit);
    }
  }
  unround_jpegio_status status = write_once(w, png, row_bytes, with_icc);
  if (status == UNROUND_JPEGIO_OK && w->again) {
    w->again = false;
    w->size = 0;
    status = write_once(w, png, row_bytes, with_icc);
  }
  if (status != UNROUND_JPEGIO_OK) {
    free(w->data);
    copy_message(message, message_size, w->message);
  } else {
    file->data = w->data;
    file->size = (uint64_t)w->size;
    copy_message(message, message_size, w->warning);
  }
  free(w);
  return status;
}
