// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Tests of the C layer's writing of PNG files. Pictures of every kind it writes --
// gray and RGB, 8 and 16 bits, every level of compression, sizes down to a pixel
// and up to several megabytes -- are read back by libpng (support/test_png.c), and
// what comes back has to be what went in, exactly, with no warning from libpng. An
// ICC profile has to come back as it went in where it is well formed and of the
// picture's color space, however small or large, and be left out, with a warning,
// where it is not. Then the refusals of wrong arguments.

#include "test_png.h"
#include "unround/jpegio.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
static int checks = 0;

static bool check_at(bool ok, const char* text, const char* file, int line) {
  checks += 1;
  if (!ok) {
    failures += 1;
    (void)fprintf(stderr, "%s:%d: check failed: %s\n", file, line, text);
  }
  return ok;
}

#define CHECK(condition) check_at((condition), #condition, __FILE__, __LINE__)

static void note(const char* what) { (void)fprintf(stderr, "  %s\n", what); }

// Deterministic pseudo-random numbers (SplitMix64).
typedef struct rng {
  uint64_t state;
} rng;

static uint64_t rng_next(rng* g) {
  g->state += UINT64_C(0x9E3779B97F4A7C15);
  uint64_t z = g->state;
  z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
  z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
  return z ^ (z >> 31);
}

static const uint8_t signature[8] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'};

// Random samples of a picture: every value of the depth is as likely.
static void* random_samples(rng* g, uint32_t width, uint32_t height, int32_t channels, int32_t bits) {
  const size_t count = (size_t)width * (size_t)height * (size_t)channels;
  if (bits == 8) {
    uint8_t* samples = (uint8_t*)malloc(count);
    if (samples == nullptr) return nullptr;
    for (size_t i = 0; i < count; ++i) samples[i] = (uint8_t)(rng_next(g) & 0xFFU);
    return samples;
  }
  uint16_t* samples = (uint16_t*)malloc(count * sizeof(uint16_t));
  if (samples == nullptr) return nullptr;
  for (size_t i = 0; i < count; ++i) samples[i] = (uint16_t)(rng_next(g) & 0xFFFFU);
  return samples;
}

static unround_jpegio_png picture_of(uint32_t width, uint32_t height, int32_t channels, int32_t bits,
                                     int32_t compression, const void* samples) {
  return (unround_jpegio_png){
      .width = width,
      .height = height,
      .channels = channels,
      .bits = bits,
      .compression = compression,
      .samples = samples,
  };
}

// Writes the picture and reads it back; checks that the file is a PNG file of the
// picture, and returns it read back in *back (to be freed), with the message; and
// the file in *kept (to be freed) when kept is not a null pointer.
static bool write_and_keep(const unround_jpegio_png* png, test_png* back, char* message, size_t message_size,
                           unround_jpegio_bytes* kept) {
  unround_jpegio_bytes file = {};
  const unround_jpegio_status status = unround_jpegio_write_png(png, &file, message, message_size);
  if (!CHECK(status == UNROUND_JPEGIO_OK)) {
    note(message);
    return false;
  }
  bool ok = CHECK(file.size > sizeof signature) && CHECK(memcmp(file.data, signature, sizeof signature) == 0);
  char read_message[256] = {};
  if (ok && !CHECK(test_png_read(file.data, (size_t)file.size, back, read_message, sizeof read_message) == 0)) {
    note(read_message);
    ok = false;
  }
  if (kept != nullptr) {
    *kept = file;
    return ok;
  }
  unround_jpegio_bytes_free(&file);
  CHECK(file.data == nullptr && file.size == 0);
  return ok;
}

static bool write_and_read(const unround_jpegio_png* png, test_png* back, char* message, size_t message_size) {
  return write_and_keep(png, back, message, message_size, nullptr);
}

static void check_same_picture(const unround_jpegio_png* png, const test_png* back) {
  CHECK(back->width == png->width);
  CHECK(back->height == png->height);
  CHECK(back->channels == png->channels);
  CHECK(back->bits == png->bits);
  CHECK(back->interlaced == 0);
  const size_t bytes = (size_t)png->width * (size_t)png->height * (size_t)png->channels * (size_t)(png->bits / 8);
  CHECK(memcmp(back->samples, png->samples, bytes) == 0);
}

// No warning from libpng in reading the file back.
static void check_no_warning(const test_png* back, const char* what) {
  if (!CHECK(back->warning[0] == '\0')) {
    note(back->warning);
    note(what);
  }
}

static void test_round_trips(void) {
  static const uint32_t sizes[][2] = {{1, 1}, {7, 5}, {33, 17}, {256, 3}, {3, 256}};
  static const int32_t levels[] = {-1, 0, 1, 6, 9};
  rng g = {.state = 1};
  for (int32_t channels = 1; channels <= 3; channels += 2) {
    for (int32_t bits = 8; bits <= 16; bits += 8) {
      for (size_t s = 0; s < sizeof sizes / sizeof sizes[0]; ++s) {
        for (size_t l = 0; l < sizeof levels / sizeof levels[0]; ++l) {
          void* samples = random_samples(&g, sizes[s][0], sizes[s][1], channels, bits);
          if (!CHECK(samples != nullptr)) return;
          const unround_jpegio_png png = picture_of(sizes[s][0], sizes[s][1], channels, bits, levels[l], samples);
          test_png back = {};
          char message[256] = {};
          if (write_and_read(&png, &back, message, sizeof message)) {
            check_same_picture(&png, &back);
            check_no_warning(&back, "a picture");
            CHECK(back.icc_profile == nullptr);
            CHECK(message[0] == '\0');
          }
          test_png_free(&back);
          free(samples);
        }
      }
    }
  }
}

// A picture of several megabytes, whose file grows its buffer many times.
static void test_large(void) {
  rng g = {.state = 2};
  void* samples = random_samples(&g, 1000, 700, 3, 16);
  if (!CHECK(samples != nullptr)) return;
  const unround_jpegio_png png = picture_of(1000, 700, 3, 16, 1, samples);
  test_png back = {};
  char message[256] = {};
  if (write_and_read(&png, &back, message, sizeof message)) {
    check_same_picture(&png, &back);
    check_no_warning(&back, "a large picture");
  }
  test_png_free(&back);
  free(samples);
}

// Every level of compression gives the same picture, and more of it a file no
// larger for a picture that compresses.
static void test_levels(void) {
  enum { width = 200, height = 100 };
  static uint8_t samples[width * height];
  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; ++x) samples[y * width + x] = (uint8_t)((x + y) / 3);
  }
  uint64_t none = 0;
  uint64_t most = 0;
  for (int32_t level = 0; level <= 9; level += 9) {
    const unround_jpegio_png png = picture_of(width, height, 1, 8, level, samples);
    unround_jpegio_bytes file = {};
    char message[256] = {};
    if (!CHECK(unround_jpegio_write_png(&png, &file, message, sizeof message) == UNROUND_JPEGIO_OK)) continue;
    if (level == 0) none = file.size;
    if (level == 9) most = file.size;
    unround_jpegio_bytes_free(&file);
  }
  CHECK(most > 0 && most < none);
}

static void put32(uint8_t* p, uint32_t value) {
  p[0] = (uint8_t)(value >> 24);
  p[1] = (uint8_t)(value >> 16);
  p[2] = (uint8_t)(value >> 8);
  p[3] = (uint8_t)value;
}

// Four characters of a signature, without the end of the text that gives them.
static void put_signature(uint8_t* p, const char* text) { memcpy(p, text, 4); }

// A well-formed ICC profile of the size, version, class, data color space and
// connection space given (a header, and the tags given, each four bytes after the
// tag table), its other bytes counting up. A profile shorter than its header is
// those bytes, its length first. A null pointer when the tags do not fit.
static uint8_t* profile_of(size_t size, uint8_t version, const char* profile_class, const char* space,
                           const char* connection, uint32_t tags) {
  if (size >= 132 && tags > 0 && size < 132 + 12 * (size_t)tags + 4) return nullptr;
  uint8_t* p = (uint8_t*)malloc(size > 0 ? size : 1);
  if (p == nullptr) return nullptr;
  for (size_t i = 0; i < size; ++i) p[i] = (uint8_t)(i * 7U);
  if (size >= 4) put32(p, (uint32_t)size);
  if (size < 132) return p;
  memset(p + 4, 0, 128);
  p[8] = version;
  put_signature(p + 12, profile_class);
  put_signature(p + 16, space);
  put_signature(p + 20, connection);
  put_signature(p + 36, "acsp");
  put32(p + 68, UINT32_C(0x0000F6D6));  // the illuminant D50, in s15Fixed16
  put32(p + 72, UINT32_C(0x00010000));
  put32(p + 76, UINT32_C(0x0000D32D));
  put32(p + 128, tags);
  const uint32_t data = (uint32_t)(132 + 12 * (size_t)tags);
  for (uint32_t i = 0; i < tags; ++i) {
    uint8_t* tag = p + 132 + 12 * (size_t)i;
    memcpy(tag, "desc", 4);
    tag[3] = (uint8_t)('a' + i);
    put32(tag + 4, data);
    put32(tag + 8, 4);
  }
  return p;
}

// Writes a picture of the channels with the profile, and checks that the profile
// comes back, or that it is left out with a warning; and that the C layer's check
// of it says the same. libpng's reading warns of a profile of the class named color
// (which it keeps), and of nothing else.
static void check_icc(int32_t channels, const uint8_t* profile, size_t size, bool written, const char* what) {
  if (!CHECK(profile != nullptr)) {
    note(what);
    return;
  }
  rng g = {.state = 3};
  void* samples = random_samples(&g, 9, 4, channels, 8);
  if (!CHECK(samples != nullptr)) return;
  char reason[160] = {};
  if (!CHECK(unround_jpegio_check_icc(profile, (uint64_t)size, channels, reason, sizeof reason) == (written ? 1 : 0))) {
    note(what);
  }
  CHECK(written == (reason[0] == '\0'));
  unround_jpegio_png png = picture_of(9, 4, channels, 8, -1, samples);
  png.icc_profile = profile;
  png.icc_profile_size = (uint64_t)size;
  test_png back = {};
  char message[256] = {};
  if (write_and_read(&png, &back, message, sizeof message)) {
    check_same_picture(&png, &back);
    if (written) {
      if (!CHECK(message[0] == '\0')) note(message);
      if (!CHECK(back.icc_profile != nullptr && back.icc_profile_size == (uint64_t)size &&
                 memcmp(back.icc_profile, profile, size) == 0)) {
        note(what);
      }
      CHECK(strcmp(back.icc_name, "ICC profile") == 0);
      if (memcmp(profile + 12, "nmcl", 4) == 0) {
        CHECK(strstr(back.warning, "unexpected NamedColor ICC profile class") != nullptr);
      } else {
        check_no_warning(&back, what);
      }
    } else {
      if (!CHECK(strncmp(message, "the ICC profile is not written: ", 32) == 0)) note(what);
      CHECK(strstr(message, reason) != nullptr);
      CHECK(back.icc_profile == nullptr);
      check_no_warning(&back, what);
    }
  }
  test_png_free(&back);
  free(samples);
}

static void test_icc(void) {
  // Well formed: of versions 2 and 4, gray and RGB, every class but the two that
  // do not go with a picture, with tags and without, XYZ and Lab.
  uint8_t* p = profile_of(132, 2, "mntr", "GRAY", "XYZ ", 0);
  check_icc(1, p, 132, true, "a gray profile of 132 bytes");
  free(p);
  p = profile_of(133, 2, "scnr", "RGB ", "Lab ", 0);
  check_icc(3, p, 133, true, "a profile of version 2 and 133 bytes");
  free(p);
  p = profile_of(3000, 4, "prtr", "RGB ", "XYZ ", 3);
  check_icc(3, p, 3000, true, "a profile of version 4 with three tags");
  free(p);
  p = profile_of(200, 2, "nmcl", "RGB ", "XYZ ", 1);
  check_icc(3, p, 200, true, "a named color profile");
  free(p);

  // What libpng drops a profile for, one thing at a time.
  p = profile_of(131, 2, "mntr", "RGB ", "XYZ ", 0);
  check_icc(3, p, 131, false, "shorter than its header");
  free(p);
  p = profile_of(0, 2, "mntr", "RGB ", "XYZ ", 0);
  check_icc(3, p, 0, false, "empty");
  free(p);
  p = profile_of(200, 2, "mntr", "RGB ", "XYZ ", 0);
  put32(p, 201);
  check_icc(3, p, 200, false, "of another length than it says");
  free(p);
  p = profile_of(202, 4, "mntr", "RGB ", "XYZ ", 0);
  check_icc(3, p, 202, false, "of version 4 and not a multiple of 4");
  free(p);
  p = profile_of(200, 2, "mntr", "RGB ", "XYZ ", 0);
  put32(p + 128, 6);
  check_icc(3, p, 200, false, "with more tags than fit");
  free(p);
  p = profile_of(200, 2, "mntr", "RGB ", "XYZ ", 1);
  put32(p + 132 + 4, 198);
  check_icc(3, p, 200, false, "with a tag outside it");
  free(p);
  p = profile_of(200, 2, "mntr", "RGB ", "XYZ ", 0);
  put32(p + 64, 0xFFFF);
  check_icc(3, p, 200, false, "with a rendering intent of 0xFFFF");
  free(p);
  p = profile_of(200, 2, "mntr", "RGB ", "XYZ ", 0);
  put_signature(p + 36, "ascp");
  check_icc(3, p, 200, false, "without its signature");
  free(p);
  p = profile_of(200, 2, "abst", "RGB ", "XYZ ", 0);
  check_icc(3, p, 200, false, "abstract");
  free(p);
  p = profile_of(200, 2, "link", "RGB ", "XYZ ", 0);
  check_icc(3, p, 200, false, "a device link");
  free(p);
  p = profile_of(200, 2, "mntr", "RGB ", "XYZ ", 0);
  check_icc(1, p, 200, false, "RGB for a gray picture");
  free(p);
  p = profile_of(200, 2, "mntr", "GRAY", "XYZ ", 0);
  check_icc(3, p, 200, false, "gray for an RGB picture");
  free(p);
  p = profile_of(200, 2, "prtr", "CMYK", "Lab ", 0);
  check_icc(3, p, 200, false, "CMYK");
  free(p);
  p = profile_of(200, 2, "mntr", "RGB ", "RGB ", 0);
  check_icc(3, p, 200, false, "of the connection space RGB");
  free(p);

  // The check itself: a profile not given, and channels a picture here does not have.
  char reason[160] = {};
  CHECK(unround_jpegio_check_icc(nullptr, 132, 3, reason, sizeof reason) == 0 && reason[0] != '\0');
  p = profile_of(132, 2, "mntr", "RGB ", "XYZ ", 0);
  CHECK(unround_jpegio_check_icc(p, 132, 2, reason, sizeof reason) == 0 && reason[0] != '\0');
  CHECK(unround_jpegio_check_icc(p, 132, 3, nullptr, 0) == 1);
  free(p);
}

// The length of the file's iCCP chunk, or -1 when it has none.
static int64_t iccp_length(const unround_jpegio_bytes* file) {
  for (uint64_t at = sizeof signature; at + 8 <= file->size;) {
    const uint8_t* p = file->data + at;
    const uint32_t length = ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | (uint32_t)p[3];
    if (memcmp(p + 4, "iCCP", 4) == 0) return (int64_t)length;
    at += 12 + (uint64_t)length;
  }
  return -1;
}

// Whether the file holds the profile as it is, that is, stored uncompressed.
static bool holds(const unround_jpegio_bytes* file, const uint8_t* profile, size_t size) {
  for (uint64_t at = 0; at + size <= file->size; ++at) {
    if (memcmp(file->data + at, profile, size) == 0) return true;
  }
  return false;
}

// Writes a picture with the profile at the level, and checks that it comes back
// with no warning, compressed or stored as expected, in an iCCP chunk that
// libpng's reading takes (81 bytes, and a zlib stream of 11).
static void check_icc_stored(const uint8_t* profile, size_t size, int32_t compression, int stored, const char* what) {
  if (!CHECK(profile != nullptr)) {
    note(what);
    return;
  }
  rng g = {.state = 4};
  void* samples = random_samples(&g, 5, 3, 3, 16);
  if (!CHECK(samples != nullptr)) return;
  unround_jpegio_png png = picture_of(5, 3, 3, 16, compression, samples);
  png.icc_profile = profile;
  png.icc_profile_size = (uint64_t)size;
  test_png back = {};
  unround_jpegio_bytes file = {};
  char message[256] = {};
  if (write_and_keep(&png, &back, message, sizeof message, &file)) {
    check_same_picture(&png, &back);
    if (!CHECK(message[0] == '\0')) note(message);
    if (!CHECK(back.icc_profile != nullptr && back.icc_profile_size == (uint64_t)size &&
               memcmp(back.icc_profile, profile, size) == 0)) {
      note(what);
    }
    check_no_warning(&back, what);
    if (!CHECK(iccp_length(&file) >= 81 + 11)) note(what);
    if (stored >= 0 && !CHECK(holds(&file, profile, size) == (stored == 1))) note(what);
  }
  unround_jpegio_bytes_free(&file);
  test_png_free(&back);
  free(samples);
}

// A profile that compresses well into a chunk shorter than libpng's reading takes
// is stored; one that compresses well into a longer chunk is compressed, and
// stored at level 0. Then profiles of every size from the header's up, with a
// share of random bytes from none to all, whose compressed chunks fall on both
// sides of the least that is taken.
static void test_icc_stored(void) {
  uint8_t* p = profile_of(132, 2, "mntr", "RGB ", "XYZ ", 0);
  check_icc_stored(p, 132, -1, 1, "a profile of 132 bytes");
  check_icc_stored(p, 132, 9, 1, "a profile of 132 bytes, at level 9");
  free(p);
  p = profile_of(3000, 4, "mntr", "RGB ", "XYZ ", 3);
  check_icc_stored(p, 3000, -1, 0, "a profile of 3000 bytes");
  check_icc_stored(p, 3000, 0, 1, "a profile of 3000 bytes, at level 0");
  free(p);

  rng g = {.state = 5};
  for (size_t size = 132; size <= 1200; size += 12) {
    for (uint64_t share = 0; share <= 8; share += 2) {
      p = profile_of(size, 2, "spac", "RGB ", "Lab ", 0);
      if (!CHECK(p != nullptr)) return;
      for (size_t i = 132; i < size; ++i) {
        const uint64_t r = rng_next(&g);
        p[i] = (r & 7U) < share ? (uint8_t)(r >> 8) : 0;
      }
      check_icc_stored(p, size, -1, -1, "a profile of random bytes and zeros");
      free(p);
    }
  }
}

// A profile larger than libpng's readers take by default is written, with a
// warning; one of just that size, without.
static void test_icc_large(void) {
  static const size_t sizes[] = {8000000, 8000004};
  for (size_t s = 0; s < sizeof sizes / sizeof sizes[0]; ++s) {
    uint8_t* p = profile_of(sizes[s], 4, "prtr", "RGB ", "Lab ", 2);
    if (!CHECK(p != nullptr)) return;
    rng g = {.state = 6};
    void* samples = random_samples(&g, 2, 2, 3, 8);
    if (!CHECK(samples != nullptr)) {
      free(p);
      return;
    }
    unround_jpegio_png png = picture_of(2, 2, 3, 8, 1, samples);
    png.icc_profile = p;
    png.icc_profile_size = (uint64_t)sizes[s];
    test_png back = {};
    char message[256] = {};
    if (write_and_read(&png, &back, message, sizeof message)) {
      check_same_picture(&png, &back);
      CHECK(back.icc_profile != nullptr && back.icc_profile_size == (uint64_t)sizes[s] &&
            memcmp(back.icc_profile, p, sizes[s]) == 0);
      check_no_warning(&back, "a large profile");
      if (sizes[s] > 8000000) {
        if (!CHECK(strstr(message, "8000004 bytes") != nullptr && strstr(message, "8000000") != nullptr)) {
          note(message);
        }
      } else if (!CHECK(message[0] == '\0')) {
        note(message);
      }
    }
    test_png_free(&back);
    free(samples);
    free(p);
  }
}

static void expect_refused(const unround_jpegio_png* png, const char* why) {
  // Bytes that a refusal has to empty.
  static uint8_t stale = 0;
  unround_jpegio_bytes file = {.data = &stale, .size = 1};
  char message[256] = {};
  const unround_jpegio_status status = unround_jpegio_write_png(png, &file, message, sizeof message);
  if (!CHECK(status == UNROUND_JPEGIO_ERROR_ARGUMENT)) note(why);
  CHECK(file.data == nullptr && file.size == 0);
  if (!CHECK(message[0] != '\0')) note(why);
}

static void test_arguments(void) {
  static const uint8_t samples[48] = {};
  static const uint8_t profile[4] = {};
  const unround_jpegio_png good = picture_of(4, 4, 3, 8, -1, samples);
  unround_jpegio_bytes file = {};
  char message[16] = {};
  CHECK(unround_jpegio_write_png(&good, nullptr, message, sizeof message) == UNROUND_JPEGIO_ERROR_ARGUMENT);
  CHECK(unround_jpegio_write_png(&good, &file, nullptr, 1) == UNROUND_JPEGIO_ERROR_ARGUMENT);
  CHECK(unround_jpegio_write_png(nullptr, &file, message, sizeof message) == UNROUND_JPEGIO_ERROR_ARGUMENT);
  // A message buffer of no size, and one too short, are allowed.
  CHECK(unround_jpegio_write_png(&good, &file, nullptr, 0) == UNROUND_JPEGIO_OK);
  unround_jpegio_bytes_free(&file);
  unround_jpegio_bytes_free(nullptr);

  unround_jpegio_png png = good;
  png.width = 0;
  expect_refused(&png, "no width");
  png = good;
  png.height = 0;
  expect_refused(&png, "no height");
  png = good;
  png.width = UINT32_C(0x80000000);
  expect_refused(&png, "wider than PNG allows");
  png = good;
  png.channels = 2;
  expect_refused(&png, "two channels");
  png = good;
  png.channels = 4;
  expect_refused(&png, "four channels");
  png = good;
  png.bits = 12;
  expect_refused(&png, "12 bits");
  png = good;
  png.compression = -2;
  expect_refused(&png, "compression -2");
  png = good;
  png.compression = 10;
  expect_refused(&png, "compression 10");
  png = good;
  png.reserved = 1;
  expect_refused(&png, "a reserved field set");
  png = good;
  png.samples = nullptr;
  expect_refused(&png, "no samples");
  png = good;
  png.icc_profile_size = 4;
  expect_refused(&png, "a size of a profile not given");
  png = good;
  png.icc_profile = profile;
  png.icc_profile_size = 0;
  unround_jpegio_bytes written = {};
  char long_message[256] = {};
  CHECK(unround_jpegio_write_png(&png, &written, long_message, sizeof long_message) == UNROUND_JPEGIO_OK);
  CHECK(strncmp(long_message, "the ICC profile is not written", 30) == 0);
  unround_jpegio_bytes_free(&written);
}

static void test_version(void) {
  const char* version = unround_jpegio_libpng_version();
  CHECK(strncmp(version, "libpng 1.6.", 11) == 0);
  CHECK(strstr(version, ", zlib-ng ") != nullptr);
  CHECK(unround_jpegio_abi_version() == UNROUND_JPEGIO_ABI_VERSION);
}

int main(void) {
  test_round_trips();
  test_large();
  test_levels();
  test_icc();
  test_icc_stored();
  test_icc_large();
  test_arguments();
  test_version();
  (void)fprintf(stderr, "%d checks, %d failed\n", checks, failures);
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
