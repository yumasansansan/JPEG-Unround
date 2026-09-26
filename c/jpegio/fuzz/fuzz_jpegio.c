// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The fuzz target of the C layer. Whatever the bytes, a read and a decoding of
// the planes return a status and a terminated message; a failure leaves
// nothing allocated; and a success describes exactly the memory it hands over:
// every coefficient and sample it says is there is read here, which the
// address sanitizer checks, and the sizes it reports agree with one another.
//
// The ICC profile of a file that reads, and the bytes themselves taken as a
// profile, are written into PNG files of a gray and an RGB pixel: whatever the
// profile, the file is written, and libpng reads the profile back, exactly as
// it was, where the C layer's check takes it for the picture, and not where the
// check refuses it, which the message then says. A broken promise aborts, which
// is what a fuzzer reports.
//
// Built with libFuzzer when UNROUND_BUILD_FUZZERS is on; smoke.c runs it on
// every system, on mutations of files and profiles of its own.

#include "fuzz_jpegio.h"

#include "test_png.h"
#include "unround/jpegio.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static volatile uint8_t sink;

// A promise of the C layer, as a comparison of C gives a truth: an int.
static void require(int holds) {
  if (holds == 0) abort();
}

// Reads every byte, so that the sanitizers see each of them read.
static void touch(const void* memory, size_t size) {
  const uint8_t* bytes = (const uint8_t*)memory;
  uint8_t folded = 0;
  for (size_t i = 0; i < size; ++i) folded ^= bytes[i];
  sink = folded;
}

static void require_terminated(const char* message, size_t size) { require(memchr(message, '\0', size) != nullptr); }

static void check_image(const unround_jpegio_image* image, const unround_jpegio_options* options) {
  require(image->num_components == 1 || image->num_components == 3);
  require(image->width > 0 && image->height > 0);
  require((uint64_t)image->width * (uint64_t)image->height <= options->max_pixels);
  require(image->exif_orientation >= 0 && image->exif_orientation <= 8);
  require(image->warnings >= 0);
  for (int32_t ci = 0; ci < image->num_components; ++ci) {
    const unround_jpegio_component* c = &image->components[ci];
    require(c->h_samp_factor >= 1 && c->h_samp_factor <= image->max_h_samp_factor);
    require(c->v_samp_factor >= 1 && c->v_samp_factor <= image->max_v_samp_factor);
    require(c->quant_table_slot >= 0 && c->quant_table_slot <= 3);
    require(c->width_in_blocks == (c->width + 7) / 8 && c->height_in_blocks == (c->height + 7) / 8);
    require(c->coefficients != nullptr);
    touch(c->coefficients,
          (size_t)c->width_in_blocks * c->height_in_blocks * UNROUND_JPEGIO_BLOCK_SIZE * sizeof(int16_t));
  }
  for (int32_t ci = image->num_components; ci < UNROUND_JPEGIO_MAX_COMPONENTS; ++ci) {
    require(image->components[ci].coefficients == nullptr);
  }
  require((image->icc_profile == nullptr) == (image->icc_profile_size == 0));
  if (image->icc_profile != nullptr) touch(image->icc_profile, (size_t)image->icc_profile_size);
}

static void check_planes(const unround_jpegio_planes* planes) {
  require(planes->num_components == 1 || planes->num_components == 3);
  for (int32_t ci = 0; ci < planes->num_components; ++ci) {
    const unround_jpegio_plane* p = &planes->planes[ci];
    require(p->width > 0 && p->width % 8 == 0 && p->height > 0 && p->height % 8 == 0 && p->stride >= p->width);
    require(p->samples != nullptr);
    touch(p->samples, (size_t)p->stride * p->height);
  }
}

static void check_icc_png(const uint8_t* profile, size_t size) {
  static const uint8_t samples[3] = {1, 2, 3};
  for (int32_t channels = 1; channels <= 3; channels += 2) {
    char reason[128];
    memset(reason, 'x', sizeof reason);
    const int32_t taken = unround_jpegio_check_icc(profile, (uint64_t)size, channels, reason, sizeof reason);
    require(taken == 0 || taken == 1);
    require_terminated(reason, sizeof reason);
    require((taken == 1) == (reason[0] == '\0'));
    const unround_jpegio_png png = {
        .width = 1,
        .height = 1,
        .channels = channels,
        .bits = 8,
        .compression = 1,
        .samples = samples,
        .icc_profile = profile,
        .icc_profile_size = (uint64_t)size,
    };
    unround_jpegio_bytes file = {};
    char message[256];
    memset(message, 'x', sizeof message);
    require(unround_jpegio_write_png(&png, &file, message, sizeof message) == UNROUND_JPEGIO_OK);
    require_terminated(message, sizeof message);
    require(file.data != nullptr && file.size > 8);
    test_png back = {};
    char read_message[256];
    require(test_png_read(file.data, (size_t)file.size, &back, read_message, sizeof read_message) == 0);
    require(back.width == 1 && back.height == 1 && back.channels == channels && back.bits == 8);
    require(memcmp(back.samples, samples, (size_t)channels) == 0);
    if (taken == 1) {
      require(back.icc_profile != nullptr && back.icc_profile_size == (uint64_t)size);
      require(memcmp(back.icc_profile, profile, size) == 0);
    } else {
      require(back.icc_profile == nullptr);
      require(strncmp(message, "the ICC profile is not written: ", 32) == 0);
    }
    test_png_free(&back);
    unround_jpegio_bytes_free(&file);
    require(file.data == nullptr && file.size == 0);
  }
}

int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  // Small limits keep each input quick: a header can ask for any size.
  const unround_jpegio_options options = {.max_pixels = UINT64_C(1) << 20, .max_scans = 64, .warnings_are_errors = 0};
  char message[128];

  unround_jpegio_image image = {};
  memset(message, 'x', sizeof message);
  if (unround_jpegio_read(data, size, &options, &image, message, sizeof message) == UNROUND_JPEGIO_OK) {
    check_image(&image, &options);
    if (image.icc_profile != nullptr) check_icc_png(image.icc_profile, (size_t)image.icc_profile_size);
  } else {
    require(image.num_components == 0 && image.components[0].coefficients == nullptr && image.icc_profile == nullptr);
  }
  require_terminated(message, sizeof message);
  unround_jpegio_image_free(&image);

  unround_jpegio_planes planes = {};
  memset(message, 'x', sizeof message);
  if (unround_jpegio_decode_planes(data, size, &options, &planes, message, sizeof message) == UNROUND_JPEGIO_OK) {
    check_planes(&planes);
  } else {
    require(planes.num_components == 0 && planes.planes[0].samples == nullptr);
  }
  require_terminated(message, sizeof message);
  unround_jpegio_planes_free(&planes);

  check_icc_png(data, size);
  return 0;
}
