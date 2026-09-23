// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The fuzz target of the C layer. Whatever the bytes, a read and a decoding of
// the planes return a status and a terminated message; a failure leaves
// nothing allocated; and a success describes exactly the memory it hands over:
// every coefficient and sample it says is there is read here, which the
// address sanitizer checks, and the sizes it reports agree with one another.
// A broken promise aborts, which is what a fuzzer reports.
//
// Built with libFuzzer when UNROUND_BUILD_FUZZERS is on; smoke.c runs it on
// every system, on mutations of files of its own.

#include "unround/jpegio.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size);

static volatile uint8_t sink;

static void require(bool condition) {
  if (!condition) abort();
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

int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  // Small limits keep each input quick: a header can ask for any size.
  const unround_jpegio_options options = {.max_pixels = UINT64_C(1) << 20, .max_scans = 64, .warnings_are_errors = 0};
  char message[128];

  unround_jpegio_image image = {};
  memset(message, 'x', sizeof message);
  if (unround_jpegio_read(data, size, &options, &image, message, sizeof message) == UNROUND_JPEGIO_OK) {
    check_image(&image, &options);
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
  return 0;
}
