// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

#include "test_jpeg.h"

#include <setjmp.h>
#include <stdckdint.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <jerror.h>
#include <jpeglib.h>

// What libjpeg's error handler needs: where to jump, and room for the message.
// Every state below begins with one, and client_data points to it.
typedef struct failure {
  jmp_buf jump;
  char message[JMSG_LENGTH_MAX];
} failure;

[[noreturn]] static void on_error_exit(j_common_ptr cinfo) {
  failure* f = (failure*)cinfo->client_data;
  (*cinfo->err->format_message)(cinfo, f->message);
  longjmp(f->jump, 1);
}

static void on_output_message(j_common_ptr /* cinfo */) {}

[[noreturn]] static void fail(j_common_ptr cinfo, int code) {
  cinfo->err->msg_code = code;
  (*cinfo->err->error_exit)(cinfo);
  abort();  // error_exit does not return
}

static void copy_message(char* message, size_t message_size, const char* text) {
  if (message == nullptr || message_size == 0) return;
  (void)snprintf(message, message_size, "%s", text);
}

int32_t test_jpeg_components(const test_jpeg* spec) {
  switch (spec->color_space) {
    case TEST_JPEG_GRAYSCALE:
      return 1;
    case TEST_JPEG_YCBCR:
    case TEST_JPEG_RGB:
      return 3;
    case TEST_JPEG_CMYK:
      return 4;
    default:
      return 0;
  }
}

static int32_t factor_or_one(int32_t factor) { return factor > 0 ? factor : 1; }

static int32_t max_factor(const int32_t factors[TEST_JPEG_MAX_COMPONENTS], int32_t components) {
  int32_t largest = 1;
  for (int32_t ci = 0; ci < components; ++ci) {
    if (factor_or_one(factors[ci]) > largest) largest = factor_or_one(factors[ci]);
  }
  return largest;
}

static uint32_t divide_rounding_up(uint64_t numerator, uint64_t denominator) {
  return (uint32_t)((numerator + denominator - 1) / denominator);
}

uint32_t test_jpeg_blocks_wide(const test_jpeg* spec, int32_t component) {
  const int32_t components = test_jpeg_components(spec);
  const uint64_t h = (uint64_t)factor_or_one(spec->h_samp_factor[component]);
  const uint64_t h_max = (uint64_t)max_factor(spec->h_samp_factor, components);
  return divide_rounding_up((uint64_t)spec->width * h, h_max * DCTSIZE);
}

uint32_t test_jpeg_blocks_high(const test_jpeg* spec, int32_t component) {
  const int32_t components = test_jpeg_components(spec);
  const uint64_t v = (uint64_t)factor_or_one(spec->v_samp_factor[component]);
  const uint64_t v_max = (uint64_t)max_factor(spec->v_samp_factor, components);
  return divide_rounding_up((uint64_t)spec->height * v, v_max * DCTSIZE);
}

// ---------------------------------------------------------------------------
// Encoding

typedef struct writer {
  failure fail;  // first: client_data points here
  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr error;
  struct jpeg_destination_mgr destination;
  jpeg_scan_info scans[TEST_JPEG_MAX_COMPONENTS];
  uint8_t* buffer;
  size_t capacity;
  size_t size;
} writer;

static writer* writer_of(j_compress_ptr cinfo) { return (writer*)cinfo->client_data; }

static void on_init_destination(j_compress_ptr cinfo) {
  writer* w = writer_of(cinfo);
  w->capacity = 4096;
  w->buffer = (uint8_t*)malloc(w->capacity);
  if (w->buffer == nullptr) fail((j_common_ptr)cinfo, JERR_OUT_OF_MEMORY);
  cinfo->dest->next_output_byte = w->buffer;
  cinfo->dest->free_in_buffer = w->capacity;
}

// The buffer is full: it grows, and everything written so far stays in it.
static boolean on_empty_output_buffer(j_compress_ptr cinfo) {
  writer* w = writer_of(cinfo);
  size_t capacity = 0;
  if (ckd_mul(&capacity, w->capacity, (size_t)2)) fail((j_common_ptr)cinfo, JERR_OUT_OF_MEMORY);
  uint8_t* grown = (uint8_t*)realloc(w->buffer, capacity);
  if (grown == nullptr) fail((j_common_ptr)cinfo, JERR_OUT_OF_MEMORY);
  w->buffer = grown;
  cinfo->dest->next_output_byte = grown + w->capacity;
  cinfo->dest->free_in_buffer = capacity - w->capacity;
  w->capacity = capacity;
  return TRUE;
}

static void on_term_destination(j_compress_ptr cinfo) {
  writer* w = writer_of(cinfo);
  w->size = w->capacity - cinfo->dest->free_in_buffer;
}

static writer* writer_new(void) {
  writer* w = (writer*)calloc(1, sizeof *w);
  if (w == nullptr) return nullptr;
  w->cinfo.err = jpeg_std_error(&w->error);
  w->error.error_exit = on_error_exit;
  w->error.output_message = on_output_message;
  w->cinfo.client_data = w;
  w->destination.init_destination = on_init_destination;
  w->destination.empty_output_buffer = on_empty_output_buffer;
  w->destination.term_destination = on_term_destination;
  return w;
}

static void encode_coefficients(writer* w, const test_jpeg* spec) {
  struct jpeg_compress_struct* const c = &w->cinfo;
  jpeg_create_compress(c);
  c->dest = &w->destination;

  const int32_t components = test_jpeg_components(spec);
  J_COLOR_SPACE space = JCS_UNKNOWN;
  switch (spec->color_space) {
    case TEST_JPEG_GRAYSCALE:
      space = JCS_GRAYSCALE;
      break;
    case TEST_JPEG_YCBCR:
      space = JCS_YCbCr;
      break;
    case TEST_JPEG_RGB:
      space = JCS_RGB;
      break;
    case TEST_JPEG_CMYK:
      space = JCS_CMYK;
      break;
    default:
      fail((j_common_ptr)c, JERR_BAD_J_COLORSPACE);
  }
  c->image_width = (JDIMENSION)spec->width;
  c->image_height = (JDIMENSION)spec->height;
  c->input_components = (int)components;
  c->in_color_space = space;
  c->data_precision = spec->data_precision != 0 ? (int)spec->data_precision : 8;
  jpeg_set_defaults(c);
  jpeg_set_colorspace(c, space);

  bool slot_used[NUM_QUANT_TBLS] = {};
  for (int32_t ci = 0; ci < components; ++ci) {
    const int32_t slot = spec->quant_table_slot[ci];
    if (slot < 0 || slot >= NUM_QUANT_TBLS) fail((j_common_ptr)c, JERR_NO_QUANT_TABLE);
    c->comp_info[ci].h_samp_factor = (int)factor_or_one(spec->h_samp_factor[ci]);
    c->comp_info[ci].v_samp_factor = (int)factor_or_one(spec->v_samp_factor[ci]);
    c->comp_info[ci].quant_tbl_no = (int)slot;
    slot_used[slot] = true;
  }
  for (int slot = 0; slot < NUM_QUANT_TBLS; ++slot) {
    if (!slot_used[slot]) continue;
    unsigned int table[DCTSIZE2];
    for (int k = 0; k < DCTSIZE2; ++k) table[k] = (unsigned int)spec->quant_tables[slot][k];
    // A scale of 100 percent keeps the entries as they are.
    jpeg_add_quant_table(c, slot, table, 100, FALSE);
  }
  c->arith_code = spec->arithmetic != 0 ? TRUE : FALSE;
  c->optimize_coding = (spec->optimize_coding != 0 || c->data_precision != 8) ? TRUE : FALSE;
  c->restart_interval = (unsigned int)spec->restart_interval;
  if (spec->progressive != 0) {
    jpeg_simple_progression(c);
  } else if (spec->sequential_scans != 0) {
    for (int32_t ci = 0; ci < components; ++ci) {
      w->scans[ci] = (jpeg_scan_info){
          .comps_in_scan = 1, .component_index = {(int)ci}, .Ss = 0, .Se = DCTSIZE2 - 1, .Ah = 0, .Al = 0};
    }
    c->scan_info = w->scans;
    c->num_scans = (int)components;
  }

  // The arrays are as large as whole MCUs; the encoder makes up the dummy
  // blocks itself and reads only the blocks of image data.
  jvirt_barray_ptr arrays[TEST_JPEG_MAX_COMPONENTS] = {};
  for (int32_t ci = 0; ci < components; ++ci) {
    const uint32_t h = (uint32_t)factor_or_one(spec->h_samp_factor[ci]);
    const uint32_t v = (uint32_t)factor_or_one(spec->v_samp_factor[ci]);
    const uint32_t wide = test_jpeg_blocks_wide(spec, ci);
    const uint32_t high = test_jpeg_blocks_high(spec, ci);
    arrays[ci] =
        (*c->mem->request_virt_barray)((j_common_ptr)c, JPOOL_IMAGE, TRUE, (JDIMENSION)((wide + h - 1) / h * h),
                                       (JDIMENSION)((high + v - 1) / v * v), (JDIMENSION)v);
  }
  jpeg_write_coefficients(c, arrays);
  for (int32_t ci = 0; ci < components; ++ci) {
    const uint32_t wide = test_jpeg_blocks_wide(spec, ci);
    const uint32_t high = test_jpeg_blocks_high(spec, ci);
    for (uint32_t row = 0; row < high; ++row) {
      JBLOCKARRAY blocks =
          (*c->mem->access_virt_barray)((j_common_ptr)c, arrays[ci], (JDIMENSION)row, (JDIMENSION)1, TRUE);
      const int16_t* source = spec->coefficients[ci] + (size_t)row * wide * DCTSIZE2;
      memcpy(blocks[0], source, (size_t)wide * DCTSIZE2 * sizeof(JCOEF));
    }
  }
  if (spec->icc_profile != nullptr) jpeg_write_icc_profile(c, spec->icc_profile, spec->icc_profile_size);
  if (spec->app1 != nullptr) jpeg_write_marker(c, JPEG_APP0 + 1, spec->app1, spec->app1_size);
  jpeg_finish_compress(c);
}

static void encode_lossless(writer* w, uint32_t width, uint32_t height, const uint8_t* samples) {
  struct jpeg_compress_struct* const c = &w->cinfo;
  jpeg_create_compress(c);
  c->dest = &w->destination;
  c->image_width = (JDIMENSION)width;
  c->image_height = (JDIMENSION)height;
  c->input_components = 1;
  c->in_color_space = JCS_GRAYSCALE;
  jpeg_set_defaults(c);
  jpeg_enable_lossless(c, 1, 0);
  jpeg_start_compress(c, TRUE);
  // libjpeg takes rows that are not const, so each is copied into one of its own.
  JSAMPARRAY row = (*c->mem->alloc_sarray)((j_common_ptr)c, JPOOL_IMAGE, (JDIMENSION)width, (JDIMENSION)1);
  while (c->next_scanline < c->image_height) {
    memcpy(row[0], samples + (size_t)c->next_scanline * width, (size_t)width);
    (void)jpeg_write_scanlines(c, row, (JDIMENSION)1);
  }
  jpeg_finish_compress(c);
}

static int finish_writing(writer* w, int result, uint8_t** data, size_t* size, char* message, size_t message_size) {
  jpeg_destroy_compress(&w->cinfo);
  if (result == 0) {
    *data = w->buffer;
    *size = w->size;
    copy_message(message, message_size, "");
  } else {
    free(w->buffer);
    copy_message(message, message_size, w->fail.message);
  }
  free(w);
  return result;
}

int test_jpeg_write(const test_jpeg* spec, uint8_t** data, size_t* size, char* message, size_t message_size) {
  *data = nullptr;
  *size = 0;
  writer* w = writer_new();
  if (w == nullptr) {
    copy_message(message, message_size, "out of memory");
    return 1;
  }
  int result = 0;
  switch (setjmp(w->fail.jump)) {
    case 0:
      encode_coefficients(w, spec);
      result = 0;
      break;
    default:
      result = 1;
      break;
  }
  return finish_writing(w, result, data, size, message, message_size);
}

int test_jpeg_write_lossless(uint32_t width, uint32_t height, const uint8_t* samples, uint8_t** data, size_t* size,
                             char* message, size_t message_size) {
  *data = nullptr;
  *size = 0;
  writer* w = writer_new();
  if (w == nullptr) {
    copy_message(message, message_size, "out of memory");
    return 1;
  }
  int result = 0;
  switch (setjmp(w->fail.jump)) {
    case 0:
      encode_lossless(w, width, height, samples);
      result = 0;
      break;
    default:
      result = 1;
      break;
  }
  return finish_writing(w, result, data, size, message, message_size);
}

// ---------------------------------------------------------------------------
// Decoding

typedef struct decoder {
  failure fail;  // first: client_data points here
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr error;
  uint8_t* samples;
} decoder;

static void decode_samples(decoder* d, const uint8_t* data, size_t size) {
  struct jpeg_decompress_struct* const c = &d->cinfo;
  jpeg_create_decompress(c);
  jpeg_mem_src(c, data, (unsigned long)size);
  (void)jpeg_read_header(c, TRUE);
  c->out_color_space = c->jpeg_color_space;
  c->dct_method = JDCT_ISLOW;
  c->do_fancy_upsampling = FALSE;
  c->do_block_smoothing = FALSE;
  (void)jpeg_start_decompress(c);
  const size_t row_size = (size_t)c->output_width * (size_t)c->output_components;
  d->samples = (uint8_t*)malloc(row_size * (size_t)c->output_height + 1);
  if (d->samples == nullptr) fail((j_common_ptr)c, JERR_OUT_OF_MEMORY);
  while (c->output_scanline < c->output_height) {
    JSAMPROW row = d->samples + (size_t)c->output_scanline * row_size;
    (void)jpeg_read_scanlines(c, &row, (JDIMENSION)1);
  }
}

int test_jpeg_decode(const uint8_t* data, size_t size, uint8_t** samples, uint32_t* width, uint32_t* height,
                     int32_t* components, char* message, size_t message_size) {
  *samples = nullptr;
  *width = 0;
  *height = 0;
  *components = 0;
  decoder* d = (decoder*)calloc(1, sizeof *d);
  if (d == nullptr) {
    copy_message(message, message_size, "out of memory");
    return 1;
  }
  d->cinfo.err = jpeg_std_error(&d->error);
  d->error.error_exit = on_error_exit;
  d->error.output_message = on_output_message;
  d->cinfo.client_data = d;
  int result = 0;
  switch (setjmp(d->fail.jump)) {
    case 0:
      decode_samples(d, data, size);
      *width = (uint32_t)d->cinfo.output_width;
      *height = (uint32_t)d->cinfo.output_height;
      *components = (int32_t)d->cinfo.output_components;
      (void)jpeg_finish_decompress(&d->cinfo);
      result = 0;
      break;
    default:
      result = 1;
      break;
  }
  jpeg_destroy_decompress(&d->cinfo);
  if (result == 0) {
    *samples = d->samples;
    copy_message(message, message_size, "");
  } else {
    free(d->samples);
    *width = 0;
    *height = 0;
    *components = 0;
    copy_message(message, message_size, d->fail.message);
  }
  free(d);
  return result;
}
