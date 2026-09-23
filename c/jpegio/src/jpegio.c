// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// libjpeg reports an error by calling error_exit, which must not return. Here
// it longjmps back to the one setjmp of each public function, and the jump
// never crosses a frame that is not C: the setjmp is in the function the
// caller called, every libjpeg call runs below it, and the state that must
// survive the jump lives in a heap object the function allocates before it
// sets the jump, so nothing the jump skips over needs cleaning up. setjmp
// stands only where C allows it, as the whole controlling expression of a
// switch.

#include "unround/jpegio.h"

#include <limits.h>
#include <setjmp.h>
#include <stdckdint.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// jpeglib.h expects stdio.h (for FILE) and size_t to be declared first.
#include <jerror.h>
#include <jpeglib.h>

static_assert(sizeof(JCOEF) == sizeof(int16_t), "libjpeg's coefficients are 16-bit");
static_assert(sizeof(JSAMPLE) == sizeof(uint8_t), "libjpeg's 8-bit samples are bytes");
static_assert(DCTSIZE2 == UNROUND_JPEGIO_BLOCK_SIZE, "a block holds 64 coefficients");
static_assert(MAX_COMPS_IN_SCAN <= UNROUND_JPEGIO_MAX_COMPONENTS, "a scan fits the components this layer reports");

#define UNROUND_TEXT_OF(x) #x
#define UNROUND_TEXT(x) UNROUND_TEXT_OF(x)

static constexpr uint64_t default_max_pixels = UINT64_C(1) << 28;
static constexpr int32_t default_max_scans = 500;

typedef struct reader {
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr error;
  struct jpeg_progress_mgr progress;
  jmp_buf jump;
  unround_jpegio_status failure;
  int32_t max_scans;
  int32_t warnings_are_errors;
  int32_t warnings;
  uint64_t max_pixels;
  char message[JMSG_LENGTH_MAX];
} reader;

static reader* reader_of(j_common_ptr cinfo) { return (reader*)cinfo->client_data; }

[[noreturn]] static void fail(reader* r, unround_jpegio_status status) {
  r->failure = status;
  longjmp(r->jump, 1);
}

[[noreturn]] static void fail_with(reader* r, unround_jpegio_status status, const char* text) {
  (void)snprintf(r->message, sizeof r->message, "%s", text);
  fail(r, status);
}

static unround_jpegio_status status_of_message(int msg_code) {
  switch (msg_code) {
    case JERR_OUT_OF_MEMORY:
      return UNROUND_JPEGIO_ERROR_MEMORY;
    case JERR_BAD_PRECISION:
    case JERR_NOTIMPL:
      return UNROUND_JPEGIO_ERROR_UNSUPPORTED;
    case JERR_IMAGE_TOO_BIG:
    case JERR_WIDTH_OVERFLOW:
      return UNROUND_JPEGIO_ERROR_LIMIT;
    default:
      return UNROUND_JPEGIO_ERROR_DECODE;
  }
}

[[noreturn]] static void on_error_exit(j_common_ptr cinfo) {
  reader* r = reader_of(cinfo);
  (*cinfo->err->format_message)(cinfo, r->message);
  fail(r, status_of_message(cinfo->err->msg_code));
}

// libjpeg calls this for warnings (level -1) and trace messages (level >= 0).
// A warning means corrupt data that libjpeg worked around; the first one is
// kept as the message, and every one is counted.
static void on_emit_message(j_common_ptr cinfo, int msg_level) {
  if (msg_level >= 0) return;
  reader* r = reader_of(cinfo);
  if (r->warnings == 0) (*cinfo->err->format_message)(cinfo, r->message);
  if (r->warnings < INT32_MAX) r->warnings += 1;
  cinfo->err->num_warnings += 1L;
  if (r->warnings_are_errors != 0) fail(r, UNROUND_JPEGIO_ERROR_DECODE);
}

// Nothing is printed: what a caller needs to know is in the message it gets.
static void on_output_message(j_common_ptr /* cinfo */) {}

// A progressive file may hold any number of scans, and each costs a pass over
// the coefficients; a crafted one can make a read take arbitrarily long.
static void on_progress(j_common_ptr cinfo) {
  if (cinfo->is_decompressor == FALSE) return;
  reader* r = reader_of(cinfo);
  const j_decompress_ptr decompress = (j_decompress_ptr)cinfo;
  if (decompress->input_scan_number > r->max_scans) {
    (void)snprintf(r->message, sizeof r->message, "the file has more than %d scans", (int)r->max_scans);
    fail(r, UNROUND_JPEGIO_ERROR_LIMIT);
  }
}

static reader* reader_new(const unround_jpegio_options* options) {
  reader* r = (reader*)calloc(1, sizeof *r);
  if (r == nullptr) return nullptr;
  r->cinfo.err = jpeg_std_error(&r->error);
  r->error.error_exit = on_error_exit;
  r->error.emit_message = on_emit_message;
  r->error.output_message = on_output_message;
  r->progress.progress_monitor = on_progress;
  r->cinfo.client_data = r;  // jpeg_create_decompress() keeps err and client_data
  r->max_pixels = (options != nullptr && options->max_pixels != 0) ? options->max_pixels : default_max_pixels;
  r->max_scans = (options != nullptr && options->max_scans > 0) ? options->max_scans : default_max_scans;
  r->warnings_are_errors = (options != nullptr) ? options->warnings_are_errors : 0;
  r->failure = UNROUND_JPEGIO_OK;
  return r;
}

static void copy_message(char* message, size_t message_size, const char* text) {
  if (message == nullptr || message_size == 0) return;
  (void)snprintf(message, message_size, "%s", text);
}

// Whether libjpeg's memory source, which takes an unsigned long, can be handed
// the whole of the data.
static bool size_fits(size_t size) {
#if SIZE_MAX > ULONG_MAX
  return size <= (size_t)ULONG_MAX;
#else
  (void)size;
  return true;
#endif
}

// Big- or little-endian integers of a TIFF structure.
static uint32_t tiff_u16(const uint8_t* p, bool little) {
  return little ? ((uint32_t)p[0] | ((uint32_t)p[1] << 8)) : (((uint32_t)p[0] << 8) | (uint32_t)p[1]);
}

static uint32_t tiff_u32(const uint8_t* p, bool little) {
  return little ? ((uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24))
                : (((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | (uint32_t)p[3]);
}

// The Orientation tag (0x0112) of the first IFD of an EXIF APP1 marker, or 0
// when the marker holds none. Every offset is checked against the marker's
// length before it is read.
static int32_t exif_orientation(const uint8_t* data, size_t size) {
  static const uint8_t signature[6] = {'E', 'x', 'i', 'f', 0, 0};
  if (size < sizeof signature + 8 || memcmp(data, signature, sizeof signature) != 0) return 0;
  const uint8_t* tiff = data + sizeof signature;
  const size_t tiff_size = size - sizeof signature;
  bool little = false;
  if (tiff[0] == 'I' && tiff[1] == 'I') {
    little = true;
  } else if (!(tiff[0] == 'M' && tiff[1] == 'M')) {
    return 0;
  }
  if (tiff_u16(tiff + 2, little) != 42) return 0;
  const size_t ifd = (size_t)tiff_u32(tiff + 4, little);
  if (ifd > tiff_size || tiff_size - ifd < 2) return 0;
  const size_t count = (size_t)tiff_u16(tiff + ifd, little);
  for (size_t i = 0; i < count; ++i) {
    size_t entry = 0;
    if (ckd_mul(&entry, i, (size_t)12) || ckd_add(&entry, entry, ifd + 2)) return 0;
    if (entry > tiff_size || tiff_size - entry < 12) return 0;
    const uint8_t* e = tiff + entry;
    if (tiff_u16(e, little) != 0x0112) continue;
    // type SHORT (3), one value, stored in the first two bytes of the value field
    if (tiff_u16(e + 2, little) != 3 || tiff_u32(e + 4, little) != 1) return 0;
    const uint32_t value = tiff_u16(e + 8, little);
    return (value >= 1 && value <= 8) ? (int32_t)value : 0;
  }
  return 0;
}

// Opens the data and reads the header, and refuses what this layer does not
// read. Everything that can fail here longjmps.
static void open_source(reader* r, const uint8_t* data, size_t size) {
  jpeg_create_decompress(&r->cinfo);
  r->cinfo.progress = &r->progress;
  jpeg_mem_src(&r->cinfo, data, (unsigned long)size);
  jpeg_save_markers(&r->cinfo, JPEG_APP0 + 1, 0xFFFF);  // EXIF
  jpeg_save_markers(&r->cinfo, JPEG_APP0 + 2, 0xFFFF);  // ICC profile
  (void)jpeg_read_header(&r->cinfo, TRUE);

  const j_decompress_ptr c = &r->cinfo;
  // In the lossless process a data unit is one sample rather than a block of
  // eight by eight, and libjpeg says so here from the header on.
#if JPEG_LIB_VERSION >= 70
  const int data_unit = c->min_DCT_h_scaled_size;
#else
  const int data_unit = c->min_DCT_scaled_size;
#endif
  if (data_unit != DCTSIZE) fail_with(r, UNROUND_JPEGIO_ERROR_UNSUPPORTED, "lossless JPEG is not supported");
  if (c->data_precision != 8) {
    (void)snprintf(r->message, sizeof r->message, "%d-bit JPEG is not supported", (int)c->data_precision);
    fail(r, UNROUND_JPEGIO_ERROR_UNSUPPORTED);
  }
  const bool grayscale = c->num_components == 1 && c->jpeg_color_space == JCS_GRAYSCALE;
  const bool three = c->num_components == 3 && (c->jpeg_color_space == JCS_YCbCr || c->jpeg_color_space == JCS_RGB);
  if (!grayscale && !three) {
    (void)snprintf(r->message, sizeof r->message, "a JPEG of %d components in color space %d is not supported",
                   (int)c->num_components, (int)c->jpeg_color_space);
    fail(r, UNROUND_JPEGIO_ERROR_UNSUPPORTED);
  }
  uint64_t pixels = 0;
  if (ckd_mul(&pixels, (uint64_t)c->image_width, (uint64_t)c->image_height) || pixels > r->max_pixels) {
    (void)snprintf(r->message, sizeof r->message, "the image has %lu x %lu pixels, more than %llu",
                   (unsigned long)c->image_width, (unsigned long)c->image_height, (unsigned long long)r->max_pixels);
    fail(r, UNROUND_JPEGIO_ERROR_LIMIT);
  }
}

static unround_jpegio_color_space color_space_of(J_COLOR_SPACE space) {
  switch (space) {
    case JCS_GRAYSCALE:
      return UNROUND_JPEGIO_GRAYSCALE;
    case JCS_RGB:
      return UNROUND_JPEGIO_RGB;
    default:
      return UNROUND_JPEGIO_YCBCR;
  }
}

static void* allocate(reader* r, size_t count, size_t element_size) {
  size_t bytes = 0;
  if (ckd_mul(&bytes, count, element_size)) fail_with(r, UNROUND_JPEGIO_ERROR_LIMIT, "the image is too large");
  void* p = malloc(bytes > 0 ? bytes : 1);
  if (p == nullptr) fail_with(r, UNROUND_JPEGIO_ERROR_MEMORY, "out of memory");
  return p;
}

static void read_coefficients(reader* r, unround_jpegio_image* image) {
  const j_decompress_ptr c = &r->cinfo;
  jvirt_barray_ptr* arrays = jpeg_read_coefficients(c);
  if (arrays == nullptr) fail_with(r, UNROUND_JPEGIO_ERROR_DECODE, "libjpeg returned no coefficients");

  image->width = (uint32_t)c->image_width;
  image->height = (uint32_t)c->image_height;
  image->color_space = (int32_t)color_space_of(c->jpeg_color_space);
  image->num_components = (int32_t)c->num_components;
  image->max_h_samp_factor = (int32_t)c->max_h_samp_factor;
  image->max_v_samp_factor = (int32_t)c->max_v_samp_factor;
  image->progressive = c->progressive_mode != FALSE ? 1 : 0;
  image->arithmetic = c->arith_code != FALSE ? 1 : 0;

  for (int ci = 0; ci < c->num_components; ++ci) {
    const jpeg_component_info* comp = &c->comp_info[ci];
    unround_jpegio_component* out = &image->components[ci];
    out->id = (int32_t)comp->component_id;
    out->h_samp_factor = (int32_t)comp->h_samp_factor;
    out->v_samp_factor = (int32_t)comp->v_samp_factor;
    out->quant_table_slot = (int32_t)comp->quant_tbl_no;
    out->width = (uint32_t)comp->downsampled_width;
    out->height = (uint32_t)comp->downsampled_height;
    out->width_in_blocks = (uint32_t)comp->width_in_blocks;
    out->height_in_blocks = (uint32_t)comp->height_in_blocks;

    // libjpeg keeps a copy of the table of a component's slot when the
    // component's first scan begins. A component without one never had a
    // scan: the file ended before any of its data.
    const JQUANT_TBL* table = comp->quant_table;
    if (table == nullptr) {
      fail_with(r, UNROUND_JPEGIO_ERROR_DECODE, "the file ends before the data of every component");
    }
    for (int k = 0; k < DCTSIZE2; ++k) out->quant_table[k] = (uint16_t)table->quantval[k];

    size_t blocks = 0;
    size_t count = 0;
    if (ckd_mul(&blocks, (size_t)comp->width_in_blocks, (size_t)comp->height_in_blocks) ||
        ckd_mul(&count, blocks, (size_t)DCTSIZE2)) {
      fail_with(r, UNROUND_JPEGIO_ERROR_LIMIT, "the image is too large");
    }
    out->coefficients = (int16_t*)allocate(r, count, sizeof(int16_t));
    const size_t row_count = (size_t)comp->width_in_blocks * (size_t)DCTSIZE2;
    for (JDIMENSION row = 0; row < comp->height_in_blocks; ++row) {
      JBLOCKARRAY rows = (*c->mem->access_virt_barray)((j_common_ptr)c, arrays[ci], row, (JDIMENSION)1, FALSE);
      memcpy(out->coefficients + (size_t)row * row_count, rows[0], row_count * sizeof(JCOEF));
    }
  }

  JOCTET* icc = nullptr;
  unsigned int icc_size = 0;
  if (jpeg_read_icc_profile(c, &icc, &icc_size) != FALSE) {
    image->icc_profile = (uint8_t*)icc;  // allocated with malloc(), freed with free()
    image->icc_profile_size = (uint64_t)icc_size;
  }
  for (jpeg_saved_marker_ptr m = c->marker_list; m != nullptr; m = m->next) {
    if (m->marker != JPEG_APP0 + 1) continue;
    const int32_t orientation = exif_orientation((const uint8_t*)m->data, (size_t)m->data_length);
    if (orientation != 0) {
      image->exif_orientation = orientation;
      break;
    }
  }

  (void)jpeg_finish_decompress(c);
  image->warnings = r->warnings;
}

int32_t unround_jpegio_abi_version(void) { return UNROUND_JPEGIO_ABI_VERSION; }

const char* unround_jpegio_libjpeg_version(void) {
#if defined(LIBJPEG_TURBO_VERSION)
  return "libjpeg-turbo " UNROUND_TEXT(LIBJPEG_TURBO_VERSION) " (libjpeg API " UNROUND_TEXT(JPEG_LIB_VERSION) ")";
#else
  return "libjpeg (libjpeg API " UNROUND_TEXT(JPEG_LIB_VERSION) ")";
#endif
}

void unround_jpegio_image_free(unround_jpegio_image* image) {
  if (image == nullptr) return;
  for (size_t ci = 0; ci < UNROUND_JPEGIO_MAX_COMPONENTS; ++ci) free(image->components[ci].coefficients);
  free(image->icc_profile);
  *image = (unround_jpegio_image){};
}

unround_jpegio_status unround_jpegio_read(const uint8_t* data, size_t size, const unround_jpegio_options* options,
                                          unround_jpegio_image* image, char* message, size_t message_size) {
  if (image == nullptr || (message == nullptr && message_size > 0)) return UNROUND_JPEGIO_ERROR_ARGUMENT;
  *image = (unround_jpegio_image){};
  if ((data == nullptr && size > 0) || !size_fits(size)) {
    copy_message(message, message_size, "the data is not addressable as one buffer");
    return UNROUND_JPEGIO_ERROR_ARGUMENT;
  }
  reader* r = reader_new(options);
  if (r == nullptr) {
    copy_message(message, message_size, "out of memory");
    return UNROUND_JPEGIO_ERROR_MEMORY;
  }
  unround_jpegio_status status = UNROUND_JPEGIO_OK;
  switch (setjmp(r->jump)) {
    case 0:
      open_source(r, data, size);
      read_coefficients(r, image);
      status = UNROUND_JPEGIO_OK;
      break;
    default:
      status = r->failure;
      break;
  }
  jpeg_destroy_decompress(&r->cinfo);
  if (status != UNROUND_JPEGIO_OK) {
    unround_jpegio_image_free(image);
    copy_message(message, message_size, r->message);
  } else {
    copy_message(message, message_size, r->warnings > 0 ? r->message : "");
  }
  free(r);
  return status;
}

static void decode(reader* r, unround_jpegio_planes* planes) {
  const j_decompress_ptr c = &r->cinfo;
  c->raw_data_out = TRUE;
  c->do_block_smoothing = FALSE;  // the planes are the inverse DCT of the coefficients and nothing else
  c->do_fancy_upsampling = FALSE;
  c->dct_method = JDCT_ISLOW;
  c->out_color_space = c->jpeg_color_space;
  (void)jpeg_start_decompress(c);
  if (c->raw_data_out == FALSE) {
    fail_with(r, UNROUND_JPEGIO_ERROR_UNSUPPORTED, "libjpeg cannot return the planes of this file");
  }

  planes->num_components = (int32_t)c->num_components;
  // Each call returns one iMCU row: v_samp_factor * 8 rows of every
  // component. In the last one libjpeg writes only the rows of blocks that
  // carry image data, and in every one only the columns of such blocks, but
  // it is handed pointers to every row; the planes are allocated as tall as
  // the calls reach and report the rows of image blocks.
  for (int ci = 0; ci < c->num_components; ++ci) {
    const jpeg_component_info* comp = &c->comp_info[ci];
    unround_jpegio_plane* plane = &planes->planes[ci];
    plane->width = (uint32_t)comp->width_in_blocks * (uint32_t)DCTSIZE;
    plane->height = (uint32_t)comp->height_in_blocks * (uint32_t)DCTSIZE;
    plane->stride = plane->width;
    size_t rows = 0;
    size_t samples = 0;
    if (ckd_mul(&rows, (size_t)c->total_iMCU_rows, (size_t)comp->v_samp_factor * (size_t)DCTSIZE) ||
        ckd_mul(&samples, rows, (size_t)plane->stride)) {
      fail_with(r, UNROUND_JPEGIO_ERROR_LIMIT, "the image is too large");
    }
    plane->samples = (uint8_t*)allocate(r, samples, sizeof(uint8_t));
  }

  JSAMPROW row_pointers[UNROUND_JPEGIO_MAX_COMPONENTS][MAX_SAMP_FACTOR * DCTSIZE];
  JSAMPARRAY component_rows[UNROUND_JPEGIO_MAX_COMPONENTS];
  const JDIMENSION rows_per_call = (JDIMENSION)(c->max_v_samp_factor * DCTSIZE);
  size_t imcu_row = 0;
  while (c->output_scanline < c->output_height) {
    for (int ci = 0; ci < c->num_components; ++ci) {
      const size_t tall = (size_t)c->comp_info[ci].v_samp_factor * (size_t)DCTSIZE;
      const unround_jpegio_plane* plane = &planes->planes[ci];
      for (size_t i = 0; i < tall; ++i) {
        row_pointers[ci][i] = (JSAMPROW)(plane->samples + (imcu_row * tall + i) * (size_t)plane->stride);
      }
      component_rows[ci] = row_pointers[ci];
    }
    if (jpeg_read_raw_data(c, component_rows, rows_per_call) == 0) {
      fail_with(r, UNROUND_JPEGIO_ERROR_DECODE, "libjpeg returned no rows");
    }
    imcu_row += 1;
  }
  (void)jpeg_finish_decompress(c);
}

void unround_jpegio_planes_free(unround_jpegio_planes* planes) {
  if (planes == nullptr) return;
  for (size_t ci = 0; ci < UNROUND_JPEGIO_MAX_COMPONENTS; ++ci) free(planes->planes[ci].samples);
  *planes = (unround_jpegio_planes){};
}

unround_jpegio_status unround_jpegio_decode_planes(const uint8_t* data, size_t size,
                                                   const unround_jpegio_options* options, unround_jpegio_planes* planes,
                                                   char* message, size_t message_size) {
  if (planes == nullptr || (message == nullptr && message_size > 0)) return UNROUND_JPEGIO_ERROR_ARGUMENT;
  *planes = (unround_jpegio_planes){};
  if ((data == nullptr && size > 0) || !size_fits(size)) {
    copy_message(message, message_size, "the data is not addressable as one buffer");
    return UNROUND_JPEGIO_ERROR_ARGUMENT;
  }
  reader* r = reader_new(options);
  if (r == nullptr) {
    copy_message(message, message_size, "out of memory");
    return UNROUND_JPEGIO_ERROR_MEMORY;
  }
  unround_jpegio_status status = UNROUND_JPEGIO_OK;
  switch (setjmp(r->jump)) {
    case 0:
      open_source(r, data, size);
      decode(r, planes);
      status = UNROUND_JPEGIO_OK;
      break;
    default:
      status = r->failure;
      break;
  }
  jpeg_destroy_decompress(&r->cinfo);
  if (status != UNROUND_JPEGIO_OK) {
    unround_jpegio_planes_free(planes);
    copy_message(message, message_size, r->message);
  } else {
    copy_message(message, message_size, r->warnings > 0 ? r->message : "");
  }
  free(r);
  return status;
}
