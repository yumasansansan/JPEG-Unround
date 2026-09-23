// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The C layer between libjpeg-turbo and the three implementations of
// JPEG-Unround (C++, Rust and Python). It reads what the reconstruction needs
// from a JPEG file held in memory -- the quantized DCT coefficients of every
// component, the quantization table each was quantized with, the sampling
// factors, the ICC profile and the EXIF orientation -- and it decodes the
// component planes the way libjpeg's standard decoder does, before upsampling
// and color conversion.
//
// libjpeg reports errors with longjmp, and the jump never leaves this layer:
// every function returns a status, and the message of a failure is copied into
// the caller's buffer. Nothing here keeps global state, so different images
// can be read on different threads at once.
//
// Coefficients and quantization tables are in natural (row-major) order, not
// in the zigzag order of the file: coefficient k of a block, times entry k of
// its component's table, is the value the encoder quantized, and it lies
// within half a step of the DCT of the encoder's input.
//
// The structures are laid out with fixed-width fields only, so that ctypes
// (Python) and a hand-written FFI (Rust) can mirror them;
// unround_jpegio_abi_version() changes whenever their layout does.
#ifndef UNROUND_JPEGIO_H
#define UNROUND_JPEGIO_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#if defined(UNROUND_JPEGIO_BUILDING_SHARED)
#define UNROUND_JPEGIO_API __declspec(dllexport)
#else
#define UNROUND_JPEGIO_API
#endif
#else
#define UNROUND_JPEGIO_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define UNROUND_JPEGIO_ABI_VERSION 1
#define UNROUND_JPEGIO_MAX_COMPONENTS 4
#define UNROUND_JPEGIO_BLOCK_SIZE 64

// What a function of this layer reports.
typedef enum unround_jpegio_status : int32_t {
  UNROUND_JPEGIO_OK = 0,
  UNROUND_JPEGIO_ERROR_ARGUMENT = 1,     // a null pointer, or a size this layer cannot pass on
  UNROUND_JPEGIO_ERROR_DECODE = 2,       // libjpeg could not read the data
  UNROUND_JPEGIO_ERROR_UNSUPPORTED = 3,  // a JPEG outside this project's scope (12-bit, lossless, CMYK, ...)
  UNROUND_JPEGIO_ERROR_LIMIT = 4,        // larger than the options allow
  UNROUND_JPEGIO_ERROR_MEMORY = 5,       // an allocation failed
} unround_jpegio_status;

// The color space of the components, as the file declares it.
typedef enum unround_jpegio_color_space : int32_t {
  UNROUND_JPEGIO_GRAYSCALE = 1,
  UNROUND_JPEGIO_YCBCR = 2,
  UNROUND_JPEGIO_RGB = 3,  // three components stored without a color transform
} unround_jpegio_color_space;

// Limits a read accepts. A null pointer where options are expected means the
// defaults. A file declares its size before any of its data, and libjpeg sets
// aside the memory for all of it from that declaration: max_pixels is what
// keeps a small file from asking for gigabytes.
typedef struct unround_jpegio_options {
  uint64_t max_pixels;          // width * height; 0 means 1 << 28
  int32_t max_scans;            // SOS markers of a progressive file; 0 means 500
  int32_t warnings_are_errors;  // nonzero: a corrupt-data warning fails the read
} unround_jpegio_options;

typedef struct unround_jpegio_component {
  int32_t id;                // the component identifier of the SOF marker
  int32_t h_samp_factor;     // 1 to 4
  int32_t v_samp_factor;     // 1 to 4
  int32_t quant_table_slot;  // 0 to 3: the DQT slot the SOF marker names for the component
  uint32_t width;            // samples of this component that cover the image
  uint32_t height;
  uint32_t width_in_blocks;  // ceil(width / 8): the blocks that carry image data
  uint32_t height_in_blocks;
  // The quantization table the component was quantized with, in natural
  // order: the contents of its slot when its first scan began, as libjpeg
  // keeps them. (A file may define a slot anew between scans.) Entries are 1
  // to 255 for baseline files and up to 65535 for extended ones.
  uint16_t quant_table[UNROUND_JPEGIO_BLOCK_SIZE];
  // height_in_blocks * width_in_blocks blocks of 64 coefficients, block rows
  // from the top, blocks from the left, coefficients in natural order. The
  // dummy blocks an encoder adds to complete an MCU are not included: they
  // carry no image data.
  int16_t* coefficients;
} unround_jpegio_component;

typedef struct unround_jpegio_image {
  uint32_t width;
  uint32_t height;
  int32_t color_space;     // unround_jpegio_color_space
  int32_t num_components;  // 1 or 3
  int32_t max_h_samp_factor;
  int32_t max_v_samp_factor;
  int32_t progressive;       // 1 when the file is progressive
  int32_t arithmetic;        // 1 when the file is arithmetic-coded
  int32_t exif_orientation;  // 1 to 8 from the EXIF APP1 marker, or 0 when there is none
  int32_t warnings;          // corrupt-data warnings libjpeg reported while reading
  unround_jpegio_component components[UNROUND_JPEGIO_MAX_COMPONENTS];
  uint8_t* icc_profile;  // the ICC profile of the APP2 markers, or a null pointer
  uint64_t icc_profile_size;
} unround_jpegio_image;

// One component decoded to 8-bit samples by libjpeg: the inverse DCT of its
// coefficients with libjpeg's accurate integer method, level-shifted and
// clamped to 0-255, with no block smoothing, no upsampling and no color
// conversion.
typedef struct unround_jpegio_plane {
  uint32_t width;   // width_in_blocks * 8: the image samples, padded to whole blocks
  uint32_t height;  // height_in_blocks * 8
  uint32_t stride;  // samples from one row to the next
  uint32_t reserved;
  uint8_t* samples;
} unround_jpegio_plane;

typedef struct unround_jpegio_planes {
  int32_t num_components;
  int32_t reserved;
  unround_jpegio_plane planes[UNROUND_JPEGIO_MAX_COMPONENTS];
} unround_jpegio_planes;

// UNROUND_JPEGIO_ABI_VERSION of the library that is loaded.
UNROUND_JPEGIO_API int32_t unround_jpegio_abi_version(void);

// The name and version of the libjpeg this layer is built with, such as
// "libjpeg-turbo 3.2.0 (libjpeg API 62)".
UNROUND_JPEGIO_API const char* unround_jpegio_libjpeg_version(void);

// Reads the coefficients and the metadata of a JPEG file in memory. On
// success, image owns what it points to until unround_jpegio_image_free(), and
// message (when message_size > 0) holds the first corrupt-data warning, or an
// empty string when there was none. On failure, image is left empty and
// message holds a description of the failure. Messages are NUL-terminated and
// cut to message_size; message may be a null pointer when message_size is 0.
[[nodiscard]] UNROUND_JPEGIO_API unround_jpegio_status unround_jpegio_read(const uint8_t* data, size_t size,
                                                                           const unround_jpegio_options* options,
                                                                           unround_jpegio_image* image, char* message,
                                                                           size_t message_size);

// Frees what a successful unround_jpegio_read() allocated, and empties image.
// A null pointer, or an empty image, is accepted.
UNROUND_JPEGIO_API void unround_jpegio_image_free(unround_jpegio_image* image);

// Decodes the component planes of a JPEG file in memory, with the same
// contract as unround_jpegio_read().
[[nodiscard]] UNROUND_JPEGIO_API unround_jpegio_status
unround_jpegio_decode_planes(const uint8_t* data, size_t size, const unround_jpegio_options* options,
                             unround_jpegio_planes* planes, char* message, size_t message_size);

// Frees what a successful unround_jpegio_decode_planes() allocated, and
// empties planes. A null pointer, or empty planes, is accepted.
UNROUND_JPEGIO_API void unround_jpegio_planes_free(unround_jpegio_planes* planes);

#ifdef __cplusplus
}
#endif

#endif
