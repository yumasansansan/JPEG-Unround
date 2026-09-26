// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The C interface of JPEG-Unround: the reference implementation (Rust) as a
// library for any language that calls C. The Python package calls it through
// ctypes.
//
// Settings are made from options by the names of the command line
// (docs/cli.md), as strings. A JPEG file in memory, or components given as
// arrays, are reconstructed into a result, whose arrays stay valid until the
// result is freed; an observer can follow the solver's records and stop it.
// The writers of TIFF, PNG and PNM, the turning of a picture upright, JFIF's
// conversion, the command line itself, and the C layer's reading of JPEG files
// (unround/jpegio.h) are here too.
//
// Every function that computes returns a status, and says what went wrong in a
// message buffer that the caller gives (NUL-terminated, cut at a character's
// boundary where it is too short; a null buffer is allowed). No error of the
// implementation crosses this interface as anything but a status. Different
// settings and results can be used on different threads at once, and one can be
// read by several.
//
// The structures are laid out with fixed-width fields only, so that ctypes can
// mirror them; UNROUND_ABI_VERSION changes whenever their layout, or a function's
// declaration, does.
#ifndef UNROUND_H
#define UNROUND_H

#include <stddef.h>
#include <stdint.h>

#include "unround/jpegio.h"

#ifdef __cplusplus
extern "C" {
#endif

#define UNROUND_ABI_VERSION 2

// What a function of the interface reports.
typedef enum unround_status : int32_t {
  UNROUND_OK = 0,
  UNROUND_ERROR_ARGUMENT = 1,     // a null pointer where one is needed, or a size that does not fit
  UNROUND_ERROR_READ = 2,         // a file that the C layer could not read, or would not
  UNROUND_ERROR_UNSUPPORTED = 3,  // a file whose layout JPEG-Unround does not take
  UNROUND_ERROR_OPTIONS = 4,      // options, or arrays, out of their ranges
  UNROUND_ERROR_IO = 5,           // a file that could not be written or read
  UNROUND_ERROR_INTERNAL = 6,     // an error of JPEG-Unround itself
} unround_status;

int32_t unround_abi_version(void);

// The version of JPEG-Unround and the implementation: static storage.
const char* unround_version(void);

// ---------------------------------------------------------------------------
// Settings

typedef struct unround_settings unround_settings;

// Settings from count options, each a NUL-terminated UTF-8 string, as the command
// line takes them without its files and its options of the output: "--mu",
// "0.01", "--method=tv". Null where they are refused, with every refusal in the
// message.
unround_settings* unround_settings_new(size_t count, const char* const* options, char* message, size_t message_size);
void unround_settings_free(unround_settings* settings);

// Writes the options that give the settings, as unround_settings_new takes them,
// into buffer: "--name=value" or "--flag" one after another, a space between, each
// number as the shortest decimal that reads back as the same double;
// NUL-terminated, and cut where the buffer is too short. Returns the length of the
// whole text, without the NUL; 0 for null settings.
size_t unround_settings_options(const unround_settings* settings, char* buffer, size_t size);

// ---------------------------------------------------------------------------
// Reconstruction

// A record of the solver, as an observer sees it: the iteration, its gap per
// sample (what the tolerance is compared with), the primal and dual values, and
// the canvas of channels x height x width samples, valid for the length of the
// call.
typedef struct unround_record {
  uint64_t iteration;
  double gap;
  double primal;
  double dual;
  uint64_t channels;
  uint64_t height;
  uint64_t width;
  const double* canvas;
} unround_record;

// Called with each record after the first; a nonzero return stops the solver.
typedef int32_t (*unround_observer)(void* user, const unround_record* record);

// A component given as arrays: rows x columns blocks of 64 levels in natural
// order, block rows from the top.
typedef struct unround_component {
  const int16_t* levels;
  uint32_t rows;
  uint32_t columns;
  int32_t h_samp_factor;
  int32_t v_samp_factor;
  uint16_t quant_table[64];  // natural order
} unround_component;

// Components given as arrays: the picture's size, its color space
// (unround_jpegio_color_space), and count components.
typedef struct unround_input {
  uint32_t height;
  uint32_t width;
  int32_t color_space;
  uint32_t count;
  const unround_component* components;
} unround_input;

typedef struct unround_result unround_result;

// Reconstructs a JPEG file of size bytes at data into *result, which
// unround_result_free frees. observer, where it is not null, is called with user
// and each record after the first.
unround_status unround_decode(const uint8_t* data, size_t size, const unround_settings* settings,
                              unround_observer observer, void* user, unround_result** result, char* message,
                              size_t message_size);

// Reconstructs components given as arrays, as unround_decode does a file.
unround_status unround_solve(const unround_input* input, const unround_settings* settings, unround_observer observer,
                             void* user, unround_result** result, char* message, size_t message_size);

void unround_result_free(unround_result* result);

// The picture: height x width x channels binary64 samples, interleaved,
// greyscale or RGB. The sizes are written where the pointers are not null.
const double* unround_result_picture(const unround_result* result, uint64_t* height, uint64_t* width,
                                     uint64_t* channels);

// The canvas cut to the picture, channel by channel: the Y, Cb and Cr of a file
// in YCbCr.
const double* unround_result_planes(const unround_result* result, uint64_t* count);

// The file's color space (unround_jpegio_color_space); 0 for a null result.
int32_t unround_result_color_space(const unround_result* result);

// The whole canvas, channels x height x width, as the solver left it.
const double* unround_result_canvas(const unround_result* result, uint64_t* channels, uint64_t* height,
                                    uint64_t* width);

uint64_t unround_result_components(const unround_result* result);

// The coefficients of a component, each within its interval, 64 to a block;
// null for a component that is not there.
const double* unround_result_coefficients(const unround_result* result, uint64_t component, uint64_t* count);

// What the model of a component holds: which 0, 1 and 2 the lower and upper ends
// of the intervals and the centres of the data term, one for each coefficient;
// 3 and 4 the quantization steps and the weights of the data term, one for each
// frequency (64).
const double* unround_result_problem(const unround_result* result, uint64_t component, int32_t which, uint64_t* count);

// A component's block rows and columns, and the canvas's samples per sample of
// the component down and across.
unround_status unround_result_component(const unround_result* result, uint64_t component, uint64_t* rows,
                                        uint64_t* columns, uint64_t* down, uint64_t* across);

// 1 where a solver made the result, with its iterations and why it stopped (1 a
// tolerance, 2 the most iterations, 3 the observer, 4 a subgradient of 0); 0 for
// the decoder of the centres.
int32_t unround_result_solver(const unround_result* result, uint64_t* iterations, int32_t* stop);

// The iterations of the solver's records.
const uint64_t* unround_result_record_iterations(const unround_result* result, uint64_t* count);

// A value of every record: which 0 the solver's seconds, 1 the primal value, 2 the
// dual value, 3 TGV's scaling of the dual, 4 TGV's partial gap (NaN where not
// taken).
const double* unround_result_record_values(const unround_result* result, int32_t which, uint64_t* count);

// The file's ICC profile, or null; and its EXIF orientation, 1 to 8, or 0.
const uint8_t* unround_result_icc_profile(const unround_result* result, uint64_t* size);
int32_t unround_result_exif_orientation(const unround_result* result);

// ---------------------------------------------------------------------------
// Output and conversion

// The bytes of a TIFF of binary64 samples, height x width x channels (1, or 3
// interleaved), exactly as they are; with ycbcr nonzero, JFIF's Y, Cb and Cr,
// which the file declares. The ICC profile of icc_size bytes at icc (none where
// icc is null) is embedded where it goes with the picture (unround_jpeg_check_icc);
// where it does not, it is left out, and on success the message says why (it
// holds that warning, or an empty string). unround_bytes_free frees the bytes.
unround_status unround_tiff(const double* samples, uint64_t height, uint64_t width, uint64_t channels, int32_t ycbcr,
                            const uint8_t* icc, uint64_t icc_size, uint8_t** bytes, uint64_t* size, char* message,
                            size_t message_size);

// The bytes of a PNG file of 8 bits, or with sixteen nonzero 16 bits
// (docs/cli.md), compressed at zlib's level compression, 0 to 9, or -1 for
// zlib's default; with the ICC profile as unround_tiff takes it.
unround_status unround_png(const double* samples, uint64_t height, uint64_t width, uint64_t channels, int32_t sixteen,
                           int32_t compression, const uint8_t* icc, uint64_t icc_size, uint8_t** bytes, uint64_t* size,
                           char* message, size_t message_size);

// The bytes of a PNM file of 8 bits, or with sixteen nonzero 16 bits
// (docs/cli.md).
unround_status unround_pnm(const double* samples, uint64_t height, uint64_t width, uint64_t channels, int32_t sixteen,
                           uint8_t** bytes, uint64_t* size, char* message, size_t message_size);

void unround_bytes_free(uint8_t* bytes, uint64_t size);

// The picture of height x width pixels of channels interleaved samples turned
// upright as EXIF's orientation says (1 to 8, or 0 for none), into turned, which
// holds as many doubles; its rows and columns into *turned_height and
// *turned_width where they are not null. The samples move bit for bit.
unround_status unround_orient(const double* samples, uint64_t height, uint64_t width, uint64_t channels,
                              int32_t orientation, double* turned, uint64_t* turned_height, uint64_t* turned_width,
                              char* message, size_t message_size);

// JFIF's RGB, height x width x 3 interleaved, of Y, Cb and Cr planes one after
// another; and back.
unround_status unround_to_rgb(const double* planes, uint64_t height, uint64_t width, double* picture);
unround_status unround_to_ycbcr(const double* picture, uint64_t height, uint64_t width, double* planes);

// The command line on count arguments, without the program's name: its exit
// status (docs/cli.md).
int32_t unround_main(size_t count, const char* const* arguments);

// ---------------------------------------------------------------------------
// The C layer (unround/jpegio.h), through this library

int32_t unround_jpeg_abi_version(void);
const char* unround_jpeg_libjpeg_version(void);
const char* unround_jpeg_libpng_version(void);
int32_t unround_jpeg_check_icc(const uint8_t* profile, uint64_t size, int32_t channels, char* reason,
                               size_t reason_size);
unround_jpegio_status unround_jpeg_read(const uint8_t* data, size_t size, const unround_jpegio_options* options,
                                        unround_jpegio_image* image, char* message, size_t message_size);
void unround_jpeg_image_free(unround_jpegio_image* image);
unround_jpegio_status unround_jpeg_decode_planes(const uint8_t* data, size_t size,
                                                 const unround_jpegio_options* options, unround_jpegio_planes* planes,
                                                 char* message, size_t message_size);
void unround_jpeg_planes_free(unround_jpegio_planes* planes);

#ifdef __cplusplus
}
#endif

static_assert(sizeof(unround_record) == 64, "the layout of unround_record");
static_assert(sizeof(unround_component) == 152, "the layout of unround_component");
static_assert(offsetof(unround_component, quant_table) == 24, "the layout of unround_component");
static_assert(sizeof(unround_input) == 24, "the layout of unround_input");
static_assert(offsetof(unround_input, components) == 16, "the layout of unround_input");

#endif
