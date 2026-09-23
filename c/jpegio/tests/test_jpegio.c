// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Tests of the C layer. Files are written by libjpeg's encoder from
// coefficients chosen here (support/test_jpeg.c), in every mode the layer
// reads -- sequential and progressive, Huffman and arithmetic coding, with and
// without restart markers, every usual chroma subsampling, sizes that are and
// are not whole MCUs -- and what the layer reads back has to be what went in,
// exactly. The planes have to be the inverse DCT of those coefficients: to
// within one level against an inverse DCT computed here in double precision,
// and exactly against libjpeg's own decoding where no upsampling is involved.
// Then the refusals, the limits, damaged files and wrong arguments.

#include "test_jpeg.h"
#include "unround/jpegio.h"

#include <math.h>
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

// ---------------------------------------------------------------------------
// Deterministic pseudo-random numbers (SplitMix64)

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

// A number from low to high, both included.
static int32_t rng_range(rng* g, int32_t low, int32_t high) {
  const uint64_t span = (uint64_t)((int64_t)high - (int64_t)low + 1);
  return (int32_t)((int64_t)low + (int64_t)(rng_next(g) % span));
}

// ---------------------------------------------------------------------------
// The DCT of JPEG, in double precision

static constexpr double pi = 3.14159265358979323846;

// cosines[k][n] = C(k) / 2 * cos((2n + 1) k pi / 16), so that the forward and
// the inverse transform are both sums of products with it.
static double cosines[8][8];

static void init_cosines(void) {
  for (int k = 0; k < 8; ++k) {
    const double scale = (k == 0 ? sqrt(0.5) : 1.0) / 2.0;
    for (int n = 0; n < 8; ++n) cosines[k][n] = scale * cos((double)(2 * n + 1) * (double)k * pi / 16.0);
  }
}

static void forward_dct(const double samples[64], double coefficients[64]) {
  for (int v = 0; v < 8; ++v) {
    for (int u = 0; u < 8; ++u) {
      double sum = 0.0;
      for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) sum += samples[y * 8 + x] * cosines[u][x] * cosines[v][y];
      }
      coefficients[v * 8 + u] = sum;
    }
  }
}

static void inverse_dct(const double coefficients[64], double samples[64]) {
  for (int y = 0; y < 8; ++y) {
    for (int x = 0; x < 8; ++x) {
      double sum = 0.0;
      for (int v = 0; v < 8; ++v) {
        for (int u = 0; u < 8; ++u) sum += coefficients[v * 8 + u] * cosines[u][x] * cosines[v][y];
      }
      samples[y * 8 + x] = sum;
    }
  }
}

static double clamp(double value, double low, double high) { return value < low ? low : (value > high ? high : value); }

// ---------------------------------------------------------------------------
// Test images

static void random_table(rng* g, uint16_t table[64], int32_t low, int32_t high) {
  for (int k = 0; k < 64; ++k) table[k] = (uint16_t)rng_range(g, low, high);
}

// The coefficients of a picture of waves, steps and noise, quantized with the
// table: what an encoder of 8-bit samples could have written.
static int16_t* picture_coefficients(rng* g, uint32_t wide, uint32_t high, const uint16_t table[64]) {
  int16_t* coefficients = (int16_t*)malloc((size_t)wide * high * 64 * sizeof(int16_t) + 1);
  if (coefficients == nullptr) abort();
  const double phase = (double)rng_range(g, 0, 628) / 100.0;
  for (uint32_t by = 0; by < high; ++by) {
    for (uint32_t bx = 0; bx < wide; ++bx) {
      double samples[64];
      for (uint32_t y = 0; y < 8; ++y) {
        for (uint32_t x = 0; x < 8; ++x) {
          const uint32_t px = bx * 8 + x;
          const uint32_t py = by * 8 + y;
          double value = 128.0 + 70.0 * sin((double)px / 9.0 + phase) * cos((double)py / 7.0);
          if ((px / 13 + py / 11) % 3 == 0) value += 60.0;
          value += (double)rng_range(g, -24, 24);
          samples[y * 8 + x] = clamp(value, 0.0, 255.0) - 128.0;
        }
      }
      double transformed[64];
      forward_dct(samples, transformed);
      int16_t* block = coefficients + ((size_t)by * wide + bx) * 64;
      for (int k = 0; k < 64; ++k) block[k] = (int16_t)lround(transformed[k] / (double)table[k]);
    }
  }
  return coefficients;
}

// Coefficients from anywhere in the range an 8-bit baseline file can code: a
// DC coefficient within 1016 of zero, so that two neighbours differ by less
// than 2048, and AC coefficients of up to 1023.
static int16_t* extreme_coefficients(rng* g, uint32_t wide, uint32_t high) {
  const size_t count = (size_t)wide * high * 64;
  int16_t* coefficients = (int16_t*)malloc(count * sizeof(int16_t) + 1);
  if (coefficients == nullptr) abort();
  for (size_t i = 0; i < count; ++i) {
    const bool dc = i % 64 == 0;
    coefficients[i] =
        (int16_t)(dc ? rng_range(g, -1016, 1016) : (rng_range(g, 0, 2) == 0 ? rng_range(g, -1023, 1023) : 0));
  }
  return coefficients;
}

typedef struct encoded {
  test_jpeg spec;
  int16_t* coefficients[TEST_JPEG_MAX_COMPONENTS];
  uint8_t* data;
  size_t size;
} encoded;

static void encoded_free(encoded* e) {
  for (int ci = 0; ci < TEST_JPEG_MAX_COMPONENTS; ++ci) free(e->coefficients[ci]);
  free(e->data);
  *e = (encoded){};
}

// Fills in the coefficients of every component and encodes the file.
static bool encode(encoded* e, rng* g, bool extreme) {
  const int32_t components = test_jpeg_components(&e->spec);
  for (int32_t ci = 0; ci < components; ++ci) {
    const uint32_t wide = test_jpeg_blocks_wide(&e->spec, ci);
    const uint32_t high = test_jpeg_blocks_high(&e->spec, ci);
    const uint16_t* table = e->spec.quant_tables[e->spec.quant_table_slot[ci]];
    e->coefficients[ci] = extreme ? extreme_coefficients(g, wide, high) : picture_coefficients(g, wide, high, table);
    e->spec.coefficients[ci] = e->coefficients[ci];
  }
  char message[256];
  const int result = test_jpeg_write(&e->spec, &e->data, &e->size, message, sizeof message);
  if (result != 0) (void)fprintf(stderr, "  the test encoder failed: %s\n", message);
  return CHECK(result == 0);
}

// ---------------------------------------------------------------------------
// Reading back

static void check_read_back(const encoded* e, const unround_jpegio_image* image) {
  const test_jpeg* spec = &e->spec;
  const int32_t components = test_jpeg_components(spec);
  CHECK(image->width == spec->width);
  CHECK(image->height == spec->height);
  CHECK(image->num_components == components);
  CHECK(image->color_space == (spec->color_space == TEST_JPEG_GRAYSCALE ? UNROUND_JPEGIO_GRAYSCALE
                               : spec->color_space == TEST_JPEG_RGB     ? UNROUND_JPEGIO_RGB
                                                                        : UNROUND_JPEGIO_YCBCR));
  CHECK(image->progressive == (spec->progressive != 0 ? 1 : 0));
  CHECK(image->arithmetic == (spec->arithmetic != 0 ? 1 : 0));
  CHECK(image->warnings == 0);
  CHECK(image->exif_orientation == 0);
  int32_t h_max = 1;
  int32_t v_max = 1;
  for (int32_t ci = 0; ci < components; ++ci) {
    const int32_t h = spec->h_samp_factor[ci] > 0 ? spec->h_samp_factor[ci] : 1;
    const int32_t v = spec->v_samp_factor[ci] > 0 ? spec->v_samp_factor[ci] : 1;
    h_max = h > h_max ? h : h_max;
    v_max = v > v_max ? v : v_max;
  }
  CHECK(image->max_h_samp_factor == h_max);
  CHECK(image->max_v_samp_factor == v_max);
  for (int32_t ci = 0; ci < components; ++ci) {
    const unround_jpegio_component* c = &image->components[ci];
    const int32_t h = spec->h_samp_factor[ci] > 0 ? spec->h_samp_factor[ci] : 1;
    const int32_t v = spec->v_samp_factor[ci] > 0 ? spec->v_samp_factor[ci] : 1;
    CHECK(c->h_samp_factor == h);
    CHECK(c->v_samp_factor == v);
    CHECK(c->quant_table_slot == spec->quant_table_slot[ci]);
    CHECK(c->width == (uint32_t)(((uint64_t)spec->width * (uint64_t)h + (uint64_t)h_max - 1) / (uint64_t)h_max));
    CHECK(c->height == (uint32_t)(((uint64_t)spec->height * (uint64_t)v + (uint64_t)v_max - 1) / (uint64_t)v_max));
    CHECK(c->width_in_blocks == test_jpeg_blocks_wide(spec, ci));
    CHECK(c->height_in_blocks == test_jpeg_blocks_high(spec, ci));
    CHECK(c->width_in_blocks == (c->width + 7) / 8);
    CHECK(c->height_in_blocks == (c->height + 7) / 8);
    CHECK(memcmp(c->quant_table, spec->quant_tables[spec->quant_table_slot[ci]], sizeof c->quant_table) == 0);
    const size_t count = (size_t)c->width_in_blocks * c->height_in_blocks * 64;
    CHECK(c->coefficients != nullptr);
    if (c->coefficients != nullptr) CHECK(memcmp(c->coefficients, e->coefficients[ci], count * sizeof(int16_t)) == 0);
  }
  for (int32_t ci = components; ci < UNROUND_JPEGIO_MAX_COMPONENTS; ++ci) {
    CHECK(image->components[ci].coefficients == nullptr);
  }
  CHECK(image->icc_profile == nullptr && image->icc_profile_size == 0);
}

// The largest difference between a plane and the inverse DCT, computed here,
// of the coefficients it was decoded from.
static int plane_error(const unround_jpegio_plane* plane, const unround_jpegio_component* c) {
  int worst = 0;
  for (uint32_t by = 0; by < c->height_in_blocks; ++by) {
    for (uint32_t bx = 0; bx < c->width_in_blocks; ++bx) {
      const int16_t* block = c->coefficients + ((size_t)by * c->width_in_blocks + bx) * 64;
      double dequantized[64];
      for (int k = 0; k < 64; ++k) dequantized[k] = (double)block[k] * (double)c->quant_table[k];
      double samples[64];
      inverse_dct(dequantized, samples);
      for (uint32_t y = 0; y < 8; ++y) {
        for (uint32_t x = 0; x < 8; ++x) {
          const long expected = lround(clamp(samples[y * 8 + x] + 128.0, 0.0, 255.0));
          const int actual = plane->samples[((size_t)by * 8 + y) * plane->stride + (size_t)bx * 8 + x];
          const int difference = abs(actual - (int)expected);
          worst = difference > worst ? difference : worst;
        }
      }
    }
  }
  return worst;
}

static void check_planes(const encoded* e, const unround_jpegio_image* image) {
  unround_jpegio_planes planes = {};
  char message[256];
  const unround_jpegio_status status =
      unround_jpegio_decode_planes(e->data, e->size, nullptr, &planes, message, sizeof message);
  if (!CHECK(status == UNROUND_JPEGIO_OK)) {
    (void)fprintf(stderr, "  %s\n", message);
    return;
  }
  bool decoded = CHECK(planes.num_components == image->num_components);
  bool upsampled = false;
  for (int32_t ci = 0; ci < image->num_components; ++ci) {
    const unround_jpegio_component* c = &image->components[ci];
    const unround_jpegio_plane* p = &planes.planes[ci];
    CHECK(p->width == c->width_in_blocks * 8);
    CHECK(p->height == c->height_in_blocks * 8);
    CHECK(p->stride >= p->width);
    if (CHECK(p->samples != nullptr)) {
      CHECK(plane_error(p, c) <= 1);
    } else {
      decoded = false;
    }
    if (c->h_samp_factor != image->max_h_samp_factor || c->v_samp_factor != image->max_v_samp_factor) {
      upsampled = true;
    }
  }

  // Without upsampling, libjpeg's standard decoding is the planes, cropped.
  if (decoded && !upsampled) {
    uint8_t* samples = nullptr;
    uint32_t width = 0;
    uint32_t height = 0;
    int32_t components = 0;
    if (CHECK(test_jpeg_decode(e->data, e->size, &samples, &width, &height, &components, message, sizeof message) ==
              0) &&
        CHECK(width == image->width && height == image->height && components == image->num_components)) {
      bool same = true;
      for (uint32_t y = 0; y < height && same; ++y) {
        for (uint32_t x = 0; x < width && same; ++x) {
          for (int32_t ci = 0; ci < components; ++ci) {
            const unround_jpegio_plane* p = &planes.planes[ci];
            if (samples[((size_t)y * width + x) * (size_t)components + (size_t)ci] !=
                p->samples[(size_t)y * p->stride + x]) {
              same = false;
            }
          }
        }
      }
      CHECK(same);
    }
    free(samples);
  }
  unround_jpegio_planes_free(&planes);
  CHECK(planes.num_components == 0 && planes.planes[0].samples == nullptr);
}

typedef struct layout {
  const char* name;
  int32_t color_space;
  int32_t h[3];
  int32_t v[3];
} layout;

typedef struct mode {
  const char* name;
  int32_t progressive;
  int32_t arithmetic;
  int32_t optimize_coding;
  int32_t restart_interval;
  int32_t sequential_scans;
} mode;

static const layout layouts[] = {
    {.name = "grayscale", .color_space = TEST_JPEG_GRAYSCALE, .h = {1}, .v = {1}},
    {.name = "grayscale 2x2", .color_space = TEST_JPEG_GRAYSCALE, .h = {2}, .v = {2}},
    {.name = "4:4:4", .color_space = TEST_JPEG_YCBCR, .h = {1, 1, 1}, .v = {1, 1, 1}},
    {.name = "4:2:2", .color_space = TEST_JPEG_YCBCR, .h = {2, 1, 1}, .v = {1, 1, 1}},
    {.name = "4:2:0", .color_space = TEST_JPEG_YCBCR, .h = {2, 1, 1}, .v = {2, 1, 1}},
    {.name = "4:4:0", .color_space = TEST_JPEG_YCBCR, .h = {1, 1, 1}, .v = {2, 1, 1}},
    {.name = "4:1:1", .color_space = TEST_JPEG_YCBCR, .h = {4, 1, 1}, .v = {1, 1, 1}},
    {.name = "RGB", .color_space = TEST_JPEG_RGB, .h = {1, 1, 1}, .v = {1, 1, 1}},
    {.name = "RGB, green 2x1", .color_space = TEST_JPEG_RGB, .h = {1, 2, 1}, .v = {1, 1, 1}},
};

static const mode modes[] = {
    {.name = "baseline"},
    {.name = "optimized Huffman", .optimize_coding = 1},
    {.name = "restart every MCU", .restart_interval = 1},
    {.name = "one scan per component", .sequential_scans = 1},
    {.name = "progressive", .progressive = 1},
    {.name = "progressive with restarts", .progressive = 1, .restart_interval = 3},
    {.name = "arithmetic", .arithmetic = 1},
    {.name = "arithmetic progressive", .progressive = 1, .arithmetic = 1, .restart_interval = 2},
};

static const uint32_t sizes[][2] = {{1, 1}, {8, 8}, {16, 16}, {17, 9}, {33, 47}, {64, 40}, {7, 70}};

static void set_layout(test_jpeg* spec, const layout* l, rng* g) {
  spec->color_space = l->color_space;
  const int32_t components = test_jpeg_components(spec);
  for (int32_t ci = 0; ci < components; ++ci) {
    spec->h_samp_factor[ci] = l->h[ci];
    spec->v_samp_factor[ci] = l->v[ci];
    spec->quant_table_slot[ci] = ci == 0 ? 0 : 1;
  }
  random_table(g, spec->quant_tables[0], 1, 99);
  random_table(g, spec->quant_tables[1], 1, 99);
}

static void test_round_trips(void) {
  note("round trips");
  rng g = {.state = 20260923};
  for (size_t li = 0; li < sizeof layouts / sizeof layouts[0]; ++li) {
    for (size_t mi = 0; mi < sizeof modes / sizeof modes[0]; ++mi) {
      for (size_t si = 0; si < sizeof sizes / sizeof sizes[0]; ++si) {
        encoded e = {};
        e.spec.width = sizes[si][0];
        e.spec.height = sizes[si][1];
        set_layout(&e.spec, &layouts[li], &g);
        e.spec.progressive = modes[mi].progressive;
        e.spec.arithmetic = modes[mi].arithmetic;
        e.spec.optimize_coding = modes[mi].optimize_coding;
        e.spec.restart_interval = modes[mi].restart_interval;
        e.spec.sequential_scans = modes[mi].sequential_scans;
        if (!encode(&e, &g, false)) {
          (void)fprintf(stderr, "  %s, %s, %ux%u\n", layouts[li].name, modes[mi].name, (unsigned)sizes[si][0],
                        (unsigned)sizes[si][1]);
          encoded_free(&e);
          continue;
        }
        const int before = failures;
        unround_jpegio_image image = {};
        char message[256];
        const unround_jpegio_status status =
            unround_jpegio_read(e.data, e.size, nullptr, &image, message, sizeof message);
        if (CHECK(status == UNROUND_JPEGIO_OK)) {
          CHECK(message[0] == '\0');
          check_read_back(&e, &image);
          check_planes(&e, &image);
        } else {
          (void)fprintf(stderr, "  %s\n", message);
        }
        if (failures != before) {
          (void)fprintf(stderr, "  in %s, %s, %ux%u\n", layouts[li].name, modes[mi].name, (unsigned)sizes[si][0],
                        (unsigned)sizes[si][1]);
        }
        unround_jpegio_image_free(&image);
        encoded_free(&e);
      }
    }
  }
}

// Coefficients from the whole range, and tables of every size: 1 to 255 in 8
// bits, and up to 32767 in the 16 bits of an extended file.
static void test_extreme_coefficients(void) {
  note("extreme coefficients and tables");
  rng g = {.state = 7};
  const int32_t table_limits[] = {1, 255, 32767};
  for (size_t ti = 0; ti < sizeof table_limits / sizeof table_limits[0]; ++ti) {
    for (size_t mi = 0; mi < sizeof modes / sizeof modes[0]; ++mi) {
      encoded e = {};
      e.spec.width = 45;
      e.spec.height = 29;
      set_layout(&e.spec, &layouts[4], &g);  // 4:2:0
      random_table(&g, e.spec.quant_tables[0], 1, table_limits[ti]);
      random_table(&g, e.spec.quant_tables[1], 1, table_limits[ti]);
      e.spec.progressive = modes[mi].progressive;
      e.spec.arithmetic = modes[mi].arithmetic;
      e.spec.optimize_coding = modes[mi].optimize_coding;
      e.spec.restart_interval = modes[mi].restart_interval;
      e.spec.sequential_scans = modes[mi].sequential_scans;
      if (encode(&e, &g, true)) {
        unround_jpegio_image image = {};
        char message[256];
        if (CHECK(unround_jpegio_read(e.data, e.size, nullptr, &image, message, sizeof message) == UNROUND_JPEGIO_OK)) {
          check_read_back(&e, &image);
        } else {
          (void)fprintf(stderr, "  %s\n", message);
        }
        unround_jpegio_image_free(&image);
      }
      encoded_free(&e);
    }
  }
}

// A small 4:2:0 file with a picture in it, for the tests below.
static bool small_file(encoded* e, rng* g) {
  *e = (encoded){};
  e->spec.width = 40;
  e->spec.height = 30;
  set_layout(&e->spec, &layouts[4], g);
  return encode(e, g, false);
}

// ---------------------------------------------------------------------------
// Markers

static void put16(uint8_t* p, uint32_t value, bool little) {
  p[little ? 0 : 1] = (uint8_t)(value & 0xFFu);
  p[little ? 1 : 0] = (uint8_t)((value >> 8) & 0xFFu);
}

static void put32(uint8_t* p, uint32_t value, bool little) {
  for (int i = 0; i < 4; ++i) p[little ? i : 3 - i] = (uint8_t)((value >> (8 * i)) & 0xFFu);
}

// An EXIF payload with an IFD of two entries: an image width, then the
// orientation, with the type and count given.
static size_t exif_payload(uint8_t* out, bool little, uint32_t orientation, uint32_t type, uint32_t count) {
  static const uint8_t signature[6] = {'E', 'x', 'i', 'f', 0, 0};
  memcpy(out, signature, sizeof signature);
  uint8_t* tiff = out + 6;
  tiff[0] = little ? 'I' : 'M';
  tiff[1] = tiff[0];
  put16(tiff + 2, 42, little);
  put32(tiff + 4, 8, little);
  put16(tiff + 8, 2, little);
  uint8_t* entry = tiff + 10;
  put16(entry, 0x0100, little);
  put16(entry + 2, 3, little);
  put32(entry + 4, 1, little);
  put32(entry + 8, 0, little);
  put16(entry + 8, 640, little);
  entry += 12;
  put16(entry, 0x0112, little);
  put16(entry + 2, type, little);
  put32(entry + 4, count, little);
  put32(entry + 8, 0, little);
  put16(entry + 8, orientation, little);
  put32(entry + 12, 0, little);  // no next IFD
  return 6 + 10 + 2 * 12 + 4;
}

static int32_t orientation_of(const uint8_t* app1, uint32_t app1_size) {
  rng g = {.state = 99};
  encoded e = {};
  e.spec.width = 16;
  e.spec.height = 16;
  set_layout(&e.spec, &layouts[0], &g);
  e.spec.app1 = app1;
  e.spec.app1_size = app1_size;
  int32_t orientation = -1;
  if (encode(&e, &g, false)) {
    unround_jpegio_image image = {};
    char message[256];
    if (CHECK(unround_jpegio_read(e.data, e.size, nullptr, &image, message, sizeof message) == UNROUND_JPEGIO_OK)) {
      orientation = image.exif_orientation;
    }
    unround_jpegio_image_free(&image);
  }
  encoded_free(&e);
  return orientation;
}

static void test_exif(void) {
  note("EXIF orientation");
  uint8_t payload[64];
  for (uint32_t o = 1; o <= 8; ++o) {
    CHECK(orientation_of(payload, (uint32_t)exif_payload(payload, true, o, 3, 1)) == (int32_t)o);
    CHECK(orientation_of(payload, (uint32_t)exif_payload(payload, false, o, 3, 1)) == (int32_t)o);
  }
  CHECK(orientation_of(payload, (uint32_t)exif_payload(payload, true, 0, 3, 1)) == 0);
  CHECK(orientation_of(payload, (uint32_t)exif_payload(payload, true, 9, 3, 1)) == 0);
  CHECK(orientation_of(payload, (uint32_t)exif_payload(payload, true, 6, 4, 1)) == 0);   // a LONG
  CHECK(orientation_of(payload, (uint32_t)exif_payload(payload, false, 6, 3, 2)) == 0);  // two values

  // Cut short: the IFD says it has two entries, and the orientation is missing.
  const size_t full = exif_payload(payload, true, 6, 3, 1);
  CHECK(orientation_of(payload, (uint32_t)(full - 16)) == 0);
  CHECK(orientation_of(payload, (uint32_t)(full - 5)) == 0);
  CHECK(orientation_of(payload, (uint32_t)(full - 4)) == 6);  // only the next-IFD offset is missing

  // An IFD offset past the end, and one that leaves no room for the count.
  (void)exif_payload(payload, false, 6, 3, 1);
  put32(payload + 6 + 4, 1000, false);
  CHECK(orientation_of(payload, (uint32_t)full) == 0);
  put32(payload + 6 + 4, (uint32_t)(full - 6 - 1), false);
  CHECK(orientation_of(payload, (uint32_t)full) == 0);

  // A huge entry count, bounded by the marker.
  (void)exif_payload(payload, true, 6, 3, 1);
  put16(payload + 6 + 8, 0xFFFF, true);
  CHECK(orientation_of(payload, (uint32_t)full) == 6);
  (void)exif_payload(payload, true, 6, 3, 1);
  put16(payload + 6 + 10 + 12, 0x0113, true);  // the orientation's tag renamed: nothing to find
  put16(payload + 6 + 8, 0xFFFF, true);
  CHECK(orientation_of(payload, (uint32_t)full) == 0);

  // Not EXIF, and not TIFF.
  memcpy(payload, "XMP\0\0\0ABCDEFGH", 14);
  CHECK(orientation_of(payload, 14) == 0);
  (void)exif_payload(payload, true, 6, 3, 1);
  payload[6] = 'X';
  CHECK(orientation_of(payload, (uint32_t)full) == 0);
  (void)exif_payload(payload, true, 6, 3, 1);
  put16(payload + 6 + 2, 43, true);
  CHECK(orientation_of(payload, (uint32_t)full) == 0);
}

static void test_icc(void) {
  note("ICC profiles");
  rng g = {.state = 11};
  const uint32_t lengths[] = {1, 100, 65519, 65520, 140000};  // one marker holds at most 65519 bytes of it
  for (size_t li = 0; li < sizeof lengths / sizeof lengths[0]; ++li) {
    uint8_t* profile = (uint8_t*)malloc(lengths[li]);
    if (profile == nullptr) abort();
    for (uint32_t i = 0; i < lengths[li]; ++i) profile[i] = (uint8_t)rng_range(&g, 0, 255);
    encoded e = {};
    e.spec.width = 24;
    e.spec.height = 8;
    set_layout(&e.spec, &layouts[2], &g);
    e.spec.icc_profile = profile;
    e.spec.icc_profile_size = lengths[li];
    if (encode(&e, &g, false)) {
      unround_jpegio_image image = {};
      char message[256];
      if (CHECK(unround_jpegio_read(e.data, e.size, nullptr, &image, message, sizeof message) == UNROUND_JPEGIO_OK)) {
        CHECK(image.icc_profile_size == lengths[li]);
        CHECK(image.icc_profile != nullptr && memcmp(image.icc_profile, profile, lengths[li]) == 0);
      }
      unround_jpegio_image_free(&image);
      CHECK(image.icc_profile == nullptr && image.icc_profile_size == 0);
    }
    encoded_free(&e);
    free(profile);
  }
}

// ---------------------------------------------------------------------------
// What the layer refuses

static void expect_status(const uint8_t* data, size_t size, const unround_jpegio_options* options,
                          unround_jpegio_status expected, const char* part_of_message) {
  unround_jpegio_image image = {};
  char message[256];
  const unround_jpegio_status status = unround_jpegio_read(data, size, options, &image, message, sizeof message);
  CHECK(status == expected);
  if (part_of_message != nullptr && !CHECK(strstr(message, part_of_message) != nullptr)) {
    (void)fprintf(stderr, "  the message was: %s\n", message);
  }
  if (status != UNROUND_JPEGIO_OK) {
    CHECK(image.num_components == 0 && image.components[0].coefficients == nullptr && image.icc_profile == nullptr);
  }
  unround_jpegio_image_free(&image);

  unround_jpegio_planes planes = {};
  const unround_jpegio_status planes_status =
      unround_jpegio_decode_planes(data, size, options, &planes, message, sizeof message);
  CHECK(planes_status == expected);
  if (planes_status != UNROUND_JPEGIO_OK) CHECK(planes.num_components == 0 && planes.planes[0].samples == nullptr);
  unround_jpegio_planes_free(&planes);
}

static void test_refusals(void) {
  note("refusals");
  rng g = {.state = 5};

  encoded twelve = {};
  twelve.spec.width = 16;
  twelve.spec.height = 16;
  set_layout(&twelve.spec, &layouts[0], &g);
  twelve.spec.data_precision = 12;
  if (encode(&twelve, &g, false))
    expect_status(twelve.data, twelve.size, nullptr, UNROUND_JPEGIO_ERROR_UNSUPPORTED, "12-bit");
  encoded_free(&twelve);

  encoded cmyk = {};
  cmyk.spec.width = 16;
  cmyk.spec.height = 16;
  cmyk.spec.color_space = TEST_JPEG_CMYK;
  random_table(&g, cmyk.spec.quant_tables[0], 1, 99);
  if (encode(&cmyk, &g, false))
    expect_status(cmyk.data, cmyk.size, nullptr, UNROUND_JPEGIO_ERROR_UNSUPPORTED, "4 components");
  encoded_free(&cmyk);

  uint8_t samples[20 * 12];
  for (size_t i = 0; i < sizeof samples; ++i) samples[i] = (uint8_t)rng_range(&g, 0, 255);
  uint8_t* lossless = nullptr;
  size_t lossless_size = 0;
  char message[256];
  if (CHECK(test_jpeg_write_lossless(20, 12, samples, &lossless, &lossless_size, message, sizeof message) == 0)) {
    expect_status(lossless, lossless_size, nullptr, UNROUND_JPEGIO_ERROR_UNSUPPORTED, "lossless");
  }
  free(lossless);
}

static void test_limits(void) {
  note("limits");
  rng g = {.state = 3};
  encoded e = {};
  if (small_file(&e, &g)) {
    unround_jpegio_options options = {.max_pixels = (uint64_t)40 * 30 - 1};
    expect_status(e.data, e.size, &options, UNROUND_JPEGIO_ERROR_LIMIT, "40 x 30");
    options.max_pixels = (uint64_t)40 * 30;
    expect_status(e.data, e.size, &options, UNROUND_JPEGIO_OK, nullptr);
  }
  encoded_free(&e);

  // libjpeg's simple progression codes three components in 10 scans.
  encoded progressive = {};
  progressive.spec.width = 40;
  progressive.spec.height = 30;
  set_layout(&progressive.spec, &layouts[4], &g);
  progressive.spec.progressive = 1;
  if (encode(&progressive, &g, false)) {
    unround_jpegio_options options = {.max_scans = 9};
    expect_status(progressive.data, progressive.size, &options, UNROUND_JPEGIO_ERROR_LIMIT, "more than 9 scans");
    options.max_scans = 10;
    expect_status(progressive.data, progressive.size, &options, UNROUND_JPEGIO_OK, nullptr);
  }
  encoded_free(&progressive);
}

// ---------------------------------------------------------------------------
// Damaged files

// The offset of the n-th (from 0) marker of a kind, found by walking the
// segments and the entropy-coded data between them; or the size when there is
// no such marker.
static size_t find_marker(const uint8_t* data, size_t size, uint8_t marker, int n) {
  size_t i = 2;  // after SOI
  int seen = 0;
  while (i + 4 <= size) {
    if (data[i] != 0xFF) return size;
    const uint8_t kind = data[i + 1];
    if (kind == marker) {
      if (seen == n) return i;
      seen += 1;
    }
    if (kind == 0xD9) return size;  // EOI
    const size_t length = ((size_t)data[i + 2] << 8) | (size_t)data[i + 3];
    i += 2 + length;
    if (kind == 0xDA) {  // entropy-coded data, up to the next marker that is not a restart
      while (i + 1 < size && !(data[i] == 0xFF && data[i + 1] != 0x00 && (data[i + 1] < 0xD0 || data[i + 1] > 0xD7))) {
        ++i;
      }
    }
  }
  return size;
}

// The offset of the first byte of entropy-coded data.
static size_t data_start(const uint8_t* data, size_t size) {
  const size_t sos = find_marker(data, size, 0xDA, 0);
  if (sos + 4 > size) return size;
  return sos + 2 + (((size_t)data[sos + 2] << 8) | (size_t)data[sos + 3]);
}

static void test_damaged(void) {
  note("damaged files");
  rng g = {.state = 13};

  static const uint8_t text[] = "this is not a JPEG file";
  expect_status(text, sizeof text - 1, nullptr, UNROUND_JPEGIO_ERROR_DECODE, "Not a JPEG file");
  expect_status(text, 0, nullptr, UNROUND_JPEGIO_ERROR_DECODE, "Empty");
  expect_status(nullptr, 0, nullptr, UNROUND_JPEGIO_ERROR_DECODE, nullptr);

  encoded e = {};
  if (small_file(&e, &g)) {
    // Cut in the middle of the entropy-coded data: libjpeg warns and fills in
    // what is missing, and the read succeeds unless warnings are errors.
    const size_t start = data_start(e.data, e.size);
    CHECK(start < e.size);
    const size_t cut = start + (e.size - start) / 2;
    unround_jpegio_image image = {};
    char message[256];
    if (CHECK(unround_jpegio_read(e.data, cut, nullptr, &image, message, sizeof message) == UNROUND_JPEGIO_OK)) {
      CHECK(image.warnings > 0);
      CHECK(strstr(message, "Premature end") != nullptr);
    }
    unround_jpegio_image_free(&image);
    unround_jpegio_planes planes = {};
    if (CHECK(unround_jpegio_decode_planes(e.data, cut, nullptr, &planes, message, sizeof message) ==
              UNROUND_JPEGIO_OK)) {
      CHECK(strstr(message, "Premature end") != nullptr);
    }
    unround_jpegio_planes_free(&planes);
    const unround_jpegio_options strict = {.warnings_are_errors = 1};
    expect_status(e.data, cut, &strict, UNROUND_JPEGIO_ERROR_DECODE, "Premature end");

    // Cut in the header.
    expect_status(e.data, 20, nullptr, UNROUND_JPEGIO_ERROR_DECODE, nullptr);
  }
  encoded_free(&e);

  // One scan per component, cut before the second: the chroma components have
  // no data at all.
  encoded scans = {};
  scans.spec.width = 40;
  scans.spec.height = 30;
  set_layout(&scans.spec, &layouts[4], &g);
  scans.spec.sequential_scans = 1;
  if (encode(&scans, &g, false)) {
    const size_t second = find_marker(scans.data, scans.size, 0xDA, 1);
    if (CHECK(second < scans.size)) {
      unround_jpegio_image image = {};
      char message[256];
      CHECK(unround_jpegio_read(scans.data, second, nullptr, &image, message, sizeof message) ==
            UNROUND_JPEGIO_ERROR_DECODE);
      CHECK(strstr(message, "before the data of every component") != nullptr);
      CHECK(image.num_components == 0);
      unround_jpegio_image_free(&image);
    }
  }
  encoded_free(&scans);
}

// ---------------------------------------------------------------------------
// Arguments

static void test_arguments(void) {
  note("arguments");
  rng g = {.state = 17};
  unround_jpegio_image image = {};
  unround_jpegio_planes planes = {};
  char message[256];
  static const uint8_t byte = 0xFF;

  CHECK(unround_jpegio_read(&byte, 1, nullptr, nullptr, message, sizeof message) == UNROUND_JPEGIO_ERROR_ARGUMENT);
  CHECK(unround_jpegio_read(&byte, 1, nullptr, &image, nullptr, 1) == UNROUND_JPEGIO_ERROR_ARGUMENT);
  CHECK(unround_jpegio_read(nullptr, 1, nullptr, &image, message, sizeof message) == UNROUND_JPEGIO_ERROR_ARGUMENT);
  CHECK(strstr(message, "not addressable") != nullptr);
  CHECK(unround_jpegio_decode_planes(&byte, 1, nullptr, nullptr, message, sizeof message) ==
        UNROUND_JPEGIO_ERROR_ARGUMENT);
  CHECK(unround_jpegio_decode_planes(nullptr, 1, nullptr, &planes, message, sizeof message) ==
        UNROUND_JPEGIO_ERROR_ARGUMENT);

  // No room for a message, or little.
  CHECK(unround_jpegio_read(&byte, 1, nullptr, &image, nullptr, 0) == UNROUND_JPEGIO_ERROR_DECODE);
  char small[8];
  memset(small, 'x', sizeof small);
  CHECK(unround_jpegio_read(&byte, 1, nullptr, &image, small, sizeof small) == UNROUND_JPEGIO_ERROR_DECODE);
  CHECK(strlen(small) == sizeof small - 1);

  // Freeing nothing, or twice.
  unround_jpegio_image_free(nullptr);
  unround_jpegio_planes_free(nullptr);
  unround_jpegio_image_free(&image);
  unround_jpegio_planes_free(&planes);
  encoded e = {};
  if (small_file(&e, &g)) {
    CHECK(unround_jpegio_read(e.data, e.size, nullptr, &image, message, sizeof message) == UNROUND_JPEGIO_OK);
    unround_jpegio_image_free(&image);
    unround_jpegio_image_free(&image);
    CHECK(image.num_components == 0);
    // An image that is read into again is emptied first: what it held is the
    // caller's to free before.
    CHECK(unround_jpegio_read(e.data, e.size, nullptr, &image, message, sizeof message) == UNROUND_JPEGIO_OK);
    unround_jpegio_image_free(&image);
  }
  encoded_free(&e);

  CHECK(unround_jpegio_abi_version() == UNROUND_JPEGIO_ABI_VERSION);
  CHECK(strncmp(unround_jpegio_libjpeg_version(), "libjpeg-turbo 3.2.0 (libjpeg API 62)", 64) == 0);
}

int main(void) {
  init_cosines();
  test_round_trips();
  test_extreme_coefficients();
  test_exif();
  test_icc();
  test_refusals();
  test_limits();
  test_damaged();
  test_arguments();
  (void)fprintf(stderr, "%d of %d checks failed\n", failures, checks);
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
