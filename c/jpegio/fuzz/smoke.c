// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Runs the fuzz target of the C layer without libFuzzer, on every system:
// given files, on each of them (to replay what a fuzzer found); given none, on
// files of its own in several modes, with a well-formed ICC profile or with
// bytes that are none, and on well-formed profiles alone, and on thousands of
// mutations of them -- flipped bits, changed and inserted bytes, cuts and
// repeated stretches -- drawn from a fixed seed, so that every run tries the
// same inputs. It is a test in every build, and under the sanitizers in theirs.
//
// smoke --seeds DIRECTORY writes its own files and profiles there instead, as
// the corpus a fuzzing run starts from (ci/fuzz.sh).

#include "fuzz_jpegio.h"
#include "test_jpeg.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

// A number below bound, which is not 0.
static size_t rng_below(rng* g, size_t bound) { return (size_t)(rng_next(g) % (uint64_t)bound); }

static int replay(int count, char** paths) {
  for (int i = 0; i < count; ++i) {
    FILE* file = fopen(paths[i], "rb");
    if (file == nullptr) {
      (void)fprintf(stderr, "cannot open %s\n", paths[i]);
      return EXIT_FAILURE;
    }
    size_t capacity = (size_t)1 << 16;
    size_t size = 0;
    uint8_t* data = (uint8_t*)malloc(capacity);
    while (data != nullptr) {
      size += fread(data + size, 1, capacity - size, file);
      if (size < capacity) break;
      capacity *= 2;
      uint8_t* grown = (uint8_t*)realloc(data, capacity);
      if (grown == nullptr) free(data);
      data = grown;
    }
    const bool failed = ferror(file) != 0;
    (void)fclose(file);
    if (data == nullptr || failed) {
      (void)fprintf(stderr, "cannot read %s\n", paths[i]);
      free(data);
      return EXIT_FAILURE;
    }
    (void)LLVMFuzzerTestOneInput(data, size);
    free(data);
  }
  (void)fprintf(stderr, "%d inputs replayed\n", count);
  return EXIT_SUCCESS;
}

// Changes the input in place, or makes it shorter or longer within capacity;
// returns the new size.
static size_t mutate(rng* g, uint8_t* data, size_t size, size_t capacity) {
  if (size == 0) return 0;
  switch (rng_below(g, 6)) {
    case 0: {  // flip one to four bits
      const size_t flips = 1 + rng_below(g, 4);
      for (size_t i = 0; i < flips; ++i) data[rng_below(g, size)] ^= (uint8_t)(1u << rng_below(g, 8));
      return size;
    }
    case 1: {  // set a byte to a value markers are made of, or to any
      static const uint8_t values[] = {0x00, 0xFF, 0xD8, 0xD9, 0xDA, 0xC0, 0xC2, 0xC4, 0xDB, 0xDD, 0x01, 0x7F, 0x80};
      const size_t at = rng_below(g, size);
      data[at] = rng_below(g, 2) == 0 ? values[rng_below(g, sizeof values)] : (uint8_t)rng_below(g, 256);
      return size;
    }
    case 2:  // cut
      return rng_below(g, size);
    case 3: {  // remove a stretch
      const size_t at = rng_below(g, size);
      const size_t length = 1 + rng_below(g, size - at);
      memmove(data + at, data + at + length, size - at - length);
      return size - length;
    }
    case 4: {  // repeat a stretch
      const size_t at = rng_below(g, size);
      size_t length = 1 + rng_below(g, size - at < 64 ? size - at : 64);
      if (length > capacity - size) length = capacity - size;
      memmove(data + at + length, data + at, size - at);
      return size + length;
    }
    default: {  // insert a random byte
      if (size == capacity) return size;
      const size_t at = rng_below(g, size + 1);
      memmove(data + at + 1, data + at, size - at);
      data[at] = (uint8_t)rng_below(g, 256);
      return size + 1;
    }
  }
}

static void put32(uint8_t* p, uint32_t value) {
  p[0] = (uint8_t)(value >> 24);
  p[1] = (uint8_t)(value >> 16);
  p[2] = (uint8_t)(value >> 8);
  p[3] = (uint8_t)value;
}

// Four characters of a signature, without the end of the text that gives them.
static void put_signature(uint8_t* p, const char* text) { memcpy(p, text, 4); }

// A well-formed ICC profile of the size (at least 156 bytes), a monitor's of
// version 2 of the data color space given ('GRAY' or 'RGB '), with two tags in
// it, its other bytes random.
static void well_formed_profile(rng* g, uint8_t* p, size_t size, const char* space) {
  for (size_t i = 0; i < size; ++i) p[i] = (uint8_t)rng_below(g, 256);
  memset(p, 0, 132);
  put32(p, (uint32_t)size);
  p[8] = 2;
  put_signature(p + 12, "mntr");
  put_signature(p + 16, space);
  put_signature(p + 20, "XYZ ");
  put_signature(p + 36, "acsp");
  put32(p + 68, UINT32_C(0x0000F6D6));  // the illuminant D50
  put32(p + 72, UINT32_C(0x00010000));
  put32(p + 76, UINT32_C(0x0000D32D));
  put32(p + 128, 2);
  put_signature(p + 132, "desc");
  put32(p + 136, 156);
  put32(p + 140, (uint32_t)(size - 156));
  put_signature(p + 144, "wtpt");
  put32(p + 148, 156);
  put32(p + 152, 0);
}

static bool seed_file(rng* g, int32_t kind, uint8_t** data, size_t* size) {
  test_jpeg spec = {};
  spec.width = 13 + (uint32_t)rng_below(g, 40);
  spec.height = 7 + (uint32_t)rng_below(g, 40);
  spec.color_space = kind % 3 == 0 ? TEST_JPEG_GRAYSCALE : TEST_JPEG_YCBCR;
  if (kind % 3 == 2) {
    spec.h_samp_factor[0] = 2;
    spec.v_samp_factor[0] = 2;
  }
  spec.quant_table_slot[1] = 1;
  spec.quant_table_slot[2] = 1;
  spec.progressive = (kind / 3) % 2;
  spec.arithmetic = (kind / 6) % 2;
  spec.restart_interval = (kind / 12) % 2 == 0 ? 0 : 2;
  static const uint8_t exif[] = {'E',  'x',  'i', 'f', 0, 0, 'I', 'I', 42, 0, 8, 0, 0, 0, 1, 0,
                                 0x12, 0x01, 3,   0,   1, 0, 0,   0,   6,  0, 0, 0, 0, 0, 0, 0};
  // Every other file carries a well-formed profile of its color space; the rest,
  // bytes that are none.
  static const uint8_t not_a_profile[] = "not really a profile, but carried all the same";
  uint8_t icc[300];
  well_formed_profile(g, icc, sizeof icc, spec.color_space == TEST_JPEG_GRAYSCALE ? "GRAY" : "RGB ");
  spec.app1 = exif;
  spec.app1_size = (uint32_t)sizeof exif;
  spec.icc_profile = kind % 2 == 0 ? icc : not_a_profile;
  spec.icc_profile_size = kind % 2 == 0 ? (uint32_t)sizeof icc : (uint32_t)sizeof not_a_profile;
  int16_t* coefficients[3] = {};
  const int32_t components = test_jpeg_components(&spec);
  for (int32_t slot = 0; slot < 2; ++slot) {
    for (int k = 0; k < 64; ++k) spec.quant_tables[slot][k] = (uint16_t)(1 + rng_below(g, 60));
  }
  for (int32_t ci = 0; ci < components; ++ci) {
    const size_t count = (size_t)test_jpeg_blocks_wide(&spec, ci) * test_jpeg_blocks_high(&spec, ci) * 64;
    coefficients[ci] = (int16_t*)calloc(count, sizeof(int16_t));
    if (coefficients[ci] == nullptr) abort();
    for (size_t i = 0; i < count; ++i) {
      if (i % 64 == 0) {
        coefficients[ci][i] = (int16_t)((int32_t)rng_below(g, 401) - 200);
      } else if (i % 64 < 10 && rng_below(g, 3) == 0) {
        coefficients[ci][i] = (int16_t)((int32_t)rng_below(g, 61) - 30);
      }
    }
    spec.coefficients[ci] = coefficients[ci];
  }
  char message[256];
  const int result = test_jpeg_write(&spec, data, size, message, sizeof message);
  for (int32_t ci = 0; ci < components; ++ci) free(coefficients[ci]);
  if (result != 0) (void)fprintf(stderr, "the test encoder failed: %s\n", message);
  return result == 0;
}

static bool write_file(const char* directory, const char* name, int32_t kind, const uint8_t* data, size_t size) {
  char path[4096];
  const int length = snprintf(path, sizeof path, "%s/%s-%02d", directory, name, (int)kind);
  if (length < 0 || (size_t)length >= sizeof path) return false;
  FILE* file = fopen(path, "wb");
  if (file == nullptr) return false;
  const bool written = fwrite(data, 1, size, file) == size;
  return (bool)(fclose(file) == 0 && written);
}

static constexpr int32_t kinds = 24;
static constexpr int32_t profiles = 4;

// Tries the input and mutations of it; returns how many inputs it tried.
static size_t try_mutations(rng* g, const uint8_t* seed, size_t seed_size, int mutations) {
  (void)LLVMFuzzerTestOneInput(seed, seed_size);
  const size_t capacity = seed_size + 256;
  uint8_t* work = (uint8_t*)malloc(capacity);
  if (work == nullptr) abort();
  for (int m = 0; m < mutations; ++m) {
    memcpy(work, seed, seed_size);
    size_t size = seed_size;
    const size_t rounds = 1 + rng_below(g, 3);
    for (size_t r = 0; r < rounds; ++r) size = mutate(g, work, size, capacity);
    // An input of its own, of exactly its size, so that a read past its end
    // is one the address sanitizer sees.
    uint8_t* input = (uint8_t*)malloc(size > 0 ? size : 1);
    if (input == nullptr) abort();
    memcpy(input, work, size);
    (void)LLVMFuzzerTestOneInput(input, size);
    free(input);
  }
  free(work);
  return 1 + (size_t)mutations;
}

int main(int argc, char** argv) {
  const bool seeds_only = (bool)(argc == 3 && strcmp(argv[1], "--seeds") == 0);
  if (argc > 1 && !seeds_only) return replay(argc - 1, argv + 1);

  rng g = {.state = UINT64_C(0x756E726F756E64)};  // "unround"
  const int mutations = 400;
  size_t inputs = 0;
  for (int32_t kind = 0; kind < kinds; ++kind) {
    uint8_t* seed = nullptr;
    size_t seed_size = 0;
    if (!seed_file(&g, kind, &seed, &seed_size)) return EXIT_FAILURE;
    if (seeds_only) {
      const bool written = write_file(argv[2], "seed", kind, seed, seed_size);
      free(seed);
      if (!written) {
        (void)fprintf(stderr, "cannot write a seed into %s\n", argv[2]);
        return EXIT_FAILURE;
      }
      continue;
    }
    inputs += try_mutations(&g, seed, seed_size, mutations);
    free(seed);
  }
  // Profiles alone, gray and RGB, of a size a multiple of 4 and of one that is not.
  for (int32_t kind = 0; kind < profiles; ++kind) {
    uint8_t profile[401];
    const size_t size = kind < 2 ? 400 : 401;
    well_formed_profile(&g, profile, size, kind % 2 == 0 ? "GRAY" : "RGB ");
    if (seeds_only) {
      if (!write_file(argv[2], "profile", kind, profile, size)) {
        (void)fprintf(stderr, "cannot write a seed into %s\n", argv[2]);
        return EXIT_FAILURE;
      }
      continue;
    }
    inputs += try_mutations(&g, profile, size, mutations);
  }
  if (seeds_only) {
    (void)fprintf(stderr, "%d seeds written into %s\n", (int)(kinds + profiles), argv[2]);
  } else {
    (void)fprintf(stderr, "%zu inputs tried\n", inputs);
  }
  return EXIT_SUCCESS;
}
