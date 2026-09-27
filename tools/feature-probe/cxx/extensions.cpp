// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Feature probes: what Clang and its runtimes offer beyond the standard, for
// SIMD, dispatch on the instruction set, half precision and threads. Clang is
// the one compiler of JPEG-Unround on every system, so these are candidates
// wherever the standard has nothing portable yet. Each probe is compiled,
// linked and run on its own by run.py, and passes when it exits with status 0.
// The probes follow the project's rules for C++: an empty parameter list is
// written (void), and no conversion is left implicit.

//=== probe: clang_vector_builtins
//--- title: Clang ext_vector_type with __builtin_elementwise_* (portable SIMD)
//--- paper: Clang extension
typedef float f32x8 __attribute__((ext_vector_type(8)));
int main(void) {
  float in[8];
  float out[8];
  for (int i = 0; i < 8; ++i) in[i] = static_cast<float>(i) - 4.0f;
  f32x8 v;
  __builtin_memcpy(&v, in, sizeof v);
  const f32x8 lo = -1.0f;
  const f32x8 hi = 2.0f;
  f32x8 clipped = __builtin_elementwise_min(__builtin_elementwise_max(v, lo), hi);
  f32x8 fused = __builtin_elementwise_fma(v, v, clipped);
  f32x8 root = __builtin_elementwise_sqrt(__builtin_elementwise_abs(v));
  f32x8 rounded = __builtin_elementwise_roundeven(v * 0.5f);
  f32x8 floored = __builtin_elementwise_floor(v * 0.5f);
  __builtin_memcpy(out, &fused, sizeof out);
  float s = 0.0f;
  for (int i = 0; i < 8; ++i) s += clipped[i];
  float m = __builtin_reduce_max(clipped);
  // clipped: -1 -1 -1 -1 0 1 2 2 (sum 1); fused[0] = 16 - 1; root[0] = 2; roundeven(-1.5) = -2
  return s == 1.0f && m == 2.0f && out[0] == 15.0f && root[0] == 2.0f && rounded[1] == -2.0f &&
                 floored[1] == -2.0f
             ? 0
             : 1;
}

//=== probe: clang_masked_load_store
//--- title: __builtin_masked_load / __builtin_masked_store (vector tails)
//--- paper: Clang extension
typedef float f32x8 __attribute__((ext_vector_type(8)));
typedef bool m8 __attribute__((ext_vector_type(8)));
int main(void) {
  float in[5] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  float out[5] = {};
  const m8 mask = {true, true, true, true, true, false, false, false};
  f32x8 v = __builtin_masked_load(mask, &in[0]);
  __builtin_masked_store(mask, v * 2.0f, &out[0]);
  return out[4] == 10.0f ? 0 : 1;
}

//=== probe: target_clones
//--- title: function multiversioning with target_clones (x86-64)
//--- paper: Clang extension
//--- only: x86_64
//--- libs:  | -lclang_rt.builtins-x86_64
__attribute__((target_clones("arch=x86-64-v3", "default"))) static float sum(const float* p, int n) {
  float s = 0.0f;
  for (int i = 0; i < n; ++i) s += p[i];
  return s;
}
int main(void) {
  float d[64];
  for (int i = 0; i < 64; ++i) d[i] = 1.0f;
  return sum(d, 64) == 64.0f ? 0 : 1;
}

//=== probe: target_attribute_dispatch
//--- title: target("avx2,fma") functions chosen with __builtin_cpu_supports (x86-64)
//--- paper: Clang extension
//--- only: x86_64
//--- libs:  | -lclang_rt.builtins-x86_64
__attribute__((target("avx2,fma"))) static float dot_avx2(const float* a, const float* b, int n) {
  float s = 0.0f;
  for (int i = 0; i < n; ++i) s += a[i] * b[i];
  return s;
}
static float dot_base(const float* a, const float* b, int n) {
  float s = 0.0f;
  for (int i = 0; i < n; ++i) s += a[i] * b[i];
  return s;
}
int main(void) {
  __builtin_cpu_init();
  const bool fast = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
  float a[32];
  float b[32];
  for (int i = 0; i < 32; ++i) a[i] = b[i] = 1.0f;
  const float r = fast ? dot_avx2(a, b, 32) : dot_base(a, b, 32);
  return r == 32.0f ? 0 : 1;
}

//=== probe: aarch64_fmv
//--- title: function multiversioning with target_version (AArch64)
//--- paper: Clang extension (ACLE FMV)
//--- only: arm64
__attribute__((target_version("sve2"))) static int kind(void) { return 2; }
__attribute__((target_version("default"))) static int kind(void) { return 1; }
int main(void) { return kind() >= 1 ? 0 : 1; }

//=== probe: float16_extension
//--- title: _Float16 arithmetic (Clang)
//--- paper: Clang extension (ISO/IEC TS 18661-3 type)
int main(void) {
  const _Float16 a = static_cast<_Float16>(1.5f);
  const _Float16 b = static_cast<_Float16>(a * static_cast<_Float16>(2.0f));
  return static_cast<float>(b) == 3.0f ? 0 : 1;
}

//=== probe: bfloat16_extension
//--- title: __bf16 arithmetic (Clang)
//--- paper: Clang extension
int main(void) {
  const __bf16 a = static_cast<__bf16>(1.5f);
  const __bf16 b = static_cast<__bf16>(a * static_cast<__bf16>(2.0f));
  return static_cast<float>(b) == 3.0f ? 0 : 1;
}

//=== probe: openmp
//--- title: OpenMP parallel regions (-fopenmp); prints the threads it ran
//--- paper: OpenMP 5.x
//--- flags: -fopenmp
//--- inspect: imports
#include <cstdio>
#include <omp.h>
int main(void) {
  int n = 0;
#pragma omp parallel reduction(+ : n)
  n += 1;
  std::printf("threads=%d max=%d\n", n, omp_get_max_threads());
  return n >= 1 ? 0 : 1;
}

//=== probe: openmp_simd
//--- title: #pragma omp simd without the OpenMP runtime (-fopenmp-simd)
//--- paper: OpenMP 5.x
//--- flags: -fopenmp-simd
//--- inspect: imports
int main(void) {
  float a[256];
  float b[256];
  for (int i = 0; i < 256; ++i) a[i] = static_cast<float>(i);
  float s = 0.0f;
#pragma omp simd reduction(+ : s)
  for (int i = 0; i < 256; ++i) {
    b[i] = a[i] * 2.0f;
    s += b[i];
  }
  return s == 65280.0f ? 0 : 1;
}

//=== probe: int128_arithmetic
//--- title: __int128 and unsigned __int128: sums, products, shifts and comparisons
//--- paper: Clang extension
int main(void) {
  // volatile keeps the compiler from folding the arithmetic away.
  volatile unsigned long long a = 0xFFFFFFFFFFFFFFFFull;
  volatile unsigned long long b = 0x0123456789ABCDEFull;
  const unsigned __int128 wide = static_cast<unsigned __int128>(a) * static_cast<unsigned __int128>(b);
  const unsigned long long high = static_cast<unsigned long long>(wide >> 64);
  const unsigned long long low = static_cast<unsigned long long>(wide);
  // (2^64 - 1) b = b 2^64 - b: the high word is b - 1 and the low word 2^64 - b.
  const bool product = high == b - 1ull && low == 0ull - b;
  const __int128 negative = -static_cast<__int128>(wide);
  const __int128 sum = negative + static_cast<__int128>(wide);
  const unsigned __int128 square = wide * wide;  // modulo 2^128
  // b lies within [2^56, 2^57), and so the product within (2^119, 2^121).
  const unsigned __int128 one = 1u;
  const bool order = negative < 0 && sum == 0 && (square >> 127) <= 1u && (one << 119) < wide && wide < (one << 121);
  return product && order ? 0 : 1;
}

//=== probe: int128_division_conversion
//--- title: __int128 division and remainder, and conversion from and to double (calls into the compiler's runtime on some targets)
//--- paper: Clang extension
//--- libs:  | -lclang_rt.builtins-x86_64
int main(void) {
  volatile unsigned long long a = 0xFFFFFFFFFFFFFFFFull;
  volatile unsigned long long b = 0x0123456789ABCDEFull;
  const unsigned __int128 wide = static_cast<unsigned __int128>(a) * static_cast<unsigned __int128>(b);
  const unsigned __int128 quotient = wide / static_cast<unsigned __int128>(b);
  const unsigned __int128 remainder = wide % static_cast<unsigned __int128>(a);
  const __int128 signed_quotient = -static_cast<__int128>(wide) / static_cast<__int128>(b);
  volatile double big = 1.0e30;
  const __int128 truncated = static_cast<__int128>(big);
  const double back = static_cast<double>(truncated);
  const double unsigned_back = static_cast<double>(wide);
  return quotient == a && remainder == 0u && signed_quotient == -static_cast<__int128>(a) && back == 1.0e30 &&
                 unsigned_back > 1.0e36 && unsigned_back < 2.0e36
             ? 0
             : 1;
}

//=== probe: int128_in_library_templates
//--- title: __int128 in std::optional, std::array and std::pair keeps its alignment of 16
//--- paper: Clang extension
#include <array>
#include <optional>
#include <utility>
// A library that defines its templates under #pragma pack(8) lays out a member of 16
// bytes' alignment at 8, where aligned loads of it fault.
static_assert(alignof(std::optional<unsigned __int128>) >= alignof(unsigned __int128));
static_assert(alignof(std::array<unsigned __int128, 2>) >= alignof(unsigned __int128));
static_assert(alignof(std::pair<unsigned __int128, int>) >= alignof(unsigned __int128));
int main(void) {
  const std::optional<unsigned __int128> value = static_cast<unsigned __int128>(3u);
  return *value == 3u ? 0 : 1;
}

//=== probe: int128_numeric_limits
//--- title: std::numeric_limits specialized for __int128 (its min() the least value, not 0)
//--- paper: Clang extension
#include <limits>
static_assert(std::numeric_limits<__int128>::is_specialized);
int main(void) { return std::numeric_limits<__int128>::min() < 0 ? 0 : 1; }

//=== probe: cxx_runtime_linkage
//--- title: the C++ library a program links (should be the system's, as a shared library)
//--- paper: -
//--- inspect: imports
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
int main(void) {
  try {
    throw std::runtime_error(std::string("x"));
  } catch (const std::exception& e) {
    // The stream and the thread count live in the shared C++ library itself
    // (msvcp140.dll, libstdc++.so.6, libc++.1.dylib), not only in its headers.
    std::cout << e.what() << ' ' << (std::thread::hardware_concurrency() > 0u) << '\n';
  }
  return 0;
}
