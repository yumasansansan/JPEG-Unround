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
