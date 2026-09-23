// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Feature probes: C23, for the C code of JPEG-Unround (the thin wrapper over
// libjpeg-turbo, where setjmp/longjmp has to stay inside C). The language
// comes from Clang; the headers of the C library are the system's (UCRT,
// glibc, Apple's libc), except those Clang ships itself. Each probe is
// compiled, linked and run on its own by run.py, and passes when it exits with
// status 0. The probes follow the project's rules for C: an empty parameter
// list is written (void), and no conversion is left implicit.

//=== probe: c23_version
//--- title: __STDC_VERSION__ is 202311L
#if __STDC_VERSION__ < 202311L
#error "not C23"
#endif
int main(void) { return 0; }

//=== probe: c23_nullptr
//--- title: nullptr and nullptr_t
#include <stddef.h>
int main(void) {
  int* p = nullptr;
  nullptr_t n = nullptr;
  return p == n ? 0 : 1;
}

//=== probe: c23_constexpr
//--- title: constexpr objects
#include <stddef.h>
constexpr int block = 8;
constexpr double scale = 0.125;
static int table[block * block];
int main(void) {
  return sizeof table / sizeof table[0] == (size_t)64 && scale * (double)block == 1.0 ? 0 : 1;
}

//=== probe: c23_typeof
//--- title: typeof and typeof_unqual
int main(void) {
  const int x = 3;
  typeof_unqual(x) y = x;
  y += 1;
  typeof(x) z = y;
  return z == 4 ? 0 : 1;
}

//=== probe: c23_keywords
//--- title: bool/true/false, static_assert, alignas, alignof, thread_local as keywords
#include <stddef.h>
#include <stdint.h>
static_assert(sizeof(int) >= 2);
static thread_local int counter;
int main(void) {
  bool b = true;
  alignas(32) float v[8] = {};
  counter += b ? 1 : 0;
  return counter == 1 && alignof(max_align_t) >= (size_t)8 && ((uintptr_t)(void*)v % (uintptr_t)32) == (uintptr_t)0 &&
                 !false
             ? 0
             : 1;
}

//=== probe: c23_auto
//--- title: auto type inference
int main(void) {
  auto n = 4;
  auto d = 0.5;
  return (int)((double)n * d) == 2 ? 0 : 1;
}

//=== probe: c23_attributes
//--- title: [[nodiscard("...")]], [[maybe_unused]], [[deprecated]], [[fallthrough]], [[noreturn]]
#include <stdlib.h>
[[nodiscard("check the status")]] static int status(void) { return 0; }
[[noreturn]] static void die(void) { abort(); }
[[deprecated("use status")]] [[maybe_unused]] static int old(void) { return 0; }
int main(int argc, char** argv) {
  [[maybe_unused]] int unused = 0;
  (void)argv;
  switch (argc) {
    case 0:
      die();
    case 1:
      [[fallthrough]];
    default:
      break;
  }
  return status();
}

//=== probe: c23_empty_initializer
//--- title: empty initializer = {}
struct Params {
  int max_iter;
  double tol;
  float weights[4];
};
int main(void) {
  struct Params p = {};
  int n = {};
  return p.max_iter == 0 && p.tol == 0.0 && p.weights[3] == 0.0f && n == 0 ? 0 : 1;
}

//=== probe: c23_literals
//--- title: binary literals and digit separators
int main(void) { return 0b1010 == 10 && 1'000'000 == 1000000 ? 0 : 1; }

//=== probe: c23_enum_underlying
//--- title: enumerations with a fixed underlying type
#include <stddef.h>
enum Kind : unsigned char { luma, chroma };
int main(void) { return sizeof(enum Kind) == (size_t)1 && chroma == (enum Kind)1 ? 0 : 1; }

//=== probe: c23_bitint
//--- title: _BitInt(N)
int main(void) {
  unsigned _BitInt(12) x = 4095uwb;
  x += 1uwb;
  return x == 0uwb ? 0 : 1;
}

//=== probe: c23_embed
//--- title: #embed
#include <stddef.h>
static const unsigned char self[] = {
#embed __FILE__
};
int main(void) { return sizeof self > (size_t)100 && self[0] == (unsigned char)'/' ? 0 : 1; }

//=== probe: c23_stdckdint
//--- title: <stdckdint.h> ckd_add / ckd_mul (checked sizes)
#include <stdckdint.h>
#include <stddef.h>
int main(void) {
  size_t bytes = 0;
  bool overflow = ckd_mul(&bytes, (size_t)1 << 40, (size_t)1 << 40);
  int sum = 0;
  bool fine = !ckd_add(&sum, 2, 3);
  return overflow && fine && sum == 5 ? 0 : 1;
}

//=== probe: c23_stdbit
//--- title: <stdbit.h> bit utilities
#include <stdbit.h>
int main(void) {
  unsigned x = 0x0F0u;
  return stdc_count_ones(x) == 4u && stdc_trailing_zeros(x) == 4u && stdc_has_single_bit(64u) ? 0 : 1;
}

//=== probe: c23_unreachable
//--- title: unreachable() from <stddef.h>
#include <stddef.h>
int main(int argc, char** argv) {
  (void)argv;
  if (argc < 0) unreachable();
  return 0;
}

//=== probe: c23_memset_explicit
//--- title: memset_explicit
#include <string.h>
int main(void) {
  char secret[8] = "abcdefg";
  (void)memset_explicit(secret, 0, sizeof secret);
  return secret[0] == '\0' ? 0 : 1;
}

//=== probe: c23_strdup
//--- title: strdup / strndup in <string.h>
#include <stdlib.h>
#include <string.h>
int main(void) {
  char* a = strdup("abc");
  char* b = strndup("abcdef", 2);
  bool ok = a != nullptr && b != nullptr && strcmp(a, "abc") == 0 && strcmp(b, "ab") == 0;
  free(a);
  free(b);
  return ok ? 0 : 1;
}

//=== probe: c23_va_start
//--- title: va_start with one argument, variadic without a named parameter
#include <stdarg.h>
static int sum(...) {
  va_list ap;
  va_start(ap);
  int a = va_arg(ap, int);
  int b = va_arg(ap, int);
  va_end(ap);
  return a + b;
}
int main(void) { return sum(2, 3) == 5 ? 0 : 1; }

//=== probe: c23_unnamed_parameters
//--- title: unnamed parameters in function definitions
static int ignore(int, int b) { return b; }
int main(void) { return ignore(1, 0); }

//=== probe: c23_labels
//--- title: labels before declarations and at the end of blocks
int main(void) {
  int n = 0;
  goto next;
next:
  int m = 2;
  n += m;
  {
    goto end;
  end:
  }
  return n == 2 ? 0 : 1;
}

//=== probe: c23_elifdef
//--- title: #elifdef / #elifndef
#define PROBE_A
#ifdef PROBE_NOT_DEFINED
#error "wrong branch"
#elifdef PROBE_A
static const bool ok = true;
#else
#error "wrong branch"
#endif
int main(void) { return ok ? 0 : 1; }

//=== probe: c23_free_sized
//--- title: free_sized / free_aligned_sized
#include <stdlib.h>
int main(void) {
  void* p = malloc(16);
  free_sized(p, 16);
  void* q = aligned_alloc(64, 128);
  free_aligned_sized(q, 64, 128);
  return 0;
}

//=== probe: c23_strfromd
//--- title: strfromd / strfromf
#include <stdlib.h>
#include <string.h>
int main(void) {
  char buf[32];
  (void)strfromd(buf, sizeof buf, "%.2f", 0.5);
  return strcmp(buf, "0.50") == 0 ? 0 : 1;
}

//=== probe: c11_threads
//--- title: <threads.h> (C11 threads, optional in C23)
#include <threads.h>
static int work(void* arg) { return *(int*)arg + 1; }
int main(void) {
  thrd_t t;
  int arg = 41;
  int result = 0;
  if (thrd_create(&t, work, &arg) != thrd_success) return 1;
  if (thrd_join(t, &result) != thrd_success) return 1;
  return result == 42 ? 0 : 1;
}

//=== probe: c_setjmp_longjmp
//--- title: setjmp / longjmp (how libjpeg reports errors)
#include <setjmp.h>
static jmp_buf env;
[[noreturn]] static void fail(void) { longjmp(env, 7); }
int main(void) {
  volatile int stage = 0;
  // setjmp stands only where C allows it: here, the whole controlling
  // expression of a switch.
  switch (setjmp(env)) {
    case 0:
      stage = 1;
      fail();
    case 7:
      return stage == 1 ? 0 : 1;
    default:
      return 1;
  }
}
