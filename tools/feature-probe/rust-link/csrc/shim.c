// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The shape of the C layer between libjpeg and the implementations: an error
// deep inside is reported with longjmp, as libjpeg reports its errors, and the
// jump never leaves C, since it may not cross the frames of Rust or C++. The
// caller sees a status code.

#include <setjmp.h>
#include <stdckdint.h>
#include <stdint.h>

static jmp_buf probe_env;

[[noreturn]] static void probe_fail(void) { longjmp(probe_env, 1); }

static void probe_add_or_fail(int32_t a, int32_t b, int32_t* out) {
  if (ckd_add(out, a, b)) probe_fail();
}

[[nodiscard]] int32_t unround_probe_checked_add(int32_t a, int32_t b, int32_t* out);

[[nodiscard]] int32_t unround_probe_checked_add(int32_t a, int32_t b, int32_t* out) {
  // setjmp stands only where C allows it: the whole controlling expression.
  switch (setjmp(probe_env)) {
    case 0:
      probe_add_or_fail(a, b, out);
      return 0;
    default:
      return -1;
  }
}
