// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Cases for ci/void-params.py --self-test: every line that ends in the comment
// "want: void" has to be reported, and no other line. The file compiles as C23 as it is.

#include <stddef.h>

int answer();  // want: void
int answer(void) { return 42; }

static int twice(int value) { return 2 * value; }

void (*handler)();              // want: void
typedef int function_type();    // want: void
typedef int proper_type(void);  // written as it should be

struct callbacks {
  int (*callback)();  // want: void
  int (*proper)(void);
};

#define ZERO() 0
#define CALL(function) function()

#if 0
int inactive();
#endif

static int uses(void) {
  struct callbacks table = {.callback = answer, .proper = answer};
  int sum = answer() + twice(answer()) + table.proper() + ZERO() + CALL(answer);
  sum += (int)sizeof(answer());
  int (*local)() = answer;  // want: void
  return sum + local();
}

int main(void) { return uses() == 0; }
