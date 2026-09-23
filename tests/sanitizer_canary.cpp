// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Makes on purpose the mistake a sanitizer is there to find, so that a
// sanitizer build can show that its sanitizers run: a test that says nothing
// cannot be told from a check that never ran. Each test of this program passes
// only when the sanitizer's report of the mistake is in its output
// (tests/CMakeLists.txt). The mistakes go through volatile objects, which the
// optimizer has to leave as they are written.
//
//   sanitizer_canary address | leak | undefined | thread

#include <climits>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <thread>

namespace {

// A read one past the end of an allocation. The index goes through a volatile,
// so that the compiler cannot see the mistake and leave the read out.
int read_past_the_end(void) {
  int* numbers = new int[4]{1, 2, 3, 4};
  volatile std::size_t index = 4;
  const int value = numbers[index];
  delete[] numbers;
  return value;
}

// Memory that nothing points to when the program ends.
int leak(void) {
  volatile int* lost = new int[16];
  lost[0] = 1;
  lost = nullptr;
  return 0;
}

int overflow(void) {
  volatile int large = INT_MAX;
  const int sum = large + 1;
  return sum == 0 ? 1 : 0;
}

// Two threads write one variable with nothing to order the writes.
int counter = 0;

int race(void) {
  std::thread first{[](void) {
    for (int i = 0; i < 100000; ++i) counter = counter + 1;
  }};
  std::thread second{[](void) {
    for (int i = 0; i < 100000; ++i) counter = counter + 1;
  }};
  first.join();
  second.join();
  return counter == 0 ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fputs("usage: sanitizer_canary address|leak|undefined|thread\n", stderr);
    return 2;
  }
  const char* kind = argv[1];
  if (std::strcmp(kind, "address") == 0) return read_past_the_end();
  if (std::strcmp(kind, "leak") == 0) return leak();
  if (std::strcmp(kind, "undefined") == 0) return overflow();
  if (std::strcmp(kind, "thread") == 0) return race();
  std::fputs("unknown mistake\n", stderr);
  return 2;
}
