# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
#
# The C layer (C23) over libjpeg-turbo, as a static library for the C++ and
# Rust implementations; its tests; and its fuzz target.

add_library(unround_jpegio STATIC src/jpegio.c)
add_library(unround::jpegio ALIAS unround_jpegio)
target_include_directories(unround_jpegio PUBLIC include)
target_link_libraries(unround_jpegio PRIVATE unround::libjpeg)
unround_target_defaults(unround_jpegio)

if(UNROUND_BUILD_TESTS)
  # libjpeg's encoder and its standard decoding, for the tests of this layer
  # and of what is built on it.
  add_library(unround_test_jpeg STATIC tests/support/test_jpeg.c)
  target_include_directories(unround_test_jpeg PUBLIC tests/support)
  target_link_libraries(unround_test_jpeg PRIVATE unround::libjpeg)
  unround_target_defaults(unround_test_jpeg)

  add_executable(unround_jpegio_test tests/test_jpegio.c)
  target_link_libraries(unround_jpegio_test PRIVATE unround::jpegio unround_test_jpeg $<$<PLATFORM_ID:Linux>:m>)
  unround_target_defaults(unround_jpegio_test)
  unround_add_test(c.jpegio unround_jpegio_test)

  # The fuzz target, run without libFuzzer on inputs of its own.
  add_executable(unround_jpegio_fuzz_smoke fuzz/smoke.c fuzz/fuzz_jpegio.c)
  target_link_libraries(unround_jpegio_fuzz_smoke PRIVATE unround::jpegio unround_test_jpeg)
  unround_target_defaults(unround_jpegio_fuzz_smoke)
  unround_add_test(c.jpegio.fuzz-smoke unround_jpegio_fuzz_smoke)
endif()

if(UNROUND_BUILD_FUZZERS)
  add_executable(unround_jpegio_fuzz fuzz/fuzz_jpegio.c)
  target_link_options(unround_jpegio_fuzz PRIVATE -fsanitize=fuzzer)
  target_link_libraries(unround_jpegio_fuzz PRIVATE unround::jpegio)
  unround_target_defaults(unround_jpegio_fuzz)
endif()
