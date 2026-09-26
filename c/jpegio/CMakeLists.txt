# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
#
# The C layer (C23) over libjpeg-turbo and libpng, as a static library for the
# C++ and Rust implementations and as a shared library for the Python one; its
# tests; and its fuzz target.

add_library(unround_jpegio STATIC src/jpegio.c src/png.c)
add_library(unround::jpegio ALIAS unround_jpegio)
target_include_directories(unround_jpegio PUBLIC include)
target_link_libraries(unround_jpegio PRIVATE unround::libjpeg unround::png)
unround_target_defaults(unround_jpegio)

if(UNROUND_WITH_PYTHON)
  # The shared library that the Python implementation loads with ctypes, with
  # libjpeg-turbo, libpng and zlib-ng linked in. It exports the functions of
  # unround/jpegio.h and nothing else (tests/check_exports.cmake checks), and it
  # is written into python/ of the build directory, apart from the static
  # library of the same name. Linux is asked to leave no symbol undefined;
  # Apple's linker and Windows' do so anyway. A sanitizer build cannot: the
  # sanitizer's runtime comes with the program that loads the library.
  add_library(unround_jpegio_shared SHARED src/jpegio.c src/png.c)
  set_target_properties(unround_jpegio_shared PROPERTIES
    OUTPUT_NAME unround_jpegio
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python")
  target_include_directories(unround_jpegio_shared PUBLIC include)
  target_compile_definitions(unround_jpegio_shared PRIVATE UNROUND_JPEGIO_BUILDING_SHARED)
  target_link_libraries(unround_jpegio_shared PRIVATE unround::libjpeg unround::png)
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT UNROUND_SANITIZER_LIST)
    target_link_options(unround_jpegio_shared PRIVATE "LINKER:-z,defs")
  endif()
  unround_target_defaults(unround_jpegio_shared)
endif()

if(UNROUND_BUILD_TESTS OR UNROUND_BUILD_FUZZERS)
  # libpng's reading of PNG files, for the tests of the writing of them here and
  # in what is built on this layer, and for the fuzz target.
  add_library(unround_test_png STATIC tests/support/test_png.c)
  target_include_directories(unround_test_png PUBLIC tests/support)
  target_link_libraries(unround_test_png PRIVATE unround::png)
  unround_target_defaults(unround_test_png)
endif()

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

  add_executable(unround_png_test tests/test_png.c)
  target_link_libraries(unround_png_test PRIVATE unround::jpegio unround_test_png)
  unround_target_defaults(unround_png_test)
  unround_add_test(c.png unround_png_test)

  # The fuzz target, run without libFuzzer on inputs of its own.
  add_executable(unround_jpegio_fuzz_smoke fuzz/smoke.c fuzz/fuzz_jpegio.c)
  target_link_libraries(unround_jpegio_fuzz_smoke PRIVATE unround::jpegio unround_test_jpeg unround_test_png)
  unround_target_defaults(unround_jpegio_fuzz_smoke)
  unround_add_test(c.jpegio.fuzz-smoke unround_jpegio_fuzz_smoke)
endif()

if(UNROUND_BUILD_FUZZERS)
  add_executable(unround_jpegio_fuzz fuzz/fuzz_jpegio.c)
  target_link_options(unround_jpegio_fuzz PRIVATE -fsanitize=fuzzer)
  target_link_libraries(unround_jpegio_fuzz PRIVATE unround::jpegio unround_test_png)
  unround_target_defaults(unround_jpegio_fuzz)
endif()
