# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
#
# The C++ implementation: a library for programs in C++, written from
# docs/math.md apart from the reference implementation in Rust, so that each
# checks the other through the conformance cases. Its headers are in
# include/unround/, and so is the C++ face of the C layer (jpegio.hpp); the
# tests read PNG files back with libpng.

find_package(Threads REQUIRED)

add_library(unround_cpp STATIC
  src/dct.cpp
  src/exact.cpp
  src/laplace.cpp
  src/model.cpp
  src/operators.cpp)
add_library(unround::cpp ALIAS unround_cpp)
target_include_directories(unround_cpp PUBLIC include)
target_link_libraries(unround_cpp PUBLIC unround::jpegio Threads::Threads)
target_compile_features(unround_cpp PUBLIC cxx_std_26)
unround_target_defaults(unround_cpp)

# On x86-64 the library computes with AVX2 and FMA (x86-64-v3), as the reference
# implementation does (docs/cli.md, Builds): a product may be fused with the sum
# that takes it (docs/math.md, Arithmetic). The tests are built alike, so that they
# check the code as it runs.
set(UNROUND_CPP_TARGET_FLAGS "")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(AMD64|x86_64)$")
  set(UNROUND_CPP_TARGET_FLAGS -march=x86-64-v3)
endif()
target_compile_options(unround_cpp PRIVATE ${UNROUND_CPP_TARGET_FLAGS})

if(UNROUND_BUILD_TESTS)
  add_executable(unround_cpp_jpegio_test tests/test_jpegio.cpp)
  target_link_libraries(unround_cpp_jpegio_test PRIVATE unround::cpp unround_test_jpeg unround_test_png)
  unround_target_defaults(unround_cpp_jpegio_test)
  unround_add_test(cpp.jpegio unround_cpp_jpegio_test)

  # Arguments and file names in UTF-8 (the manifest of CMakeLists.txt). The
  # text is the one that the test expects: kanji, Latin letters beyond ASCII,
  # and a character beyond the Basic Multilingual Plane.
  add_executable(unround_cpp_utf8_arguments_test tests/test_utf8_arguments.cpp)
  unround_target_defaults(unround_cpp_utf8_arguments_test)
  unround_add_test(cpp.utf8-arguments unround_cpp_utf8_arguments_test
    "日本語 ×ü 🙂" "${CMAKE_CURRENT_BINARY_DIR}/utf8-arguments")

  # The tests of the library; those that compare with the exact references take
  # their directory.
  set(UNROUND_REFERENCES "${PROJECT_SOURCE_DIR}/conformance/references")
  foreach(name IN ITEMS exact dct laplace model operators)
    add_executable(unround_cpp_${name}_test tests/test_${name}.cpp)
    target_include_directories(unround_cpp_${name}_test PRIVATE tests)
    target_link_libraries(unround_cpp_${name}_test PRIVATE unround::cpp)
    unround_target_defaults(unround_cpp_${name}_test)
    target_compile_options(unround_cpp_${name}_test PRIVATE ${UNROUND_CPP_TARGET_FLAGS})
    unround_add_test(cpp.${name} unround_cpp_${name}_test "${UNROUND_REFERENCES}")
  endforeach()
endif()
