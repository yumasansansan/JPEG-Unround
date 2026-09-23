# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
#
# The C++ implementation. So far it is the C++ face of the C layer
# (include/unround/jpegio.hpp), with its tests.

find_package(Threads REQUIRED)

add_library(unround_cpp INTERFACE)
add_library(unround::cpp ALIAS unround_cpp)
target_include_directories(unround_cpp INTERFACE include)
target_link_libraries(unround_cpp INTERFACE unround::jpegio Threads::Threads)
target_compile_features(unround_cpp INTERFACE cxx_std_26)

if(UNROUND_BUILD_TESTS)
  add_executable(unround_cpp_jpegio_test tests/test_jpegio.cpp)
  target_link_libraries(unround_cpp_jpegio_test PRIVATE unround::cpp unround_test_jpeg)
  unround_target_defaults(unround_cpp_jpegio_test)
  unround_add_test(cpp.jpegio unround_cpp_jpegio_test)

  # Arguments and file names in UTF-8 (the manifest of CMakeLists.txt). The
  # text is the one that the test expects: kanji, Latin letters beyond ASCII,
  # and a character beyond the Basic Multilingual Plane.
  add_executable(unround_cpp_utf8_arguments_test tests/test_utf8_arguments.cpp)
  unround_target_defaults(unround_cpp_utf8_arguments_test)
  unround_add_test(cpp.utf8-arguments unround_cpp_utf8_arguments_test
    "日本語 ×ü 🙂" "${CMAKE_CURRENT_BINARY_DIR}/utf8-arguments")
endif()
