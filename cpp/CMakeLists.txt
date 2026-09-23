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
endif()
