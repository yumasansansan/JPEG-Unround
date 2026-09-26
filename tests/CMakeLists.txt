# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Checks of the build itself.
#
# In a sanitizer build, one test for each of its sanitizers makes the mistake
# that sanitizer finds and passes only when the sanitizer's report is in the
# output, so that the silence of the other tests means something. Leaks are
# looked for where LeakSanitizer runs, which is on Linux (CMakeLists.txt).
#
# Where the shared C layer is built, a test checks that it exports its API and
# nothing else (check_exports.cmake).
#
# The header of the C interface of the Rust implementation
# (rust/capi/include/unround.h) is compiled as C23 and as C++, so that its
# declarations and the layout of its structures are checked by the compiler.

if(UNROUND_SANITIZER_LIST)
  find_package(Threads REQUIRED)
  add_executable(unround_sanitizer_canary sanitizer_canary.cpp)
  target_link_libraries(unround_sanitizer_canary PRIVATE Threads::Threads)
  unround_target_defaults(unround_sanitizer_canary)

  set(canaries "")
  if("address" IN_LIST UNROUND_SANITIZER_LIST)
    list(APPEND canaries "address=AddressSanitizer: heap-buffer-overflow")
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
      list(APPEND canaries "leak=LeakSanitizer: detected memory leaks")
    endif()
  endif()
  if("undefined" IN_LIST UNROUND_SANITIZER_LIST)
    list(APPEND canaries "undefined=runtime error: signed integer overflow")
  endif()
  if("thread" IN_LIST UNROUND_SANITIZER_LIST)
    list(APPEND canaries "thread=ThreadSanitizer: data race")
  endif()

  foreach(canary IN LISTS canaries)
    string(FIND "${canary}" "=" split)
    string(SUBSTRING "${canary}" 0 ${split} kind)
    math(EXPR start "${split} + 1")
    string(SUBSTRING "${canary}" ${start} -1 report)
    unround_add_test(sanitizer.${kind} unround_sanitizer_canary ${kind})
    set_tests_properties(sanitizer.${kind} PROPERTIES PASS_REGULAR_EXPRESSION "${report}")
  endforeach()
endif()

if(UNROUND_WITH_PYTHON)
  add_test(NAME c.jpegio.exports
    COMMAND "${CMAKE_COMMAND}"
      "-DLIBRARY=$<TARGET_FILE:unround_jpegio_shared>"
      "-DHEADER=${PROJECT_SOURCE_DIR}/c/jpegio/include/unround/jpegio.h"
      "-DNM=${CMAKE_NM}" "-DOBJDUMP=${CMAKE_OBJDUMP}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/check_exports.cmake")
endif()

add_library(unround_header_check OBJECT check_unround_header.c)
target_include_directories(unround_header_check PRIVATE "${PROJECT_SOURCE_DIR}/rust/capi/include")
target_link_libraries(unround_header_check PRIVATE unround::jpegio)
unround_target_defaults(unround_header_check)
if(UNROUND_WITH_CPP)
  add_library(unround_header_check_cpp OBJECT check_unround_header.cpp)
  target_include_directories(unround_header_check_cpp PRIVATE "${PROJECT_SOURCE_DIR}/rust/capi/include")
  target_link_libraries(unround_header_check_cpp PRIVATE unround::jpegio)
  unround_target_defaults(unround_header_check_cpp)
endif()
