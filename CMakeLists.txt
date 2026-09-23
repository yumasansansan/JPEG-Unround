# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
#
# The native build of JPEG-Unround: libjpeg-turbo (extern/), the C layer over
# it that all three implementations read JPEG files with (c/jpegio), and the
# C++ implementation (cpp/). The Rust implementation is built by cargo and the
# Python one by pip; both take the C layer from here.

cmake_minimum_required(VERSION 3.30)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# macOS builds are for Apple Silicon and macOS 26 or later. Both have to be set
# before project().
set(CMAKE_OSX_DEPLOYMENT_TARGET "26.0" CACHE STRING "Minimum macOS version")
set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "Architectures to build for on macOS")

# Windows builds use the C runtime as the DLL of its release build, in every
# configuration and in every target, libjpeg-turbo's included: memory that one
# part allocates, another frees, which works only within one runtime, and the
# debug runtime is not one that systems have.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")

project(JPEG-Unround
  VERSION 0.1.0
  DESCRIPTION "JPEG decoding that removes compression artifacts within the quantization intervals"
  HOMEPAGE_URL "https://github.com/yumasansansan/JPEG-Unround"
  LANGUAGES C CXX)

# --------------------------------------------------------------------------
# The presets are how this project is configured
#
# CMakePresets.json chooses the compilers, the linker, the generator and the
# settings that JPEG-Unround is built and tested with. A configuration put
# together by hand differs from them without saying so; a build that has to
# differ can inherit a preset of its own in CMakeUserPresets.json.
set(UNROUND_PRESET "" CACHE STRING "The preset this build was configured with; CMakePresets.json sets it")
if(NOT UNROUND_PRESET)
  message(FATAL_ERROR
    "Configure through a preset: cmake --preset debug, and cmake --list-presets names them "
    "all. The presets choose the compilers, the linker, the generator and the settings this "
    "project is built and tested with. A build that has to differ can inherit a preset of its "
    "own in CMakeUserPresets.json, or say -DUNROUND_PRESET=<a name for it> and own the "
    "settings it is then left with.")
endif()
message(STATUS "Preset: ${UNROUND_PRESET}")

if(NOT CMAKE_C_COMPILER_ID MATCHES "Clang" OR NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  message(FATAL_ERROR
    "JPEG-Unround is built with Clang, not ${CMAKE_CXX_COMPILER_ID}. Configure with "
    "CMAKE_C_COMPILER=clang and CMAKE_CXX_COMPILER=clang++, as the presets do.")
endif()
if(NOT CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "GNU")
  message(FATAL_ERROR
    "JPEG-Unround is built with Clang's GNU-style driver (clang, clang++), not "
    "${CMAKE_CXX_COMPILER_FRONTEND_VARIANT}-style (clang-cl).")
endif()

# LLD links everything, on every system: CMake gives CMAKE_LINKER_TYPE to every
# target and to try_compile().
if(NOT CMAKE_LINKER_TYPE STREQUAL "LLD")
  message(FATAL_ERROR
    "JPEG-Unround is linked with LLD, but CMAKE_LINKER_TYPE is '${CMAKE_LINKER_TYPE}'. "
    "Configure with -DCMAKE_LINKER_TYPE=LLD, as the presets do.")
endif()

# The toolchain has to be LLVM's, of one version (cmake/LLVMToolchain.cmake).
include(LLVMToolchain)

# --------------------------------------------------------------------------
# Options

option(UNROUND_WERROR "Treat warnings in JPEG-Unround's own sources as errors" OFF)
option(UNROUND_BUILD_TESTS "Build the tests and register them with CTest" ON)
option(UNROUND_BUILD_FUZZERS "Build the fuzz targets with libFuzzer (not on Windows)" OFF)
option(UNROUND_WITH_CPP "Build the C++ implementation" ON)
option(UNROUND_WITH_PYTHON "Build the C layer as the shared library that the Python implementation loads" OFF)
set(UNROUND_SANITIZERS "" CACHE STRING "Sanitizers to build with, comma-separated (address, undefined, thread)")

# libFuzzer runs on Linux and macOS. On Windows it cannot be linked at all: the
# libFuzzer that LLVM builds for Windows uses the static C runtime, and this
# project the dynamic one, and the two cannot go into one program. The fuzz
# targets still run as tests there, on inputs of their own (c/jpegio/fuzz).
if(UNROUND_BUILD_FUZZERS AND CMAKE_SYSTEM_NAME STREQUAL "Windows")
  message(FATAL_ERROR
    "UNROUND_BUILD_FUZZERS is not for Windows: LLVM's libFuzzer for Windows is linked with the "
    "static C runtime (/MT), and JPEG-Unround with the dynamic one (/MD).")
endif()

# No file of this project is a C++ module, and none imports one: the C++ is
# written with headers until import std is available on every system this
# project supports. Left alone, CMake scans every C++ source for modules.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

set(CMAKE_C_VISIBILITY_PRESET "hidden")
set(CMAKE_CXX_VISIBILITY_PRESET "hidden")
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --------------------------------------------------------------------------
# Sanitizers
#
# UNROUND_SANITIZERS names the sanitizers of a build, comma-separated, and gives
# them to every target of this project; libjpeg-turbo is given them too, but for
# undefined (cmake/Libjpeg.cmake). Undefined behaviour stops the program
# (-fno-sanitize-recover=all) instead of being reported and passed over, and the
# frame pointer stays, so that the reports name the callers. address and thread
# cannot be built together, and Clang has no thread sanitizer for Windows.
# Sanitizers are not for Release builds: the sanitize presets configure
# RelWithDebInfo.
#
# The tests run with the environment below. A sanitizer that finds something
# ends the program with an exit status (abort_on_error=0), as it does by default
# everywhere but on Apple's systems, where it calls abort(): CTest takes a
# program that a signal ended for one that crashed, whatever its output says, so
# a test that passes only on the report of a mistake made on purpose (tests/)
# could not pass there, and a real finding would read as a crash rather than as
# the report it is. LeakSanitizer comes with the address sanitizer and is asked
# for on Linux, where it runs; asking for it elsewhere only makes the address
# sanitizer refuse to start. On Windows the address sanitizer's runtime is a DLL
# beside the compiler, where a test does not look for it, so the tests have its
# directory put at the front of PATH.
set(UNROUND_SANITIZER_LIST "")
set(UNROUND_SANITIZER_TEST_ENVIRONMENT "")
set(UNROUND_SANITIZER_TEST_PATH "")
if(UNROUND_SANITIZERS)
  if(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(FATAL_ERROR
      "UNROUND_SANITIZERS is not for Release builds. Configure RelWithDebInfo, as the sanitize "
      "presets do.")
  endif()
  string(REPLACE "," ";" UNROUND_SANITIZER_LIST "${UNROUND_SANITIZERS}")
  foreach(sanitizer IN LISTS UNROUND_SANITIZER_LIST)
    if(NOT sanitizer MATCHES "^(address|undefined|thread)$")
      message(FATAL_ERROR
        "UNROUND_SANITIZERS names '${sanitizer}', which is not one of address, undefined, thread.")
    endif()
  endforeach()
  if("address" IN_LIST UNROUND_SANITIZER_LIST AND "thread" IN_LIST UNROUND_SANITIZER_LIST)
    message(FATAL_ERROR "UNROUND_SANITIZERS names both address and thread, which cannot be built together.")
  endif()
  if("thread" IN_LIST UNROUND_SANITIZER_LIST AND CMAKE_SYSTEM_NAME STREQUAL "Windows")
    message(FATAL_ERROR "UNROUND_SANITIZERS names thread, for which Clang has no sanitizer on Windows.")
  endif()
  list(JOIN UNROUND_SANITIZER_LIST "," UNROUND_SANITIZER_OPTION)
  message(STATUS "Sanitizers: ${UNROUND_SANITIZER_OPTION}")
  add_compile_options("-fsanitize=${UNROUND_SANITIZER_OPTION}" -fno-sanitize-recover=all -fno-omit-frame-pointer)
  add_link_options("-fsanitize=${UNROUND_SANITIZER_OPTION}")

  list(APPEND UNROUND_SANITIZER_TEST_ENVIRONMENT "UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1:abort_on_error=0")
  if("address" IN_LIST UNROUND_SANITIZER_LIST)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
      list(APPEND UNROUND_SANITIZER_TEST_ENVIRONMENT "ASAN_OPTIONS=abort_on_error=0:detect_leaks=1")
    else()
      list(APPEND UNROUND_SANITIZER_TEST_ENVIRONMENT "ASAN_OPTIONS=abort_on_error=0")
    endif()
  endif()
  if("thread" IN_LIST UNROUND_SANITIZER_LIST)
    list(APPEND UNROUND_SANITIZER_TEST_ENVIRONMENT
      "TSAN_OPTIONS=halt_on_error=1:second_deadlock_stack=1:abort_on_error=0")
  endif()
  if(CMAKE_SYSTEM_NAME STREQUAL "Windows" AND "address" IN_LIST UNROUND_SANITIZER_LIST)
    execute_process(
      COMMAND "${CMAKE_CXX_COMPILER}" -print-file-name=clang_rt.asan_dynamic-x86_64.dll
      OUTPUT_VARIABLE UNROUND_ASAN_RUNTIME OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(IS_ABSOLUTE "${UNROUND_ASAN_RUNTIME}" AND EXISTS "${UNROUND_ASAN_RUNTIME}")
      cmake_path(GET UNROUND_ASAN_RUNTIME PARENT_PATH UNROUND_ASAN_RUNTIME_DIRECTORY)
      set(UNROUND_SANITIZER_TEST_PATH "PATH=path_list_prepend:${UNROUND_ASAN_RUNTIME_DIRECTORY}")
    else()
      message(WARNING
        "The address sanitizer's runtime DLL was not found beside the compiler; the tests will "
        "need its directory on PATH.")
    endif()
  endif()
endif()

# A fuzzing build gives all of this project's code the coverage that libFuzzer
# steers by; the fuzz targets alone link libFuzzer itself.
if(UNROUND_BUILD_FUZZERS)
  add_compile_options(-fsanitize=fuzzer-no-link)
endif()

# Registers a test, with the sanitizers' environment.
function(unround_add_test name)
  add_test(NAME "${name}" COMMAND ${ARGN})
  if(UNROUND_SANITIZER_TEST_ENVIRONMENT)
    set_property(TEST "${name}" APPEND PROPERTY ENVIRONMENT ${UNROUND_SANITIZER_TEST_ENVIRONMENT})
  endif()
  if(UNROUND_SANITIZER_TEST_PATH)
    set_property(TEST "${name}" APPEND PROPERTY ENVIRONMENT_MODIFICATION "${UNROUND_SANITIZER_TEST_PATH}")
  endif()
endfunction()

# --------------------------------------------------------------------------
# Warnings, language standards and hardening of this project's own code
#
# Every warning below is an error in the presets (UNROUND_WERROR). The code is
# clean of them, and anything new is either a fault to fix or a decision to
# write down where it is made.
set(UNROUND_WARNING_FLAGS
  -Wall -Wextra -Wpedantic
  -Wconversion -Wsign-conversion -Wshorten-64-to-32 -Wimplicit-float-conversion
  -Wdouble-promotion -Wshadow-all -Wcast-qual -Wcast-align
  -Wformat=2 -Wformat-signedness -Wimplicit-fallthrough -Wundef
  -Wunreachable-code -Wcomma -Wmissing-prototypes -Wmissing-variable-declarations
  -Wdeprecated -Wextra-semi -Wextra-semi-stmt -Wnewline-eof -Wreserved-identifier
  -Wzero-as-null-pointer-constant
  $<$<COMPILE_LANGUAGE:CXX>:-Wold-style-cast>
  $<$<COMPILE_LANGUAGE:CXX>:-Wnon-virtual-dtor>
  $<$<COMPILE_LANGUAGE:CXX>:-Woverloaded-virtual>
  $<$<COMPILE_LANGUAGE:CXX>:-Wsuggest-override>
  $<$<COMPILE_LANGUAGE:CXX>:-Wsuggest-destructor-override>
  $<$<COMPILE_LANGUAGE:CXX>:-Wundefined-func-template>
  $<$<COMPILE_LANGUAGE:CXX>:-Wunused-member-function>
  $<$<COMPILE_LANGUAGE:CXX>:-Wunused-template>)
if(UNROUND_WERROR)
  list(APPEND UNROUND_WARNING_FLAGS -Werror)
else()
  # Falling off the end of a non-void function is undefined behaviour.
  list(APPEND UNROUND_WARNING_FLAGS -Werror=return-type)
endif()

# The hardening of each C++ library stops an out-of-range access to a span, a
# vector or a string rather than let it read or write memory it was not given.
# The stack protector and _FORTIFY_SOURCE do the same for the C of the C
# library's buffers, where glibc and Apple's libc check them; _FORTIFY_SOURCE
# needs an optimized build.
set(UNROUND_DEFINITIONS "")
set(UNROUND_HARDENING_FLAGS "")
if(UNROUND_CXX_LIBRARY STREQUAL "MSVC STL")
  list(APPEND UNROUND_DEFINITIONS $<$<COMPILE_LANGUAGE:CXX>:_MSVC_STL_HARDENING=1>)
elseif(UNROUND_CXX_LIBRARY STREQUAL "libstdc++")
  list(APPEND UNROUND_DEFINITIONS $<$<COMPILE_LANGUAGE:CXX>:_GLIBCXX_ASSERTIONS>)
elseif(UNROUND_CXX_LIBRARY STREQUAL "libc++")
  list(APPEND UNROUND_DEFINITIONS $<$<COMPILE_LANGUAGE:CXX>:_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_FAST>)
endif()
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  # fopen and its kin are used as the C standard has them; the Microsoft C
  # library would ask for its own *_s functions instead.
  list(APPEND UNROUND_DEFINITIONS _CRT_SECURE_NO_WARNINGS)
else()
  list(APPEND UNROUND_HARDENING_FLAGS -fstack-protector-strong)
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    list(APPEND UNROUND_HARDENING_FLAGS
      "$<$<NOT:$<CONFIG:Debug>>:-U_FORTIFY_SOURCE>"
      "$<$<NOT:$<CONFIG:Debug>>:-D_FORTIFY_SOURCE=3>")
  endif()
endif()

# Text on Windows is UTF-8 as it is everywhere else. A program runs there with a
# code page, in which main() gets its arguments and in which fopen(),
# std::ifstream and std::filesystem::path take a name given as char; it is the
# code page of the system's language unless the program's manifest names one
# (932 on Japanese Windows, 1252 on English), and most of Unicode is not in it.
# Every program of this project names UTF-8 (cmake/utf8.manifest), which
# Windows 10 1903 and later honour, and the C++ implementation then handles
# arguments and names as the UTF-8 that they are on Linux and macOS; a test
# (cpp/tests/test_utf8_arguments.cpp) checks it. The standard libraries of Rust
# and Python take both from Windows in UTF-16 and need nothing of the kind.
set(UNROUND_UTF8_MANIFEST "${CMAKE_CURRENT_SOURCE_DIR}/cmake/utf8.manifest")

# Gives a target of this project its language standards, warnings and
# hardening, and a program its manifest on Windows.
function(unround_target_defaults target)
  set_target_properties("${target}" PROPERTIES
    C_STANDARD 23 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF
    CXX_STANDARD 26 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
  target_compile_options("${target}" PRIVATE ${UNROUND_WARNING_FLAGS} ${UNROUND_HARDENING_FLAGS})
  target_compile_definitions("${target}" PRIVATE ${UNROUND_DEFINITIONS})
  get_target_property(type "${target}" TYPE)
  if(CMAKE_SYSTEM_NAME STREQUAL "Windows" AND type STREQUAL "EXECUTABLE")
    target_sources("${target}" PRIVATE "${UNROUND_UTF8_MANIFEST}")
  endif()
endfunction()

if(UNROUND_BUILD_TESTS)
  enable_testing()
endif()

# --------------------------------------------------------------------------
# The parts

include(Libjpeg)
add_subdirectory(c/jpegio)
if(UNROUND_WITH_CPP)
  add_subdirectory(cpp)
endif()
if(UNROUND_BUILD_TESTS)
  add_subdirectory(tests)
endif()
