# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
#
# The tools that JPEG-Unround is compared against, built from their sources
# with the toolchain that JPEG-Unround is built with (LLVM: Clang, LLD and
# LLVM's archivers, which cmake/LLVMToolchain.cmake checks), on Windows, Linux
# and macOS:
#
#   jpeg2png       victorvde/jpeg2png v1.01                   GPL-3.0-or-later
#   jpegqs         ilyakurdyukov/jpeg-quantsmooth 1.20260122  LGPL-2.1-or-later
#   knusperli      google/knusperli, its commit of 2026-03-09 Apache-2.0
#                  (it has no releases), with lodepng         Zlib
#   mozjpeg-cjpeg  mozilla/mozjpeg v4.1.5: cjpeg, which       IJG, BSD-3-Clause, Zlib
#                  quantizes with a trellis
#
# jpeg2png and jpegqs read JPEG files with libjpeg-turbo: the one of the
# submodule extern/libjpeg-turbo, built as JPEG-Unround builds it
# (cmake/Libjpeg.cmake). jpeg2png writes PNG files with libpng 1.6.58 and zlib
# 1.3.2.
#
# The sources are fetched when the build is configured, each at a commit named
# here in full, into third_party/src/, which git ignores: none of them is part
# of this repository. A checkout that is not at its commit, or that has been
# changed, stops the configuration. The programs are installed into
# third_party/<system>-<processor>/bin, with sources.txt, which says what they
# were built from.
#
# The tools come with builds of their own -- GNU make and GCC for jpeg2png and
# jpegqs, Bazel for knusperli -- which do not run on every system that this
# project is built on. Their programs are described here instead: the files
# that their builds compile, with the definitions and the optimization that
# their builds give them. What differs:
#
#   - jpeg2png's SSE code (USE_SIMD) is for x86 alone; other processors get its
#     plain C code, which its build gives with SIMD=0.
#   - jpegqs is built on x86-64 as its build makes it with SIMD=select: the code
#     for SSE2, AVX2 and AVX-512 in one program, which chooses as it runs. On
#     arm64 it is built with NEON, which every 64-bit Arm processor has, where
#     its build would compile for the processor that it runs on.
#   - knusperli is optimized (-O2) as Bazel's opt mode would.
#   - The threads of jpeg2png and jpegqs are OpenMP's, LLVM's runtime of it,
#     when the toolchain has it. Without it both are built without threads.
#   - mozjpeg is built without its SIMD code (which needs NASM on x86-64) and
#     without contracting floating-point operations into fused ones, so that
#     its trellis, which computes in floating point, chooses alike on x86-64
#     and on arm64, where Clang would contract.
#
# scripts/comparison-tools.sh configures, builds, tests and installs this.

cmake_minimum_required(VERSION 3.30)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake")

set(CMAKE_OSX_DEPLOYMENT_TARGET "26.0" CACHE STRING "Minimum macOS version")
set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "Architectures to build for on macOS")
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")

project(JPEG-Unround-comparison-tools
  DESCRIPTION "The tools that JPEG-Unround is compared against"
  LANGUAGES C CXX)

if(NOT CMAKE_C_COMPILER_ID MATCHES "Clang" OR NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang"
    OR NOT CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "GNU"
    OR NOT CMAKE_LINKER_TYPE STREQUAL "LLD")
  message(FATAL_ERROR
    "The tools are built with the toolchain of JPEG-Unround: clang, clang++ and LLD. "
    "scripts/comparison-tools.sh configures the build with it.")
endif()
include(LLVMToolchain)

if(NOT CMAKE_BUILD_TYPE STREQUAL "Release")
  message(FATAL_ERROR "The tools are built for release (CMAKE_BUILD_TYPE=Release).")
endif()

# Where the programs go: a directory of each system and processor, so that the
# builds of Windows and of WSL, say, can share one checkout.
string(TOLOWER "${CMAKE_SYSTEM_NAME}" UNROUND_SYSTEM)
if(UNROUND_SYSTEM STREQUAL "darwin")
  set(UNROUND_SYSTEM macos)
endif()
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" UNROUND_PROCESSOR)
if(UNROUND_PROCESSOR MATCHES "^(amd64|x86_64|x64)$")
  set(UNROUND_PROCESSOR x86_64)
elseif(UNROUND_PROCESSOR MATCHES "^(arm64|aarch64)$")
  set(UNROUND_PROCESSOR arm64)
else()
  message(FATAL_ERROR "The tools are built for x86-64 and arm64, not for ${CMAKE_SYSTEM_PROCESSOR}.")
endif()
cmake_path(SET UNROUND_THIRD_PARTY NORMALIZE "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party")
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${UNROUND_THIRD_PARTY}/${UNROUND_SYSTEM}-${UNROUND_PROCESSOR}"
    CACHE PATH "Where the tools are installed" FORCE)
endif()
# The programs installed keep the paths of the libraries that they were linked
# with (LLVM's OpenMP runtime on Linux and macOS).
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)

# --------------------------------------------------------------------------
# The sources

include(FetchContent)
find_package(Git REQUIRED)

set(UNROUND_SOURCES_FILE "${CMAKE_BINARY_DIR}/sources.txt")
file(WRITE "${UNROUND_SOURCES_FILE}" "")

# Fetches a repository of GitHub at a commit into third_party/src/<name>, and
# checks that the checkout is at that commit and unchanged. The checkout is
# made without converting line ends, as git does on Windows by default, so that
# it is the same wherever it is checked out, and so that the build of another
# system can check it.
function(unround_fetch name repository commit license)
  set(directory "${UNROUND_THIRD_PARTY}/src/${name}")
  FetchContent_Declare(${name}
    GIT_REPOSITORY "https://github.com/${repository}.git"
    GIT_TAG "${commit}"
    GIT_CONFIG core.autocrlf=false
    SOURCE_DIR "${directory}"
    # A directory that none of them has: FetchContent_MakeAvailable() then
    # fetches the sources and builds none of them itself.
    SOURCE_SUBDIR "not-a-directory")
  FetchContent_MakeAvailable(${name})

  execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${directory}" rev-parse HEAD
    OUTPUT_VARIABLE head OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
  if(NOT head STREQUAL commit)
    message(FATAL_ERROR
      "${directory} is at the commit ${head}, not at ${commit}. Remove it, and the "
      "configuration fetches it again.")
  endif()
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${directory}" -c core.autocrlf=false -c core.fileMode=false
      status --porcelain --untracked-files=no
    OUTPUT_VARIABLE changed OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
  if(changed)
    message(FATAL_ERROR
      "${directory} has been changed since it was checked out, and the tools are compared as "
      "their authors wrote them:\n${changed}\nRemove it, and the configuration fetches it again.")
  endif()
  message(STATUS "${name}: ${repository} at ${commit}")
  file(APPEND "${UNROUND_SOURCES_FILE}" "${name}\thttps://github.com/${repository}\t${commit}\t${license}\n")
  set(${name}_SOURCE_DIR "${directory}" PARENT_SCOPE)
endfunction()

unround_fetch(jpeg2png victorvde/jpeg2png
  04054a2ee78b08cfbe69c71f84f9d84c0aedd9f3 "GPL-3.0-or-later")                # v1.01
unround_fetch(jpeg-quantsmooth ilyakurdyukov/jpeg-quantsmooth
  9417f99d3f827d37d2f653713d3a6f92a2256f13 "LGPL-2.1-or-later")               # 1.20260122
unround_fetch(knusperli google/knusperli
  affcb3561f15ae0173ea6891ec935459b5bcf206 "Apache-2.0")                      # 2026-03-09
unround_fetch(lodepng lvandeve/lodepng
  93e348fc8ee50729b2bfc3d602df2cae6051a40e "Zlib")                            # knusperli's
unround_fetch(mozjpeg mozilla/mozjpeg
  6c9f0897afa1c2738d7222a0a9ab49e8b536a267 "IJG AND BSD-3-Clause AND Zlib")   # v4.1.5
unround_fetch(zlib madler/zlib
  da607da739fa6047df13e66a2af6b8bec7c2a498 "Zlib")                            # v1.3.2
unround_fetch(libpng pnggroup/libpng
  3061454d980de7d53608f594194cfac722721d2a "libpng-2.0")                      # v1.6.58

# libjpeg-turbo, as JPEG-Unround builds it: unround::libjpeg, and its cjpeg
# and djpeg, which the tests below use.
include(Libjpeg)
execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${UNROUND_LIBJPEG_SOURCE}" rev-parse HEAD
  OUTPUT_VARIABLE UNROUND_LIBJPEG_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
file(APPEND "${UNROUND_SOURCES_FILE}"
  "libjpeg-turbo\thttps://github.com/libjpeg-turbo/libjpeg-turbo\t${UNROUND_LIBJPEG_COMMIT}\t"
  "IJG AND BSD-3-Clause AND Zlib\n")

# --------------------------------------------------------------------------
# The libraries with builds of their own: zlib, libpng and mozjpeg, each built
# by its CMake build, static, and installed into this build directory.

include(ExternalProject)

set(UNROUND_EXTERNAL_CACHE_ARGS
  "-DCMAKE_BUILD_TYPE:STRING=Release"
  "-DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}"
  "-DCMAKE_AR:FILEPATH=${CMAKE_AR}"
  "-DCMAKE_RANLIB:FILEPATH=${CMAKE_RANLIB}"
  "-DCMAKE_C_COMPILER_AR:FILEPATH=${CMAKE_C_COMPILER_AR}"
  "-DCMAKE_C_COMPILER_RANLIB:FILEPATH=${CMAKE_C_COMPILER_RANLIB}"
  "-DCMAKE_LINKER_TYPE:STRING=LLD"
  "-DCMAKE_EXE_LINKER_FLAGS:STRING=-fuse-ld=lld"
  "-DCMAKE_MSVC_RUNTIME_LIBRARY:STRING=${CMAKE_MSVC_RUNTIME_LIBRARY}"
  "-DCMAKE_INSTALL_LIBDIR:PATH=lib"
  "-DCMAKE_INSTALL_BINDIR:PATH=bin"
  "-DCMAKE_INSTALL_INCLUDEDIR:PATH=include")
if(APPLE)
  list(APPEND UNROUND_EXTERNAL_CACHE_ARGS
    "-DCMAKE_OSX_DEPLOYMENT_TARGET:STRING=${CMAKE_OSX_DEPLOYMENT_TARGET}"
    "-DCMAKE_OSX_ARCHITECTURES:STRING=${CMAKE_OSX_ARCHITECTURES}"
    "-DCMAKE_OSX_SYSROOT:PATH=${CMAKE_OSX_SYSROOT}")
endif()

# Adds a static library that an external project installs as an imported
# target, which depends on the external project.
function(unround_imported_library target project library include)
  file(MAKE_DIRECTORY "${include}")
  add_library(${target} STATIC IMPORTED GLOBAL)
  set_target_properties(${target} PROPERTIES
    IMPORTED_LOCATION "${library}"
    INTERFACE_INCLUDE_DIRECTORIES "${include}")
  add_dependencies(${target} ${project})
endfunction()

set(UNROUND_ZLIB_INSTALL "${CMAKE_BINARY_DIR}/zlib/install")
if(WIN32)
  set(UNROUND_ZLIB_LIBRARY "${UNROUND_ZLIB_INSTALL}/lib/zs.lib")
else()
  set(UNROUND_ZLIB_LIBRARY "${UNROUND_ZLIB_INSTALL}/lib/libz.a")
endif()
ExternalProject_Add(zlib-build
  SOURCE_DIR "${zlib_SOURCE_DIR}"
  PREFIX "${CMAKE_BINARY_DIR}/zlib"
  INSTALL_DIR "${UNROUND_ZLIB_INSTALL}"
  CMAKE_CACHE_ARGS ${UNROUND_EXTERNAL_CACHE_ARGS}
    "-DCMAKE_INSTALL_PREFIX:PATH=${UNROUND_ZLIB_INSTALL}"
    "-DZLIB_BUILD_SHARED:BOOL=OFF"
    "-DZLIB_BUILD_STATIC:BOOL=ON"
    "-DZLIB_BUILD_TESTING:BOOL=OFF"
  BUILD_BYPRODUCTS "${UNROUND_ZLIB_LIBRARY}"
  LOG_CONFIGURE ON LOG_BUILD ON LOG_INSTALL ON LOG_OUTPUT_ON_FAILURE ON)
unround_imported_library(unround_zlib zlib-build "${UNROUND_ZLIB_LIBRARY}" "${UNROUND_ZLIB_INSTALL}/include")

set(UNROUND_PNG_INSTALL "${CMAKE_BINARY_DIR}/libpng/install")
if(WIN32)
  set(UNROUND_PNG_LIBRARY "${UNROUND_PNG_INSTALL}/lib/libpng16_static.lib")
else()
  set(UNROUND_PNG_LIBRARY "${UNROUND_PNG_INSTALL}/lib/libpng16.a")
endif()
ExternalProject_Add(libpng-build
  SOURCE_DIR "${libpng_SOURCE_DIR}"
  PREFIX "${CMAKE_BINARY_DIR}/libpng"
  INSTALL_DIR "${UNROUND_PNG_INSTALL}"
  DEPENDS zlib-build
  CMAKE_CACHE_ARGS ${UNROUND_EXTERNAL_CACHE_ARGS}
    "-DCMAKE_INSTALL_PREFIX:PATH=${UNROUND_PNG_INSTALL}"
    "-DPNG_SHARED:BOOL=OFF"
    "-DPNG_STATIC:BOOL=ON"
    "-DPNG_FRAMEWORK:BOOL=OFF"
    "-DPNG_TESTS:BOOL=OFF"
    "-DPNG_TOOLS:BOOL=OFF"
    "-DZLIB_ROOT:PATH=${UNROUND_ZLIB_INSTALL}"
    "-DZLIB_USE_STATIC_LIBS:BOOL=ON"
    "-DZLIB_INCLUDE_DIR:PATH=${UNROUND_ZLIB_INSTALL}/include"
    "-DZLIB_LIBRARY:FILEPATH=${UNROUND_ZLIB_LIBRARY}"
  BUILD_BYPRODUCTS "${UNROUND_PNG_LIBRARY}"
  LOG_CONFIGURE ON LOG_BUILD ON LOG_INSTALL ON LOG_OUTPUT_ON_FAILURE ON)
unround_imported_library(unround_png libpng-build "${UNROUND_PNG_LIBRARY}" "${UNROUND_PNG_INSTALL}/include")
target_link_libraries(unround_png INTERFACE unround_zlib $<$<PLATFORM_ID:Linux>:m>)

set(UNROUND_MOZJPEG_INSTALL "${CMAKE_BINARY_DIR}/mozjpeg/install")
set(UNROUND_MOZJPEG_CJPEG "${UNROUND_MOZJPEG_INSTALL}/bin/cjpeg${CMAKE_EXECUTABLE_SUFFIX}")
# On Windows, mozjpeg 4.1.5 names the files of its installer (an NSIS script,
# which is not built here) after the compiler: "vc" for Visual C++, "gcc" for
# MinGW, and nothing for Clang, whose configuration then stops on a file that
# does not exist. Clang here builds for Visual C++'s ABI.
set(UNROUND_MOZJPEG_WINDOWS_ARGS "")
if(WIN32)
  set(UNROUND_MOZJPEG_WINDOWS_ARGS "-DINST_ID:STRING=vc")
endif()
ExternalProject_Add(mozjpeg-build
  SOURCE_DIR "${mozjpeg_SOURCE_DIR}"
  PREFIX "${CMAKE_BINARY_DIR}/mozjpeg"
  INSTALL_DIR "${UNROUND_MOZJPEG_INSTALL}"
  CMAKE_CACHE_ARGS ${UNROUND_EXTERNAL_CACHE_ARGS} ${UNROUND_MOZJPEG_WINDOWS_ARGS}
    "-DCMAKE_INSTALL_PREFIX:PATH=${UNROUND_MOZJPEG_INSTALL}"
    # mozjpeg 4.1.5 asks for CMake 2.8.12, whose ways CMake 4 no longer has.
    "-DCMAKE_POLICY_VERSION_MINIMUM:STRING=3.5"
    "-DCMAKE_C_FLAGS:STRING=-ffp-contract=off"
    "-DENABLE_SHARED:BOOL=OFF"
    "-DENABLE_STATIC:BOOL=ON"
    "-DWITH_CRT_DLL:BOOL=ON"
    "-DWITH_SIMD:BOOL=OFF"
    "-DWITH_TURBOJPEG:BOOL=OFF"
    "-DWITH_JAVA:BOOL=OFF"
    "-DPNG_SUPPORTED:BOOL=OFF"
  BUILD_BYPRODUCTS "${UNROUND_MOZJPEG_CJPEG}"
  LOG_CONFIGURE ON LOG_BUILD ON LOG_INSTALL ON LOG_OUTPUT_ON_FAILURE ON)

# --------------------------------------------------------------------------
# The programs described here

set(UNROUND_X86_64 OFF)
if(UNROUND_PROCESSOR STREQUAL "x86_64")
  set(UNROUND_X86_64 ON)
endif()

option(UNROUND_COMPARISON_OPENMP "Build jpeg2png and jpegqs with threads (OpenMP)" ON)
if(UNROUND_COMPARISON_OPENMP)
  # On Windows, Clang links LLVM's runtime from the lib directory beside its
  # own, which FindOpenMP does not look in.
  if(WIN32 AND NOT DEFINED OpenMP_libomp_LIBRARY)
    cmake_path(SET UNROUND_LIBOMP NORMALIZE "${UNROUND_LLVM_BIN}/../lib/libomp.lib")
    if(EXISTS "${UNROUND_LIBOMP}")
      set(OpenMP_libomp_LIBRARY "${UNROUND_LIBOMP}" CACHE FILEPATH "LLVM's OpenMP runtime")
    endif()
  endif()
  find_package(OpenMP COMPONENTS C)
endif()
if(OpenMP_C_FOUND)
  message(STATUS "jpeg2png and jpegqs: with threads (OpenMP ${OpenMP_C_VERSION})")
  set(UNROUND_THREADS OpenMP::OpenMP_C)
  # The DLL of the runtime goes beside the programs, which then run wherever
  # they are run from.
  if(WIN32)
    cmake_path(SET UNROUND_LIBOMP_DLL NORMALIZE "${UNROUND_LLVM_BIN}/libomp.dll")
    if(NOT EXISTS "${UNROUND_LIBOMP_DLL}")
      message(FATAL_ERROR "LLVM's OpenMP runtime has no DLL at ${UNROUND_LIBOMP_DLL}")
    endif()
    install(FILES "${UNROUND_LIBOMP_DLL}" DESTINATION bin)
  endif()
else()
  message(STATUS "jpeg2png and jpegqs: without threads (no OpenMP)")
  set(UNROUND_THREADS "")
endif()

# jpeg2png: the objects of its Makefile, with its definitions (BUILTINS=1,
# PRAGMA_FP_CONTRACT=0, SIMD=1 on x86) and -O3.
set(src "${jpeg2png_SOURCE_DIR}")
add_executable(jpeg2png
  "${src}/jpeg2png.c" "${src}/utils.c" "${src}/jpeg.c" "${src}/png.c" "${src}/box.c"
  "${src}/compute.c" "${src}/logger.c" "${src}/progressbar.c" "${src}/fp_exceptions.c"
  "${src}/gopt/gopt.c" "${src}/ooura/dct.c")
set_target_properties(jpeg2png PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
target_compile_definitions(jpeg2png PRIVATE
  BUILTIN_UNREACHABLE BUILTIN_ASSUME_ALIGNED ATTRIBUTE_UNUSED
  $<$<BOOL:${UNROUND_X86_64}>:USE_SIMD>
  $<$<PLATFORM_ID:Windows>:_CRT_SECURE_NO_WARNINGS>)
target_compile_options(jpeg2png PRIVATE -O3 -ffp-contract=off)
# jpeg2png includes <stdnoreturn.h> before <stdlib.h>, and Windows' <stdlib.h>
# declares exit() and abort() with __declspec(noreturn), which the macro
# noreturn of the former then breaks. <stdlib.h> is included first.
if(WIN32)
  target_compile_options(jpeg2png PRIVATE -include stdlib.h)
endif()
target_link_libraries(jpeg2png PRIVATE unround::libjpeg unround_png ${UNROUND_THREADS}
  $<$<PLATFORM_ID:Linux>:m>)

# jpegqs: its program (quantsmooth.c) with -O2, as its Makefile builds it. With
# SIMD=select, the library's code (libjpegqs.c) is compiled once for each
# instruction set, and the program chooses among them.
set(src "${jpeg-quantsmooth_SOURCE_DIR}")
add_executable(jpegqs "${src}/quantsmooth.c")
set(UNROUND_JPEGQS_DEFINITIONS APPNAME=jpegqs $<$<PLATFORM_ID:Windows>:_CRT_SECURE_NO_WARNINGS>)
target_compile_definitions(jpegqs PRIVATE ${UNROUND_JPEGQS_DEFINITIONS})
target_compile_options(jpegqs PRIVATE -O2)
target_link_libraries(jpegqs PRIVATE unround::libjpeg ${UNROUND_THREADS} $<$<PLATFORM_ID:Linux>:m>)
if(UNROUND_X86_64)
  target_compile_definitions(jpegqs PRIVATE SIMD_SELECT)
  set(UNROUND_JPEGQS_FLAGS_base "")
  set(UNROUND_JPEGQS_FLAGS_sse2 -msse2)
  set(UNROUND_JPEGQS_FLAGS_avx2 -mavx2 -mfma)
  set(UNROUND_JPEGQS_FLAGS_avx512 -mavx512f -mavx512dq -mavx512bw -mfma)
  foreach(simd IN ITEMS base sse2 avx2 avx512)
    string(TOUPPER "${simd}" name)
    add_library(jpegqs_${simd} OBJECT "${src}/libjpegqs.c")
    target_compile_definitions(jpegqs_${simd} PRIVATE
      ${UNROUND_JPEGQS_DEFINITIONS} TRANSCODE_ONLY WITH_LOG SIMD_SELECT SIMD_NAME=${simd} SIMD_${name})
    target_compile_options(jpegqs_${simd} PRIVATE -O2 ${UNROUND_JPEGQS_FLAGS_${simd}})
    target_link_libraries(jpegqs_${simd} PRIVATE unround::libjpeg ${UNROUND_THREADS})
    target_link_libraries(jpegqs PRIVATE jpegqs_${simd})
  endforeach()
endif()

# knusperli: the sources of its Bazel targets, with lodepng, optimized as
# Bazel's opt mode would, and dct_double.cc with -ffast-math as its BUILD says.
set(src "${knusperli_SOURCE_DIR}")
add_executable(knusperli
  "${src}/decode.cc" "${src}/jpeg_data.cc" "${src}/jpeg_data_decoder.cc"
  "${src}/jpeg_data_reader.cc" "${src}/jpeg_huffman_decode.cc" "${src}/output_image.cc"
  "${src}/dct_double.cc" "${src}/gamma_correct.cc" "${src}/idct.cc"
  "${src}/preprocess_downsample.cc" "${src}/quantize.cc" "${lodepng_SOURCE_DIR}/lodepng.cpp")
set_target_properties(knusperli PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
target_include_directories(knusperli PRIVATE "${src}" "${lodepng_SOURCE_DIR}")
target_compile_definitions(knusperli PRIVATE $<$<PLATFORM_ID:Windows>:_CRT_SECURE_NO_WARNINGS>)
target_compile_options(knusperli PRIVATE -O2)
set_source_files_properties("${src}/dct_double.cc" TARGET_DIRECTORY knusperli
  PROPERTIES COMPILE_OPTIONS -ffast-math)

# --------------------------------------------------------------------------
# Installation

install(TARGETS jpeg2png jpegqs knusperli RUNTIME DESTINATION bin)
install(PROGRAMS "${UNROUND_MOZJPEG_CJPEG}" DESTINATION bin
  RENAME "mozjpeg-cjpeg${CMAKE_EXECUTABLE_SUFFIX}")
install(FILES "${UNROUND_SOURCES_FILE}" DESTINATION .)

# --------------------------------------------------------------------------
# Tests: each program runs on a JPEG file, and what it writes is a PNG or a
# JPEG file that libjpeg-turbo's djpeg decodes.

enable_testing()
set(UNROUND_TEST_DIRECTORY "${CMAKE_BINARY_DIR}/test")
file(MAKE_DIRECTORY "${UNROUND_TEST_DIRECTORY}")

# A picture of 45 by 37 pixels -- sizes of no whole number of blocks -- with a
# ramp, an edge and a colour of each channel, as ASCII PPM, which cjpeg reads.
set(UNROUND_SAMPLE "P3\n45 37\n255\n")
foreach(y RANGE 36)
  foreach(x RANGE 44)
    math(EXPR r "(${x} * 255) / 44")
    math(EXPR g "(${y} * 255) / 36")
    if(x GREATER y)
      set(b 200)
    else()
      set(b 30)
    endif()
    string(APPEND UNROUND_SAMPLE "${r} ${g} ${b}\n")
  endforeach()
endforeach()
file(WRITE "${UNROUND_TEST_DIRECTORY}/sample.ppm" "${UNROUND_SAMPLE}")

add_test(NAME sample
  COMMAND "${UNROUND_CJPEG}" -quality 20 -sample 2x2 -outfile sample.jpg sample.ppm
  WORKING_DIRECTORY "${UNROUND_TEST_DIRECTORY}")
set_tests_properties(sample PROPERTIES FIXTURES_SETUP sample)

add_test(NAME jpeg2png
  COMMAND jpeg2png sample.jpg -o jpeg2png.png
  WORKING_DIRECTORY "${UNROUND_TEST_DIRECTORY}")
add_test(NAME jpegqs
  COMMAND jpegqs sample.jpg jpegqs.jpg
  WORKING_DIRECTORY "${UNROUND_TEST_DIRECTORY}")
add_test(NAME knusperli
  COMMAND knusperli sample.jpg knusperli.png
  WORKING_DIRECTORY "${UNROUND_TEST_DIRECTORY}")
add_test(NAME mozjpeg-cjpeg
  COMMAND "${UNROUND_MOZJPEG_CJPEG}" -quality 20 -sample 2x2 -outfile mozjpeg.jpg sample.ppm
  WORKING_DIRECTORY "${UNROUND_TEST_DIRECTORY}")
set_tests_properties(jpeg2png jpegqs knusperli mozjpeg-cjpeg PROPERTIES
  FIXTURES_REQUIRED sample FIXTURES_SETUP outputs)

add_test(NAME outputs
  COMMAND "${CMAKE_COMMAND}"
    "-DDJPEG=${UNROUND_DJPEG}"
    "-DPNG=jpeg2png.png;knusperli.png"
    "-DJPEG=jpegqs.jpg;mozjpeg.jpg"
    "-DWIDTH=45" "-DHEIGHT=37"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/check-outputs.cmake"
  WORKING_DIRECTORY "${UNROUND_TEST_DIRECTORY}")
set_tests_properties(outputs PROPERTIES FIXTURES_REQUIRED outputs)
