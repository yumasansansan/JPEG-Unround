# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
#
# libjpeg-turbo, from the submodule extern/libjpeg-turbo, as the static library
# unround::libjpeg. Its build refuses to be a subdirectory of another, so it is
# built as a project of its own (ExternalProject) with what cmake/External.cmake
# gives every such library -- this build's toolchain, Release, position-
# independent code with hidden symbols, the C runtime of this build, and its
# sanitizers -- and installed into the build directory, from which the imported
# target takes the library and the headers. What is chosen for it besides:
#
#   - Static only, with the C runtime of this build on Windows: the DLL of its
#     release build (WITH_CRT_DLL, which libjpeg-turbo would otherwise leave
#     off). The C layer frees memory that libjpeg allocates (the ICC profile).
#   - No SIMD: the SIMD code needs NASM on x86-64, which the toolchain does not
#     have, and it adds nothing this project needs -- the coefficients are read
#     by the entropy decoder, which has no SIMD, and the planes are decoded for
#     checking, not for speed.
#   - No TurboJPEG API and no regression tests. The command-line tools (cjpeg,
#     djpeg, jpegtran) are built: the test data is made with them.
#
# The sub-build runs on every build (BUILD_ALWAYS), which costs a check of an
# up-to-date tree: that way a change of the submodule is always built, and
# nothing downstream is relinked when nothing changed.

include(ExternalProject)
include(External)

# The submodule is found from this file, so that the build of the tools compared
# against (scripts/comparison-tools) can take libjpeg-turbo from here as well.
cmake_path(SET UNROUND_LIBJPEG_SOURCE NORMALIZE "${CMAKE_CURRENT_LIST_DIR}/../extern/libjpeg-turbo")
if(NOT EXISTS "${UNROUND_LIBJPEG_SOURCE}/CMakeLists.txt")
  message(FATAL_ERROR
    "extern/libjpeg-turbo is empty. Check out the submodules: "
    "git submodule update --init --depth 1")
endif()

set(UNROUND_LIBJPEG_PREFIX "${CMAKE_BINARY_DIR}/libjpeg-turbo")
set(UNROUND_LIBJPEG_INSTALL "${UNROUND_LIBJPEG_PREFIX}/install")
if(CMAKE_C_SIMULATE_ID STREQUAL "MSVC")
  set(UNROUND_LIBJPEG_LIBRARY "${UNROUND_LIBJPEG_INSTALL}/lib/jpeg-static.lib")
else()
  set(UNROUND_LIBJPEG_LIBRARY "${UNROUND_LIBJPEG_INSTALL}/lib/libjpeg.a")
endif()
set(UNROUND_CJPEG "${UNROUND_LIBJPEG_INSTALL}/bin/cjpeg${CMAKE_EXECUTABLE_SUFFIX}")
set(UNROUND_DJPEG "${UNROUND_LIBJPEG_INSTALL}/bin/djpeg${CMAKE_EXECUTABLE_SUFFIX}")
set(UNROUND_JPEGTRAN "${UNROUND_LIBJPEG_INSTALL}/bin/jpegtran${CMAKE_EXECUTABLE_SUFFIX}")

unround_external_cache_args(UNROUND_LIBJPEG_CACHE_ARGS "${UNROUND_LIBJPEG_INSTALL}")
list(APPEND UNROUND_LIBJPEG_CACHE_ARGS
  "-DENABLE_SHARED:BOOL=OFF"
  "-DENABLE_STATIC:BOOL=ON"
  "-DWITH_CRT_DLL:BOOL=ON"
  "-DWITH_SIMD:BOOL=OFF"
  "-DWITH_TURBOJPEG:BOOL=OFF"
  "-DWITH_TOOLS:BOOL=ON"
  "-DWITH_TESTS:BOOL=OFF"
  "-DWITH_FUZZ:BOOL=OFF"
  "-DWITH_ARITH_DEC:BOOL=ON"
  "-DWITH_ARITH_ENC:BOOL=ON")

ExternalProject_Add(libjpeg-turbo
  SOURCE_DIR "${UNROUND_LIBJPEG_SOURCE}"
  PREFIX "${UNROUND_LIBJPEG_PREFIX}"
  INSTALL_DIR "${UNROUND_LIBJPEG_INSTALL}"
  CMAKE_CACHE_ARGS ${UNROUND_LIBJPEG_CACHE_ARGS}
  BUILD_ALWAYS ON
  BUILD_BYPRODUCTS "${UNROUND_LIBJPEG_LIBRARY}" "${UNROUND_CJPEG}" "${UNROUND_DJPEG}" "${UNROUND_JPEGTRAN}"
  LOG_CONFIGURE ON
  LOG_BUILD ON
  LOG_INSTALL ON
  LOG_OUTPUT_ON_FAILURE ON)

# An imported target's include directory has to exist when the build is
# generated, before the sub-build has installed anything into it.
file(MAKE_DIRECTORY "${UNROUND_LIBJPEG_INSTALL}/include")
add_library(unround_libjpeg STATIC IMPORTED GLOBAL)
set_target_properties(unround_libjpeg PROPERTIES
  IMPORTED_LOCATION "${UNROUND_LIBJPEG_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${UNROUND_LIBJPEG_INSTALL}/include")
add_dependencies(unround_libjpeg libjpeg-turbo)
add_library(unround::libjpeg ALIAS unround_libjpeg)
