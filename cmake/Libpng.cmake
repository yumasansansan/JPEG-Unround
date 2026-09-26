# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
#
# zlib-ng and libpng, from the submodules extern/zlib-ng and extern/libpng, as the
# static libraries unround::zlib and unround::png, with which the C layer writes
# PNG files. Each is built as a project of its own (ExternalProject) with what
# cmake/External.cmake gives every such library -- this build's toolchain,
# Release, position-independent code with hidden symbols, the C runtime of this
# build, and its sanitizers -- and installed into the build directory, from which
# the imported targets take the libraries and the headers. What is chosen for them
# besides:
#
#   - zlib-ng static, with the API of zlib (ZLIB_COMPAT), which libpng calls. Its
#     optimized code for the processor is chosen when it runs
#     (WITH_RUNTIME_CPU_DETECTION), never for the processor that builds it
#     (WITH_NATIVE_INSTRUCTIONS). No gzip file functions, which nothing calls, and
#     no tests.
#   - libpng static, on the zlib-ng installed beside it (ZLIB_ROOT), and
#     configured by the header it ships (scripts/pnglibconf.h.prebuilt): its full
#     configuration, which it would otherwise generate with awk. No tests and no
#     tools.
#
# The sub-builds run on every build (BUILD_ALWAYS), as libjpeg-turbo's does
# (cmake/Libjpeg.cmake): a change of a submodule is always built, and nothing
# downstream is relinked when nothing changed.

include(ExternalProject)
include(External)

cmake_path(SET UNROUND_ZLIB_SOURCE NORMALIZE "${CMAKE_CURRENT_LIST_DIR}/../extern/zlib-ng")
cmake_path(SET UNROUND_LIBPNG_SOURCE NORMALIZE "${CMAKE_CURRENT_LIST_DIR}/../extern/libpng")
foreach(source IN ITEMS "${UNROUND_ZLIB_SOURCE}" "${UNROUND_LIBPNG_SOURCE}")
  if(NOT EXISTS "${source}/CMakeLists.txt")
    message(FATAL_ERROR
      "${source} is empty. Check out the submodules: "
      "git submodule update --init --depth 1")
  endif()
endforeach()

set(UNROUND_ZLIB_PREFIX "${CMAKE_BINARY_DIR}/zlib-ng")
set(UNROUND_ZLIB_INSTALL "${UNROUND_ZLIB_PREFIX}/install")
set(UNROUND_LIBPNG_PREFIX "${CMAKE_BINARY_DIR}/libpng")
set(UNROUND_LIBPNG_INSTALL "${UNROUND_LIBPNG_PREFIX}/install")
# The names the libraries give themselves: without a prefix for the toolchains of
# Windows, and libpng with a suffix that tells its static library from the import
# library of its DLL.
if(CMAKE_C_SIMULATE_ID STREQUAL "MSVC")
  set(UNROUND_ZLIB_LIBRARY "${UNROUND_ZLIB_INSTALL}/lib/z.lib")
  set(UNROUND_LIBPNG_LIBRARY "${UNROUND_LIBPNG_INSTALL}/lib/libpng16_static.lib")
else()
  set(UNROUND_ZLIB_LIBRARY "${UNROUND_ZLIB_INSTALL}/lib/libz.a")
  set(UNROUND_LIBPNG_LIBRARY "${UNROUND_LIBPNG_INSTALL}/lib/libpng16.a")
endif()

unround_external_cache_args(UNROUND_ZLIB_CACHE_ARGS "${UNROUND_ZLIB_INSTALL}")
list(APPEND UNROUND_ZLIB_CACHE_ARGS
  "-DBUILD_SHARED_LIBS:BOOL=OFF"
  "-DZLIB_COMPAT:BOOL=ON"
  "-DWITH_GZFILEOP:BOOL=OFF"
  "-DWITH_OPTIM:BOOL=ON"
  "-DWITH_RUNTIME_CPU_DETECTION:BOOL=ON"
  "-DWITH_NATIVE_INSTRUCTIONS:BOOL=OFF"
  "-DBUILD_TESTING:BOOL=OFF"
  "-DINSTALL_UTILS:BOOL=OFF")

ExternalProject_Add(zlib-ng
  SOURCE_DIR "${UNROUND_ZLIB_SOURCE}"
  PREFIX "${UNROUND_ZLIB_PREFIX}"
  INSTALL_DIR "${UNROUND_ZLIB_INSTALL}"
  CMAKE_CACHE_ARGS ${UNROUND_ZLIB_CACHE_ARGS}
  BUILD_ALWAYS ON
  BUILD_BYPRODUCTS "${UNROUND_ZLIB_LIBRARY}"
  LOG_CONFIGURE ON
  LOG_BUILD ON
  LOG_INSTALL ON
  LOG_OUTPUT_ON_FAILURE ON)

unround_external_cache_args(UNROUND_LIBPNG_CACHE_ARGS "${UNROUND_LIBPNG_INSTALL}")
list(APPEND UNROUND_LIBPNG_CACHE_ARGS
  "-DPNG_SHARED:BOOL=OFF"
  "-DPNG_STATIC:BOOL=ON"
  "-DPNG_FRAMEWORK:BOOL=OFF"
  "-DPNG_TESTS:BOOL=OFF"
  "-DPNG_TOOLS:BOOL=OFF"
  "-DPNG_LIBCONF_HEADER:FILEPATH=${UNROUND_LIBPNG_SOURCE}/scripts/pnglibconf.h.prebuilt"
  "-DZLIB_ROOT:PATH=${UNROUND_ZLIB_INSTALL}"
  "-DZLIB_LIBRARY:FILEPATH=${UNROUND_ZLIB_LIBRARY}"
  "-DZLIB_INCLUDE_DIR:PATH=${UNROUND_ZLIB_INSTALL}/include")

ExternalProject_Add(libpng
  SOURCE_DIR "${UNROUND_LIBPNG_SOURCE}"
  PREFIX "${UNROUND_LIBPNG_PREFIX}"
  INSTALL_DIR "${UNROUND_LIBPNG_INSTALL}"
  DEPENDS zlib-ng
  CMAKE_CACHE_ARGS ${UNROUND_LIBPNG_CACHE_ARGS}
  BUILD_ALWAYS ON
  BUILD_BYPRODUCTS "${UNROUND_LIBPNG_LIBRARY}"
  LOG_CONFIGURE ON
  LOG_BUILD ON
  LOG_INSTALL ON
  LOG_OUTPUT_ON_FAILURE ON)

# An imported target's include directory has to exist when the build is
# generated, before the sub-build has installed anything into it.
file(MAKE_DIRECTORY "${UNROUND_ZLIB_INSTALL}/include" "${UNROUND_LIBPNG_INSTALL}/include")
add_library(unround_zlib STATIC IMPORTED GLOBAL)
set_target_properties(unround_zlib PROPERTIES
  IMPORTED_LOCATION "${UNROUND_ZLIB_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${UNROUND_ZLIB_INSTALL}/include")
add_dependencies(unround_zlib zlib-ng)
add_library(unround::zlib ALIAS unround_zlib)

add_library(unround_png STATIC IMPORTED GLOBAL)
set_target_properties(unround_png PROPERTIES
  IMPORTED_LOCATION "${UNROUND_LIBPNG_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${UNROUND_LIBPNG_INSTALL}/include"
  INTERFACE_LINK_LIBRARIES "unround::zlib;$<$<PLATFORM_ID:Linux>:m>")
add_dependencies(unround_png libpng)
add_library(unround::png ALIAS unround_png)
