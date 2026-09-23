# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
#
# libjpeg-turbo, from the submodule extern/libjpeg-turbo, as the static library
# unround::libjpeg. Its build refuses to be a subdirectory of another, so it is
# built as a project of its own (ExternalProject) with this build's toolchain,
# and installed into the build directory, from which the imported target takes
# the library and the headers. What is chosen for it:
#
#   - Release, whatever this build's configuration: libjpeg-turbo is not what
#     this project debugs.
#   - Static only, with the C runtime of this build on Windows: the DLL of its
#     release build (WITH_CRT_DLL, which libjpeg-turbo would otherwise leave
#     off). The C layer frees memory that libjpeg allocates (the ICC profile),
#     which works only when both use one runtime.
#   - Position-independent code with hidden symbols, so that the library can go
#     into the shared C layer that Python loads without exporting libjpeg's own
#     functions from it: a process that also loads another libjpeg (Pillow's,
#     say) then keeps the two apart.
#   - No SIMD: the SIMD code needs NASM on x86-64, which the toolchain does not
#     have, and it adds nothing this project needs -- the coefficients are read
#     by the entropy decoder, which has no SIMD, and the planes are decoded for
#     checking, not for speed.
#   - No TurboJPEG API and no regression tests. The command-line tools (cjpeg,
#     djpeg, jpegtran) are built: the test data is made with them.
#   - The sanitizers of this build, but for undefined, which is left to
#     libjpeg-turbo's own testing: memory libjpeg writes for the C layer is then
#     checked where it is written. A fuzzing build adds the coverage that
#     libFuzzer steers by.
#
# The sub-build runs on every build (BUILD_ALWAYS), which costs a check of an
# up-to-date tree: that way a change of the submodule is always built, and
# nothing downstream is relinked when nothing changed.

include(ExternalProject)

set(UNROUND_LIBJPEG_SOURCE "${PROJECT_SOURCE_DIR}/extern/libjpeg-turbo")
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

set(UNROUND_LIBJPEG_C_FLAGS "")
if(UNROUND_SANITIZER_LIST)
  set(sanitizers ${UNROUND_SANITIZER_LIST})
  list(REMOVE_ITEM sanitizers undefined)
  if(sanitizers)
    list(JOIN sanitizers "," sanitizers)
    string(APPEND UNROUND_LIBJPEG_C_FLAGS " -fsanitize=${sanitizers} -fno-omit-frame-pointer")
  endif()
endif()
if(UNROUND_BUILD_FUZZERS)
  string(APPEND UNROUND_LIBJPEG_C_FLAGS " -fsanitize=fuzzer-no-link")
endif()
string(STRIP "${UNROUND_LIBJPEG_C_FLAGS}" UNROUND_LIBJPEG_C_FLAGS)

set(UNROUND_LIBJPEG_CACHE_ARGS
  "-DCMAKE_BUILD_TYPE:STRING=Release"
  "-DCMAKE_INSTALL_PREFIX:PATH=${UNROUND_LIBJPEG_INSTALL}"
  "-DCMAKE_INSTALL_LIBDIR:PATH=lib"
  "-DCMAKE_INSTALL_BINDIR:PATH=bin"
  "-DCMAKE_INSTALL_INCLUDEDIR:PATH=include"
  "-DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}"
  "-DCMAKE_AR:FILEPATH=${CMAKE_AR}"
  "-DCMAKE_RANLIB:FILEPATH=${CMAKE_RANLIB}"
  "-DCMAKE_C_COMPILER_AR:FILEPATH=${CMAKE_C_COMPILER_AR}"
  "-DCMAKE_C_COMPILER_RANLIB:FILEPATH=${CMAKE_C_COMPILER_RANLIB}"
  "-DCMAKE_LINKER_TYPE:STRING=LLD"
  "-DCMAKE_EXE_LINKER_FLAGS:STRING=-fuse-ld=lld"
  "-DCMAKE_C_FLAGS:STRING=${UNROUND_LIBJPEG_C_FLAGS}"
  "-DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON"
  "-DCMAKE_C_VISIBILITY_PRESET:STRING=hidden"
  "-DCMAKE_MSVC_RUNTIME_LIBRARY:STRING=${CMAKE_MSVC_RUNTIME_LIBRARY}"
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
if(APPLE)
  list(APPEND UNROUND_LIBJPEG_CACHE_ARGS
    "-DCMAKE_OSX_DEPLOYMENT_TARGET:STRING=${CMAKE_OSX_DEPLOYMENT_TARGET}"
    "-DCMAKE_OSX_ARCHITECTURES:STRING=${CMAKE_OSX_ARCHITECTURES}"
    "-DCMAKE_OSX_SYSROOT:PATH=${CMAKE_OSX_SYSROOT}")
endif()

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
