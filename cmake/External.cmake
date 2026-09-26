# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
#
# What every library of extern/ that is built as a project of its own
# (ExternalProject: cmake/Libjpeg.cmake, cmake/Libpng.cmake) is given:
#
#   - This build's toolchain: its C compiler, archiver and ranlib, and LLD as the
#     linker of the programs that its configuration builds to test the compiler.
#   - Release, whatever this build's configuration: the libraries are not what
#     this project debugs.
#   - Position-independent code with hidden symbols, so that the libraries can go
#     into the shared C layer that Python loads without exporting their functions
#     from it: a process that also loads another libjpeg or libpng (Pillow's, say)
#     then keeps the two apart.
#   - The C runtime of this build on Windows: the C layer frees memory that the
#     libraries allocate, which works only when both use one runtime.
#   - The sanitizers of this build, but for undefined, which is left to the
#     libraries' own testing: memory they write for the C layer is then checked
#     where it is written. A fuzzing build adds the coverage that libFuzzer steers
#     by.
#   - On Apple's systems, the deployment target, the architectures and the SDK of
#     this build.
#   - An installation into INSTALL, from which imported targets take the library
#     and its headers.
#
# unround_external_cache_args(<variable> <install>) sets <variable> to those as
# the CMAKE_CACHE_ARGS of ExternalProject_Add.

function(unround_external_cache_args variable install)
  set(flags "")
  if(UNROUND_SANITIZER_LIST)
    set(sanitizers ${UNROUND_SANITIZER_LIST})
    list(REMOVE_ITEM sanitizers undefined)
    if(sanitizers)
      list(JOIN sanitizers "," sanitizers)
      string(APPEND flags " -fsanitize=${sanitizers} -fno-omit-frame-pointer")
    endif()
  endif()
  if(UNROUND_BUILD_FUZZERS)
    string(APPEND flags " -fsanitize=fuzzer-no-link")
  endif()
  string(STRIP "${flags}" flags)

  set(arguments
    "-DCMAKE_BUILD_TYPE:STRING=Release"
    "-DCMAKE_INSTALL_PREFIX:PATH=${install}"
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
    "-DCMAKE_C_FLAGS:STRING=${flags}"
    "-DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON"
    "-DCMAKE_C_VISIBILITY_PRESET:STRING=hidden"
    "-DCMAKE_MSVC_RUNTIME_LIBRARY:STRING=${CMAKE_MSVC_RUNTIME_LIBRARY}")
  if(APPLE)
    list(APPEND arguments
      "-DCMAKE_OSX_DEPLOYMENT_TARGET:STRING=${CMAKE_OSX_DEPLOYMENT_TARGET}"
      "-DCMAKE_OSX_ARCHITECTURES:STRING=${CMAKE_OSX_ARCHITECTURES}"
      "-DCMAKE_OSX_SYSROOT:PATH=${CMAKE_OSX_SYSROOT}")
  endif()
  set(${variable} "${arguments}" PARENT_SCOPE)
endfunction()
