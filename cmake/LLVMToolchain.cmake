# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Checks, when the build is configured and so before anything is built, that
# the toolchain the build runs is LLVM's, and of one version: the C and C++
# compilers, the LLD that Clang runs for -fuse-ld=lld, and the archivers. A tool
# that reports its version has to report the compilers' version exactly. When
# the environment variable UNROUND_LLVM_MAJOR is set, as ci/setup.sh sets it in
# CI, the compilers have to be of that major version. The presets name the
# tools without a version, and this check decides what the build accepts.
#
# The C++ library is the one part that is not LLVM's everywhere: it is the
# system's -- libstdc++ on Linux, the libc++ of the SDK on macOS, the MSVC STL on
# Windows -- and a program asks the system for it when it is loaded, so it is
# recorded with its version and checked to be there to link, rather than
# required to be of a version of ours.
#
# Each tool that passes is written to llvm-toolchain.txt in the build directory,
# a line of its part, its version and its program, separated by tabs, which
# ci/build.sh shows in the log and the summary of a CI run.
#
# Included after project().

set(UNROUND_LLVM_VERSION "${CMAKE_CXX_COMPILER_VERSION}")
get_filename_component(UNROUND_LLVM_BIN "${CMAKE_CXX_COMPILER}" DIRECTORY)
file(REAL_PATH "${UNROUND_LLVM_BIN}" UNROUND_LLVM_BIN)

set(UNROUND_LLVM_TOOLCHAIN_FILE "${CMAKE_BINARY_DIR}/llvm-toolchain.txt")
file(WRITE "${UNROUND_LLVM_TOOLCHAIN_FILE}" "")

if(NOT CMAKE_C_COMPILER_VERSION VERSION_EQUAL CMAKE_CXX_COMPILER_VERSION)
  message(FATAL_ERROR
    "The C compiler is of LLVM ${CMAKE_C_COMPILER_VERSION} and the C++ compiler of LLVM "
    "${CMAKE_CXX_COMPILER_VERSION}; they have to be of one version.")
endif()

if(DEFINED ENV{UNROUND_LLVM_MAJOR})
  string(REGEX MATCH "^[0-9]+" UNROUND_LLVM_MAJOR "${UNROUND_LLVM_VERSION}")
  if(NOT UNROUND_LLVM_MAJOR STREQUAL "$ENV{UNROUND_LLVM_MAJOR}")
    message(FATAL_ERROR
      "The compilers are of LLVM ${UNROUND_LLVM_VERSION}, but UNROUND_LLVM_MAJOR asks for "
      "LLVM $ENV{UNROUND_LLVM_MAJOR}.")
  endif()
  message(STATUS "LLVM ${UNROUND_LLVM_MAJOR}, the major version that UNROUND_LLVM_MAJOR asks for")
elseif(UNROUND_LLVM_VERSION VERSION_LESS 23)
  message(FATAL_ERROR
    "The compilers are of LLVM ${UNROUND_LLVM_VERSION}; JPEG-Unround is built with LLVM 23 "
    "(on Ubuntu, put /usr/lib/llvm-23/bin first on PATH: the clang there is 23, the one of "
    "the distribution older).")
endif()

# Checks the program that plays a part of the toolchain: its file name has to
# match the pattern, and its version the compilers' version.
function(unround_check_llvm_tool part program pattern)
  if(NOT program)
    message(FATAL_ERROR "No program is set for the ${part}.")
  endif()
  if(NOT IS_ABSOLUTE "${program}")
    find_program(UNROUND_LLVM_TOOL_PATH NAMES "${program}" NO_CACHE)
    if(NOT UNROUND_LLVM_TOOL_PATH)
      message(FATAL_ERROR "The ${part}, ${program}, is not found.")
    endif()
    set(program "${UNROUND_LLVM_TOOL_PATH}")
  endif()

  get_filename_component(name "${program}" NAME)
  if(NOT name MATCHES "${pattern}")
    message(FATAL_ERROR "The ${part} is ${program}, which is not a tool of LLVM.")
  endif()

  execute_process(COMMAND "${program}" --version
    OUTPUT_VARIABLE output ERROR_VARIABLE output TIMEOUT 60)
  if(NOT output MATCHES "(clang version|LLVM version|LLD) ([0-9]+\\.[0-9]+\\.[0-9]+)")
    message(FATAL_ERROR "The ${part}, ${program}, does not say which version of LLVM it is:\n${output}")
  endif()
  set(version "${CMAKE_MATCH_2}")
  if(NOT version VERSION_EQUAL UNROUND_LLVM_VERSION)
    message(FATAL_ERROR
      "The ${part} is ${program}, of LLVM ${version}, but the compilers are of "
      "LLVM ${UNROUND_LLVM_VERSION}.")
  endif()
  message(STATUS "  ${part}: ${program} (${version})")
  file(APPEND "${UNROUND_LLVM_TOOLCHAIN_FILE}" "${part}\t${version}\t${program}\n")
endfunction()

message(STATUS "LLVM toolchain ${UNROUND_LLVM_VERSION}")
unround_check_llvm_tool("C compiler" "${CMAKE_C_COMPILER}" "^clang(-[0-9]+)?(\\.exe)?$")
unround_check_llvm_tool("C++ compiler" "${CMAKE_CXX_COMPILER}" "^clang(\\+\\+)?(-[0-9]+)?(\\.exe)?$")

# The LLD that Clang finds for -fuse-ld=lld: lld-link for Windows, ld64.lld for
# Apple's systems, ld.lld elsewhere.
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  set(UNROUND_LLD_NAME lld-link)
elseif(APPLE)
  set(UNROUND_LLD_NAME ld64.lld)
else()
  set(UNROUND_LLD_NAME ld.lld)
endif()
execute_process(COMMAND "${CMAKE_CXX_COMPILER}" "--print-prog-name=${UNROUND_LLD_NAME}"
  OUTPUT_VARIABLE UNROUND_LLD OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT IS_ABSOLUTE "${UNROUND_LLD}")
  message(FATAL_ERROR "Clang does not find ${UNROUND_LLD_NAME}, the LLD it runs for -fuse-ld=lld.")
endif()
unround_check_llvm_tool("linker" "${UNROUND_LLD}" "^(lld-link|ld64\\.lld|ld\\.lld)(-[0-9]+)?(\\.exe)?$")

foreach(UNROUND_LLVM_ARCHIVER IN ITEMS
    CMAKE_AR CMAKE_RANLIB
    CMAKE_C_COMPILER_AR CMAKE_C_COMPILER_RANLIB
    CMAKE_CXX_COMPILER_AR CMAKE_CXX_COMPILER_RANLIB)
  unround_check_llvm_tool("${UNROUND_LLVM_ARCHIVER}" "${${UNROUND_LLVM_ARCHIVER}}"
    "^llvm-(ar|ranlib)(-[0-9]+)?(\\.exe)?$")
endforeach()

# Checks the C++ library that the compilers are given, and records which one it
# is. One program says all of it. It is compiled and linked as CMake compiles and
# links anything, so that it is given what this build's compilations are given --
# the sysroot of the SDK on macOS, where the C++ headers are the SDK's and not the
# toolchain's, among the rest -- and what it says about itself, it says in a
# #pragma message, which comes back in the output. Compiling it names the library
# and the version of the headers on the include path; linking it says that the
# library is there to link and not only to include.
function(unround_check_cxx_library)
  set(check "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/cxx-library")
  file(WRITE "${check}/main.cpp"
    "#include <version>\n"
    "#include <stdexcept>\n"
    "#include <string>\n"
    "#define UNROUND_TEXT_OF(x) #x\n"
    "#define UNROUND_TEXT(x) UNROUND_TEXT_OF(x)\n"
    "#if defined(_LIBCPP_VERSION)\n"
    "#  pragma message(\"UNROUND_CXX_LIBRARY libc++ \" UNROUND_TEXT(_LIBCPP_VERSION) \" 0\")\n"
    "#elif defined(__GLIBCXX__)\n"
    "#  pragma message(\"UNROUND_CXX_LIBRARY libstdc++ \" UNROUND_TEXT(_GLIBCXX_RELEASE) \" \" UNROUND_TEXT(__GLIBCXX__))\n"
    "#elif defined(_MSVC_STL_UPDATE)\n"
    "#  pragma message(\"UNROUND_CXX_LIBRARY MSVC-STL \" UNROUND_TEXT(_MSVC_STL_VERSION) \" \" UNROUND_TEXT(_MSVC_STL_UPDATE))\n"
    "#else\n"
    "#  pragma message(\"UNROUND_CXX_LIBRARY none 0 0\")\n"
    "#endif\n"
    "int main(void) {\n"
    "  try { throw std::runtime_error{std::string{\"x\"}}; }\n"
    "  catch (const std::exception& e) { return e.what()[0] == 'x' ? 0 : 1; }\n"
    "}\n")
  try_compile(UNROUND_CXX_LIBRARY_LINKS "${check}/build" SOURCES "${check}/main.cpp"
    CMAKE_FLAGS "-DCMAKE_CXX_STANDARD=26" "-DCMAKE_LINKER_TYPE=LLD"
    OUTPUT_VARIABLE output)
  if(NOT UNROUND_CXX_LIBRARY_LINKS)
    message(FATAL_ERROR
      "A C++ program cannot be compiled and linked with the C++ library of this toolchain. On "
      "Ubuntu the library comes with libstdc++-<version>-dev, and on macOS the headers are the "
      "SDK's (ci/setup.sh):\n${output}")
  endif()
  if(NOT output MATCHES "UNROUND_CXX_LIBRARY ([A-Za-z+-]+) ([0-9]+) ([0-9]+)")
    message(FATAL_ERROR
      "A C++ program compiles and links, and the compiler did not say which C++ library its "
      "headers are:\n${output}")
  endif()
  set(library "${CMAKE_MATCH_1}")
  set(version "${CMAKE_MATCH_2}")
  set(date "${CMAKE_MATCH_3}")
  if(library STREQUAL "none")
    message(FATAL_ERROR
      "The headers on the include path are of no C++ library that this build knows: not "
      "libstdc++, not libc++, not the MSVC STL.")
  endif()
  if(library STREQUAL "MSVC-STL")
    set(library "MSVC STL")
  endif()
  set(named "${library} ${version}")
  set(record "${library}")
  if(NOT date EQUAL 0)
    string(APPEND named " (${date})")
    string(APPEND record " ${date}")
  endif()
  message(STATUS "  C++ library: ${named}")
  file(APPEND "${UNROUND_LLVM_TOOLCHAIN_FILE}" "C++ library\t${version}\t${record}\n")
  set(UNROUND_CXX_LIBRARY "${library}" PARENT_SCOPE)
endfunction()

unround_check_cxx_library()
