# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Probes C++ modules the way the build would use them: through CMake and Ninja,
# with clang-scan-deps finding what each file imports. Three programs:
#   probe_named       a named module of the project's own, imported by a program
#   probe_import_std  import std; with the system's standard library
#   probe_mixed       #include of standard headers and import std; in one file
#
# The std module is compiled here as a module library of this project, from the
# source its standard library names: the MSVC STL's modules/std.ixx, or the
# source that libstdc++.modules.json or libc++.modules.json points to. On macOS
# that has to be the manifest of the SDK's libc++, which the build compiles
# against, and not the LLVM toolchain's: that one names LLVM's own std.cppm,
# which needs headers newer than the SDK's. When the library names no std
# module, only probe_named is built, and the configuration says why.
#
# CMake's own import std support is not used: it is experimental, its gate
# value changes with each CMake release, it knows no MSVC STL under Clang, and
# it builds the module with GNU extensions unless CMAKE_CXX_EXTENSIONS=OFF is
# given before project(). With UNROUND_PROBE_IMPORT_STD=OFF only probe_named is
# built.
cmake_minimum_required(VERSION 3.30)
project(unround_module_probe LANGUAGES CXX)

option(UNROUND_PROBE_IMPORT_STD "Build the std module and the programs that import it" ON)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(probe_named_module STATIC)
target_sources(probe_named_module PUBLIC FILE_SET CXX_MODULES FILES named.cppm)
add_executable(probe_named named_main.cpp)
target_link_libraries(probe_named PRIVATE probe_named_module)

if(NOT UNROUND_PROBE_IMPORT_STD)
  return()
endif()

# Reads the std module's source, and the include directories it asks for, from
# a manifest in the format libc++ and libstdc++ share.
function(unround_probe_read_manifest manifest out_source out_includes)
  file(READ "${manifest}" manifest_text)
  # The paths in a manifest are relative to the lib directory of the LLVM or GCC
  # installation. apt.llvm.org's packages keep the file itself in the multiarch
  # directory and link to it from /usr/lib/llvm-N/lib, where the compiler does
  # not look, so the lib directory beside the compiler's is tried as well.
  cmake_path(GET manifest PARENT_PATH given_dir)
  file(REAL_PATH "${manifest}" real_manifest)
  cmake_path(GET real_manifest PARENT_PATH real_dir)
  file(REAL_PATH "${CMAKE_CXX_COMPILER}" compiler)
  cmake_path(GET compiler PARENT_PATH compiler_dir)
  cmake_path(GET compiler_dir PARENT_PATH prefix)
  set(bases "${given_dir}" "${real_dir}" "${prefix}/lib")
  set(source "")
  set(includes "")
  string(JSON count LENGTH "${manifest_text}" modules)
  if(count GREATER 0)
    math(EXPR last "${count} - 1")
    foreach(i RANGE ${last})
      string(JSON name GET "${manifest_text}" modules ${i} logical-name)
      if(name STREQUAL "std")
        string(JSON relative GET "${manifest_text}" modules ${i} source-path)
        set(manifest_dir "${given_dir}")
        foreach(base IN LISTS bases)
          if(EXISTS "${base}/${relative}")
            set(manifest_dir "${base}")
            break()
          endif()
        endforeach()
        cmake_path(ABSOLUTE_PATH relative BASE_DIRECTORY "${manifest_dir}" NORMALIZE OUTPUT_VARIABLE source)
        string(JSON dirs ERROR_VARIABLE no_dirs GET "${manifest_text}" modules ${i} local-arguments system-include-directories)
        if(NOT no_dirs)
          string(JSON n LENGTH "${dirs}")
          if(n GREATER 0)
            math(EXPR n_last "${n} - 1")
            foreach(j RANGE ${n_last})
              string(JSON dir GET "${dirs}" ${j})
              cmake_path(ABSOLUTE_PATH dir BASE_DIRECTORY "${manifest_dir}" NORMALIZE)
              list(APPEND includes "${dir}")
            endforeach()
          endif()
        endif()
      endif()
    endforeach()
  endif()
  set(${out_source} "${source}" PARENT_SCOPE)
  set(${out_includes} "${includes}" PARENT_SCOPE)
endfunction()

set(std_source "")
set(std_options -Wno-reserved-module-identifier)
set(std_includes "")
set(std_missing "")
if(CMAKE_CXX_SIMULATE_ID STREQUAL "MSVC")
  # CMake records no implicit include directories for Clang with the MSVC ABI,
  # so the MSVC tools directory is read from what the driver would run.
  execute_process(COMMAND "${CMAKE_CXX_COMPILER}" "-###" -x c++ -c -
    INPUT_FILE "${CMAKE_CURRENT_LIST_FILE}" OUTPUT_VARIABLE driver ERROR_VARIABLE driver)
  string(REPLACE "\\\\" "/" driver "${driver}")
  if(driver MATCHES "\"([^\"]*/VC/Tools/MSVC/[0-9.]+)/include\"")
    set(std_source "${CMAKE_MATCH_1}/modules/std.ixx")
  endif()
  list(APPEND std_options -Wno-include-angled-in-module-purview)
  set(std_missing "the MSVC STL's modules/std.ixx was not found")
elseif(APPLE)
  execute_process(COMMAND xcode-select -p OUTPUT_VARIABLE developer OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(candidates
    "${developer}/Toolchains/XcodeDefault.xctoolchain/usr/lib/libc++.modules.json"
    "${CMAKE_OSX_SYSROOT}/usr/lib/libc++.modules.json")
  foreach(manifest IN LISTS candidates)
    if(EXISTS "${manifest}")
      message(STATUS "std module manifest: ${manifest}")
      unround_probe_read_manifest("${manifest}" std_source std_includes)
      break()
    endif()
  endforeach()
  set(std_missing "the SDK's libc++ has no libc++.modules.json (looked for: ${candidates})")
else()
  # The manifest of the library the compiler is given, which CMake has detected
  # (a system can have both installed, as Ubuntu with LLVM's packages does).
  message(STATUS "C++ library: ${CMAKE_CXX_STANDARD_LIBRARY}")
  if(CMAKE_CXX_STANDARD_LIBRARY MATCHES "^(libc\\+\\+|libstdc\\+\\+)$")
    execute_process(COMMAND "${CMAKE_CXX_COMPILER}" ${CMAKE_CXX_FLAGS}
      "-print-file-name=${CMAKE_CXX_STANDARD_LIBRARY}.modules.json"
      OUTPUT_VARIABLE manifest OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(IS_ABSOLUTE "${manifest}" AND EXISTS "${manifest}")
      message(STATUS "std module manifest: ${manifest}")
      unround_probe_read_manifest("${manifest}" std_source std_includes)
    endif()
  endif()
  set(std_missing "${CMAKE_CXX_STANDARD_LIBRARY} names no std module source")
endif()
if(NOT std_source OR NOT EXISTS "${std_source}")
  message(WARNING "import std is not probed: ${std_missing}.")
  return()
endif()
message(STATUS "std module source: ${std_source}")

cmake_path(GET std_source PARENT_PATH std_dir)
add_library(probe_std_module STATIC)
target_sources(probe_std_module PUBLIC FILE_SET CXX_MODULES BASE_DIRS "${std_dir}" FILES "${std_source}")
target_compile_options(probe_std_module PRIVATE ${std_options})
if(std_includes)
  target_include_directories(probe_std_module SYSTEM PRIVATE ${std_includes})
endif()

add_executable(probe_import_std import_std_main.cpp)
add_executable(probe_mixed mixed_main.cpp)
target_link_libraries(probe_import_std PRIVATE probe_std_module)
target_link_libraries(probe_mixed PRIVATE probe_std_module)
