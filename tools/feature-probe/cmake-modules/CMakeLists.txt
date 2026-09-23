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
# source that libstdc++.modules.json or libc++.modules.json points to. CMake's
# own import std support is not used: it is experimental, its gate value changes
# with each CMake release, it knows no MSVC STL under Clang, and it builds the
# module with GNU extensions unless CMAKE_CXX_EXTENSIONS=OFF is given before
# project(). With UNROUND_PROBE_IMPORT_STD=OFF only probe_named is built.
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

set(std_source "")
set(std_options -Wno-reserved-module-identifier)
set(std_includes "")
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
else()
  # The manifest of the library the compiler is given, which CMake has detected
  # (a system can have both installed, as Ubuntu with LLVM's packages does).
  if(NOT CMAKE_CXX_STANDARD_LIBRARY MATCHES "^(libc\\+\\+|libstdc\\+\\+)$")
    message(FATAL_ERROR "The C++ library '${CMAKE_CXX_STANDARD_LIBRARY}' has no module manifest known here.")
  endif()
  message(STATUS "C++ library: ${CMAKE_CXX_STANDARD_LIBRARY}")
  foreach(manifest_name IN ITEMS "${CMAKE_CXX_STANDARD_LIBRARY}.modules.json")
    execute_process(COMMAND "${CMAKE_CXX_COMPILER}" ${CMAKE_CXX_FLAGS} "-print-file-name=${manifest_name}"
      OUTPUT_VARIABLE manifest OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(IS_ABSOLUTE "${manifest}" AND EXISTS "${manifest}")
      message(STATUS "std module manifest: ${manifest}")
      file(READ "${manifest}" manifest_text)
      cmake_path(GET manifest PARENT_PATH manifest_dir)
      string(JSON count LENGTH "${manifest_text}" modules)
      math(EXPR last "${count} - 1")
      foreach(i RANGE ${last})
        string(JSON name GET "${manifest_text}" modules ${i} logical-name)
        if(name STREQUAL "std")
          string(JSON relative GET "${manifest_text}" modules ${i} source-path)
          cmake_path(ABSOLUTE_PATH relative BASE_DIRECTORY "${manifest_dir}" NORMALIZE OUTPUT_VARIABLE std_source)
          string(JSON dirs ERROR_VARIABLE no_dirs GET "${manifest_text}" modules ${i} local-arguments system-include-directories)
          if(NOT no_dirs)
            string(JSON n LENGTH "${dirs}")
            if(n GREATER 0)
              math(EXPR n_last "${n} - 1")
              foreach(j RANGE ${n_last})
                string(JSON dir GET "${dirs}" ${j})
                cmake_path(ABSOLUTE_PATH dir BASE_DIRECTORY "${manifest_dir}" NORMALIZE)
                list(APPEND std_includes "${dir}")
              endforeach()
            endif()
          endif()
        endif()
      endforeach()
      break()
    endif()
  endforeach()
endif()
if(NOT std_source OR NOT EXISTS "${std_source}")
  message(FATAL_ERROR "The standard library names no std module source that exists ('${std_source}').")
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
