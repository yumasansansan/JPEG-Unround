# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
#
#   cmake -DLIBRARY=<shared library> -DHEADER=<unround/jpegio.h>
#         -DNM=<llvm-nm> -DOBJDUMP=<llvm-objdump> -P check_exports.cmake
#
# Checks that the shared C layer exports the functions that its header declares
# with UNROUND_JPEGIO_API, and nothing else. libjpeg-turbo, libpng and zlib-ng
# are linked into it, and their functions have to stay inside: a process that
# loads the library may have another libjpeg, libpng or zlib loaded as well, and
# the two must not take each other's functions. The exports are read from the
# export table of a DLL, from the dynamic symbols of an ELF object, and from the
# external symbols of a Mach-O one, whose names begin with an underscore.

foreach(variable IN ITEMS LIBRARY HEADER NM OBJDUMP)
  if(NOT DEFINED ${variable})
    message(FATAL_ERROR "check_exports.cmake needs -D${variable}=...")
  endif()
endforeach()

file(READ "${HEADER}" header)
string(REGEX MATCHALL "UNROUND_JPEGIO_API[^;(]*[ \n*]unround_jpegio_[a-z_]+\\(" declarations "${header}")
set(expected "")
foreach(declaration IN LISTS declarations)
  string(REGEX MATCH "unround_jpegio_[a-z_]+\\($" name "${declaration}")
  string(REGEX REPLACE "\\($" "" name "${name}")
  list(APPEND expected "${name}")
endforeach()
list(SORT expected)
if(NOT expected)
  message(FATAL_ERROR "${HEADER} declares no function with UNROUND_JPEGIO_API")
endif()

set(exported "")
if(LIBRARY MATCHES "\\.dll$")
  execute_process(COMMAND "${OBJDUMP}" --private-headers "${LIBRARY}"
    OUTPUT_VARIABLE dump RESULT_VARIABLE status)
  if(NOT status EQUAL 0)
    message(FATAL_ERROR "${OBJDUMP} cannot read ${LIBRARY}")
  endif()
  string(FIND "${dump}" "Export Table:" start)
  if(start LESS 0)
    message(FATAL_ERROR "${LIBRARY} has no export table")
  endif()
  string(SUBSTRING "${dump}" ${start} -1 dump)
  string(REGEX MATCHALL "\n +[0-9]+ +0x[0-9a-fA-F]+ +[A-Za-z_][A-Za-z_0-9]*" rows "${dump}")
  foreach(row IN LISTS rows)
    string(REGEX MATCH "[A-Za-z_][A-Za-z_0-9]*$" name "${row}")
    list(APPEND exported "${name}")
  endforeach()
else()
  if(LIBRARY MATCHES "\\.dylib$")
    set(arguments --extern-only --defined-only)
  else()
    set(arguments --dynamic --extern-only --defined-only)
  endif()
  execute_process(COMMAND "${NM}" ${arguments} "${LIBRARY}"
    OUTPUT_VARIABLE dump RESULT_VARIABLE status)
  if(NOT status EQUAL 0)
    message(FATAL_ERROR "${NM} cannot read ${LIBRARY}")
  endif()
  string(REGEX MATCHALL "[^\n]+" rows "${dump}")
  foreach(row IN LISTS rows)
    string(REGEX MATCH "[^ ]+$" name "${row}")
    if(LIBRARY MATCHES "\\.dylib$")
      string(REGEX REPLACE "^_" "" name "${name}")
    endif()
    list(APPEND exported "${name}")
  endforeach()
endif()
list(SORT exported)

if(NOT exported STREQUAL expected)
  list(JOIN expected "\n  " expected_text)
  list(JOIN exported "\n  " exported_text)
  message(FATAL_ERROR
    "${LIBRARY} exports\n  ${exported_text}\nand its header declares\n  ${expected_text}")
endif()
list(LENGTH exported count)
message(STATUS "${LIBRARY} exports the ${count} functions its header declares, and nothing else")
