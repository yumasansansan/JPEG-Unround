# SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
#
#   cmake -DDJPEG=<djpeg> -DPNG=<files> -DJPEG=<files> -DWIDTH=<w> -DHEIGHT=<h>
#         -P check-outputs.cmake
#
# Checks what the tools wrote in their tests (CMakeLists.txt): that each PNG
# file is one, of the size of the picture, as its header says, and that djpeg
# decodes each JPEG file into a picture of that size.

foreach(variable IN ITEMS DJPEG PNG JPEG WIDTH HEIGHT)
  if(NOT DEFINED ${variable})
    message(FATAL_ERROR "check-outputs.cmake needs -D${variable}=...")
  endif()
endforeach()

set(failures 0)

foreach(file IN LISTS PNG)
  if(NOT EXISTS "${file}")
    message(SEND_ERROR "${file} was not written")
    math(EXPR failures "${failures} + 1")
    continue()
  endif()
  # The signature, and the IHDR chunk that has to follow it: its length (13),
  # its type, then the width and the height, big-endian.
  file(READ "${file}" header LIMIT 24 HEX)
  string(SUBSTRING "${header}" 0 32 start)
  if(NOT start STREQUAL "89504e470d0a1a0a0000000d49484452")
    message(SEND_ERROR "${file} is not a PNG file: it starts ${header}")
    math(EXPR failures "${failures} + 1")
    continue()
  endif()
  string(SUBSTRING "${header}" 32 8 width)
  string(SUBSTRING "${header}" 40 8 height)
  math(EXPR width "0x${width}")
  math(EXPR height "0x${height}")
  if(NOT width EQUAL WIDTH OR NOT height EQUAL HEIGHT)
    message(SEND_ERROR "${file} is ${width} by ${height} pixels, not ${WIDTH} by ${HEIGHT}")
    math(EXPR failures "${failures} + 1")
    continue()
  endif()
  message(STATUS "${file}: PNG, ${width} by ${height}")
endforeach()

foreach(file IN LISTS JPEG)
  if(NOT EXISTS "${file}")
    message(SEND_ERROR "${file} was not written")
    math(EXPR failures "${failures} + 1")
    continue()
  endif()
  execute_process(COMMAND "${DJPEG}" -outfile "${file}.pnm" "${file}"
    RESULT_VARIABLE result ERROR_VARIABLE errors)
  if(NOT result EQUAL 0 OR errors)
    message(SEND_ERROR "djpeg does not decode ${file} cleanly (${result}): ${errors}")
    math(EXPR failures "${failures} + 1")
    continue()
  endif()
  file(READ "${file}.pnm" header LIMIT 32)
  if(NOT header MATCHES "^P[56]\n([0-9]+) ([0-9]+)\n")
    message(SEND_ERROR "djpeg decodes ${file} into something other than PNM")
    math(EXPR failures "${failures} + 1")
    continue()
  endif()
  if(NOT CMAKE_MATCH_1 EQUAL WIDTH OR NOT CMAKE_MATCH_2 EQUAL HEIGHT)
    message(SEND_ERROR
      "${file} is ${CMAKE_MATCH_1} by ${CMAKE_MATCH_2} pixels, not ${WIDTH} by ${HEIGHT}")
    math(EXPR failures "${failures} + 1")
    continue()
  endif()
  message(STATUS "${file}: JPEG, ${CMAKE_MATCH_1} by ${CMAKE_MATCH_2}")
endforeach()

if(failures GREATER 0)
  message(FATAL_ERROR "${failures} of the outputs are not as they should be")
endif()
