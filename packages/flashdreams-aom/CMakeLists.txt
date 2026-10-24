# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)
project(flashdreams_aom LANGUAGES NONE)
include(CheckLanguage)
check_language(C)
check_language(CXX)
if(NOT CMAKE_C_COMPILER OR NOT CMAKE_CXX_COMPILER)
  message(FATAL_ERROR
    "FlashDreams AV1 build is missing a C/C++ compiler. C: ${CMAKE_C_COMPILER}; C++: ${CMAKE_CXX_COMPILER}. "
    "Windows: install Visual Studio Build Tools with Desktop development with C++ and a Windows SDK, then run from an x64 Developer PowerShell. "
    "Linux: install build-essential (or your distribution's C/C++ toolchain). macOS: install Xcode command line tools.")
endif()
enable_language(C CXX)
include(FetchContent)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(ENABLE_DOCS OFF CACHE BOOL "" FORCE)
set(ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(ENABLE_TOOLS OFF CACHE BOOL "" FORCE)
set(CONFIG_AV1_DECODER 1 CACHE STRING "" FORCE)
set(CONFIG_LIBYUV 0 CACHE STRING "" FORCE)
set(CONFIG_WEBM_IO 0 CACHE STRING "" FORCE)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|amd64|x86_64|x86|i.86")
  find_program(YASM_EXECUTABLE NAMES yasm)
  if(NOT YASM_EXECUTABLE AND CMAKE_HOST_WIN32)
    FetchContent_Declare(yasm
      URL https://www.tortall.net/projects/yasm/releases/yasm-1.3.0-win64.exe
      URL_HASH SHA256=d160b1d97266f3f28a71b4420a0ad2cd088a7977c2dd3b25af155652d8d8d91f
      DOWNLOAD_NO_EXTRACT TRUE)
    FetchContent_MakeAvailable(yasm)
    find_program(YASM_EXECUTABLE NAMES yasm yasm-1.3.0-win64 PATHS "${yasm_SOURCE_DIR}" NO_DEFAULT_PATH REQUIRED)
  elseif(NOT YASM_EXECUTABLE)
    message(FATAL_ERROR "Optimized x86 AV1 requires YASM. Install yasm (for example: apt install yasm) and rebuild.")
  endif()
  set(ENABLE_NASM OFF CACHE BOOL "Use YASM for x86 assembly" FORCE)
  set(CMAKE_ASM_NASM_COMPILER "${YASM_EXECUTABLE}" CACHE FILEPATH "AV1 YASM assembler" FORCE)
  message(STATUS "AV1 x86 assembler: ${CMAKE_ASM_NASM_COMPILER}")
endif()

FetchContent_Declare(aom
  URL https://storage.googleapis.com/aom-releases/libaom-3.15.1.tar.gz
  URL_HASH SHA256=8ca0c52746174603500f0adb6f2a215d69c9ca2aab2acb3caa06fb791d8d01bf
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(aom)
set_target_properties(aom PROPERTIES AVIF_LOCAL ON)
target_include_directories(aom INTERFACE
  "$<BUILD_INTERFACE:${aom_SOURCE_DIR}>" "$<BUILD_INTERFACE:${aom_BINARY_DIR}>")
get_directory_property(AOM_HAVE_SSE2 DIRECTORY "${aom_SOURCE_DIR}" DEFINITION HAVE_SSE2)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|amd64|x86_64|x86|i.86" AND NOT AOM_HAVE_SSE2)
  message(FATAL_ERROR "AV1 x86 SIMD is disabled. Remove the old build directory and rebuild without AOM_TARGET_CPU=generic.")
endif()
set(AVIF_CODEC_AOM SYSTEM CACHE STRING "Use the pinned libaom target" FORCE)
set(AVIF_LIBYUV OFF CACHE STRING "" FORCE)
set(AVIF_BUILD_APPS ON CACHE BOOL "" FORCE)
set(AVIF_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(AVIF_JPEG LOCAL CACHE STRING "Build CLI image I/O dependencies" FORCE)
set(AVIF_ZLIBPNG LOCAL CACHE STRING "" FORCE)
# Override libavif's older pin for GHSA-qvg3-h654-xq3j.
FetchContent_Declare(libpng
  URL https://codeload.github.com/pnggroup/libpng/tar.gz/refs/tags/v1.6.59
  URL_HASH SHA256=2540302a1844ad2b2b501977abecfa850f265f97b78f065a712ab4074a89f5b5
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_Declare(avif
  URL https://codeload.github.com/AOMediaCodec/libavif/tar.gz/refs/tags/v1.4.2
  URL_HASH SHA256=2b645287340ba5a631d268b551dc2d72bd73ac33335962dd36dcdb6d8366921d
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(avif)
FetchContent_GetProperties(zlib)
FetchContent_GetProperties(libpng)
install(TARGETS avifenc avifdec RUNTIME DESTINATION flashdreams_aom COMPONENT flashdreams-aom)
install(FILES ${avif_SOURCE_DIR}/LICENSE
  DESTINATION flashdreams_aom/licenses/libavif COMPONENT flashdreams-aom)
install(DIRECTORY ${avif_SOURCE_DIR}/third_party/
  DESTINATION flashdreams_aom/licenses/libavif/third_party COMPONENT flashdreams-aom
  FILES_MATCHING PATTERN "*LICENSE*" PATTERN "*PATENTS*" PATTERN "*COPYING*")
install(FILES ${aom_SOURCE_DIR}/LICENSE ${aom_SOURCE_DIR}/PATENTS
  DESTINATION flashdreams_aom/licenses/libaom COMPONENT flashdreams-aom)
install(FILES ${zlib_SOURCE_DIR}/LICENSE
  DESTINATION flashdreams_aom/licenses/zlib COMPONENT flashdreams-aom)
install(FILES ${libpng_SOURCE_DIR}/LICENSE
  DESTINATION flashdreams_aom/licenses/libpng COMPONENT flashdreams-aom)
install(FILES ${avif_BINARY_DIR}/libjpeg/src/libjpeg/LICENSE.md
              ${avif_BINARY_DIR}/libjpeg/src/libjpeg/README.ijg
  DESTINATION flashdreams_aom/licenses/libjpeg-turbo COMPONENT flashdreams-aom)
# Preserve notices for components incorporated in the upstream codec sources.
install(DIRECTORY ${aom_SOURCE_DIR}/third_party/
  DESTINATION flashdreams_aom/licenses/third_party COMPONENT flashdreams-aom
  FILES_MATCHING PATTERN "*LICENSE*" PATTERN "*PATENTS*" PATTERN "*COPYING*")
