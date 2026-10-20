# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)
project(flashdreams_capi VERSION 0.1.0 LANGUAGES C)
if(NOT Python3_EXECUTABLE)
    if(WIN32)
        set(Python3_EXECUTABLE "${CMAKE_CURRENT_SOURCE_DIR}/../.venv/Scripts/python.exe")
    else()
        set(Python3_EXECUTABLE "${CMAKE_CURRENT_SOURCE_DIR}/../.venv/bin/python")
    endif()
endif()
find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter Development.Embed)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/python-executable.txt" "${Python3_EXECUTABLE}\n")
if(NOT IS_DIRECTORY "$ENV{VULKAN_SDK}")
    message(FATAL_ERROR "Install the Vulkan SDK with SDL3 and set VULKAN_SDK to its installation directory. No dependencies are downloaded.")
endif()
find_package(Vulkan REQUIRED)
find_package(SDL3 CONFIG REQUIRED COMPONENTS SDL3-shared
    PATHS "$ENV{VULKAN_SDK}" NO_DEFAULT_PATH)

add_library(flashdreams_capi SHARED src/flashdreams.c)
target_compile_features(flashdreams_capi PRIVATE c_std_11)
target_compile_definitions(flashdreams_capi PRIVATE FLASHDREAMS_CAPI_BUILD)
target_include_directories(flashdreams_capi PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(flashdreams_capi PRIVATE Python3::Python)
set_target_properties(flashdreams_capi PROPERTIES C_VISIBILITY_PRESET hidden)

add_executable(flashdreams_example examples/launch.c)
target_compile_features(flashdreams_example PRIVATE c_std_11)
find_package(CUDAToolkit REQUIRED)
target_link_libraries(flashdreams_example PRIVATE flashdreams_capi CUDA::cuda_driver SDL3::SDL3-shared)
if(MSVC)
    # CUDA's driver stubs request the static CRT; use the example's /MD CRT.
    target_link_options(flashdreams_example PRIVATE /NODEFAULTLIB:LIBCMT)
endif()
if(WIN32)
    add_custom_command(TARGET flashdreams_example POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:Python3::Python>" "$<TARGET_FILE_DIR:flashdreams_example>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:SDL3::SDL3-shared>" "$<TARGET_FILE_DIR:flashdreams_example>"
        VERBATIM)
endif()
install(TARGETS flashdreams_capi)
install(FILES include/flashdreams.h DESTINATION include)

add_executable(test_example_input EXCLUDE_FROM_ALL tests/test_input.c)
target_compile_features(test_example_input PRIVATE c_std_11)
target_link_libraries(test_example_input PRIVATE flashdreams_capi CUDA::cuda_driver SDL3::SDL3-shared)
if(MSVC)
    target_link_options(test_example_input PRIVATE /NODEFAULTLIB:LIBCMT)
endif()
