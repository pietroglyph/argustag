# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required (VERSION 3.8)

project(cucoslam LANGUAGES CUDA CXX)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" "${CMAKE_MODULE_PATH}")

if (NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED True)
endif()
if (NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED True)
endif()

set(THIRDPARTY_DEP_DIR "${CMAKE_SOURCE_DIR}/thirdparty/lib/cmake")

find_package(Argus REQUIRED)
find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED
    PATHS ${THIRDPARTY_DEP_DIR})
find_package(cuda-api-wrappers
    PATHS ${THIRDPARTY_DEP_DIR})
find_library(APRIL_TAGGING_LIBRARIES REQUIRED
    NAMES april_tagging libapril_tagging.a
    PATHS "${CMAKE_SOURCE_DIR}/thirdparty/lib"
    NO_DEFAULT_PATH)
if (NOT APRIL_TAGGING_LIBRARIES)
    # REQUIRED argument isn't supported until version 3.19
    message(FATAL_ERROR "april_tagging is a required library")
endif()

add_executable(${PROJECT_NAME}
    main.cpp
    argus_camera.cpp
    file_camera.cpp

    utils/ArgusHelpers.cpp
    utils/color_conversion.cu
    )

set_target_properties(${PROJECT_NAME} PROPERTIES
	CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
	--default-stream=per-thread
	--relocatable-device-code=true
	>)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${ARGUS_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    )

# If we allowed CMake >= 3.13 then we could use target_link_directories
link_directories(${PROJECT_NAME} PRIVATE "${CMAKE_SOURCE_DIR}/thirdparty/lib/")

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${ARGUS_LIBRARIES}
    ${OpenCV_LIBS}
    ${APRIL_TAGGING_LIBRARIES}
    fmt::fmt
    cuda
    pthread
    stdc++fs
    )

# Wrap everything in a COMPILE_LANGUAGE:CXX generator expression so that we don't add these flags for the CUDA language
if (MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>/W4 /WX>)
else()
    # FIXME: Argus includes are in the main project target because FindArgus.cmake is bad, and Argus wasn't designed with -pedantic on so we can't use -pedantic
    target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Werror>)
endif()

install(TARGETS ${PROJECT_NAME} DESTINATION bin)

include(CPack)
