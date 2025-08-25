# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(mlperf-automotive)

include(GNUInstallDirs)
find_package(CUDA REQUIRED)
find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG)

# Build options
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)
set(PYTHON3_CMD "python3" CACHE STRING "Python 3 command")

# Pass the Loadgen include directory from command line
add_definitions(-DLOADGEN_INCLUDE_DIR=${LOADGEN_INCLUDE_DIR})

# Workaround for TRT header warning
execute_process(COMMAND echo "Warning: setting -Wno-deprecated-declarations to avoid header warnings")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")

# Set sm versions
if(${IS_THOR})
    string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_101,code=sm_101")
endif()

project(harness LANGUAGES CXX CUDA)

# Find the static Loadgen library
unset(LOADGEN_LIB CACHE)
find_library(LOADGEN_LIB NAMES libmlperf_loadgen.a PATHS ${LOADGEN_LIB_DIR})

# Set the path to the LWIS library
unset(LWIS_INCLUDE_DIR CACHE)
set(LWIS_INCLUDE_DIR lwis/include)

# Set NVTX library path
unset(NV_TOOLS_EXT_LIB CACHE)
set(NV_TOOLS_EXT_LIB ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnvToolsExt.so)

execute_process(COMMAND uname -p OUTPUT_VARIABLE ARCH)


######### FFI Utilities for Python ###########
pybind11_add_module(FFIUtils ffi_utils/ffi_utils.cpp)
target_link_libraries(FFIUtils PRIVATE ${LOADGEN_LIB})
target_include_directories(FFIUtils
    PUBLIC
    ${LOADGEN_INCLUDE_DIR})
set_target_properties(FFIUtils PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${FFI_UTILS_DIR})


######### DEFAULT HARNESS ########
# Add the LWIS subdirectory (which will generate a static LWIS library)
add_subdirectory(lwis)

# Build the default harness which covers single_stream and offline scenarios on image benchmarks.
execute_process(COMMAND echo "Building default harness...")
add_executable(harness_default
    harness_default/main_default.cc
    common/logger.cpp
)

target_link_libraries(harness_default
    nvinfer
    nvinfer_plugin
    gflags
    glog
    ${CUDA_LIBRARIES}
    lwis
    ${LOADGEN_LIB}
)

target_include_directories(harness_default
    PUBLIC
        ${CUDA_INCLUDE_DIRS}
        ${LOADGEN_INCLUDE_DIR}
        ${LWIS_INCLUDE_DIR}
        common
)

######### DEFAULT PY_HARNESS ########
add_custom_target(py_harness_default ALL
    COMMAND ${CMAKE_COMMAND} -E echo "Building default py_harness..."
    COMMAND ${CMAKE_COMMAND} -E env CXX_FLAGS=${CMAKE_CXX_FLAGS}
            ${PYTHON3_CMD} setup.py build_ext
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/harness_default
)

add_dependencies(py_harness_default lwis)

# Remove the build directory after building py_harness_default
add_custom_command(TARGET py_harness_default POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_SOURCE_DIR}/harness_default/build
)


######### BEVFormer HARNESS ########
execute_process(COMMAND echo "Building BEVFormer harness...")
add_executable(harness_bevformer
    harness_bevformer/main_bevformer.cc
    harness_bevformer/lwis_bevformer.cpp
    common/logger.cpp
)

target_link_libraries(harness_bevformer
    nvinfer
    nvinfer_plugin
    gflags
    glog
    ${CUDA_LIBRARIES}
    ${LOADGEN_LIB}
)

target_include_directories(harness_bevformer
    PUBLIC
        ${CUDA_INCLUDE_DIRS}
        ${LOADGEN_INCLUDE_DIR}
        ${LWIS_INCLUDE_DIR}
        common
        harness_bevformer
)

######### BEVFormer PY_HARNESS ########
add_library(bevformer
    harness_bevformer/lwis_bevformer.cpp
)

set_target_properties(bevformer PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lwis_bevformer"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lwis_bevformer"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lwis_bevformer"
)

target_link_libraries(bevformer
    nvinfer
    nvinfer_plugin
    gflags
    glog
    ${CUDA_LIBRARIES}
    ${LOADGEN_LIB}
)

target_include_directories(bevformer
    PUBLIC
        ${CUDA_INCLUDE_DIRS}
        ${LOADGEN_INCLUDE_DIR}
        ${LWIS_INCLUDE_DIR}
        common
        harness_bevformer
)

add_custom_target(py_harness_bevformer ALL
    COMMAND ${CMAKE_COMMAND} -E echo "Building BEVFormer py_harness..."
    COMMAND ${CMAKE_COMMAND} -E env CXX_FLAGS=${CMAKE_CXX_FLAGS}
            ${PYTHON3_CMD} setup.py build_ext
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/harness_bevformer
)

# Remove the build directory after building py_harness_bevformer
add_custom_command(TARGET py_harness_bevformer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_SOURCE_DIR}/harness_bevformer/build
)

