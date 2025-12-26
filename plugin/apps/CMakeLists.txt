cmake_minimum_required(VERSION 3.25)

project(APPS VERSION 0.0.1 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# TensorRT NvInfer related libs
find_library(NVINFER_LIB nvinfer HINTS /usr/lib/aarch64-linux-gnu/ PATH_SUFFIXES lib lib64 REQUIRED)
find_library(NVINFER_PLUGIN_LIB nvinfer_plugin HINTS /usr/lib/aarch64-linux-gnu/ PATH_SUFFIXES lib lib64 REQUIRED)
find_library(NVONNXPARSER_LIB nvonnxparser HINTS /usr/lib/aarch64-linux-gnu/ PATH_SUFFIXES lib lib64 REQUIRED)

add_executable(build_engine build_engine.cpp)
# ADDED ${CMAKE_DL_LIBS} HERE
target_link_libraries(build_engine PRIVATE ${NVINFER_LIB} ${NVONNXPARSER_LIB} ${NVINFER_PLUGIN_LIB} CUDA::cudart ${CMAKE_DL_LIBS})

add_executable(run_engine run_engine.cpp)
# ADDED ${CMAKE_DL_LIBS} HERE AS WELL
target_link_libraries(run_engine PRIVATE ${NVINFER_LIB} CUDA::cudart ${CMAKE_DL_LIBS})