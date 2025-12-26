cmake_minimum_required(VERSION 3.25)
project(trt_sanity LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# TensorRT headers live here on Jetson
include_directories(/usr/include)

# Libraries live here on Jetson (ARM64)
link_directories(/usr/lib/aarch64-linux-gnu)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
add_executable(trt_sanity
    trt_sanity.cpp
)

target_link_libraries(trt_sanity
    nvinfer
    nvinfer_plugin
    nvonnxparser
)
