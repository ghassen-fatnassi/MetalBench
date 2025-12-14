cmake_minimum_required(VERSION 3.10)
project(fused_attention_plugin LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(fused_attention_plugin SHARED
    fused_attention_plugin.cpp
)

include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# TensorRT 7.1 libraries vary by install; minimally link nvinfer and cudart for the plugin,
# and add nvonnxparser for the engine builder tool.
target_link_libraries(fused_attention_plugin
    nvinfer
    cudart
)

add_executable(build_engine main.cpp)
target_link_libraries(build_engine
    fused_attention_plugin
    nvinfer
    nvonnxparser
)