cmake_minimum_required(VERSION 3.10)
project(fused_attention_plugin)

find_package(CUDA REQUIRED)

add_library(fused_attention_plugin SHARED
    fused_attention_plugin.cpp
)

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
