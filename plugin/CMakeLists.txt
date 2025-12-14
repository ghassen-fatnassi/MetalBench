cmake_minimum_required(VERSION 3.10)
project(fused_attention_plugin LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(fused_attention_plugin SHARED
    fused_attention_plugin.cpp
)

# Use CUDA-provided include directories and libraries
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)

# CUDA_CUDART_LIBRARY is provided by FindCUDA
target_link_libraries(fused_attention_plugin
    nvinfer
    ${CUDA_CUDART_LIBRARY}
)


add_executable(build_engine main.cpp)
target_link_libraries(build_engine
    fused_attention_plugin
    nvinfer
    nvonnxparser
)