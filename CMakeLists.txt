cmake_minimum_required(VERSION 3.13)
project(ort_cpp_example LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use absolute path so CMake knows where to look
set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/onnxruntime")

# Include and link directories
include_directories(${ONNXRUNTIME_ROOT}/include)
link_directories(${ONNXRUNTIME_ROOT}/lib)

# Your main source file (inside Bench/)
add_executable(ort_cpp_example Bench/main.cpp)

# Link against ONNX Runtime shared library
target_link_libraries(ort_cpp_example PRIVATE onnxruntime)
