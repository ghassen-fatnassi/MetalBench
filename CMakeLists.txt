cmake_minimum_required(VERSION 3.10)
project(onnx_jetson_tools LANGUAGES CXX CUDA)

# ------------------------
# C++ & CUDA settings
# ------------------------
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_ARCHITECTURES 53)  # Jetson Nano

# ------------------------
# ONNX Runtime paths
# ------------------------
set(ORT_ROOT "/home/jetson/onnxruntime")
set(ORT_INCLUDE_DIR "${ORT_ROOT}/build/Linux/Release/include")  # only include folder
set(ORT_LIB_DIR     "${ORT_ROOT}/build/Linux/Release")

include_directories(
    ${ORT_INCLUDE_DIR}          # public headers only
    /usr/local/cuda/include
)

link_directories(
    ${ORT_LIB_DIR}
    /usr/local/cuda/lib64
)

set(CMAKE_BUILD_RPATH ${ORT_LIB_DIR})
set(CMAKE_INSTALL_RPATH ${ORT_LIB_DIR})

# ------------------------
# Threads
# ------------------------
find_package(Threads REQUIRED)

# ------------------------
# ONNX Runtime shared libraries
# ------------------------
set(ONNX_RUNTIME_LIBS
    ${ORT_LIB_DIR}/libonnxruntime.so
    ${ORT_LIB_DIR}/libonnxruntime_providers_shared.so
    ${ORT_LIB_DIR}/libonnxruntime_providers_cuda.so
    ${ORT_LIB_DIR}/libonnxruntime_providers_tensorrt.so
)

# ------------------------
# TensorRT / CUDA libraries
# ------------------------
set(TRT_LIBS
    nvinfer
    nvinfer_plugin
    nvonnxparser
    cudart
    cublas
    cublasLt
    cudnn
)

# ------------------------
# Function to create executables
# ------------------------
function(add_onnxbin name src)
    add_executable(${name} ${src})
    target_include_directories(${name} PRIVATE ${ORT_INCLUDE_DIR})
    target_link_libraries(${name} PRIVATE
        ${ONNX_RUNTIME_LIBS}
        ${TRT_LIBS}
        Threads::Threads
    )
endfunction()

# ------------------------
# Executables
# ------------------------
add_onnxbin(single_run_cuda_dynamic      Bench/single_run_cuda_dynamic.cpp)
add_onnxbin(single_run_cuda_static       Bench/single_run_cuda_static.cpp)
add_onnxbin(single_run_cpu               Bench/single_run_cpu.cpp)
add_onnxbin(jetson_bench                 Bench/jetson_bench.cpp)
add_onnxbin(single_run_tensorrt_static  Bench/single_run_tensorrt_static.cpp)
add_onnxbin(list_eps                     Bench/list_EPS.cpp)
