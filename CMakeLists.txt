cmake_minimum_required(VERSION 3.10)
project(onnx_jetson_tools LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ============================================================
# ONNX Runtime install prefix
# ============================================================
set(ORT_INSTALL_DIR "/home/jetson/ort_install")

set(ORT_INCLUDE_DIR "${ORT_INSTALL_DIR}/include")
set(ORT_LIB_DIR "${ORT_INSTALL_DIR}/lib")

include_directories(${ORT_INCLUDE_DIR} ${ORT_INCLUDE_DIR}/onnxruntime/core/session)
link_directories(${ORT_LIB_DIR})

# ============================================================
# CUDA + TensorRT linkage
# ============================================================
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

set(TRT_LIBS
    nvinfer
    nvinfer_plugin
    nvonnxparser
    cudart
    cublas
    cublasLt
    cudnn
)

# ============================================================
# ONNX Runtime libraries
# ============================================================
set(ONNXRUNTIME_LIBS
    onnxruntime
    onnxruntime_providers_shared
    onnxruntime_providers_tensorrt
)

find_package(Threads REQUIRED)

# ============================================================
# Helper for adding binaries
# ============================================================
function(add_onnx_bin name src)
    add_executable(${name} ${src})
    target_include_directories(${name} PRIVATE ${ORT_INCLUDE_DIR})
    target_link_libraries(${name} PRIVATE
        ${ONNXRUNTIME_LIBS}
        ${TRT_LIBS}
        Threads::Threads
    )
endfunction()

# ============================================================
# Your tools
# ============================================================
add_onnx_bin(list_eps Bench/list_eps.cpp)
add_onnx_bin(single_run_cpu Bench/single_run_cpu.cpp)
add_onnx_bin(single_run_cuda_static Bench/single_run_cuda_static.cpp)
add_onnx_bin(single_run_cuda_dynamic Bench/single_run_cuda_dynamic.cpp)
# add_onnx_bin(single_run_cuda_graphs Bench/single_run_cuda_graphs.cpp)
add_onnx_bin(single_run_tensorrt_static Bench/single_run_tensorrt_static.cpp)
