cmake_minimum_required(VERSION 3.10)
project(onnx_jetson_tools LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(ORT_ROOT "/home/jetson/onnxruntime")

set(ORT_INCLUDE_DIR "${ORT_ROOT}/include")
set(ORT_LIB_DIR "${ORT_ROOT}/build/Linux/Release")

include_directories(${ORT_INCLUDE_DIR})

find_package(Threads REQUIRED)

# ---- REQUIRED LIBS IN CORRECT ORDER ----
set(ONNX_RUNTIME_LIBS
    ${ORT_LIB_DIR}/libonnxruntime.so

    # Shared EP dispatcher (must be before CUDA/TensorRT)
    ${ORT_LIB_DIR}/libonnxruntime_providers_shared.so

    # TensorRT provider (shared)
    ${ORT_LIB_DIR}/libonnxruntime_providers_tensorrt.so

    # CUDA provider (STATIC)
    ${ORT_LIB_DIR}/libonnxruntime_providers_cuda.a

    # Needed when linking provider static libs
    cudart
    cublas
    cublasLt
    curand
)

function(add_onnxbin name src)
    add_executable(${name} ${src})
    target_include_directories(${name} PRIVATE ${ORT_INCLUDE_DIR})
    target_link_libraries(${name} PRIVATE ${ONNX_RUNTIME_LIBS} Threads::Threads)
endfunction()

add_onnxbin(single_run_cuda single_run_cuda.cpp)
