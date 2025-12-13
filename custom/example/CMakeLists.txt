cmake_minimum_required(VERSION 3.10)
project(onnx_custom_op_bench LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# Jetson Nano: 53, Xavier: 72, Orin: 87. Adjust if needed.
set(CMAKE_CUDA_ARCHITECTURES 53) 

# ============================================================
# Path Configuration (Hardcoded as requested)
# ============================================================
set(ORT_INSTALL_DIR "/home/jetson/ort_install")
set(ORT_INCLUDE_DIR "${ORT_INSTALL_DIR}/include")
set(ORT_LIB_DIR "${ORT_INSTALL_DIR}/lib")

set(CUDA_INCLUDE_DIR "/usr/local/cuda/include")
set(CUDA_LIB_DIR "/usr/local/cuda/lib64")

include_directories(
    ${ORT_INCLUDE_DIR} 
    ${ORT_INCLUDE_DIR}/external/onnx

    ${ORT_INCLUDE_DIR}/onnxruntime/core/session
    ${ORT_INCLUDE_DIR}/onnxruntime
    ${CUDA_INCLUDE_DIR}
)

link_directories(${ORT_LIB_DIR} ${CUDA_LIB_DIR})

# Libraries to link against
set(TRT_LIBS nvinfer nvinfer_plugin nvonnxparser cudart cublas cublasLt cudnn)
set(ONNXRUNTIME_LIBS onnxruntime onnxruntime_providers_shared onnxruntime_providers_tensorrt)

find_package(Threads REQUIRED)

# ============================================================
# PHASE 1: Build the Custom Operator Library (.so)
# ============================================================
# This creates the libort_custom_fused_attn.so that main.cpp will dlopen
add_library(ort_custom_fused_attn SHARED 
    fused_attn.cc 
    fused_attn.cu
)

target_link_libraries(ort_custom_fused_attn PRIVATE 
    ${ONNXRUNTIME_LIBS} 
    ${TRT_LIBS} 
    dl
)

# ============================================================
# PHASE 2: Build the Benchmark App
# ============================================================
add_executable(ort_benchmark_app main.cpp)

target_link_libraries(ort_benchmark_app PRIVATE
    ort_custom_fused_attn  # Dependency check
    ${ONNXRUNTIME_LIBS}
    ${TRT_LIBS}
    Threads::Threads
    dl
)

# RPATH: Helps the binary find libonnxruntime.so and your custom lib at runtime
set_target_properties(ort_benchmark_app PROPERTIES 
    INSTALL_RPATH "${ORT_LIB_DIR}:\$ORIGIN"
    BUILD_RPATH "${ORT_LIB_DIR}"
)

# Copy the custom .so to the same folder as the app so dlopen("./...") works
add_custom_command(
    TARGET ort_benchmark_app POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "$<TARGET_FILE:ort_custom_fused_attn>"
    "$<TARGET_FILE_DIR:ort_benchmark_app>"
)