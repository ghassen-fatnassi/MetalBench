cmake_minimum_required(VERSION 3.10)
project(FusedAttnOpTest LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# Jetson Nano architecture
set(CMAKE_CUDA_ARCHITECTURES "53")

# ONNX Runtime paths
set(ONNXRUNTIME_ROOT "/home/jetson/onnxruntime")
set(ORT_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
# Path for internal headers like "core/framework/op_kernel.h"
set(ORT_INTERNAL_SOURCE_DIR "${ONNXRUNTIME_ROOT}/onnxruntime") 
# --- FIX: Add the path for the missing GSL headers ---
set(GSL_INCLUDE_DIR "${ORT_INTERNAL_SOURCE_DIR}/contrib/gsl-lite/include")

set(ORT_LIB_DIR "${ONNXRUNTIME_ROOT}/build/Linux/Release")

# Include public headers, internal headers, GSL, and CUDA
include_directories(${ORT_INCLUDE_DIR} ${ORT_INTERNAL_SOURCE_DIR} ${GSL_INCLUDE_DIR} /usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

add_definitions(-DONNX_ML=1)
find_package(Threads REQUIRED)

# Sources
set(SRC
    fused_attn_op.cc   # kernel implementation
    main.cc            # test executable
)

# Optional: if you have a CUDA kernel
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/fused_attn_op.cu")
    list(APPEND SRC fused_attn_op.cu)
endif()

add_executable(custom_attn_test ${SRC})

# ONNX Runtime libraries
set(ONNX_RUNTIME_LIBS
    ${ORT_LIB_DIR}/libonnxruntime.so
    ${ORT_LIB_DIR}/libonnxruntime_providers_shared.so
    ${ORT_LIB_DIR}/libonnxruntime_providers_cuda.a
)

target_link_libraries(custom_attn_test PRIVATE ${ONNX_RUNTIME_LIBS} Threads::Threads cudart dl rt)