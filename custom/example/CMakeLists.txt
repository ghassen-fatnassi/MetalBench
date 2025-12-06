cmake_minimum_required(VERSION 3.10)
project(SimpleCustomOpTest LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# Jetson TX2/Xavier default architecture
set(CMAKE_CUDA_ARCHITECTURES "53")

# -------------------------------
# ONNX Runtime paths
# -------------------------------
set(ONNXRUNTIME_ROOT "/home/jetson/onnxruntimel")
set(ORT_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
set(ORT_LIB_DIR "${ONNXRUNTIME_ROOT}/build/Linux/Release")

# -------------------------------
# Include directories
# -------------------------------
include_directories(
    .
    ${ORT_INCLUDE_DIR}
    /usr/local/cuda/include       # REQUIRED for cuda_runtime.h in .cc files
)

# CUDA libraries path
link_directories(/usr/local/cuda/lib64)

# -------------------------------
# ONNX Runtime libraries
# -------------------------------
set(ONNX_RUNTIME_LIBS
    ${ORT_LIB_DIR}/libonnxruntime.so
    ${ORT_LIB_DIR}/libonnxruntime_providers_shared.so
    ${ORT_LIB_DIR}/libonnxruntime_providers_tensorrt.so
    ${ORT_LIB_DIR}/libonnxruntime_providers_cuda.a
)

find_package(Threads REQUIRED)

# -------------------------------
# Sources
# -------------------------------
set(KERNEL_SRC simple_relu_add_kernel.cc)
set(MAIN_SRC main.cc)
set(CUDA_SRC simple_relu_add_kernel.cu)

add_executable(custom_op_test
    ${KERNEL_SRC}
    ${MAIN_SRC}
    ${CUDA_SRC}
)

# -------------------------------
# Link libraries
# -------------------------------
target_link_libraries(custom_op_test
    PRIVATE
    ${ONNX_RUNTIME_LIBS}
    Threads::Threads
    cudart                  # CUDA runtime
)
