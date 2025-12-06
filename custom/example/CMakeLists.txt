# Minimum version of CMake required
cmake_minimum_required(VERSION 3.10)

# Define the project name and enable C++ and CUDA
project(SimpleCustomOpTest LANGUAGES CXX CUDA)

# --- Configuration Variables ---
set(ONNXRUNTIME_ROOT "/home/jetson/onnxruntimel")
set(ORT_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
set(ORT_LIB_DIR "${ONNXRUNTIME_ROOT}/build/Linux/Release") 

# --- Include Directories ---

include_directories(
    .
    ${ORT_INCLUDE_DIR}
)

# --- Link Libraries (Explicitly Listing Files for Reliability) ---

# Find the Threads library for multi-threading support
find_package(Threads REQUIRED)

# List all necessary ONNX Runtime libraries
set(ONNX_RUNTIME_LIBS
    ${ORT_LIB_DIR}/libonnxruntime.so

    # Shared EP dispatcher (must be before CUDA/TensorRT)
    ${ORT_LIB_DIR}/libonnxruntime_providers_shared.so

    # TensorRT provider (shared)
    ${ORT_LIB_DIR}/libonnxruntime_providers_tensorrt.so

    # CUDA provider (STATIC)
    ${ORT_LIB_DIR}/libonnxruntime_providers_cuda.a
)

# --- Define Source Files ---

set(KERNEL_SRC simple_relu_add_kernel.cc)
set(MAIN_SRC main.cc)
set(CUDA_SRC simple_relu_add_kernel.cu)

# --- Build the Executable ---

add_executable(custom_op_test
    ${KERNEL_SRC}
    ${MAIN_SRC}
    ${CUDA_SRC}
)

# Link necessary libraries
target_link_libraries(custom_op_test
    PRIVATE
    # Explicitly listed ORT libraries
    ${ONNX_RUNTIME_LIBS}
    
    # Required system libraries
    Threads::Threads
    
    # CUDA runtime libraries
    cuda_runtime
    cudart
    
    # For certain Linux setups, may need pthreads
    # pthread
)