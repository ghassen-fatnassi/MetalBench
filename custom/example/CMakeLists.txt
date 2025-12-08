cmake_minimum_required(VERSION 3.10)
project(FusedAttnOpTest LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17) # Use C++17
set(CMAKE_CUDA_STANDARD 17) # Use C++17

# Jetson Nano architecture (Architecture 53 is valid for Jetson Nano)
set(CMAKE_CUDA_ARCHITECTURES "53")

# --- ONNX Runtime and Environment Paths ---
# You can keep these paths, assuming they are correct for your Jetson setup.
set(ONNXRUNTIME_ROOT "/home/jetson/onnxruntime")
set(ORT_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
set(ORT_LIB_DIR "${ONNXRUNTIME_ROOT}/build/Linux/Release")

include_directories(${ORT_INCLUDE_DIR} /usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

add_definitions(-DONNX_ML=1)
# Add the permissive flag for C++ CustomOpBase issues
add_compile_options(-fpermissive)
find_package(Threads REQUIRED)

# --- 1. DEFINE SHARED LIBRARY TARGET (Your Custom Op) ---
## We assume the Op definition/registration is in FusedAttnOp.cc
set(CUSTOM_OP_SRC
    FusedAttnOpKernel.cc    # The actual compute logic
    FusedAttnOp.cc          # Op definition and RegisterCustomOps function
)
# Add these lines to your CMakeLists.txt
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CUDA_FLAGS_DEBUG "-g")
# Optional: if you have a CUDA kernel implementation in .cu
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/FusedAttnOpKernel.cu")
    list(APPEND CUSTOM_OP_SRC FusedAttnOpKernel.cu)
endif()

# Create the shared library (libcustom_attn.so)
add_library(custom_attn SHARED ${CUSTOM_OP_SRC})
target_link_libraries(custom_attn PRIVATE Threads::Threads cudart dl rt)

# Set the output name for consistency
set_target_properties(custom_attn PROPERTIES OUTPUT_NAME "libcustom_attn")

# --- 2. DEFINE TEST EXECUTABLE TARGET ---
set(TEST_SRC
    main.cc # Your test file that calls the session and loads libcustom_attn.so
)

add_executable(custom_attn_test ${TEST_SRC})
# ONNX Runtime libraries
set(ONNX_RUNTIME_LIBS
    ${ORT_LIB_DIR}/libonnxruntime.so
    ${ORT_LIB_DIR}/libonnxruntime_providers_shared.so
    ${ORT_LIB_DIR}/libonnxruntime_providers_cuda.a
)

target_link_libraries(custom_attn_test PRIVATE ${ONNX_RUNTIME_LIBS} Threads::Threads cudart dl rt)