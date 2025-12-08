cmake_minimum_required(VERSION 3.10)
project(FusedAttnOpTest LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# Jetson Nano architecture
set(CMAKE_CUDA_ARCHITECTURES "53")

# ONNX Runtime paths
set(ONNXRUNTIME_ROOT "/home/jetson/onnxruntime")
set(ORT_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
# Assuming ONNX runtime source root is needed for internal headers (core/framework etc.)
# If the ORT build only exposed headers in 'include', you may need to find the internal header path.
# For simplicity, we assume internal headers are also available in ORT_INCLUDE_DIR or its parent.
# A common ORT structure:
set(ORT_INTERNAL_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include") # Adjust this if needed

set(ORT_LIB_DIR "${ONNXRUNTIME_ROOT}/build/Linux/Release")

include_directories(${ORT_INCLUDE_DIR} ${ORT_INTERNAL_INCLUDE_DIR} /usr/local/cuda/include)
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
# We now need the full ONNX Runtime static library (libonnxruntime.a) or the shared library (libonnxruntime.so) 
# as the CustomRegistry API uses internal ORT components.
# The original setup links libonnxruntime.so, which should be fine if it exposes all required symbols.
set(ONNX_RUNTIME_LIBS
    ${ORT_LIB_DIR}/libonnxruntime.so
    ${ORT_LIB_DIR}/libonnxruntime_providers_shared.so
    ${ORT_LIB_DIR}/libonnxruntime_providers_cuda.a
)

target_link_libraries(custom_attn_test PRIVATE ${ONNX_RUNTIME_LIBS} Threads::Threads cudart dl rt)