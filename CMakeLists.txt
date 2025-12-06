cmake_minimum_required(VERSION 3.17) # Bumped for FindCUDAToolkit
project(onnx_jetson_tools LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 1. Configurable Path (Allows overriding via command line)
set(ORT_ROOT "/home/jetson/onnxruntime" CACHE PATH "Path to ONNX Runtime root")

set(ORT_INCLUDE_DIR "${ORT_ROOT}/include")
set(ORT_LIB_DIR "${ORT_ROOT}/build/Linux/Release")

include_directories(${ORT_INCLUDE_DIR})

# 2. Find Dependencies
find_package(Threads REQUIRED)

# Find CUDA Toolkit to get the system libs (cudart, cublas, etc.)
# implied by linking the static providers_cuda.a
find_package(CUDAToolkit REQUIRED)

# 3. Define ONNX Runtime Libraries
# Order is critical: Providers -> Shared Dispatcher -> Core
set(ONNX_RUNTIME_LIBS
    # TensorRT Provider (Shared)
    ${ORT_LIB_DIR}/libonnxruntime_providers_tensorrt.so
    
    # CUDA Provider (STATIC)
    # Because this is static, we MUST link system CUDA libs below
    ${ORT_LIB_DIR}/libonnxruntime_providers_cuda.a
    
    # Shared EP dispatcher
    ${ORT_LIB_DIR}/libonnxruntime_providers_shared.so
    
    # Core Shared Library
    ${ORT_LIB_DIR}/libonnxruntime.so
)

function(add_onnxbin name src)
    add_executable(${name} ${src})
    
    target_include_directories(${name} PRIVATE ${ORT_INCLUDE_DIR})
    
    # 4. Link ORT + System CUDA Libraries
    target_link_libraries(${name} PRIVATE 
        ${ONNX_RUNTIME_LIBS}
        Threads::Threads
        CUDA::cudart
        CUDA::cudnn
        CUDA::cublas
        CUDA::curand
        # If using TensorRT provider, you might also need:
        # CUDA::nvinfer
    )
    
    # Add RPATH so the executable finds the .so files at runtime without LD_LIBRARY_PATH
    set_target_properties(${name} PROPERTIES BUILD_RPATH "${ORT_LIB_DIR}")
endfunction()

add_onnxbin(single_run_cuda Bench/single_run_cuda.cpp)
add_onnxbin(single_run_cpu Bench/single_run_cpu.cpp)
add_onnxbin(jetson_bench Bench/jetson_bench.cpp)