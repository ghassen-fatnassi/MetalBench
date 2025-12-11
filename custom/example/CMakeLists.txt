# Use a modern, stable CMake version
cmake_minimum_required(VERSION 3.10)
project(custom_op_benchmark LANGUAGES CXX CUDA) 

set(ORT_ROOT_DIR "/home/jetson/onnxruntime/build")

set(ONNX_INCLUDE_DIR "${ORT_ROOT_DIR}/include/onnxruntime")

set(ONNX_LIB_DIR "${ORT_ROOT_DIR}/Linux/Release") 

set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda") 

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDA REQUIRED PATHS ${CUDA_TOOLKIT_ROOT_DIR} NO_DEFAULT_PATH)

set(CMAKE_CUDA_ARCHITECTURES 53) 

include_directories(
    ${ONNX_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
)

add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fPIC -Wall>)
cuda_add_compile_options(-Xcompiler -fPIC) 

set(CUSTOM_OP_LIB_NAME "ort_custom_fused_attn")
set(CUSTOM_OP_SOURCES
    fused_attn.cc
    fused_attn.cu
)

add_library(${CUSTOM_OP_LIB_NAME} SHARED ${CUSTOM_OP_SOURCES})

target_link_libraries(${CUSTOM_OP_LIB_NAME}
    PRIVATE
    ${ONNX_LIB_DIR}/libonnxruntime.so  
    
    ${CUDA_LIBRARIES}
)

set(APP_NAME "ort_benchmark_app")

add_executable(${APP_NAME} main.cpp)

target_link_libraries(${APP_NAME}
    PRIVATE
    ${ONNX_LIB_DIR}/libonnxruntime.so
    ${ONNX_LIB_DIR}/libonnxruntime_providers_cuda.a
    ${ONNX_LIB_DIR}/libonnxruntime_providers_shared.so
    ${CUSTOM_OP_LIB_NAME}
    
    dl
)

set_target_properties(${APP_NAME} PROPERTIES 
    INSTALL_RPATH "${ONNX_LIB_DIR}:\$ORIGIN"
    BUILD_RPATH "${ONNX_LIB_DIR}"
)

add_custom_command(
    TARGET ${APP_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "$<TARGET_FILE:${CUSTOM_OP_LIB_NAME}>"
    "$<TARGET_FILE_DIR:${APP_NAME}>"
)