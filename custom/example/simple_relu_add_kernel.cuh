// simple_relu_add_kernel.cuh
#pragma once
#include <cstddef>   // for size_t

// CUDA kernel
__global__ void SimpleReLUAddKernel(const float* input1, const float* input2, float* output, size_t size);
