#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "custom_op.h" // Now includes the declaration for SimpleReLUAddKernelLaunch

// Since SimpleReLUAddKernel is a device function and launched within 
// SimpleReLUAddKernelLaunch (in simple_relu_add_kernel.cu), 
// we cannot launch it directly here in a host-only test.

// --- Host-side verification test (CUDA kernel launch logic) ---
// NOTE: For a host-side test to work, you MUST use the declared 
// SimpleReLUAddKernelLaunch function from custom_op.h, NOT the direct kernel 
// launch syntax, unless you wrap SimpleReLUAddKernel in an extern "C" function.
// For simplicity in this test, let's assume SimpleReLUAddKernelLaunch is 
// what is used to launch the kernel.

// REVISITED: The original main.cc was trying to launch the kernel directly.
// To fix the error while preserving the intent of a simple CUDA test:
// 1. You must declare the SimpleReLUAddKernel signature.
// 2. You must remove the #include "simple_relu_add_kernel.cu".

// External declaration of the kernel for the direct launch test.
// NOTE: This assumes the kernel is visible, which it may not be 
// if nvcc is compiling the .cu file separately without an extern "C". 
// A safer approach is to use SimpleReLUAddKernelLaunch, but for this 
// stand-alone test, the original method is often desired.

__global__ void SimpleReLUAddKernel(const float* input1, const float* input2, float* output, size_t size);


int main() {
    const size_t size = 8;

    // Input data
    std::vector<float> input1(size);
    std::vector<float> input2(size);
    for (size_t i = 0; i < size; ++i) {
        input1[i] = static_cast<float>(i) - 2.0f; // some negative values
        input2[i] = static_cast<float>(i) * 0.5f;
    }

    // Allocate device memory
    float *d_input1, *d_input2, *d_output;
    cudaMalloc(&d_input1, size * sizeof(float));
    cudaMalloc(&d_input2, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));

    cudaMemcpy(d_input1, input1.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input2, input2.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    const int threads_per_block = 256;
    const int blocks = (size + threads_per_block - 1) / threads_per_block;
    
    // Direct kernel launch is now possible because we removed the include 
    // of the .cu file and instead declared the kernel (SimpleReLUAddKernel)
    // using the __global__ specifier.
    SimpleReLUAddKernel<<<blocks, threads_per_block>>>(d_input1, d_input2, d_output, size);

    // Copy back results
    std::vector<float> output(size);
    cudaMemcpy(output.data(), d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Print results
    std::cout << "SimpleReLUAdd output:\n";
    for (size_t i = 0; i < size; ++i) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory
    cudaFree(d_input1);
    cudaFree(d_input2);
    cudaFree(d_output);

    return 0;
}