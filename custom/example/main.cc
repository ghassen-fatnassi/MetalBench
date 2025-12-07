#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "custom_op.h"

// Declare the kernel for direct testing
__global__ void SimpleReLUAddKernel(const float* input1, 
                                     const float* input2, 
                                     float* output, 
                                     size_t size);

int main() {
    const size_t size = 8;
    
    // Input data
    std::vector<float> input1(size);
    std::vector<float> input2(size);
    for (size_t i = 0; i < size; ++i) {
        input1[i] = static_cast<float>(i) - 2.0f;  // some negative values
        input2[i] = static_cast<float>(i) * 0.5f;
    }
    
    std::cout << "Input 1: ";
    for (size_t i = 0; i < size; ++i) {
        std::cout << input1[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Input 2: ";
    for (size_t i = 0; i < size; ++i) {
        std::cout << input2[i] << " ";
    }
    std::cout << std::endl;
    
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
    
    SimpleReLUAddKernel<<<blocks, threads_per_block>>>(d_input1, d_input2, d_output, size);
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    
    // Wait for kernel to complete
    cudaDeviceSynchronize();
    
    // Copy back results
    std::vector<float> output(size);
    cudaMemcpy(output.data(), d_output, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print results
    std::cout << "\nSimpleReLUAdd output:" << std::endl;
    for (size_t i = 0; i < size; ++i) {
        std::cout << "ReLU(" << input1[i] << ") + " << input2[i] 
                  << " = " << output[i] << std::endl;
    }
    
    // Free device memory
    cudaFree(d_input1);
    cudaFree(d_input2);
    cudaFree(d_output);
    
    std::cout << "\nTest completed successfully!" << std::endl;
    return 0;
}