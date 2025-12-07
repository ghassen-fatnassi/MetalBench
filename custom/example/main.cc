#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "custom_op.h" 

// We do NOT declare the __global__ kernel here because we cannot call it from .cc
// We use the wrapper function from custom_op.h instead.

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

    // Launch kernel via the C++ wrapper
    // We pass '0' (or nullptr) as the stream for default stream
    SimpleReLUAddKernelLaunch(0, d_input1, d_input2, d_output, size);

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