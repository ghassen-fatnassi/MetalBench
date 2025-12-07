#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "custom_op.h" 

int main() {
    // Keep size 8 for this test
    const size_t size = 8;

    // --- REPLACEMENT START ---
    // Hardcoded deterministic data
    // Input 1: Has negatives to prove ReLU works (negatives become 0)
    std::vector<float> input1 = { -10.0f, -5.0f, -1.0f, 0.0f, 1.0f, 5.0f, 10.0f, 100.0f };
    
    // Input 2: Values to add
    std::vector<float> input2 = {   1.0f,  2.0f,  3.0f, 4.0f, 5.0f, 6.0f,  7.0f,   8.0f };
    // --- REPLACEMENT END ---

    // Expected Output Logic:
    // Index 0: ReLU(-10) + 1 = 0 + 1 = 1
    // Index 1: ReLU(-5)  + 2 = 0 + 2 = 2
    // Index 4: ReLU(1)   + 5 = 1 + 5 = 6

    // Allocate device memory
    float *d_input1, *d_input2, *d_output;
    cudaMalloc(&d_input1, size * sizeof(float));
    cudaMalloc(&d_input2, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));

    cudaMemcpy(d_input1, input1.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input2, input2.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel via wrapper (passing 0 for default stream)
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
    
    // Expected output: 1 2 3 4 6 11 17 108

    // Free device memory
    cudaFree(d_input1);
    cudaFree(d_input2);
    cudaFree(d_output);

    return 0;
}