#include "custom_op.h"

// CUDA kernel
__global__ void SimpleReLUAddKernel(const float* input1, const float* input2, float* output, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float relu_val = fmaxf(0.0f, input1[idx]);
        output[idx] = relu_val + input2[idx];
    }
}

// Launch function
void SimpleReLUAddKernelLaunch(cudaStream_t stream,
                               const float* input1,
                               const float* input2,
                               float* output,
                               size_t size) {
    int threads_per_block = 256;
    int num_blocks = (size + threads_per_block - 1) / threads_per_block;
    SimpleReLUAddKernel<<<num_blocks, threads_per_block, 0, stream>>>(input1, input2, output, size);
}
