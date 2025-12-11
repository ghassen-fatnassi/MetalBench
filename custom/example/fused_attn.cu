#include <cuda_runtime.h>

// 1. Generic CUDA Kernel
// Works for any tensor shape because we treat it as a flat array of size 'count'
template <typename T>
__global__ void FusedAttnKernel(const T* input, T* output, size_t count) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        output[i] = input[i];
    }
}

// 2. Launcher
void LaunchFusedAttnKernel(const float* input, float* output, size_t count, cudaStream_t stream) {
    int blockSize = 256;
    int numBlocks = (count + blockSize - 1) / blockSize;

    FusedAttnKernel<float><<<numBlocks, blockSize, 0, stream>>>(input, output, count);
}