#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

namespace custom
{

template <typename T>
__global__ void fusedAttnKernel(int n, const T* input, T* output)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        // Identity: Pass input directly to output
        output[idx] = input[idx];
    }
}

int computeFusedAttn(cudaStream_t stream, int n, const float* input, float* output)
{
    constexpr int blockSize = 256;
    const int gridSize = (n + blockSize - 1) / blockSize;
    fusedAttnKernel<float><<<gridSize, blockSize, 0, stream>>>(n, input, output);

    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

} // namespace custom