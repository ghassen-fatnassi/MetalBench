#include <cuda_runtime.h>
#include <algorithm> // for fmaxf

extern "C" __global__ void SimpleReLUAddKernel(const float* input1, const float* input2, float* output, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float relu_val = fmaxf(0.0f, input1[idx]);
    output[idx] = relu_val + input2[idx];
  }
}

extern "C" void SimpleReLUAddKernelLaunch(cudaStream_t stream, const float* input1, const float* input2, float* output, size_t size) {
  const int threads_per_block = 256;
  const int num_blocks = static_cast<int>((size + threads_per_block - 1) / threads_per_block);
  // Note: passing stream (may be 0 meaning default stream)
  SimpleReLUAddKernel<<<num_blocks, threads_per_block, 0, stream>>>(input1, input2, output, size);
}
