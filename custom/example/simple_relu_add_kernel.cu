#include <cuda_runtime.h>

__global__ void SimpleReLUAddKernel(const float* in1, const float* in2, float* out, size_t n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        float x = in1[idx];
        float relu = (x > 0.f ? x : 0.f);
        out[idx] = relu + in2[idx];
    }
}

extern "C" void SimpleReLUAddKernelLaunch(cudaStream_t stream,
                                          const float* in1,
                                          const float* in2,
                                          float* out,
                                          size_t n)
{
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    SimpleReLUAddKernel<<<blocks, threads, 0, stream>>>(in1, in2, out, n);
}
