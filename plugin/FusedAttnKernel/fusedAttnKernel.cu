/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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