#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cmath>

namespace custom
{

// 1. Projection and Split Kernel: X -> Q, K, V
// Applies 1x1 Conv (W1) and Bias (B1)
// Input: (N, C, H, W)
// Weights: (3*C, C, 1, 1) -> Treated as MatMul per pixel
// Output: Q, K, V separated (each N, C, H, W)
template <typename T>
__global__ void qkv_proj_kernel(int n, int c, int h, int w, 
                                const T* __restrict__ input, 
                                const T* __restrict__ weights, 
                                const T* __restrict__ bias,
                                T* __restrict__ q, 
                                T* __restrict__ k, 
                                T* __restrict__ v)
{
    // Grid: (H * W * N), Block: C (assuming C <= 1024)
    // One block per pixel per batch item. 
    // Threads calculate 3*C output channels.
    // Note: Since 3*C might be > 1024 or variable, let's invert loops.
    // Optimized for simplicity: Grid (N, H, W), Loop over output channels.
    
    int b_idx = blockIdx.z;
    int r = blockIdx.y;
    int c_idx = blockIdx.x;
    
    int px_offset = b_idx * (c * h * w) + r * w + c_idx; // Input is NCHW, we need to stride C
    
    // We compute 3*C outputs for this pixel.
    // Q: channels 0..C-1
    // K: channels C..2C-1
    // V: channels 2C..3C-1
    
    int tid = threadIdx.x;
    int stride = blockDim.x;
    
    int num_out = 3 * c;
    
    for (int out_c = tid; out_c < num_out; ++out_c) {
        float sum = bias[out_c];
        
        // Dot product with input channel vector at this pixel
        for (int in_c = 0; in_c < c; ++in_c) {
            // Input Address: b * (C*H*W) + in_c * (H*W) + r * W + c_idx
            int in_addr = b_idx * (c * h * w) + in_c * (h * w) + r * w + c_idx;
            
            // Weight Address: out_c * C + in_c (Weights are Out x In x 1 x 1)
            int w_addr = out_c * c + in_c;
            
            sum += (float)input[in_addr] * (float)weights[w_addr];
        }
        
        // Store result
        int out_plane = out_c % c;
        int out_type = out_c / c; // 0=Q, 1=K, 2=V
        
        int out_addr = b_idx * (c * h * w) + out_plane * (h * w) + r * w + c_idx;
        
        if (out_type == 0) q[out_addr] = (T)sum;
        else if (out_type == 1) k[out_addr] = (T)sum;
        else v[out_addr] = (T)sum;
    }
}

// 2. Depthwise Conv Kernel: V -> V_dw
// 7x7 Convolution, Pad=3
template <typename T>
__global__ void dw_conv_kernel(int n, int c, int h, int w,
                               const T* __restrict__ input,
                               const T* __restrict__ weights,
                               const T* __restrict__ bias,
                               T* __restrict__ output)
{
    // Grid: (N, H, W), Block: C
    int b_idx = blockIdx.z;
    int r = blockIdx.y;
    int c_idx = blockIdx.x;
    
    int channel = threadIdx.x;
    if (channel >= c) return;

    // DW Weights: (C, 1, 7, 7) -> (C, 7, 7)
    // Bias: (C)
    
    float sum = (float)bias[channel];
    
    int k_size = 7;
    int pad = 3;
    
    for (int ky = 0; ky < k_size; ++ky) {
        for (int kx = 0; kx < k_size; ++kx) {
            int in_r = r + ky - pad;
            int in_c = c_idx + kx - pad;
            
            if (in_r >= 0 && in_r < h && in_c >= 0 && in_c < w) {
                int in_addr = b_idx * (c * h * w) + channel * (h * w) + in_r * w + in_c;
                int w_addr = channel * (k_size * k_size) + ky * k_size + kx;
                sum += (float)input[in_addr] * (float)weights[w_addr];
            }
        }
    }
    
    int out_addr = b_idx * (c * h * w) + channel * (h * w) + r * w + c_idx;
    output[out_addr] = (T)sum;
}

// 3. Attention Kernels (Naive)
// A = Q * K^T
// Shape: (HW, C) * (C, HW) -> (HW, HW)
__global__ void qk_matmul_kernel(int n, int hw, int c, 
                                 const float* __restrict__ Q, 
                                 const float* __restrict__ K, 
                                 float* __restrict__ AttnMap,
                                 float scale)
{
    // Grid: N, HW, HW (or tiled)
    // Let's perform one dot product per thread (Naive)
    // Global Idx: batch, row_q (pixel), col_k (pixel)
    
    int b = blockIdx.z;
    int i = blockIdx.y; // 0..HW-1
    int j = blockIdx.x * blockDim.x + threadIdx.x; // 0..HW-1
    
    if (j >= hw) return;
    
    float sum = 0.0f;
    for (int k = 0; k < c; ++k) {
        // Q: N, C, HW -> Q[b, k, i] (Assuming NCHW layout means channel first)
        // K: N, C, HW -> K[b, k, j]
        int q_idx = b * (c * hw) + k * hw + i;
        int k_idx = b * (c * hw) + k * hw + j;
        sum += Q[q_idx] * K[k_idx];
    }
    
    int out_idx = b * (hw * hw) + i * hw + j;
    AttnMap[out_idx] = sum * scale;
}

// Softmax Row-wise
__global__ void softmax_kernel(int n, int hw, float* __restrict__ map)
{
    int b = blockIdx.y;
    int row = blockIdx.x; // 0..HW-1
    
    int row_offset = b * (hw * hw) + row * hw;
    
    // Find Max
    float max_val = -1e20f;
    for (int j = 0; j < hw; ++j) {
        max_val = max(max_val, map[row_offset + j]);
    }
    
    // Exp sum
    float sum = 0.0f;
    for (int j = 0; j < hw; ++j) {
        float val = expf(map[row_offset + j] - max_val);
        map[row_offset + j] = val;
        sum += val;
    }
    
    // Normalize
    for (int j = 0; j < hw; ++j) {
        map[row_offset + j] /= sum;
    }
}

// O = AttnMap * V
// Shape: (HW, HW) * (HW, C) -> (HW, C) (Output is Transposed NCHW again)
__global__ void av_matmul_kernel(int n, int hw, int c, 
                                 const float* __restrict__ AttnMap, 
                                 const float* __restrict__ V, 
                                 float* __restrict__ Output)
{
    int b = blockIdx.z;
    int channel = blockIdx.y; // 0..C-1
    int pixel = blockIdx.x * blockDim.x + threadIdx.x; // 0..HW-1
    
    if (pixel >= hw) return;
    
    float sum = 0.0f;
    for (int k = 0; k < hw; ++k) {
        // Attn: b, pixel, k
        int a_idx = b * (hw * hw) + pixel * hw + k;
        // V: b, channel, k (NCHW)
        int v_idx = b * (c * hw) + channel * hw + k;
        sum += AttnMap[a_idx] * V[v_idx];
    }
    
    // Output: b, channel, pixel
    int out_idx = b * (c * hw) + channel * hw + pixel;
    Output[out_idx] = sum;
}

// 4. Residual Add & Output Proj
// Out = (AttnOut + V_dw) * W3 + B3
template <typename T>
__global__ void output_proj_kernel(int n, int c, int h, int w,
                                   const T* __restrict__ attn_out,
                                   const T* __restrict__ v_dw,
                                   const T* __restrict__ weights,
                                   const T* __restrict__ bias,
                                   T* __restrict__ final_out)
{
    // Grid: (N, H, W), Block: C
    int b_idx = blockIdx.z;
    int r = blockIdx.y;
    int c_idx = blockIdx.x;
    
    // Thread processes one output channel
    int tid = threadIdx.x;
    int stride = blockDim.x;
    
    for (int out_c = tid; out_c < c; out_c += stride) {
        float sum = (float)bias[out_c];
        
        // Dot product over C input channels
        for (int in_c = 0; in_c < c; ++in_c) {
            int px_idx = b_idx * (c * h * w) + in_c * (h * w) + r * w + c_idx;
            float val = (float)attn_out[px_idx] + (float)v_dw[px_idx]; // Residual add
            
            // Weight: out_c, in_c
            int w_idx = out_c * c + in_c;
            sum += val * (float)weights[w_idx];
        }
        
        int out_idx = b_idx * (c * h * w) + out_c * (h * w) + r * w + c_idx;
        final_out[out_idx] = (T)sum;
    }
}


int computeFusedAttn(cudaStream_t stream, 
    int batchSize, int seqLen, int hiddenDim, 
    const float* input, 
    const float* w1, const float* b1, 
    const float* w2, const float* b2, 
    const float* w3, const float* b3, 
    float* output, 
    void* workspace, 
    float attnScale)
{
    int h = (int)sqrt(seqLen); // Assuming square image for simple reconstruction
    int w = seqLen / h;
    int c = hiddenDim;
    int n = batchSize;
    
    // Workspace layout
    size_t sizeImage = n * c * seqLen * sizeof(float);
    size_t sizeAttn = n * seqLen * seqLen * sizeof(float);
    
    float* d_Q = (float*)workspace;
    float* d_K = d_Q + (n*c*seqLen);
    float* d_V = d_K + (n*c*seqLen);
    float* d_Vdw = d_V + (n*c*seqLen);
    float* d_AttnMap = d_Vdw + (n*c*seqLen);
    float* d_AttnOut = d_AttnMap + (n*seqLen*seqLen);

    // 1. QKV Proj
    dim3 grid1(w, h, n);
    dim3 block1(256); // Iterate channels inside
    qkv_proj_kernel<<<grid1, block1, 0, stream>>>(n, c, h, w, input, w1, b1, d_Q, d_K, d_V);
    
    // 2. DW Conv on V -> Vdw
    dim3 block2(c <= 1024 ? c : 1024);
    dw_conv_kernel<<<grid1, block2, 0, stream>>>(n, c, h, w, d_V, w2, b2, d_Vdw);

    // 3. Attention
    // 3a. Q * K^T
    dim3 grid3((seqLen + 255)/256, seqLen, n);
    dim3 block3(256);
    qk_matmul_kernel<<<grid3, block3, 0, stream>>>(n, seqLen, c, d_Q, d_K, d_AttnMap, attnScale);
    
    // 3b. Softmax
    dim3 gridSoft(seqLen, n);
    softmax_kernel<<<gridSoft, 1, 0, stream>>>(n, seqLen, d_AttnMap); // Single thread per row for simplicity/safety
    
    // 3c. AttnMap * Vdw (Note: Using Vdw here? Graph says V goes to DW, but logic implies Attn * V)
    // Actually graph says: Attn(Q,K) * V_dw.
    // Wait, graph shows V branch goes to DW Conv (7x7), then that result is used in MatMul.
    // So yes, we multiply AttnMap by V_dw.
    dim3 grid4((seqLen + 255)/256, c, n);
    av_matmul_kernel<<<grid4, block3, 0, stream>>>(n, seqLen, c, d_AttnMap, d_Vdw, d_AttnOut);
    
    // 4. Final Projection
    output_proj_kernel<<<grid1, block2, 0, stream>>>(n, c, h, w, d_AttnOut, d_Vdw, w3, b3, output);

    return 0;
}

} // namespace custom