#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cmath>
#include <cfloat>

namespace custom
{

// Constants matching PyTorch model
constexpr int NUM_HEADS = 2;
constexpr int HEAD_DIM = 32;
constexpr int HIDDEN_DIM = 64;  // NUM_HEADS * HEAD_DIM
constexpr int QKV_DIM = 192;    // 3 * HIDDEN_DIM

// ============================================================================
// Kernel 1: QKV Projection (1x1 Conv:  64 -> 192)
// Input:   (N, 64, H, W) in NCHW format
// Output: (N, 192, H, W) in NCHW format
// ============================================================================
template <typename T>
__global__ void qkv_projection_kernel(
    int N, int C_in, int C_out, int H, int W,
    const T* __restrict__ input,
    const T* __restrict__ weights,  // (192, 64, 1, 1)
    const T* __restrict__ bias,     // (192,)
    T* __restrict__ output)
{
    // Each thread computes one output element
    int idx = blockIdx.x * blockDim. x + threadIdx. x;
    int total = N * C_out * H * W;
    
    if (idx >= total) return;
    
    // Decode index:  output is NCHW
    int w_idx = idx % W;
    int h_idx = (idx / W) % H;
    int c_out = (idx / (W * H)) % C_out;
    int n = idx / (W * H * C_out);
    
    float sum = (float)bias[c_out];
    
    // 1x1 conv is just a dot product over input channels at this spatial location
    for (int c_in = 0; c_in < C_in; ++c_in) {
        int in_idx = n * (C_in * H * W) + c_in * (H * W) + h_idx * W + w_idx;
        int w_idx_addr = c_out * C_in + c_in;  // weights:  (C_out, C_in)
        sum += (float)input[in_idx] * (float)weights[w_idx_addr];
    }
    
    output[idx] = (T)sum;
}

// ============================================================================
// Kernel 2: Reshape QKV and compute attention
// This kernel implements the complex reshape/permute/split/attention logic
// 
// PyTorch operations:
//   qkv:  (1, 192, H, W)
//   x1 = qkv.view(1, 192, HW)
//   x2 = x1.transpose(1, 2)           -> (1, HW, 192)
//   x3 = x2.view(1, HW, 2, 96)        -> (1, HW, 2, 96)
//   x4 = x3.permute(0, 2, 3, 1)       -> (1, 2, 96, HW)
//   q_raw, k_raw, v_raw = split(x4, 32, dim=2)  -> each (1, 2, 32, HW)
//   q = q_raw.transpose(2, 3)         -> (1, 2, HW, 32)
//   k = k_raw.transpose(2, 3)         -> (1, 2, HW, 32)
//   v = v_raw.transpose(2, 3)         -> (1, 2, HW, 32)
//   attn = (q @ k.T) * scale          -> (1, 2, HW, HW)
//   attn = softmax(attn, dim=-1)
//   attn_perm = attn. permute(0,1,3,2) -> (1, 2, HW, HW)
//   x_attn_raw = v_raw @ attn_perm    -> (1, 2, 32, HW)
// ============================================================================

// Helper:  Get QKV value after all the reshapes
// qkv is (N, 192, H, W) in NCHW
// After reshapes, we want to access as (N, 2, 96, HW) then split into Q/K/V
__device__ inline float get_qkv_reshaped(
    const float* qkv, 
    int n, int head, int channel_in_head, int spatial,
    int H, int W)
{
    // x4[n, head, channel_in_head, spatial] comes from: 
    // x3[n, spatial, head, channel_in_head]
    // x2[n, spatial, head * 96 + channel_in_head]
    // x1[n, head * 96 + channel_in_head, spatial]
    // qkv[n, head * 96 + channel_in_head, h, w] where spatial = h * W + w
    
    int qkv_channel = head * 96 + channel_in_head;
    int h = spatial / W;
    int w = spatial % W;
    int HW = H * W;
    
    return qkv[n * (192 * HW) + qkv_channel * HW + h * W + w];
}

// Kernel to compute Q @ K^T for attention scores
// Q, K are conceptually (N, NUM_HEADS, HW, HEAD_DIM) after transpose
// But we read directly from QKV buffer using the reshape logic
// Output: attn_scores (N, NUM_HEADS, HW, HW)
__global__ void compute_attention_scores_kernel(
    int N, int H, int W,
    const float* __restrict__ qkv,
    float* __restrict__ attn_scores,
    float scale)
{
    int HW = H * W;
    
    // Each thread computes one attention score
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * NUM_HEADS * HW * HW;
    
    if (idx >= total) return;
    
    // Decode:  attn_scores[n, head, i, j]
    int j = idx % HW;
    int i = (idx / HW) % HW;
    int head = (idx / (HW * HW)) % NUM_HEADS;
    int n = idx / (HW * HW * NUM_HEADS);
    
    // Q is at channels 0-31 in the 96-channel head slice (q_raw from split)
    // K is at channels 32-63 in the 96-channel head slice (k_raw from split)
    // 
    // q_raw[n, head, : , i] = x4[n, head, 0:32, i]
    // k_raw[n, head, :, j] = x4[n, head, 32:64, j]
    // 
    // After transpose: q[n, head, i, : ] and k[n, head, j, :]
    // Dot product: sum over HEAD_DIM
    
    float sum = 0.0f;
    for (int d = 0; d < HEAD_DIM; ++d) {
        // Q channel offset in 96-dim:  0 + d
        float q_val = get_qkv_reshaped(qkv, n, head, d, i, H, W);
        // K channel offset in 96-dim: 32 + d
        float k_val = get_qkv_reshaped(qkv, n, head, 32 + d, j, H, W);
        sum += q_val * k_val;
    }
    
    attn_scores[idx] = sum * scale;
}

// Kernel for row-wise softmax on attention scores
// attn_scores: (N, NUM_HEADS, HW, HW)
// Softmax over last dimension (j)
__global__ void softmax_attention_kernel(
    int N, int H, int W,
    float* __restrict__ attn_scores)
{
    int HW = H * W;
    
    // Each thread handles one row
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_rows = N * NUM_HEADS * HW;
    
    if (row_idx >= total_rows) return;
    
    int i = row_idx % HW;
    int head = (row_idx / HW) % NUM_HEADS;
    int n = row_idx / (HW * NUM_HEADS);
    
    int row_offset = n * (NUM_HEADS * HW * HW) + head * (HW * HW) + i * HW;
    
    // Find max for numerical stability
    float max_val = -FLT_MAX;
    for (int j = 0; j < HW; ++j) {
        max_val = fmaxf(max_val, attn_scores[row_offset + j]);
    }
    
    // Compute exp and sum
    float sum = 0.0f;
    for (int j = 0; j < HW; ++j) {
        float val = expf(attn_scores[row_offset + j] - max_val);
        attn_scores[row_offset + j] = val;
        sum += val;
    }
    
    // Normalize
    float inv_sum = 1.0f / sum;
    for (int j = 0; j < HW; ++j) {
        attn_scores[row_offset + j] *= inv_sum;
    }
}

// Kernel to compute attention output:  v_raw @ attn. permute(0,1,3,2)
// v_raw: (N, NUM_HEADS, HEAD_DIM, HW) conceptually
// attn after permute: (N, NUM_HEADS, HW, HW) -> accessing as attn[n,h,j,i] 
// Output: x_attn_raw (N, NUM_HEADS, HEAD_DIM, HW)
// 
// x_attn_raw[n, head, d, i] = sum_j( v_raw[n, head, d, j] * attn_perm[n, head, d, j, i] )
//                           = sum_j( v_raw[n, head, d, j] * attn[n, head, i, j] )  <- after permute swap
// Wait, let me re-check: 
// attn_perm = attn.permute(0,1,3,2) means attn_perm[n,h,i,j] = attn[n,h,j,i]
// x_attn_raw = v_raw @ attn_perm
// v_raw is (1,2,32,HW), attn_perm is (1,2,HW,HW)
// Matmul: (32, HW) @ (HW, HW) -> (32, HW)
// x_attn_raw[n,h,d,i] = sum_j v_raw[n,h,d,j] * attn_perm[n,h,j,i]
//                     = sum_j v_raw[n,h,d,j] * attn[n,h,i,j]
__global__ void apply_attention_kernel(
    int N, int H, int W,
    const float* __restrict__ qkv,
    const float* __restrict__ attn_scores,
    float* __restrict__ attn_output)  // (N, NUM_HEADS, HEAD_DIM, HW)
{
    int HW = H * W;
    
    // Each thread computes one output element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * NUM_HEADS * HEAD_DIM * HW;
    
    if (idx >= total) return;
    
    // Decode: attn_output[n, head, d, i]
    int i = idx % HW;
    int d = (idx / HW) % HEAD_DIM;
    int head = (idx / (HW * HEAD_DIM)) % NUM_HEADS;
    int n = idx / (HW * HEAD_DIM * NUM_HEADS);
    
    float sum = 0.0f;
    for (int j = 0; j < HW; ++j) {
        // v_raw[n, head, d, j] -> V channel offset in 96-dim:  64 + d
        float v_val = get_qkv_reshaped(qkv, n, head, 64 + d, j, H, W);
        
        // attn[n, head, i, j] (we want attn_perm[n,h,j,i] = attn[n,h,i,j])
        int attn_idx = n * (NUM_HEADS * HW * HW) + head * (HW * HW) + i * HW + j;
        float attn_val = attn_scores[attn_idx];
        
        sum += v_val * attn_val;
    }
    
    attn_output[idx] = sum;
}

// Kernel to reshape attention output to image format
// Input: attn_output (N, NUM_HEADS, HEAD_DIM, HW) 
// This is x_attn_raw in PyTorch
// 
// PyTorch:  x_attn = x_attn_raw.permute(0,3,1,2)  -> (1, HW, 2, 32)
//          x_attn = x_attn. reshape(1, 40, 40, 64).permute(0, 3, 1, 2)  -> (1, 64, 40, 40)
// 
// So final layout is (N, 64, H, W) where:
// output[n, c, h, w] where c = head * HEAD_DIM + d
// We need to trace back: 
// x_attn after first permute: (N, HW, NUM_HEADS, HEAD_DIM)
// x_attn[n, spatial, head, d] = x_attn_raw[n, head, d, spatial]
// After reshape to (N, H, W, 64): x_attn[n, h, w, head*32+d]
// After permute to (N, 64, H, W): output[n, head*32+d, h, w]
__global__ void reshape_attn_output_kernel(
    int N, int H, int W,
    const float* __restrict__ attn_output,  // (N, NUM_HEADS, HEAD_DIM, HW)
    float* __restrict__ output)              // (N, 64, H, W)
{
    int HW = H * W;
    
    int idx = blockIdx. x * blockDim.x + threadIdx.x;
    int total = N * HIDDEN_DIM * HW;
    
    if (idx >= total) return;
    
    // Decode output[n, c, h, w]
    int w_idx = idx % W;
    int h_idx = (idx / W) % H;
    int c = (idx / HW) % HIDDEN_DIM;
    int n = idx / (HW * HIDDEN_DIM);
    
    int head = c / HEAD_DIM;
    int d = c % HEAD_DIM;
    int spatial = h_idx * W + w_idx;
    
    // Read from attn_output[n, head, d, spatial]
    int in_idx = n * (NUM_HEADS * HEAD_DIM * HW) + head * (HEAD_DIM * HW) + d * HW + spatial;
    
    output[idx] = attn_output[in_idx];
}

// ============================================================================
// Kernel 3: Build V image for PE branch
// v = v_raw. transpose(2, 3)  -> (1, 2, HW, 32)
// v_img = v.transpose(1, 2).reshape(1, H, W, 64).permute(0, 3, 1, 2) -> (1, 64, H, W)
// 
// Let's trace: 
// v[n, head, spatial, d] = v_raw[n, head, d, spatial]
// After transpose(1,2): (1, HW, 2, 32) -> temp[n, spatial, head, d] = v[n, head, spatial, d]
// After reshape: (1, H, W, 64) -> temp2[n, h, w, head*32+d]
// After permute: (1, 64, H, W) -> v_img[n, head*32+d, h, w]
// 
// So:  v_img[n, c, h, w] = v_raw[n, head, d, h*W+w] where c = head*32 + d
// ============================================================================
__global__ void build_v_image_kernel(
    int N, int H, int W,
    const float* __restrict__ qkv,
    float* __restrict__ v_img)  // (N, 64, H, W)
{
    int HW = H * W;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * HIDDEN_DIM * HW;
    
    if (idx >= total) return;
    
    // Decode v_img[n, c, h, w]
    int w_idx = idx % W;
    int h_idx = (idx / W) % H;
    int c = (idx / HW) % HIDDEN_DIM;
    int n = idx / (HW * HIDDEN_DIM);
    
    int head = c / HEAD_DIM;
    int d = c % HEAD_DIM;
    int spatial = h_idx * W + w_idx;
    
    // v_raw channel offset in 96-dim: 64 + d
    float val = get_qkv_reshaped(qkv, n, head, 64 + d, spatial, H, W);
    
    v_img[idx] = val;
}

// ============================================================================
// Kernel 4: Depthwise 7x7 Convolution for PE
// Input:   (N, 64, H, W)
// Output: (N, 64, H, W)
// Weights: (64, 1, 7, 7)
// Bias: (64,)
// ============================================================================
template <typename T>
__global__ void depthwise_conv_7x7_kernel(
    int N, int C, int H, int W,
    const T* __restrict__ input,
    const T* __restrict__ weights,  // (C, 1, 7, 7) -> (C, 49)
    const T* __restrict__ bias,     // (C,)
    T* __restrict__ output)
{
    int idx = blockIdx.x * blockDim. x + threadIdx. x;
    int total = N * C * H * W;
    
    if (idx >= total) return;
    
    // Decode output[n, c, h, w]
    int w_idx = idx % W;
    int h_idx = (idx / W) % H;
    int c = (idx / (H * W)) % C;
    int n = idx / (H * W * C);
    
    const int K = 7;
    const int PAD = 3;
    
    float sum = (float)bias[c];
    
    for (int ky = 0; ky < K; ++ky) {
        for (int kx = 0; kx < K; ++kx) {
            int in_h = h_idx + ky - PAD;
            int in_w = w_idx + kx - PAD;
            
            if (in_h >= 0 && in_h < H && in_w >= 0 && in_w < W) {
                int in_idx = n * (C * H * W) + c * (H * W) + in_h * W + in_w;
                int w_offset = c * (K * K) + ky * K + kx;
                sum += (float)input[in_idx] * (float)weights[w_offset];
            }
        }
    }
    
    output[idx] = (T)sum;
}

// ============================================================================
// Kernel 5: Final projection with residual add
// Output = proj(attn_out + pe_out)
// 1x1 Conv: (N, 64, H, W) -> (N, 64, H, W)
// ============================================================================
template <typename T>
__global__ void final_projection_kernel(
    int N, int C, int H, int W,
    const T* __restrict__ attn_out,
    const T* __restrict__ pe_out,
    const T* __restrict__ weights,  // (64, 64, 1, 1)
    const T* __restrict__ bias,     // (64,)
    T* __restrict__ output)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    
    if (idx >= total) return;
    
    // Decode output[n, c_out, h, w]
    int w_idx = idx % W;
    int h_idx = (idx / W) % H;
    int c_out = (idx / (H * W)) % C;
    int n = idx / (H * W * C);
    
    float sum = (float)bias[c_out];
    
    // 1x1 conv over input channels
    for (int c_in = 0; c_in < C; ++c_in) {
        int in_idx = n * (C * H * W) + c_in * (H * W) + h_idx * W + w_idx;
        
        // Residual add
        float in_val = (float)attn_out[in_idx] + (float)pe_out[in_idx];
        
        // Weight
        int w_offset = c_out * C + c_in;
        sum += in_val * (float)weights[w_offset];
    }
    
    output[idx] = (T)sum;
}

// ============================================================================
// Main compute function
// ============================================================================
int computeFusedAttn(
    cudaStream_t stream,
    int batchSize,      // N
    int height,         // H (e.g., 40)
    int width,          // W (e.g., 40)
    const float* input, // (N, 64, H, W)
    const float* qkv_weights,   // (192, 64, 1, 1)
    const float* qkv_bias,      // (192,)
    const float* pe_weights,    // (64, 1, 7, 7)
    const float* pe_bias,       // (64,)
    const float* proj_weights,  // (64, 64, 1, 1)
    const float* proj_bias,     // (64,)
    float* output,              // (N, 64, H, W)
    void* workspace)
{
    int N = batchSize;
    int H = height;
    int W = width;
    int HW = H * W;
    
    // Scale factor from PyTorch:  0.1767766952966369 = 1/sqrt(32)
    float attn_scale = 1.0f / sqrtf((float)HEAD_DIM);
    
    // Workspace allocation
    // qkv_output: (N, 192, H, W)
    // attn_scores: (N, NUM_HEADS, HW, HW)
    // attn_output: (N, NUM_HEADS, HEAD_DIM, HW)
    // attn_reshaped: (N, 64, H, W)
    // v_img: (N, 64, H, W)
    // pe_out: (N, 64, H, W)
    
    size_t qkv_size = N * QKV_DIM * HW;
    size_t attn_scores_size = N * NUM_HEADS * HW * HW;
    size_t attn_output_size = N * NUM_HEADS * HEAD_DIM * HW;
    size_t image_size = N * HIDDEN_DIM * HW;
    
    float* qkv_output = (float*)workspace;
    float* attn_scores = qkv_output + qkv_size;
    float* attn_output = attn_scores + attn_scores_size;
    float* attn_reshaped = attn_output + attn_output_size;
    float* v_img = attn_reshaped + image_size;
    float* pe_out = v_img + image_size;
    
    const int BLOCK_SIZE = 256;
    
    // Step 1: QKV Projection
    {
        int total = N * QKV_DIM * HW;
        int grid = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
        qkv_projection_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
            N, HIDDEN_DIM, QKV_DIM, H, W,
            input, qkv_weights, qkv_bias, qkv_output);
    }
    
    // Step 2: Compute attention scores (Q @ K^T * scale)
    {
        int total = N * NUM_HEADS * HW * HW;
        int grid = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
        compute_attention_scores_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
            N, H, W, qkv_output, attn_scores, attn_scale);
    }
    
    // Step 3: Softmax on attention scores
    {
        int total_rows = N * NUM_HEADS * HW;
        int grid = (total_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
        softmax_attention_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
            N, H, W, attn_scores);
    }
    
    // Step 4: Apply attention (v_raw @ attn_permuted)
    {
        int total = N * NUM_HEADS * HEAD_DIM * HW;
        int grid = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
        apply_attention_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
            N, H, W, qkv_output, attn_scores, attn_output);
    }
    
    // Step 5: Reshape attention output to (N, 64, H, W)
    {
        int total = N * HIDDEN_DIM * HW;
        int grid = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
        reshape_attn_output_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
            N, H, W, attn_output, attn_reshaped);
    }
    
    // Step 6: Build V image for PE path
    {
        int total = N * HIDDEN_DIM * HW;
        int grid = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
        build_v_image_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
            N, H, W, qkv_output, v_img);
    }
    
    // Step 7: Depthwise 7x7 conv for PE
    {
        int total = N * HIDDEN_DIM * HW;
        int grid = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
        depthwise_conv_7x7_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
            N, HIDDEN_DIM, H, W,
            v_img, pe_weights, pe_bias, pe_out);
    }
    
    // Step 8: Final projection with residual
    {
        int total = N * HIDDEN_DIM * HW;
        int grid = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
        final_projection_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
            N, HIDDEN_DIM, H, W,
            attn_reshaped, pe_out, proj_weights, proj_bias, output);
    }
    
    return 0;
}

// Helper function to calculate required workspace size
size_t getWorkspaceSize(int batchSize, int height, int width)
{
    int N = batchSize;
    int HW = height * width;
    
    size_t qkv_size = N * QKV_DIM * HW;
    size_t attn_scores_size = N * NUM_HEADS * HW * HW;
    size_t attn_output_size = N * NUM_HEADS * HEAD_DIM * HW;
    size_t image_size = N * HIDDEN_DIM * HW;
    
    // Total:  qkv + attn_scores + attn_output + attn_reshaped + v_img + pe_out
    size_t total = qkv_size + attn_scores_size + attn_output_size + 3 * image_size;
    
    return total * sizeof(float);
}

} // namespace custom