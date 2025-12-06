// simple_relu_add_kernel.cc
#include "custom_op.h"
#include <cuda_runtime.h>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <stdexcept>
#include <numeric> // Required for std::accumulate

// Forward declaration
void SimpleReLUAddKernelLaunch(cudaStream_t stream, const float* input1, const float* input2, float* output, size_t size);

struct SimpleReLUAddOpKernel {
    SimpleReLUAddOpKernel(const OrtApi& api, const OrtKernelInfo* info) {}

    void Compute(OrtKernelContext* context) {
        Ort::KernelContext ctx(context);

        // 1. Get Input Tensors
        Ort::Value input1_tensor = ctx.GetInput(0);
        Ort::Value input2_tensor = ctx.GetInput(1);

        // 2. Get Data Pointers (GPU)
        const float* input1 = input1_tensor.GetTensorData<float>();
        const float* input2 = input2_tensor.GetTensorData<float>();

        // 3. Shape Validation & Calculation
        auto type_info = input1_tensor.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> dims = type_info.GetShape();
        
        // (Simplified size calc)
        size_t size = 1;
        for (auto dim : dims) size *= dim;

        // 4. Get Output Tensor
        Ort::Value output_tensor = ctx.GetOutput(0, dims.data(), dims.size());
        float* output = output_tensor.GetTensorMutableData<float>();

        // 5. Get CUDA Stream (ORT 1.6 Specific approach)
        // In ORT 1.6, C++ API support for streams in CustomOps was limited.
        // We often rely on the assumption that we use the default stream or 
        // try to extract the resource. For safety in 1.6, we will use the 
        // underlying C API to get the stream handle.
        
        // NOTE: If this fails in 1.6, pass 0 (default stream), 
        // but this is the standard way to try getting the handle:
        cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.GetGPUComputeStream());

        // 6. Launch Kernel
        SimpleReLUAddKernelLaunch(stream, input1, input2, output, size);
    }
};

void* SimpleReLUAddOp::CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
    return new SimpleReLUAddOpKernel(api, info);
}

const char* SimpleReLUAddOp::GetKernelTypeInfoName() const {
    return "SimpleReLUAdd_CUDA"; 
}

void RegisterSimpleReLUAdd(Ort::CustomOpDomain& domain) {
    static SimpleReLUAddOp op;
    domain.Add(&op);
}