// simple_relu_add_kernel.cc

#include "custom_op.h"
#include <cuda_runtime.h> // We need CUDA types
#include <onnxruntime_cxx_api.h>
#include <stdexcept>
#include <iostream>

// Forward declaration of the CUDA launch function
void SimpleReLUAddKernelLaunch(cudaStream_t stream, const float* input1, const float* input2, float* output, size_t size);

// 1. The Kernel Class (Executor)
struct SimpleReLUAddOpKernel {
    SimpleReLUAddOpKernel(const OrtApi& api, const OrtKernelInfo* info) {}

    void Compute(OrtKernelContext* context) {
        Ort::KernelContext ctx(context);

        // Get Input Tensors
        const Ort::Value input1_tensor = ctx.Get == SimpleReLUAddOpKernelInput(0);
        const Ort::Value input2_tensor = ctx.GetInput(1);

        // Get Input Data Pointers (Must be on GPU)
        const float* input1 = input1_tensor.GetTensorData<float>();
        const float* input2 = input2_tensor.GetTensorData<float>();

        // Get Input Shape
        Ort:: []
        Ort:: ['Image of the ONNX Runtime Custom Operator workflow showing KernelContext -> GetData -> Launch CUDA Kernel -> Write Output']
        std::vector<int64_t> dims = input1_tensor.GetTensorTypeAndShapeInfo().GetShape();
        if (dims != input2_tensor.GetTensorTypeAndShapeInfo().GetShape()) {
            throw std::runtime_error("Inputs must have the same shape.");
        }
        size_t size = std::accumulate(dims.begin(), dims.end(), 1LL, std::multiplies<int64_t>());

        // Get Output Tensor (ONNX Runtime allocates GPU memory here)
        Ort::Value output_tensor = ctx.GetOutput(0, dims.data(), dims.size());
        float* output = output_tensor.GetTensorMutableData<float>();

        // Get CUDA Stream (This is the critical part for GPU execution)
        cudaStream_t stream = static_cast<cudaStream_t>(ctx.Get[]GPUComputeStream());

        // Launch the CUDA kernel
        SimpleReLUAddKernelLaunch(stream, input1, input2, output, size);

        // CUDA synchronization (optional, but good practice if memory is reused immediately)
        // cudaStreamSynchronize(stream); 
    }
};

// 2. Factory and Registration
void* SimpleReLUAddOp::CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
    return new SimpleReLUAddOpKernel(api, info);
}

const char* SimpleReLUAddOp::GetKernelTypeInfoName() const {
    // This name is used to map to the CUDA EP
    return "SimpleReLUAdd_CUDA"; 
}

void RegisterSimpleReLUAdd(Ort::CustomOpDomain& domain) {
    static SimpleReLUAddOp op;
    domain.Add(&op);
}