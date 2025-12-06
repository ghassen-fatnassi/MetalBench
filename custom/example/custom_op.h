#pragma once
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <vector>
#include <cuda_runtime.h>
#include <algorithm>

// -------------------------------
// CUDA kernel declaration
// -------------------------------
void SimpleReLUAddKernelLaunch(cudaStream_t stream,
                               const float* input1,
                               const float* input2,
                               float* output,
                               size_t size);

// -------------------------------
// Kernel class
// -------------------------------
struct SimpleReLUAddOpKernel {
    SimpleReLUAddOpKernel(const OrtApi& api, const OrtKernelInfo* /*info*/) : api_(api) {}

    void Compute(OrtKernelContext* context);

    const OrtApi& api_;
};

// -------------------------------
// Custom op class
// -------------------------------
struct SimpleReLUAddOp : Ort::CustomOpBase<SimpleReLUAddOp, SimpleReLUAddOpKernel> {

    void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
        return new SimpleReLUAddOpKernel(api, info);
    }

    const char* GetName() const { return "SimpleReLUAdd"; }

    const char* GetExecutionProviderType() const { return "CUDAExecutionProvider"; }

    size_t GetInputTypeCount() const { return 2; }

    ONNXTensorElementDataType GetInputType(size_t /*index*/) const {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }

    size_t GetOutputTypeCount() const { return 1; }

    ONNXTensorElementDataType GetOutputType(size_t /*index*/) const {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
};
