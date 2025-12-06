#pragma once
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

struct SimpleReLUAddOpKernel;

// Provide the full struct definition
struct SimpleReLUAddOpKernel {
    SimpleReLUAddOpKernel(const OrtApi& api, const OrtKernelInfo* info) : api_(api) {}
    void Compute(OrtKernelContext* context);

private:
    const OrtApi& api_;
};
struct SimpleReLUAddOp : Ort::CustomOpBase<SimpleReLUAddOp, SimpleReLUAddOpKernel> {
    void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const;
    const char* GetName() const { return "SimpleReLUAdd"; }
    const char* GetExecutionProviderType() const { return "CUDAExecutionProvider"; }

    size_t GetInputTypeCount() const { return 2; }
    ONNXTensorElementDataType GetInputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }

    size_t GetOutputTypeCount() const { return 1; }
    ONNXTensorElementDataType GetOutputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
};

void RegisterSimpleReLUAdd(Ort::CustomOpDomain& domain);
