#pragma once
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

struct FusedAttnOp : Ort::CustomOpBase<FusedAttnOp, FusedAttnOp> {
    FusedAttnOp();
    void Compute(OrtKernelContext* context);
    const char* GetName() const { return "FusedAttnOp"; }
    const char* GetExecutionProviderType() const { return "CUDA"; }
    size_t GetInputTypeCount() const { return 1; }
    ONNXTensorElementDataType GetInputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
    size_t GetOutputTypeCount() const { return 1; }
    ONNXTensorElementDataType GetOutputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
};
