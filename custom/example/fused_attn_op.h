// fused_attn_op.h
#pragma once
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

struct FusedAttnOp : Ort::CustomOpBase<FusedAttnOp, FusedAttnOp> {
    FusedAttnOp();
    
    // The kernel is the op itself in the C++ API wrapper.
    void Compute(OrtKernelContext* context);
    
    // Op properties
    const char* GetName() const { return "FusedAttnOp"; }
    
    // This correctly specifies the CUDA Execution Provider
    const char* GetExecutionProviderType() const { return "CUDA"; } 
    
    // Input/Output definitions
    size_t GetInputTypeCount() const { return 1; }
    ONNXTensorElementDataType GetInputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
    size_t GetOutputTypeCount() const { return 1; }
    ONNXTensorElementDataType GetOutputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
};
// Removed the closing '}' here.