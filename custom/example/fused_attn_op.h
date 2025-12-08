#pragma once
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <vector>

// 1. The C++ Kernel implementation (formerly the TKernel template parameter)
class FusedAttnKernel {
public:
    // This constructor matches the expected signature for a CustomOp kernel constructor
    // (Ort::CustomOpApi ort, const OrtKernelInfo* info)
    FusedAttnKernel(const OrtApi& api, const OrtKernelInfo* info);

    // This is the core compute function
    void Compute(OrtKernelContext* context);

private:
    Ort::CustomOpApi api_;
    // Add any private members needed for the kernel here (e.g., attributes)
};

// 2. Declaration of C-style functions that the C API requires for registration
OrtCustomOp* CreateFusedAttnOp();

// These functions will be implemented in the .cc file and assigned to the OrtCustomOp struct
const char* ORT_API_CALL FusedAttnOp_GetName(const void* op);
const char* ORT_API_CALL FusedAttnOp_GetExecutionProviderType(const void* op);
size_t ORT_API_CALL FusedAttnOp_GetInputTypeCount(const void* op);
ONNXTensorElementDataType ORT_API_CALL FusedAttnOp_GetInputType(const void* op, size_t index);
size_t ORT_API_CALL FusedAttnOp_GetOutputTypeCount(const void* op);
ONNXTensorElementDataType ORT_API_CALL FusedAttnOp_GetOutputType(const void* op, size_t index);
void* ORT_API_CALL FusedAttnOp_CreateKernel(const void* op, const OrtApi* api, const OrtKernelInfo* info);
void ORT_API_CALL FusedAttnOp_KernelCompute(void* op_kernel, OrtKernelContext* context);
void ORT_API_CALL FusedAttnOp_KernelDestroy(void* op_kernel);