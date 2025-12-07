#pragma once
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <vector>
#include <cuda_runtime.h>
#include <algorithm>
#include <iostream>

// --- Helper to check ORT Status ---
#define ORT_CHECK(api, status) \
    do { \
        OrtStatus* _s = (status); \
        if (_s != nullptr) { \
            const char* msg = api->GetErrorMessage(_s); \
            std::cerr << "ORT Error: " << msg << " in " << __FILE__ << ":" << __LINE__ << std::endl; \
            api->ReleaseStatus(_s); \
            abort(); \
        } \
    } while(0)

// -------------------------------
// CUDA kernel wrapper declaration
// -------------------------------
void SimpleReLUAddKernelLaunch(cudaStream_t stream,
                               const float* input1,
                               const float* input2,
                               float* output,
                               size_t size);
// -------------------------------
// Kernel class: Implements the Compute logic
// -------------------------------
struct SimpleReLUAddOpKernel {
    // Constructor receives the API handle and optional metadata
    SimpleReLUAddOpKernel(const OrtApi& api, const OrtKernelInfo* /*info*/) : api_(api) {}
    // The method called by ONNX Runtime to execute the operation
    void Compute(OrtKernelContext* context);
    const OrtApi& api_;
};
// -------------------------------
// Custom op class: Defines the operation metadata
// -------------------------------
struct SimpleReLUAddOp : Ort::CustomOpBase<SimpleReLUAddOp, SimpleReLUAddOpKernel> {
    void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
        return new SimpleReLUAddOpKernel(api, info);
    }
    const char* GetName() const { return "SimpleReLUAdd"; }
    // IMPORTANT: Use the same domain you register below (e.g., com.your.custom)
    // For simplicity in this example, we rely on standard ONNX domain if using ORT's internal test mechanism.
    // For your real model, you need a custom domain.
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

// -------------------------------
// Custom Op Library Registration Class
// -------------------------------
class CustomOpLibrary {
public:
    // This is the function ORT expects to register all custom ops in your library.
    OrtStatus* RegisterOps(OrtSessionOptions* options, const OrtApiBase* api_base);
};