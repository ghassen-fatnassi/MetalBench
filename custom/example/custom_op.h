#pragma once
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <vector>

struct SimpleReLUAddOpKernel; // forward

// Custom Op definition: note both template params: Op and Kernel
struct SimpleReLUAddOp : Ort::CustomOpBase<SimpleReLUAddOp, SimpleReLUAddOpKernel> {
  SimpleReLUAddOp(const char* provider = "CUDAExecutionProvider") : provider_(provider) {}

  // CreateKernel signature: pass OrtApi and OrtKernelInfo (docs example)
  void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
    return new SimpleReLUAddOpKernel(api, info);
  }

  const char* GetName() const { return "SimpleReLUAdd"; }
  const char* GetExecutionProviderType() const { return provider_; }

  size_t GetInputTypeCount() const { return 2; }
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }

  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }

 private:
  const char* provider_;
};

// Helper to register
void RegisterSimpleReLUAdd(Ort::CustomOpDomain& domain);
