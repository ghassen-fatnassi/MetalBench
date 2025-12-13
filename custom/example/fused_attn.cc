#include "onnxruntime_cxx_api.h"
#include "onnxruntime/core/framework/op_kernel.h"
#include "onnxruntime/core/framework/customregistry.h"
#include <cuda_runtime.h>

using namespace onnxruntime;
using namespace onnxruntime::common;

// Forward declaration of your CUDA launcher
void LaunchFusedAttnKernel(const float* input, float* output, size_t count, cudaStream_t stream);

class FusedAttnKernel : public OpKernel {
 public:
  FusedAttnKernel(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* context) const override {
    const auto* X = context->Input<Tensor>(0);
    const auto& shape = X->Shape();
    const float* X_Data = X->Data<float>();
    
    auto* Y = context->Output(0, shape);
    float* Y_Data = Y->MutableData<float>();

    // Get the stream directly from the context
    // This doesn't need the CUDA Provider header, just cuda_runtime.h
    void* stream_ptr = context->GetComputeStream();
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

    LaunchFusedAttnKernel(X_Data, Y_Data, shape.Size(), stream);

    return Status::OK();
  }
};

OpKernel* CreateFusedAttnKernel(const OpKernelInfo& info) {
  return new FusedAttnKernel(info);
}

KernelDefBuilder FusedAttnKernelDef() {
  KernelDefBuilder def;
  def.SetName("FusedAttnOp")             
     .SetDomain("custom.attn")           
     .SinceVersion(1)                   
     .Provider(onnxruntime::kCudaExecutionProvider) 
     .TypeConstraint("T", DataTypeImpl::GetTensorType<float>());
  return def;
}

ONNX_NAMESPACE::OpSchema GetFusedAttnSchema() {
  ONNX_NAMESPACE::OpSchema schema("FusedAttnOp", __FILE__, __LINE__);
  schema.SetDomain("custom.attn"); 
  schema.Input(0, "input", "Input tensor", "T");
  schema.Output(0, "output", "Output tensor", "T");
  schema.TypeConstraint("T", {"tensor(float)"}, "float");
  schema.SinceVersion(1);
  return schema;
}

// Ensure this function is exported for dlopen/dlsym
extern "C" {
    std::shared_ptr<CustomRegistry> RegisterFusedAttnOps() {
      auto registry = std::make_shared<CustomRegistry>();
      std::vector<ONNX_NAMESPACE::OpSchema> schemas = { GetFusedAttnSchema() };
      registry->RegisterOpSet(schemas, "custom.attn", 1, 100);
      registry->RegisterCustomKernel(FusedAttnKernelDef(), CreateFusedAttnKernel);
      return registry;
    }
}