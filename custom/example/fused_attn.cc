#include "core/framework/op_kernel.h"
#include "core/framework/execution_provider.h"
#include "core/framework/customregistry.h"
#include "core/graph/schema_registry.h"
#include "core/providers/cuda/cuda_execution_provider.h" 

using namespace onnxruntime;
using namespace onnxruntime::common;

// Forward declaration
void LaunchFusedAttnKernel(const float* input, float* output, size_t count, cudaStream_t stream);

// ---------------------------------------------------------------
// 1. The Kernel Implementation
// ---------------------------------------------------------------
class FusedAttnKernel : public OpKernel {
 public:
  FusedAttnKernel(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* context) const override {
    // A. Get Input (Dynamic handling)
    // This works for custom.attn_0 (small inputs) AND custom.attn_7 (huge inputs)
    // because we query the shape at runtime.
    const auto* X = context->Input<Tensor>(0);
    
    const auto& shape = X->Shape();
    const float* X_Data = X->Data<float>();
    size_t count = shape.Size();

    // B. Allocate Output
    // Since this is identity, we set output shape = input shape.
    auto* Y = context->Output(0, shape);
    float* Y_Data = Y->MutableData<float>();

    // C. Get Stream
    cudaStream_t stream = static_cast<cudaStream_t>(context->GetComputeStream());

    // D. Launch
    LaunchFusedAttnKernel(X_Data, Y_Data, count, stream);

    return Status::OK();
  }
};

OpKernel* CreateFusedAttnKernel(const OpKernelInfo& info) {
  return new FusedAttnKernel(info);
}

// ---------------------------------------------------------------
// 2. The Kernel Definition
// ---------------------------------------------------------------
KernelDefBuilder FusedAttnKernelDef() {
  KernelDefBuilder def;
  // **CRITICAL MATCHING**: Matches the 'op_type' and 'domain' from your model list
  def.SetName("FusedAttnOp")             
     .SetDomain("custom.attn")           
     .SinceVersion(1)                   
     .Provider(onnxruntime::kCudaExecutionProvider) 
     .TypeConstraint("T", DataTypeImpl::GetTensorType<float>());
  return def;
}

// ---------------------------------------------------------------
// 3. The Schema Definition
// ---------------------------------------------------------------
ONNX_NAMESPACE::OpSchema GetFusedAttnSchema() {
  ONNX_NAMESPACE::OpSchema schema("FusedAttnOp", __FILE__, __LINE__);
  
  // **CRITICAL MATCHING**: Must match the domain in the model
  schema.SetDomain("custom.attn"); 
  
  schema.Input(0, "input", "Input tensor", "T");
  schema.Output(0, "output", "Output tensor", "T");
  
  // Allow variable input/output types if needed, here constrained to float
  schema.TypeConstraint(
      "T", 
      {"tensor(float)"}, 
      "Constrain to float tensors");
      
  schema.SinceVersion(1);
  return schema;
}

// ---------------------------------------------------------------
// 4. Registration Function (Call this in your Session setup)
// ---------------------------------------------------------------
std::shared_ptr<CustomRegistry> RegisterFusedAttnOps() {
  std::shared_ptr<CustomRegistry> registry = std::make_shared<CustomRegistry>();

  // Register Schema
  std::vector<ONNX_NAMESPACE::OpSchema> schemas = { GetFusedAttnSchema() };
  registry->RegisterOpSet(schemas, "custom.attn", 1, 100);

  // Register Kernel
  registry->RegisterCustomKernel(FusedAttnKernelDef(), CreateFusedAttnKernel);

  return registry;
}