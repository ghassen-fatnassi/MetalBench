// fused_attn_op.cc
#include "fused_attn_op.h"
#include "core/framework/tensor.h"
#include "core/framework/kernel_registry.h"
#include "core/providers/cuda/cuda_kernel.h" // For CUDA provider specifics

using namespace onnxruntime;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace custom_op {

// --- 1. Kernel Implementation ---
// Implement the compute method for the float (T=float) version of the kernel.
// For simplicity, we keep the original no-op (copy input to output).
template <>
Status FusedAttnKernel<float>::Compute(OpKernelContext* context) const {
    // Get input tensor
    const Tensor* input_tensor = context->Input<Tensor>(0);
    const float* input_data = input_tensor->Data<float>();

    // Create output tensor with same shape
    const TensorShape& shape = input_tensor->Shape();
    Tensor* output_tensor = context->Output(0, shape);
    float* output_data = output_tensor->MutableData<float>();

    // No-op: just copy input to output
    size_t total_len = shape.Size();
    std::copy(input_data, input_data + total_len, output_data);

    // Replace with actual GPU kernel launch later. This part is typically where
    // you would call a CUDA function, managing memory and streams appropriately.

    return Status::OK();
}

// --- 2. Schema Definition ---
ONNX_NAMESPACE::OpSchema GetFusedAttnSchema() {
    ONNX_NAMESPACE::OpSchema schema("FusedAttnOp", "custom.attn", 1); // Domain must match main.cc
    schema.Input(0, "X", "Input tensor (batch, N, res, res)", "T");
    schema.Output(0, "Y", "Output tensor (same shape as X)", "T");
    schema.TypeConstraint(
        "T",
        OpSchema::numeric_types_for_math_reduction(),
        "Constrain input and output types to high-precision numeric tensors.");
    schema.SinceVersion(1); // Start versioning from 1
    return schema;
}

// --- 3. Kernel Definition ---
KernelDefBuilder FusedAttnKernelDef() {
    KernelDefBuilder def;
    def.SetName("FusedAttnOp")
        .SetDomain("custom.attn")
        .SinceVersion(1)
        // IMPORTANT: Set to CUDA execution provider as specified in original file
        .Provider(kCudaExecutionProvider) 
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>());
    return def;
}

// Function to create the kernel instance
OpKernel* CreateFusedAttnKernel(const OpKernelInfo& kernel_info) {
    return new FusedAttnKernel<float>(kernel_info);
}

// --- 4. Registration Function ---
// This function registers both the schema and the kernel definition with the registry.
Status RegisterFusedAttnCustomOps(onnxruntime::CustomRegistry& registry) {
    // Register the schema
    auto schema = GetFusedAttnSchema();
    std::vector<OpSchema> schemas = {schema};
    // The opset must match the version in the model if it's used there
    ORT_RETURN_IF_ERROR(registry.RegisterOpSet(schemas, "custom.attn", 1, 10)); // Register opset version 1 to 10

    // Register the kernel
    auto def = FusedAttnKernelDef();
    return registry.RegisterCustomKernel(def, CreateFusedAttnKernel);
}

} // namespace custom_op
} // namespace onnxruntime