// fused_attn_op.h
#pragma once

#include "onnxruntime/core/framework/op_kernel.h"
#include "core/graph/onnx_protobuf.h"
#include "core/framework/customregistry.h"

// Forward declaration of the OpKernel class
namespace onnxruntime {
namespace custom_op {

// The FusedAttn kernel is templated on the data type (T)
template <typename T>
class FusedAttnKernel : public OpKernel {
public:
    // Constructor requires OpKernelInfo
    FusedAttnKernel(const OpKernelInfo& info) : OpKernel(info) {}

    // The main computation method
    Status Compute(OpKernelContext* context) const override;
};

// Function to register the custom operators (kernels and schema)
Status RegisterFusedAttnCustomOps(onnxruntime::CustomRegistry& registry);

} // namespace custom_op
} // namespace onnxruntime