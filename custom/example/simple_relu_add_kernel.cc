#include "custom_op.h"
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <stdexcept>
#include <vector>
#include <numeric>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

// forward of the CUDA launcher
extern void SimpleReLUAddKernelLaunch(cudaStream_t stream, const float* input1, const float* input2, float* output, size_t size);

struct SimpleReLUAddOpKernel {
  // Constructor: store the OrtApi reference if needed
  SimpleReLUAddOpKernel(const OrtApi& api, const OrtKernelInfo* /*info*/) : api_(api) {}

  void Compute(OrtKernelContext* context) {
    Ort::KernelContext ctx(context);

    // get inputs
    Ort::Value input1_tensor = ctx.GetInput(0);
    Ort::Value input2_tensor = ctx.GetInput(1);

    // shape & size calculation (supporting N-D)
    auto shape_info = input1_tensor.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> dims = shape_info.GetShape();
    size_t size = 1;
    for (auto d : dims) {
      if (d < 0) throw std::runtime_error("dynamic dims not supported in this example");
      size *= static_cast<size_t>(d);
    }

    // output tensor (same shape)
    Ort::Value output_tensor = ctx.GetOutput(0, dims.data(), dims.size());

    // Get raw pointers (note: if execution provider is CUDA, these may already be device pointers)
    const float* input1 = input1_tensor.GetTensorData<float>();
    const float* input2 = input2_tensor.GetTensorData<float>();
    float* output = output_tensor.GetTensorMutableData<float>();

    // Try to obtain a CUDA stream if available. If not, use default stream (0).
    cudaStream_t stream = 0;
#ifdef USE_CUDA
    // The C++ KernelContext wrapper doesn't expose a typed GetGPUComputeStream in all ORT versions.
    // If the pointer is available via the C API, you could retrieve it; for portability, use default stream.
    // If you have a specific ORT version that exposes the stream, replace this with the proper call.
    // For now we'll use the default stream which is valid for most simple tests.
    stream = 0;
#endif

    // Launch fused CUDA kernel (works on host pointers if ORT copied to device — ORT will manage)
    SimpleReLUAddKernelLaunch(stream, input1, input2, output, size);
  }

 private:
  const OrtApi& api_;
};

// CreateKernel wrapper expected by the Op object (the Op class uses this)
void* SimpleReLUAddOp::CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
  return new SimpleReLUAddOpKernel(api, info);
}

// register helper
void RegisterSimpleReLUAdd(Ort::CustomOpDomain& domain) {
  static SimpleReLUAddOp op("CUDAExecutionProvider");
  domain.Add(&op);
}
