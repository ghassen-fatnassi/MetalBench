#include "custom_op.h"
#include <cuda_runtime.h>
#include <vector>

extern "C" void SimpleReLUAddKernelLaunch(cudaStream_t stream,
                                          const float* in1,
                                          const float* in2,
                                          float* out,
                                          size_t n);

struct SimpleReLUAddOpKernel {

    SimpleReLUAddOpKernel(const OrtApi& api, const OrtKernelInfo* info)
        : api_(api), info_(info) {}

    void Compute(OrtKernelContext* context) {

        // C API usage for ORT 1.6
        const OrtValue* in1_val = api_.KernelContext_GetInput(context, 0);
        const OrtValue* in2_val = api_.KernelContext_GetInput(context, 1);

        const float* in1 = api_.GetTensorData<float>(in1_val);
        const float* in2 = api_.GetTensorData<float>(in2_val);

        OrtTensorTypeAndShapeInfo* shape_info = api_.GetTensorTypeAndShape(in1_val);

        size_t dim_count = api_.GetDimensionsCount(shape_info);
        std::vector<int64_t> shape(dim_count);
        api_.GetDimensions(shape_info, shape.data(), dim_count);

        size_t size = 1;
        for (auto d : shape) size *= d;

        api_.ReleaseTensorTypeAndShapeInfo(shape_info);

        OrtValue* out_val =
            api_.KernelContext_GetOutput(context, 0, shape.data(), dim_count);

        float* out = api_.GetTensorMutableData<float>(out_val);

        cudaStream_t stream = 0; // default CUDA stream

        SimpleReLUAddKernelLaunch(stream, in1, in2, out, size);
    }

    const OrtApi& api_;
    const OrtKernelInfo* info_;
};

void* SimpleReLUAddOp::CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
    return new SimpleReLUAddOpKernel(api, info);
}

void RegisterSimpleReLUAdd(Ort::CustomOpDomain& domain) {
    static SimpleReLUAddOp op;
    domain.Add(&op);
}
