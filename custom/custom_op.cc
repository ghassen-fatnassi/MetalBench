#include "onnxruntime_cxx_api.h"
#include <vector>

struct MyCustomOpKernel : Ort::CustomOpBase<MyCustomOpKernel, OrtKernel> {
    void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
        return new MyCustomOpKernel(api, info);
    }

    const char* GetName() const { return "MyCustomOp"; }
    const char* GetDomain() const { return "mydomain"; }
    size_t GetInputTypeCount() const { return 1; }
    size_t GetOutputTypeCount() const { return 1; }

    ONNXTensorElementDataType GetInputType(size_t /*idx*/) const {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    ONNXTensorElementDataType GetOutputType(size_t /*idx*/) const {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }

    MyCustomOpKernel(const OrtApi& api, const OrtKernelInfo* info) : api_(api) {}

    void Compute(OrtKernelContext* ctx) {
        // Get input tensor
        const OrtValue* input = api_.KernelContext_GetInput(ctx, 0);
        const float* in_data = api_.GetTensorData<float>(input);

        // Shape
        OrtTensorTypeAndShapeInfo* shape_info = api_.GetTensorTypeAndShape(input);
        std::vector<int64_t> shape(api_.GetDimensionsCount(shape_info));
        api_.GetDimensions(shape_info, shape.data(), shape.size());
        api_.ReleaseTensorTypeAndShapeInfo(shape_info);

        size_t n = 1;
        for (auto s : shape) n *= s;

        // Output tensor
        OrtValue* output = api_.KernelContext_GetOutput(ctx, 0, shape.data(), shape.size());
        float* out_data = api_.GetTensorMutableData<float>(output);

        // Kernel logic: identity
        for (size_t i = 0; i < n; i++) out_data[i] = in_data[i];
    }

    const OrtApi& api_;
};

struct MyCustomOpLibrary {
    MyCustomOpKernel op;
};

extern "C" {
    OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api) {
        static MyCustomOpLibrary c;
        Ort::CustomOpDomain custom_domain("mydomain");
        custom_domain.Add(&c.op);

        Ort::UnownedSessionOptions session_options(options);
        session_options.Add(custom_domain);

        return nullptr;
    }
}
