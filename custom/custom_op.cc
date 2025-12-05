#include "onnxruntime_cxx_api.h"
#include <vector>

// Define a simple custom op: identity operation
struct MyCustomOp : Ort::CustomOpBase<MyCustomOp, MyCustomOp> {
    void* CreateKernel(const OrtApi& /*api*/, const OrtKernelInfo* /*info*/) const {
        return new MyCustomOp();
    }

    const char* GetName() const { return "MyCustomOp"; }
    const char* GetDomain() const { return "mydomain"; }
    size_t GetInputTypeCount() const { return 1; }
    size_t GetOutputTypeCount() const { return 1; }

    ONNXTensorElementDataType GetInputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
    ONNXTensorElementDataType GetOutputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }

    void Compute(OrtKernelContext* context) {
        Ort::CustomOpApi ort{Ort::GetApi()};

        // Get input
        const OrtValue* input = ort.KernelContext_GetInput(context, 0);
        const float* in_data = ort.GetTensorData<float>(input);

        // Get input shape
        OrtTensorTypeAndShapeInfo* shape_info = ort.GetTensorTypeAndShape(input);
        size_t num_dims = ort.GetDimensionsCount(shape_info);
        std::vector<int64_t> shape(num_dims);
        ort.GetDimensions(shape_info, shape.data(), num_dims);
        ort.ReleaseTensorTypeAndShapeInfo(shape_info);

        // Allocate output
        OrtValue* output = ort.KernelContext_GetOutput(context, 0, shape.data(), shape.size());
        float* out_data = ort.GetTensorMutableData<float>(output);

        // Copy input -> output (identity)
        size_t N = 1;
        for (auto d : shape) N *= d;
        for (size_t i = 0; i < N; i++) out_data[i] = in_data[i];
    }
};

// Register custom op
extern "C" OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api_base) {
    const OrtApi* api = api_base->GetApi(ORT_API_VERSION);
    Ort::CustomOpDomain custom_domain("mydomain");
    static MyCustomOp op;
    custom_domain.Add(&op);
    return api->AddCustomOpDomain(options, custom_domain);
}
