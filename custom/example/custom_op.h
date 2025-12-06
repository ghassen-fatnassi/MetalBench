// custom_op.h

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

// A Simple Custom Op with one input and one output
struct SimpleReLUAddOp : Ort::CustomOpBase<SimpleReLUAddOp> {
    void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const;
    const char* GetKernelTypeInfoName() const;

    // Defines the ONNX properties
    size_t GetInputTypeCount() const { return 2; }
    ONNXTensorElementDataType GetInputType(size_t index) const {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; // float for both X1 and X2
    }

    size_t GetOutputTypeCount() const { return 1; }
    ONNXTensorElementDataType GetOutputType(size_t index) const {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; // float for Y
    }
};

// Function to register the custom ops
void RegisterSimpleReLUAdd(Ort::CustomOpDomain& domain);