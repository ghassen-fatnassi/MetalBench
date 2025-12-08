#include "onnxruntime_cxx_api.h"
#include "FusedAttnOpKernel.h"

struct FusedAttnOp : Ort::CustomOpBase<FusedAttnOp, FusedAttnOpKernel> {
    
    void* CreateKernel(Ort::CustomOpApi api, const OrtKernelInfo* info) const { 
        return new FusedAttnOpKernel(api, info); 
    };

    // Name and Domain must match your Python fusion script
    const char* GetName() const { return "FusedAttnOp"; };
    const char* GetDomain() const { return "custom.attn"; }; 
    const char* GetExecutionProviderType() const { 
        return "CPUExecutionProvider";
    };

    // Configuration for Identity Op: one input, one output
    size_t GetInputTypeCount() const { return 1; }; 
    ONNXTensorElementDataType GetInputType(size_t /*index*/) const { 
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; 
    };
    
    size_t GetOutputTypeCount() const { return 1; }; 
    ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { 
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; 
    };
};

// Function required for shared library registration
OrtStatus* ORT_API_CALL RegisterCustomOps(Ort::CustomOpDomain& domain) {
    static FusedAttnOp custom_op;
    return domain.Add(&custom_op);
}