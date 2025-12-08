// FusedAttnOp.cc

#include "onnxruntime/core/session/onnxruntime_cxx_api.h"
// This includes the definition of FusedAttnOpKernel and the utility OrtTensorDimensions
#include "FusedAttnOpKernel.h" 

// --- 1. Define the FusedAttnOp structure ---

struct FusedAttnOp : Ort::CustomOpBase<FusedAttnOp, FusedAttnOpKernel> {
    
    // This method is called by ONNX Runtime to instantiate the kernel 
    void* CreateKernel(Ort::CustomOpApi api, const OrtKernelInfo* info) const { 
        return new FusedAttnOpKernel(api, info); 
    };

    // The name of the operator. MUST match the op_type in your Python script.
    const char* GetName() const { return "FusedAttnOp"; };
    
    // The domain of the operator. MUST match the domain in your Python script.
    const char* GetDomain() const { return "custom.attn"; }; 
    
    // Define the Execution Provider (EP).
    const char* GetExecutionProviderType() const { 
        // We defined the kernel only for CPU, so we specify CPU here.
        return "CPUExecutionProvider";
    };

    // --- Input/Output Type Definitions ---
    // Single Input (The tensor going into the first node of the fused block)
    size_t GetInputTypeCount() const { return 1; }; 
    ONNXTensorElementDataType GetInputType(size_t /*index*/) const { 
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; 
    };
    
    // Single Output (The tensor coming out of the last node of the fused block)
    size_t GetOutputTypeCount() const { return 1; }; 
    ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { 
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; 
    };
    
    // NOTE: If you had attributes (like an 'epsilon' value), you would override 
    // Get)AttributeType and Get)AttributeCount here.
};

// --- 2. The Mandatory Registration Function (FIXED) ---

// FIX: The extern "C" block prevents C++ name mangling, 
// ensuring the function is exported under the exact name 
// 'RegisterCustomOps' that ONNX Runtime looks for.
extern "C" {
    OrtStatus* ORT_API_CALL RegisterCustomOps(Ort::CustomOpDomain& domain) {
        static FusedAttnOp custom_op;
        
        // 1. Add the op (returns void, as previously fixed)
        domain.Add(&custom_op);
        
        // 2. Return nullptr to signal success
        return nullptr;
    }
} // End extern "C" block