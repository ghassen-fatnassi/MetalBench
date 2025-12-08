#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include "fused_attn_op.h" // Includes the C-style registration function
#include <iostream>

// Forward declaration for the op creator function
extern OrtCustomOp* CreateFusedAttnOp();

int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");

    Ort::SessionOptions session_options;
    
    // 1. Create an OrtCustomOpDomain with the domain name used in the ONNX model
    Ort::CustomOpDomain custom_domain("custom.attn"); 
    
    // 2. Create the OrtCustomOp struct using the C-style function
    OrtCustomOp* fused_attn_op_c_struct = CreateFusedAttnOp();
    
    // 3. Add the custom op C struct to the custom domain
    Ort::ThrowOnError(OrtAddCustomOp(custom_domain, fused_attn_op_c_struct));
    
    // 4. Add the custom domain to the session options
    session_options.Add(custom_domain);

    // Note: If you don't delete the C struct, you'll have a memory leak, 
    // but the session needs it, so handle cleanup carefully in production code.
    
    Ort::Session session(env, "/home/jetson/MetalBench/Models/model_fused.onnx", session_options);

    std::cout << "Session created, fused attention op registered.\n";

    // TODO: Prepare input tensor (batch, N, res, res) and run inference
    
    return 0;
}