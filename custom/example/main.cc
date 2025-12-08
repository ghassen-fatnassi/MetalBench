// main.cc
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include "fused_attn_op.h"
#include <iostream>

int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");

    Ort::SessionOptions session_options;
    
    // 1. Create an OrtCustomOpDomain with the domain name used in the ONNX model
    Ort::CustomOpDomain custom_domain("custom.attn"); 
    
    // 2. Create the custom op instance
    FusedAttnOp fused_attn_op;
    
    // 3. Add the op instance to the custom domain
    custom_domain.Add(&fused_attn_op);
    
    // 4. Add the custom domain to the session options
    session_options.Add(custom_domain);

    // Note: Ensure the model path is correct (e.g., /home/jetson/...)
    Ort::Session session(env, "/home/jetson/MetalBench/Models/model_fused.onnx", session_options);

    std::cout << "Session created, fused attention op registered.\n";

    // TODO: Prepare input tensor (batch, N, res, res) and run inference
    
    return 0;
}