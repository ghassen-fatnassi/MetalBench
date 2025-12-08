// main.cc
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include "fused_attn_op.h"
#include <iostream>

int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");

    Ort::SessionOptions session_options;
    Ort::CustomOpDomain custom_domain("custom.attn"); // match domain in Python
    FusedAttnOp fused_attn_op;
    custom_domain.Add(&fused_attn_op);
    session_options.Add(custom_domain);

    Ort::Session session(env, "home/jetson/MetalBench/Models/model_fused.onnx", session_options);

    std::cout << "Session created, fused attention op registered.\n";

    // TODO: Prepare input tensor (batch, N, res, res) and run inference
}
