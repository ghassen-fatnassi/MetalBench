#include "custom_op.h"
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>

int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;

    // Register custom op
    SimpleReLUAddOp my_op;
    Ort::CustomOpDomain custom_op_domain("");
    custom_op_domain.Add(&my_op);
    session_options.Add(custom_op_domain);

    // Load your ONNX model
    const char* model_path = "../Models/yolo12n_op12.onnx";
    Ort::Session session(env, model_path, session_options);

    std::cout << "Custom op session created successfully!" << std::endl;
    return 0;
}
