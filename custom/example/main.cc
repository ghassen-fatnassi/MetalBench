// main.cc
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include "fused_attn_op.h"
#include <iostream>
#include "core/framework/customregistry.h" // Needed for CustomRegistry

// Note: Using the onnxruntime namespace as the registration function lives there
using namespace onnxruntime;

int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");

    Ort::SessionOptions session_options;

    // --- New Registration Method ---
    // 1. Create the CustomRegistry object.
    std::shared_ptr<CustomRegistry> custom_registry = std::make_shared<CustomRegistry>();

    // 2. Register the custom ops (schema and kernel) with the registry.
    // The custom op logic now resides inside the custom_op namespace.
    if (!custom_op::RegisterFusedAttnCustomOps(*custom_registry).IsOK()) {
        std::cerr << "Failed to register FusedAttn custom ops." << std::endl;
        return 1;
    }

    // 3. Register the CustomRegistry with the SessionOptions.
    // Use the C API function wrapper for RegisterCustomRegistry
    if (OrtApis::Get->SessionOptionsRegisterCustomRegistry(session_options, custom_registry.get()) != NULL) {
        std::cerr << "Failed to register custom registry with session options." << std::endl;
        return 1;
    }
    
    // Original path to the model file
    Ort::Session session(env, "/home/jetson/MetalBench/Models/model_fused.onnx", session_options);

    std::cout << "Session created, fused attention op registered via CustomRegistry.\n";

    // TODO: Prepare input tensor (batch, N, res, res) and run inference
    
    return 0;
}