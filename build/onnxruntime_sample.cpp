#include <iostream>
#include "onnxruntime_cxx_api.h"

int main() {
    std::cout << "ONNX Runtime C++ sample - probing environment\n";
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    auto providers = Ort::GetAvailableProviders();
    std::cout << "Available execution providers (compiled):\n";
    for (const auto& p : providers) {
        std::cout << " - " << p << std::endl;
    }
    std::cout << "onnxruntime installed and configured correctly" << std::endl;
    return 0;
}
