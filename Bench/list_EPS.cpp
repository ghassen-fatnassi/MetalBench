#include <iostream>
#include <vector>
#include <onnxruntime_cxx_api.h>

int main() {
    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ListEPs");

        // Query providers
        auto providers = Ort::GetAvailableProviders();

        std::cout << "=== Available Execution Providers ===\n";
        for (const auto& p : providers) {
            std::cout << " - " << p << "\n";
        }

        std::cout << "=====================================\n";

    } catch (const Ort::Exception& e) {
        std::cerr << "ORT Error: " << e.what() << "\n";
    }

    return 0;
}
