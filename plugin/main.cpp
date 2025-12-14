#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvOnnxParser.h>

#include <dlfcn.h>
#include <iostream>
#include <cstdlib>

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity s, const char* msg) override {
        if (s <= Severity::kINFO)
            std::cout << "[TRT] " << msg << std::endl;
    }
};

static void loadPluginLibrary()
{
    std::cout << "=== Loading plugin library ===" << std::endl;

    void* handle = dlopen("./plugin/build/libfused_attention_plugin.so", RTLD_NOW);
    if (!handle) {
        std::cerr << "❌ dlopen failed: " << dlerror() << std::endl;
        std::exit(1);
    }

    std::cout << "✅ Plugin library loaded successfully" << std::endl;
}

static void dumpPluginRegistry()
{
    std::cout << "=== Dumping TensorRT plugin registry ===" << std::endl;

    auto* registry = nvinfer1::getPluginRegistry();
    int n = registry->getNbPluginCreators();

    std::cout << "Registered plugin creators: " << n << std::endl;

    bool found = false;
    for (int i = 0; i < n; ++i) {
        auto* c = registry->getPluginCreator(i);
        std::cout
            << "  [" << i << "] "
            << c->getPluginName()
            << " | v" << c->getPluginVersion()
            << " | ns='" << c->getPluginNamespace() << "'"
            << std::endl;

        if (std::string(c->getPluginName()) == "FusedAttnOp")
            found = true;
    }

    if (!found) {
        std::cerr << "❌ FusedAttnOp NOT found in registry" << std::endl;
    } else {
        std::cout << "✅ FusedAttnOp found in registry" << std::endl;
    }
}

int main()
{
    Logger logger;

    std::cout << "\n=== Initializing TensorRT plugins ===" << std::endl;
    initLibNvInferPlugins(&logger, "");

    loadPluginLibrary();
    dumpPluginRegistry();

    std::cout << "\n=== Building network ===" << std::endl;

    auto builder = nvinfer1::createInferBuilder(logger);
    auto config  = builder->createBuilderConfig();
    auto network = builder->createNetworkV2(1U << 0);
    auto parser  = nvonnxparser::createParser(*network, logger);

    std::cout << "=== Parsing ONNX ===" << std::endl;
    if (!parser->parseFromFile(
            "../Models/model_fused.onnx",
            static_cast<int>(nvinfer1::ILogger::Severity::kINFO))) {

        std::cerr << "❌ ONNX parse failed" << std::endl;
        return 1;
    }

    std::cout << "✅ ONNX parsed successfully" << std::endl;

    std::cout << "=== Building engine ===" << std::endl;
    auto engine = builder->buildEngineWithConfig(*network, *config);
    if (!engine) {
        std::cerr << "❌ Engine build failed" << std::endl;
        return 1;
    }

    std::cout << "✅ Engine built successfully" << std::endl;
    return 0;
}
