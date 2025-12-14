#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <fstream>
#include <iostream>

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity s, const char* msg) override {
        if (s <= Severity::kINFO) std::cout << msg << std::endl;
    }
};

int main() {
    Logger logger;

    auto builder = nvinfer1::createInferBuilder(logger);
    auto config = builder->createBuilderConfig();
    auto network = builder->createNetworkV2(1U << 0);
    auto parser = nvonnxparser::createParser(*network, logger);

    if (!parser->parseFromFile("Models/model_fused.onnx",
                               (int)nvinfer1::ILogger::Severity::kINFO)) {
        std::cerr << "ONNX parse failed\n";
        return 1;
    }

    auto engine = builder->buildEngineWithConfig(*network, *config);
    if (!engine) {
        std::cerr << "Engine build failed\n";
        return 1;
    }

    std::cout << "Engine built successfully\n";
    return 0;
}