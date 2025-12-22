#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvOnnxParser.h>
#include <iostream>
#include <cuda_runtime_api.h>

using namespace nvinfer1;

// ------------------- Logger -------------------
class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
};

Logger gLogger;

// ------------------- Dummy Plugin -------------------
class SanityPlugin : public IPluginV2IOExt {
public:
    SanityPlugin() {}
    ~SanityPlugin() override {}

    // IPluginV2IOExt overrides
    const char* getPluginType() const noexcept override { return "SanityPlugin"; }
    const char* getPluginVersion() const noexcept override { return "1"; }
    int getNbOutputs() const noexcept override { return 1; }
    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept override { return inputs[0]; }
    int initialize() noexcept override { return 0; }
    void terminate() noexcept override {}
    size_t getWorkspaceSize(int) const noexcept override { return 0; }
    int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) noexcept override { return 0; }
    size_t getSerializationSize() const noexcept override { return 0; }
    void serialize(void* buffer) const noexcept override {}
    void destroy() noexcept override { delete this; }
    IPluginV2IOExt* clone() const noexcept override { return new SanityPlugin(); }
    void setPluginNamespace(const char* pluginNamespace) noexcept override {}
    const char* getPluginNamespace() const noexcept override { return ""; }

    bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const noexcept override {
        return true;
    }

    DataType getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const noexcept override {
        return inputTypes[0];
    }

    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const noexcept override { return false; }
    bool canBroadcastInputAcrossBatch(int inputIndex) const noexcept override { return false; }
};

// ------------------- Main -------------------
int main() {
    std::cout << "Creating TensorRT builder..." << std::endl;
    auto builder = createInferBuilder(gLogger);
    if (!builder) {
        std::cerr << "Failed to create builder!" << std::endl;
        return -1;
    }

    std::cout << "Builder created successfully. TensorRT is properly installed!" << std::endl;
    builder->destroy();
    return 0;
}
