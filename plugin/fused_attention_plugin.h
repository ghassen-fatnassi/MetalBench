#pragma once
#include <NvInfer.h>
#include <string>
#include <vector>

class FusedAttentionPlugin : public nvinfer1::IPluginV2DynamicExt {
public:
    FusedAttentionPlugin() {}
    FusedAttentionPlugin(const void* data, size_t length) {}

    // --- Mandatory overrides ---
    const char* getPluginType() const override {
        return "FusedAttnOp";
    }

    

    const char* getPluginVersion() const override {
        return "1";
    }

    int getNbOutputs() const override {
        return 1;
    }

    nvinfer1::IPluginV2DynamicExt* clone() const override {
        return new FusedAttentionPlugin();
    }

    nvinfer1::DimsExprs getOutputDimensions(
        int outputIndex,
        const nvinfer1::DimsExprs* inputs,
        int nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) override
    {
        return inputs[0]; // output shape = input shape (for now)
    }

    bool supportsFormatCombination(
        int pos,
        const nvinfer1::PluginTensorDesc* inOut,
        int nbInputs,
        int nbOutputs) override
    {
        return inOut[pos].format == nvinfer1::TensorFormat::kLINEAR &&
               inOut[pos].type == nvinfer1::DataType::kFLOAT;
    }

    void configurePlugin(
        const nvinfer1::DynamicPluginTensorDesc* inputs,
        int nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* outputs,
        int nbOutputs) override {}

    size_t getWorkspaceSize(
        const nvinfer1::PluginTensorDesc* inputs,
        int nbInputs,
        const nvinfer1::PluginTensorDesc* outputs,
        int nbOutputs) const override
    {
        return 0;
    }

    int enqueue(
        const nvinfer1::PluginTensorDesc* inputDesc,
        const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs,
        void* const* outputs,
        void* workspace,
        cudaStream_t stream) override;

    // --- Boilerplate ---
    int initialize() override { return 0; }
    void terminate() override {}
    size_t getSerializationSize() const override { return 0; }
    void serialize(void* buffer) const override {}
    void destroy() override { delete this; }

    void setPluginNamespace(const char* ns) override {
        mNamespace = ns;
    }

    const char* getPluginNamespace() const override {
        return mNamespace.c_str();
    }

    nvinfer1::DataType getOutputDataType(
        int index,
        const nvinfer1::DataType* inputTypes,
        int nbInputs) const override
    {
        return inputTypes[0];
    }

private:
    std::string mNamespace;
};