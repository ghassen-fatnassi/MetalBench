#pragma once
#include <NvInfer.h>
#include <string>

class FusedAttentionPlugin : public nvinfer1::IPluginV2DynamicExt {
public:
    FusedAttentionPlugin() {}
    FusedAttentionPlugin(const void*, size_t) {}

    const char* getPluginType() const override { return "FusedAttnOp"; }
    const char* getPluginVersion() const override { return "1"; }
    int getNbOutputs() const override { return 1; }
    nvinfer1::IPluginV2DynamicExt* clone() const override { return new FusedAttentionPlugin(); }

    nvinfer1::DimsExprs getOutputDimensions(int, const nvinfer1::DimsExprs* inputs, int, nvinfer1::IExprBuilder&) override {
        return inputs[0];
    }

    bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int, int) override {
        return inOut[pos].format == nvinfer1::TensorFormat::kLINEAR &&
               inOut[pos].type == nvinfer1::DataType::kFLOAT;
    }

    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc*, int, const nvinfer1::DynamicPluginTensorDesc*, int) override {}

    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc*, int, const nvinfer1::PluginTensorDesc*, int) const override { return 0; }

    int enqueue(const nvinfer1::PluginTensorDesc*, const nvinfer1::PluginTensorDesc*, const void* const*, void* const*, void*, cudaStream_t) override;

    int initialize() override { return 0; }
    void terminate() override {}
    size_t getSerializationSize() const override { return 0; }
    void serialize(void*) const override {}
    void destroy() override { delete this; }

    void setPluginNamespace(const char* ns) override { mNamespace = ns ? ns : ""; }
    const char* getPluginNamespace() const override { return mNamespace.c_str(); }

    nvinfer1::DataType getOutputDataType(int, const nvinfer1::DataType* inputTypes, int) const override { return inputTypes[0]; }

private:
    std::string mNamespace;
};