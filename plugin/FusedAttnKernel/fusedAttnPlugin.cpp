

#include "fusedAttnPlugin.h"
#include <cstring>
#include <iostream>
#include <vector>
#include <cassert>

using namespace nvinfer1;

namespace custom
{

namespace
{
static const char* FUSED_ATTN_PLUGIN_VERSION{"1"};
static const char* FUSED_ATTN_PLUGIN_NAME{"FusedAttnOp"}; // Must match the ONNX "type" or the name registered in ONNX parser
} // namespace

// Static class fields initialization
PluginFieldCollection FusedAttnPluginCreator::mFC{};
std::vector<PluginField> FusedAttnPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(FusedAttnPluginCreator);

// Helper for serialization
template <typename T>
void serialize_value(void** buffer, T value)
{
    T* d = static_cast<T*>(*buffer);
    *d = value;
    *buffer = static_cast<void*>(d + 1);
}

template <typename T>
void deserialize_value(const void** buffer, size_t* buffer_size, T* value)
{
    assert(*buffer_size >= sizeof(T));
    const T* d = static_cast<const T*>(*buffer);
    *value = *d;
    *buffer = static_cast<const void*>(d + 1);
    *buffer_size -= sizeof(T);
}

FusedAttnPlugin::FusedAttnPlugin(const std::string name, float attnScale)
    : mLayerName(name)
    , mAttnScale(attnScale)
{
}

FusedAttnPlugin::FusedAttnPlugin(const std::string name, const void* data, size_t length)
    : mLayerName(name)
{
    // Deserialize
    deserialize_value(&data, &length, &mAttnScale);
}

nvinfer1::IPluginV2DynamicExt* FusedAttnPlugin::clone() const
{
    auto plugin = new FusedAttnPlugin(mLayerName, mAttnScale);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs FusedAttnPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder)
{
    // For identity/attention, output shape often matches input[0] (Batch, Seq, Hidden)
    return inputs[0];
}

bool FusedAttnPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs)
{
    // 7 inputs, 1 output. Total 8 descriptors.
    // Input 0: Data
    // Input 1-6: Weights (if passed as inputs)
    // Output 0: Result
    
    // Check for FP32 and Linear format for all inputs and outputs
    bool condition = (inOut[pos].type == DataType::kFLOAT) && (inOut[pos].format == TensorFormat::kLINEAR);
    return condition;
}

void FusedAttnPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs)
{
    // Verification or setup if needed
    // assert(nbInputs == 7);
    // assert(nbOutputs == 1);
}

size_t FusedAttnPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const
{
    return 0;
}

int FusedAttnPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream)
{
    // Input 0 is the main data input
    // Inputs 1-6 are weights (unused in identity)
    
    // Calculate total elements to process for Input 0
    size_t inputVolume = 1;
    for (int i = 0; i < inputDesc[0].dims.nbDims; ++i)
    {
        inputVolume *= inputDesc[0].dims.d[i];
    }

    const float* inputData = static_cast<const float*>(inputs[0]);
    float* outputData = static_cast<float*>(outputs[0]);

    // Launch identity kernel
    return computeFusedAttn(stream, (int)inputVolume, inputData, outputData);
}

nvinfer1::DataType FusedAttnPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    return DataType::kFLOAT;
}

const char* FusedAttnPlugin::getPluginType() const
{
    return FUSED_ATTN_PLUGIN_NAME;
}

const char* FusedAttnPlugin::getPluginVersion() const
{
    return FUSED_ATTN_PLUGIN_VERSION;
}

int FusedAttnPlugin::getNbOutputs() const
{
    return 1;
}

int FusedAttnPlugin::initialize()
{
    return 0;
}

void FusedAttnPlugin::terminate()
{
}

size_t FusedAttnPlugin::getSerializationSize() const
{
    return sizeof(mAttnScale);
}

void FusedAttnPlugin::serialize(void* buffer) const
{
    serialize_value(&buffer, mAttnScale);
}

void FusedAttnPlugin::destroy()
{
    delete this;
}

void FusedAttnPlugin::setPluginNamespace(const char* libNamespace)
{
    mNamespace = libNamespace;
}

const char* FusedAttnPlugin::getPluginNamespace() const
{
    return mNamespace.c_str();
}

/////////////// Creator ///////////////

FusedAttnPluginCreator::FusedAttnPluginCreator()
{
    // Define the "attn_scale" attribute
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("attn_scale", nullptr, PluginFieldType::kFLOAT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* FusedAttnPluginCreator::getPluginName() const
{
    return FUSED_ATTN_PLUGIN_NAME;
}

const char* FusedAttnPluginCreator::getPluginVersion() const
{
    return FUSED_ATTN_PLUGIN_VERSION;
}

const PluginFieldCollection* FusedAttnPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2* FusedAttnPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    float attnScale = 1.0f; // Default value

    for (int i = 0; i < fc->nbFields; i++)
    {
        std::string field_name(fc->fields[i].name);
        if (field_name.compare("attn_scale") == 0)
        {
            attnScale = *static_cast<const float*>(fc->fields[i].data);
        }
    }

    return new FusedAttnPlugin(name, attnScale);
}

IPluginV2* FusedAttnPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    return new FusedAttnPlugin(name, serialData, serialLength);
}

void FusedAttnPluginCreator::setPluginNamespace(const char* libNamespace)
{
    mNamespace = libNamespace;
}

const char* FusedAttnPluginCreator::getPluginNamespace() const
{
    return mNamespace.c_str();
}

} // namespace custom