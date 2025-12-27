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
static const char* FUSED_ATTN_PLUGIN_NAME{"FusedAttnOp"}; 
} // namespace

// Static class fields initialization
PluginFieldCollection FusedAttnPluginCreator::mFC{};
std::vector<PluginField> FusedAttnPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(FusedAttnPluginCreator);

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
    // Output shape matches input shape (N, C, H, W)
    return inputs[0];
}

bool FusedAttnPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs)
{
    // 7 inputs (Data + 6 weights/bias), 1 output.
    // Ensure everything is Float32 and Linear
    return (inOut[pos].type == DataType::kFLOAT) && (inOut[pos].format == TensorFormat::kLINEAR);
}

void FusedAttnPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs)
{
}

size_t FusedAttnPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const
{
    // We need workspace for intermediate tensors:
    // Input is N, C, H, W
    // 1. Q, K, V (each N, C, H, W) after initial projection
    // 2. V_dw (N, C, H, W) after depthwise
    // 3. AttnMatrix (N, HW, HW) - This is the largest if H,W are big. 
    //    For 40x40=1600, 1600x1600 floats = ~10MB.
    // 4. AttnOutput (N, C, H, W)
    
    // Note: inputs[0].dims is (N, C, H, W)
    int n = inputs[0].dims.d[0];
    int c = inputs[0].dims.d[1];
    int h = inputs[0].dims.d[2];
    int w = inputs[0].dims.d[3];
    
    // Safety check for dynamic dims (-1)
    if (n < 0 || c < 0 || h < 0 || w < 0) return 0; // Runtime will call again with real dims

    size_t sizeImage = n * c * h * w * sizeof(float);
    size_t sizeAttn = n * (h * w) * (h * w) * sizeof(float);

    size_t total = 0;
    total += sizeImage; // Q
    total += sizeImage; // K
    total += sizeImage; // V
    total += sizeImage; // V_dw
    total += sizeAttn;  // Attention Map (HW x HW)
    total += sizeImage; // Attention Output
    
    // Align to 256 bytes for safety
    return total + 256 * 6;
}

int FusedAttnPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream)
{
    int n = inputDesc[0].dims.d[0];
    int c = inputDesc[0].dims.d[1];
    int h = inputDesc[0].dims.d[2];
    int w = inputDesc[0].dims.d[3];

    // Inputs layout:
    // 0: Data
    // 1: W1 (192, 64, 1, 1) or similar
    // 2: B1 (192)
    // 3: W2 (64, 1, 7, 7)
    // 4: B2 (64)
    // 5: W3 (64, 64, 1, 1)
    // 6: B3 (64)
    
    return computeFusedAttn(stream, 
        n, h, w, c, // Batch, SeqLen (HW), Hidden
        static_cast<const float*>(inputs[0]),
        static_cast<const float*>(inputs[1]),
        static_cast<const float*>(inputs[2]),
        static_cast<const float*>(inputs[3]),
        static_cast<const float*>(inputs[4]),
        static_cast<const float*>(inputs[5]),
        static_cast<const float*>(inputs[6]),
        static_cast<float*>(outputs[0]),
        workspace,
        mAttnScale);
}

nvinfer1::DataType FusedAttnPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    return DataType::kFLOAT;
}

const char* FusedAttnPlugin::getPluginType() const { return FUSED_ATTN_PLUGIN_NAME; }
const char* FusedAttnPlugin::getPluginVersion() const { return FUSED_ATTN_PLUGIN_VERSION; }
int FusedAttnPlugin::getNbOutputs() const { return 1; }
int FusedAttnPlugin::initialize() { return 0; }
void FusedAttnPlugin::terminate() {}
size_t FusedAttnPlugin::getSerializationSize() const { return sizeof(mAttnScale); }
void FusedAttnPlugin::serialize(void* buffer) const { serialize_value(&buffer, mAttnScale); }
void FusedAttnPlugin::destroy() { delete this; }
void FusedAttnPlugin::setPluginNamespace(const char* libNamespace) { mNamespace = libNamespace; }
const char* FusedAttnPlugin::getPluginNamespace() const { return mNamespace.c_str(); }

// Creator Implementation
FusedAttnPluginCreator::FusedAttnPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("attn_scale", nullptr, PluginFieldType::kFLOAT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* FusedAttnPluginCreator::getPluginName() const { return FUSED_ATTN_PLUGIN_NAME; }
const char* FusedAttnPluginCreator::getPluginVersion() const { return FUSED_ATTN_PLUGIN_VERSION; }
const PluginFieldCollection* FusedAttnPluginCreator::getFieldNames() { return &mFC; }

IPluginV2* FusedAttnPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    float attnScale = 1.0f;
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

void FusedAttnPluginCreator::setPluginNamespace(const char* libNamespace) { mNamespace = libNamespace; }
const char* FusedAttnPluginCreator::getPluginNamespace() const { return mNamespace.c_str(); }

} // namespace custom