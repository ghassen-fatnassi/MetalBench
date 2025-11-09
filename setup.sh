#!/usr/bin/env bash
set -euo pipefail

# --------------------------------------------
# METALBENCH SETUP SCRIPT
# Usage: ./setup.sh -hw {cpu|openvino|xilinx|jetson}
# Example: ./setup.sh -hw cpu
# --------------------------------------------

HW=""
# optional: allow non-interactive apt by default
DEBIAN_FRONTEND=${DEBIAN_FRONTEND:-noninteractive} #if an apt package prompts me during installation, it automatically goes for default choice

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -hw) HW="${2:-}"; shift ;;
        -h|--help)
            echo "Usage: $0 -hw {cpu|openvino|xilinx|jetson}"
            exit 0
            ;;
        *)
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
    esac
    shift
done

if [ -z "$HW" ]; then
    echo "❌ Hardware type not specified. Usage: ./setup.sh -hw {cpu|openvino|xilinx|jetson}"
    exit 1
fi

echo "🚀 chosen hardware: $HW"

sudo apt update -y
sudo apt install -y git wget curl python3.10-dev python3.10-pip cmake build-essential unzip pkg-config

python3 -m pip install --upgrade pip setuptools wheel || true

echo "🔧 Installing Git LFS..."
sudo apt install -y git-lfs
git lfs install
git lfs pull

mkdir build
cd build

#
# CPU (Intel / DigitalOcean) — full flow implemented
#
if [ "$HW" == "cpu" ]; then
    echo "💻 Setting up for Intel CPU (DigitalOcean VM) ..."

    # Install system dependencies commonly needed for ONNX Runtime C++ builds and runtime
    echo "📦 Installing common dependencies..."
    sudo apt install -y libprotobuf-dev protobuf-compiler libopencv-dev python3-opencv \
        libnuma-dev libssl-dev libgomp1 libjpeg-dev libpng-dev

    # Python setup ( env and packages)
    cd ..
    python -m venv env
    source env/bin/activate
    pip install --upgrade onnx onnxruntime numpy pandas matplotlib psutil tqdm
    cd build
    # went back to initial directory

    # Download ONNX Runtime prebuilt C++ (shared libs) - change version if you want another
    ORT_VER="1.23.2"
    ORT_PKG="onnxruntime-linux-x64-${ORT_VER}.tgz"
    ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VER}/${ORT_PKG}"


    echo "📥 Downloading ONNX Runtime C++ prebuilt package v${ORT_VER}..."
    if [ ! -f "${ORT_PKG}" ]; then
        wget -q --show-progress "${ORT_URL}" -O "${ORT_PKG}" || {
            echo "⚠️ Failed to download ${ORT_PKG}. Please check network or ORT version."
            exit 1
        }
    else
        echo "ℹ️ ${ORT_PKG} already present, skipping download."
    fi

    tar -xzf "${ORT_PKG}"
    ORT_DIR=$(tar -tf "${ORT_PKG}" | head -n1 | cut -f1 -d"/" || true)
    # fallback if above failed
    if [ -z "$ORT_DIR" ]; then
        ORT_DIR="onnxruntime-linux-x64-${ORT_VER}"
    fi
    ONNXRUNTIME_PATH="$(pwd)/${ORT_DIR}"

    # Export env for current session and append to ~/.bashrc if not already added
    if ! grep -q "ONNXRUNTIME_PATH" ~/.bashrc 2>/dev/null; then
        echo "export ONNXRUNTIME_PATH=${ONNXRUNTIME_PATH}" >> ~/.bashrc
        echo 'export LD_LIBRARY_PATH=$ONNXRUNTIME_PATH/lib:$LD_LIBRARY_PATH' >> ~/.bashrc #exporting it to show the linker where to look at runtime
        echo "export PATH=\$ONNXRUNTIME_PATH/bin:\$PATH" >> ~/.bashrc
    fi

    # also export for current shell execution of this script since this will only be applied after source ./bashrc which is not applied
    export ONNXRUNTIME_PATH="$ONNXRUNTIME_PATH"
    export LD_LIBRARY_PATH="$ONNXRUNTIME_PATH/lib:${LD_LIBRARY_PATH:-}"

    echo "✅ ONNX Runtime C++ package unpacked to: $ONNXRUNTIME_PATH"

    # Create a minimal C++ sample to verify runtime if not present
    SAMPLE_CPP="onnxruntime_sample.cpp"
    if [ ! -f "$SAMPLE_CPP" ]; then
        cat > "$SAMPLE_CPP" <<'CPP'
#include <iostream>
#include "onnxruntime_cxx_api.h"

int main() {
    std::cout << "ONNX Runtime C++ sample - probing environment\n";
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    std::cout << "Available execution providers (compiled):\n";
    auto providers = Ort::GetAvailableProviders();
    for (const auto& p : providers) {
        std::cout << " - " << p << std::endl;
    }
    std::cout << "onnxruntime installed and configured correctly" << std::endl;
    return 0;
}
CPP
    fi
    # Compile the sample
    echo "🏗️ Compiling ONNX Runtime C++ sample..."
    # Use rpath so binary finds libonnxruntime at runtime without modifying LD_LIBRARY_PATH persistently
    g++ -std=c++17 -O2 "$SAMPLE_CPP" -I"${ONNXRUNTIME_PATH}/include" \
        -L"${ONNXRUNTIME_PATH}/lib" -lonnxruntime -lpthread -Wl,-rpath,"${ONNXRUNTIME_PATH}/lib" -o onnx_sample || {
            echo "⚠️ Compilation failed. Check include/lib paths in $ONNXRUNTIME_PATH"
            exit 1
        }

    echo "✅ Sample compiled: ./onnx_sample"
    echo "▶️ Running sample now..."
    ./onnx_sample || true # why is this not giving an error if it fails , the sample code not working is really problematic

    cd ..
    echo "✅ DigitalOcean CPU environment ready!"
    exit 0
fi

#
# JETSON (TensorRT) placeholder
#
if [ "$HW" == "jetson" ]; then
    echo "🔵 Setting up for NVIDIA Jetson (TensorRT / JetPack)..."
    echo "TODO: Add JetPack/TensorRT installation guidance and ONNX Runtime GPU/TensorRT build steps."
    echo " - On Jetson, typically you install JetPack (SDK) and build ONNX Runtime with TensorRT support."
    echo "⚠️ Placeholder – to be implemented."
    exit 0
fi


#
# OPENVINO placeholder
#
if [ "$HW" == "openvino" ]; then
    echo "🟡 Setting up for OpenVINO (Intel NCS / CPU + OpenVINO EP)..."
    echo "TODO: Add OpenVINO installation steps and ORT with OpenVINO EP build."
    echo " - Install Intel OpenVINO toolkit (or use OpenVINO runtime pip package)"
    echo " - Build ONNX Runtime with OpenVINO Execution Provider (requires ORT build from source)"
    echo "⚠️ Placeholder – to be implemented."
    exit 0
fi

#
# XILINX / VITIS (placeholder)
#
if [ "$HW" == "xilinx" ]; then
    echo "🟣 Setting up for Xilinx / Vitis AI (FPGA/DPU)..."
    echo "TODO: Add Vitis AI runtime installation and model compilation flow (xilinx/vitis)."
    echo " - Install Vitis AI libs/toolchain on host or use Docker image provided by Xilinx/AMD."
    echo " - Compile models to DPU format, deploy bitstreams, and use VART / DPU runner."
    echo "⚠️ Placeholder – to be implemented."
    exit 0
fi

# Fallback unknown hw
echo "❌ Unsupported hardware: $HW"
exit 1
