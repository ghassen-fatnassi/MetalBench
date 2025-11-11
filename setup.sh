#!/usr/bin/env bash
set -euo pipefail

# --------------------------------------------------
# METALBENCH SETUP SCRIPT (Improved)
# Usage: ./setup.sh -hw {cpu|openvino|xilinx|jetson}
# --------------------------------------------------

HW=""
DEBIAN_FRONTEND=${DEBIAN_FRONTEND:-noninteractive}

# -------------------------------
# Parse arguments
# -------------------------------
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -hw) HW="${2:-}"; shift ;;
        -h|--help)
            echo "Usage: $0 -hw {cpu|openvino|xilinx|jetson}"
            exit 0 ;;
        *)
            echo "Unknown parameter: $1"
            exit 1 ;;
    esac
    shift
done

if [ -z "$HW" ]; then
    echo "❌ Hardware type not specified."
    echo "Usage: ./setup.sh -hw {cpu|openvino|xilinx|jetson}"
    exit 1
fi

echo "🚀 Selected hardware: $HW"

# -------------------------------
# Helper functions
# -------------------------------
log() { echo -e "\n👉 $1\n"; }

ensure_pkg() {
    if ! dpkg -l | grep -q "$1" >/dev/null 2>&1; then
        sudo apt install -y "$1"
    else
        echo "✅ $1 already installed"
    fi
}

ensure_python() {
    if ! command -v python3 >/dev/null 2>&1; then
        log "⚠️ Python3 not found, installing..."
        sudo apt install -y python3 python3-venv python3-dev
    fi
}

detect_python_version() {
    python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
}

create_env() {
    PY_VER=$(detect_python_version)
    log "🐍 Creating Python venv (Python $PY_VER)"
    python3 -m venv env
    source env/bin/activate
    pip install --upgrade pip setuptools wheel
    pip install --upgrade onnx onnxruntime numpy pandas matplotlib psutil tqdm
}

compile_and_run_sample() {
    local ort_path="$1"
    local sample_cpp="onnxruntime_sample.cpp"
    log "🏗️ Compiling ONNX Runtime C++ sample..."
    cat > "$sample_cpp" <<'CPP'
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
CPP

    g++ -std=c++17 -O2 "$sample_cpp" -I"${ort_path}/include" \
        -L"${ort_path}/lib" -lonnxruntime -lpthread \
        -Wl,-rpath,"${ort_path}/lib" -o onnx_sample

    log "▶️ Running sample..."
    if ! ./onnx_sample; then
        echo "❌ ONNX Runtime test failed"
        exit 1
    fi
    echo "✅ Sample executed successfully"
}

setup_env_vars() {
    local ort_path="$1"
    if ! grep -q "ONNXRUNTIME_PATH" ~/.bashrc 2>/dev/null; then
        echo "export ONNXRUNTIME_PATH=${ort_path}" >> ~/.bashrc
        echo 'export LD_LIBRARY_PATH=$ONNXRUNTIME_PATH/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
        echo 'export PATH=$ONNXRUNTIME_PATH/bin:$PATH' >> ~/.bashrc
    fi
    export ONNXRUNTIME_PATH="$ort_path"
    export LD_LIBRARY_PATH="$ort_path/lib:${LD_LIBRARY_PATH:-}"
}

# -------------------------------
# Common base setup
# -------------------------------
sudo apt update -y
sudo apt install -y git wget curl cmake build-essential unzip pkg-config
sudo apt install -y git-lfs
git lfs install
git lfs pull

mkdir -p build && cd build

# -------------------------------
# CPU Setup
# -------------------------------
if [ "$HW" == "cpu" ]; then
    log "💻 Setting up for Intel/CPU (DigitalOcean or desktop)"
    ensure_python
    sudo apt install -y libprotobuf-dev protobuf-compiler libopencv-dev python3-opencv \
        libnuma-dev libssl-dev libgomp1 libjpeg-dev libpng-dev

    cd ..
    create_env
    cd build

    ORT_VER="1.23.2"
    ORT_PKG="onnxruntime-linux-x64-${ORT_VER}.tgz"
    ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VER}/${ORT_PKG}"

    if [ ! -f "$ORT_PKG" ]; then
        log "📥 Downloading ONNX Runtime v${ORT_VER}..."
        wget -q --show-progress "$ORT_URL" -O "$ORT_PKG"
    else
        echo "ℹ️ ${ORT_PKG} already present"
    fi

    tar -xzf "$ORT_PKG"
    ORT_DIR=$(tar -tf "$ORT_PKG" | head -n1 | cut -f1 -d"/" || true)
    ORT_DIR=${ORT_DIR:-"onnxruntime-linux-x64-${ORT_VER}"}
    ONNXRUNTIME_PATH="$(pwd)/${ORT_DIR}"

    setup_env_vars "$ONNXRUNTIME_PATH"
    compile_and_run_sample "$ONNXRUNTIME_PATH"

    cd ..
    log "✅ CPU environment ready!"
    exit 0
fi

# -------------------------------
# Jetson (TensorRT)
# -------------------------------
if [ "$HW" == "jetson" ]; then
    log "💻 Setting up for Jetson (TensorRT)"
    ensure_python
    sudo apt install -y libprotobuf-dev protobuf-compiler libopencv-dev python3-opencv \
        libnuma-dev libssl-dev libgomp1 libjpeg-dev libpng-dev

    # TensorRT system libs
    ensure_pkg libnvinfer8
    ensure_pkg libnvinfer-plugin8

    cd ..
    create_env
    cd build

    ORT_VER="1.23.2"
    ORT_PKG="onnxruntime-linux-aarch64-${ORT_VER}.tgz"
    ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VER}/${ORT_PKG}"

    if [ ! -f "$ORT_PKG" ]; then
        log "📥 Downloading ONNX Runtime v${ORT_VER}..."
        wget -q --show-progress "$ORT_URL" -O "$ORT_PKG"
    else
        echo "ℹ️ ${ORT_PKG} already present"
    fi

    tar -xzf "$ORT_PKG"
    ORT_DIR=$(tar -tf "$ORT_PKG" | head -n1 | cut -f1 -d"/" || true)
    ORT_DIR=${ORT_DIR:-"onnxruntime-linux-aarch64-${ORT_VER}"}
    ONNXRUNTIME_PATH="$(pwd)/${ORT_DIR}"

    setup_env_vars "$ONNXRUNTIME_PATH"
    compile_and_run_sample "$ONNXRUNTIME_PATH"

    cd ..
    log "✅ Jetson environment ready!"
    exit 0
fi

# -------------------------------
# OpenVINO (Placeholder)
# -------------------------------
if [ "$HW" == "openvino" ]; then
    log "🟡 Setting up for OpenVINO (Intel NCS / CPU + EP)"
    echo "⚠️ TODO: Add OpenVINO runtime or build ORT with OpenVINO EP."
    echo "   - Option 1: pip install openvino-dev"
    echo "   - Option 2: build ONNX Runtime with OpenVINO EP enabled"
    exit 77
fi

# -------------------------------
# Xilinx (Placeholder)
# -------------------------------
if [ "$HW" == "xilinx" ]; then
    log "🟣 Setting up for Xilinx / Vitis AI (FPGA/DPU)"
    echo "⚠️ TODO: Add Vitis AI runtime setup and model compilation flow."
    echo "   - Install Vitis AI SDK or use official Docker image"
    echo "   - Deploy bitstreams and use VART or DPU Runner"
    exit 77
fi

# -------------------------------
# Unsupported HW
# -------------------------------
echo "❌ Unsupported hardware: $HW"
exit 1
