#grab all the models using git lfs
#make correct installation of prerequisites of onnxruntime based on the hardware i'm running on then grab the correct onnxruntime
#install and build onnxruntime
#run the corresponding onnxruntime cpp file

#all this will be acheieved by doing ./setup.sh and giving CLI arguments of the hardware type which will basically be either vitis , openvino or cuda


#!/bin/bash
set -e

# --------------------------------------------
# METALBENCH SETUP SCRIPT
# Usage: ./setup.sh -hw {digitalocean|jetson|avnet}
# --------------------------------------------

HW=""
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -hw) HW="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$HW" ]; then
    echo "❌ Hardware type not specified. Usage: ./setup.sh -hw {intel-xeon|jetson|avnet}"
    exit 1
fi

echo "🚀 Setting up METALBENCH for hardware: $HW"
sudo apt update -y
sudo apt install -y git wget curl python3 python3-pip cmake build-essential unzip

# --------------------------------------------
# Common setup
# --------------------------------------------
echo "📦 Installing common dependencies..."
sudo apt install -y libprotobuf-dev protobuf-compiler libopencv-dev python3-opencv
pip3 install --upgrade pip
pip3 install onnx onnxruntime numpy pandas matplotlib psutil tqdm 

# Git LFS for model weights
echo "🔧 Installing Git LFS..."
sudo apt install -y git-lfs
git lfs install
git lfs pull || true

# Create build dir
mkdir -p build && cd build

# --------------------------------------------
# DIGITALOCEAN CPU SETUP
# --------------------------------------------
if [ "$HW" == "intel-xeon" ]; then
    echo "💻 Setting up for Intel Xeon (DigitalOcean VM) ..."
    
    # Install ONNX Runtime C++ API
    echo "📥 Downloading ONNX Runtime C++ prebuilt package..."
    wget -q https://github.com/microsoft/onnxruntime/releases/download/v1.20.0/onnxruntime-linux-x64-1.20.0.tgz
    tar -xzf onnxruntime-linux-x64-1.20.0.tgz
    echo "export ONNXRUNTIME_PATH=$(pwd)/onnxruntime-linux-x64-1.20.0" >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=$ONNXRUNTIME_PATH/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
    source ~/.bashrc

    echo "✅ ONNX Runtime installed successfully."
    echo "🏗️ Ready to build and benchmark models."

    cd ..
    echo "✅ DigitalOcean environment ready!"
    exit 0
fi

# --------------------------------------------
# NVIDIA Jetson Nano SETUP (Placeholder)
# --------------------------------------------
if [ "$HW" == "jetson" ]; then
    echo "🟢 Setting up for NVIDIA Jetson Nano..."
    echo "TODO: Add TensorRT + JetPack dependencies and ONNX Runtime GPU build steps."
    echo "⚠️ Placeholder – to be implemented."
    exit 0
fi

# --------------------------------------------
# Avnet Ultra96-V2 SETUP (Placeholder)
# --------------------------------------------
if [ "$HW" == "avnet" ]; then
    echo "🟣 Setting up for Avnet Ultra96-V2..."
    echo "TODO: Add Vitis AI runtime installation, DPU firmware deployment, and model compilation flow."
    echo "⚠️ Placeholder – to be implemented."
    exit 0
fi

