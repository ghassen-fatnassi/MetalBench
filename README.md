
# MetalBench

A tutored project titled **"Hardware Space Exploration For Tiny ML"** dedicated to exploring AI hardware, benchmarking, and compiler frameworks across diverse edge devices.

---

## Project Overview

**MetalBench** is a comprehensive benchmarking suite targeting TinyML and edge AI workloads. It enables users to evaluate, compare, and optimize neural network deployments across a variety of hardware platforms, compilers, and frameworks. The project draws inspiration from [FPGA_DPU](https://github.com/markcxli/FPGA_DPU) and integrates best practices from industry and academia to provide deep insights into hardware/software co-design for AI inference at the edge.

---

## Supported Languages

- **C++** (~68%)
- **Python** (~20%)
- **CUDA** (~10%)
- **CMake**, **Makefile** (~2%)

---

## Supported Devices

- **Avnet Ultra96v2 (Zynq Ultrascale MPSoC)**
    - Vitis AI, hardware acceleration, BLAS, quantization ([Docs](https://www.xilinx.com/support/documents/sw_manuals/xilinx2022_2/ug1137-zynq-ultrascale-mpsoc-swdev.pdf), [BLAS Library](https://xilinx.github.io/Vitis_Libraries/blas/2022.1/index.html))

- **NVIDIA Jetson Nano**
    - CUDA, cuDNN, TensorRT ([TensorRT Guide](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-843/pdf/TensorRT-Developer-Guide.pdf))


---

## AI Compiler & ML Frameworks

- **TVM**, **IREE**, **Vitis AI**, **TensorRT**, **OpenVINO**, **oneDNN**, **cutlass**
    - [Introduction to ML Compilers & Optimizers](https://huyenchip.com/2021/09/07/a-friendly-introduction-to-machine-learning-compilers-and-optimizers.html)
    - [TVM & Related Papers](https://arxiv.org/abs/2207.04296)
    - [AI Compiler Study Notion](https://www.notion.so/AI-Compiler-Study-2cc71f48eb1140d09a439ab0b10bdb7b?pvs=21)

---

## Benchmarking Metrics

- **Cache Pressure**
- **Memory Bandwidth Saturation**
- **Instruction-Level Parallelism (ILP) & Pipeline Utilization**
- **TLB Pressure (Translation Lookaside Buffer)**
- **Warp/Thread Divergence**
- **Contention/Synchronization Overhead**
- **Memory Footprint & Working Set**

> Learn more: [Benchmarking Readme](attachment:482916c1-d994-4a21-b011-f77d44fb8ea3:tinyml_benchmark_readme.md), [mlsysbook.ai](https://www.mlsysbook.ai/contents/core/benchmarking/benchmarking.html)

## Reference Projects & Further Reading

- [FPGA_DPU](https://github.com/markcxli/FPGA_DPU)
- [TinyML Benchmarking](https://www.mlsysbook.ai/contents/core/benchmarking/benchmarking.html)
- [TVM Related Papers](https://www.notion.so/TVM-Related-Papers-2809f1788efd804cbdb7d2c48f8d7ee4?pvs=21)
- [Learnings ZYNQ-MPSOC](https://www.notion.so/Learnings-ZYNQ-MPSOC-29c9f1788efd8028affcdd003fe99a40?pvs=21)
- [STM32 Notes Handy Bits and Pieces](https://www.notion.so/STM32-Notes-Handy-Bits-and-Pieces-2899f1788efd81bc8cacc8f35cb08eb8?pvs=21)
- [PyTorch Executorch](https://pytorch.org/blog/introducing-executorch-1-0/)
- [COS597 Princeton - Scaling, Infra, Compilers](https://www.cs.princeton.edu/~ravian/COS597_F24/)
- [Vitis AI Tutorials](https://github.com/Xilinx/Vitis-AI-Tutorials)

---

## Miscellaneous Resources

- [Analog University Wiki](https://wiki.analog.com/university)
- [FlameGraphs](https://www.brendangregg.com/FlameGraphs/cpuflamegraphs.html)
- [Cutlass Tutorial](https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/)
- [Triton Distributed Overview](https://triton-distributed.readthedocs.io/en/latest/build.html)
