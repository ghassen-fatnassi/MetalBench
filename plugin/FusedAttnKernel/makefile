# CUDA directory: /usr/local/cuda is standard on Jetson
CUDA_PATH = /usr/local/cuda

# TensorRT paths for Jetson (System Install)
# Headers are typically in /usr/include/aarch64-linux-gnu
# Libraries are typically in /usr/lib/aarch64-linux-gnu
TRT_INC_DIR = /usr/include/aarch64-linux-gnu
TRT_LIB_DIR = /usr/lib/aarch64-linux-gnu

# Compiler settings
CC = g++
NVCC = $(CUDA_PATH)/bin/nvcc

# Compute Capability (SM)
# Adjust based on your Jetson model:
# Nano: 53, TX2: 62, Xavier: 72, Orin: 87
SMS = 53
GENCODE_FLAGS = -gencode arch=compute_$(SMS),code=sm_$(SMS)

# Flags
# -std=c++14 is often required for newer TRT versions
CXXFLAGS = -O3 -std=c++14 -fPIC -Wall
CUFLAGS  = -O3 -std=c++14 -Xcompiler -fPIC $(GENCODE_FLAGS)

# Includes and Libraries
INCLUDES = -I$(CUDA_PATH)/include -I$(TRT_INC_DIR)
LDFLAGS  = -L$(CUDA_PATH)/lib64 -L$(TRT_LIB_DIR) -lnvinfer -lcudart

# Target file
TARGET_LIB = libfusedAttnPlugin.so

# Source files
SRCS_CPP = fusedAttnPlugin.cpp
SRCS_CU  = fusedAttnKernel.cu

# Object files
OBJS = $(SRCS_CPP:.cpp=.o) $(SRCS_CU:.cu=.o)

all: $(TARGET_LIB)

$(TARGET_LIB): $(OBJS)
	$(CC) -shared -o $@ $(OBJS) $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

%.o: %.cu
	$(NVCC) $(CUFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET_LIB)

.PHONY: all clean