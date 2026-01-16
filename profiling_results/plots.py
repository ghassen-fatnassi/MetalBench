import matplotlib.pyplot as plt
import numpy as np

# Models and execution providers
models = ['UNet', 'MobileNetV2']
eps = ['CUDA', 'TensorRT']

# Data: throughput (FPS), jitter (ms)
data = {
    'UNet': {
        'CUDA': {'throughput': 4.73, 'variance': 1.07},
        'TensorRT': {'throughput': 6.01, 'variance': 1.09}
    },
    'MobileNetV2': {
        'CUDA': {'throughput': 38.24, 'variance': 1.18},
        'TensorRT': {'throughput': 67.56, 'variance': 1.18}
    }
}

# 1. Throughput Plot
fig, ax = plt.subplots(figsize=(8,6))
x = np.arange(len(models))
width = 0.35

throughput_cuda = [data[m]['CUDA']['throughput'] for m in models]
throughput_trt  = [data[m]['TensorRT']['throughput'] for m in models]

ax.bar(x - width/2, throughput_cuda, width, label='Throughput CUDA (FPS)', color='skyblue')
ax.bar(x + width/2, throughput_trt, width, label='Throughput TRT (FPS)', color='dodgerblue')

ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel('Throughput (FPS)')
ax.set_title('Throughput Comparison')
ax.legend()
plt.tight_layout()
plt.savefig('throughput_comparison.png')
plt.close()


# 2. Jitter Plot
fig, ax = plt.subplots(figsize=(6,5))
jitter_cuda = [data[m]['CUDA']['variance'] for m in models]
jitter_trt  = [data[m]['TensorRT']['variance'] for m in models]

ax.bar(x - width/2, jitter_cuda, width, label='variance CUDA (ratio))', color='lightgreen')
ax.bar(x + width/2, jitter_trt, width, label='variance TRT (ratio)', color='green')

ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel('variance (ratio)')
ax.set_title('variance Comparison')
ax.legend()
plt.tight_layout()
plt.savefig('variance_comparison.png')
plt.close()
