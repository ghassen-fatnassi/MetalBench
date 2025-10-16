import onnxruntime as ort
import numpy as np
import time
import matplotlib.pyplot as plt

def benchmark_histogram(model_path="Models/yolo12n.onnx", runs=100, use_gpu=False):
    providers = ["CUDAExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
    session = ort.InferenceSession(model_path, providers=providers)

    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    input_data = np.random.rand(*[dim if isinstance(dim, int) else 1 for dim in input_shape]).astype(np.float32)

    times = []
    for _ in range(runs):
        t0 = time.time()
        _ = session.run(None, {input_name: input_data})
        t1 = time.time()
        times.append((t1 - t0) * 1000)

    plt.hist(times, bins=50)
    plt.xlabel("Inference time (ms)")
    plt.ylabel("Frequency")
    plt.title(f"Inference time distribution ({'GPU' if use_gpu else 'CPU'})")
    plt.show()

if __name__ == "__main__":
    benchmark_histogram("Models/yolo12n.onnx", runs=100, use_gpu=False)
