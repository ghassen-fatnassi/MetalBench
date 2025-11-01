import onnxruntime as ort
import numpy as np
import time

def benchmark_bandwidth(model_path="Models/yolo12n.onnx", runs=100, use_gpu=False):
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

    avg_time_s = sum(times) / len(times) / 1000
    num_elements = np.prod([dim if isinstance(dim, int) else 1 for dim in input_shape])
    total_bytes = num_elements * 4  # float32
    bandwidth_gb_s = (total_bytes / avg_time_s) / (1024 ** 3)
    print(f"Approximate bandwidth: {bandwidth_gb_s:.3f} GB/s")

if __name__ == "__main__":
    benchmark_bandwidth("Models/yolo12n.onnx", runs=100, use_gpu=False)
