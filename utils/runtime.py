import onnxruntime as ort
import numpy as np
import time
import statistics
import argparse

def benchmark_basic(model_path="Models/yolo12n.onnx", warmup_runs=10, runs=100, use_gpu=False):
    providers = ["CUDAExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
    session = ort.InferenceSession(model_path, providers=providers)

    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    input_data = np.random.rand(*[dim if isinstance(dim, int) else 1 for dim in input_shape]).astype(np.float32)

    # Cold start
    t0 = time.time()
    _ = session.run(None, {input_name: input_data})
    t1 = time.time()
    cold_time = (t1 - t0) * 1000

    # Warmup
    warmup_times = []
    for _ in range(warmup_runs):
        t0 = time.time()
        _ = session.run(None, {input_name: input_data})
        t1 = time.time()
        warmup_times.append((t1 - t0) * 1000)

    # Main benchmark
    times = []
    for _ in range(runs):
        t0 = time.time()
        _ = session.run(None, {input_name: input_data})
        t1 = time.time()
        times.append((t1 - t0) * 1000)

    print(f"\n📊 Benchmark results for {model_path}")
    print(f"Provider: {'CUDA' if use_gpu else 'CPU'}")
    print(f"Cold start: {cold_time:.3f} ms")
    print(f"Warmup avg: {statistics.mean(warmup_times):.3f} ms ± {statistics.stdev(warmup_times):.3f}")
    print(f"Steady avg: {statistics.mean(times):.3f} ms ± {statistics.stdev(times):.3f}")
    print(f"Best run: {min(times):.3f} ms | Worst run: {max(times):.3f} ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true", help="Use CUDAExecutionProvider")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()
    benchmark_basic("Models/yolo12n.onnx", args.warmup, args.runs, use_gpu=args.gpu)
