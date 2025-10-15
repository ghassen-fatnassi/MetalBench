import onnxruntime as ort
import numpy as np
import time

def benchmark_threads(model_path="Models/yolo12n.onnx", runs=50, threads_list=[1,2,8], use_gpu=False):
    for threads in threads_list:
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = threads
        sess_options.inter_op_num_threads = threads
        providers = ["CPUExecutionProvider"] if not use_gpu else ["CUDAExecutionProvider"]
        session = ort.InferenceSession(model_path, sess_options, providers=providers)

        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        input_data = np.random.rand(*[dim if isinstance(dim, int) else 1 for dim in input_shape]).astype(np.float32)

        times = []
        for _ in range(runs):
            t0 = time.time()
            _ = session.run(None, {input_name: input_data})
            t1 = time.time()
            times.append((t1 - t0) * 1000)

        avg_time = sum(times) / len(times)
        print(f"Threads={threads} -> Avg inference: {avg_time:.3f} ms")

if __name__ == "__main__":
    benchmark_threads("Models/yolo12n.onnx", runs=50, threads_list=[1,2,4], use_gpu=False)
