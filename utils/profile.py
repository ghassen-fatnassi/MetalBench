import onnxruntime as ort
import numpy as np

def benchmark_profiling(model_path="Models/yolo12n.onnx", use_gpu=False):
    providers = ["CUDAExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
    sess_options = ort.SessionOptions()
    sess_options.enable_profiling = True
    session = ort.InferenceSession(model_path, sess_options, providers=providers)

    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    input_data = np.random.rand(*[dim if isinstance(dim, int) else 1 for dim in input_shape]).astype(np.float32)

    _ = session.run(None, {input_name: input_data})
    profile_file = session.end_profiling()
    print(f"Profiling JSON saved to {profile_file}. Open in Chrome tracing or convert to flamegraph.")

if __name__ == "__main__":
    benchmark_profiling("Models/yolo12n.onnx", use_gpu=False)
