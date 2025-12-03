import onnxruntime as ort
import numpy as np
import os

# ================= CONFIG =================
MODEL_PATH = "Models/yolo12n_op12.onnx"
IMG_SHAPE = (3, 640, 640)
BATCH = 1
DEVICE_ID = 0

# ================= INPUT GENERATION =================
def generate_input(batch_size):
    np.random.seed(42)
    # standard NumPy array
    return np.random.rand(batch_size, *IMG_SHAPE).astype(np.float32)

# ================= SESSION CREATION =================
def create_cuda_session(path):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    so.enable_cpu_mem_arena = True
    so.enable_mem_pattern = True
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ort.InferenceSession(path, sess_options=so, providers=providers)

# ================= RUN WITH IO BINDING =================
def run_inference(session, input_data):
    binding = session.io_binding()

    # Bind input on GPU
    ort_input = ort.OrtValue.ortvalue_from_numpy(input_data, device="cuda", device_id=DEVICE_ID)
    in_name = session.get_inputs()[0].name
    binding.bind_ortvalue_input(in_name, ort_input)

    # Preallocate output on GPU
    out_name = session.get_outputs()[0].name
    out_shape = session.get_outputs()[0].shape
    out_tensor = ort.OrtValue.ortvalue_from_shape_and_type(out_shape, np.float32,
                                                           device="cuda", device_id=DEVICE_ID)
    binding.bind_ortvalue_output(out_name, out_tensor)

    session.run_with_iobinding(binding)
    return out_tensor

# ================= MAIN =================
def main():
    if not os.path.exists(MODEL_PATH):
        print("Error: Model not found.")
        return

    print(f"Loading model: {MODEL_PATH}")
    session = create_cuda_session(MODEL_PATH)

    input_data = generate_input(BATCH)
    print(f"Running inference with input shape: {input_data.shape}")
    output = run_inference(session, input_data)

    print("Done. Output shape:", output.shape())

if __name__ == "__main__":
    main()
