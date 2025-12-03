import onnxruntime as ort
import numpy as np
import os

# ================= CONFIG =================
MODEL_PATH = "Models/yolo12n_op12.onnx"
IMG_SHAPE = (3, 128, 128)
BATCH_SIZE = 1

# ================= INPUT GENERATION =================
def generate_input(batch_size):
    np.random.seed(42)
    return np.random.rand(batch_size, *IMG_SHAPE).astype(np.float32)

# ================= SESSION CREATION (GPU) =================
def create_optimized_cuda_session(model_path):
    so = ort.SessionOptions()
    
    # ---------------- OPTIMIZATIONS ----------------
    # Enable extended graph optimizations (fusions, constant folding, etc.)
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

    # Enable CPU memory arena (even with GPU EP, useful for CPU fallback)
    so.enable_cpu_mem_arena = True

    # Enable memory pattern optimization (reuses memory buffers)
    so.enable_mem_pattern = True

    # Intra-op threads (affects CPU kernels, useful if fallback occurs)
    so.intra_op_num_threads = 2

    # Inter-op threads (affects parallel execution of independent nodes)
    so.inter_op_num_threads = 2

    # ---------------- END OPTIMIZATIONS ----------------
    
    # Use CUDAExecutionProvider first, fallback to CPUExecutionProvider
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    return ort.InferenceSession(model_path, sess_options=so, providers=providers)

# ================= INFERENCE =================
def run_single_inference(session, input_data):
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_data})
    print("Single GPU inference completed")
    return outputs

# ================= MAIN =================
def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    print(f"Loading model: {MODEL_PATH} ({os.path.getsize(MODEL_PATH)/1024/1024:.1f} MB)")
    session = create_optimized_cuda_session(MODEL_PATH)

    input_data = generate_input(BATCH_SIZE)
    print(f"Running single GPU inference with input shape: {input_data.shape}")
    run_single_inference(session, input_data)

if __name__ == "__main__":
    main()
