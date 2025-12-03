import onnxruntime as ort
import numpy as np
import os

MODEL_PATH = "Models/yolo12n_op12.onnx"
IMG_SHAPE = (3, 640, 640)
BATCH = 1

def generate_input(batch_size):
    np.random.seed(42)
    return np.random.rand(batch_size, *IMG_SHAPE).astype(np.float32)

def create_cuda_session(model_path):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    so.enable_cpu_mem_arena = True
    so.enable_mem_pattern = True
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ort.InferenceSession(model_path, sess_options=so, providers=providers)

def run_inference(session, input_data):
    binding = session.io_binding()

    # Bind input
    input_name = session.get_inputs()[0].name
    binding.bind_input(name=input_name, device_type='cuda', element_type=np.float32, shape=input_data.shape, buffer_ptr=input_data.__array_interface__['data'][0])

    # Bind output
    output_name = session.get_outputs()[0].name
    out_shape = session.get_outputs()[0].shape
    output_np = np.empty(out_shape, dtype=np.float32)
    binding.bind_output(name=output_name, device_type='cuda', element_type=np.float32, shape=out_shape, buffer_ptr=output_np.__array_interface__['data'][0])

    session.run_with_iobinding(binding)
    return output_np

def main():
    if not os.path.exists(MODEL_PATH):
        print("Model not found")
        return

    print("Loading model:", MODEL_PATH)
    sess = create_cuda_session(MODEL_PATH)

    input_data = generate_input(BATCH)
    print("Running inference with input shape:", input_data.shape)
    output = run_inference(sess, input_data)

    print("Done. Output shape:", output.shape)

if __name__ == "__main__":
    main()
