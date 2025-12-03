import onnxruntime as ort
import numpy as np
import torch
import os

# ================= CONFIG =================
MODEL_PATH = "Models/yolo12n_op12.onnx"
IMG_SHAPE = (3, 640, 640)
BATCH = 1
USE_CUSTOM_STREAM = True     # set False if you don't want custom stream
DEVICE_ID = 0


# ================= DEVICE INPUT GENERATION =================
def generate_device_tensor(batch):
    np.random.seed(42)
    cpu_arr = np.random.rand(batch, *IMG_SHAPE).astype(np.float32)

    # pinned CPU memory – async transfers
    pinned = torch.empty(cpu_arr.shape, dtype=torch.float32, pin_memory=True)
    pinned[:] = torch.from_numpy(cpu_arr)

    # Allocate directly on GPU as ORT device tensor
    ort_in = ort.OrtValue.ortvalue_from_numpy(
        pinned.numpy(),
        device="cuda",
        device_id=DEVICE_ID
    )
    return ort_in


# ================= SESSION CREATION =================
def create_cuda_session(path):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    so.enable_cpu_mem_arena = True
    so.enable_mem_pattern = True

    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1

    provider = "CUDAExecutionProvider"
    ep_options = {}

    # attach custom CUDA compute stream
    if USE_CUSTOM_STREAM:
        s = torch.cuda.Stream(device=DEVICE_ID)
        ep_options["user_compute_stream"] = str(s.cuda_stream)

    return ort.InferenceSession(
        path,
        sess_options=so,
        providers=[provider],
        provider_options=[ep_options] if ep_options else [{}]
    )


# ================= RUN WITH IO BINDING =================
def infer_with_iobinding(session, ort_input):
    binding = session.io_binding()

    in_name = session.get_inputs()[0].name
    in_device = ort_input.device_name()
    binding.bind_ortvalue_input(in_name, ort_input)

    # Allocate output on GPU as device tensor
    out_name = session.get_outputs()[0].name
    out_shape = session.get_outputs()[0].shape
    out_tensor = ort.OrtValue.ortvalue_from_shape_and_type(
        out_shape,
        np.float32,
        device="cuda",
        device_id=DEVICE_ID
    )
    binding.bind_ortvalue_output(out_name, out_tensor)

    session.run_with_iobinding(binding)
    return out_tensor


# ================= MAIN =================
def main():
    if not os.path.exists(MODEL_PATH):
        print("Model not found.")
        return

    print("Loading model:", MODEL_PATH)
    sess = create_cuda_session(MODEL_PATH)

    print("Allocating device tensors...")
    device_input = generate_device_tensor(BATCH)

    print("Running GPU inference (zero-copy)...")
    out = infer_with_iobinding(sess, device_input)

    print("Done. Output GPU tensor shape:", out.shape())


if __name__ == "__main__":
    main()
