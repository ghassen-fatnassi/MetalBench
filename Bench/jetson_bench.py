import onnxruntime as ort
import numpy as np
import itertools
import time
import json
from tqdm import tqdm
import os

# ================= CONFIG =================
MODEL_PATH = "path/to/your/yolo12n.onnx"  # <-- replace with your ONNX model path
NUM_WARMUP = 5
NUM_RUNS = 10
IMG_SHAPE = (3, 640, 640)  # (C, H, W)

# ================= HELPER FUNCTIONS =================
def generate_input(batch_size):
    return np.random.rand(batch_size, *IMG_SHAPE).astype(np.float32)

def run_inference(session, input_data, input_name):
    start = time.time()
    outputs = session.run(None, {input_name: input_data})
    end = time.time()
    return outputs, end - start

def create_session(optimization, intra, inter, execution_provider, enable_profiling=False, profile_file=None):
    so = ort.SessionOptions()
    so.intra_op_num_threads = intra
    so.inter_op_num_threads = inter
    so.graph_optimization_level = optimization
    if enable_profiling:
        so.enable_profiling = True
        if profile_file:
            so.profile_file_prefix = profile_file
    if execution_provider == "CPU":
        return ort.InferenceSession(MODEL_PATH, sess_options=so, providers=["CPUExecutionProvider"])
    elif execution_provider == "CUDA":
        return ort.InferenceSession(MODEL_PATH, sess_options=so, providers=["CUDAExecutionProvider"])
    else:
        raise ValueError(f"Unknown execution provider: {execution_provider}")

# ================= CONFIGURATIONS =================
optimizations = [
    ("Disabled", ort.GraphOptimizationLevel.ORT_DISABLE_ALL),
    ("Basic", ort.GraphOptimizationLevel.ORT_ENABLE_BASIC),
    ("Extended", ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED)
]
intra_threads = [1, 4]
inter_threads = [1, 2, 4]
batch_sizes = [1, 2, 4, 8]
warmup_options = [True, False]
execution_providers = ["CPU", "CUDA"]

# ================= RUN ABLATION =================
results = []

for ep in execution_providers:
    config_list = list(itertools.product(optimizations, intra_threads, inter_threads, batch_sizes, warmup_options))
    
    for opt, intra, inter, batch, warmup in tqdm(config_list, desc=f"Ablation on {ep}", ncols=100):
        opt_name, opt_value = opt
        desc = f"EP:{ep}, Opt:{opt_name}, intra:{intra}, inter:{inter}, batch:{batch}, warmup:{warmup}"
        input_data = generate_input(batch)
        
        # --- Warmup (without profiling) ---
        session = create_session(opt_value, intra, inter, ep)
        input_name = session.get_inputs()[0].name
        if warmup:
            for _ in range(NUM_WARMUP):
                _ = session.run(None, {input_name: input_data})

        # --- Run normal inference ---
        latencies = []
        for _ in tqdm(range(NUM_RUNS), desc=f"Normal inference {desc}", leave=False):
            _, latency = run_inference(session, input_data, input_name)
            latencies.append(latency)
        
        result = {
            "description": desc,
            "avg_latency_ms": np.mean(latencies) * 1000,
            "std_latency_ms": np.std(latencies) * 1000,
            "profiling_enabled": False,
            "profile_file": None
        }
        results.append(result)

        # --- Run inference with profiling enabled ---
        profile_prefix = f"profile_{ep}_{opt_name}_intra{intra}_inter{inter}_batch{batch}_warmup{warmup}"
        session_prof = create_session(opt_value, intra, inter, ep, enable_profiling=True, profile_file=profile_prefix)
        input_name_prof = session_prof.get_inputs()[0].name

        # Warmup for profiling session
        if warmup:
            for _ in range(NUM_WARMUP):
                _ = session_prof.run(None, {input_name_prof: input_data})

        latencies_prof = []
        for _ in tqdm(range(NUM_RUNS), desc=f"Profiling inference {desc}", leave=False):
            _, latency = run_inference(session_prof, input_data, input_name_prof)
            latencies_prof.append(latency)

        # Get profile file
        profile_file_path = session_prof.end_profiling()

        result_prof = {
            "description": desc,
            "avg_latency_ms": np.mean(latencies_prof) * 1000,
            "std_latency_ms": np.std(latencies_prof) * 1000,
            "profiling_enabled": True,
            "profile_file": profile_file_path
        }
        results.append(result_prof)

# ================= SAVE RESULTS =================
with open("onnx_ablation_cpu_gpu_profiling.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nAblation study completed. Results saved to onnx_ablation_cpu_gpu_profiling.json")
