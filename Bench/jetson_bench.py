import onnxruntime as ort
import numpy as np
import itertools
import time
import json
from tqdm import tqdm
import subprocess
import threading
import queue
import re

# ================= CONFIG =================
MODEL_PATH = "Models/yolo12n_op12.onnx"  # <-- replace with your ONNX model path
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

# ================= TEGRASTATS CAPTURE =================
def start_tegrastats():
    """
    Launches tegrastats in the background and captures its output.
    """
    q = queue.Queue()

    proc = subprocess.Popen(
        ["tegrastats", "--interval", "200"],   # 200 ms sampling
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    def reader():
        for line in proc.stdout:
            q.put(line.strip())

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    return proc, q

def stop_tegrastats(proc):
    """
    Stops the tegrastats process cleanly.
    """
    proc.terminate()
    try:
        proc.wait(timeout=1)
    except:
        proc.kill()

def parse_tegrastats_line(line):
    """
    Extract useful metrics from one tegrastats line.
    """
    result = {}

    # RAM: RAM 400/3950MB
    m = re.search(r"RAM\s+(\d+)/(\d+)MB", line)
    if m:
        result["ram_used_mb"] = int(m.group(1))
        result["ram_total_mb"] = int(m.group(2))

    # GPU load: GR3D_FREQ 50%
    m = re.search(r"GR3D_FREQ (\d+)%", line)
    if m:
        result["gpu_load_percent"] = int(m.group(1))

    # GPU temperature: GPU@37.5C
    m = re.search(r"GPU@(\d+(\.\d+)?)C", line)
    if m:
        result["gpu_temp_c"] = float(m.group(1))

    # CPU temperature: CPU@40C
    m = re.search(r"CPU@(\d+(\.\d+)?)C", line)
    if m:
        result["cpu_temp_c"] = float(m.group(1))

    # EMC / memory controller: EMC_FREQ 20%
    m = re.search(r"EMC_FREQ (\d+)%", line)
    if m:
        result["emc_load_percent"] = int(m.group(1))

    return result

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

        # =================================================== #
        # Start tegrastats capture
        # =================================================== #
        proc, q = start_tegrastats()
        tegra_samples = []

        # --- Warmup ---
        session = create_session(opt_value, intra, inter, ep)
        input_name = session.get_inputs()[0].name

        if warmup:
            for _ in range(NUM_WARMUP):
                _ = session.run(None, {input_name: input_data})

        # --- Normal inference ---
        latencies = []
        for _ in tqdm(range(NUM_RUNS), desc=f"Normal inference {desc}", leave=False):
            _, latency = run_inference(session, input_data, input_name)
            latencies.append(latency)

            # Collect tegrastats lines
            while not q.empty():
                parsed = parse_tegrastats_line(q.get())
                if parsed:
                    tegra_samples.append(parsed)

        # Stop tegrastats
        stop_tegrastats(proc)

        # Save result
        result = {
            "description": desc,
            "avg_latency_ms": float(np.mean(latencies) * 1000),
            "std_latency_ms": float(np.std(latencies) * 1000),
            "profiling_enabled": False,
            "profile_file": None,
            "tegrastats": tegra_samples
        }
        results.append(result)

        # =================================================== #
        # PROFILING SESSION (with tegrastats)
        # =================================================== #
        proc_p, q_p = start_tegrastats()
        tegra_samples_prof = []

        profile_prefix = f"profile_{ep}_{opt_name}_i{intra}_o{inter}_b{batch}_w{warmup}"
        session_prof = create_session(opt_value, intra, inter, ep, enable_profiling=True, profile_file=profile_prefix)
        input_name_prof = session_prof.get_inputs()[0].name

        if warmup:
            for _ in range(NUM_WARMUP):
                _ = session_prof.run(None, {input_name_prof: input_data})

        latencies_prof = []
        for _ in tqdm(range(NUM_RUNS), desc=f"Profiling inference {desc}", leave=False):
            _, latency = run_inference(session_prof, input_data, input_name_prof)
            latencies_prof.append(latency)

            while not q_p.empty():
                parsed = parse_tegrastats_line(q_p.get())
                if parsed:
                    tegra_samples_prof.append(parsed)

        stop_tegrastats(proc_p)

        profile_file_path = session_prof.end_profiling()

        result_prof = {
            "description": desc,
            "avg_latency_ms": float(np.mean(latencies_prof) * 1000),
            "std_latency_ms": float(np.std(latencies_prof) * 1000),
            "profiling_enabled": True,
            "profile_file": profile_file_path,
            "tegrastats": tegra_samples_prof
        }
        results.append(result_prof)

# ================= SAVE RESULTS =================
with open("onnx_ablation_cpu_gpu_profiling_tegrastats.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nAblation study + tegrastats monitoring completed.")
