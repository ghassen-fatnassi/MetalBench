"""
Comprehensive ONNX Runtime Benchmarking Script for Jetson
Fixed for Python 3.6 compatibility and Jetson-specific issues
"""
import onnxruntime as ort
import numpy as np
import itertools
import time
import json
import gc
from tqdm import tqdm
import subprocess
import threading
import queue
import re
import sys
import os
from datetime import datetime
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# ================= CONFIG =================
MODEL_PATH = "Models/yolo12n_op12.onnx"
NUM_WARMUP = 3  # Reduced for Jetson
NUM_RUNS = 30   # Reduced but still statistically significant
IMG_SHAPE = (3, 128, 128)
COOLING_DELAY = 5.0  # Reduced cooling delay
TEGRASTATS_INTERVAL = 100  # ms - Jetson-compatible
MIN_RUN_TIME = 1.0
TIMEOUT_SECONDS = 60.0  # Increased timeout for Jetson
RESULTS_PATH = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"


def measure_inference(session, input_data, input_name):
    """Measure inference time with high precision"""
    start = time.perf_counter()
    outputs = session.run(None, {input_name: input_data})
    end = time.perf_counter()
    return outputs, end - start

def create_session(optimization, intra, inter, execution_provider, enable_profiling=False, profile_file=None):
    """Create ONNX Runtime session with options"""
    so = ort.SessionOptions()
    so.intra_op_num_threads = intra
    so.inter_op_num_threads = inter
    so.graph_optimization_level = optimization
    
    # Disable some optimizations for Jetson stability
    so.enable_cpu_mem_arena = True
    so.enable_mem_pattern = True
    
    # Enable profiling if requested
    if enable_profiling:
        so.enable_profiling = True
        if profile_file:
            so.profile_file_prefix = profile_file
    
    # Select execution provider - Jetson specific
    if execution_provider == "CPU":
        providers = ["CPUExecutionProvider"]
    elif execution_provider == "CUDA":
        # Jetson-specific CUDA provider settings
        providers = [("CUDAExecutionProvider", {
            'device_id': 0,
            'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
        }), "CPUExecutionProvider"]
    else:
        raise ValueError(f"Unknown execution provider: {execution_provider}")
    
    try:
        session = ort.InferenceSession(MODEL_PATH, sess_options=so, providers=providers)
        # Warm up session creation
        _ = session.get_inputs()[0].name
        return session
    except Exception as e:
        print(f"Failed to create session: {e}")
        # Try with default providers
        return ort.InferenceSession(MODEL_PATH, sess_options=so)

def reset_system_state():
    """
    Reset system state between runs.
    """
    print(f"Cooling down for {COOLING_DELAY:.1f} seconds...")
    time.sleep(COOLING_DELAY)
    
    # Force garbage collection
    gc.collect()

# ================= TEGRASTATS MONITOR (Python 3.6 compatible) =================
class TegrastatsMonitor:
    """Thread-safe tegrastats monitor for Jetson"""
    
    def __init__(self, interval_ms=100):
        self.interval_ms = interval_ms
        self.proc = None
        self.queue = queue.Queue()
        self.reader_thread = None
        self.running = False
        self.samples = []  # List of (timestamp, metrics)
        self.start_time = None
        
    def start(self):
        """Start tegrastats monitoring"""
        if self.running:
            return
        
        try:
            # Python 3.6 compatible subprocess call
            self.proc = subprocess.Popen(
                ["tegrastats", "--interval", str(self.interval_ms)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True  # Python 3.6 compatibility
            )
            
            self.running = True
            self.start_time = time.perf_counter()
            self.reader_thread = threading.Thread(target=self._reader)
            self.reader_thread.daemon = True
            self.reader_thread.start()
            print(f"Tegrastats started with {self.interval_ms}ms interval")
            
        except Exception as e:
            print(f"Warning: Could not start tegrastats: {e}")
            self.running = False
    
    def _reader(self):
        """Read tegrastats output"""
        while self.running and self.proc:
            try:
                line = self.proc.stdout.readline()
                if not line:  # EOF
                    break
                timestamp = time.perf_counter() - self.start_time
                metrics = self.parse_tegrastats_line(line.strip())
                if metrics:
                    self.queue.put((timestamp, metrics))
            except Exception as e:
                print(f"Error reading tegrastats: {e}")
                break
    
    def get_sample_at_time(self, target_time, window_ms=50):
        """
        Get the tegrastats sample closest to target_time within window_ms
        """
        if not self.running:
            return None
        
        # Try to get a sample from queue
        try:
            # Process any pending samples
            while True:
                timestamp, metrics = self.queue.get_nowait()
                self.samples.append((timestamp, metrics))
        except queue.Empty:
            pass
        
        # Find closest sample
        if not self.samples:
            return None
        
        closest_sample = None
        closest_diff = float('inf')
        
        for timestamp, metrics in self.samples:
            diff = abs(timestamp - target_time)
            if diff <= window_ms / 1000.0 and diff < closest_diff:
                closest_diff = diff
                closest_sample = metrics
        
        return closest_sample
    
    def get_all_samples(self):
        """Get all collected samples"""
        # Drain the queue
        try:
            while True:
                self.samples.append(self.queue.get_nowait())
        except queue.Empty:
            pass
        return [s[1] for s in self.samples if s[1]]
    
    @staticmethod
    def parse_tegrastats_line(line):
        """Parse tegrastats output line - handles Python 3.6 bytes/string"""
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
        
        # CPU frequencies
        m = re.search(r"CPU (\d+)%@(\d+)", line)
        if m:
            result["cpu_load_percent"] = int(m.group(1))
            result["cpu_freq_mhz"] = int(m.group(2))
        
        return result if result else None
    
    def stop(self):
        """Stop tegrastats monitoring"""
        if self.running:
            self.running = False
            if self.proc:
                try:
                    self.proc.terminate()
                    self.proc.wait(timeout=1)
                except:
                    try:
                        self.proc.kill()
                    except:
                        pass
            if self.reader_thread:
                self.reader_thread.join(timeout=2)
            print("Tegrastats stopped")

# ================= STATISTICAL ANALYSIS =================
def calculate_statistics(data, confidence_level=0.95):
    """Calculate comprehensive statistics"""
    if not data:
        return {}
    
    try:
        data_array = np.array(data)
        n = len(data_array)
        
        # Basic statistics
        mean = float(np.mean(data_array))
        std = float(np.std(data_array))
        median = float(np.median(data_array))
        min_val = float(np.min(data_array))
        max_val = float(np.max(data_array))
        
        # Percentiles
        percentiles = {
            'p5': float(np.percentile(data_array, 5)),
            'p25': float(np.percentile(data_array, 25)),
            'p50': float(median),
            'p75': float(np.percentile(data_array, 75)),
            'p95': float(np.percentile(data_array, 95))
        }
        
        # Throughput calculation
        throughput = 1.0 / mean if mean > 0 else 0
        
        return {
            'n_samples': n,
            'mean_ms': mean * 1000,
            'std_ms': std * 1000,
            'median_ms': median * 1000,
            'min_ms': min_val * 1000,
            'max_ms': max_val * 1000,
            'percentiles_ms': {k: v * 1000 for k, v in percentiles.items()},
            'throughput_fps': throughput,
            'cv_percent': (std / mean * 100) if mean > 0 else 0.0
        }
    except Exception as e:
        print(f"Error calculating statistics: {e}")
        # Fallback to simple stats
        if data:
            mean = sum(data) / len(data)
            return {
                'n_samples': len(data),
                'mean_ms': mean * 1000,
                'throughput_fps': 1.0 / mean if mean > 0 else 0
            }
        return {}

# ================= BENCHMARKING FUNCTION =================
def benchmark_configuration(config_dict, monitor=None, enable_profiling=False):
    """
    Benchmark a single configuration with proper isolation
    """
    print(f"\n{'='*60}")
    print(f"Testing: {config_dict['description']}")
    print(f"{'='*60}")
    
    # Reset system state before starting
    reset_system_state()
    
    session = None
    result = {
        "config": config_dict,
        "latency_stats": {},
        "system_metrics": [],
        "profile_file": None,
        "timestamp": datetime.now().isoformat(),
        "success": False,
        "error_message": None
    }
    
    try:
        # Create session ONCE (excluded from timing)
        print("Creating session...")
        session = create_session(
            optimization=config_dict['optimization'],
            intra=config_dict['intra'],
            inter=config_dict['inter'],
            execution_provider=config_dict['execution_provider'],
            enable_profiling=enable_profiling,
            profile_file=config_dict.get('profile_prefix')
        )
        
        # Get input name and prepare data
        input_name = session.get_inputs()[0].name
        print(f"Input name: {input_name}")
        
        # Generate input data
        input_data = generate_input(config_dict['batch'],config_dict['resolution'])
        print(f"Input shape: {input_data.shape}")
        
        # Start monitoring if available
        if monitor:
            monitor.start()
            time.sleep(0.2)  # Allow monitor to start
        
        # Warmup phase (separate from measurements)
        print(f"Warming up ({NUM_WARMUP} iterations)...")
        warmup_start = time.time()
        
        warmup_success = 0
        for i in range(NUM_WARMUP):
            try:
                _ = session.run(None, {input_name: input_data})
                warmup_success += 1
                if (i + 1) % 10 == 0:
                    print(f"  Warmup iteration {i+1}/{NUM_WARMUP}")
            except Exception as e:
                print(f"  Warmup iteration {i+1} failed: {e}")
        
        if warmup_success == 0:
            raise RuntimeError("All warmup iterations failed")
        
        warmup_time = time.time() - warmup_start
        print(f"Warmup completed: {warmup_success}/{NUM_WARMUP} in {warmup_time:.2f}s")
        
        # Main measurement loop
        print(f"Running {NUM_RUNS} inference iterations...")

        latencies = []
        system_samples = []
        
        run_start = time.time()
        iterations_completed = 0
        
        # Progress bar
        pbar = tqdm(total=NUM_RUNS, desc="Inference", unit="iter", ncols=80)
        
        for i in range(NUM_RUNS):
            # Check timeout
            if time.time() - run_start > TIMEOUT_SECONDS:
                print(f"Timeout after {TIMEOUT_SECONDS}s")
                break
            
            # Mark inference start time
            inference_start = time.perf_counter()
            
            try:
                # Run inference
                outputs = session.run(None, {input_name: input_data})
                
                # Calculate latency
                latency = time.perf_counter() - inference_start
                latencies.append(latency/config_dict['batch'])
                iterations_completed += 1
                
                # Get synchronized system metrics
                if monitor:
                    sample = monitor.get_sample_at_time(inference_start, window_ms=50)
                    if sample:
                        sample['iteration'] = i
                        sample['latency_ms'] = latency * 1000
                        system_samples.append(sample)
                
                # Update progress bar
                pbar.update(1)
                if iterations_completed > 0:
                    avg_latency = sum(latencies) / len(latencies) * 1000
                    pbar.set_postfix({'avg_ms': f'{avg_latency:.1f}'})
                
            except Exception as e:
                print(f"\nInference iteration {i+1} failed: {e}")
                # Continue with next iteration
        
        pbar.close()
        run_time = time.time() - run_start
        
        if iterations_completed == 0:
            raise RuntimeError("No successful inference iterations")
        
        # Calculate statistics
        stats = calculate_statistics(latencies)
        
        # Collect remaining system metrics
        if monitor:
            remaining_samples = monitor.get_all_samples()
            system_samples.extend(remaining_samples)
            monitor.stop()
        
        # Create result
        result["latency_stats"] = stats
        result["system_metrics"] = system_samples
        result["success"] = True
        
        # Get profile file if profiling was enabled
        if enable_profiling and session:
            try:
                result["profile_file"] = session.end_profiling()
            except:
                pass
        
        print(f"Benchmark completed: {stats.get('mean_ms', 0):.1f} ± {stats.get('std_ms', 0):.1f} ms")
        print(f"Throughput: {stats.get('throughput_fps', 0):.1f} FPS")
        print(f"Completed {iterations_completed}/{NUM_RUNS} iterations in {run_time:.1f}s")
        
    except Exception as e:
        result["success"] = False
        result["error_message"] = str(e)
        print(f"Error during benchmarking: {e}")
        
        # Try to get error details
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        if monitor and monitor.running:
            monitor.stop()
        
        if session:
            try:
                # End profiling if active
                if enable_profiling:
                    try:
                        session.end_profiling()
                    except:
                        pass
            except:
                pass
            del session
        
        # Force garbage collection
        gc.collect()
    
    return result

# ================= EXPERIMENTAL DESIGN =================
def generate_test_configurations():
    """
    Generate test configurations for Jetson, iterating over resolutions from 64 to 640
    """
    # Map optimization levels
    opt_map = {
        "Disabled": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "Basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "Extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    }
    
    configurations = []
    
    # Test CPU vs CUDA
    for ep in ["CUDA"]:
        for opt_name in ["Disabled","Extended"]:  # Just test extended for now
            for batch in [1, 2, 4, 8]:
                for warmup in [True]:
                    # Set reasonable thread counts for Jetson
                    if ep == "CPU":
                        intra_options = [2]  # Jetson typically has 4+ cores
                        inter_options = [2]
                    else:
                        intra_options =[2]
                        inter_options = [2]
                    
                    for intra in intra_options:
                        for inter in inter_options:
                            for res in range(128, 513, 128):
                                config = {
                                    'optimization': opt_map[opt_name],
                                    'intra': intra,
                                    'inter': inter,
                                    'batch': batch,
                                    'warmup': warmup,
                                    'execution_provider': ep,
                                    'resolution': res,
                                    'description': f"EP:{ep}, Opt:{opt_name}, intra:{intra}, inter:{inter}, batch:{batch}, res:{res}, warmup:{warmup}",
                                    'profile_prefix': f"profile_{ep}_{opt_name}_i{intra}_o{inter}_b{batch}_r{res}_w{warmup}"
                                }
                                configurations.append(config)
    
    return configurations

# ================= HELPER FUNCTION UPDATE =================
def generate_input(batch_size, resolution=None):
    """Generate consistent random input"""
    np.random.seed(42)
    if resolution is None:
        resolution = 128
    return np.random.rand(batch_size, 3, resolution, resolution).astype(np.float32)


# ================= MAIN EXECUTION =================
def main():
    """Main benchmarking execution"""
    print("="*70)
    print("ONNX Runtime Benchmarking for Jetson")
    print(f"Model: {MODEL_PATH}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print(f"Current directory: {os.getcwd()}")
        print(f"Looking for: {os.path.abspath(MODEL_PATH)}")
        return
    
    print(f"Model found: {os.path.getsize(MODEL_PATH) / 1024 / 1024:.1f} MB")
    
    # Generate configurations
    configurations = generate_test_configurations()
    print(f"\nGenerated {len(configurations)} test configurations")
    
    # Initialize monitoring
    monitor = TegrastatsMonitor(interval_ms=TEGRASTATS_INTERVAL)
    
    # Check if tegrastats is available
    try:
        subprocess.run(["which", "tegrastats"], check=True, stdout=subprocess.PIPE)
        print("Tegrastats is available")
    except:
        print("Warning: tegrastats not found. System metrics will not be collected.")
        monitor = None
    
    # Baseline measurement (idle system)
    if monitor:
        print("\nMeasuring baseline system state...")
        monitor.start()
        time.sleep(2.0)
        baseline_samples = monitor.get_all_samples()
        monitor.stop()
        print(f"Collected {len(baseline_samples)} baseline samples")
    else:
        baseline_samples = []
    
    # Run benchmarks
    all_results = []
    
    for i, config_dict in enumerate(configurations):
        print(f"\n{'#'*70}")
        print(f"Configuration {i+1}/{len(configurations)}")
        print(f"Description: {config_dict['description']}")
        print(f"{'#'*70}")
        
        # Run without profiling first
        result = benchmark_configuration(config_dict, monitor=monitor, enable_profiling=True)
        
        if result["success"]:
            all_results.append(result)
            
            # Save intermediate results
            with open(RESULTS_PATH, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
            print(f"\nIntermediate results saved to {RESULTS_PATH}")
        else:
            print(f"\nConfiguration failed: {result.get('error_message', 'Unknown error')}")
            # Save failed result for debugging
            all_results.append(result)
    
    # Generate summary
    generate_summary(all_results)
    
    print("\n" + "="*70)
    print("Benchmarking completed!")
    print(f"Results saved to: {RESULTS_PATH}")
    print("="*70)

def generate_summary(results):
    """Generate a summary report"""
    successful = [r for r in results if r.get("success", False)]
    
    if not successful:
        print("No successful benchmarks to summarize")
        return
    
    summary = {
        'total_configurations': len(results),
        'successful_configurations': len(successful),
        'failed_configurations': len(results) - len(successful),
        'timestamp': datetime.now().isoformat(),
        'best_performance': None,
        'worst_performance': None,
        'by_execution_provider': {},
        'by_batch_size': {}
    }
    
    # Find best and worst performance
    latencies = []
    for result in successful:
        stats = result.get('latency_stats', {})
        if 'mean_ms' in stats:
            latencies.append((stats['mean_ms'], result['config']['description']))
    
    if latencies:
        best = min(latencies, key=lambda x: x[0])
        worst = max(latencies, key=lambda x: x[0])
        
        summary['best_performance'] = {
            'latency_ms': best[0],
            'configuration': best[1]
        }
        summary['worst_performance'] = {
            'latency_ms': worst[0],
            'configuration': worst[1]
        }
    
    # Group by execution provider
    for result in successful:
        ep = result['config']['execution_provider']
        if ep not in summary['by_execution_provider']:
            summary['by_execution_provider'][ep] = []
        stats = result.get('latency_stats', {})
        summary['by_execution_provider'][ep].append({
            'latency_ms': stats.get('mean_ms', 0),
            'throughput_fps': stats.get('throughput_fps', 0),
            'configuration': result['config']['description']
        })
    
    # Group by batch size
    for result in successful:
        batch = result['config']['batch']
        if batch not in summary['by_batch_size']:
            summary['by_batch_size'][batch] = []
        stats = result.get('latency_stats', {})
        summary['by_batch_size'][batch].append({
            'latency_ms': stats.get('mean_ms', 0),
            'throughput_fps': stats.get('throughput_fps', 0),
            'configuration': result['config']['description']
        })
    
    # Save summary
    summary_path = RESULTS_PATH.replace('.json', '_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {summary_path}")
    
    # Print quick summary
    print("\n" + "="*70)
    print("QUICK SUMMARY")
    print("="*70)
    print(f"Successful: {summary['successful_configurations']}/{summary['total_configurations']}")
    
    if summary['best_performance']:
        print(f"\nBest latency: {summary['best_performance']['latency_ms']:.1f}ms")
        print(f"Config: {summary['best_performance']['configuration']}")
    
    # Print by execution provider
    print("\nBy Execution Provider:")
    for ep, configs in summary['by_execution_provider'].items():
        if configs:
            avg_latency = np.mean([c['latency_ms'] for c in configs]) if configs else 0
            avg_throughput = np.mean([c['throughput_fps'] for c in configs]) if configs else 0
            print(f"  {ep}: {avg_latency:.1f}ms, {avg_throughput:.1f} FPS")

if __name__ == "__main__":
    main()
