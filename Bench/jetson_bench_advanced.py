"""
Comprehensive ONNX Runtime Benchmarking Script for Jetson
Enhanced with multi-layer profiling capabilities
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
import psutil
import platform
import multiprocessing
from collections import defaultdict

# Suppress warnings
warnings.filterwarnings('ignore')

# ================= CONFIG =================
MODEL_PATH = "Models/yolo12n_op12.onnx"
NUM_WARMUP = 30
NUM_RUNS = 100
RESOLUTIONS = [64, 128, 256, 512, 640]
COOLING_DELAY = 5.0
TEGRASTATS_INTERVAL = 100
MIN_RUN_TIME = 1.0
TIMEOUT_SECONDS = 60.0
RESULTS_PATH = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

# Feature flags
ENABLE_ADVANCED_PROFILING = True  # Set to False to disable advanced metrics
ENABLE_PERF_COUNTERS = True       # Linux perf counters
ENABLE_MEMORY_TRACKING = True     # Detailed memory tracking
ENABLE_ENERGY_ESTIMATION = True   # Power/energy estimation

# ================= PROFILING MODULES =================
class BaseProfiler:
    """Base class for all profilers"""
    def start(self):
        pass
    
    def stop(self):
        pass
    
    def get_metrics(self):
        return {}

class SystemProfiler(BaseProfiler):
    """System-level profiling using psutil"""
    
    def __init__(self, pid=None):
        self.pid = pid or os.getpid()
        self.process = psutil.Process(self.pid)
        self.start_time = None
        self.start_cpu_times = None
        self.start_memory_info = None
        self.io_counters_start = None
        self.metrics_history = []
        
    def start(self):
        self.start_time = time.perf_counter()
        self.start_cpu_times = self.process.cpu_times()
        self.start_memory_info = self.process.memory_info()
        
        try:
            self.io_counters_start = self.process.io_counters()
        except:
            self.io_counters_start = None
            
        self.metrics_history = []
        
    def capture_snapshot(self):
        """Capture current system metrics"""
        try:
            snapshot = {
                'timestamp': time.perf_counter() - self.start_time,
                'cpu_percent': self.process.cpu_percent(interval=None),
                'memory_rss_mb': self.process.memory_info().rss / 1024 / 1024,
                'memory_vms_mb': self.process.memory_info().vms / 1024 / 1024,
                'num_threads': self.process.num_threads(),
                'num_fds': self.process.num_fds() if hasattr(self.process, 'num_fds') else 0,
            }
            
            # CPU times
            cpu_times = self.process.cpu_times()
            snapshot.update({
                'user_cpu_s': cpu_times.user,
                'system_cpu_s': cpu_times.system,
            })
            
            self.metrics_history.append(snapshot)
            return snapshot
            
        except Exception as e:
            print(f"Warning: Could not capture system snapshot: {e}")
            return {}
    
    def stop(self):
        """Stop profiling and calculate metrics"""
        elapsed = time.perf_counter() - self.start_time
        
        try:
            end_cpu_times = self.process.cpu_times()
            end_memory_info = self.process.memory_info()
            
            # Calculate CPU usage percentages
            user_cpu_diff = end_cpu_times.user - self.start_cpu_times.user
            system_cpu_diff = end_cpu_times.system - self.start_cpu_times.system
            total_cpu_diff = user_cpu_diff + system_cpu_diff
            
            cpu_percent_total = (total_cpu_diff / elapsed) * 100 if elapsed > 0 else 0
            
            # Memory usage stats
            memory_stats = {}
            if self.metrics_history:
                rss_values = [m['memory_rss_mb'] for m in self.metrics_history if 'memory_rss_mb' in m]
                memory_stats = {
                    'memory_rss_avg_mb': np.mean(rss_values) if rss_values else 0,
                    'memory_rss_max_mb': max(rss_values) if rss_values else 0,
                    'memory_rss_min_mb': min(rss_values) if rss_values else 0,
                }
            
            # I/O statistics if available
            io_stats = {}
            if self.io_counters_start:
                try:
                    io_counters_end = self.process.io_counters()
                    io_stats = {
                        'read_bytes': io_counters_end.read_bytes - self.io_counters_start.read_bytes,
                        'write_bytes': io_counters_end.write_bytes - self.io_counters_start.write_bytes,
                        'read_count': io_counters_end.read_count - self.io_counters_start.read_count,
                        'write_count': io_counters_end.write_count - self.io_counters_start.write_count,
                    }
                except:
                    pass
            
            return {
                'elapsed_time_s': elapsed,
                'cpu_user_s': user_cpu_diff,
                'cpu_system_s': system_cpu_diff,
                'cpu_total_percent': cpu_percent_total,
                'cpu_user_percent': (user_cpu_diff / elapsed) * 100 if elapsed > 0 else 0,
                'cpu_system_percent': (system_cpu_diff / elapsed) * 100 if elapsed > 0 else 0,
                'final_memory_rss_mb': end_memory_info.rss / 1024 / 1024,
                'final_memory_vms_mb': end_memory_info.vms / 1024 / 1024,
                'num_threads': self.process.num_threads(),
                **memory_stats,
                **io_stats,
                'samples_collected': len(self.metrics_history),
            }
            
        except Exception as e:
            print(f"Warning: Error in SystemProfiler.stop(): {e}")
            return {'error': str(e)}

class LinuxPerfProfiler(BaseProfiler):
    """Linux perf event monitoring (if available)"""
    
    def __init__(self, pid=None):
        self.pid = pid or os.getpid()
        self.perf_process = None
        self.output_file = None
        
    def start(self):
        """Start perf stat monitoring"""
        if not ENABLE_PERF_COUNTERS:
            return
            
        try:
            # Create output file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.output_file = f"perf_stats_{self.pid}_{timestamp}.txt"
            
            # Basic perf events for ARM/ Jetson
            events = [
                'cpu-cycles',
                'instructions',
                'cache-references',
                'cache-misses',
                'branch-instructions',
                'branch-misses',
                # ARM specific events if available
                # 'armv8_cortex_a57/br_mis_pred/',
                # 'armv8_cortex_a57/br_pred/',
            ]
            
            # Start perf stat
            cmd = ['perf', 'stat', '-p', str(self.pid)]
            for event in events:
                cmd.extend(['-e', event])
            cmd.extend(['-o', self.output_file])
            
            self.perf_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Give perf time to attach
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Warning: perf not available or failed: {e}")
            self.perf_process = None
    
    def stop(self):
        """Stop perf and parse results"""
        if self.perf_process:
            try:
                self.perf_process.terminate()
                self.perf_process.wait(timeout=2)
                
                # Parse perf output
                if os.path.exists(self.output_file):
                    return self.parse_perf_output(self.output_file)
                    
            except Exception as e:
                print(f"Warning: Error stopping perf: {e}")
                if self.perf_process:
                    try:
                        self.perf_process.kill()
                    except:
                        pass
        
        return {}
    
    @staticmethod
    def parse_perf_output(filename):
        """Parse perf stat output file"""
        metrics = {}
        
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                # Parse lines like "     5,123,456      cpu-cycles"
                parts = line.strip().split()
                if len(parts) >= 2:
                    # Remove commas from numbers
                    value_str = parts[0].replace(',', '')
                    event_name = parts[-1]
                    
                    try:
                        if value_str.replace('.', '').isdigit():
                            value = float(value_str) if '.' in value_str else int(value_str)
                            metrics[event_name] = value
                    except ValueError:
                        pass
                        
                # Parse percentage lines
                if '%' in line and 'of' in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part.endswith('%'):
                            try:
                                percent = float(part[:-1])
                                context = ' '.join(parts[i+1:])
                                metrics[f'{context}_percent'] = percent
                            except:
                                pass
        
        except Exception as e:
            print(f"Warning: Could not parse perf output: {e}")
        
        return metrics

class ONNXProfiler(BaseProfiler):
    """ONNX Runtime specific profiling"""
    
    def __init__(self, session_options=None):
        self.session_options = session_options or ort.SessionOptions()
        self.profile_file = None
        
    def start(self, profile_prefix=None):
        """Enable ONNX Runtime profiling"""
        self.session_options.enable_profiling = True
        if profile_prefix:
            self.session_options.profile_file_prefix = profile_prefix
            
    def stop(self, session):
        """End profiling and get results"""
        try:
            self.profile_file = session.end_profiling()
            
            # Parse profiling data
            if self.profile_file and os.path.exists(self.profile_file):
                return self.parse_onnx_profile(self.profile_file)
                
        except Exception as e:
            print(f"Warning: ONNX profiling error: {e}")
        
        return {}
    
    @staticmethod
    def parse_onnx_profile(profile_file):
        """Parse ONNX Runtime profiling JSON"""
        try:
            with open(profile_file, 'r') as f:
                profile_data = json.load(f)
            
            metrics = {
                'total_duration_ms': 0,
                'node_count': 0,
                'operator_breakdown': defaultdict(float),
                'longest_node_ms': 0,
                'longest_node_name': '',
            }
            
            if isinstance(profile_data, list) and len(profile_data) > 0:
                for event in profile_data:
                    if 'dur' in event:
                        duration_ms = event['dur'] / 1000.0  # Convert ns to ms
                        metrics['total_duration_ms'] += duration_ms
                        
                        # Track operator types
                        if 'name' in event:
                            op_name = event['name']
                            # Extract operator type (e.g., Conv, Add, etc.)
                            for op_type in ['Conv', 'Add', 'Mul', 'Relu', 'BatchNormalization', 'Reshape']:
                                if op_type in op_name:
                                    metrics['operator_breakdown'][op_type] += duration_ms
                            
                            # Track longest node
                            if duration_ms > metrics['longest_node_ms']:
                                metrics['longest_node_ms'] = duration_ms
                                metrics['longest_node_name'] = op_name
                
                metrics['node_count'] = len(profile_data)
            
            return metrics
            
        except Exception as e:
            print(f"Warning: Could not parse ONNX profile: {e}")
            return {}

class EnergyEstimator:
    """Simple energy estimation for Jetson"""
    
    @staticmethod
    def estimate_energy(cpu_time_s, gpu_time_s=None, cpu_power_w=5, gpu_power_w=10):
        """
        Simple energy estimation based on time and typical power consumption
        CPU: ~5W, GPU: ~10W (Jetson Nano estimates)
        """
        cpu_energy_j = cpu_time_s * cpu_power_w
        total_energy_j = cpu_energy_j
        
        if gpu_time_s:
            gpu_energy_j = gpu_time_s * gpu_power_w
            total_energy_j += gpu_energy_j
            
        return {
            'estimated_cpu_energy_j': cpu_energy_j,
            'estimated_gpu_energy_j': gpu_energy_j if gpu_time_s else 0,
            'estimated_total_energy_j': total_energy_j,
            'estimated_cpu_power_w': cpu_power_w,
            'estimated_gpu_power_w': gpu_power_w if gpu_time_s else 0,
        }

class AdvancedProfilingManager:
    """Manager for all profiling components"""
    
    def __init__(self, enable_advanced=True):
        self.enable_advanced = enable_advanced and ENABLE_ADVANCED_PROFILING
        self.profilers = []
        self.metrics = {}
        
    def initialize(self, pid=None):
        """Initialize all profilers"""
        if not self.enable_advanced:
            return
            
        pid = pid or os.getpid()
        
        # System profiler (always available)
        self.system_profiler = SystemProfiler(pid)
        self.profilers.append(self.system_profiler)
        
        # Linux perf profiler (if enabled)
        if ENABLE_PERF_COUNTERS:
            self.perf_profiler = LinuxPerfProfiler(pid)
            self.profilers.append(self.perf_profiler)
        
        # ONNX profiler
        self.onnx_profiler = ONNXProfiler()
        self.profilers.append(self.onnx_profiler)
        
        print(f"Advanced profiling enabled with {len(self.profilers)} profilers")
    
    def start(self):
        """Start all profilers"""
        if not self.enable_advanced:
            return
            
        for profiler in self.profilers:
            try:
                profiler.start()
            except Exception as e:
                print(f"Warning: Profiler {profiler.__class__.__name__} failed to start: {e}")
    
    def capture_system_snapshot(self):
        """Capture a system snapshot during run"""
        if self.enable_advanced and hasattr(self, 'system_profiler'):
            return self.system_profiler.capture_snapshot()
        return {}
    
    def stop(self, session=None):
        """Stop all profilers and collect metrics"""
        if not self.enable_advanced:
            return {}
        
        all_metrics = {}
        
        # Stop profilers in reverse order
        for profiler in reversed(self.profilers):
            try:
                if isinstance(profiler, ONNXProfiler) and session:
                    metrics = profiler.stop(session)
                else:
                    metrics = profiler.stop()
                
                profiler_name = profiler.__class__.__name__.replace('Profiler', '').lower()
                all_metrics[profiler_name] = metrics
                
            except Exception as e:
                print(f"Warning: Profiler {profiler.__class__.__name__} failed to stop: {e}")
        
        self.metrics = all_metrics
        return all_metrics
    
    def get_metrics(self):
        """Get all collected metrics"""
        return self.metrics

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
        
        # Power measurements if available
        m = re.search(r"POM_5V_IN\s+(\d+)/(\d+)", line)
        if m:
            result["power_5v_current_ma"] = int(m.group(1))
            result["power_5v_voltage_mv"] = int(m.group(2))
        
        m = re.search(r"POM_5V_GPU\s+(\d+)/(\d+)", line)
        if m:
            result["power_gpu_current_ma"] = int(m.group(1))
            result["power_gpu_voltage_mv"] = int(m.group(2))
        
        m = re.search(r"POM_5V_CPU\s+(\d+)/(\d+)", line)
        if m:
            result["power_cpu_current_ma"] = int(m.group(1))
            result["power_cpu_voltage_mv"] = int(m.group(2))
        
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

# ================= HELPER FUNCTIONS =================
def generate_input(batch_size, img_shape):
    """Generate consistent random input"""
    # Use fixed seed for reproducibility
    np.random.seed(42)
    return np.random.rand(batch_size, *img_shape).astype(np.float32)

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
    
    # Select execution provider - Jetson specific
    if execution_provider == "CPU":
        providers = ["CPUExecutionProvider"]
    elif execution_provider == "CUDA":
        # Jetson-specific CUDA provider settings
        providers = [("CUDAExecutionProvider", {
            'device_id': 0,
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            'do_copy_in_default_stream': True,
        }), "CPUExecutionProvider"]
    else:
        raise ValueError(f"Unknown execution provider: {execution_provider}")
    
    try:
        session = ort.InferenceSession(MODEL_PATH, sess_options=so, providers=providers)
        # Warm up session creation
        _ = session.get_inputs()[0].name
        return session, so
    except Exception as e:
        print(f"Failed to create session: {e}")
        # Try with default providers
        return ort.InferenceSession(MODEL_PATH, sess_options=so), so

def reset_system_state():
    """
    Reset system state between runs.
    """
    print(f"Cooling down for {COOLING_DELAY:.1f} seconds...")
    time.sleep(COOLING_DELAY)
    
    # Force garbage collection
    gc.collect()

def get_system_info():
    """Collect comprehensive system information"""
    info = {
        'timestamp': datetime.now().isoformat(),
        'python_version': platform.python_version(),
        'onnxruntime_version': ort.__version__,
        'system': platform.system(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'cpu_count': multiprocessing.cpu_count(),
        'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {},
        'memory_total_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024,
    }
    
    # Get GPU info if available
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_info = {
            'gpu_name': pynvml.nvmlDeviceGetName(handle).decode('utf-8'),
            'gpu_memory_total_mb': pynvml.nvmlDeviceGetMemoryInfo(handle).total / 1024 / 1024,
            'gpu_driver_version': pynvml.nvmlSystemGetDriverVersion().decode('utf-8'),
        }
        info['gpu'] = gpu_info
        pynvml.nvmlShutdown()
    except:
        info['gpu'] = {'available': False}
    
    # Jetson-specific info
    try:
        # Check if we're on Jetson
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read().strip('\x00')
            info['device_model'] = model
    except:
        pass
    
    return info

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
        
        # Coefficient of variation
        cv_percent = (std / mean * 100) if mean > 0 else 0.0
        
        # Confidence interval (simplified)
        if n > 1:
            from scipy import stats
            t_value = stats.t.ppf((1 + confidence_level) / 2, n - 1)
            ci_half_width = t_value * std / np.sqrt(n)
            confidence_interval = {
                'lower_ms': (mean - ci_half_width) * 1000,
                'upper_ms': (mean + ci_half_width) * 1000,
                'width_ms': (2 * ci_half_width) * 1000,
            }
        else:
            confidence_interval = {}
        
        return {
            'n_samples': n,
            'mean_ms': mean * 1000,
            'std_ms': std * 1000,
            'median_ms': median * 1000,
            'min_ms': min_val * 1000,
            'max_ms': max_val * 1000,
            'percentiles_ms': {k: v * 1000 for k, v in percentiles.items()},
            'throughput_fps': throughput,
            'cv_percent': cv_percent,
            'confidence_interval_ms': confidence_interval,
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
def benchmark_configuration(config_dict, monitor=None, advanced_profiling=None):
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
        "advanced_metrics": {},
        "profile_file": None,
        "timestamp": datetime.now().isoformat(),
        "success": False,
        "error_message": None
    }
    
    try:
        # Get image shape from config
        resolution = config_dict['resolution']
        img_shape = (3, resolution, resolution)
        
        # Initialize advanced profiling
        if advanced_profiling:
            advanced_profiling.initialize()
        
        # Create session ONCE (excluded from timing)
        print(f"Creating session for resolution {resolution}x{resolution}...")
        session, session_options = create_session(
            optimization=config_dict['optimization'],
            intra=config_dict['intra'],
            inter=config_dict['inter'],
            execution_provider=config_dict['execution_provider']
        )
        
        # Setup ONNX profiling if advanced profiling is enabled
        if advanced_profiling and advanced_profiler.enable_advanced:
            advanced_profiler.onnx_profiler.session_options = session_options
            advanced_profiler.onnx_profiler.start(config_dict.get('profile_prefix'))
        
        # Get input name and prepare data
        input_name = session.get_inputs()[0].name
        print(f"Input name: {input_name}")
        
        # Generate input data
        input_data = generate_input(config_dict['batch'], img_shape)
        print(f"Input shape: {input_data.shape}")
        
        # Start monitoring if available
        if monitor:
            monitor.start()
            time.sleep(0.2)  # Allow monitor to start
        
        # Start advanced profiling
        if advanced_profiling:
            advanced_profiling.start()
        
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
                # Capture system snapshot if advanced profiling
                if advanced_profiling:
                    snapshot = advanced_profiling.capture_system_snapshot()
                    if snapshot:
                        snapshot['iteration'] = i
                        system_samples.append(snapshot)
                
                # Run inference
                outputs = session.run(None, {input_name: input_data})
                
                # Calculate latency
                latency = time.perf_counter() - inference_start
                latencies.append(latency/config_dict['batch'])
                iterations_completed += 1
                
                # Get synchronized system metrics from tegrastats
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
        
        # Stop advanced profiling and collect metrics
        advanced_metrics = {}
        if advanced_profiling:
            advanced_metrics = advanced_profiling.stop(session)
            result["advanced_metrics"] = advanced_metrics
        
        # Add energy estimation
        if ENABLE_ENERGY_ESTIMATION and stats.get('mean_ms'):
            total_inference_time = stats['mean_ms'] * iterations_completed / 1000
            energy_estimation = EnergyEstimator.estimate_energy(
                cpu_time_s=total_inference_time,
                gpu_time_s=total_inference_time if config_dict['execution_provider'] == 'CUDA' else None
            )
            stats['energy_estimation'] = energy_estimation
        
        # Create result
        result["latency_stats"] = stats
        result["system_metrics"] = system_samples
        result["success"] = True
        
        # Get profile file if profiling was enabled
        if advanced_profiling and advanced_profiling.onnx_profiler.profile_file:
            result["profile_file"] = advanced_profiling.onnx_profiler.profile_file
        
        print(f"\nBenchmark completed:")
        print(f"  Latency: {stats.get('mean_ms', 0):.1f} ± {stats.get('std_ms', 0):.1f} ms")
        print(f"  Throughput: {stats.get('throughput_fps', 0):.1f} FPS")
        print(f"  CV: {stats.get('cv_percent', 0):.1f}%")
        print(f"  Completed {iterations_completed}/{NUM_RUNS} iterations in {run_time:.1f}s")
        
        # Print advanced metrics summary
        if advanced_metrics:
            print(f"\nAdvanced Metrics:")
            for profiler_name, metrics in advanced_metrics.items():
                if metrics:
                    print(f"  {profiler_name}: {len(metrics)} metrics collected")
                    if 'cpu_total_percent' in metrics:
                        print(f"    CPU Usage: {metrics['cpu_total_percent']:.1f}%")
                    if 'estimated_total_energy_j' in metrics:
                        print(f"    Energy: {metrics['estimated_total_energy_j']:.2f} J")
        
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
                if advanced_profiling:
                    try:
                        advanced_profiling.stop(session)
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
    Generate test configurations for Jetson with multiple resolutions
    """
    # Map optimization levels
    opt_map = {
        "Disabled": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "Basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "Extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    }
    
    configurations = []
    
    # Test all resolutions
    for resolution in RESOLUTIONS:
        # Test CPU vs CUDA
        for ep in ["CPU", "CUDA"]:
            for opt_name in ["Extended"]:  # Just test extended for now
                for batch in [1, 2, 4, 8]:
                    for warmup in [True]:
                        # Set reasonable thread counts for Jetson
                        if ep == "CPU":
                            intra_options = [4]  # Jetson typically has 4+ cores
                            inter_options = [1]
                        else:
                            intra_options = [1]
                            inter_options = [1]
                        
                        for intra in intra_options:
                            for inter in inter_options:
                                config = {
                                    'resolution': resolution,
                                    'optimization': opt_map[opt_name],
                                    'intra': intra,
                                    'inter': inter,
                                    'batch': batch,
                                    'warmup': warmup,
                                    'execution_provider': ep,
                                    'description': f"Res:{resolution}x{resolution}, EP:{ep}, Opt:{opt_name}, intra:{intra}, inter:{inter}, batch:{batch}, warmup:{warmup}",
                                    'profile_prefix': f"profile_r{resolution}_{ep}_{opt_name}_i{intra}_o{inter}_b{batch}_w{warmup}"
                                }
                                configurations.append(config)
    
    return configurations

# ================= MAIN EXECUTION =================
def main():
    """Main benchmarking execution"""
    print("="*80)
    print("ENHANCED ONNX Runtime Benchmarking for Jetson")
    print(f"Model: {MODEL_PATH}")
    print(f"Testing resolutions: {RESOLUTIONS}")
    print(f"Advanced profiling: {'ENABLED' if ENABLE_ADVANCED_PROFILING else 'DISABLED'}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Collect system information
    system_info = get_system_info()
    print(f"\nSystem Information:")
    for key, value in system_info.items():
        if key != 'gpu' or isinstance(value, dict):
            print(f"  {key}: {value}")
    
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
    
    # Initialize advanced profiling
    advanced_profiler = AdvancedProfilingManager(enable_advanced=ENABLE_ADVANCED_PROFILING)
    
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
        print(f"\n{'#'*80}")
        print(f"Configuration {i+1}/{len(configurations)}")
        print(f"Description: {config_dict['description']}")
        print(f"{'#'*80}")
        
        # Run benchmark
        result = benchmark_configuration(
            config_dict, 
            monitor=monitor, 
            advanced_profiling=advanced_profiler
        )
        
        if result["success"]:
            all_results.append(result)
            
            # Save intermediate results
            with open(RESULTS_PATH, 'w') as f:
                json.dump({
                    'system_info': system_info,
                    'configurations_tested': i + 1,
                    'results': all_results
                }, f, indent=2, default=str)
            print(f"\nIntermediate results saved to {RESULTS_PATH}")
        else:
            print(f"\nConfiguration failed: {result.get('error_message', 'Unknown error')}")
            # Save failed result for debugging
            all_results.append(result)
    
    # Generate summary
    generate_summary(all_results, system_info)
    
    print("\n" + "="*80)
    print("Benchmarking completed!")
    print(f"Results saved to: {RESULTS_PATH}")
    print("="*80)

def generate_summary(results, system_info):
    """Generate a summary report"""
    successful = [r for r in results if r.get("success", False)]
    
    if not successful:
        print("No successful benchmarks to summarize")
        return
    
    summary = {
        'system_info': system_info,
        'total_configurations': len(results),
        'successful_configurations': len(successful),
        'failed_configurations': len(results) - len(successful),
        'timestamp': datetime.now().isoformat(),
        'best_performance': None,
        'worst_performance': None,
        'most_efficient': None,
        'by_resolution': {},
        'by_execution_provider': {},
        'by_batch_size': {},
        'advanced_metrics_summary': {}
    }
    
    # Find best and worst performance
    latencies = []
    energy_efficiencies = []
    
    for result in successful:
        stats = result.get('latency_stats', {})
        config = result['config']
        
        if 'mean_ms' in stats:
            latencies.append((stats['mean_ms'], config['description'], config))
            
            # Calculate energy efficiency (FPS per Joule)
            if 'energy_estimation' in stats and stats['energy_estimation'].get('estimated_total_energy_j', 0) > 0:
                fps = stats.get('throughput_fps', 0)
                energy_j = stats['energy_estimation']['estimated_total_energy_j']
                efficiency = fps / energy_j if energy_j > 0 else 0
                energy_efficiencies.append((efficiency, config['description'], config))
    
    # Performance metrics
    if latencies:
        best = min(latencies, key=lambda x: x[0])
        worst = max(latencies, key=lambda x: x[0])
        
        summary['best_performance'] = {
            'latency_ms': best[0],
            'throughput_fps': 1000 / best[0] if best[0] > 0 else 0,
            'configuration': best[1],
            'config_details': best[2]
        }
        summary['worst_performance'] = {
            'latency_ms': worst[0],
            'throughput_fps': 1000 / worst[0] if worst[0] > 0 else 0,
            'configuration': worst[1],
            'config_details': worst[2]
        }
    
    # Energy efficiency
    if energy_efficiencies:
        most_efficient = max(energy_efficiencies, key=lambda x: x[0])
        summary['most_efficient'] = {
            'efficiency_fps_per_j': most_efficient[0],
            'configuration': most_efficient[1],
            'config_details': most_efficient[2]
        }
    
    # Group by resolution
    for result in successful:
        res = result['config']['resolution']
        if res not in summary['by_resolution']:
            summary['by_resolution'][res] = {
                'latencies': [],
                'throughputs': [],
                'energy_joules': [],
                'configurations': []
            }
        
        stats = result.get('latency_stats', {})
        if 'mean_ms' in stats:
            summary['by_resolution'][res]['latencies'].append(stats['mean_ms'])
            summary['by_resolution'][res]['throughputs'].append(stats.get('throughput_fps', 0))
            
            if 'energy_estimation' in stats:
                summary['by_resolution'][res]['energy_joules'].append(
                    stats['energy_estimation'].get('estimated_total_energy_j', 0)
                )
            
            summary['by_resolution'][res]['configurations'].append(result['config']['description'])
    
    # Calculate averages for resolutions
    for res, data in summary['by_resolution'].items():
        if data['latencies']:
            data['avg_latency_ms'] = np.mean(data['latencies'])
            data['avg_throughput_fps'] = np.mean(data['throughputs'])
            if data['energy_joules']:
                data['avg_energy_j'] = np.mean(data['energy_joules'])
    
    # Group by execution provider
    for result in successful:
        ep = result['config']['execution_provider']
        if ep not in summary['by_execution_provider']:
            summary['by_execution_provider'][ep] = {
                'latencies': [],
                'throughputs': [],
                'configurations': []
            }
        
        stats = result.get('latency_stats', {})
        if 'mean_ms' in stats:
            summary['by_execution_provider'][ep]['latencies'].append(stats['mean_ms'])
            summary['by_execution_provider'][ep]['throughputs'].append(stats.get('throughput_fps', 0))
            summary['by_execution_provider'][ep]['configurations'].append(result['config']['description'])
    
    # Group by batch size
    for result in successful:
        batch = result['config']['batch']
        if batch not in summary['by_batch_size']:
            summary['by_batch_size'][batch] = {
                'latencies': [],
                'throughputs': [],
                'configurations': []
            }
        
        stats = result.get('latency_stats', {})
        if 'mean_ms' in stats:
            summary['by_batch_size'][batch]['latencies'].append(stats['mean_ms'])
            summary['by_batch_size'][batch]['throughputs'].append(stats.get('throughput_fps', 0))
            summary['by_batch_size'][batch]['configurations'].append(result['config']['description'])
    
    # Advanced metrics summary
    if ENABLE_ADVANCED_PROFILING:
        cpu_usages = []
        memory_usages = []
        
        for result in successful:
            if 'advanced_metrics' in result and 'system' in result['advanced_metrics']:
                sys_metrics = result['advanced_metrics']['system']
                if 'cpu_total_percent' in sys_metrics:
                    cpu_usages.append(sys_metrics['cpu_total_percent'])
                if 'memory_rss_avg_mb' in sys_metrics:
                    memory_usages.append(sys_metrics['memory_rss_avg_mb'])
        
        if cpu_usages:
            summary['advanced_metrics_summary']['cpu_usage'] = {
                'avg_percent': np.mean(cpu_usages),
                'max_percent': max(cpu_usages),
                'min_percent': min(cpu_usages)
            }
        
        if memory_usages:
            summary['advanced_metrics_summary']['memory_usage'] = {
                'avg_mb': np.mean(memory_usages),
                'max_mb': max(memory_usages),
                'min_mb': min(memory_usages)
            }
    
    # Save summary
    summary_path = RESULTS_PATH.replace('.json', '_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {summary_path}")
    
    # Print comprehensive summary
    print("\n" + "="*80)
    print("COMPREHENSIVE SUMMARY")
    print("="*80)
    print(f"Successful: {summary['successful_configurations']}/{summary['total_configurations']}")
    
    if summary['best_performance']:
        print(f"\n🏆 BEST PERFORMANCE:")
        print(f"  Latency: {summary['best_performance']['latency_ms']:.1f}ms")
        print(f"  Throughput: {summary['best_performance']['throughput_fps']:.1f} FPS")
        print(f"  Config: {summary['best_performance']['configuration']}")
    
    if summary['most_efficient']:
        print(f"\n💡 MOST ENERGY EFFICIENT:")
        print(f"  Efficiency: {summary['most_efficient']['efficiency_fps_per_j']:.3f} FPS/J")
        print(f"  Config: {summary['most_efficient']['configuration']}")
    
    print(f"\n📊 BY RESOLUTION:")
    for res, data in sorted(summary['by_resolution'].items()):
        if 'avg_latency_ms' in data:
            print(f"  {res}x{res}: {data['avg_latency_ms']:.1f}ms, {data['avg_throughput_fps']:.1f} FPS", end="")
            if 'avg_energy_j' in data:
                print(f", {data['avg_energy_j']:.2f} J")
            else:
                print()
    
    print(f"\n⚡ BY EXECUTION PROVIDER:")
    for ep, data in summary['by_execution_provider'].items():
        if data['latencies']:
            avg_latency = np.mean(data['latencies'])
            avg_throughput = np.mean(data['throughputs'])
            print(f"  {ep}: {avg_latency:.1f}ms, {avg_throughput:.1f} FPS")
    
    if summary['advanced_metrics_summary']:
        print(f"\n🔍 ADVANCED METRICS:")
        if 'cpu_usage' in summary['advanced_metrics_summary']:
            cpu = summary['advanced_metrics_summary']['cpu_usage']
            print(f"  CPU Usage: {cpu['avg_percent']:.1f}% (min: {cpu['min_percent']:.1f}%, max: {cpu['max_percent']:.1f}%)")
        
        if 'memory_usage' in summary['advanced_metrics_summary']:
            mem = summary['advanced_metrics_summary']['memory_usage']
            print(f"  Memory Usage: {mem['avg_mb']:.1f}MB (min: {mem['min_mb']:.1f}MB, max: {mem['max_mb']:.1f}MB)")

if __name__ == "__main__":
    main()