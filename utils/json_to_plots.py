#!/usr/bin/env python3
import json
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

def load_data(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    
    records = []
    for entry in data:
        record = {
            'batch': entry['config']['batch'],
            'exec_provider': entry['config']['execution_provider'].replace('ExecutionProvider', ''),
            'optimization': entry['config']['optimization'],
            'warmup': entry['config']['warmup'],
            'intra': entry['config']['intra'],
            'inter': entry['config']['inter'],
            
            'mean_ms': entry['latency_stats']['mean_ms'],
            'median_ms': entry['latency_stats']['median_ms'],
            'std_ms': entry['latency_stats']['std_ms'],
            'throughput_fps': entry['latency_stats']['throughput_fps'],
            'cv_percent': entry['latency_stats']['cv_percent'],
            'p95': entry['latency_stats']['percentiles_ms']['p95'],
        }
        
        if entry['system_metrics']:
            metrics = entry['system_metrics']
            record['gpu_load'] = np.mean([m['gpu_load_percent'] for m in metrics])
            record['gpu_temp'] = np.mean([m['gpu_temp_c'] for m in metrics])
            record['cpu_temp'] = np.mean([m['cpu_temp_c'] for m in metrics])
            record['ram_used'] = np.mean([m['ram_used_mb'] for m in metrics])
        
        records.append(record)
    
    return pd.DataFrame(records)

def plot_useful_comparisons(df, output_dir='plots'):
    Path(output_dir).mkdir(exist_ok=True)
    
    # Get unique values
    providers = df['exec_provider'].unique()
    optimizations = df['optimization'].unique()
    batches = sorted(df['batch'].unique())
    
    colors = {'CPU': '#3498db', 'CUDA': '#e74c3c'}
    markers = {'CPU': 'o', 'CUDA': 's'}
    
    # 1. CPU vs GPU: Latency as Batch Increases
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    for provider in providers:
        for opt in optimizations:
            subset = df[(df['exec_provider'] == provider) & (df['optimization'] == opt)]
            if len(subset) > 0:
                subset = subset.sort_values('batch')
                label = f'{provider} - {opt}'
                linestyle = '-' if opt == 'all' else '--'
                ax1.plot(subset['batch'], subset['mean_ms'], 
                        marker=markers[provider], label=label, 
                        color=colors[provider], linestyle=linestyle, 
                        linewidth=2.5, markersize=8, alpha=0.8)
    
    ax1.set_xlabel('Batch Size', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Mean Latency (ms)', fontweight='bold', fontsize=12)
    ax1.set_title('CPU vs GPU: Latency vs Batch Size', fontweight='bold', fontsize=14)
    ax1.legend(loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    
    # 2. CPU vs GPU: Throughput as Batch Increases
    for provider in providers:
        for opt in optimizations:
            subset = df[(df['exec_provider'] == provider) & (df['optimization'] == opt)]
            if len(subset) > 0:
                subset = subset.sort_values('batch')
                label = f'{provider} - {opt}'
                linestyle = '-' if opt == 'all' else '--'
                ax2.plot(subset['batch'], subset['throughput_fps'], 
                        marker=markers[provider], label=label,
                        color=colors[provider], linestyle=linestyle,
                        linewidth=2.5, markersize=8, alpha=0.8)
    
    ax2.set_xlabel('Batch Size', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Throughput (FPS)', fontweight='bold', fontsize=12)
    ax2.set_title('CPU vs GPU: Throughput vs Batch Size', fontweight='bold', fontsize=14)
    ax2.legend(loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/01_cpu_vs_gpu_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: CPU vs GPU performance comparison")
    
    # 2. Optimization Level Direct Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    for provider in providers:
        opt_data = []
        opt_labels = []
        for opt in optimizations:
            subset = df[(df['exec_provider'] == provider) & (df['optimization'] == opt)]
            if len(subset) > 0:
                opt_data.append(subset['mean_ms'].mean())
                opt_labels.append(opt)
        
        x = np.arange(len(opt_labels))
        width = 0.35
        offset = -width/2 if provider == 'CPU' else width/2
        ax1.bar(x + offset, opt_data, width, label=provider, color=colors[provider], alpha=0.8)
    
    ax1.set_xlabel('Optimization Level', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Average Latency (ms)', fontweight='bold', fontsize=12)
    ax1.set_title('CPU vs GPU: Optimization Impact on Latency', fontweight='bold', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(opt_labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    for provider in providers:
        opt_data = []
        opt_labels = []
        for opt in optimizations:
            subset = df[(df['exec_provider'] == provider) & (df['optimization'] == opt)]
            if len(subset) > 0:
                opt_data.append(subset['throughput_fps'].mean())
                opt_labels.append(opt)
        
        x = np.arange(len(opt_labels))
        offset = -width/2 if provider == 'CPU' else width/2
        ax2.bar(x + offset, opt_data, width, label=provider, color=colors[provider], alpha=0.8)
    
    ax2.set_xlabel('Optimization Level', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Average Throughput (FPS)', fontweight='bold', fontsize=12)
    ax2.set_title('CPU vs GPU: Optimization Impact on Throughput', fontweight='bold', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(opt_labels)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/02_optimization_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: Optimization comparison")
    
    # 3. Latency Distribution: CPU vs GPU side-by-side
    fig, ax = plt.subplots(figsize=(14, 7))
    
    positions = []
    labels = []
    pos = 0
    
    for batch in batches:
        for provider in providers:
            subset = df[(df['batch'] == batch) & (df['exec_provider'] == provider)]
            if len(subset) > 0:
                row = subset.iloc[0]
                bp = ax.boxplot([[row['mean_ms'] - row['std_ms'], 
                                 row['mean_ms'], 
                                 row['mean_ms'] + row['std_ms']]],
                               positions=[pos], widths=0.6,
                               patch_artist=True,
                               boxprops=dict(facecolor=colors[provider], alpha=0.7),
                               medianprops=dict(color='black', linewidth=2))
                positions.append(pos)
                labels.append(f'{provider}\nB:{batch}')
                pos += 1
        pos += 0.5  # Gap between batches
    
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Latency (ms)', fontweight='bold', fontsize=12)
    ax.set_title('Latency Distribution: CPU vs GPU Across Batch Sizes', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[p], alpha=0.7, label=p) for p in providers]
    ax.legend(handles=legend_elements, loc='best')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/03_latency_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: Latency distribution comparison")
    
    # 4. System Load: GPU Usage during CPU vs GPU execution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    for provider in providers:
        subset = df[df['exec_provider'] == provider].sort_values('batch')
        if len(subset) > 0 and 'gpu_load' in subset.columns:
            ax1.plot(subset['batch'], subset['gpu_load'], 
                    marker=markers[provider], label=provider,
                    color=colors[provider], linewidth=2.5, markersize=8)
    
    ax1.set_xlabel('Batch Size', fontweight='bold', fontsize=12)
    ax1.set_ylabel('GPU Load (%)', fontweight='bold', fontsize=12)
    ax1.set_title('GPU Utilization: CPU vs GPU Execution', fontweight='bold', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    for provider in providers:
        subset = df[df['exec_provider'] == provider].sort_values('batch')
        if len(subset) > 0 and 'gpu_temp' in subset.columns:
            ax2.plot(subset['batch'], subset['gpu_temp'], 
                    marker=markers[provider], label=provider,
                    color=colors[provider], linewidth=2.5, markersize=8)
    
    ax2.set_xlabel('Batch Size', fontweight='bold', fontsize=12)
    ax2.set_ylabel('GPU Temperature (°C)', fontweight='bold', fontsize=12)
    ax2.set_title('GPU Temperature: CPU vs GPU Execution', fontweight='bold', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/04_system_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: System metrics comparison")
    
    # 5. Speedup Factor: How much faster is GPU vs CPU
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for opt in optimizations:
        speedups = []
        batch_vals = []
        
        for batch in batches:
            cpu_subset = df[(df['batch'] == batch) & (df['exec_provider'] == 'CPU') & (df['optimization'] == opt)]
            gpu_subset = df[(df['batch'] == batch) & (df['exec_provider'] == 'CUDA') & (df['optimization'] == opt)]
            
            if len(cpu_subset) > 0 and len(gpu_subset) > 0:
                cpu_time = cpu_subset['mean_ms'].values[0]
                gpu_time = gpu_subset['mean_ms'].values[0]
                speedup = cpu_time / gpu_time
                speedups.append(speedup)
                batch_vals.append(batch)
        
        if speedups:
            linestyle = '-' if opt == 'all' else '--'
            ax.plot(batch_vals, speedups, marker='o', label=f'{opt}',
                   linewidth=2.5, markersize=10, linestyle=linestyle)
    
    ax.axhline(y=1, color='red', linestyle='--', linewidth=2, alpha=0.5, label='No speedup')
    ax.set_xlabel('Batch Size', fontweight='bold', fontsize=12)
    ax.set_ylabel('Speedup Factor (CPU time / GPU time)', fontweight='bold', fontsize=12)
    ax.set_title('GPU Speedup Over CPU', fontweight='bold', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/05_gpu_speedup.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: GPU speedup analysis")
    
    # 6. Stability Comparison (CV %)
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for provider in providers:
        for opt in optimizations:
            subset = df[(df['exec_provider'] == provider) & (df['optimization'] == opt)]
            if len(subset) > 0:
                subset = subset.sort_values('batch')
                label = f'{provider} - {opt}'
                linestyle = '-' if opt == 'all' else '--'
                ax.plot(subset['batch'], subset['cv_percent'], 
                       marker=markers[provider], label=label,
                       color=colors[provider], linestyle=linestyle,
                       linewidth=2.5, markersize=8, alpha=0.8)
    
    ax.axhline(y=5, color='green', linestyle='--', alpha=0.3, label='Good (<5%)')
    ax.axhline(y=10, color='orange', linestyle='--', alpha=0.3, label='Acceptable (<10%)')
    ax.set_xlabel('Batch Size', fontweight='bold', fontsize=12)
    ax.set_ylabel('Coefficient of Variation (%)', fontweight='bold', fontsize=12)
    ax.set_title('Performance Stability: CPU vs GPU (Lower = More Stable)', fontweight='bold', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/06_stability_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: Stability comparison")
    
    # 7. Thread Configuration Impact (if varies)
    if len(df['intra'].unique()) > 1 or len(df['inter'].unique()) > 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        cpu_data = df[df['exec_provider'] == 'CPU']
        if len(cpu_data) > 0:
            thread_config = cpu_data['intra'].astype(str) + '/' + cpu_data['inter'].astype(str)
            cpu_data = cpu_data.copy()
            cpu_data['thread_label'] = thread_config
            
            for label in cpu_data['thread_label'].unique():
                subset = cpu_data[cpu_data['thread_label'] == label].sort_values('batch')
                ax1.plot(subset['batch'], subset['mean_ms'], 
                        marker='o', label=f'Intra/Inter: {label}',
                        linewidth=2.5, markersize=8)
            
            ax1.set_xlabel('Batch Size', fontweight='bold', fontsize=12)
            ax1.set_ylabel('Mean Latency (ms)', fontweight='bold', fontsize=12)
            ax1.set_title('CPU Thread Configuration Impact', fontweight='bold', fontsize=14)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            for label in cpu_data['thread_label'].unique():
                subset = cpu_data[cpu_data['thread_label'] == label].sort_values('batch')
                ax2.plot(subset['batch'], subset['throughput_fps'], 
                        marker='o', label=f'Intra/Inter: {label}',
                        linewidth=2.5, markersize=8)
            
            ax2.set_xlabel('Batch Size', fontweight='bold', fontsize=12)
            ax2.set_ylabel('Throughput (FPS)', fontweight='bold', fontsize=12)
            ax2.set_title('CPU Thread Configuration Impact', fontweight='bold', fontsize=14)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/07_thread_config.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✅ Saved: Thread configuration analysis")

def print_summary(df):
    print("\n" + "="*80)
    print("BENCHMARK INSIGHTS")
    print("="*80)
    
    cpu = df[df['exec_provider'] == 'CPU']
    gpu = df[df['exec_provider'] == 'CUDA']
    
    if len(cpu) > 0 and len(gpu) > 0:
        avg_speedup = (cpu['mean_ms'].mean() / gpu['mean_ms'].mean())
        print(f"\n⚡ Average GPU Speedup: {avg_speedup:.2f}x faster than CPU")
        
        print(f"\n🏆 Best CPU Config:")
        best_cpu = cpu.loc[cpu['throughput_fps'].idxmax()]
        print(f"   Throughput: {best_cpu['throughput_fps']:.2f} FPS")
        print(f"   Latency: {best_cpu['mean_ms']:.2f} ms")
        print(f"   Config: {best_cpu['optimization']}, Batch {best_cpu['batch']}")
        
        print(f"\n🏆 Best GPU Config:")
        best_gpu = gpu.loc[gpu['throughput_fps'].idxmax()]
        print(f"   Throughput: {best_gpu['throughput_fps']:.2f} FPS")
        print(f"   Latency: {best_gpu['mean_ms']:.2f} ms")
        print(f"   Config: {best_gpu['optimization']}, Batch {best_gpu['batch']}")
        
        print(f"\n📊 Performance Gap:")
        print(f"   GPU is {(best_gpu['throughput_fps'] / best_cpu['throughput_fps']):.2f}x faster at best")

def main():
    if len(sys.argv) < 2:
        print("Usage: python benchmark_visualizer.py <benchmark_json>")
        sys.exit(1)
    
    df = load_data(sys.argv[1])
    print(f"📊 Loaded {len(df)} benchmark configurations")
    
    plot_useful_comparisons(df)
    print_summary(df)
    
    print("\n✨ Done! Check the 'plots/' folder")

if __name__ == '__main__':
    main()