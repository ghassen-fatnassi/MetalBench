#!/usr/bin/env python3
import json
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ================================================================
# CONFIG
# ================================================================
INPUT_FILE = "benchmark_results_20251204_060447.json"
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

sns.set(style="whitegrid", context="talk")


# ================================================================
# LOAD + NORMALIZE DATA
# ================================================================
print("Loading JSON...")
with open(INPUT_FILE, "r") as f:
    raw = json.load(f)

print("Normalizing...")
rows = []
for entry in raw:
    cfg = entry["config"]
    lat = entry["latency_stats"]
    sys = entry["system_metrics"][0]

    rows.append({
        # config
        "resolution": cfg["resolution"],
        "batch": cfg["batch"],
        "provider": cfg["execution_provider"],
        "optimization": cfg["optimization"],
        "intra": cfg["intra"],
        "inter": cfg["inter"],
        "warmup": cfg["warmup"],
        "description": cfg["description"],

        # system
        "gpu_load": sys["gpu_load_percent"],
        "emc_load": sys["emc_load_percent"],
        "gpu_temp": sys["gpu_temp_c"],
        "cpu_temp": sys["cpu_temp_c"],
        "ram_used": sys["ram_used_mb"],
        "ram_total": sys["ram_total_mb"],

        # latency
        "mean_ms": lat["mean_ms"],
        "min_ms": lat["min_ms"],
        "max_ms": lat["max_ms"],
        "median_ms": lat["median_ms"],
        "std_ms": lat["std_ms"],
        "cv_percent": lat["cv_percent"],
        "throughput_fps": lat["throughput_fps"],
        "n_samples": lat["n_samples"],

        "timestamp": entry["timestamp"],
    })

df = pd.DataFrame(rows)

# ================================================================
# DERIVED METRICS
# ================================================================
df["pixels"] = df["resolution"] ** 2
df["effective_pixels"] = df["pixels"] * df["batch"]
df["ram_util_percent"] = df["ram_used"] / df["ram_total"] * 100
df["latency_per_pixel"] = df["mean_ms"] / df["pixels"]
df["fps_per_ram"] = df["throughput_fps"] / df["ram_used"]
df["fps_per_degree"] = df["throughput_fps"] / df["gpu_temp"]


# ================================================================
# HELPER FUNCTION TO SAVE PLOTS
# ================================================================
def saveplot(filename):
    plt.tight_layout()
    plt.gcf().set_size_inches(14, 7)
    path = os.path.join(PLOT_DIR, filename + ".png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.clf()
    print("Saved:", path)


# ================================================================
# 1. Throughput vs Resolution
# ================================================================
sns.lineplot(data=df, x="resolution", y="throughput_fps",
             hue="batch", style="optimization", marker="o")
plt.title("Throughput vs Resolution")
plt.xlabel("Resolution")
plt.ylabel("Throughput (FPS)")
saveplot("throughput_vs_resolution")

# 2. Latency vs Resolution
sns.lineplot(data=df, x="resolution", y="mean_ms",
             hue="batch", style="optimization", marker="o")
plt.title("Latency vs Resolution")
plt.xlabel("Resolution")
plt.ylabel("Mean Latency (ms)")
saveplot("latency_vs_resolution")

# 3. Throughput vs Batch (per resolution)
resolutions = sorted(df["resolution"].unique())
for res in resolutions:
    sdf = df[df["resolution"] == res]
    sns.lineplot(data=sdf, x="batch", y="throughput_fps",
                 hue="optimization", marker="o")
    plt.title(f"Throughput vs Batch @ res={res}")
    plt.xlabel("Batch Size")
    plt.ylabel("Throughput (FPS)")
    saveplot(f"throughput_res{res}")

# 4. Latency vs Batch (per resolution)
for res in resolutions:
    sdf = df[df["resolution"] == res]
    sns.lineplot(data=sdf, x="batch", y="mean_ms",
                 hue="optimization", marker="o")
    plt.title(f"Latency vs Batch @ res={res}")
    plt.xlabel("Batch Size")
    plt.ylabel("Mean Latency (ms)")
    saveplot(f"latency_res{res}")

# 5. Throughput vs Effective Load
sns.scatterplot(data=df, x="effective_pixels", y="throughput_fps",
                hue="optimization", style="batch", s=120)
plt.xscale("log")
plt.title("Throughput vs Effective Load (pixels × batch)")
plt.xlabel("Effective Pixels (log scale)")
plt.ylabel("Throughput (FPS)")
saveplot("throughput_vs_load")

# 6. Latency vs Effective Load
sns.scatterplot(data=df, x="effective_pixels", y="mean_ms",
                hue="optimization", style="batch", s=120)
plt.xscale("log")
plt.title("Latency vs Effective Load (pixels × batch)")
plt.xlabel("Effective Pixels (log scale)")
plt.ylabel("Mean Latency (ms)")
saveplot("latency_vs_load")

# 7. Optimization summary (FPS and Latency)
sns.barplot(data=df, x="optimization", y="throughput_fps",
            estimator="mean", errorbar="sd")
plt.title("Mean Throughput by Optimization")
plt.xlabel("Optimization Mode")
plt.ylabel("Throughput (FPS)")
saveplot("opt_throughput_summary")

sns.barplot(data=df, x="optimization", y="mean_ms",
            estimator="mean", errorbar="sd")
plt.title("Mean Latency by Optimization")
plt.xlabel("Optimization Mode")
plt.ylabel("Latency (ms)")
saveplot("opt_latency_summary")

# ================================================================
# TinyML / Hardware Metrics
# ================================================================

# 8. GPU Temp vs Resolution
sns.lineplot(data=df, x="resolution", y="gpu_temp",
             hue="batch", style="optimization", marker="o")
plt.title("GPU Temperature vs Resolution")
plt.xlabel("Resolution")
plt.ylabel("GPU Temp (°C)")
saveplot("gpu_temp_vs_resolution")

# 9. GPU Temp vs Effective Load
sns.scatterplot(data=df, x="effective_pixels", y="gpu_temp",
                hue="optimization", style="batch", s=120)
plt.xscale("log")
plt.title("GPU Temp vs Effective Load")
plt.xlabel("Effective Pixels (log scale)")
plt.ylabel("GPU Temp (°C)")
saveplot("gpu_temp_vs_load")

# 10. GPU Load vs Throughput
sns.scatterplot(data=df, x="gpu_load", y="throughput_fps",
                hue="optimization", style="batch", s=120)
plt.title("GPU Load vs Throughput")
plt.xlabel("GPU Load (%)")
plt.ylabel("Throughput (FPS)")
saveplot("gpu_load_vs_throughput")

# 11. EMC Load vs Effective Load
sns.scatterplot(data=df, x="effective_pixels", y="emc_load",
                hue="optimization", style="batch", s=120)
plt.xscale("log")
plt.title("EMC Load vs Effective Load")
plt.xlabel("Effective Pixels (log scale)")
plt.ylabel("EMC Load (%)")
saveplot("emc_load_vs_load")

# 12. RAM Utilization vs Resolution
sns.lineplot(data=df, x="resolution", y="ram_util_percent",
             hue="batch", style="optimization", marker="o")
plt.title("RAM Utilization vs Resolution")
plt.xlabel("Resolution")
plt.ylabel("RAM Utilization (%)")
saveplot("ram_util_vs_resolution")

# 13. RAM Utilization vs Batch
for res in resolutions:
    sdf = df[df["resolution"] == res]
    sns.lineplot(data=sdf, x="batch", y="ram_util_percent",
                 hue="optimization", marker="o")
    plt.title(f"RAM Utilization vs Batch @ res={res}")
    plt.xlabel("Batch Size")
    plt.ylabel("RAM Utilization (%)")
    saveplot(f"ram_util_res{res}")

# 14. CV% vs Resolution
sns.lineplot(data=df, x="resolution", y="cv_percent",
             hue="batch", style="optimization", marker="o")
plt.title("Inference CV% vs Resolution")
plt.xlabel("Resolution")
plt.ylabel("Coefficient of Variation (%)")
saveplot("cv_percent_vs_resolution")

# 15. Latency per Pixel vs Resolution
sns.lineplot(data=df, x="resolution", y="latency_per_pixel",
             hue="batch", style="optimization", marker="o")
plt.title("Latency per Pixel vs Resolution")
plt.xlabel("Resolution")
plt.ylabel("Latency per Pixel (ms/pixel)")
saveplot("latency_per_pixel_vs_resolution")

# 16. FPS per RAM MB vs Resolution
sns.lineplot(data=df, x="resolution", y="fps_per_ram",
             hue="batch", style="optimization", marker="o")
plt.title("Throughput per RAM MB vs Resolution")
plt.xlabel("Resolution")
plt.ylabel("FPS per RAM MB")
saveplot("fps_per_ram_vs_resolution")

# 17. FPS per GPU Temp vs Resolution
sns.lineplot(data=df, x="resolution", y="fps_per_degree",
             hue="batch", style="optimization", marker="o")
plt.title("Throughput per GPU °C vs Resolution")
plt.xlabel("Resolution")
plt.ylabel("FPS per GPU °C")
saveplot("fps_per_degree_vs_resolution")

print("\nAll plots saved in:", PLOT_DIR)
