import json
import pandas as pd
import matplotlib.pyplot as plt
import os

# ===================== CONFIG =====================
profile_file = "your_onnx_profile.json"  # Replace with your ONNX profile JSON
output_dir = "profile_analysis"
os.makedirs(output_dir, exist_ok=True)

# ===================== LOAD PROFILE =====================
with open(profile_file, 'r') as f:
    profile = json.load(f)

events = profile.get("events", [])

# Extract relevant fields
rows = []
for e in events:
    if 'name' in e and 'dur' in e:
        rows.append({
            "name": e["name"],
            "start_us": e["ts"],
            "duration_us": e["dur"],
            "provider": e.get("args", {}).get("provider", "unknown")
        })
df = pd.DataFrame(rows)
df["end_us"] = df["start_us"] + df["duration_us"]

total_time_us = df["duration_us"].sum()

# ===================== 1. Operator-level execution breakdown =====================
op_summary = df.groupby("name")["duration_us"].sum().sort_values(ascending=False)
op_summary_ms = op_summary / 1000
op_summary_ms.to_csv(os.path.join(output_dir, "operator_execution_breakdown.csv"))

plt.figure(figsize=(10,6))
op_summary_ms.head(15).plot(kind="barh", color="skyblue")
plt.xlabel("Total execution time (ms)")
plt.title("Top 15 Operators by Total Execution Time")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "operator_execution_breakdown.png"))
plt.close()

# ===================== 3. Sequential vs Asynchronous Execution =====================
df_sorted = df.sort_values("start_us")
seq_time_us = 0
async_time_us = 0
current_end = 0

for _, row in df_sorted.iterrows():
    if row["start_us"] >= current_end:
        seq_time_us += row["duration_us"]
        current_end = row["end_us"]
    else:
        async_time_us += row["duration_us"]
        current_end = max(current_end, row["end_us"])

seq_async_df = pd.DataFrame({
    "type": ["Sequential", "Async"],
    "duration_us": [seq_time_us, async_time_us]
})
seq_async_df["duration_ms"] = seq_async_df["duration_us"] / 1000
seq_async_df.to_csv(os.path.join(output_dir, "sequential_vs_async.csv"), index=False)

plt.figure(figsize=(6,6))
plt.pie(seq_async_df["duration_ms"], labels=seq_async_df["type"], autopct="%1.1f%%", colors=["orange","green"])
plt.title("Sequential vs Asynchronous Execution Time")
plt.savefig(os.path.join(output_dir, "sequential_vs_async.png"))
plt.close()

# ===================== 4. Kernel Execution Dominance =====================
kernel_df = df.groupby("name")["duration_us"].sum().sort_values(ascending=False).reset_index()
kernel_df = kernel_df.rename(columns={"duration_us":"total_duration_us"})
kernel_df["total_duration_ms"] = kernel_df["total_duration_us"]/1000
kernel_df.to_csv(os.path.join(output_dir, "kernel_execution_dominance.csv"), index=False)

plt.figure(figsize=(10,6))
kernel_df.head(15).plot(x="name", y="total_duration_ms", kind="bar", legend=False, color="salmon")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Total Duration (ms)")
plt.title("Top 15 Kernels by Execution Time")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "kernel_execution_dominance.png"))
plt.close()

# ===================== 5. Overhead Analysis =====================
overhead_df = df[df["provider"] != "CUDAExecutionProvider"]
overhead_time_us = overhead_df["duration_us"].sum()
overhead_percent = overhead_time_us / total_time_us * 100

overhead_summary_df = pd.DataFrame({
    "type": ["CUDA Execution", "Overhead (Non-CUDA)"],
    "duration_us": [total_time_us - overhead_time_us, overhead_time_us],
    "duration_ms": [(total_time_us - overhead_time_us)/1000, overhead_time_us/1000],
    "percent": [100-overhead_percent, overhead_percent]
})
overhead_summary_df.to_csv(os.path.join(output_dir, "overhead_analysis.csv"), index=False)

plt.figure(figsize=(6,6))
plt.pie(overhead_summary_df["duration_ms"], labels=overhead_summary_df["type"], autopct="%1.1f%%", colors=["green","gray"])
plt.title("CUDA vs Overhead Time")
plt.savefig(os.path.join(output_dir, "overhead_analysis.png"))
plt.close()

# ===================== 8. Layer-wise Profiling Summary =====================
layer_summary_df = df.groupby(["name", "provider"]).agg(
    n_calls=("duration_us","count"),
    total_duration_us=("duration_us","sum"),
    avg_duration_us=("duration_us","mean"),
    min_duration_us=("duration_us","min"),
    max_duration_us=("duration_us","max")
).reset_index()
layer_summary_df["total_duration_ms"] = layer_summary_df["total_duration_us"]/1000
layer_summary_df["avg_duration_ms"] = layer_summary_df["avg_duration_us"]/1000
layer_summary_df.to_csv(os.path.join(output_dir, "layer_wise_summary.csv"), index=False)

print(f"Analysis complete. Results saved in '{output_dir}'")
