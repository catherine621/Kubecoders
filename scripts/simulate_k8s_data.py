import numpy as np
import pandas as pd
import random

# Set seed for reproducibility
np.random.seed(42)

# Number of records
num_samples = 1000

# Time simulation (1-minute intervals)
timestamps = pd.date_range(start="2025-01-01", periods=num_samples, freq="T")

# Simulate CPU Usage (%)
cpu_usage = np.random.normal(loc=50, scale=15, size=num_samples).clip(0, 100)

# Simulate Memory Usage (%)
memory_usage = np.random.normal(loc=60, scale=10, size=num_samples).clip(0, 100)

# Simulate Network I/O (MB/s)
network_io = np.random.normal(loc=200, scale=50, size=num_samples).clip(10, 500)

# Simulate Pod Status (0 = Running, 1 = CrashLoopBackOff, 2 = Pending)
pod_status = np.random.choice([0, 1, 2], size=num_samples, p=[0.85, 0.10, 0.05])

# Simulate Node Failures (0 = Healthy, 1 = Failed)
node_failures = np.random.choice([0, 1], size=num_samples, p=[0.97, 0.03])

# Simulate Disk Usage (%)
disk_usage = np.random.normal(loc=70, scale=10, size=num_samples).clip(0, 100)

# Create a DataFrame
df = pd.DataFrame({
    "timestamp": timestamps,
    "cpu_usage": cpu_usage,
    "memory_usage": memory_usage,
    "network_io": network_io,
    "pod_status": pod_status,
    "node_failure": node_failures,
    "disk_usage": disk_usage
})

# Introduce anomalies (simulate failures)
for _ in range(20):  # Introduce 20 failure points randomly
    index = random.randint(0, num_samples - 1)
    df.loc[index, ["cpu_usage", "memory_usage", "network_io"]] *= np.random.uniform(1.5, 2.5)  # Spike usage
    df.loc[index, "pod_status"] = 1  # CrashLoopBackOff
    df.loc[index, "node_failure"] = 1  # Node Failure

# Save to CSV
df.to_csv("simulated_k8s_metrics.csv", index=False)

print("✅ Simulated Kubernetes data saved as 'simulated_k8s_metrics.csv'")
