import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.ensemble import IsolationForest

# Load dataset
df = pd.read_csv(r"C:\Users\cathe\OneDrive\Desktop\Kubecoders\data\simulated_k8s_metrics.csv")

# Convert timestamp to datetime
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Apply rolling mean for smoothing
df["cpu_usage_smooth"] = df["cpu_usage"].rolling(window=10).mean()
df["memory_usage_smooth"] = df["memory_usage"].rolling(window=10).mean()

# Function to plot CPU & Memory Usage
def plot_resource_usage():
    plt.figure(figsize=(12, 5))
    plt.plot(df["timestamp"], df["cpu_usage_smooth"], label="CPU Usage (%)", alpha=0.7)
    plt.plot(df["timestamp"], df["memory_usage_smooth"], label="Memory Usage (%)", alpha=0.7)
    
    # Highlighting anomaly regions (if any)
    plt.axvline(pd.Timestamp("2021-01-10"), color="red", linestyle="--", label="Anomaly Detected")
    plt.fill_between(df["timestamp"], df["cpu_usage_smooth"], df["memory_usage_smooth"], color="gray", alpha=0.1)

    plt.xlabel("Time")
    plt.ylabel("Usage (%)")
    plt.title("CPU & Memory Usage Over Time (Smoothed)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

# Function to plot Node Failures
def plot_node_failures():
    plt.figure(figsize=(12, 3))
    
    # Using scatter plot instead of bar for better visualization
    plt.scatter(df["timestamp"], df["node_failure"], color="red", alpha=0.6)
    
    plt.xlabel("Time")
    plt.ylabel("Failure (0 or 1)")
    plt.title("Node Failures Over Time")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

# Function to plot correlation heatmap
def plot_correlation():
    corr = df[["cpu_usage", "memory_usage", "network_io", "disk_usage", "node_failure"]].corr()
    
    plt.figure(figsize=(8, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Between Metrics")
    plt.show()

# Function to analyze trends using seasonal decomposition
def plot_seasonal_decomposition():
    decomposition = seasonal_decompose(df["cpu_usage_smooth"].dropna(), model="additive", period=50)
    decomposition.plot()
    plt.show()

# Function to detect anomalies using Isolation Forest
def detect_anomalies():
    model = IsolationForest(contamination=0.05, random_state=42)
    df["anomaly"] = model.fit_predict(df[["cpu_usage", "memory_usage"]])

    # Plot anomalies
    plt.figure(figsize=(12, 5))
    plt.scatter(df["timestamp"], df["cpu_usage"], c=df["anomaly"], cmap="coolwarm", alpha=0.7)
    plt.xlabel("Time")
    plt.ylabel("CPU Usage (%)")
    plt.title("Anomaly Detection in CPU Usage")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

# Run all plots
if __name__ == "__main__":
    plot_resource_usage()
    plot_node_failures()
    plot_correlation()
    plot_seasonal_decomposition()
    detect_anomalies()
