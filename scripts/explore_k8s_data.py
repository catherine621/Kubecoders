import pandas as pd

# Load the dataset
df = pd.read_csv(r"C:\Users\cathe\OneDrive\Desktop\Kubecoders\data\simulated_k8s_metrics.csv")

# Display first few rows
print(df.head())

# Show basic statistics
print(df.describe())

# Check for missing values
print(df.isnull().sum())
