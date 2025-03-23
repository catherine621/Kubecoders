import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score
from imblearn.over_sampling import SMOTE

# 1️⃣ Load Dataset
file_path = r"C:\Users\cathe\OneDrive\Desktop\Kubecoders\data\simulated_k8s_metrics.csv"
  # Ensure this file exists
df = pd.read_csv(file_path)

# Debugging Step: Check CSV Structure
print("\n🔍 Checking CSV Structure...")
print(df.head())
print("\nColumns in CSV:", df.columns.tolist())

# 2️⃣ Read with correct column names based on actual CSV structure
df.columns = ["timestamp", "cpu_usage", "memory_usage", "network_io", "pod_status", "node_failure", "disk_usage"]

# 3️⃣ Convert 'timestamp' to datetime
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.dropna(subset=["timestamp"])

# 4️⃣ Extract time-based features
df["hour"] = df["timestamp"].dt.hour
df["day"] = df["timestamp"].dt.day
df["weekday"] = df["timestamp"].dt.weekday

# 5️⃣ Add rolling averages for time-series analysis (5-minute window)
df["cpu_usage_avg"] = df["cpu_usage"].rolling(window=5, min_periods=1).mean()
df["memory_usage_avg"] = df["memory_usage"].rolling(window=5, min_periods=1).mean()
df["network_io_avg"] = df["network_io"].rolling(window=5, min_periods=1).mean()
df["disk_usage_avg"] = df["disk_usage"].rolling(window=5, min_periods=1).mean()

df = df.drop(columns=["timestamp"])  # Drop original timestamp column

# 6️⃣ Convert node_failure into categorical labels
failure_threshold = df["node_failure"].median()
df["node_failure"] = df["node_failure"].apply(lambda x: 1 if x > failure_threshold else 0)

print("\n✅ Unique classes in node_failure:", df["node_failure"].unique())

# 7️⃣ Normalize Features
scaler = MinMaxScaler()
X_features = df.drop(columns=["node_failure"])
X_scaled = scaler.fit_transform(X_features)

# 8️⃣ Define Features & Target
X = X_scaled
y = df["node_failure"]

# 9️⃣ Check Class Distribution Before Balancing
print("\n📊 Class Distribution Before Balancing:")
print(y.value_counts())

# 🔟 Apply SMOTE for Balancing
print("\n🔄 Applying SMOTE to balance classes...")
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)
print("\n✅ SMOTE applied successfully!")

# 1️⃣1️⃣ Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 1️⃣2️⃣ Handle Class Imbalance with Weighted Loss
pos_weight = sum(y == 0) / sum(y == 1)  # Calculate imbalance ratio

# 1️⃣3️⃣ Hyperparameter Tuning for XGBoost
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 6, 10],
    'scale_pos_weight': [pos_weight]  # Handling imbalance
}

grid_search = GridSearchCV(XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss"), 
                           param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search.fit(X_train, y_train)

model = grid_search.best_estimator_
print("\n✅ Best Model Parameters:", grid_search.best_params_)

# 1️⃣4️⃣ Train Model with Best Parameters
model.fit(X_train, y_train)

# 1️⃣5️⃣ Evaluate Model
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability scores for ROC-AUC

# 🔥 Performance Metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("\n📊 Model Performance:")
print(f"✅ Accuracy: {accuracy:.4f}")
print(f"✅ F1-score: {f1:.4f}")
print(f"✅ ROC-AUC Score: {roc_auc:.4f}")
print(classification_report(y_test, y_pred))

# 1️⃣6️⃣ Feature Importance Visualization
feature_importance = model.feature_importances_
features = X_features.columns
indices = np.argsort(feature_importance)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(len(features)), feature_importance[indices])
plt.xticks(range(len(features)), np.array(features)[indices], rotation=45)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Feature Importance")
plt.show()

# 1️⃣7️⃣ Save Model & Scaler
joblib.dump(model, "k8s_issue_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("\n✅ Model saved as k8s_issue_model.pkl")
print("✅ Scaler saved as scaler.pkl")
