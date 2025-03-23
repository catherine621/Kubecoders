import requests
import numpy as np
import joblib
import logging
from flask import Flask, jsonify

# Initialize Flask application
app = Flask(__name__)

# Load trained model and scaler (ensure these files exist)
try:
    model = joblib.load("../model/k8s_issue_model.pkl")
    scaler = joblib.load("../model/scaler.pkl")
    logging.info("Model and Scaler loaded successfully")
except Exception as e:
    logging.error(f"Error loading model or scaler: {e}")
    model = None
    scaler = None

# Prometheus API URL
PROMETHEUS_URL = "http://localhost:9090/api/v1/query"

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Function to fetch Prometheus metrics
def fetch_prometheus_data():
    queries = {
        "cpu": 'windows_cpu_time_total',
        
        
    }

    metrics = {}

    for metric, query in queries.items():
        try:
            response = requests.get(PROMETHEUS_URL, params={'query': query})
            response.raise_for_status()
            data = response.json()

            # ✅ Debugging logs to check if Prometheus is returning correct values
            print(f"Query: {query}")
            print(f"Response from Prometheus: {data}")

            if data.get("status") == "success" and "data" in data and "result" in data["data"]:
                values = [float(entry['value'][1]) for entry in data["data"]["result"] if "value" in entry] or [0]
                metrics[metric] = values[:5]  # Take first 5 data points
            else:
                logging.warning(f"No data returned for {metric}")
                metrics[metric] = [0]  # Default to 0 if no data

        except Exception as e:
            logging.error(f"Error fetching {metric} data: {e}")
            metrics[metric] = [0]  # Default to 0 if error occurs

    print(f"Final fetched metrics: {metrics}")  # ✅ Print final data
    return metrics
# Home route for testing API availability
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Kubernetes Failure Prediction API is running!"})

# Prediction route
@app.route('/predict', methods=['GET'])
def predict():
    if not model or not scaler:
        return jsonify({"error": "Model or Scaler not loaded"}), 500

    try:
        # Fetch Prometheus data
        metrics = fetch_prometheus_data()
        
        logging.debug(f"Fetched data: {metrics}")

        # Validate if we have at least 2 values for each metric
        if all(len(metrics[key]) >= 2 for key in metrics):
            # Combine first 2 values from each metric into a single array (8 features)
            input_data = np.array([metrics["cpu"][:2] + metrics["memory"][:2] +
                                   metrics["disk"][:2] + metrics["network"][:2]])

            logging.debug(f"Data before scaling: {input_data}")

            # Scale input data
            scaled_data = scaler.transform(input_data)
            logging.debug(f"Scaled data: {scaled_data}")

            # Make prediction
            prediction = model.predict(scaled_data)[0]
            result = "Yes" if prediction else "No"
            
            logging.debug(f"Prediction result: {result}")

            return jsonify({"Predicted Failure": result})
        
        else:
            return jsonify({"error": "Insufficient valid Prometheus data"}), 500

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({"error": f"Error during prediction processing: {e}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
