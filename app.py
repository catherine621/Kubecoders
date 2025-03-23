from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open("model/k8s_issue_model.pkl", "rb"))

@app.route("/", methods=["GET"])
def home():
    return "Kubernetes Failure Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Ensure JSON data is present
        if not request.json or "features" not in request.json:
            return jsonify({"error": "Missing 'features' in request body"}), 400
        
        data = request.json["features"]
        
        # Ensure input is a list
        if not isinstance(data, list):
            return jsonify({"error": "Features should be a list of numbers"}), 400
        
        # Debugging prints
        print("Received input:", data)
        print("Expected input shape:", model.n_features_in_)

        # Convert to numpy array
        data_array = np.array(data).reshape(1, -1)

        # Validate input shape
        if data_array.shape[1] != model.n_features_in_:
            return jsonify({"error": f"Expected {model.n_features_in_} features, but got {data_array.shape[1]}"}), 400
        
        # Make prediction
        prediction = model.predict(data_array)[0]
        return jsonify({"failure": bool(prediction)})

    except Exception as e:
        print("Error:", str(e))  # Log error for debugging
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
