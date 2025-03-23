This project is a Flask-based API designed to predict Kubernetes node failures using metrics fetched from Prometheus. The API processes CPU, memory, disk, and network metrics, scales the data, and uses a pre-trained machine learning model to predict potential failures.
Real-time Metrics Fetching: Fetches CPU, memory, disk, and network metrics from Prometheus.

Failure Prediction: Uses a pre-trained machine learning model to predict Kubernetes node failures.

Scalable and Lightweight: Built with Flask, making it easy to deploy and scale.

Error Handling: Robust error handling to ensure the API remains functional even if data fetching or processing fails.
