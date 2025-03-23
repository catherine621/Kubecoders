import requests

def fetch_metrics():
    response = requests.get("http://localhost:9090/api/v1/targets")
    print(response.json())

if __name__ == "__main__":
    fetch_metrics()
