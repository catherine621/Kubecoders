import requests

PROMETHEUS_URL = "http://localhost:9090/api/v1/query"

def query_prometheus(query):
    response = requests.get(PROMETHEUS_URL, params={"query": query})
    return response.json()

if __name__ == "__main__":
    query = "node_cpu_seconds_total"
    data = query_prometheus(query)
    print(data)
