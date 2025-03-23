import numpy as np

def process_prometheus_data(data):
    results = data['data']['result']
    cpu_usage = [float(entry['value'][1]) for entry in results]
    
    # Convert to NumPy array
    return np.array(cpu_usage).reshape(1, -1)

cpu_features = process_prometheus_data(data)
print(cpu_features)
