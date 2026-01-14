# Step 3: Run this to understand how server averages client models weights in Federated Learning

import torch 
from model import SimpleCNN
import copy # to deep copy model weights

# Load Client Models Weights
def load_client_weights(client_files):
    client_weights = []
    client_sizes = []

    for file, data_size in client_files:
        weights = torch.load(file)
        client_weights.append(weights)
        client_sizes.append(data_size)

    return client_weights, client_sizes


# Federated Averaging - FedAvg
def fed_avg(client_weights, client_sizes):
    global_weights = copy.deepcopy(client_weights[0])
    total_size = sum(client_sizes)

    for key in global_weights.keys():
        global_weights[key] = sum(
            client_weights[i][key] * (client_sizes[i] / total_size)
            for i in range(len(client_weights))
        )

    return global_weights
