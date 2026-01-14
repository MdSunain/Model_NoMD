# FEDERATED FINAL 
# Step 4: Run this to perform Federated Learning across multiple clients

import torch
from client_train import train_client
from server import fed_avg
from model import SimpleCNN
from torchvision import datasets, transforms

# Image Cleaning Process
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

def get_data_size(path):
    return len(datasets.ImageFolder(path, transform=transform))

# Clients to work with. |  {name, path}
clients = {
    "client1": "federated_data/client1",
    "client2": "federated_data/client2",
    "client3": "federated_data/client3",
} 

# Model Initialized
global_model = SimpleCNN(num_classes=9)

# Model Weights Initialization
global_weights = global_model.state_dict()  # state_dict holds model parameters

# number of federated rounds
rounds = 5

for r in range(rounds):# for all rounds
    print(f"\n\t--- Federated Round {r+1} ---")

    client_updates = []# to store all client weights
    client_sizes = []  # to store all client data sizes

    for name, path in clients.items():# for all clients
        #trains for each path
        train_client(path, name, global_weights) 
        # weights after training
        weights = torch.load(f"{name}_model.pth")
        # collect weights and data size
        client_updates.append(weights)
        client_sizes.append(get_data_size(path))

    # average the weights for all clients
    global_weights = fed_avg(client_updates, client_sizes)

# Save the final global model after updating all rounds
torch.save(global_weights, "global_model_final.pth")
print("\n\tFederated Training Completed.")

