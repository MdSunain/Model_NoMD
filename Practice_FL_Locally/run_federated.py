# Optional: Run this to, understand how one Federated Round is executed

import torch
from server import load_client_weights, fed_avg
from model import SimpleCNN
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])


def get_data_size(client_path):
    dataset = datasets.ImageFolder(client_path, transform=transform)
    return len(dataset)

client_info = [
    ("client1_model.pth", get_data_size("federated_data/client1")),
    ("client2_model.pth", get_data_size("federated_data/client2")),
    ("client3_model.pth", get_data_size("federated_data/client3")),
]

client_weights, client_sizes = load_client_weights(client_info)

global_weights = fed_avg(client_weights, client_sizes)

torch.save(global_weights, "global_model_round1.pth")

print("Federated Round 1 completed.")
