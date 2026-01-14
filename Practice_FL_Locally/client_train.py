# Step 2: Run this to understand how to train a client 

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import SimpleCNN

# Image processing
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# training client
def train_client(client_path, client_name, global_weights=None):
    dataset = datasets.ImageFolder(
        root=client_path,
        transform=transform
    )

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SimpleCNN(num_classes=len(dataset.classes)).to(device)

    # 🔥 THIS IS THE IMPORTANT LINE
    if global_weights is not None:
        model.load_state_dict(global_weights)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 1

    for epoch in range(epochs):
        total_loss = 0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"{client_name} | Epoch {epoch+1}, Loss: {total_loss:.4f}")
    print()
    torch.save(model.state_dict(), f"{client_name}_model.pth")
