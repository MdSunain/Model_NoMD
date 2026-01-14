# CLIENT TRAINING
# Step 2: Run this to understand how to train a client 

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import SimpleCNN

# Image processing
transform = transforms.Compose([
    transforms.Resize((64, 64)), # resize images to 64x64
    transforms.ToTensor(), # convert image to vector
])

# training client
def train_client(client_path, client_name, global_weights=None):


    # Load client dataset
    dataset = datasets.ImageFolder(
        root=client_path, # path to client data
        transform=transform # image transformations
    )

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True) #PyTorch class to determine how many photos are fed to the model at the same time in memory + shuffled

    device = "cuda" if torch.cuda.is_available() else "cpu" # check for GPU if Available

    model = SimpleCNN(num_classes=len(dataset.classes)).to(device) # create model instance

    # 🔥 THIS IS THE IMPORTANT LINE
    if global_weights is not None:
        model.load_state_dict(global_weights) # load global weights into client model


    # we define loss function and optimizer in this training process
    criterion = nn.CrossEntropyLoss() # loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001) # optimizing algorithm ADAM

    # Training loops
    epochs = 4

    # looping
    for epoch in range(epochs):
        total_loss = 0 

        # iterate over dataloader batches
        for images, labels in dataloader:
            
            optimizer.zero_grad() #This crucial step clears the the memory (gradient) of previous calculations from the previous batch's processing. If you don't do this, gradients accumulate across batches, leading to incorrect model updates.
            outputs = model(images) # predicting next batch
            loss = criterion(outputs, labels) # calculating loss
            loss.backward()  # Actual Algorithm:To compute the gradient of the loss with respect to each parameter (weight and bias) in model.
            optimizer.step() # updating weights based on calculated gradients

            total_loss += loss.item() # calculating total loss for the epoch

        print(f"{client_name} | Epoch {epoch+1}, Loss: {total_loss:.4f}")
    print()
    # Save the trained model weights for the client
    torch.save(model.state_dict(), f"{client_name}_model.pth")
