# Optional: Run this to, Understand How Centralized Training Works


import torch # to use Pytorch
import torch.nn as nn # neural network modules
import torch.optim as optim # optimization module, which provides algorithms to update model parameters during training
from torchvision import datasets, transforms # PyTorch's computer vision utilities for datasets and image transformations
from torch.utils.data import DataLoader # to load data in batches


# image Cleaning
transform_image = transforms.Compose([  # transform function from PyTorch 
    transforms.Resize((64, 64)),   # make all images same size
    transforms.ToTensor(),         # convert image → numbers |Tensor: multi-dimensional array used for numerical computation, similar to a NumPy array
])


# load dataset
dataset = datasets.ImageFolder( # Load images into PyTorch dataset
    root="images",
    transform=transform_image
)

dataloader = DataLoader(
    dataset,
    batch_size=20,   # number of images processed together in one iteration
    shuffle=True
)

print("Classes:", dataset.classes)


# Neural Network Model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Linear(32 * 16 * 16, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)   # flatten
        return self.classifier(x)

# Neural Network Model End


# Training Setup
device = "cuda" if torch.cuda.is_available() else "cpu"

model = SimpleCNN(num_classes=len(dataset.classes)).to(device)

criterion = nn.CrossEntropyLoss() #confident and wrong → BIG punishment, confident and right → SMALL punishment
optimizer = optim.Adam(model.parameters(), lr=0.001) # model is going wildly → slow down, If learning is smooth → speed up


# Training Loop
epochs = 5

for epoch in range(epochs):
    total_loss = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
print("Training complete.")

torch.save(model.state_dict(), "centralized_model.pth")
