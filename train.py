from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.simple_unet import SimpleUNet

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# Dataset path
dataset_path = Path("dataset/chest_xray/train")

# Preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Dataset
train_dataset = datasets.ImageFolder(
    root=dataset_path,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)

# Model
model = SimpleUNet().to(device)

# Loss function
criterion = nn.MSELoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
epochs = 3

for epoch in range(epochs):

    running_loss = 0

    for images, _ in train_loader:

        images = images.to(device)

        noise = torch.randn_like(images)

        noisy_images = images + 0.5 * noise
        noisy_images = torch.clamp(noisy_images, -1, 1)

        predicted_noise = model(noisy_images)

        loss = criterion(predicted_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} Loss: {running_loss/len(train_loader):.4f}")

print("Training Completed!")

# Save the trained model
torch.save(model.state_dict(), "checkpoints/unet_model.pth")

print("Model saved successfully!")