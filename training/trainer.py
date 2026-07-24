from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from models.unet import TimeConditionedUNet
from utils.diffusion import Diffusion


class DDPMTrainer:
    def __init__(
        self,
        dataset_path: str = "dataset/chest_xray/train",
        image_size: int = 128,
        batch_size: int = 4,
        learning_rate: float = 1e-4,
        noise_steps: int = 1000,
        device: str | None = None
    ):
        self.dataset_path = Path(dataset_path)
        self.image_size = image_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.noise_steps = noise_steps

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        print("Using device:", self.device)

        self.train_loader = self.create_dataloader()

        self.model = TimeConditionedUNet(
            image_channels=1,
            base_channels=32,
            time_embedding_dim=256
        ).to(self.device)

        self.diffusion = Diffusion(
            noise_steps=self.noise_steps,
            image_size=self.image_size,
            device=self.device
        )

        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.learning_rate
        )

        self.loss_function = nn.MSELoss()

    def create_dataloader(self) -> DataLoader:

        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(
                (self.image_size, self.image_size)
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        dataset = datasets.ImageFolder(
            root=self.dataset_path,
            transform=transform
        )

        print("Classes:", dataset.classes)
        print("Full dataset images:", len(dataset))

        # Quick test with only 100 images
        test_size = min(500, len(dataset))
        dataset = Subset(dataset, range(test_size))

        print("Quick test images:", len(dataset))

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )

        return loader

    def train_one_epoch(self, epoch: int) -> float:

        self.model.train()

        total_loss = 0.0

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}"
        )

        for images, _ in progress_bar:

            images = images.to(self.device)

            timesteps = self.diffusion.sample_timesteps(
                images.shape[0]
            )

            noisy_images, actual_noise = self.diffusion.add_noise(
                images,
                timesteps
            )

            predicted_noise = self.model(
                noisy_images,
                timesteps
            )

            loss = self.loss_function(
                predicted_noise,
                actual_noise
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        average_loss = total_loss / len(self.train_loader)

        return average_loss

    def save_checkpoint(self, epoch: int):

        checkpoint_folder = Path("checkpoints")
        checkpoint_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        checkpoint_path = checkpoint_folder / f"ddpm_epoch_{epoch}.pth"

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict()
            },
            checkpoint_path
        )

        print(f"Checkpoint saved: {checkpoint_path}")

    def train(self, epochs: int = 1):

        for epoch in range(1, epochs + 1):

            average_loss = self.train_one_epoch(epoch)

            print(
                f"Epoch {epoch}/{epochs} "
                f"Average Loss: {average_loss:.4f}"
            )

            self.save_checkpoint(epoch)

        print("DDPM training completed!")