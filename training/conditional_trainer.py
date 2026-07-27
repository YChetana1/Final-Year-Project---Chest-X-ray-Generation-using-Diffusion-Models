
from pathlib import Path
import csv
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from tqdm import tqdm

from models.conditional_unet import ConditionalUNet
from utils.diffusion import Diffusion
from utils.ema import EMA


class ConditionalDDPMTrainer:
    def __init__(
        self,
        dataset_path: str = "dataset/chest_xray/train",
        image_size: int = 128,
        batch_size: int = 4,
        learning_rate: float = 1e-4,
        noise_steps: int = 1000,
        number_of_classes: int = 2,
        ema_decay: float = 0.999,
        device: str | None = None,
    ):
        self.dataset_path = Path(dataset_path)
        self.image_size = image_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.noise_steps = noise_steps
        self.number_of_classes = number_of_classes

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        print("Using device:", self.device)

        self.train_loader = self.create_dataloader()

        self.model = ConditionalUNet(
            image_channels=1,
            base_channels=32,
            time_embedding_dim=256,
            number_of_classes=self.number_of_classes,
        ).to(self.device)

        self.ema = EMA(
            model=self.model,
            decay=ema_decay,
        ).to(self.device)

        self.diffusion = Diffusion(
            noise_steps=self.noise_steps,
            image_size=self.image_size,
            device=self.device,
        )

        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.learning_rate,
        )

        self.loss_function = nn.MSELoss()

        self.best_loss = float("inf")
        self.training_history = []

        self.checkpoint_dir = Path(
            "conditional_checkpoints"
        )
        self.results_dir = Path(
            "conditional_results"
        )

        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.results_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def create_dataloader(self) -> DataLoader:
        transform = transforms.Compose(
            [
                transforms.Grayscale(
                    num_output_channels=1
                ),
                transforms.Resize(
                    (
                        self.image_size,
                        self.image_size,
                    )
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5,),
                    (0.5,),
                ),
            ]
        )

        dataset = datasets.ImageFolder(
            root=self.dataset_path,
            transform=transform,
        )

        print("Classes:", dataset.classes)
        print("Class mapping:", dataset.class_to_idx)
        print("Total images:", len(dataset))

        class_counts = torch.zeros(
            len(dataset.classes),
            dtype=torch.long,
        )

        for _, label in dataset.samples:
            class_counts[label] += 1

        print("Class counts:", class_counts.tolist())

        class_weights = 1.0 / class_counts.float()

        sample_weights = torch.tensor(
            [
                class_weights[label]
                for _, label in dataset.samples
            ],
            dtype=torch.double,
        )

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )

    def train_one_epoch(
        self,
        epoch: int,
    ) -> float:
        self.model.train()

        total_loss = 0.0

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Conditional Epoch {epoch}",
        )

        for images, labels in progress_bar:
            images = images.to(
                self.device,
                non_blocking=True,
            )

            labels = labels.to(
                self.device,
                non_blocking=True,
            )

            timesteps = (
                self.diffusion.sample_timesteps(
                    images.shape[0]
                ).to(self.device)
            )

            noisy_images, actual_noise = (
                self.diffusion.add_noise(
                    images,
                    timesteps,
                )
            )

            predicted_noise = self.model(
                noisy_images,
                timesteps,
                labels,
            )

            loss = self.loss_function(
                predicted_noise,
                actual_noise,
            )

            self.optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=1.0,
            )

            self.optimizer.step()

            self.ema.update(self.model)

            total_loss += loss.item()

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        return total_loss / len(
            self.train_loader
        )

    def save_checkpoint(
        self,
        epoch: int,
        average_loss: float,
    ) -> None:
        checkpoint_path = (
            self.checkpoint_dir
            / f"conditional_epoch_{epoch}.pth"
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": (
                    self.model.state_dict()
                ),
                "ema_model_state_dict": (
                    self.ema.state_dict()
                ),
                "optimizer_state_dict": (
                    self.optimizer.state_dict()
                ),
                "average_loss": average_loss,
                "best_loss": self.best_loss,
                "training_history": (
                    self.training_history
                ),
                "class_names": [
                    "NORMAL",
                    "PNEUMONIA",
                ],
            },
            checkpoint_path,
        )

        print(
            "Checkpoint saved:",
            checkpoint_path,
        )

    def save_best_model(
        self,
        epoch: int,
        average_loss: float,
    ) -> None:
        if average_loss < self.best_loss:
            self.best_loss = average_loss

            best_path = (
                self.checkpoint_dir
                / "conditional_best_model.pth"
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": (
                        self.model.state_dict()
                    ),
                    "ema_model_state_dict": (
                        self.ema.state_dict()
                    ),
                    "optimizer_state_dict": (
                        self.optimizer.state_dict()
                    ),
                    "average_loss": average_loss,
                    "best_loss": self.best_loss,
                    "training_history": (
                        self.training_history
                    ),
                    "class_names": [
                        "NORMAL",
                        "PNEUMONIA",
                    ],
                },
                best_path,
            )

            print(
                "New best conditional model saved:",
                best_path,
            )

    def save_training_history(self) -> None:
        csv_path = (
            self.results_dir
            / "conditional_training_history.csv"
        )

        with csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "epoch",
                    "average_loss",
                    "time_seconds",
                    "learning_rate",
                ],
            )

            writer.writeheader()
            writer.writerows(
                self.training_history
            )

        print(
            "Training history saved:",
            csv_path,
        )

    def save_loss_graph(self) -> None:
        epochs = [
            row["epoch"]
            for row in self.training_history
        ]

        losses = [
            row["average_loss"]
            for row in self.training_history
        ]

        plt.figure(figsize=(8, 5))
        plt.plot(
            epochs,
            losses,
            marker="o",
        )

        plt.xlabel("Epoch")
        plt.ylabel("Average Loss")
        plt.title(
            "Conditional DDPM Training Loss"
        )
        plt.grid(True)
        plt.tight_layout()

        graph_path = (
            self.results_dir
            / "conditional_training_loss.png"
        )

        plt.savefig(
            graph_path,
            dpi=300,
        )

        plt.close()

        print(
            "Loss graph saved:",
            graph_path,
        )

    def train(
        self,
        total_epochs: int,
    ) -> None:
        for epoch in range(
            1,
            total_epochs + 1,
        ):
            start_time = time.time()

            average_loss = (
                self.train_one_epoch(epoch)
            )

            elapsed_time = (
                time.time() - start_time
            )

            print(
                f"Epoch {epoch}/{total_epochs} "
                f"Average Loss: "
                f"{average_loss:.4f}"
            )

            self.training_history.append(
                {
                    "epoch": epoch,
                    "average_loss": average_loss,
                    "time_seconds": round(
                        elapsed_time,
                        2,
                    ),
                    "learning_rate": (
                        self.optimizer
                        .param_groups[0]["lr"]
                    ),
                }
            )

            self.save_best_model(
                epoch,
                average_loss,
            )

            self.save_checkpoint(
                epoch,
                average_loss,
            )

            self.save_training_history()
            self.save_loss_graph()

        print(
            "Conditional DDPM training completed!"
        )
