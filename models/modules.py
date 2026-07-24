import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """
    Converts diffusion timesteps into vector embeddings.
    """

    def __init__(self, embedding_dim: int):
        super().__init__()

        if embedding_dim % 2 != 0:
            raise ValueError("embedding_dim must be even")

        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.embedding_dim // 2

        scale = math.log(10000) / (half_dim - 1)

        frequencies = torch.exp(
            torch.arange(
                half_dim,
                device=timesteps.device,
                dtype=torch.float32
            ) * -scale
        )

        angles = timesteps.float()[:, None] * frequencies[None, :]

        embedding = torch.cat(
            [torch.sin(angles), torch.cos(angles)],
            dim=1
        )

        return embedding


class ResidualBlock(nn.Module):
    """
    Residual convolution block conditioned on timestep embeddings.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embedding_dim: int
    ):
        super().__init__()

        self.norm1 = nn.GroupNorm(
            num_groups=8,
            num_channels=in_channels
        )

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                time_embedding_dim,
                out_channels
            )
        )

        self.norm2 = nn.GroupNorm(
            num_groups=8,
            num_channels=out_channels
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

        if in_channels == out_channels:
            self.residual_connection = nn.Identity()
        else:
            self.residual_connection = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1
            )

    def forward(
        self,
        x: torch.Tensor,
        time_embedding: torch.Tensor
    ) -> torch.Tensor:

        residual = self.residual_connection(x)

        x = self.norm1(x)
        x = F.silu(x)
        x = self.conv1(x)

        time_features = self.time_projection(time_embedding)

        time_features = time_features[:, :, None, None]

        x = x + time_features

        x = self.norm2(x)
        x = F.silu(x)
        x = self.conv2(x)

        return x + residual