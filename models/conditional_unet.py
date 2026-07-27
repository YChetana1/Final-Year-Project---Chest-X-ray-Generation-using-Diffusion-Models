
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules import ResidualBlock, SinusoidalTimeEmbedding


class AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.norm = nn.GroupNorm(8, channels)

        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=4,
            batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape

        residual = x

        x = self.norm(x)

        x = x.view(batch_size, channels, height * width)
        x = x.permute(0, 2, 1)

        attention_output, _ = self.attention(x, x, x)

        attention_output = attention_output.permute(0, 2, 1)
        attention_output = attention_output.view(
            batch_size,
            channels,
            height,
            width
        )

        return attention_output + residual


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embedding_dim: int,
        use_attention: bool = False
    ):
        super().__init__()

        self.residual_block = ResidualBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            time_embedding_dim=time_embedding_dim
        )

        self.attention = (
            AttentionBlock(out_channels)
            if use_attention
            else nn.Identity()
        )

        self.downsample = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1
        )

    def forward(
        self,
        x: torch.Tensor,
        condition_embedding: torch.Tensor
    ):
        x = self.residual_block(
            x,
            condition_embedding
        )

        x = self.attention(x)

        skip_connection = x
        x = self.downsample(x)

        return x, skip_connection


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        time_embedding_dim: int,
        use_attention: bool = False
    ):
        super().__init__()

        self.upsample = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.residual_block = ResidualBlock(
            in_channels=out_channels + skip_channels,
            out_channels=out_channels,
            time_embedding_dim=time_embedding_dim
        )

        self.attention = (
            AttentionBlock(out_channels)
            if use_attention
            else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        skip_connection: torch.Tensor,
        condition_embedding: torch.Tensor
    ) -> torch.Tensor:

        x = self.upsample(x)

        if x.shape[-2:] != skip_connection.shape[-2:]:
            x = F.interpolate(
                x,
                size=skip_connection.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        x = torch.cat(
            [x, skip_connection],
            dim=1
        )

        x = self.residual_block(
            x,
            condition_embedding
        )

        x = self.attention(x)

        return x


class ConditionalUNet(nn.Module):
    def __init__(
        self,
        image_channels: int = 1,
        base_channels: int = 32,
        time_embedding_dim: int = 256,
        number_of_classes: int = 2
    ):
        super().__init__()

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(
                time_embedding_dim
            ),
            nn.Linear(
                time_embedding_dim,
                time_embedding_dim
            ),
            nn.SiLU(),
            nn.Linear(
                time_embedding_dim,
                time_embedding_dim
            )
        )

        self.class_embedding = nn.Embedding(
            number_of_classes,
            time_embedding_dim
        )

        self.condition_projection = nn.Sequential(
            nn.Linear(
                time_embedding_dim,
                time_embedding_dim
            ),
            nn.SiLU(),
            nn.Linear(
                time_embedding_dim,
                time_embedding_dim
            )
        )

        self.input_layer = nn.Conv2d(
            image_channels,
            base_channels,
            kernel_size=3,
            padding=1
        )

        self.down1 = DownBlock(
            in_channels=base_channels,
            out_channels=base_channels,
            time_embedding_dim=time_embedding_dim
        )

        self.down2 = DownBlock(
            in_channels=base_channels,
            out_channels=base_channels * 2,
            time_embedding_dim=time_embedding_dim,
            use_attention=True
        )

        self.down3 = DownBlock(
            in_channels=base_channels * 2,
            out_channels=base_channels * 4,
            time_embedding_dim=time_embedding_dim,
            use_attention=True
        )

        self.middle1 = ResidualBlock(
            in_channels=base_channels * 4,
            out_channels=base_channels * 8,
            time_embedding_dim=time_embedding_dim
        )

        self.middle_attention = AttentionBlock(
            base_channels * 8
        )

        self.middle2 = ResidualBlock(
            in_channels=base_channels * 8,
            out_channels=base_channels * 4,
            time_embedding_dim=time_embedding_dim
        )

        self.up1 = UpBlock(
            in_channels=base_channels * 4,
            skip_channels=base_channels * 4,
            out_channels=base_channels * 4,
            time_embedding_dim=time_embedding_dim,
            use_attention=True
        )

        self.up2 = UpBlock(
            in_channels=base_channels * 4,
            skip_channels=base_channels * 2,
            out_channels=base_channels * 2,
            time_embedding_dim=time_embedding_dim,
            use_attention=True
        )

        self.up3 = UpBlock(
            in_channels=base_channels * 2,
            skip_channels=base_channels,
            out_channels=base_channels,
            time_embedding_dim=time_embedding_dim
        )

        self.output_layer = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(
                base_channels,
                image_channels,
                kernel_size=3,
                padding=1
            )
        )

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        class_labels: torch.Tensor
    ) -> torch.Tensor:

        time_embedding = self.time_embedding(
            timesteps
        )

        class_embedding = self.class_embedding(
            class_labels
        )

        condition_embedding = (
            time_embedding + class_embedding
        )

        condition_embedding = (
            self.condition_projection(
                condition_embedding
            )
        )

        x = self.input_layer(x)

        x, skip1 = self.down1(
            x,
            condition_embedding
        )

        x, skip2 = self.down2(
            x,
            condition_embedding
        )

        x, skip3 = self.down3(
            x,
            condition_embedding
        )

        x = self.middle1(
            x,
            condition_embedding
        )

        x = self.middle_attention(x)

        x = self.middle2(
            x,
            condition_embedding
        )

        x = self.up1(
            x,
            skip3,
            condition_embedding
        )

        x = self.up2(
            x,
            skip2,
            condition_embedding
        )

        x = self.up3(
            x,
            skip1,
            condition_embedding
        )

        return self.output_layer(x)
