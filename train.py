import torch

from models.modules import (
    ResidualBlock,
    SinusoidalTimeEmbedding
)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

time_embedding_dim = 256


time_embedding_layer = SinusoidalTimeEmbedding(
    embedding_dim=time_embedding_dim
).to(device)


residual_block = ResidualBlock(
    in_channels=32,
    out_channels=64,
    time_embedding_dim=time_embedding_dim
).to(device)


sample_images = torch.randn(
    4,
    32,
    64,
    64
).to(device)


sample_timesteps = torch.tensor(
    [10, 100, 500, 900],
    device=device
)


time_embeddings = time_embedding_layer(
    sample_timesteps
)


output = residual_block(
    sample_images,
    time_embeddings
)


print("Device:", device)
print("Input image shape:", sample_images.shape)
print("Time embedding shape:", time_embeddings.shape)
print("Output image shape:", output.shape)