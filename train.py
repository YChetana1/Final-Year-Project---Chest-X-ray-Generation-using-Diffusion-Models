import torch

from models.unet import TimeConditionedUNet


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = TimeConditionedUNet().to(device)

sample_images = torch.randn(
    2,
    1,
    128,
    128
).to(device)

sample_timesteps = torch.tensor(
    [10, 500],
    device=device
)

with torch.no_grad():
    output = model(
        sample_images,
        sample_timesteps
    )

print("Device:", device)
print("Input shape:", sample_images.shape)
print("Timesteps shape:", sample_timesteps.shape)
print("Output shape:", output.shape)