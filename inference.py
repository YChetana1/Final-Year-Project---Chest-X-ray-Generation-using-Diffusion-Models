from pathlib import Path

import torch
from torchvision.utils import save_image

from models.unet import TimeConditionedUNet
from utils.diffusion import Diffusion


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    model = TimeConditionedUNet(
        image_channels=1,
        base_channels=32,
        time_embedding_dim=256
    ).to(device)

    checkpoint_path = Path(
        "checkpoints/ddpm_epoch_1.pth"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print("Checkpoint loaded successfully!")

    diffusion = Diffusion(
        noise_steps=1000,
        image_size=128,
        device=device
    )

    print("Generating image...")

    generated_images = diffusion.sample(
        model=model,
        number_of_images=1,
        image_channels=1
    )

    output_folder = Path("outputs")
    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = output_folder / "generated_xray.png"

    save_image(
        generated_images,
        output_path
    )

    print("Generated image saved at:", output_path)


if __name__ == "__main__":
    main()