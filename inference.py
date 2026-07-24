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
        "checkpoints/ddpm_epoch_3.pth"
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

    number_of_images = 1

    print(
        f"Generating {number_of_images} images..."
    )

    generated_images = diffusion.sample(
        model=model,
        number_of_images=number_of_images,
        image_channels=1
    )

    output_folder = Path("outputs")
    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    for index, image in enumerate(
        generated_images,
        start=1
    ):
        output_path = (
            output_folder
            / f"generated_xray_{index}.png"
        )

        save_image(
            image,
            output_path
        )

        print("Saved:", output_path)

    grid_path = output_folder / "generated_xray_grid.png"

    save_image(
        generated_images,
        grid_path,
        nrow=2
    )

    print("Grid saved:", grid_path)
    print("Image generation completed!")


if __name__ == "__main__":
    main()