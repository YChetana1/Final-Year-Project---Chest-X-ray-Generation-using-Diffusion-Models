import torch


class Diffusion:
    def __init__(
        self,
        noise_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        image_size: int = 128,
        device: str = "cpu"
    ):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.image_size = image_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(self.device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self) -> torch.Tensor:
        return torch.linspace(
            self.beta_start,
            self.beta_end,
            self.noise_steps
        )

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        return torch.randint(
            low=1,
            high=self.noise_steps,
            size=(batch_size,),
            device=self.device
        )

    def add_noise(
        self,
        images: torch.Tensor,
        timesteps: torch.Tensor
    ):
        sqrt_alpha_hat = torch.sqrt(
            self.alpha_hat[timesteps]
        )[:, None, None, None]

        sqrt_one_minus_alpha_hat = torch.sqrt(
            1.0 - self.alpha_hat[timesteps]
        )[:, None, None, None]

        noise = torch.randn_like(images)

        noisy_images = (
            sqrt_alpha_hat * images
            + sqrt_one_minus_alpha_hat * noise
        )

        return noisy_images, noise