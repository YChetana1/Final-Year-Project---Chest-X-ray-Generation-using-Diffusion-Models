
import copy

import torch
import torch.nn as nn


class EMA:
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
    ):
        self.decay = decay

        self.ema_model = copy.deepcopy(model)

        self.ema_model.eval()

        for parameter in self.ema_model.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        model_parameters = dict(
            model.named_parameters()
        )

        ema_parameters = dict(
            self.ema_model.named_parameters()
        )

        for name, ema_parameter in ema_parameters.items():
            model_parameter = model_parameters[name]

            ema_parameter.data.mul_(self.decay)

            ema_parameter.data.add_(
                model_parameter.data,
                alpha=1.0 - self.decay,
            )

        model_buffers = dict(
            model.named_buffers()
        )

        ema_buffers = dict(
            self.ema_model.named_buffers()
        )

        for name, ema_buffer in ema_buffers.items():
            ema_buffer.copy_(
                model_buffers[name]
            )

    def state_dict(self):
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict) -> None:
        self.ema_model.load_state_dict(
            state_dict
        )

    def to(self, device):
        self.ema_model.to(device)
        return self
