import torch
import torch.nn as nn


class InjectNoise(nn.Module):

    def __init__(self, channels):
        super().__init__()
        random_weight = torch.randn((1, channels, 1, 1))
        self.weight = nn.Parameter(
            random_weight

        )

    def forward(self, image):

        n_sample, channel, height, width = image.shape
        noise_shape = (n_sample, 1, height, width)

        noise = torch.randn(noise_shape, device=image.device)  # Creates the random noise
        return image + self.weight * noise  # Applies to image after multiplying by the weight for each channel


