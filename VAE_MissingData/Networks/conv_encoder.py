

from .utils import Flatten
import torch.nn as nn


class ConvEncoderMNIST(nn.Module):
    def __init__(self, input_size : tuple = (1, 28, 28,), latent_dim = 10) -> None:
        super(ConvEncoderMNIST, self).__init__()
        self.input_channel = input_size[0]
        self.module = nn.Sequential(
                nn.Conv2d(in_channels=self.input_channel, out_channels=16, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                Flatten(1),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, latent_dim),
            )
    def forward(self, x):
        return self.module(x)