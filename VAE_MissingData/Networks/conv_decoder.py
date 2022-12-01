import torch.nn as nn
import numpy as np
from .utils import View


class ConvDecoderMNIST(nn.Module):
    def __init__(self, input_size: tuple = (1,10), output_size : int = 1,):
        super(ConvDecoderMNIST, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.out_channel = output_size[0]
        self.latent_dim = np.prod(input_size)
        self.conv_layers = nn.Sequential(nn.Linear(self.latent_dim, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, 512),
                                        nn.ReLU(),
                                        View((-1, 32, 4, 4)),
                                        nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=2),
                                        nn.ReLU(),
                                        nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=2, output_padding=1),
                                        nn.ReLU(),
                                        nn.ConvTranspose2d(in_channels=16, out_channels=self.out_channel, kernel_size=5, stride=2, padding=2, output_padding=1),
                                    )

    def forward(self, x):
        # print(x.shape)
        _out = self.conv_layers(x)
        return _out