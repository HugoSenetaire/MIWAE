import torch.nn as nn
import numpy as np


class ConvDecoderMaskMNIST(nn.Module):
    def __init__(self, input_size: tuple = (1,10), output_size : int = (1,28,28),):
        super(ConvDecoderMaskMNIST, self).__init__()
        self.input_channel = input_size[0]
        self.output_size = output_size
        self.out_channel = output_size[0]
        self.conv_layers = nn.Sequential(nn.Conv2d(in_channels=self.input_channel, out_channels=16, kernel_size=5, stride=2, padding=2),
                                        nn.ReLU(),
                                        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2),
                                        nn.ReLU(),
                                        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=2),
                                        nn.ReLU(),
                                        nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=2),
                                        nn.ReLU(),
                                        nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=2, output_padding=1),
                                        nn.ReLU(),
                                        nn.ConvTranspose2d(in_channels=16, out_channels=self.out_channel, kernel_size=5, stride=2, padding=2, output_padding=1),
                                    )

    def forward(self, x):
        _out = self.conv_layers(x)
        return _out