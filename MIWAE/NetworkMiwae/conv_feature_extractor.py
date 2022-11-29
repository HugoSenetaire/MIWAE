


import torch.nn as nn


class ConvFeatureExtractor(nn.Module):
    def __init__(self, input_channel : int = 1) -> None:
        super(ConvFeatureExtractor, self).__init__()
        self.input_channel = input_channel        
        self.module = nn.Sequential(
                nn.Conv2d(in_channels=input_channel, out_channels=16, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
            )
    def forward(self, x):
        return self.module(x)