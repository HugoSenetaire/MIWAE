import torch.nn as nn
from .utils import View


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim: int, output_channel : int):
        super(ConvDecoder, self).__init__()
        self.conv_layers = nn.Sequential(nn.Linear(latent_dim, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, 512),
                                        nn.ReLU(),
                                        View((-1, 32, 4, 4)),
                                        nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=2),
                                        nn.ReLU(),
                                        nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=2, output_padding=1),
                                        nn.ReLU(),
                                        nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=5, stride=2, padding=2, output_padding=1),
                                    )

    def forward(self, x):
        
        _out = self.conv_layers(x)
        return _out