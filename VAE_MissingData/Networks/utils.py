
import torch
import torch.nn as nn


class View(nn.Module):
    """ For reshaping tensors inside Sequential objects"""
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

class Flatten(nn.Module):
    """ For reshaping tensors inside Sequential objects"""
    def __init__(self, position =1):
        super(Flatten, self).__init__()
        self.position = position

    def forward(self, tensor):
        return tensor.flatten(self.position)