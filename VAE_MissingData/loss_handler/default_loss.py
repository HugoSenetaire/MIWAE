from torch import Tensor
import torch
from typing import Callable, Optional
from torch.nn.modules.loss import NLLLoss, MSELoss

class WeightsMultiplication(MSELoss):
    """
    Do a weighted sum of the input with a given weight. 
    Mostly used for training the Gaussian Generative model where the output of the model is the log-likelihood of the data.
    """
    def __init__(self, size_average=None, reduce=None,  reduction: str = 'none') -> None:
        super().__init__(size_average, reduce = None, reduction = "none")
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor, weights: Tensor = None) -> Tensor:
        if self.reduction == 'none':
            return input
        else :
            if weights is None :
                return torch.sum(input, -1)
            return torch.dot(input, weights)