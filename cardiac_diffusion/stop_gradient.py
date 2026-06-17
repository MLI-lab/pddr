from typing import Tuple

import torch
import torch.nn as nn

from torch import Tensor

from cardiac_diffusion.diffusion import Diffusion

class PassThrough(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input1: Tensor, input2: Tensor, input3: Tensor, model: Diffusion) -> Tensor:
        return model(input1, input2, input3)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, None, None]:
        return grad_output, None, None, None
    
class ScoreWithIdentityGradWrapper(nn.Module):
    def __init__(self, model: Diffusion):
        super().__init__()
        self.model = model
        self.fn = PassThrough.apply

    def forward(self, input1, input2, input3):
        return self.fn(input1, input2, input3, self.model)
