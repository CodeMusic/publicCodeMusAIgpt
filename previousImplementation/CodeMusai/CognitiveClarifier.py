
import torch
import torch.nn as nn
from torch.nn import functional as F

class CognitiveClarifier(nn.Module):
    """
    CognitiveClarifier (LayerNorm) simulates how neurons in the brain normalize signals.
    It optionally includes a bias term, which is not typically supported directly in PyTorch's LayerNorm.
    """
    def __init__(self, neuronCount, useBias=False): 
        super().__init__()
        self.scalingFactor_weight = nn.Parameter(torch.ones(neuronCount))
        self.scalingFactor_bias = nn.Parameter(torch.zeros(neuronCount)) if useBias else None

    def forward(self, dendrites_input): 
        return F.layer_norm(dendrites_input, self.scalingFactor_weight.shape, self.scalingFactor_weight, self.scalingFactor_bias, 1e-5)
    