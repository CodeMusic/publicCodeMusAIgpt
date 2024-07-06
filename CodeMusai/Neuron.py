import torch
import torch.nn as nn

class Neuron(nn.Module):

    def __init__(self, in_features, out_features, bias=True, dropout=0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear(x)
        x = self.gelu(x)
        x = self.dropout(x)
        return x