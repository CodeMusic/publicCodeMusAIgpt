import torch
import torch.nn as nn
from CodeMusai.CognitiveClarifier import CognitiveClarifier
from CodeMusai.TemporalAttention import TemporalAttention
from CodeMusai.NeuralNetwork import NeuralNetwork

class ThoughtProcessor(nn.Module):
    """
    A block consisting of multi-head self-attention, feed-forward network, layer normalization, and residual connections.
    
    Arguments:
        config (object): Configuration object with attributes n_embd, n_head, dropout, block_size, bias.
    """
    def __init__(self, config):
        super().__init__()
        self.ln1 = CognitiveClarifier(config.n_embd, bias=config.bias)
        self.ln2 = CognitiveClarifier(config.n_embd, bias=config.bias)
        self.attn = TemporalAttention(config)
        self.mlp = NeuralNetwork([config.n_embd, 4 * config.n_embd, config.n_embd], bias=config.bias, dropout=config.dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x