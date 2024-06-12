
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as f


@dataclass
class CodeMusaiConfig:
    memorySpan: int = 256 #this is the context awareness size, traditionally block_size
    vocabularySize: int = 65 #this is the vocabulary size, traditionally vocab_size
    depth: int = 6 #this is the depth of the model, the number of hidden layers, traditionally n_layers
    awarenessNode: int = 6 #this is the awareness node size, or the number of heads of attention, traditionally n_heads
    conceptualDensity: int = 384 #this is the conceptual density, the ability to embed concepts in a multimension space in relations to the tokens, traditionally n_embd



