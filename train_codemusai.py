
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as f
import math


@dataclass
class CodeMusaiConfig:
    memorySpan: int = 256 #this is the context awareness size, traditionally block_size
    vocabularySize: int = 65 #this is the vocabulary size, traditionally vocab_size
    depth: int = 6 #this is the depth of the model, the number of hidden layers, traditionally n_layers
    awarenessNodes: int = 6 #this is the awareness node size, or the number of heads of attention, traditionally n_heads
    conceptualDensity: int = 384 #this is the conceptual density, the ability to embed concepts in a multimension space in relations to the tokens, traditionally n_embd


class CognitiveClarity(nn.Module):
    def __init__(self, config: CodeMusaiConfig):
        super().__init__()
        self.config = config

    def forward(self, x):
        return nn.LayerNorm(x)


class TemporalFocus(nn.Module):

    def __init__(self, config: CodeMusaiConfig):
        super().__init__()
        #self.config = config
        assert config.conceptualDensity % config.awarenessNodes == 0

        #key, quests, values projects for all heads, but in a batch
        self.c_attn = nn.Linear(config.conceptualDensity, 3 * config.conceptualDensity)
        #output projection
        self.c_proj = nn.Linear(config.conceptualDensity, config.conceptualDensity)
        #regularization
        self.n_head = config.awarenessNodes
        self.n_embd = config.conceptualDensity

        #bias mask following openai/hf names so we can load them
        self.register_buffer("bias", torch.tril(torch.ones(config.memorySpan, config.memorySpan))
                                        .view(1, 1, config.memorySpan, config.memorySpan))
    
    def forward(self, x):
        B, T, C = x.size() #batch size, sequence length, and embedding dimensionality (n_embd)
        #calcualtes query, key, and values for all heads in batch and move head forward to be the batch
        #nh is 'number of heads', hs is 'head size', and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, and so nh*hs=C=12*64=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=-2)
        #q, k, v = qkv.chunk(3, dim=-1)
        #calculate attention scores ('affinities') by applying the bias and then applying the causal mask
        #affinities = (q @ k.transpose(-2, -1)) * (1 / math.sqrt(hs)) + bias
        #affinities = q @ k.transpose(-2, -1) * (1 / math.sqrt(hs)) + self.bias

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        #attention scores ('affinities') by applying the bias and then applying the causal mask
        #affinities = (q @ k.transpose(-2, -1)) * (1 / math.sqrt(hs)) + bias
        #affinities = q @ k.transpose(-2, -1) * (1 / math.sqrt(hs)) + self.bias
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) #+ self.bias
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = att.softmax(att, dim=-1)

        y = att @ v #(B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) #re-align nh and hs

        #output projection
        y = self.c_proj(y)
        return y
        #att = self.attn_drop(att)
        

class NeuralNetwork(nn.Module):
    def __init__(self, config: CodeMusaiConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.conceptualDensity, 4 * config.conceptualDensity)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.conceptualDensity, config.conceptualDensity)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class ThoughtProcessor(nn.Module):
    
    def __init__(self, config: CodeMusaiConfig):
        super().__init__()
        self.config = config

        self.ln_1 = nn.LayerNorm(config.conceptualDensity)
        self.attn = TemporalFocus(config)
        self.ln_2 = nn.LayerNorm(config.conceptualDensity)
        self.mlp = NeuralNetwork(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Cortex(nn.Module):

    def __init__(self, config: CodeMusaiConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocabularySize, config.conceptualDensity),
            wpe = nn.Embedding(config.memorySpan, config.conceptualDensity),
            h = nn.ModuleList([ThoughtProcessor(config) for _ in range(config.depth)]),
            ln_f = nn.LayerNorm(config.conceptualDensity),
            ))
        self.lm_head = nn.Linear(config.conceptualDensity, config.vocabularySize)
        #self.embeddings = nn.Embedding(config.vocabularySize, config.conceptualDensity)


