import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from CodeMusai.NeuralConfig import NeuralCircuitSettings
class TemporalAttention(nn.Module):
    """
    TemporalAttention (CausalSelfAttention) mimics the brain's attention mechanism by focusing only on earlier or current information
    in the sequence to predict the next part of the sequence, ensuring the flow of time in predictions.
    """
    
    def __init__(self, neuroConfig: NeuralCircuitSettings, isTemporal: bool = True): 
        super().__init__()
        assert neuroConfig.neuronDensity % neuroConfig.parallelAttentionPathways == 0
        self.isTemporal = isTemporal

        self.keyQueryValueProjection = nn.Linear(neuroConfig.neuronDensity, 3 * neuroConfig.neuronDensity, bias=neuroConfig.useBias) # key, query, value projections for all heads, but in a batch        
        self.outputProjection = nn.Linear(neuroConfig.neuronDensity, neuroConfig.neuronDensity, bias=neuroConfig.useBias) # output projection
        
        self.attentionDropout = nn.Dropout(neuroConfig.dropoutRate)
        self.residualDropout = nn.Dropout(neuroConfig.dropoutRate) 

        self.parallelAttentionPathways = neuroConfig.parallelAttentionPathways
        self.neuronDensity = neuroConfig.neuronDensity
        self.dropout = neuroConfig.dropoutRate

        if self.isTemporal:
            self.register_buffer("bias", torch.tril(torch.ones(neuroConfig.memorySpan, neuroConfig.memorySpan)) 
                                        .view(1, 1, neuroConfig.memorySpan, neuroConfig.memorySpan))

    def forward(self, dendrites_input): 
        parallelThoughtBatch, thoughtSequence, multiSensoryRichnessEmbedding = dendrites_input.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        query, key, value  = self.keyQueryValueProjection(dendrites_input).split(self.neuronDensity, dim=2)
        key = key.view(parallelThoughtBatch, thoughtSequence, self.parallelAttentionPathways, multiSensoryRichnessEmbedding // self.parallelAttentionPathways).transpose(1, 2) # (B, nh, T, hs)
        query = query.view(parallelThoughtBatch, thoughtSequence, self.parallelAttentionPathways, multiSensoryRichnessEmbedding // self.parallelAttentionPathways).transpose(1, 2) # (B, nh, T, hs)
        value = value.view(parallelThoughtBatch, thoughtSequence, self.parallelAttentionPathways, multiSensoryRichnessEmbedding // self.parallelAttentionPathways).transpose(1, 2) # (B, nh, T, hs)
        #(B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        attention = (query @ key.transpose(-2, -1)) * (1.0 / math.sqrt(key.size(-1)))
        
        if self.isTemporal:
            attention = attention.masked_fill(self.bias[:,:,:thoughtSequence,:thoughtSequence] == 0, float('-inf'))
        
        attention = F.softmax(attention, dim=-1)
        attention = self.attentionDropout(attention)
        axon_output = attention @ value # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        axon_output = axon_output.transpose(1, 2).contiguous().view(parallelThoughtBatch, thoughtSequence, multiSensoryRichnessEmbedding) 

        axon_output = self.residualDropout(self.outputProjection(axon_output))
        return axon_output

