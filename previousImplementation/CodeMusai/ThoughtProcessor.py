import torch
import torch.nn as nn
from CodeMusai.NeuralConfig import NeuralCircuitSettings
from CodeMusai.CognitiveClarifier import CognitiveClarifier
from CodeMusai.TemporalAttention import TemporalAttention
from CodeMusai.NeuralNetwork import NeuralNetwork

class ThoughtProcessor(nn.Module):
    """
    ThoughtProcessor (TransformerBlock) represents a single processing unit of the brain, combining attention and neural network layers
    to process and refine information.
    """
    def __init__(self, neuroConfig : NeuralCircuitSettings): 
        super().__init__()
        self.signalNormalizerPre = CognitiveClarifier(neuroConfig.neuronDensity, useBias=neuroConfig.useBias)
        self.temporalAttention = TemporalAttention(neuroConfig)
        self.signalNormalizerPost = CognitiveClarifier(neuroConfig.neuronDensity, useBias=neuroConfig.useBias)
        self.neuralNetwork_mlp = NeuralNetwork(neuroConfig)
        self.device = neuroConfig.device

    def forward(self, dendrites_input, external_context=None): 
        dendrites_input = self.signalNormalizerPre(dendrites_input).to(self.device)
        dendrites_input = dendrites_input + self.temporalAttention(dendrites_input).to(self.device)
        
        #if external_context is not None:
        #   external_context = external_context.to(dtype=torch.float32).to(self.device)
        #    # Apply cross-attention
        #    cross_attn_output = self.crossAttention(dendrites_input, external_context).to(self.device)
        #    dendrites_input = dendrites_input + cross_attn_output

        axon_output = dendrites_input + self.neuralNetwork_mlp(self.signalNormalizerPost(dendrites_input))
        return axon_output
