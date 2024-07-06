import torch
import torch.nn as nn
from CodeMusai.Neuron import Neuron

class NeuralNetwork(nn.Module):

    def __init__(self, numberOfNeurons, bias=True, dropout=0.0):
        super().__init__()
        self.neurons = nn.ModuleList()
        for I in range(len(numberOfNeurons) - 1):
            self.neurons.append(Neuron(numberOfNeurons[I], numberOfNeurons[I + 1], bias=bias, dropout=dropout))

    def forward(self, x):
        for aNeuron in self.neurons:
            x = aNeuron(x)
        return x