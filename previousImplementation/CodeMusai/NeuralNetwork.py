import torch.nn as nn
from CodeMusai.NeuralConfig import NeuralCircuitSettings

class NeuralNetwork(nn.Module):
    """
    NeuralNetwork (MLP) simulates a densely connected neural network layer in the brain, where each neuron's output can influence many other neurons.
    It's used here to transform the output of the attention mechanism before it's sent to the next layer or used as a final output.
    """

    def __init__(self, neuroConfig: NeuralCircuitSettings): 
        super().__init__()
        self.denseLayer = nn.Linear(neuroConfig.neuronDensity, 4 * neuroConfig.neuronDensity, bias=neuroConfig.useBias)
        self.activationFunction  = nn.GELU()
        self.outputLayer = nn.Linear(4 * neuroConfig.neuronDensity, neuroConfig.neuronDensity, bias=neuroConfig.useBias)
        self.dropoutLayer = nn.Dropout(neuroConfig.dropoutRate)

    def forward(self, dendrites_input): 
    
        dendrites_input = self.denseLayer(dendrites_input) # Apply first linear transformation 
        primedDendrites = self.activationFunction(dendrites_input) # Apply GELU activation function, similar to the way neurons are activated in the brain 
        axon_output = self.outputLayer(primedDendrites) # Apply second linear transformation 
        axon_output = self.dropoutLayer(axon_output) # Apply dropout for regularization 
  
        return axon_output