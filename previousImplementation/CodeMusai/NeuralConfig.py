from dataclasses import dataclass

@dataclass
class NeuralCircuitSettings:
#{
    useEmotionCore: bool = False
    memorySpan: int = 1024  #block_size - Length of the sequences processed by the model, akin to the 'memory span' in neural circuits
    vocabularySize: int = 50304  #vocab_size - Number of unique tokens recognized by the model, similar to the 'vocabulary' of a brain
    depth: int = 12  #n_layer - Number of processing layers, representing the depth of neural processing stages
    parallelAttentionPathways: int = 12  #n_head - Number of attention heads, analogous to parallel processing pathways in the brain
    neuronDensity: int = 768  #n_embd - Dimensionality of feature representations, similar to the complexity of neural signals
    dropoutRate: float = 0.0  #dropout - Regularization technique to prevent overfitting, like synaptic pruning in neural development
    useBias: bool = True  #bias - Whether to use bias in linear transformations and normalization, enhancing the model's flexibility
    device: str = 'mps'  #device - The device to use for computations, can be 'cpu', 'cuda', or 'mps' for Apple Silicon devices
