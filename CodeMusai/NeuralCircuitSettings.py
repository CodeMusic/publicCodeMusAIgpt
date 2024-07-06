from dataclasses import dataclass

@dataclass
class NeuralCircuitSettings:
    n_embd: int = 64
    n_head: int = 8
    dropout: float = 0.1
    block_size: int = 128
    bias: bool = True
    vocabSize: int = 50257
    n_layer: int = 12