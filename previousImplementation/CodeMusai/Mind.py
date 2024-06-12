from CodeMusai.NeuralConfig import NeuralCircuitSettings
from CodeMusai.Cortex import Cortex



import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import inspect
import tiktoken
import os
import pickle

class Mind(nn.Module):
    """
    Mind is the interface between the user and the cortices.
    """
    def __init__(self, neuroConfig : NeuralCircuitSettings): 
        super().__init__()

        if neuroConfig is None:
            #neuroConfig = NeuralCircuitSettings()
            print("No configuration provided, using default settings.")

        assert neuroConfig.vocabularySize is not None
        assert neuroConfig.memorySpan is not None            

        self.neuroConfig = neuroConfig

        
        self.EmotionCortex = None; #trained on music
        #self.CognitiveCortex = Cortex(neuroConfig);
        self.AuditoryCortex = None; #trained on sound patterns
        self.VisionCortex = None; #trained on visual patterns
        self.SomatosensoryCortex = None; #trained on touch patterns

        self.Thalamus = None; #todo
        

        self.cortices = nn.ModuleDict(dict(
            LanguageCortex = Cortex(self.neuroConfig), #trained on language
            EmotionCortex = None, #trained on music
            AuditoryCortex = None, #trained on sound patterns
            VisionCortex = None, #trained on visual patterns
            SomatosensoryCortex = None, #trained on touch patterns
            Thalamus = None, #todo
        ))
        
        

    def forward(self, dendrites_input, targets=None, external_context=None): 
        #for now we just have the one working cortex
        device = dendrites_input.device
        
        logits, loss = self.cortices.LanguageCortex(dendrites_input, targets, external_context)

        return logits, loss
    
    def loadMemories(self, initializationMethod, dropout=0.0):
        self.cortices.LanguageCortex = self.cortices.LanguageCortex.loadMemories(initializationMethod, dropout)

    def configureNeuralOptimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()} # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad} # filter out those that do not require grad

        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        neuralOptimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused Adaptive Moment Estimation W: {use_fused}")

        return neuralOptimizer

    def estimate_cognitiveLoad(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        
        numberOfSynapses = self.getNumberOfSynapses()
        cfg = self.neuroConfig
        depth, parallelAttentionPathways, neuronDensity, memorySpan = cfg.depth, cfg.parallelAttentionPathways, cfg.neuronDensity, cfg.memorySpan
        flops_per_token = 6*numberOfSynapses + 12*depth*parallelAttentionPathways*neuronDensity//parallelAttentionPathways*memorySpan
        flops_per_fwdbwd = flops_per_token * memorySpan
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu
    
    def getNumberOfSynapses(self, non_embedding=True):
        numberOfSynapses = 0
        for cortex in self.cortices.values():
            if (cortex is not None):
                numberOfSynapses += cortex.getNumberOfSynapses(non_embedding)
        return numberOfSynapses

