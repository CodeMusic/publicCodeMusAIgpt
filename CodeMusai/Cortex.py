import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from CodeMusai.NeuralCircuitSettings import NeuralCircuitSettings
from CodeMusai.CognitiveClarifier import CognitiveClarifier
from CodeMusai.ThoughtProcessor import ThoughtProcessor
import tiktoken
import os
import pickle

class Cortex(nn.Module):
    """
    Cortex (GPT) is like the whole brain, composed of multiple ThoughtProcessors,
    which together process input data sequentially to generate output, mimicking the flow of thought.
    """
    def __init__(self, neuroConfig: NeuralCircuitSettings): 
        super().__init__()

        if neuroConfig is None:
            print("No configuration provided, using default settings.")

        assert neuroConfig.vocabSize is not None
        assert neuroConfig.block_size is not None            

        self.neuroConfig = neuroConfig

        self.tokenEmbedding = nn.Embedding(neuroConfig.vocabSize, neuroConfig.n_embd)
        self.positionalEmbedding = nn.Embedding(neuroConfig.block_size, neuroConfig.n_embd)
        self.dropout = nn.Dropout(neuroConfig.dropout)
        self.thoughtProcessors = nn.ModuleList([ThoughtProcessor(neuroConfig) for _ in range(neuroConfig.n_layer)])
        self.outputNormalizer = CognitiveClarifier(neuroConfig.n_embd, bias=neuroConfig.bias)
        self.outputLayer = nn.Linear(neuroConfig.n_embd, neuroConfig.vocabSize, bias=False)
        self.tokenEmbedding.weight = self.outputLayer.weight  # Weight tying

        self.apply(self._init_lessonScalingFactors)
        
        for pn, p in self.named_parameters():  # Apply special scaled init to the residual projections, per GPT-2 paper
            if pn.endswith('outputProjection.weight') or pn.endswith('outputLayer.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * neuroConfig.n_layer))

    def getNumberOfSynapses(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        synapsesCount_n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            synapsesCount_n_params -= self.positionalEmbedding.weight.numel()
        return synapsesCount_n_params

    def _init_lessonScalingFactors(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, dendrites_input, targets=None, external_context=None):
        device = dendrites_input.device
        cognitiveBatchSize, cognitiveSequenceLength = dendrites_input.size()
        assert cognitiveSequenceLength <= self.neuroConfig.block_size, f"Cannot forward sequence of length {cognitiveSequenceLength}, block size is only {self.neuroConfig.block_size}"
        temporalPositions = torch.arange(0, cognitiveSequenceLength, dtype=torch.long, device=device)

        tokenEmbeddings = self.tokenEmbedding(dendrites_input)
        positionalEmbeddings = self.positionalEmbedding(temporalPositions)
        sensoryInput = self.dropout(tokenEmbeddings + positionalEmbeddings)
        
        for aThoughtProcessor in self.thoughtProcessors:
            sensoryInput = aThoughtProcessor(sensoryInput)

        behaviouralResponse = self.outputNormalizer(sensoryInput)
        
        if targets is not None:
            logits = self.outputLayer(behaviouralResponse)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.outputLayer(behaviouralResponse[:, [-1], :])
            loss = None

        return logits, loss

    def reduceThoughtProcessorSize_crop_block_size(self, block_size):
        assert block_size <= self.neuroConfig.block_size
        self.neuroConfig.block_size = block_size
        self.positionalEmbedding.weight = nn.Parameter(self.positionalEmbedding.weight[:block_size])
        for aThoughtProcessor in self.thoughtProcessors:
            if hasattr(aThoughtProcessor.attn, 'bias'):
                aThoughtProcessor.attn.bias = aThoughtProcessor.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def loadMemories(cls, modelType, overrideArgs=None):
        assert modelType in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        overrideArgs = overrideArgs or {}
        
        assert all(k == 'dropout' for k in overrideArgs)
        from transformers import GPT2LMHeadModel
        print(f"Loading weights from pretrained GPT: {modelType}")
        
        configArgs = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),
        }[modelType]
        configArgs.update({'vocabSize': 50257, 'block_size': 1024, 'bias': True})

        if 'dropout' in overrideArgs:
            configArgs['dropout'] = overrideArgs['dropout']

        neuroConfig = NeuralCircuitSettings(**configArgs)
        model = cls(neuroConfig)
        memories = model.state_dict()
        memoryKeys_sd_keys = [key for key in memories.keys() if not key.endswith('.attn.bias')]

        model_hf = GPT2LMHeadModel.from_pretrained(modelType)
        memories_hf = model_hf.state_dict()
        memories_keys_hf = [key for key in memories_hf.keys() if not key.endswith('attn.masked_bias') and not key.endswith('attn.bias')]

        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        
        print(f"Number of keys in Hugging Face model: {len(memories_keys_hf)}")
        print(f"Number of keys in our model: {len(memoryKeys_sd_keys)}")

        if len(memories_keys_hf) != len(memoryKeys_sd_keys):
            print(f"Mismatched keys: {len(memories_keys_hf)} != {len(memoryKeys_sd_keys)}")
            print("The following keys are missing or extra:")
            for key in set(memories_keys_hf).symmetric_difference(memoryKeys_sd_keys):
                print(key)

        for key in memories_keys_hf:
            if key in memories:
                if any(key.endswith(w) for w in transposed):
                    assert memories_hf[key].shape[::-1] == memories[key].shape
                    with torch.no_grad():
                        memories[key].copy_(memories_hf[key].t())
                else:
                    assert memories_hf[key].shape == memories[key].shape
                    with torch.no_grad():
                        memories[key].copy_(memories_hf[key])
            else:
                print(f"Skipping key {key} as it is not present in the model state dict")

        return model

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, gradualOutput=False, external_context=None):
        encode, decode = self.getCognitiveInterpreters()
        print('')
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.neuroConfig.block_size else idx[:, -self.neuroConfig.block_size:]
            logits, _ = self(idx_cond, external_context=external_context)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probabilities = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probabilities, num_samples=1)

            if idx_next.item() == 50256 or idx_next.item() == encode(""):
                print("\nNatural Stop\n")
                break

            if idx_next < 0:
                print(f"Negative token index encountered: {idx_next}")
                raise ValueError(f"Negative token index encountered: {idx_next}")
            
            idx = torch.cat((idx, idx_next), dim=1)

            if gradualOutput:
                next_token_text = decode([idx_next])
                print(next_token_text, end='', flush=True)

        return decode(idx[0].tolist())
    
    def save_model(self, file_path):
        torch.save(self.state_dict(), file_path)
        print(f"Model saved to {file_path}")

    @classmethod
    def load_model(cls, file_path, config):
        model = cls(config)
        model.load_state_dict(torch.load(file_path, map_location=torch.device('cpu')))
        print(f"Model loaded from {file_path}")
        return model
          
    def getCognitiveInterpreters(self, metaPath=None):
        if metaPath is None:
            encoder = tiktoken.get_encoding("gpt2")
            encode = lambda s: encoder.encode(s, allowed_special={""})
            decode = lambda l: encoder.decode(l)
        else:
            if os.path.exists(metaPath):
                print(f"Loading meta from {metaPath}…")
                with open(metaPath, 'rb') as file:
                    meta = pickle.load(file)
                toStringIndex, indexToString = meta['indexToString'], meta['stringToIndex']
            encode = lambda s: [toStringIndex[c] for c in s]
            decode = lambda l: ''.join([indexToString[I] for I in l])
        return encode, decode