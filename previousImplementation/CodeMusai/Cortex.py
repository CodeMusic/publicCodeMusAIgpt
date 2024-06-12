import torch
import torch.nn as nn
from CodeMusai.NeuralConfig import NeuralCircuitSettings
from CodeMusai.CognitiveClarifier import CognitiveClarifier
from CodeMusai.ThoughtProcessor import ThoughtProcessor
import math
import torch.nn.functional as F
import inspect
import tiktoken
import os
import pickle

class Cortex(nn.Module):
    """
    Cortex (GPT) is like the whole brain, composed of multiple ThoughtProcessors,
    which together process input data sequentially to generate output, mimicking the flow of thought.
    """
    def __init__(self, neuroConfig : NeuralCircuitSettings): 
        super().__init__()

        if neuroConfig is None:
            #neuroConfig = NeuralCircuitSettings()
            print("No configuration provided, using default settings.")

        assert neuroConfig.vocabularySize is not None
        assert neuroConfig.memorySpan is not None            

        self.neuroConfig = neuroConfig

        self.transformer = nn.ModuleDict(dict(
            tokenEmbedding = nn.Embedding(neuroConfig.vocabularySize, neuroConfig.neuronDensity),
            positionalEmbedding = nn.Embedding(neuroConfig.memorySpan, neuroConfig.neuronDensity),
            dropout = nn.Dropout(neuroConfig.dropoutRate),
            thoughtProcessors = nn.ModuleList([ThoughtProcessor(neuroConfig) for _ in range(neuroConfig.depth)]),
            outputNormalizer = CognitiveClarifier(neuroConfig.neuronDensity, useBias=neuroConfig.useBias),
        ))
        self.outputLayer = nn.Linear(neuroConfig.neuronDensity, neuroConfig.vocabularySize, bias=False)
        self.transformer.tokenEmbedding.weight = self.outputLayer.weight # https://paperswithcode.com/method/weight-tying

        self.apply(self._init_lessonScalingFactors)
        
        for pn, p in self.named_parameters(): # apply special scaled init to the residual projections, per GPT-2 paper
            if pn.endswith('outputProjection.weight') or pn.endswith('outputLayer.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * neuroConfig.depth))

    def getNumberOfSynapses(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        synapsesCount_n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            synapsesCount_n_params -= self.transformer.positionalEmbedding.weight.numel()
        return synapsesCount_n_params

    def _init_lessonScalingFactors(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, dendrites_input, targets=None, external_context=None):  #neural_inputs=idx # Sequentially pass the input through each ThoughtProcessor
        
        device = dendrites_input.device
        cognitiveBatchSize, cognitiveSequenceLength = dendrites_input.size()
        assert cognitiveSequenceLength <= self.neuroConfig.memorySpan, f"Cannot forward sequence of length {cognitiveSequenceLength}, block size is only {self.neuroConfig.memorySpan}"
        temporalPositions = torch.arange(0, cognitiveSequenceLength, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tokenEmbeddings = self.transformer.tokenEmbedding(dendrites_input) # token embeddings of shape (b, t, n_embd)
        positionalEmbeddings = self.transformer.positionalEmbedding(temporalPositions) # position embeddings of shape (t, n_embd)
        sensoryInput = self.transformer.dropout(tokenEmbeddings + positionalEmbeddings)
        for aThoughtProcessor in self.transformer.thoughtProcessors:
            sensoryInput = aThoughtProcessor(sensoryInput, external_context)
        behaviouralResponse = self.transformer.outputNormalizer(sensoryInput)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.outputLayer(behaviouralResponse)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.outputLayer(behaviouralResponse[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def reduceThoughtProcessorSize_crop_block_size(self, memorySpan):
        assert memorySpan <= self.neuroConfig.memorySpan
        self.neuroConfig.memorySpan = memorySpan
        self.transformer.positionalEmbedding.weight = nn.Parameter(self.transformer.positionalEmbedding.weight[:memorySpan])
        for aThoughtProcessor in self.transformer.thoughtProcessors:
            if hasattr(aThoughtProcessor.temporalAttention, 'bias'):
                aThoughtProcessor.temporalAttention.bias = aThoughtProcessor.temporalAttention.bias[:,:,:memorySpan,:memorySpan]

    @classmethod
    def loadMemories(cls, modelType, override_args=None):
        assert modelType in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        overrideArgs = overrideArgs or {} # default to empty dict
        
        assert all(k == 'dropoutRate_dropout' for k in overrideArgs) # only dropout can be overridden see more notes below
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % modelType)
        
        configArgs = { # n_layer, n_head and n_embd are determined from model_type
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[modelType]
        print("forcing vocabularySize=50257, memorySpan=1024, bias=True")
        configArgs['vocabularySize'] = 50257 # always 50257 for GPT model checkpoints
        configArgs['memorySpan'] = 1024 # always 1024 for GPT model checkpoints
        configArgs['useBias'] = True # always True for GPT model checkpoints

        # we can override the dropout rate, if desired
        if 'dropoutRate' in overrideArgs:
            print(f"overriding dropout rate to {overrideArgs['dropoutRate']}")
            configArgs['dropoutRate'] = overrideArgs['dropoutRate']
        # create a from-scratch initialized minGPT model
        neuroConfig = NeuralCircuitSettings(**configArgs)
        model = Cortex(neuroConfig)
        memories = model.state_dict()
        memoryKeys_sd_keys = memories.keys()
        memoryKeys_sd_keys = [key for key in memoryKeys_sd_keys if not key.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(modelType)
        memories_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        memories_keys_hf = memories_hf.keys()
        memories_keys_hf = [key for key in memories_keys_hf if not key.endswith('attn.masked_bias')] # ignore these, just a buffer
        memories_keys_hf = [key for key in memories_keys_hf if not key.endswith('attn.bias')] # same, just the mask (buffer)
        transposed = ['temporalAttention_attn.keyQueryValueProjection_c_attn.weight', 'temporalAttention_attn.outputProjection_c_proj.weight', 'NeuralNetwork.denseLayer_c_fc.weight', 'NeuralNetwork.outputLayer_c_proj.weight']
        
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(memories_keys_hf) == len(memoryKeys_sd_keys), f"mismatched keys: {len(memories_keys_hf)} != {len(memoryKeys_sd_keys)}"
        for key in memories_keys_hf:
            if any(key.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert memories_hf[key].shape[::-1] == memories[key].shape
                with torch.no_grad():
                    memories[key].copy_(memories_hf[key].t())
            else:
                # vanilla copy over the other parameters
                assert memories_hf[key].shape == memories[key].shape
                with torch.no_grad():
                    memories[key].copy_(memories_hf[key])

        return model


    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, gradualOutput=False, external_context=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """

        encode, decode = getCognitiveInterpreters()
        #external_context = self.prepare_system_message(external_context)
        #external_context = encode(external_context)

        for _ in range(max_new_tokens):

            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.neuroConfig.memorySpan else idx[:, -self.neuroConfig.memorySpan:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond, external_context=external_context)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probabilities = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probabilities, num_samples=1)

            if idx_next.item() == 50256 or idx_next.item() == encode("<|endoftext|>"):
                print("\nNatural Stop\n")
                break

            if idx_next < 0:
                raise ValueError(f"Negative token index encountered: {idx_next}")
            
            idx = torch.cat((idx, idx_next), dim=1) # append sampled index to the running sequence and continue

            if gradualOutput:
                next_token_text = decode([idx_next])
                print(next_token_text, end='', flush=True)

        return decode(idx[0].tolist())

def getCognitiveInterpreters(metaPath=None):
    if metaPath is None:
        encoder = tiktoken.get_encoding("gpt2") #gpt2
        encode = lambda s: encoder.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: encoder.decode(l)
    else:
        if os.path.exists(metaPath):
            print(f"Loading meta from {metaPath}...")
            with open(metaPath, 'rb') as file:
                meta = pickle.load(file)
            toStringIndex, indexToString = meta['indexToString'], meta['stringToIndex']
            encode = lambda s: [toStringIndex[c] for c in s]
            decode = lambda l: ''.join([indexToString[i] for i in l])
    return encode, decode

