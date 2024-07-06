from CodeMusai.CognitiveClarifier import CognitiveClarifier
from CodeMusai.ThoughtProcessor import ThoughtProcessor
from CodeMusai.Cortex import Cortex
from CodeMusai.NeuralCircuitSettings import NeuralCircuitSettings
from CodeMusai.Mind import Mind
import torch
import os.path as path

from torch.utils.data import DataLoader, Dataset

# Define a sample configuration
class SampleConfig(NeuralCircuitSettings):
    def __init__(self):
        self.vocabSize = 50257
        self.memorySpan = 128
        self.n_embd = 768
        self.n_head = 12
        self.dropout = 0
        self.n_layer = 12
        self.bias = True

# Sample dataset
class RandomDataset(Dataset):
    def __init__(self, vocab_size, sequence_length, num_samples):
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_data = torch.randint(0, self.vocab_size, (self.sequence_length,))
        target_data = input_data.clone()
        return input_data, target_data

# Create an instance of the Mind class
config = SampleConfig()
mind = Mind(config)

# Load the random model
mind.loadMemories('gpt2')

#load additional model
thePath = path.join(path.dirname(__file__),'_activeMinds', 'codemusai.pt')
if path.exists(thePath):
    mind.cortices.LanguageCortex.load_model(thePath, config)
    print(f"Previous model loaded from {thePath}")
else:
    print(f"Previous model not found at {thePath}")


# Prepare the dataset and dataloader
dataset = RandomDataset(config.vocabSize, config.memorySpan, 1000)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Configure the optimizer
weight_decay = 0.01
learning_rate = 3e-4
betas = (0.9, 0.95)
device_type = 'cuda' if torch.cuda.is_available() else 'mps'

optimizer = mind.configureNeuralOptimizers(weight_decay, learning_rate, betas, device_type)

# Training loop
mind.to(device_type)
mind.train()

num_epochs = 1000
for epoch in range(num_epochs):
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device_type), targets.to(device_type)
        optimizer.zero_grad()
        logits, loss = mind(inputs, targets)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")
            thePath = path.join(path.dirname(__file__),'_activeMinds', 'codemusai.pt')
            mind.cortices.LanguageCortex.save_model(thePath)    
            print(f"Checkpoint saved to {thePath}")

#save model
thePath = path.join(path.dirname(__file__),'_activeMinds', 'codemusai.pt')
mind.cortices.LanguageCortex.save_model(thePath)
print("Training complete.")


"""

from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math


@dataclass
class CodeMusaiConfig:
    block_size: int = 1024#memorySpan#256 #this is the context awareness size, traditionally block_size
    vocab_size: int = 50257#vocabularySize#65 #this is the vocabulary size, traditionally vocab_size
    n_layer: int = 12#depth 6 #this is the n_layer of the model, the number of hidden layers, traditionally n_layer
    n_head: int = 12#n_head#6 #this is the awareness node size, or the number of heads of attention, traditionally n_head
    n_embd: int = 768#conceptualDensity#384 #this is the conceptual density, the ability to embed concepts in a multimension space in relations to the tokens, traditionally n_embd
    dropout: float = 0.0 #this is the dropout rate, traditionally dropout
    bias: bool = True #this is the bias, traditionally bias

class CognitiveClarity(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)



class TemporalFocus(nn.Module):

    def __init__(self, config: CodeMusaiConfig):
        super().__init__()
        #self.config = config
        assert config.n_embd % config.n_head == 0

        #key, quests, values projects for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        #output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        #regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout


        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

        #bias mask following openai/hf names so we can load them
        #self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
        #                                 .view(1, 1, config.block_size, config.block_size))
    
    def forward(self, x):

        B, T, C = x.size() #batch size, sequence length, and embedding dimensionality (n_embd)
        #calcualtes query, key, and values for all heads in batch and move head forward to be the batch
        #nh is 'number of heads', hs is 'head size', and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, and so nh*hs=C=12*64=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=-1)
        #q, k, v = qkv.chunk(3, dim=-1)
        #calculate attention scores ('affinities') by applying the bias and then applying the causal mask
        #affinities = (q @ k.transpose(-2, -1)) * (1 / math.sqrt(hs)) + bias
        #affinities = q @ k.transpose(-2, -1) * (1 / math.sqrt(hs)) + self.bias

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            #attention scores ('affinities') by applying the bias and then applying the causal mask
            #affinities = (q @ k.transpose(-2, -1)) * (1 / math.sqrt(hs)) + bias
            #affinities = q @ k.transpose(-2, -1) * (1 / math.sqrt(hs)) + self.bias
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) #+ self.bias
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = att.softmax(att, dim=-1)
            y = self.attn_dropout(y)
            y = att @ v #(B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C) #re-align nh and hs

        #output projection
        y = self.c_proj(y)
        y = self.resid_dropout(y)
        return y
        #att = self.attn_drop(att)
        

class NeuralNetwork(nn.Module):
    def __init__(self, config: CodeMusaiConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)


    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class ThoughtProcessor(nn.Module):
    
    def __init__(self, config: CodeMusaiConfig):
        super().__init__()
        self.config = config

        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = TemporalFocus(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = NeuralNetwork(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Cortex(nn.Module):

    def __init__(self, config: CodeMusaiConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([ThoughtProcessor(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
            ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        #self.embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def forward(self, idx):
        #idx is of shape (B,T)
        B, T  = idx.shape
        assert T <= self.config.block_size, f"input sequence length is too long: {T} > {self.config.block_size}"
        
        #forward the token and positional embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) #shape (T)
        pos_emb = self.transformer.wpe(pos) #positional embedding of shape (T,n_embd)
        tok_emb = self.transformer.wte(idx) #token embedding of shape (B,T,n_embd)
        x = tok_emb + pos_emb #add the positional embedding to the token embedding
        x = self.transformer.drop(x) #dropout
        for layer in self.transformer.h:
            x = layer(x) #block layer

        #forward the final layernorm and the classifier
        x = self.transformer.ln_f(x) #final layer norm
        logits = self.lm_head(x) #(B,T,vocab_size)#output projection (classifier)
        return logits


    def get_num_params(self, non_embedding=True):
-
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
-
        
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    
    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = CodeMusaiConfig(**config_args)
        model = Cortex(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()



        

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        #print(sd_keys_hf) #this one has 149
        #print(sd_keys) #this one has 150
        #what is the missing key?
        #print(set(sd_keys))
        #print(set(sd_keys_hf))
        #assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


#-------------
num_return_sequences = 1
max_length = 150
topK = 50

#detect device
device = 'cpu' 
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
print(f"using device: {device}")

model = Cortex.from_pretrained('gpt2')
#model = Cortex(CodeMusaiConfig())
print('model loaded')
model.eval()
model.to(device)
#print(model)

#prefix
import tiktoken
enc = tiktoken.encoding_for_model('gpt2')
tokens = enc.encode('Hello, I am CodeMusai,') #Hello, I am CodeMusai,
tokens = torch.tensor(tokens, dtype=torch.long) #(8,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) #(5,8)
x = tokens.to(device)
#print(tokens)
#print(enc.decode(tokens))

#generate! rn x is (B,T) where B=5, T=8
#set seed to 42
#torch.manual_seed(42)
#torch.cuda.manual_seed(42)

while x.size(1) < max_length:
    #forward the model to get the logits
    with torch.no_grad():
        logits = model(x)
        #get the last token
        last_token = logits[:, -1, :]
        #get the probability distribution
        probs = F.softmax(last_token, dim=-1)

        #do top-k sampling of 50
        topk_probs, topk_indices = torch.topk(probs, topK, dim=-1)

        #sample a token from top-k probabilities
        ix = torch.multinomial(topk_probs, num_samples=1) #(B,1)
        #gather the corresponding indices
        xcol = torch.gather(topk_indices, 1, ix) #(B,1)
        #add/append the token to the sequence
        x = torch.cat([x, xcol], dim=1)

for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print('~~~~')
    print(">", decoded)
    print('~~~~')
"""