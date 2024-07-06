from CodeMusai.NeuralCircuitSettings import NeuralCircuitSettings
from CodeMusai.Mind import Mind
import torch
import os.path as path

# Define a sample configuration
class SampleConfig(NeuralCircuitSettings):
    def __init__(self):
        self.vocabSize = 50257
        self.memorySpan = 128
        self.n_embd = 768
        self.n_head = 12
        self.dropout = 0.1
        self.n_layer = 12
        self.bias = True

# Create an instance of the Mind class
config = SampleConfig()
mind = Mind(config)

# Load the random model
mind.loadMemories('gpt2')

#load additional model
thePath = path.join(path.dirname(__file__),'_activeMinds', 'codemusai.pt')
if path.exists(thePath):
    mind.cortices.LanguageCortex.load_model(thePath, config)
    print(f"Model loaded from {thePath}")
else:
    print(f"Model not found at {thePath}")

# Prepare dummy input data
initial_prompt = "Once upon a time"
encode, decode = mind.cortices.LanguageCortex.getCognitiveInterpreters()
initial_input = torch.tensor([encode(initial_prompt)], dtype=torch.long)

# Generate output
gradualOutput = True
if gradualOutput:
    print('')
    print(initial_prompt, end='', flush=True)

mind.eval()
with torch.no_grad():
    generated_output = mind.cortices.LanguageCortex.generate(initial_input, max_new_tokens=50, gradualOutput=gradualOutput, temperature=1.0, top_k=10)

# Print the decoded output
print('')
print('')
print("Generated Output:")
print(generated_output)