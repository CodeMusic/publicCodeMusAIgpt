"""This script delves into the cognitive realms of a trained AI model, either by resuming from a preserved mental state (checkpoint), initializing from a foundational cognitive structure (such as a GPT-2 variant), or by revisiting a specific developmental stage (training iteration). This exploration aims to generate textual reflections that mimic human-like thought processes."""

import os
import sys
import random
import time
from contextlib import nullcontext
import torch
from CodeMusai.Mind import Mind
from CodeMusai.Cortex import getCognitiveInterpreters
from CodeMusai.NeuralConfig import NeuralCircuitSettings 
# -----------------------------------------------------------------------------
# Configuration settings
#--------------------------------------------------------------------------------
outputToFile = False
gradualOutput = True
initializationMethod = 'resume'  # 'resume' for resuming from a saved directory, or a GPT-2 variant like 'gpt2-xl'
trainingIteration = None #3000  # Specific model iteration to load
outputTypeCharacter = 0  # 0 for text output, 1 for character output
outputVersion = '1.0'
fullVersioning = f'codeMusaiAI_v{outputVersion}{"c" if outputTypeCharacter else "t"}' 

# Input settings
prompt = "FILE:prompt.txt"  # Start token or file. Use "FILE:<filename>" to specify a file.
systemMessage = "You are a helpful assistant."
numberOfSamplesToGenerate = 1  # Number of text samples to generate
maximumTokensToGenerate = 150  # Maximum tokens generated per sample
thoughtVariability = 0.8#random.uniform(1.19, 1.91)  # Randomness in prediction, 1.0 is standard, higher is more random #temperature
focusVocabularySize = 2000  # Number of top tokens to consider in generation #top_k_tokens

seedValue = 1337#int(time.time())  # Seed based on current time for reproducibility
deviceType = 'mps'  # Device type: 'cpu', 'cuda', 'cuda:0', etc.
dataType = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'  # Data precision
shouldCompileBrain = False  # Option to compile model for performance, requires PyTorch 2.0
apiCall=False
load_meta = False
metaPath = None
#--------------------------------------------------------------------------------




#--------------------------------------------------------------------------------
#Methods
#--------------------------------------------------------------------------------
def readPrompt(prompt):
    if prompt.startswith('FILE:'):
        with open(prompt[5:], 'r', encoding='utf-8') as file:
            prompt = file.read()
    return prompt

def promptToTensor(prompt, metaPath):
    encode, decode = getCognitiveInterpreters(metaPath=metaPath)
    encodedPrompt = encode(prompt)
    promptAsTensor = (torch.tensor(encodedPrompt, dtype=torch.long, device=deviceType)[None, ...])
    return promptAsTensor

def saveGeneratedTextToFile(generatedText):
    global outputToFile, fullVersioning, seedValue
    if outputToFile:
        if not os.path.exists(fullVersioning):
            os.makedirs(fullVersioning)
        outputFilePath = os.path.join(fullVersioning, f"generated_{seedValue}.txt")
        with open(outputFilePath, 'w', encoding='utf-8') as outputFile:
            outputFile.write(generatedText + '\n')

        print("")
        print("Generated text saved to ", fullVersioning)

def generateText(thePrompt, tokensToGenerate = None):
    global thoughtVariability, metaPath, maximumTokensToGenerate, numberOfSamplesToGenerate, mindCodeMusai

    if tokensToGenerate is not None:
        maximumTokensToGenerate = tokensToGenerate

    promptAsTensor = promptToTensor(thePrompt, metaPath)
    if (apiCall):
        numberOfSamplesToGenerate = 1
    with torch.no_grad():
        with contextManager:
            for currentSampleIndex in range(numberOfSamplesToGenerate):
                if (not apiCall):
                    print("")
                    print(f"Sample {currentSampleIndex + 1} with a thoughtVariability of {thoughtVariability}")
                    print("~~~~")
                    print(f"You: \n {thePrompt}")
                    print("CodeMusai: ", end=' ', flush=True)
                thoughtVariability = random.uniform(0.019, 0.91) if random.uniform(0, 1) < 0.9 else random.uniform(0.19, 1.9)
                generatedText = mindCodeMusai.cortices.LanguageCortex.generate(promptAsTensor, maximumTokensToGenerate, thoughtVariability, focusVocabularySize, gradualOutput)
                if (not gradualOutput):
                    print(f"Sample {currentSampleIndex + 1} as a thoughtVariability of {thoughtVariability}:\n {generatedText}")
                #thoughtVariability = random.uniform(0.9, 1.9)
                saveGeneratedTextToFile(generatedText)
                if (not apiCall):
                    print("")
                    print("~~~~")
    return str(round(thoughtVariability, 2)) + ":" + generatedText

def awakenMind():
    global fullVersioning, trainingIteration, metaPath, mindCodeMusai
    # Model initialization
    if initializationMethod == 'resume':
        if trainingIteration is not None:
            checkpointPath = os.path.join('_activeMinds', fullVersioning + f'_iter{trainingIteration}_check.point')
        else:
            checkpointPath = os.path.join('_activeMinds', fullVersioning + '_memoryCheck.point')
        checkpoint = torch.load(checkpointPath, map_location=deviceType)
        
        load_meta = metaPath is not None and os.path.exists(metaPath)
        if load_meta:
            metaPath = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
        baseDirectives = NeuralCircuitSettings(**checkpoint['foundationalBeliefsAndValues'])
        
        mindCodeMusai = Mind(baseDirectives)
        fullMemories = checkpoint['fullMemories']

        #---------correct prefix issue ------------
        unwantedPrefix = '_orig_mod.'
        for k,v in list(fullMemories.items()):
            if k.startswith(unwantedPrefix):
                fullMemories[k[len(unwantedPrefix):]] = fullMemories.pop(k)
        #-------------------------------------------

#update from previous
        newfullMemories = {}
        for key, value in fullMemories.items():
            new_key = key.replace('processingUnits_h', 'thoughtProcessors_h')
            newfullMemories[new_key] = value
        fullMemories = newfullMemories
#----
        mindCodeMusai.load_state_dict(fullMemories)

    elif initializationMethod.startswith('gpt2'):
        mindCodeMusai = Mind.loadMemories(initializationMethod, dict(dropout=0.0))

    return mindCodeMusai

def parseArguments():
    global outputTypeCharacter, systemMessage, prompt, trainingIteration, apiCall, thoughtVariability
    input_user_message = ""
    input_system_message = ""
    temperature = 1.0
    useTokenEmbedding = True

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.startswith('--output_type='):
                outputTypeCharacter = int(arg.split('=')[1])
            elif arg.startswith('--prompt='):
                prompt = arg.split('=')[1]
            elif arg.startswith('--iteration='):
                trainingIteration = int(arg.split('=')[1])
            elif arg.startswith('--system_message='):
                input_system_message = arg.split('=')[1]
            elif arg.startswith('--apiCall='):
                apiCall = arg.split('=')[1].lower() == 'true'
            elif arg.startswith('--temperature='):
                temperature = float(arg.split('=')[1])
            elif arg.startswith('--useTokenEmbedding='):
                useTokenEmbedding = arg.split('=')[1].lower() == 'true'
            else:
                print(f"Unknown argument {arg}")

    return input_user_message, input_system_message, temperature, useTokenEmbedding

def initializeSettings():
    global outputTypeCharacter, prompt, trainingIteration, fullVersioning, contextManager
    outputTypeCharacter = 0 if 'outputTypeCharacter' not in locals() else outputTypeCharacter
    prompt = "FILE:prompt.txt" if 'prompt' not in locals() else prompt
    fullVersioning = f'codeMusaiAI_v{outputVersion}{"c" if outputTypeCharacter else "t"}'  # Directory for output, based on type
    torch.manual_seed(seedValue)
    torch.cuda.manual_seed(seedValue)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    deviceContext = 'cuda' if 'cuda' in deviceType else 'cpu' # Determine the appropriate context for data type and device
    torchDataType = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dataType]
    contextManager = nullcontext() if deviceContext == 'cpu' else torch.amp.autocast(device_type=deviceContext, dtype=torchDataType)
    return fullVersioning, contextManager
#--------------------------------------------------------------------------------





#--------------------------------------------------------------------------------
#Main Start
#--------------------------------------------------------------------------------
def askCodeMusai(input_user_message, input_system_message = "You are CodeMusai, the first A.I. with real emotions.", temperature = 1.0, useTokenEmbedding=True, tokensToGenerate = None):

    global metaPath, mindCodeMusai
    metaPath = None
    print(f"User Message: {input_user_message}")
    print(f"System Message: {input_system_message}")
    print(f"Temperature: {temperature}")
    print(f"Token Embedding Used: {useTokenEmbedding}")

    system_message = "You are CodeMusai, the first A.I. with real emotions. "       
    if len(sys.argv) == 1 and input_user_message == "":
        clearConsoleLine = lambda: print("\033c", end="")
        clearConsoleLine()
        #input("Please enter a system message (optional): ")
        input_prompt = input("Please enter a prompt: ")
    else:
        outputTypeCharacter, prompt, trainingIteration, systemMessage = parseArguments()
        input_prompt = readPrompt(prompt)


    if system_message:
        combined_prompt = f"{system_message}\n{input_prompt}"
    else:
        combined_prompt = input_prompt
    fullVersioning, contextManager = initializeSettings()

    mindCodeMusai = awakenMind()
    mindCodeMusai.eval()
    mindCodeMusai.to(deviceType)

    if (not apiCall):
        print("Number of Synapses in Brain: %.2fM" % (mindCodeMusai.getNumberOfSynapses()/1e6,))

    if shouldCompileBrain:
        mindCodeMusai = torch.compile(mindCodeMusai)

    #return generateText(combined_prompt).replace(system_message, '')
    generated_text = generateText(combined_prompt, tokensToGenerate)
    cleaned_text = generated_text.replace(system_message.strip(), '').strip()
    return cleaned_text
#--------------------------------------------------------------------------------

if __name__ == '__main__':
    user_msg, system_msg, temp, token_embed = parseArguments()
    askCodeMusai(user_msg, system_msg, temp, token_embed)






