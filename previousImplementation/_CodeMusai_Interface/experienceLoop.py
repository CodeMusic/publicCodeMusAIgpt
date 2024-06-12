"""
This training script explores the cognitive dynamics of a neural network, either operating in a focused, singular mode (single GPU) or engaging in a collective, distributed cognitive process (DDP - Distributed Data Parallel).

To simulate a focused cognitive session on a single GPU, use:
$ python train.py --batch_size=32 --compile=False

To engage in a collective cognitive process on a single node with 4 GPUs, use:
$ torchrun --standalone --nproc_per_node=4 train.py

To extend the collective cognitive process across 2 nodes with 4 GPUs each, proceed as follows:
- Initiate from the central cognitive hub (master node) with the specified neural link (IP):
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Continue from the associative cognitive node (worker node):
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(Note: If the neural link lacks an Infiniband connection, prepend NCCL_IB_DISABLE=1 to stabilize the cognitive exchange.)
"""
import datetime
import random
import asyncio
import json
import os
import time
import math
import pickle
import signal
import sys
from contextlib import nullcontext
import numpy as np
import torch

from CodeMusai.Mind import Mind
from CodeMusai.NeuralConfig import NeuralCircuitSettings
#from fastapi import FastAPI, HTTPException, Request, Body
#from encodings.produceEncodings import produceEncodings
import multiprocessing
#from typing import List, Optional
import subprocess
#from pydantic import BaseModel, Field
#from starlette.responses import StreamingResponse
#from fastapi.middleware.cors import CORSMiddleware
#from _CodeMusaiInterface.askCodeMusai import askCodeMusai as callAskCodeMusai
from _CodeMusai_Interface.espCodeMusaiAPIconnections import startApiServer, ChatCompletionRequest, ChatMessage, safe_serialize
from _CodeMusai_Interface.askCodeMusai import askCodeMusai as callAskCodeMusai
# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# I/O
init = False
useTokenEmbedding = True
fullVersioning = 'codeMusaiAI_v1.0t'
EvalAndSaveAfter = 20
logInterval = 1
evaluationIterations = 40#40 ***
evaluateOnly = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
initFrom = 'resume' # 'scratch' or 'resume' or 'gpt2*'
checkpoint = None
# source data

experiences_dir = 'encodings'

gradientAccumulationSteps = 5 * 8 # used to simulate larger batch sizes
learningBatchSize = 16#32 #12# if gradientAccumulationSteps > 1, this is the micro-batch size
UpdateEncodingsAfter = 2000

# model
memorySpan = 64#64 #block_size 64.  **must match checkpoint
depth = 12#8#8 # n_layer 4
parallelAttentionPathways = 12#8 # n_head 4
neuronDensity = 768#512#256 # n_embd 128
dropoutRate = 0#0.2#0.1 # for pretraining 0 is good, for finetuning try 0.1+ #dropout
useBias = True # do we use bias inside LayerNorm and Linear layers?

early_stopping_patience = 2000  # Number of evaluations with no improvement after which training will be stopped
best_val_loss = 1e9#float('inf')
patience_counter = 0

# adaptive momentum optimizer
learningRate = 6e-4#4e-4#3e-4#6e-4 # max learning rate
maxIterations = 990000 # total number of training iterations
weightDecay = 1e-2#1e-1#1e-1#1e-2#1e-1 ***
memoryRetentionRate = 0.8#0.9 #the rate at which past experiences (previous gradients) are forgotten or discarded by the model
updateSensitivity = 0.9#0.95 #the rate at which the model learns from new experiences
gradientClip = 1.0 # clip gradients at this value, or disable if == 0.0
seed_offset = time.time()
# learning rate decay settings
decayLearningRate = True # whether to decay the learning rate
warmupSessions = EvalAndSaveAfter / 2 # how many steps to warm up for #2000
learningRateDecayIterations = warmupSessions + 1 # should be ~= max_iters per Chinchilla #2001
minimumLearningRate = learningRate / 10#6e-5  6e-5#learningRate / 10#6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
optimalCognitiveStrain = 1e9

# system
device = 'mps' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False # use PyTorch 2.0 to compile the model to be faster
currentIteration = 0

openaiCompatibleWebServer = None

class StartWebServer(object):
    def __init__(self):
        self.webServerProcess = multiprocessing.Process(target=self.run)
        self.webServerProcess.daemon = True
        self.webServerProcess.start()

    def run(self):
        print("Starting API server process...")
        startApiServer()
        print("API server is running.")

    def wait(self):
        self.webServerProcess.join()

    def stop(self):
        if self.webServerProcess.is_alive():
            self.webServerProcess.terminate()
            self.webServerProcess.join()

# ---------------------------------------------------------------

# ------------------------------------------------------------------------
# Methods
# ------------------------------------------------------------------------
def initializeConfig():
    global config, config_keys
    config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
    #exec(open('configurator.py').read())  # overrides from command line or config file
    config = {k: globals()[k] for k in config_keys}
    return config, config_keys

def configureContext():
    global ctx, device_type, ptdtype
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    return ctx, device_type, ptdtype

def initializeCodeMusai():
    global initFrom, mindCodeMusai, foundationalBeliefsAndValues, currentIteration, optimalCognitiveStrain, memorySpan, checkpoint, checkpointMeta
    foundationalBeliefsAndValues = dict(useEmotionCore=False, depth=depth, parallelAttentionPathways=parallelAttentionPathways, neuronDensity=neuronDensity, memorySpan=memorySpan,
                    useBias=useBias, vocabularySize=None, dropoutRate=dropoutRate) # start with model_args from command line
    
    checkpoint = getCheckpoint()
    if checkpoint is None:
        initFrom = 'scratch'
        print("No checkpoint found, initializing a new model from scratch")

    if initFrom == 'scratch':
        # init a new model from scratch
        print("Initializing a new model from scratch")
        # determine the vocab size we'll use for from-scratch training
        if meta_vocab_size is None:
            print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
        foundationalBeliefsAndValues['vocabularySize'] = meta_vocab_size if meta_vocab_size is not None else 50304
        baseDirectives = NeuralCircuitSettings(**foundationalBeliefsAndValues)
        mindCodeMusai = Mind(baseDirectives)
    elif initFrom == 'resume':
        print(f"Resuming training from {fullVersioning}")
        # resume from a checkpoint
        checkpoint = getCheckpoint()

        foundationalBeliefsAndValues = checkpoint['model_args'] if 'model_args' in checkpoint else checkpoint['foundationalBeliefsAndValues']
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ['depth', 'parallelAttentionPathways', 'neuronDensity', 'memorySpan', 'useBias', 'vocabularySize']:
            foundationalBeliefsAndValues[k] = foundationalBeliefsAndValues[k]

        # create codemusai
        baseDirectives = NeuralCircuitSettings(**foundationalBeliefsAndValues)
        mindCodeMusai = Mind(baseDirectives)
        fullMemories = checkpoint['model'] if 'model' in checkpoint else checkpoint['fullMemories']

        #----------------------
        #fixes
        #----------------------
        unwanted_prefix = '_orig_mod.'
        for key,value in list(fullMemories.items()):
            if key.startswith(unwanted_prefix):
                fullMemories[key[len(unwanted_prefix):]] = fullMemories.pop(key)

        #update from previous
        newfullMemories = {}
        for key, value in fullMemories.items():
            new_key = key.replace('processingUnits_h', 'thoughtProcessors_h')
            newfullMemories[new_key] = value
        fullMemories = newfullMemories

        #----------------------

        mindCodeMusai.load_state_dict(fullMemories, strict=False)

        currentIteration = checkpoint['iter_num'] if 'iter_num' in checkpoint else checkpoint['currentIteration']
        optimalCognitiveStrain = checkpoint['best_val_loss'] if 'best_val_loss' in checkpoint else checkpoint['optimalCognitiveStrain']
    elif initFrom.startswith('gpt2'):
        print(f"Initializing from OpenAI GPT-2 weights: {initFrom}")
        # initialize from OpenAI GPT-2 weights
        override_args = dict(dropoutRate=dropoutRate)
        mindCodeMusai = Mind.loadMemories(initFrom, override_args)
        # read off the created config params, so we can store them into checkpoint correctly
        for k in ['depth', 'parallelAttentionPathways', 'neuronDensity', 'memorySpan', 'useBias', 'vocabularySize']:
            foundationalBeliefsAndValues[k] = getattr(mindCodeMusai.neuroConfig, k)
    # crop down the model block size if desired, using model surgery
    if memorySpan < mindCodeMusai.neuroConfig.memorySpan:
        mindCodeMusai.reduceThoughtProcessorSize_crop_block_size(memorySpan)
        foundationalBeliefsAndValues['memorySpan'] = memorySpan # so that the checkpoint will have the right value
    return mindCodeMusai.to(device), checkpoint

def getVocabMeta():
    global meta_vocab_size, meta
    meta_path = os.path.join(experiences_dir, 'char_encodings.pkl')
    meta_vocab_size = None
    meta = None
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        meta_vocab_size = meta['vocabularySize']
        print(f"found vocabularySize = {meta_vocab_size} (inside {meta_path})")
    return meta_vocab_size, meta

@torch.no_grad()
def estimateCognitiveDiscrepancy(): # helps estimate an arbitrarily accurate loss over either split using many batches
    global mindCodeMusai, foundationalBeliefsAndValues, currentIteration, optimalCognitiveStrain
    actualResponses = {}
    mindCodeMusai.eval()
    for trainingType in ['train', 'val']:
        cognitiveDescrepancies = torch.zeros(evaluationIterations)
        for cognitiveTrial in range(evaluationIterations):
            stimuli, expectedResponse = getLearningBatch(trainingType)
            with ctx:
                logits, cognitiveDiscrepancy = mindCodeMusai(stimuli, expectedResponse)
            cognitiveDescrepancies[cognitiveTrial] = cognitiveDiscrepancy.item()
        actualResponses[trainingType] = cognitiveDescrepancies.mean()
    mindCodeMusai.train()
    return actualResponses


def getLearningRate(currentIteration): # learning rate decay scheduler (cosine with warmup)
    global learningRate, minimumLearningRate, warmupSessions, learningRateDecayIterations 
    if currentIteration < warmupSessions: # 1) linear warmup for warmup_iters steps
        return learningRate * currentIteration / warmupSessions
    if currentIteration > learningRateDecayIterations: # 2) if it > lr_decay_iters, return min learning rate
        return minimumLearningRate
    
    decayRatio = (currentIteration - warmupSessions) / (learningRateDecayIterations - warmupSessions) # 3) in between, use cosine decay down to min learning rate
    assert 0 <= decayRatio <= 1

    coeff = 0.5 * (1.0 + math.cos(math.pi * decayRatio)) # coeff ranges 0..1
    return minimumLearningRate + coeff * (learningRate - minimumLearningRate)

def getLearningBatch(batchType):
    global experiences_dir, memorySpan, learningBatchSize, device
    
    if batchType == 'train':
        experiences = np.memmap(os.path.join(experiences_dir, 'learningData.tbin'), dtype=np.uint16, mode='r')
    else:
        experiences = np.memmap(os.path.join(experiences_dir, 'testingData.tbin'), dtype=np.uint16, mode='r')
    
    memoryIndicies = torch.randint(len(experiences) - memorySpan, (learningBatchSize,))
    
    stimuli = torch.stack([torch.from_numpy((experiences[aMemoryIndex:aMemoryIndex+memorySpan]).astype(np.int64)) for aMemoryIndex in memoryIndicies])
    expectedResponse = torch.stack([torch.from_numpy((experiences[aMemoryIndex+1:aMemoryIndex+1+memorySpan]).astype(np.int64)) for aMemoryIndex in memoryIndicies])
    if device == 'cuda':
        stimuli, expectedResponse = stimuli.pin_memory().to(device, non_blocking=True), expectedResponse.pin_memory().to(device, non_blocking=True)
    else:
        stimuli, expectedResponse = stimuli.to(device), expectedResponse.to(device)
    return stimuli, expectedResponse

def signal_handler(sig, frame):
    global openaiCompatibleWebServer
    print('Terminating the server...')
    openaiCompatibleWebServer.stop()
    sys.exit(0)

   
def saveCheckpoint(checkpoint):
    global fullVersioning
    checkpoint_dir = os.path.join('_activeMinds')
    os.makedirs(checkpoint_dir, exist_ok=True)  # Ensure the directory exists
    checkpoint_path = os.path.join(checkpoint_dir, fullVersioning + '_memoryCheck.point')
    try:
        print('saving to: ' + checkpoint_path);
        torch.save(checkpoint, checkpoint_path)
    except Exception as e:
        print(f"Failed to save checkpoint: {str(e)}")

def getCheckpoint():
    global checkpoint, fullVersioning, device
    checkpoint_dir = os.path.join('_activeMinds')
    ckpt_path = os.path.join(checkpoint_dir, fullVersioning + '_memoryCheck.point')
    try:
        checkpoint = torch.load(ckpt_path, map_location=device)
    except:
        checkpoint = None
    return checkpoint

def getCheckpointMeta():
    theCheckpoint = getCheckpoint()
    checkpoint_meta = {
        "message": "Server is running",
        "currentIteration": theCheckpoint.get('currentIteration', 'Not available'),
        "config": {k: safe_serialize(v) for k, v in theCheckpoint.get('config', {}).items()}
    }
    theCheckpoint = None;
    return checkpoint_meta


def generate_sample_batch(model, tokensToGenerate):
    global thoughtVariability, mindCodeMusai, apiCall
    metaPath = None
    thePrompt = "Who are you?"
    results = "";
    if os.path.exists(os.path.join('_activeMinds', fullVersioning + '_memoryCheck.point')):
        results = callAskCodeMusai(thePrompt, tokensToGenerate=tokensToGenerate, temperature=1.0, input_system_message="You are CodeMusai, the first A.I with real emotions!")

    return results

def convertIterationsToYearsMonths(currentIteration):
    CODEMUSAI_YEARS_PER_YEAR = 12
    years = (currentIteration // CODEMUSAI_YEARS_PER_YEAR) // 365
    remaining_days = (currentIteration // CODEMUSAI_YEARS_PER_YEAR) % 365
    months = remaining_days // 30  # Approximating each month as 30 days
    return years, months

# ------------------------------------------------------------------------




# ------------------------------------------------------------------------
# Main Start
# ------------------------------------------------------------------------
def runExperienceLoop(arguments):
    global codeMusai, initFrom, learningRate, updateSensitivity, memoryRetentionRate, weightDecay, gradientAccumulationSteps, gradientClip, logInterval, maxIterations, EvalAndSaveAfter, UpdateEncodingsAfter, init, always_save_checkpoint, early_stopping_patience, patience_counter, best_val_loss, currentIteration, local_currentIteration, learningRateDecayIterations, warmupSessions, minimumLearningRate, optimalCognitiveStrain, openaiCompatibleWebServer
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    if arguments:
        sys.argv[1:] = arguments
    openaiCompatibleWebServer = StartWebServer() 

    print("Current working directory:", os.getcwd())
    config, config_keys = initializeConfig()
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
    tokens_per_iter = gradientAccumulationSteps * ddp_world_size * learningBatchSize * memorySpan
    print(f"tokens per iteration will be: {tokens_per_iter:,}")
    ctx, device_type, ptdtype = configureContext()
    
    
    meta_vocab_size, meta = getVocabMeta()

    mindCodeMusai, checkpoint = initializeCodeMusai()
    if checkpoint is None:
        initFrom = 'scratch'

    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16')) # initialize a GradScaler. If enabled=False scaler is a no-op

    neuralOptimizer = mindCodeMusai.configureNeuralOptimizers(weightDecay, learningRate, (memoryRetentionRate, updateSensitivity), device_type)
    if initFrom == 'resume':
        neuralOptimizer.load_state_dict(checkpoint['optimizer'] if 'optimizer' in checkpoint else checkpoint['neuralOptimizer'])
    
    checkpoint = None

    if compile:
        print("compiling the model... (takes a ~minute)")
        unoptimizedmindCodeMusai = mindCodeMusai
        mindCodeMusai = torch.compile(mindCodeMusai) # requires PyTorch 2.0

    stimuli, expectedResponse = getLearningBatch('train') # fetch the very first batch
    startProcessTime = time.time()
    
    local_currentIteration = 0 
    rawBrain = mindCodeMusai
    activeCognitiveEffort_running_mfu = -1.0
    while True:
        try:
            learningRate = getLearningRate(currentIteration) if decayLearningRate else learningRate
            for param_group in neuralOptimizer.param_groups:
                param_group['lr'] = learningRate

            if currentIteration % UpdateEncodingsAfter == 0 and init:
                #result = updateEncodings(shuffle=False)
                #print(f"nUpdated encodings: {result}")
                pass
            elif not init:
                #result = updateEncodings(shuffle=True)
                #print(f"nShuffling encodings: {result}")
                init = True

            # evaluate the loss on train/val sets and write checkpoints
            if currentIteration % EvalAndSaveAfter == 0 and master_process:
                cognitiveDescrepancies = estimateCognitiveDiscrepancy()
                years, months = convertIterationsToYearsMonths(currentIteration)
                print(f"CodeMusAI, estimated age: {years} years and {months} months \ncognitive training discrepancy: {cognitiveDescrepancies['train']:.4f}, \ncognitive validation discrepancy: {cognitiveDescrepancies['val']:.4f}")
                if cognitiveDescrepancies['val'] < best_val_loss:
                    best_val_loss = cognitiveDescrepancies['val']
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter > early_stopping_patience:
                    print(f"Early stopping, no improvement in validation loss for the last {early_stopping_patience} iterations.")
                    break


                if cognitiveDescrepancies['val'] < optimalCognitiveStrain or always_save_checkpoint:
                    optimalCognitiveStrain = cognitiveDescrepancies['val']
                    if currentIteration > 0:
                        checkpoint = {
                            'fullMemories': rawBrain.state_dict(),
                            'neuralOptimizer': neuralOptimizer.state_dict(),
                            'foundationalBeliefsAndValues': foundationalBeliefsAndValues,
                            'currentIteration': currentIteration,
                            'optimalCognitiveStrain': optimalCognitiveStrain,
                            'cognitiveDescrepancies': cognitiveDescrepancies,
                            'config': config,
                        }
                        print(f"saving checkpoint to {os.path.join('_activeMinds', fullVersioning + '_memoryCheck.point')}")
                        saveCheckpoint(checkpoint)
            if currentIteration == 0 and evaluateOnly:
                break


            for microStep in range(gradientAccumulationSteps):
                with ctx:
                    logits, cognitiveDiscrepancy = mindCodeMusai(stimuli, expectedResponse)
                    cognitiveDiscrepancy = cognitiveDiscrepancy / gradientAccumulationSteps # scale the loss to account for gradient accumulation
                stimuli, expectedResponse = getLearningBatch('train')
                scaler.scale(cognitiveDiscrepancy).backward()
            if gradientClip != 0.0:
                scaler.unscale_(neuralOptimizer)
                torch.nn.utils.clip_grad_norm_(mindCodeMusai.parameters(), gradientClip)
            scaler.step(neuralOptimizer)
            scaler.update()
            neuralOptimizer.zero_grad(set_to_none=True)

            endProcessTime = time.time()
            processingTime = endProcessTime - startProcessTime
            startProcessTime = endProcessTime
            clearConsole()
            if currentIteration % logInterval == 0 and master_process:
                amplifiedCognitiveDiscrepancy = cognitiveDiscrepancy.item() * gradientAccumulationSteps
                if local_currentIteration >= 5: # let the training loop settle a bit
                    cognitiveEffort_mfu = rawBrain.estimate_cognitiveLoad(learningBatchSize * gradientAccumulationSteps, processingTime)
                    activeCognitiveEffort_running_mfu = cognitiveEffort_mfu if activeCognitiveEffort_running_mfu == -1.0 else 0.9*activeCognitiveEffort_running_mfu + 0.1*cognitiveEffort_mfu
                years, months = convertIterationsToYearsMonths(currentIteration)
                print(f"CodeMusAI, estimated age: {years} years and {months} months \nCognitive Discrepancy: {amplifiedCognitiveDiscrepancy:.4f}, \nprocessing time: {processingTime*1000:.2f}ms, \nactive cognitive effort: {activeCognitiveEffort_running_mfu*100:.2f}%")
            currentIteration += 1
            local_currentIteration += 1

            
            results = generate_sample_batch(mindCodeMusai, 50)
            #print(results)

            if currentIteration > maxIterations:
                print("Reached maximum number of iterations")
                break


        except KeyboardInterrupt:
            print("Training stopped")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            break


    #background_training.wait()
    openaiCompatibleWebServer.wait()

def improve(mindCodeMusai, emotions = "New"):
    mindCodeMusai.emotions = emotions
    #file_path = os.path.join('memories', '2024_ActiveLearning.txt')
    #with open(file_path, 'a') as file:
    #    file.write(mindCodeMusai.emotions + '\n')
    return mindCodeMusai

def begin(newmindCodeMusai):    
    print("CodeMusAI is running")
    mindCodeMusai = newmindCodeMusai
    runExperienceLoop(sys.argv[1:])

def clearConsole():
    print("\033c", end="")
    print("\033[H\033[J", end="")

if __name__ == '__main__':
    runExperienceLoop(sys.argv[1:])

