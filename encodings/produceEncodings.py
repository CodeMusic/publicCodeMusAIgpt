import tiktoken
from consolidateMemories import gatherMemories
import os
import numpy as np
import pickle

#import sentencepiece as sp  # add google sentencepiece as an encoding type

rehearsalProportion = 0.9

def generateTikTokenEncodings(train_data, val_data):
    encoder = tiktoken.get_encoding("gpt2") 

    learningIndices = encoder.encode_ordinary(train_data)
    testingIndices = encoder.encode_ordinary(val_data)

    saveEncodings(learningIndices, os.path.dirname(__file__), 'learningData.tbin')
    saveEncodings(testingIndices, os.path.dirname(__file__), 'testingData.tbin')

    print(f"");
    print(f"[Token encoding]");
    print(f"Learning Tokens: {len(learningIndices):,} ~ Testing Tokens: {len(testingIndices):,}")

def generateCharacterEncodings(train_data, val_data):
    vocabulary = sorted(list(set(train_data + val_data)))
    stringToIndex = {ch: i for i, ch in enumerate(vocabulary)} 
    indexToString = {i: ch for i, ch in enumerate(vocabulary)} 

    learningIndices = [[stringToIndex[aCharacter] for aCharacter in aString] for aString in train_data] 
    testingIndices = [[stringToIndex[aCharacter] for aCharacter in aString] for aString in val_data] 

    saveEncodings(learningIndices, os.path.dirname(__file__), 'learningData.cbin') 
    saveEncodings(testingIndices, os.path.dirname(__file__), 'testingData.cbin')

    meta = {'vocabularySize': len(vocabulary), 'indexToString': indexToString, 'stringToIndex': stringToIndex} 
    with open(os.path.join(os.path.dirname(__file__), 'char_encodings.pkl'), 'wb') as f:
        pickle.dump(meta, f)

    print(f"");
    print(f"[Character encoding]");
    print(f"Found Vocabulary: {''.join(vocabulary)}")
    print(f"Vocabulary Size: {len(vocabulary):,} ~ Learning Tokens: {len(learningIndices):,} ~ Testing Tokens: {len(testingIndices):,}")


def saveEncodings(encoding_ids, directory, filename):
    
    encoding_ids = np.array(encoding_ids, dtype=np.uint16)
    encoding_ids.tofile(os.path.join(directory, filename))
    print(f"Saved {len(encoding_ids):,} tokens to {os.path.join(directory, filename)}")


def produceEncodings(shuffle=False):
    rawMemories = gatherMemories(shuffle)

    memoryLength = len(rawMemories)
    learningData = rawMemories[:int(memoryLength*rehearsalProportion)] 
    testingData = rawMemories[int(memoryLength*rehearsalProportion):] 
    generateTikTokenEncodings(learningData, testingData)
    generateCharacterEncodings(learningData, testingData)


if __name__ == "__main__":
    produceEncodings()

