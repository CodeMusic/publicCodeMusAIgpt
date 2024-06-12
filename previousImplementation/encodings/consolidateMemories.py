import os
import random
import pickle
import argparse


def gatherMemories(shuffle):
    #shuffle is experimental, may cause overfitting due to val data peeking

    directoryPath = "encodings"
    allMemories = ""
    fileList = os.listdir(directoryPath + '/memories')
    if shuffle:
        print("Shuffling files...")
        random.shuffle(fileList)  # Shuffle the list of files randomly
    else:
        print("Sorting files...")
        fileList = sorted(fileList)

    for fileName in fileList:
        if fileName.endswith(".txt"):
            filePath = os.path.join(directoryPath + '/memories', fileName)
            with open(filePath, 'r', encoding='utf-8') as file:
                allMemories += file.read() + "\n" #grab all text files in memories, and add them to allMemories separated by newlines

    print("Writing combined memory...")
    with open(os.path.join(directoryPath, 'combined.txt'), 'w', encoding='utf-8') as file:
        file.write(allMemories)
    print(f"Total characters: {len(allMemories)}")
    print("Done!")
    return allMemories



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process and encode text data.")
    parser.add_argument('-shuffle', action='store_true', help='Shuffle the files before processing')
    args = parser.parse_args()

    gatherMemories(args.shuffle)

