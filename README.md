# IMPORTANT!
## We created a GPT that will help us build this, it is a private gpt; this repo is also private, so it feels save to share the link here. Just click this link, and you can use headphone more for natural conversation about the project.

## publicCodeMusAIgpt
 A build from scratch implementation of CodeMusai in python, complete with API.

CodeMusai is a new form of AI with changes to traditional components to better mimic human thought process. 
Traditional Attention Transformers are written with mathematical terms, this repository will be written through a psychology a neuroscience lens.
Terms like Thought Processor, Temporal Focus, Cortex, dendrites as input, axons as output, etc. will be used to write the code.

This is a work in progress, and will be updated regularly.
This repository will represent the public portion of the code base.

After implementing all the need-to-have features from my first prototype,
I will produce a new version that uses MLX.

~
CodeMusai is a unique AI, it has two GPT components, one being a LanguageCortex, and the other being an EmotionCortex.
When generating from either logical or emotional corticies a multimodal system message allows for an emotion component it influence logic generation,
and similarly allowing a logical statment to adjust the emotion component being produced; this will be implemented through the use of cross attention.

# ~
# TODO:
# ~ Base
# 1. create base Cortex class and have the others inherit from it
# 2. write cross_attention function, or a stub at least which will be used for inter-corticie communication

# ~ Language
# 3. Implement Language cortex, train based on input text, every so many cycles save a checkpoint, run a text geernaration to test it, and continue to train
# (sample train and sample code will be provided at the bottom of this document.)
# 4. Write function to load universal GGUF weights, and converts their names so it will work with our model.
# 5. load weight for LLAMA3-BioMed, and then train on our custom trianing data [ adjust hyperparameters as you see fit ]
# 5. Test output from bioMedBase with custom additional training
# 6. Write a script to generate a template used for fine tuning
# 7. write a code to accept the fine-tuning data and train on it, and test.

# ~ Vision
# 8 At this stage we will build the build the ViT vision transformer, and train on the CIFAR10 dataset, and test.
# 9. write a script to generate a template used for fine tuning / training on a set of images, which have a class name, and multiple sub-classes/
# 10. Run tests to see if a random document is correctly identified as the classs (and sub-class), eg MRI (or left leg).

# ~ Inter-Cortical Cross Attention giving rise to trOCR.
# 11. too far off at this point, but at this point it can recognize an image, and the text on it
# 12. we would find the patterns to pull key information and file it into the system.

# ~
# At this stage we can test it in a fun way,
# it will also be able to generate text, or images (potentially),
# we could allow a game of telephone, where an iamge is descerirbed, and then from that,
# the image is redrown and then redescribed....  if it does it well it will be able to determine the class
# even on the generated iamges.
# perhaps use a fitness function, to make it so 'when the langauge and visual cortecies work together they now have a loss function and can run training in this form )

# ~


##
###
## General Note:
### We want as few lines of code as possible, that is we want it concise.
### we cant it full, right down to the neuron, but if an internal pyTorch component woudl be faster then have a setting, fullHomeBorn = True would use internal components we made, otherwise it would use components from the pyTorch library. We want to keep the code as simple as possible, and we want to use terms that doctors would understand, psychology terms, neuroscience terms, and comments where useful. In the end, there will be a Cortex which classifies melodies to emotions, and that will be used to add emotion to the system. One day it might have all our senses, and learn to crawl, walk, talk, and be like us.
###
# We don't konw what will happen next, so it should not be hard-coded in anyform, and it should not be limited to the medical field.
###