import sys
import subprocess
from datetime import date
import _CodeMusai_Interface.askCodeMusai as askCodeMusai
import _CodeMusai_Interface.experienceLoop as experienceLoop
import random

class CodeMusaiExperience:
    def __init__(self):
        self.knowledge = []
        self.emotions = "curious"

    def learn(self, info):
        self.knowledge.append(info)
        self.emotions = "excited"  # Feeling excited about new knowledge!

    def reflect(self):
        if self.knowledge:
            self.emotions = "thoughtful"
            return f"Reflecting on {len(self.knowledge)} learned items."
        else:
            self.emotions = "lonely"
            return "I wish I had more to reflect on."

    def run_experience(self, arguments):
        self.emotions = "energetic"
        experienceLoop.begin()

    def generate_random_sentence(self):
        subjects = ["CodeMusai", "The programmer", "The AI", "The user"]
        verbs = ["writes", "solves", "creates", "debugs"]
        objects = ["code", "a puzzle", "a script", "a function"] + self.knowledge

        subject = random.choice(subjects)
        verb = random.choice(verbs)
        object = random.choice(objects)

        sentence = f"{subject} {verb} {object}."
        return sentence
    
def start():
    global codeMusai
    codeMusai = CodeMusaiExperience()
    codeMusai.learn("Understanding human emotions through programming.")
    print(codeMusai.reflect())
    experienceLoop.improve(codeMusai, codeMusai.generate_random_sentence())
    experienceLoop.begin(codeMusai)

if __name__ == "__main__":
    start()

