import sys
import subprocess
from _CodeMusai_Interface.askCodeMusai import askCodeMusai

def callAskCodeMusai():
    # Retrieve arguments excluding the script name itself
    arguments = sys.argv[1:]
    # Prepare the command to call the other script with the arguments
    askCodeMusai(arguments)

if __name__ == "__main__":
    callAskCodeMusai()