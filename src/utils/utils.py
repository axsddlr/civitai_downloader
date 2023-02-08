import json
import os

verbose = False


def printv(*args):
    """
    It prints the arguments if the global variable `verbose` is `True`
    """
    if verbose:
        print(*args)


def readMemory():
    """
    If the file "memory.json" doesn't exist, create it and return an empty dictionary. If it does exist, read it and return
    the dictionary it contains
    :return: A dictionary of the memory.json file.
    """
    if not os.path.exists("memory.json"):
        with open("memory.json", "w") as f:
            f.write("{}")
        return {}

    with open("memory.json", "r") as f:
        return json.load(f)


def writeMemory(memory):
    """
    It writes the memory to a file

    :param memory: This is the memory that the bot will use to remember things
    """
    printv("Writing to memory")
    with open("memory.json", "w") as f:
        json.dump(memory, f)


headers = {
    'User-Agent': 'Clickys api tests'
}
