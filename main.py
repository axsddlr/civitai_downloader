import json
import os
import re
import threading

import requests
from tqdm import tqdm

from src.downloader import downloadFile
from src.utils.utils import headers
from src.utils.utils import readMemory, writeMemory, printv


def get_model_id():
    """
    It checks if the file `id.txt` exists, if it does not exist, it creates it, if it exists, it reads the file and returns
    a list of all the numbers in the file.
    :return: A list of all the model ids.
    """
    # Checking if the file id.txt exists. If it does not exist, it will create it.
    if not os.path.exists("id.txt"):
        with open("id.txt", "w") as f:
            f.write("")
            print("id.txt file created, run the script again")
    else:
        with open("id.txt", "r") as f:
            # Getting all the numbers from the file `id.txt` and putting them in a list.
            model_id_list = [re.search("\d+", line.strip()).group() for line in f if line.strip()]
        return model_id_list


# Define the list only once
aListOfAllDownloadedModels = []

# Call the readMemory function to get updated values
updatedList = readMemory()

# Update the list with the updated values
aListOfAllDownloadedModels = updatedList


def getModels(session):
    """
    It takes a session object and returns a list of model objects

    :param session: the session object that you created earlier
    :return: A list of dictionaries.
    """
    modelIDs = get_model_id()
    models = []
    for modelID in modelIDs:
        modelR = session.get(f"https://civitai.com/api/v1/models/{str(modelID)}", headers=headers)
        models.append(json.loads(modelR.text))
    # print(models)
    return models


session_downloadedFileCount = 0
session_downloadedBytes = 0
failedHashes = 0


def isModelAlreadyInMemory(modelIdStr, model):
    """
    If the model used as a key, in the dictionary of all models, returns the same version id as the model's version id, then
    the model has already been downloaded

    :param modelIdStr: The model's id, as a string
    :param model: The model object from the API
    :return: A list of all the models that have been downloaded
    """
    if modelIdStr in aListOfAllDownloadedModels:
        if aListOfAllDownloadedModels[modelIdStr] == model["modelVersions"][0]["id"]:
            printv("Skipping model, already in memory", model["name"])
            return True
        print("Update found for model" + model["name"])
    return False


def findFiles(model):
    """
    It takes a model as input and returns a list of all the files in the model's directory.

    :param model: The model you want to find files for
    """
    files = []
    acceptableTypes = ["Pruned Model", "Model", "VAE"]
    modelFound = False
    for file in model["modelVersions"][0]["files"]:
        # print(f"{file}")
        if file["type"] in acceptableTypes:
            # if file["type"] == "Model" and modelFound:  # Prevent multiple models from being downloaded
            #    continue
            # if file["type"] == "Model":
            #    modelFound = True
            print(f"Added {file['name']}")
            files.append(file)

    return files


# It creates a class called File.
class File:
    def __init__(self, name, downloadUrl, imageUrl, hash):
        self.name = name
        self.downloadUrl = downloadUrl
        self.imageUrl = imageUrl
        self.hash = hash


def getFiles(model):
    """
    It takes a model object and returns a list of File objects

    :param model: The model object that we're getting the files for
    :return: A list of File objects.
    """
    files = findFiles(model)
    filesAsObjects = []

    safetensorfound = False
    pickletensorfound = False
    for file in files:
        # print(f"{file}")
        # find a safetensor first, if any
        if file["format"] == "SafeTensor":
            # use this one
            safetensorfound = True
            modelCkptFileName = file["name"]
            modelCkptDownloadLink = file["downloadUrl"]
            try:
                modelCkptHash = file["hashes"]["SHA256"]
            except KeyError:
                modelCkptHash = None
            modelImage = model["modelVersions"][0]["images"][0]["url"]
            break
    if safetensorfound == False:
        # grab a Pickle instead
        pickletensorfound = False
        for file in files:
            if file["format"] == "PickleTensor":
                # use this one
                pickletensorfound = True
                modelCkptFileName = file["name"]
                modelCkptDownloadLink = file["downloadUrl"]
                try:
                    modelCkptHash = file["hashes"]["SHA256"]
                except KeyError:
                    modelCkptHash = None
                modelImage = model["modelVersions"][0]["images"][0]["url"]
                break
    if pickletensorfound == False and safetensorfound == False:
        # Houston we have a problem
        print(f"Error, neither a safetensor nor pickletensor found... update your code")
    else:
        filesAsObjects.append(File(modelCkptFileName, modelCkptDownloadLink, modelImage, modelCkptHash))

    return filesAsObjects


def iterateAModel(model):
    """
    > Iterate through all the layers in a model and print out the layer name and the number of parameters in that layer

    :param model: the model to iterate
    """
    global aListOfAllDownloadedModels

    files = getFiles(model)

    # It creates a directory called JSON if it doesn't already exist.
    os.makedirs("JSON", exist_ok=True)
    with open(os.path.join("JSON", files[0].name + ".json"), "w") as f:
        json.dump(model, f, indent=4)
        # print("Dumped!")

    # Iterating through all the files in the model.
    for file in files:
        downloadSuccessful = downloadFile(file.downloadUrl, file.imageUrl, modelType=model["type"], hash=file.hash)

        if not downloadSuccessful:
            printv(f"Download failed, {file.name} skipping...")
            return

        # Iterating through all the model versions and getting all the trained words.
        trainedWords = [item for sublist in model["modelVersions"] for item in sublist["trainedWords"]]

        if len(trainedWords) > 0:  # Check if the list `trainedWords` is not empty.
            model_to_file = {
                "LORA": "LORA/" + file.name + ".txt",
                "TextualInversion": "TextualInversion/" + file.name + ".txt",
                "Checkpoint": "Checkpoint/" + file.name + ".txt",
            }

            # Writing the trained words to a file.
            file_path = model_to_file.get(model["type"], None)
            if file_path:
                with open(file_path, "w") as f:
                    for item in trainedWords:
                        f.write("%s " % item)

    # Adding the model to the list of all downloaded models.
    aListOfAllDownloadedModels[model["id"]] = model["modelVersions"][0]["id"]


def waitUntilAllThreadsAreDone(allThreads):
    """
    It waits until all the threads in the list `allThreads` are done

    :param allThreads: a list of all the threads that are running
    """
    for thread in allThreads:
        thread.join()


allThreads = []
multiThreadLimit = 20
session = requests.Session()
allModels = getModels(session)

# Iterating through all the models in the list `allModels` and printing out a progress bar.
# Downloading all the models from the session.
for model in tqdm(getModels(session), desc="Downloading models", leave=True, position=0):
    #  Check if the model has already been downloaded
    modelExists = isModelAlreadyInMemory(str(model["id"]),
                                         model)  # Keep in mind that modelExists will be false if there is an update
    modelIsCheckpoint = model["type"] == "Checkpoint"

    if not modelExists:
        if modelIsCheckpoint:  # If the model is a checkpoint then go in single threading to avoid errors
            iterateAModel(model)
            continue
        else:
            #  Start new thread
            newThread = threading.Thread(target=iterateAModel, args=(model,))
            newThread.start()
            allThreads.append(newThread)

    if session_downloadedFileCount > 10 or session_downloadedBytes > 1024 * 1024 * 1024:  # Downloaded 50 files or 1GB
        session_downloadedBytes = 0
        writeMemory(aListOfAllDownloadedModels)

    if len(threading.enumerate()) > 20:  # 50 threads or a checkpoint
        print("Waiting for threads to finish...")
        waitUntilAllThreadsAreDone(allThreads)  # Wait up for all the downloads to finish

    if len(threading.enumerate()) > 20 or model["type"] == "Checkpoint":  # 50 threads or a checkpoint
        waitUntilAllThreadsAreDone(allThreads)  # Wait up for all the downloads to finish

# It waits until all the threads in the list `allThreads` are done.
waitUntilAllThreadsAreDone(allThreads)
# Writing the list of all downloaded models to a file called `memory.json`.
writeMemory(aListOfAllDownloadedModels)
# Printing the number of failed hashes.
print(f"Failed hashes: {failedHashes}")
