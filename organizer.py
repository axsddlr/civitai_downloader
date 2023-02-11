import json
import os
import re

import requests

from src.utils.utils import headers


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


def get_models(session):
    """
    It takes a session object and returns a list of model objects

    :param session: the session object that you created earlier
    :return: A list of dictionaries.
    """
    model_ids = get_model_id()
    models = []
    for model_id in model_ids:
        model_response = session.get(f"https://civitai.com/api/v1/models/{str(model_id)}", headers=headers)
        models.append(json.loads(model_response.text))
    if len(models) == 0:
        print("id.txt is empty, please add model ids to it")
        exit()
    else:
        return models


def main():
    session = requests.Session()
    models = get_models(session)
    for model in models:
        type_ = model["type"]
        # for model_version in model["modelVersions"]:
        #     for file in model_version["files"]:
        link = "https://civitai.com/models/" + str(model["id"])
        with open(f"{type_}.txt", "a") as f:
            f.write(link + "\n")


if __name__ == "__main__":
    main()
