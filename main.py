import argparse
import asyncio
import json
import os

import httpx
import rich.progress

with open('config.json', 'r') as f:
    config = json.load(f)

api_key = config["civitai_api_key"]

version = "0.0.1"

border = "=" * 50
message = f"       Running civitai_download.py v{version}"
print(f"\n{border}\n{message}\n{border}\n")

# It's creating a parser object, and adding an argument to it.
parser = argparse.ArgumentParser()
parser.add_argument("--verbose", action="store_true", help="Print debug messages")

# Parsing the arguments
args = parser.parse_args()


async def get_all_models():
    """
    It's getting all the models from the Civitai API
    :return: A list of all the models.
    """
    async with httpx.AsyncClient() as client:
        all_models = []
        params = {
            "types": ("Checkpoint", "TextualInversion", "Hypernetwork", "LORA", "AestheticGradient"),
            "period": "AllTime",
            "limit": 100,
            "page": 1
        }
        url = "https://civitai.com/api/v1/models"
        querystring = {"sort": "Newest", "favorites": "true"}
        headers = {"Authorization": f"Bearer {api_key}"}

        # Retrieve all available models.
        response = await client.get(url, headers=headers, params={**params, **querystring})
        if response.status_code == 200:
            response_json = response.json()
            all_models.extend(response_json["items"])

            # It's getting all the models from the Civitai API.
            for page_num in range(2, response_json["metadata"]["totalPages"] + 1):
                params["page"] = page_num
                response = await client.get(url, headers=headers, params={**params, **querystring})
                if response.status_code == 200:
                    response_json = response.json()
                    all_models.extend(response_json["items"])
            if args.verbose:
                print(f"Found {len(all_models)} models")
            return all_models
        else:
            print(f"Request failed with status code {response.status_code}")


async def map_api():
    """
    It returns a list of all the model version IDs, a list of all the model IDs, the type of the first model in the
    list of all models, and a list of the first model version for each model :return: model_versions_id, model_ids,
    file_type, modelver_list
    """
    # It's getting all the models from the Civitai API.
    all_models = await get_all_models()

    file_type = [model["type"] for model in all_models]
    model_versions_id = [version["id"] for model in all_models for version in model["modelVersions"]]
    model_ids = [model["modelVersions"][0]["modelId"] for model in all_models]
    modelver_list = [model["modelVersions"][0] for model in all_models]
    return model_versions_id, model_ids, file_type, modelver_list


async def download_file(download_url, filename: str) -> None:
    """
    Download the file from the URL and save it to the current working directory

    :param download_url: The URL to download the file from
    :param filename: the name of the file you want to download
    :type filename: str
    :return: Nothing is being returned.
    """
    _, _, file_type, modelver_list = await map_api()

    for modelver, ftype in zip(modelver_list, file_type):
        if args.verbose:
            print(f"Checking model version {modelver['name']}...")
        files = modelver["files"]
        for file in files:
            if file["name"] != filename:
                continue
            if args.verbose:
                print(f"Checking if {filename} exists...")
            # create filepath if it doesn't exist
            filepath = os.path.join(os.getcwd(), ftype)
            if not os.path.exists(filepath):
                os.makedirs(filepath)
                if args.verbose:
                    print(f"Created {ftype} directory.")
            else:
                print(f"{ftype} directory already exists. Skipping creation.")

            filepath = os.path.join(filepath, filename)

            # check if file already exists
            if os.path.exists(filepath):
                # check if file size matches expected size
                filesize = os.path.getsize(filepath)
                if filesize == file["sizeKB"] * 1024:
                    if args.verbose:
                        print(f"{filename} already exists and is the correct size. Skipping download...")
                    return

            print(f"\nDownloading {filename} from {download_url}...")
            block_size = 1024 * 1024 * 4  # 4 MB

            async with httpx.AsyncClient() as client:
                async with client.stream("GET", download_url, follow_redirects=True) as response:
                    response.raise_for_status()
                    total = int(response.headers["Content-Length"])

                    with rich.progress.Progress(
                            "[progress.percentage]{task.percentage:>3.0f}%",
                            rich.progress.BarColumn(bar_width=70),
                            rich.progress.DownloadColumn(),
                            rich.progress.TransferSpeedColumn(),
                    ) as progress:
                        download_task = progress.add_task("Download", total=total)
                        with open(filepath, "wb") as fr:
                            async for chunk in response.aiter_bytes(chunk_size=block_size):
                                fr.write(chunk)
                                progress.update(download_task, advance=len(chunk))
            return
    print(f"Could not find file {filename} in available model versions")


class File:
    def __init__(self, name, download_url, sha256_hash):
        """
        This function takes in three arguments, and assigns them to the three attributes of the class

        :param name: The name of the file
        :param download_url: The URL to download the file from
        :param sha256_hash: The SHA256 hash of the file. This is used to verify the integrity of the file
        """
        self.name = name
        self.download_url = download_url
        self.sha256_hash = sha256_hash

    @staticmethod
    async def get_files(modelver):
        """
        > It takes a model version object and returns a list of File objects

        :param modelver: The model version object that you want to get the files from
        :return: A list of File objects.
        """
        files = modelver["files"]
        files_as_objects = []
        required_extensions = [".ckpt", ".safetensors", ".zip", ".pt", ".bin"]

        for ext in required_extensions:
            for file in files:
                if "downloadUrl" in file and "hashes" in file and "SHA256" in file["hashes"]:
                    file_name = file["name"]
                    download_url = file["downloadUrl"]
                    sha256_hash = file["hashes"]["SHA256"]
                    file_type = os.path.splitext(file_name)[1]
                    if file_type == ext:
                        files_as_objects.append(File(file_name, download_url, sha256_hash))

        return files_as_objects


async def get_all_files():
    """
    It gets all the files for all the model versions
    :return: A list of all the files in the database.
    """
    # It's unpacking the tuple returned by get_model_versions_and_ids() into three variables.
    _, _, _, modelver_list = await map_api()
    all_files = []
    for modelver in modelver_list:
        files = await File.get_files(modelver)
        all_files.extend(files)
    return all_files


async def main():
    """
    It downloads all the files in the files_as_objects list
    """
    # Get all files
    files_as_objects = await get_all_files()

    # It's downloading all the files in the files_as_objects list.
    for file in files_as_objects:
        await download_file(file.download_url, file.name)


if __name__ == '__main__':
    # It's running the main() function asynchronously.
    asyncio.run(main())
