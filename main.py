import argparse
import asyncio
import json
import os
import tempfile

import httpx
import rich.progress

with open('config.json', 'r') as f:
    config = json.load(f)

api_key = config["civitai_api_key"]

version = "0.0.2"

border = "=" * 50
message = f"       Running civitai_download.py v{version}"
print(f"\n{border}\n{message}\n{border}\n")

# It's creating a parser object, and adding an argument to it.
parser = argparse.ArgumentParser()
parser.add_argument("--verbose", action="store_true", help="Print debug messages")
parser.add_argument("--preview", action="store_true", help="do not download preview images")

# Parsing the arguments
args = parser.parse_args()

user_agent = 'CivitaiLink:Automatic1111'


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
        headers = {"User-Agent": user_agent, "Authorization": f"Bearer {api_key}"}

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
            if len(all_models) == 0:
                print("No models found. Please check your API key or like some models and try again.")
                exit(1)
            else:
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
    It downloads the file and preview image from the given URL

    :param download_url: The URL to download the file from
    :param filename: The name of the file you want to download
    :type filename: str
    :return: None
    """
    _, _, file_type, modelver_list = await map_api()

    for modelver, ftype in zip(modelver_list, file_type):
        if args.verbose:
            print(f"Checking model version {modelver['name']}...")
        files = modelver["files"]
        modelImage = [pimage["url"] for pimage in modelver["images"]]
        for file in files:
            if file["name"] != filename:
                continue

            # create filepath if it doesn't exist
            file_dir = os.path.join(os.getcwd(), ftype)
            if not os.path.exists(file_dir):
                os.makedirs(file_dir)

            # create file paths for saving
            file_path = os.path.join(file_dir, filename)

            # check if file already exists
            if os.path.exists(file_path):
                # check if file size matches expected size
                filesize = os.path.getsize(file_path)
                if filesize == file["sizeKB"] * 1024:
                    if args.verbose:
                        print(f"{filename} already exists and is the correct size. Skipping download...")
                    return

            # download image and file
            if not args.preview:
                preview_file_name = os.path.splitext(filename)[0] + ".preview.png"
                image_path = os.path.join(file_dir, preview_file_name)

            print(f"\nDownloading {filename} from {download_url}...")
            block_size = 8192  # set the block size to 8192 bytes.

            async with httpx.AsyncClient() as client:
                async with client.stream("GET", download_url, follow_redirects=True,
                                         headers={"User-Agent": user_agent}) as response:

                    response.raise_for_status()
                    total = int(response.headers["Content-Length"])

                    # It's creating a temporary file in the file_dir directory.
                    with tempfile.NamedTemporaryFile(mode="wb", delete=False, dir=file_dir) as tmp_file:
                        with rich.progress.Progress(
                                "[progress.percentage]{task.percentage:>3.0f}%",
                                rich.progress.BarColumn(bar_width=70),
                                rich.progress.DownloadColumn(),
                                rich.progress.TransferSpeedColumn(),
                        ) as progress:
                            download_task = progress.add_task("Download", total=total)
                            async for chunk in response.aiter_bytes(chunk_size=block_size):
                                tmp_file.write(chunk)
                                progress.update(download_task, advance=len(chunk))

                    # move the temporary file to the final destination
                    os.replace(tmp_file.name, file_path)

            if not args.preview:
                preview_url = modelImage[0]
                if args.verbose:
                    print(f"\nDownloading preview image from {preview_url}...")
                async with httpx.AsyncClient() as client:
                    response = await client.get(preview_url)
                    response.raise_for_status()
                    with open(image_path, "wb") as fi:
                        fi.write(response.content)
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
        allowed_file_types = ["Model", "VAE", "Pruned Model"]

        for ext in required_extensions:
            for file in files:
                if "downloadUrl" in file and "hashes" in file and "SHA256" in file["hashes"]:
                    file_name = file["name"]
                    download_url = file["downloadUrl"]
                    sha256_hash = file["hashes"]["SHA256"]
                    file_type = os.path.splitext(file_name)[1]

                    if file_type == ext and "primary" in file and file["primary"] and "type" in file and file[
                        "type"] in allowed_file_types:
                        if "format" in file:
                            if file["format"] == "SafeTensor":
                                download_url = f"{download_url}?type={file['type']}&format=SafeTensor"
                            elif file["format"] == "PickleTensor":
                                download_url = f"{download_url}?type={file['type']}&format=PickleTensor"
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
