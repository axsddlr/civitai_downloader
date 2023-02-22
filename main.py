import asyncio
import httpx
import json
import os
import argparse
import rich.progress

with open('config.json', 'r') as f:
    config = json.load(f)

api_key = config["civitai_api_key"]
APIKEY = f"Bearer {api_key}"

parser = argparse.ArgumentParser()

# Adding the arguments for the file names
parser.add_argument("--pickle", action="store_true", help="Only download PickleTensor files")
parser.add_argument("--verbose", action="store_true", help="Print debug messages")

# Parsing the arguments
args = parser.parse_args()


async def get_all_models():
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
        headers = {"Authorization": APIKEY}

        response = await client.get(url, headers=headers, params={**params, **querystring})
        if response.status_code == 200:
            response_json = response.json()
            all_models.extend(response_json["items"])

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
    _, _, file_type, modelver_list = await map_api()
    for ftype in file_type:
        for modelver in modelver_list:
            files = modelver["files"]
            for file in files:
                if file["name"] != filename:
                    continue
                if args.verbose:
                    print(f"Checking if {filename} exists...")
                # create filepath if it doesn't exist
                if not os.path.exists(os.path.join(os.getcwd(), str(ftype))):
                    os.makedirs(os.path.join(os.getcwd(), str(ftype)))
                # else:
                #     print(f"{ftype} directory already exists. Skipping creation...")

                filepath = os.path.join(os.getcwd(), ftype, filename)

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
                                rich.progress.BarColumn(bar_width=50),
                                rich.progress.DownloadColumn(),
                                rich.progress.TransferSpeedColumn(),
                        ) as progress:
                            download_task = progress.add_task("Download", total=total)
                            with open(filepath, "wb") as fr:
                                async for chunk in response.aiter_bytes(block_size):
                                    fr.write(chunk)
                                    progress.update(download_task, completed=response.num_bytes_downloaded)
                if args.verbose:
                    print(f"File downloaded: {filepath}")
                return
        print(f"Could not find file {filename} in available model versions")


class File:
    def __init__(self, name, download_url, sha256_hash):
        self.name = name
        self.download_url = download_url
        self.sha256_hash = sha256_hash

    @staticmethod
    async def get_files(modelver):
        files = modelver["files"]
        files_as_objects = []

        if files:
            # First, search for safetensors files (if not prioritized by the user)
            if not args.pickle:
                for file in files:
                    if "downloadUrl" in file and "hashes" in file and "SHA256" in file["hashes"]:
                        file_name = file["name"]
                        download_url = file["downloadUrl"]
                        sha256_hash = file["hashes"]["SHA256"]
                        file_type = os.path.splitext(file_name)[1]
                        if file_type == ".safetensors":
                            files_as_objects.append(File(file_name, download_url, sha256_hash))

            # If the user wants to prioritize ckpt files, search for them first
            if args.pickle:
                for file in files:
                    if "downloadUrl" in file and "hashes" in file and "SHA256" in file["hashes"]:
                        file_name = file["name"]
                        download_url = file["downloadUrl"]
                        sha256_hash = file["hashes"]["SHA256"]
                        file_type = os.path.splitext(file_name)[1]
                        if file_type == ".ckpt":
                            files_as_objects.append(File(file_name, download_url, sha256_hash))

            # If no safetensors or ckpt file is found (or if the user did not prioritize),
            # search for ckpt files (if not already prioritized)
            if not files_as_objects:
                for file in files:
                    if "downloadUrl" in file and "hashes" in file and "SHA256" in file["hashes"]:
                        file_name = file["name"]
                        download_url = file["downloadUrl"]
                        sha256_hash = file["hashes"]["SHA256"]
                        file_type = os.path.splitext(file_name)[1]
                        if file_type == ".ckpt":
                            files_as_objects.append(File(file_name, download_url, sha256_hash))

            return files_as_objects


async def get_all_files():
    # It's unpacking the tuple returned by get_model_versions_and_ids() into three variables.
    _, _, _, modelver_list = await map_api()
    all_files = []
    for modelver in modelver_list:
        files = await File.get_files(modelver)
        all_files.extend(files)
    return all_files


async def main():
    # Get all files
    files_as_objects = await get_all_files()

    # Print the file names and URLs
    for file in files_as_objects:
        await download_file(file.download_url, file.name)


if __name__ == '__main__':
    asyncio.run(main())
