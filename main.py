import asyncio
import httpx
import json
import os

with open('config.json', 'r') as f:
    config = json.load(f)

api_key = config["civitai_api_key"]
APIKEY = f"Bearer {api_key}"


async def get_all_models():
    async with httpx.AsyncClient() as client:
        all_models = []
        params = {
            "types": ("TextualInversion", "Hypernetwork", "LORA", "AestheticGradient"),
            "period": "AllTime",
            "limit": 100,
            "page": 1
        }
        url = "https://civitai.com/api/v1/models"
        querystring = {"sort": "Newest", "favorites": "true"}
        headers = {"Authorization": APIKEY}

        response = await client.get(url, headers=headers, params=querystring)
        response_json = response.json()
        all_models.extend(response_json["items"])

        for page_num in range(2, response_json["metadata"]["totalPages"] + 1):
            params["page"] = page_num
            response = await client.get(url, headers=headers, params=querystring)
            response_json = response.json()
            all_models.extend(response_json["items"])

        params = {
            "types": "Checkpoint",  # All types: TextualInversion, Hypernetwork, Checkpoint, LORA, AestheticGradient
            "period": "AllTime",
            "limit": 30,
            "page": 1
        }

        response = await client.get(url, headers=headers, params=querystring)
        response_json = response.json()
        all_models.extend(response_json["items"])

        print(f"Found {len(all_models)} models")
        return all_models


async def get_model_versions_and_ids():
    all_models = await get_all_models()
    # Here, the modified model_versions list comprehension iterates over each model in all_models, then over each
    # version in the modelVersions list for that model, and returns the "id" attribute for each version.
    model_versions_id = [version["id"] for model in all_models for version in model["modelVersions"]]

    # This is a list comprehension that iterates over each model in all_models, then over each version in the
    # modelVersions list for that model, and returns the "id" attribute for each version.
    model_ids = [model["modelVersions"][0]["modelId"] for model in all_models]

    # It's a list comprehension that iterates over each model in all_models, then over each version in the
    # modelVersions list for that model, and returns the "id" attribute for each version.
    modelver_list = [model["modelVersions"][0] for model in all_models]
    return model_versions_id, model_ids, modelver_list


def checkIfFileExists(filename, size):
    if os.path.exists(filename):
        if os.path.getsize(filename) == size:
            return True
    return False


def compareSizes(filename, size):
    if os.path.exists(filename):
        if os.path.getsize(filename) == size:
            return True
    return False


async def download_file(url, filename):
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        with open(filename, "wb") as f:
            f.write(response.content)


class File:
    def __init__(self, name, download_url, sha256_hash):
        self.name = name
        self.download_url = download_url
        self.sha256_hash = sha256_hash

    async def download_file(self, file_name):
        async with httpx.AsyncClient() as client:
            async with client.stream("GET", self.download_url) as response:
                response.raise_for_status()
                with open(file_name, 'wb') as fr:
                    async for chunk in response.aiter_bytes(1024 * 1024 * 20):
                        fr.write(chunk)

    async def get_files(modelver):
        files = modelver["files"]
        files_as_objects = []

        if files:
            for file in files:
                if "downloadUrl" in file and "hashes" in file and "SHA256" in file["hashes"]:
                    file_name = file["name"]
                    download_url = file["downloadUrl"]
                    sha256_hash = file["hashes"]["SHA256"]
                    file_type = os.path.splitext(file_name)[1]
                    if file_type == ".safetensors" or file_type == ".ckpt":
                        files_as_objects.append(File(file_name, download_url, sha256_hash))
                        # await download_file(download_url, file_name)

        # print(files_as_objects)
        return files_as_objects


async def get_all_files():
    _, _, modelver_list = await get_model_versions_and_ids()
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
        print(f"Name: {file.name}\nURL: {file.download_url}\n")


if __name__ == '__main__':
    asyncio.run(main())
