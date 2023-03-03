import argparse
import json
import re
import os
import requests
from tqdm import tqdm

with open('tst_config.json', 'r') as f:
    config = json.load(f)

api_key = config["civitai_api_key"]

version = "0.0.2"

border = "=" * 50
message = f"       Running civitai_download.py v{version}"
print(f"\n{border}\n{message}\n{border}\n")

parser = argparse.ArgumentParser()
parser.add_argument("--verbose", action="store_true", help="Print debug messages")
parser.add_argument("--all", action="store_true", help="do not download preview images")

# Parsing the arguments
args = parser.parse_args()


def get_all_models():
    url = "https://civitai.com/api/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {
        "types": ("Checkpoint", "TextualInversion", "Hypernetwork", "LORA", "AestheticGradient"),
        "period": "AllTime",
        "limit": 100,
        "page": 1
    }
    querystring = {"sort": "Newest", "favorites": "true"}

    response = requests.get(url, headers=headers, params={**params, **querystring})

    if response.status_code != 200:
        print(f"Request failed with status code {response.status_code}")
        return []

    response_json = response.json()
    all_models = response_json["items"]

    while "nextPage" in response_json["metadata"]:
        params["page"] += 1
        response = requests.get(url, headers=headers, params={**params, **querystring})
        response_json = response.json()
        all_models += response_json["items"]

    if args.verbose:
        print(f"Found {len(all_models)} models")

    if args.verbose:
        print("models found from likes: " + str(all_models))
    return all_models


def get_model_ids():
    all_models = get_all_models()
    primary_model_ids = []

    # Loop through each model in the list of all models
    for model in all_models:
        version = model.get("modelVersions")[0]  # Get the first version of the model
        if 'files' in version:
            for file in version["files"]:
                if ('id' in file) and ('id' in version):
                    if file.get("primary") is True:
                        file_id = file["id"]
                        file_modelId = version["id"]

                        # Append the current "file_modelId" to the list of primary model IDs
                        if file_modelId not in primary_model_ids:
                            primary_model_ids.append(file_modelId)
    if args.verbose:
        print(f"Found {len(primary_model_ids)} primary model ids: {primary_model_ids}")
    if not primary_model_ids:
        raise ValueError("No primary model available")
    return primary_model_ids


def get_sizesKB():
    models = get_all_models()
    sizes = [
        float(file['sizeKB'])
        for model in models
        for version in model.get('modelVersions', [])
        for file in version.get('files', [])
        if file.get('primary') is True
    ]
    return sizes if args.all else [sizes[0]]


def download_files(save_dir, model_id_list):
    # Make dir if not exists
    os.makedirs(save_dir, exist_ok=True)
    existing_files = set(os.listdir(save_dir))

    # Create a session object from requests library
    with requests.Session() as session:
        # Download files
        print(f"Downloading {len(model_id_list)} files...")
        for model_id in tqdm(model_id_list):
            url = f"https://civitai.com/api/download/models/{model_id}"

            # Skip if downloaded
            if f"{model_id}_" in existing_files:
                print(f"[NOTICE] {url} is already downloaded, skipping.")
                continue

            # Download file
            print(f"Downloading {url}...")
            tmp_filename = os.path.join(save_dir, f"{model_id}.tmp")
            with session.get(url, stream=True, allow_redirects=True) as response:
                # Handle 404 errors
                if response.status_code == 404:
                    print(f"[INFO] {url} not found, skipping.")
                    continue

                # Extract filename from headers
                file_name = response.headers.get("content-disposition", "filename=")
                file_name = re.findall("filename=(.+)", file_name)[0].strip('"')
                file_name = file_name.replace(":", "_")

                save_name = os.path.join(save_dir, f"{model_id}_{file_name}")

                # Rename temporary files to filename
                with open(tmp_filename, 'wb') as fr:
                    for chunk in tqdm(response.iter_content(chunk_size=8192),
                                      total=int(response.headers.get('content-length', 0)) // 8192, unit="KB"):
                        fr.write(chunk)

                if os.path.exists(save_name):
                    os.remove(save_name)

                os.rename(tmp_filename, save_name)

                # Download finished
                print(f"[INFO] Downloaded {url} to {save_name}!")


download_files("downloads", get_model_ids())
