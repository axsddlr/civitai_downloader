import argparse
import json
import re
import os

import cloudscraper

scraper = cloudscraper.create_scraper(browser='chrome')

with open('config.json', 'r') as f:
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

    response = scraper.get(url, headers=headers, params={**params, **querystring})

    if response.status_code != 200:
        print(f"Request failed with status code {response.status_code}")
        return []

    response_json = response.json()
    all_models = response_json["items"]

    while "nextPage" in response_json["metadata"]:
        params["page"] += 1
        response = scraper.get(url, headers=headers, params={**params, **querystring})
        response_json = response.json()
        all_models += response_json["items"]

    if args.verbose:
        print(f"Found {len(all_models)} models")

    # print(all_models)
    return all_models


def get_primary_model_ids():
    all_models = get_all_models()
    primary_model_ids = [version["id"] for model in all_models for version in model.get("modelVersions") if
                         any(file.get("primary") is True for file in version.get("files", []))]
    if args.verbose:
        print(f"Found {len(primary_model_ids)} model ids: {primary_model_ids}")
    return primary_model_ids if args.all else [primary_model_ids[0]]


def mkdir(mdir):
    if not os.path.exists(mdir):
        os.makedirs(mdir)


def file_exists(save_dir, model_id):
    for file in os.listdir(save_dir):
        if file.startswith(f"{model_id}_"):
            return True
    return False


def get_download_url():
    all_models = get_all_models()
    primary_model_ids = [
        ver["downloadUrl"]
        for model in all_models
        for ver in model.get("modelVersions")
        if any(file.get("primary") is True for file in ver.get("files", []))
    ]
    if args.verbose:
        print(f"Found {len(primary_model_ids)} model ids: {primary_model_ids}")
    return primary_model_ids[0] if not args.all else primary_model_ids


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


def download_civitai_by_id_list(save_dir, model_id_list, download_url):
    # Make dir if not exists
    mkdir(save_dir)
    existing_files = set(os.listdir(save_dir))

    # Download files
    print(f"Downloading {len(model_id_list)} files...")
    for model_id in model_id_list:
        url = download_url

        # Skip if downloaded
        if f"{model_id}_" in existing_files:
            print(f"[NOTICE] {url} is already downloaded, skipping.")
            continue

        # Download file
        print(f"Downloading {url}...")
        tmp_filename = os.path.join(save_dir, f"{model_id}.tmp")
        with scraper.get(url, stream=True) as r:
            # Set encoding to utf-8 to avoid character issues
            r.encoding = 'utf-8'
            with open(tmp_filename, 'wb') as fr:
                for chunk in r.iter_content(chunk_size=8192):
                    # If you have chunk encoded response uncomment if
                    # and set chunk_size parameter to None.
                    # if chunk
                    fr.write(chunk)
        print(r.headers)
        if "Content-Disposition" not in r.headers:
            os.remove(tmp_filename)
            print(f"[INFO] {url} not found, skipping.")
            continue

        # Encode it into ISO-8859-1 and then return it to utf-8 (https://www.rfc-editor.org/rfc/rfc5987.txt)
        headers_utf8 = r.headers["Content-Disposition"].encode('ISO-8859-1').decode()
        file_name_raw = re.search(r'filename=\"(.*)\"', headers_utf8).group(1)
        file_name = file_name_raw.replace(":", "_")
        save_name = os.path.join(save_dir, f"{model_id}_{file_name}")

        # Rename temporary files to filename
        if os.path.exists(save_name):
            os.remove(save_name)
        os.rename(tmp_filename, save_name)

        # Download finished
        print(f"[INFO] Downloaded {url} to {save_name}!")


download_civitai_by_id_list("downloads", get_primary_model_ids(), get_download_url())
print(get_download_url())
