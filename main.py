import requests
import json
import os
import tempfile

# Read configuration from the file
with open("config.json", "r") as f:
    config = json.load(f)

api_key = config["civitai_api_key"]


def get_all_models():
    """
    It's getting all the models from the Civitai API
    :return: A list of all the models.
    """
    all_models = []
    params = {
        "types": (
            "Checkpoint",
            "TextualInversion",
            "Hypernetwork",
            "LORA",
            "AestheticGradient",
        ),
        "period": "AllTime",
        "limit": 100,
        "page": 1,
    }
    url = "https://civitai.com/api/v1/models"
    querystring = {"sort": "Newest", "favorites": "true"}
    headers = {
        "User-Agent": "CivitaiLink:Automatic1111",
        "Authorization": f"Bearer {api_key}",
    }

    # Retrieve all available models.
    response = requests.get(url, headers=headers, params={**params, **querystring})
    if response.status_code == 200:
        response_json = response.json()
        all_models.extend(response_json["items"])

        # It's getting all the models from the Civitai API.
        for page_num in range(2, response_json["metadata"]["totalPages"] + 1):
            params["page"] = page_num
            response = requests.get(
                url, headers=headers, params={**params, **querystring}
            )
            if response.status_code == 200:
                response_json = response.json()
                all_models.extend(response_json["items"])
        return all_models
    else:
        print(f"Request failed with status code {response.status_code}")
        return []


def extract_files_data(data):
    files_data = []
    for item in data:
        primary_key = item.get(
            "primary"
        )  # Replace 'primary_key' with the actual key name
        model_type = item.get("type")  # Get the type from the item
        model_versions = item.get("modelVersions", [])

        if model_versions:  # Check if there are any model versions available
            latest_model_version = model_versions[0]  # Get the latest model version
            files = latest_model_version.get("files", [])
            for file in files:
                file_info = {
                    "primary_key": primary_key,
                    "type": model_type,
                    "id": file.get("id"),
                    "name": file.get("name"),
                    "sizeKB": file.get("sizeKB"),
                    "file_type": file.get("type"),
                    "metadata": file.get("metadata"),
                    "hashes": file.get("hashes"),
                    "downloadUrl": file.get("downloadUrl"),
                }
                files_data.append(file_info)
    return files_data


def download_file(downloadUrl, filename, folder, sizeKB):
    # Create the folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)

    filepath = os.path.join(folder, filename)

    # Check if the file already exists
    if os.path.exists(filepath):
        # Check the existing file size
        existing_file_size_bytes = os.path.getsize(filepath)
        existing_file_size_kb = existing_file_size_bytes / 1024

        # Compare the existing file size with the expected sizeKB
        if round(existing_file_size_kb, 2) == round(sizeKB, 2):
            print(
                f"File '{filepath}' already exists and has the same size. Skipping download."
            )
            return
        else:
            print(
                f"File '{filepath}' already exists but has a different size. Downloading new file..."
            )

    # Get the file with a GET request
    response = requests.get(downloadUrl, stream=True)
    response.raise_for_status()

    # Get the Content-Length header to find the file size in bytes
    file_size_bytes = int(response.headers.get("Content-Length", 0))
    # Convert file size to kilobytes
    file_size_kb = file_size_bytes / 1024

    # Compare the file size with the expected sizeKB
    if round(file_size_kb, 2) != round(sizeKB, 2):
        print(
            f"File size mismatch: expected {sizeKB:.2f} KB, but found {file_size_kb:.2f} KB. Skipping download."
        )
        return

    # Save the file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, mode="wb", dir=folder) as temp_file:
        for chunk in response.iter_content(chunk_size=8192):
            temp_file.write(chunk)

    # Move the temporary file to the specified folder
    os.replace(temp_file.name, filepath)
    print(f"File '{filepath}' ({file_size_kb:.2f} KB) downloaded successfully.")


# Get models data
models_data = get_all_models()

# Extract files data
files_data = extract_files_data(models_data)

# Download files with primary_key
for file_info in files_data:
    if "primary_key" in file_info:
        downloadUrl = file_info["downloadUrl"]
        filename = file_info["name"]
        folder = file_info["type"]
        sizeKB = file_info["sizeKB"]
        download_file(downloadUrl, filename, folder, sizeKB)
    else:
        print(f"Skipping '{file_info['name']}' as it does not have a primary key.")
