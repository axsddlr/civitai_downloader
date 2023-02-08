import os
import time

import requests
from tqdm import tqdm
from src.utils.fileutils import checkIfFileExists, compareSizes
from src.utils.utils import headers

session_downloadedFileCount = 0
session_downloadedBytes = 0


def downloadFile(url, imageurl, modelType, hash, retries=20):
    """
    Download a file from a URL, and check if the download was successful

    :param url: The URL of the model
    :param modelType: The type of model, e.g. "3d_models"
    :param hash: The hash of the model, used to check if the download was successful
    :param retries: How many times to retry the download if it fails, defaults to 4 (optional)
    :return: A boolean value, True if the download was successful, False if it failed.
    """
    global session_downloadedBytes
    global session_downloadedFileCount
    global failedHashes
    for retryN in range(retries):
        # Download the model with a progress bar
        response = requests.get(url, stream=True, headers=headers)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024 * 100  # 100 MB chunk size
        filename = response.headers.get('Content-Disposition').split('filename=')[1].replace('"', '')
        filename = os.path.join(modelType, filename)  # Put the model in a folder based on its type

        #  Check if the file already exists size+hash
        if checkIfFileExists(filename, total_size_in_bytes):
            if compareSizes(filename, total_size_in_bytes):
                print(f"{filename} size matches, skipping...")
                session_downloadedBytes += total_size_in_bytes
                return True

        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, desc=f"try: {retryN} {filename}",
                            leave=False)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as file:  # Download the file in chunks
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()

        #  Check if the download was successful
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print(f"ERROR, something went wrong with {filename}, retrying...")
            continue  # Retry

        #  Check hash using sha256
        time.sleep(1)
        if not compareSizes(filename, total_size_in_bytes):  # If the hash doesn't match
            print(f"{filename} Size doesn't match, retrying...")
            failedHashes += 1
            os.remove(filename)  # Delete the file
            continue  # Retry
        else:
            imgresponse = requests.get(imageurl)
            if imgresponse.status_code:
                fp = open(filename.rsplit(".", 1)[0] + ".preview.png", 'wb')
                fp.write(imgresponse.content)
                fp.close()
        session_downloadedBytes += total_size_in_bytes
        session_downloadedFileCount += 1
        return True  # If we reached here, the download was successful
    return False  # Out of tries
