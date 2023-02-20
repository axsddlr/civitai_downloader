import asyncio
import httpx
import json


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
    model_versions = [version["id"] for model in all_models for version in model["modelVersions"]]
    # This is a list comprehension that iterates over each model in all_models, then over each version in the
    # modelVersions list for that model, and returns the "id" attribute for each version.
    model_ids = [model["modelVersions"][0]["modelId"] for model in all_models]
    return model_versions, model_ids


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    model_versions, model_ids = loop.run_until_complete(get_model_versions_and_ids())
    print(model_versions)
