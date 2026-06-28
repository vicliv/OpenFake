import json
import os

completed_models = []

with open('data.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

    # Iterate through the dictionary
    for model_name, details in data.items():
        if details.get("status") == "COMPLETED":
            completed_models.append(model_name)