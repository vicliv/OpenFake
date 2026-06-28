import pandas as pd
from datetime import datetime
from huggingface_hub import HfApi, model_info, snapshot_download

def check_safetensors_available(model_id): #verify safetensors available
    # print("verifying tensors...")
    try:
        info = model_info(model_id)
        # Scan all files in the repo for the .safetensors extension
        # print(f"scanning through {len(info.siblings)} items")
        file_names = [f.rfilename for f in info.siblings]
        return any(fname.endswith(".safetensors") for fname in file_names)
    except Exception as e:
        # print(f"Error checking files for {model_id}: {e}")
        return False

api = HfApi()

print("Fetching txt2img models from Hugging Face...")
models = api.list_models(filter="diffusers", sort="downloads", full=True, limit=1000)

model_count = 0 

for model in models:
        downloads = getattr(model, "downloads", 0)
        if downloads < 30000:
            continue

        else: 
            print(f"{model.id},{downloads}")
            
        if not check_safetensors_available(model.id):
            continue

        pipeline_task = getattr(model, "pipeline_tag", "") or ""
        if any(bad_word in pipeline_task.lower() for bad_word in ["video", "audio", "3d"]):
            continue

        model_count += 1

    
print(model_count)

