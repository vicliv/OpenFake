from transformers import AutoModelForImageClassification, AutoImageProcessor
import os

model_id = "Falconsai/nsfw_image_detection"
save_path = "data/models/nsfw_filtering_model"

if not os.path.exists(save_path):
    os.makedirs(save_path)

print(f"Downloading {model_id} components from HuggingFace...")

# Download the underlying model and processor explicitly
model = AutoModelForImageClassification.from_pretrained(model_id)
processor = AutoImageProcessor.from_pretrained(model_id)

print(f"Saving model artifacts to {save_path}...")
model.save_pretrained(save_path)
processor.save_pretrained(save_path)

print("Download complete. The model is ready for offline use.")