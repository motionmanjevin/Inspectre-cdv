import os
from PIL import Image
from tqdm import tqdm

import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# === CONFIG ===
IMAGE_FOLDER = "movement_frames/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_IMAGES = 50
QUESTION = "Describe what is happening in this image."

# === Load BLIP-2 FLAN-T5 XL ===
print("Loading BLIP-2 FLAN-T5 model...")
processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-flan-t5-xl", torch_dtype=torch.float16
).to(DEVICE)

def describe_image_blip2(image: Image.Image, question: str) -> str:
    inputs = processor(images=image, text=question, return_tensors="pt").to(DEVICE, torch.float16)
    generated_ids = model.generate(**inputs, max_new_tokens=50)
    return processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

def process_images(folder_path):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))][:MAX_IMAGES]
    results = {}

    for img_name in tqdm(image_files, desc="Analyzing images"):
        try:
            img_path = os.path.join(folder_path, img_name)
            img = Image.open(img_path).convert("RGB")
            caption = describe_image_blip2(img, QUESTION)
            results[img_name] = caption
        except Exception as e:
            print(f"Error with {img_name}: {e}")

    return results

if __name__ == "__main__":
    print("Starting...")
    captions = process_images(IMAGE_FOLDER)

    print("\n=== IMAGE CONTEXTS ===")
    for name, cap in captions.items():
        print(f"{name}: {cap}")
