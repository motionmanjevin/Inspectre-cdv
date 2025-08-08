import os
from PIL import Image
from tqdm import tqdm

# Option 1: Local AI (BLIP-2)
from transformers import BlipProcessor, BlipForConditionalGeneration

# Option 2: API-based (e.g., OpenAI's GPT-4V) — optional
# import openai

# === CONFIGURATION ===
IMAGE_FOLDER = 'movement_frames/'  # folder where you store your video frames
USE_LOCAL_AI = True  # Set to False to use OpenAI GPT-4V (API required)
MAX_IMAGES = 50  # Limit for performance

# === LOAD LOCAL MODEL ===
if USE_LOCAL_AI:
    print("Loading BLIP model...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def describe_image_local(image: Image.Image) -> str:
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# === OPTIONAL: GPT-4V or GPT API ===
# def describe_image_openai(image_path):
#     with open(image_path, "rb") as image_file:
#         response = openai.ChatCompletion.create(
#             model="gpt-4-vision-preview",
#             messages=[
#                 {"role": "user", "content": [
#                     {"type": "text", "text": "Describe what's happening in this image."},
#                     {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(image_file.read()).decode()}" }}
#                 ]}
#             ],
#             max_tokens=100
#         )
#         return response['choices'][0]['message']['content']

def process_images(folder_path):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:MAX_IMAGES]
    results = {}

    for img_name in tqdm(image_files, desc="Processing images"):
        try:
            img_path = os.path.join(folder_path, img_name)
            img = Image.open(img_path).convert("RGB")

            if USE_LOCAL_AI:
                description = describe_image_local(img)
            else:
                description = describe_image_openai(img_path)

            results[img_name] = description
        except Exception as e:
            print(f"Error processing {img_name}: {e}")

    return results

if __name__ == "__main__":
    print("Starting image context analysis...")
    descriptions = process_images(IMAGE_FOLDER)

    print("\n--- IMAGE CONTEXT RESULTS ---")
    for img, desc in descriptions.items():
        print(f"{img}: {desc}")
