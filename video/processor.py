from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import torchvision.transforms as T
import numpy as np

class FrameProcessor:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.labels = ["person", "cup", "phone", "chair", "table", "book", "laptop"]

    def process(self, frame):
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = self.processor(text=self.labels, images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1).squeeze()
        results = [label for label, p in zip(self.labels, probs) if p > 0.3]
        return results