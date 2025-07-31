import cv2
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torchvision.transforms as T
from collections import deque

class FrameProcessor:
    def __init__(self, 
                 base_labels=None,
                 max_history=5,
                 top_k=5,
                 threshold=0.25):
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        self.labels = base_labels or [
            "person", "cup", "phone", "chair", 
            "table", "book", "laptop", "bag", "bottle"
        ]
        self.top_k = top_k
        self.threshold = threshold
        
        # Store recent detections for temporal smoothing
        self.history = deque(maxlen=max_history)

        # Optional transform for resizing/normalizing
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()
        ])
    
    def _predict_labels(self, image):
        """
        Run CLIP inference for an image and candidate labels.
        """
        inputs = self.processor(text=self.labels, images=image, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1).squeeze()

        # Select top-k labels above threshold
        top_indices = torch.topk(probs, self.top_k).indices.tolist()
        results = [(self.labels[i], float(probs[i])) for i in top_indices if probs[i] > self.threshold]
        return results
    
    def _smooth_predictions(self, predictions):
        """
        Smooth results over recent frames to stabilize noisy detections.
        """
        self.history.append(predictions)
        all_preds = {}
        for frame_preds in self.history:
            for label, conf in frame_preds:
                all_preds[label] = max(all_preds.get(label, 0), conf)
        
        # Return averaged predictions
        return [(label, conf) for label, conf in sorted(all_preds.items(), key=lambda x: x[1], reverse=True)]
    
    def process(self, frame):
        """
        Main entry: process a frame, return smoothed labels with confidence.
        """
        # Convert frame to PIL image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Predict labels using CLIP
        predictions = self._predict_labels(image)
        
        # Smooth predictions across history
        smoothed_results = self._smooth_predictions(predictions)
        
        return smoothed_results
