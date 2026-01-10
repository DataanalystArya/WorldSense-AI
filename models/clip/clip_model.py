import torch
import clip
from PIL import Image


class CLIPZeroShot:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    # 🔹 FEATURE 1: Zero-shot concept detection (OLD FEATURE — SAFE)
    def detect(self, image_path, text_queries):
        image = self.preprocess(
            Image.open(image_path).convert("RGB")
        ).unsqueeze(0).to(self.device)

        text_tokens = clip.tokenize(text_queries).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text_tokens)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarities = (image_features @ text_features.T).softmax(dim=-1)

        results = []
        for label, score in zip(text_queries, similarities[0]):
            results.append({
                "label": label,
                "confidence": float(score)
            })

        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results

    # 🔹 FEATURE 2: Scene classification (NEW FEATURE — ADDITIVE)
    def classify_scene(self, image_path, scene_labels):
        image = self.preprocess(
            Image.open(image_path).convert("RGB")
        ).unsqueeze(0).to(self.device)

        text_tokens = clip.tokenize(scene_labels).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text_tokens)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarities = (image_features @ text_features.T).softmax(dim=-1)

        results = []
        for label, score in zip(scene_labels, similarities[0]):
            results.append({
                "scene": label,
                "confidence": float(score)
            })

        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results
