import os
import json
import torch
import clip
from PIL import Image

IMAGE_DIR = "data/samples/images"
OUTPUT_DIR = "output/image_reasoning"

# Hardcoded reasoning questions
QUESTIONS = {
    "Is this place crowded?": [
        "a crowded place with many people",
        "a large crowd of people",
        "an empty place with no people"
    ],
    "Is this an outdoor scene?": [
        "an outdoor scene",
        "an indoor room"
    ],
    "Is there traffic in the image?": [
        "vehicles on the road",
        "traffic jam",
        "empty road with no vehicles"
    ]
}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    for image_name in os.listdir(IMAGE_DIR):
        if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(IMAGE_DIR, image_name)
        print(f"\n🔍 Reasoning on: {image_name}")

        image = preprocess(
            Image.open(image_path).convert("RGB")
        ).unsqueeze(0).to(device)

        image_results = {
            "image": image_name,
            "reasoning": []
        }

        with torch.no_grad():
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            for question, statements in QUESTIONS.items():
                text_tokens = clip.tokenize(statements).to(device)
                text_features = model.encode_text(text_tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                similarities = (image_features @ text_features.T).softmax(dim=-1)
                best_idx = similarities[0].argmax().item()

                image_results["reasoning"].append({
                    "question": question,
                    "answer": statements[best_idx],
                    "confidence": float(similarities[0][best_idx])
                })

        # save per-image JSON
        output_path = os.path.join(
            OUTPUT_DIR, image_name.rsplit(".", 1)[0] + "_reasoning.json"
        )

        with open(output_path, "w") as f:
            json.dump(image_results, f, indent=2)

        print(f" Saved: {output_path}")


if __name__ == "__main__":
    main()
