import os
import json
from models.clip.clip_model import CLIPZeroShot


OBJECT_PROMPTS = [
    "person",
    "crowd",
    "road",
    "vehicle",
    "market",
    "highway",
    "animal",
    "tree",
    "building",
    "food"
]

SCENE_PROMPTS = [
    "highway",
    "city street",
    "marketplace",
    "forest",
    "indoor room",
    "crowded place",
    "village",
    "urban area"
]



def analyze_image(image_path, threshold=0.12):
    clip_model = CLIPZeroShot()

    # Object detection
    object_results = clip_model.detect(image_path, OBJECT_PROMPTS)

    objects = [
        obj["label"]
        for obj in object_results
        if obj["confidence"] >= threshold
    ]

    # Scene detection
    scene_results = clip_model.detect(image_path, SCENE_PROMPTS)
    scene = max(scene_results, key=lambda x: x["confidence"])["label"]

    return objects, scene



IMAGE_DIR = "data/samples/images"
OUTPUT_DIR = "output"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    clip_model = CLIPZeroShot()

    for image_name in os.listdir(IMAGE_DIR):
        if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(IMAGE_DIR, image_name)
        print(f"\n🔍 Processing: {image_name}")

        # Object detection
        object_results = clip_model.detect(image_path, OBJECT_PROMPTS)
        top_objects = sorted(
            object_results,
            key=lambda x: x["confidence"],
            reverse=True
        )[:5]

        # Scene detection
        scene_results = clip_model.detect(image_path, SCENE_PROMPTS)
        top_scene = max(scene_results, key=lambda x: x["confidence"])

        print("Top objects:")
        for obj in top_objects:
            print(f" - {obj['label']} : {obj['confidence']:.3f}")

        print(f"Scene: {top_scene['label']} ({top_scene['confidence']:.3f})")

        image_result = {
            "image": image_name,
            "objects": top_objects,
            "scene": top_scene
        }

        json_name = os.path.splitext(image_name)[0] + ".json"
        json_path = os.path.join(OUTPUT_DIR, json_name)

        with open(json_path, "w") as f:
            json.dump(image_result, f, indent=2)

        print(f" Saved: {json_path}")


if __name__ == "__main__":
    main()
