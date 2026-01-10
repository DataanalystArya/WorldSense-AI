import os
from models.clip.clip_model import CLIPZeroShot

IMAGE_DIR = "data/samples/images"

SCENE_LABELS = [
    "highway",
    "market",
    "forest",
    "beach",
    "city street",
    "residential area",
    "office",
    "indoor room",
    "parking lot",
    "mountain"
]

def main():
    clip_model = CLIPZeroShot()

    for image_name in os.listdir(IMAGE_DIR):
        if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(IMAGE_DIR, image_name)
        print(f"\n Image: {image_name}")

        results = clip_model.classify_scene(image_path, SCENE_LABELS)

        top_scene = results[0]
        print(f" Scene detected: {top_scene['scene']} ({top_scene['confidence']:.3f})")

if __name__ == "__main__":
    main()
