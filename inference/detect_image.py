# inference/detect_image.py

import os
import cv2
import subprocess
from models.yolo.yolov8_loader import load_yolo
from utils.visualization import draw_boxes

INPUT_DIR = "data/samples/images"
OUTPUT_DIR = "output/images"

def detect_images_from_folder():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = load_yolo()

    images = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    if not images:
        print(" No images found in input folder")
        return

    for img_name in images:
        img_path = os.path.join(INPUT_DIR, img_name)
        out_path = os.path.join(OUTPUT_DIR, img_name)

        results = model(img_path)[0]
        image = cv2.imread(img_path)

        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()

        output_image = draw_boxes(image, boxes, classes, scores)
        cv2.imwrite(out_path, output_image)

        print(f" Saved: {out_path}")

        
        subprocess.run(["open", out_path])

if __name__ == "__main__":
    detect_images_from_folder()
