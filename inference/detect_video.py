import cv2
from ultralytics import YOLO
from pathlib import Path
import os
import sys


def detect_video(video_path, output_path):
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened(), f" Video is not opening: {video_path}"

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    print(f"▶️ Processing: {video_path.name}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.30)
        annotated_frame = results[0].plot()

        out.write(annotated_frame)

    cap.release()
    out.release()

    print(f" Output saved at: {output_path}")


if __name__ == "__main__":

    INPUT_DIR = Path("data/samples/videos")
    OUTPUT_DIR = Path("output/videos")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Case 1: specific video passed
    if len(sys.argv) > 1:
        video_name = sys.argv[1]
        video_path = INPUT_DIR / video_name

        if not video_path.exists():
            print(f" Video not found: {video_name}")
            exit(1)

        output_video = OUTPUT_DIR / f"{video_path.stem}_detected.mp4"
        detect_video(video_path, output_video)

    #  Case 2: no argument → run all videos
    else:
        for video in INPUT_DIR.glob("*.mp4"):
            output_video = OUTPUT_DIR / f"{video.stem}_detected.mp4"
            detect_video(video, output_video)

    print("DONE")

