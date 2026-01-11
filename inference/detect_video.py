import sys
import os

# -------- PROJECT ROOT FIX (IMPORTANT) --------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# -------- IMPORTS --------
import cv2
from ultralytics import YOLO
from pathlib import Path
from utils.analytics import Analytics
from utils.alerts import AlertSystem


def detect_video(video_path, output_path):
    model = YOLO("yolov8n.pt")

    # INIT ANALYTICS + ALERTS
    analytics = Analytics()
    alerts = AlertSystem(person_threshold=5)

    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened(), f"Video is not opening: {video_path}"

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    print(f" Processing: {video_path.name}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO inference
        results = model(frame, conf=0.30)
        annotated_frame = results[0].plot()

        # ANALYTICS
        detections = results[0].boxes
        current_fps = analytics.update_fps()
        counts = analytics.count_objects(detections, model.names)

        # ALERTS
        alert_triggered = alerts.check_alerts(counts)

        # -------- VISUAL OVERLAY --------
        cv2.rectangle(
            annotated_frame,
            (10, 10),
            (400, 200),
            (0, 0, 0),
            -1
        )

        # FPS
        cv2.putText(
            annotated_frame,
            f"FPS: {current_fps}",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (0, 255, 0),
            3
        )

        # Object counts
        y_offset = 85
        for label, count in counts.items():
            cv2.putText(
                annotated_frame,
                f"{label}: {count}",
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2
            )
            y_offset += 35
        
        # -------- ALERT OVERLAY --------
        if alert_triggered:
            cv2.putText(
                annotated_frame,
                "ALERT: HIGH CROWD",
                (width - 380, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),   # RED
                3
            )


        out.write(annotated_frame)

    cap.release()
    out.release()
    print(f" Output saved at: {output_path}")


if __name__ == "__main__":

    INPUT_DIR = Path("data/samples/videos")
    OUTPUT_DIR = Path("output/videos")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Case 1: specific video
    if len(sys.argv) > 1:
        video_name = sys.argv[1]
        video_path = INPUT_DIR / video_name

        if not video_path.exists():
            print(f" Video not found: {video_name}")
            sys.exit(1)

        output_video = OUTPUT_DIR / f"{video_path.stem}_detected.mp4"
        detect_video(video_path, output_video)

    # Case 2: batch mode
    else:
        for video in INPUT_DIR.glob("*.mp4"):
            output_video = OUTPUT_DIR / f"{video.stem}_detected.mp4"
            detect_video(video, output_video)

    print(" DONE")
