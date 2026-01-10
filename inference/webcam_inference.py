import cv2
from ultralytics import YOLO
import torch


def run_webcam_detection(
    model_path="models/yolov8n.pt",
    device="cuda"
):
    print("📷 Starting webcam...")
    print("Press 'q' to quit")

    # Load model
    model = YOLO(model_path)
    model.to(device)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Webcam not accessible")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO inference
        results = model(
            frame,
            conf=0.4,
            device=device,
            verbose=False
        )

        annotated_frame = results[0].plot()

        cv2.imshow("WorldSense-AI | Live Webcam Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("🛑 Webcam stopped")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_webcam_detection(device=device)
