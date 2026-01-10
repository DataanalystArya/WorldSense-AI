WorldSense-AI  
### Real-Time Image & Video Object Detection System (YOLOv8 + CLIP)

This project started as my attempt to understand how object detection actually works in real-world systems.
Instead of writing one long script, I structured this project like an industry-style system, where:
models,inference pipelines,scene understanding
,reasoning and utilities are all clearly separated.The focus of this project is not only “what is visible”, but also “what is happening, why it is happening, and what this scene means.”

### Features Implemented
### Core Detection:
Image Object Detection (YOLOv8), Video Object Detection (single & batch), Multi-object detection, Automatic saving of output images and videos, CLI-based execution

### Scene Intelligence (New):
Video-level scene understanding using CLIP, Confusion-aware scene output (top competing scenes), Confidence-level estimation (LOW / MEDIUM / HIGH), Human-readable “WHY” explanations for predictions

### Live & Real-Time (Experimental):
Live webcam object detection, Real-time inference loop (OpenCV based)

### Image Reasoning:
Reasoning over objects and context in an image, Structured JSON-based outputs, Answers questions like “What is happening in this image?”

### Zero-Shot Intelligence (CLIP-based):
Uses CLIP vision–language embeddings, Can reason about unseen or unknown scenes, No hardcoded labels required

### Voice Interface (Experimental):
Voice input support, Voice-based output responses, Foundation for a multimodal AI assistant

### Structured Outputs:
Clean and readable JSON output, Confidence scores, Top scene candidates, Reasoning and explanations

### Tech Stack
-Python,YOLOv8 (Ultralytics),OpenCV,PyTorch,CLIP (Vision-Language Model)

## ▶️ How to Run
### 1. Create virtual environment & install dependencies:-
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
### 2. How to Run Inference:-

Image Detection
Put your input images inside:
data/samples/images/

Run the image detection pipeline:
python -m inference.detect_image

Detected images will be automatically saved to:
output/images/

### 3. Video Detection (Single Video):-

Place the video file inside:
data/samples/videos/

Run detection on a specific video:
python -m inference.detect_video market.mp4

The detected video will be saved to:
output/videos/

### Video Detection (Batch Mode):-

To run detection on all videos present in the folder:
python -m inference.detect_video

### Image Detection Results
Sample Outputs
![Image Detection 1](assets/demo_image1.jpg)
![Image Detection 2](assets/demo_image2.jpg)
![Image Detection 3](assets/demo_image3.jpg)

### 🎥 Video Detection Demo
### Market Scene Detection
GIF generated from object detection on a crowded market video:

![Market Video Detection](assets/demo_video.gif)

### Highway Scene Detection
GIF generated from object detection on a highway traffic video:

![Highway Video Detection](assets/highway_demo.gif)

How to Run (Voice + Zero-Shot Reasoning)
### Activate Virtual Environment
source venv310/bin/activate
[Voice-based features currently run in venv310]
(due to microphone + speech dependencies)

### Video Scene Understanding run:-
python main.py
This will:Extract video frames,Identify the scene,Generate confidence-aware predictions,Produce,human-readable explanations,Save structured JSON outputs

### Image Reasoning
Image reasoning results are automatically saved to:
output/image_reasoning/

### Voice Interface (Experimental)
python voice/voice_input.py

### Future Work
Object tracking (DeepSORT / ByteTrack),Event-level understanding,Temporal memory & video Q&A,Voice-based interaction,Web dashboard (Streamlit),Performance analytics & logs

### Author
Built with ❤️ by Arya Verma

