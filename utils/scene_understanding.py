import cv2
import torch
from collections import defaultdict
import clip
from PIL import Image


SCENE_LABELS = [
    "indoor office",
    "indoor home",
    "outdoor street",
    "traffic road",
    "crowded public place",
    "shopping mall",
    "parking lot",
    "factory or industrial area",
    "forest or park",
    "fire or smoke scene",
    "night time urban scene",
    "day time outdoor scene"
]


def sample_video_frames(video_path, frame_interval=30):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    if not cap.isOpened():
        return frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frames.append(frame)

        frame_count += 1

    cap.release()
    return frames


def video_scene_understanding(
    video_path,
    clip_model,
    clip_preprocess,
    device="cpu",
    frame_interval=30
):
    frames = sample_video_frames(video_path, frame_interval)

    if len(frames) == 0:
        return {"scene": "unknown", "confidence": 0.0}

    # Text features
    text_tokens = torch.cat(
        [clip.tokenize(f"a photo of {label}") for label in SCENE_LABELS]
    ).to(device)

    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    scene_scores = defaultdict(float)

    for frame in frames:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = clip_preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = clip_model.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarity = image_features @ text_features.T

        for idx, label in enumerate(SCENE_LABELS):
            scene_scores[label] += similarity[0][idx].item()

    # Sort scenes
    sorted_scenes = sorted(scene_scores.items(), key=lambda x: x[1], reverse=True)

    best_scene, best_score = sorted_scenes[0]
    second_best_score = sorted_scenes[1][1]

    confidence = (best_score - second_best_score) / abs(best_score)

    # TOP 3 only
    top_k = 3
    top_candidates = [
        {"scene": scene, "score": round(score, 3)}
        for scene, score in sorted_scenes[:top_k]
    ]

    return {
        "scene": best_scene,
        "confidence": round(max(0.0, min(confidence, 1.0)), 3),
        "top_candidates": top_candidates
    }
