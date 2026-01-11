import sys
import json
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import clip
from utils.confusion_aware_scene import confusion_aware_scene
from utils.scene_understanding import video_scene_understanding
from utils.why_explanations import generate_why_explanation
from utils.temporal_memory import TemporalMemory
from utils.event_summary import generate_event_summary

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# -------- LOAD CLIP (tumhare existing style me) --------
device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()
memory = TemporalMemory()

# -------- INPUT VIDEO --------
video_path = input("\n🎥 Enter video path: ").strip()

# -------- RUN SCENE UNDERSTANDING --------
scene_result = video_scene_understanding(
    video_path=video_path,
    clip_model=clip_model,
    clip_preprocess=preprocess,
    device=device
)
scene_result = confusion_aware_scene(scene_result)
memory.update(scene_result)


scene_result["why"] = generate_why_explanation(
    scene_result["scene"],
    scene_result["confidence_level"]
)
scene_result["event_summary"] = generate_event_summary(scene_result)

print("\n🎬 VIDEO SCENE UNDERSTANDING RESULT:")
print(json.dumps(scene_result, indent=2))

print("\n TEMPORAL MEMORY SUMMARY:")
print(json.dumps(memory.summary(), indent=2))
